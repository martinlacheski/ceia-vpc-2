"""Entrena el clasificador de huecos decimales (src/meter_reader/gaps.py) y mide si generaliza.

Los ejemplos salen de los visores de energía ya recortados y etiquetados (<raíz>/digits/) y del
formato de cada medidor (columna `modelo` de readings.csv): con las cajas de los dígitos se arman
los recortes de los huecos, y la disposición (enteros, fracción) dice cuál es el decimal. No hace
falta etiquetar el punto a mano. Las fotos cuya cantidad de dígitos no coincide con la disposición
quedan afuera, porque no se sabe qué hueco es el decimal.

La pregunta es si sirve para medidores que nunca vio, así que la validación por defecto deja
afuera un modelo de medidor entero por vez (--cv modelo). Con --cv medidor se hacen k folds por
medidor, para comparar con el caso de modelos conocidos. Quedan afuera las fotos de la muestra de
prueba y las demás de esos medidores (como en scripts/prepare_digits.py): son el test de punta a punta.

Con --dot-data se suman al entrenamiento datasets YOLO externos con el punto anotado como clase
'.' (por ejemplo LCD de energía de Roboflow): el hueco donde cae el punto es el positivo. Solo
entrenan; la evaluación sigue siendo sobre los medidores propios.

Guarda:
  - experiments/<nombre>/cv.json: exactitud de la posición del decimal por foto, total y por modelo;
  - experiments/<nombre>/oof.csv: predicción out-of-fold de cada foto;
  - models/<nombre>.pt: pesos entrenados con todas las fotos.

Uso (con el entorno que tiene torch, por ejemplo .venv-rocm):
  python scripts/train_gap_model.py --name gap_energia_mnv3s
  python scripts/train_gap_model.py --name gap_energia_lcd_mnv3s --dot-data data/raw/energia/meter-digits-5srka
"""

import argparse
import csv
import json
import random
import sys
import zlib
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from meter_reader.gaps import PATCH_SIZE, decimal_gap, gap_patches, gap_target  # noqa: E402
from meter_reader.layout import layout_of  # noqa: E402
from meter_reader.meter_model import ARCHS, build_model, to_tensor  # noqa: E402
from meter_reader.reading import Digit  # noqa: E402


def read_digits(label_path: Path, width: int, height: int) -> list[Digit]:
    """Cajas YOLO (clase cx cy w h normalizados) → dígitos en píxeles del recorte."""
    digits = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) != 5:
            continue
        cls, cx, cy, w, h = int(values[0]), *map(float, values[1:])
        digits.append(Digit(cls, 1.0, (cx - w / 2) * width, (cy - h / 2) * height,
                            (cx + w / 2) * width, (cy + h / 2) * height))
    return digits


def load_photos(root: Path, skip_unreadable: bool = False) -> tuple[list[dict], dict[str, int]]:
    """Fotos con sus recortes de huecos: {file, model, meter, patches, target}.

    target es la cantidad de dígitos enteros (dónde va el punto) o None si no hay fracción.
    Devuelve también cuántas fotos quedaron afuera y por qué.
    """
    with (root / "readings.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with (root / "metadata.csv").open(newline="", encoding="utf-8") as f:
        meter_of = {r["file"]: r["meter_id"] for r in csv.DictReader(f)}
    model_of = {r["file"]: r.get("modelo", "") for r in rows}
    sample = {r["file"] for r in rows if r.get("muestra", "").strip().lower() == "si"}
    test_meters = {meter_of[name] for name in sample if name in meter_of}
    unreadable = set()
    if skip_unreadable:
        with (root / "digits" / "revision.csv").open(newline="", encoding="utf-8") as f:
            unreadable = {r["file"] for r in csv.DictReader(f) if r["estado"] == "ilegible"}

    photos, skipped = [], defaultdict(int)
    for image_path in sorted((root / "digits" / "images").glob("*.jpg")):
        name = image_path.name
        meter, layout = meter_of.get(name), layout_of(model_of.get(name, ""))
        label_path = root / "digits" / "labels" / f"{image_path.stem}.txt"
        if meter is None or layout is None or not label_path.exists():
            skipped["sin modelo o etiquetas"] += 1
            continue
        if meter in test_meters:
            skipped["muestra de prueba"] += 1
            continue
        if name in unreadable:
            skipped["ilegible"] += 1
            continue
        crop = cv2.imread(str(image_path))
        digits = read_digits(label_path, crop.shape[1], crop.shape[0])
        gap = gap_target(len(digits), layout)
        if gap == -1 or len(digits) < 2:
            skipped["dígitos no coinciden con el formato"] += 1
            continue
        photos.append({"file": name, "model": model_of[name], "meter": meter,
                       "patches": gap_patches(crop, digits),
                       "target": None if gap is None else gap + 1})
    return photos, dict(skipped)


def dot_target(digits: list[Digit], dot_x: float) -> int:
    """Cantidad de dígitos enteros según dónde cae el punto, o -1 si no cae entre dos dígitos."""
    centers = sorted((d.x1 + d.x2) / 2 for d in digits)
    for i, (left, right) in enumerate(zip(centers, centers[1:])):
        if left < dot_x < right:
            return i + 1
    return -1


def class_names(data_yaml: Path) -> list[str]:
    """Nombres de clase de un data.yaml de Roboflow (lista `names:` con un nombre por renglón)."""
    names, inside = [], False
    for line in data_yaml.read_text(encoding="utf-8").splitlines():
        if line.startswith("names:"):
            inside = True
        elif inside and line.startswith("- "):
            names.append(line[2:].strip().strip("'\""))
        elif inside:
            break
    return names


def load_dot_dataset(root: Path) -> list[dict]:
    """Fotos de un dataset externo con el punto anotado: solo las que tienen un único punto entre
    dos dígitos (sin punto puede ser una etiqueta que falta). Los recortes salen de la foto tal cual."""
    names = class_names(root / "data.yaml")
    photos = []
    for label_path in sorted(root.glob("*/labels/*.txt")):
        image_path = next(label_path.parent.parent.joinpath("images").glob(f"{label_path.stem}.*"), None)
        if image_path is None:
            continue
        rows = [line.split() for line in label_path.read_text(encoding="utf-8").splitlines()]
        rows = [r for r in rows if len(r) == 5 and int(r[0]) < len(names)]
        dots = [r for r in rows if names[int(r[0])] == "."]
        boxes = [r for r in rows if names[int(r[0])].isdigit()]
        if len(dots) != 1 or len(boxes) < 2:
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        digits = [Digit(int(names[int(r[0])]), 1.0, (float(r[1]) - float(r[3]) / 2) * width,
                        (float(r[2]) - float(r[4]) / 2) * height, (float(r[1]) + float(r[3]) / 2) * width,
                        (float(r[2]) + float(r[4]) / 2) * height) for r in boxes]
        target = dot_target(digits, float(dots[0][1]) * width)
        if target == -1:
            continue
        photos.append({"file": image_path.name, "model": f"externo:{root.name}", "meter": None,
                       "patches": gap_patches(image, digits), "target": target})
    return photos


def gap_labels(photo: dict) -> list[int]:
    """1 en el hueco decimal, 0 en los demás."""
    return [int(photo["target"] is not None and i == photo["target"] - 1) for i in range(len(photo["patches"]))]


def folds_by(photos: list[dict], cv: str, k: int) -> list[int]:
    """Fold de cada foto: un modelo de medidor por fold (cv=modelo) o k folds por medidor."""
    if cv == "modelo":
        models = sorted({p["model"] for p in photos})
        return [models.index(p["model"]) for p in photos]
    return [zlib.crc32(p["meter"].encode()) % k for p in photos]


def augment(patch: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Brillo, contraste, desenfoque y un leve corrimiento: sin giros, el lado del hueco importa."""
    height, width = patch.shape[:2]
    matrix = np.float32([[1, 0, rng.uniform(-0.08, 0.08) * width], [0, 1, rng.uniform(-0.08, 0.08) * height]])
    out = cv2.warpAffine(patch, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    out = np.clip(out * rng.uniform(0.7, 1.3) + rng.uniform(-30, 30), 0, 255)
    if rng.random() < 0.3:
        out = cv2.GaussianBlur(out, (3, 3), 0)
    return out.astype(np.uint8)


def fit(patches: list[np.ndarray], labels: list[int], args, device):
    """CNN binaria (hueco decimal o no) con pérdida ponderada: hay un decimal cada varios huecos."""
    import torch

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    net = build_model(args.arch, 2).to(device)
    counts = np.bincount(labels, minlength=2).astype(np.float32)
    weights = torch.tensor((counts.max() / np.maximum(counts, 1)) ** 0.5, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * max(1, len(patches) // args.batch)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=steps)
    target = torch.tensor(labels)
    step = 0
    for _ in range(args.epochs):
        net.train()
        order = rng.permutation(len(patches))
        for start in range(0, len(order) - args.batch + 1, args.batch):
            batch = order[start:start + args.batch]
            x = to_tensor([augment(patches[i], rng) for i in batch]).to(device)
            loss = loss_fn(net(x), target[batch].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step < steps:
                scheduler.step()
            step += 1
    return net


def gap_probs(net, patches: list[np.ndarray], device) -> np.ndarray:
    """Probabilidad de que cada hueco sea el decimal."""
    import torch

    if not patches:
        return np.zeros(0, dtype=np.float32)
    net.eval()
    with torch.no_grad():
        return net(to_tensor(patches).to(device)).softmax(1)[:, 1].cpu().numpy()


def summarize(photos: list[dict], predicted: list[int | None]) -> dict:
    """Exactitud de la posición del decimal por foto: total, con y sin fracción, y por modelo."""
    hits = [p["target"] == q for p, q in zip(photos, predicted)]
    by_model = defaultdict(list)
    for p, hit in zip(photos, hits):
        by_model[p["model"]].append(hit)

    def accuracy(values: list[bool]) -> float | None:
        return round(float(np.mean(values)), 4) if values else None

    return {
        "photos": len(photos),
        "accuracy": accuracy(hits),
        "accuracy_with_fraction": accuracy([h for p, h in zip(photos, hits) if p["target"] is not None]),
        "accuracy_without_fraction": accuracy([h for p, h in zip(photos, hits) if p["target"] is None]),
        "per_model": {m: {"photos": len(v), "accuracy": accuracy(v)} for m, v in sorted(by_model.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True)
    parser.add_argument("--root", type=Path, default=ROOT / "data/ceml_energia")
    parser.add_argument("--arch", choices=ARCHS, default="mobilenet_v3_small")
    parser.add_argument("--cv", choices=["modelo", "medidor"], default="modelo")
    parser.add_argument("--folds", type=int, default=5, help="solo con --cv medidor")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-unreadable", action="store_true")
    parser.add_argument("--dot-data", type=Path, action="append", default=[],
                        help="dataset YOLO externo con la clase '.', solo para entrenar")
    args = parser.parse_args()

    import torch

    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    photos, skipped = load_photos(args.root, args.skip_unreadable)
    print(f"{len(photos)} fotos, {sum(len(p['patches']) for p in photos)} huecos; afuera: {skipped}")
    extra = [p for root in args.dot_data for p in load_dot_dataset(root)]
    if extra:
        print(f"externas (solo entrenamiento): {len(extra)} fotos, {sum(len(p['patches']) for p in extra)} huecos")

    folds = folds_by(photos, args.cv, args.folds)
    probs: list[np.ndarray | None] = [None] * len(photos)
    for k in sorted(set(folds)):
        train = [p for p, f in zip(photos, folds) if f != k] + extra
        net = fit([x for p in train for x in p["patches"]], [y for p in train for y in gap_labels(p)], args, device)
        for i, f in enumerate(folds):
            if f == k:
                probs[i] = gap_probs(net, photos[i]["patches"], device)
        held = [i for i, f in enumerate(folds) if f == k]
        hits = np.mean([decimal_gap(probs[i]) == photos[i]["target"] for i in held])
        print(f"fold {k + 1}: {len(held)} fotos ({photos[held[0]]['model'] if args.cv == 'modelo' else 'medidores'}), "
              f"exactitud {hits:.3f}", flush=True)

    predicted = [decimal_gap(p) for p in probs]
    metrics = summarize(photos, predicted)
    metrics.update({"cv": args.cv, "skipped": skipped, "arch": args.arch, "epochs": args.epochs,
                    "lr": args.lr, "batch": args.batch, "seed": args.seed, "patch_size": list(PATCH_SIZE),
                    "skip_unreadable": args.skip_unreadable, "dot_data": [str(d) for d in args.dot_data],
                    "extra_photos": len(extra)})

    exp_dir = ROOT / "experiments" / args.name
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "cv.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    with (exp_dir / "oof.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["file", "modelo", "real", "predicho", "probabilidades", "fold"])
        for p, q, pr, fold in zip(photos, predicted, probs, folds):
            writer.writerow([p["file"], p["model"], p["target"] or "", q or "",
                             " ".join(f"{v:.2f}" for v in pr), fold])
    print(json.dumps({k: metrics[k] for k in ("accuracy", "accuracy_with_fraction", "accuracy_without_fraction")}))

    final = photos + extra
    net = fit([x for p in final for x in p["patches"]], [y for p in final for y in gap_labels(p)], args, device)
    weights = ROOT / "models" / f"{args.name}.pt"
    torch.save({"arch": args.arch, "patch_size": list(PATCH_SIZE), "state_dict": net.cpu().state_dict()}, weights)
    print(f"Modelo final en {weights}")


if __name__ == "__main__":
    main()
