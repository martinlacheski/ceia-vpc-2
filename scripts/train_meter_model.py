"""Entrena un clasificador del modelo de medidor (CNN preentrenada en ImageNet + fine-tuning).

El modelo de medidor fija cuántos dígitos son m³ y cuántos fracción (src/meter_reader/layout.py).
La entrada es el visor enderezado con su entorno (logo, marco, "m³"), armado con el polígono
etiquetado del visor. Las etiquetas salen de la columna `modelo` de data/ceml/readings.csv.

Con tan pocas fotos (324) se evalúa con validación cruzada estratificada de k folds: cada foto
se predice con un modelo que no la vio (predicción out-of-fold). Esas predicciones se usan
después para evaluar el pipeline completo sobre CEML sin fuga de datos. Al final se entrena
con todas las fotos el modelo que usa la app.

  --target modelo: una clase por modelo de medidor;
  --target formato: una clase por disposición (5_2, 4_2, ...), que junta modelos distintos
                    con la misma cantidad de dígitos.
Las clases con menos de --min-samples fotos no se entrenan (quedan como "sin clase").

Guarda:
  - experiments/<nombre>/cv.json: exactitud, F1 macro, exactitud del formato y matriz de confusión;
  - experiments/<nombre>/oof.csv: predicción out-of-fold de cada foto;
  - models/<nombre>.pt: pesos entrenados con todas las fotos, con la lista de clases.

Uso (con el entorno que tiene torch, por ejemplo .venv-rocm):
  python scripts/train_meter_model.py --name meter_model_mnv3s
"""

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from meter_reader.layout import layout_of, model_input  # noqa: E402
from meter_reader.meter_model import ARCHS, build_model, predict_probs, to_tensor  # noqa: E402



def layout_name(model: str) -> str:
    integer, fraction = layout_of(model)
    return f"{integer}_{fraction}"


def read_polygon(label_path: Path, width: int, height: int) -> np.ndarray | None:
    values = label_path.read_text(encoding="utf-8").split() if label_path.exists() else []
    if len(values) < 7:
        return None
    return np.array(values[1:], dtype=float).reshape(-1, 2) * [width, height]


def load_examples(sheet: Path, images_dir: Path, labels_dir: Path) -> list[dict]:
    """Fotos con modelo y visor etiquetado: {file, model, image} (entrada del clasificador, BGR)."""
    examples = []
    with sheet.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if layout_of(row.get("modelo", "")) is None:  # sin modelo o sin disposición
                continue
            image = cv2.imread(str(images_dir / row["file"]))  # aplica la rotación EXIF
            if image is None:
                continue
            height, width = image.shape[:2]
            polygon = read_polygon(labels_dir / f"{Path(row['file']).stem}.txt", width, height)
            if polygon is None:
                continue
            examples.append({"file": row["file"], "model": row["modelo"], "image": model_input(image, polygon)})
    return examples


def target_of(model: str, target: str) -> str:
    return model if target == "modelo" else layout_name(model)


def trainable_classes(labels: list[str], min_samples: int) -> list[str]:
    counts = Counter(labels)
    return sorted(c for c, n in counts.items() if n >= min_samples)


def stratified_folds(labels: list[str], k: int, seed: int) -> list[int]:
    """Asigna un fold a cada ejemplo repartiendo cada clase de forma pareja entre los k folds."""
    rng = random.Random(seed)
    folds = [0] * len(labels)
    by_class: dict[str, list[int]] = {}
    for i, label in enumerate(labels):
        by_class.setdefault(label, []).append(i)
    offset = 0
    for label in sorted(by_class):
        indices = by_class[label]
        rng.shuffle(indices)
        for j, i in enumerate(indices):
            folds[i] = (offset + j) % k
        offset += len(indices)  # las clases chicas no caen siempre en el fold 0
    return folds


def classification_metrics(true: list[str], pred: list[str], classes: list[str]) -> dict:
    """Exactitud, F1 macro y matriz de confusión (filas: real, columnas: predicho)."""
    index = {c: i for i, c in enumerate(classes)}
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(true, pred):
        matrix[index[t], index[p]] += 1
    f1 = []
    for i in range(len(classes)):
        tp = matrix[i, i]
        precision = tp / matrix[:, i].sum() if matrix[:, i].sum() else 0.0
        recall = tp / matrix[i].sum() if matrix[i].sum() else 0.0
        f1.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return {
        "accuracy": round(float(np.trace(matrix) / max(1, matrix.sum())), 4),
        "macro_f1": round(float(np.mean(f1)), 4),
        "per_class_f1": {c: round(float(v), 4) for c, v in zip(classes, f1)},
        "confusion": matrix.tolist(),
    }


def augment(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Variaciones de campo: visor girado 180°, leve rotación y escala, brillo, contraste y desenfoque."""
    out = image
    if rng.random() < 0.5:
        out = out[::-1, ::-1]
    height, width = out.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), rng.uniform(-6, 6), rng.uniform(0.9, 1.1))
    matrix[:, 2] += rng.uniform(-0.05, 0.05, 2) * [width, height]
    out = cv2.warpAffine(np.ascontiguousarray(out), matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)
    out = np.clip(out.astype(np.float32) * rng.uniform(0.7, 1.3) + rng.uniform(-30, 30), 0, 255)
    if rng.random() < 0.3:
        out = cv2.GaussianBlur(out, (5, 5), 0)
    return out.astype(np.uint8)


def fit(images: list[np.ndarray], labels: list[int], num_classes: int, args, device):
    """Entrena una CNN con pérdida ponderada por clase (las clases chicas pesan más)."""
    import torch

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    net = build_model(args.arch, num_classes).to(device)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    weights = torch.tensor((counts.max() / np.maximum(counts, 1)) ** 0.5, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * max(1, len(images) // args.batch)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=steps)
    target = torch.tensor(labels)
    step = 0
    for _ in range(args.epochs):
        net.train()
        order = rng.permutation(len(images))
        for start in range(0, len(order) - args.batch + 1, args.batch):
            batch = order[start:start + args.batch]
            x = to_tensor([augment(images[i], rng) for i in batch]).to(device)
            loss = loss_fn(net(x), target[batch].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step < steps:
                scheduler.step()
            step += 1
    return net


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True)
    parser.add_argument("--arch", choices=ARCHS, default="mobilenet_v3_small")
    parser.add_argument("--target", choices=["modelo", "formato"], default="modelo")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sheet", type=Path, default=ROOT / "data/ceml/readings.csv")
    parser.add_argument("--images", type=Path, default=ROOT / "data/ceml/images")
    parser.add_argument("--labels", type=Path, default=ROOT / "data/ceml/labels")
    args = parser.parse_args()

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    examples = load_examples(args.sheet, args.images, args.labels)
    labels = [target_of(e["model"], args.target) for e in examples]
    classes = trainable_classes(labels, args.min_samples)
    used = [i for i, label in enumerate(labels) if label in classes]
    skipped = sorted(Counter(labels[i] for i in range(len(labels)) if labels[i] not in classes).items())
    print(f"{len(examples)} fotos, {len(used)} en {len(classes)} clases; sin clase: {skipped}")

    images = [examples[i]["image"] for i in used]
    y = [classes.index(labels[i]) for i in used]
    folds = stratified_folds([labels[i] for i in used], args.folds, args.seed)
    oof = np.zeros((len(used), len(classes)), dtype=np.float32)
    for k in range(args.folds):
        train = [j for j, f in enumerate(folds) if f != k]
        test = [j for j, f in enumerate(folds) if f == k]
        net = fit([images[j] for j in train], [y[j] for j in train], len(classes), args, device)
        oof[test] = predict_probs(net, [images[j] for j in test], device)
        print(f"fold {k + 1}/{args.folds}: exactitud {np.mean(oof[test].argmax(1) == np.array(y)[test]):.3f}", flush=True)

    true = [classes[c] for c in y]
    pred = [classes[c] for c in oof.argmax(1)]
    metrics = classification_metrics(true, pred, classes)
    # Lo que importa para leer: que la disposición (enteros, fracción) sea la correcta
    true_layout = [layout_name(examples[i]["model"]) for i in used]
    pred_layout = [p if args.target == "formato" else layout_name(p) for p in pred]
    metrics["layout_accuracy"] = round(float(np.mean([t == p for t, p in zip(true_layout, pred_layout)])), 4)
    metrics.update({"classes": classes, "skipped": dict(skipped), "images": len(used), "arch": args.arch,
                    "target": args.target, "folds": args.folds, "epochs": args.epochs, "lr": args.lr,
                    "batch": args.batch, "seed": args.seed})

    exp_dir = ROOT / "experiments" / args.name
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "cv.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (exp_dir / "oof.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["file", "modelo", "real", "predicho", "probabilidad", "fold"])
        for j, i in enumerate(used):
            c = int(oof[j].argmax())
            writer.writerow([examples[i]["file"], examples[i]["model"], true[j], classes[c],
                             f"{oof[j, c]:.3f}", folds[j]])
    print(json.dumps({k: metrics[k] for k in ("accuracy", "macro_f1", "layout_accuracy")}))

    net = fit(images, y, len(classes), args, device)
    weights = ROOT / "models" / f"{args.name}.pt"
    torch.save({"arch": args.arch, "target": args.target, "classes": classes,
                "state_dict": net.cpu().state_dict()}, weights)
    print(f"Modelo final en {weights}")


if __name__ == "__main__":
    main()
