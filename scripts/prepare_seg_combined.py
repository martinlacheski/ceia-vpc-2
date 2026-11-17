"""Combina Toloka con visores derivados de water-meter-dataset (yfzw4) para la segmentación.

La versión 3 de water-meter-dataset solo anota dígitos (clases 0-9), no el
visor. Como cada imagen anota todos los dígitos de la lectura, el rectángulo
rotado mínimo que los contiene (con un margen) aproxima la ventana del visor.

Reglas:
  - solo se usan imágenes de yfzw4 con al menos MIN_DIGITS dígitos anotados
    (con menos, el rectángulo quedaría corto);
  - todas las imágenes de yfzw4 van a train; val y test son los de Toloka, para
    que los resultados sean comparables con el entrenamiento solo con Toloka;
  - se descartan imágenes de yfzw4 casi idénticas a alguna de val/test de Toloka;
  - con --extra se suman a train otros conjuntos con el visor etiquetado a mano
    (readings.csv, images/, labels/; por ejemplo data/ceml_energia). Las fotos con
    muestra=si en readings.csv quedan afuera: son el test de punta a punta.

Requiere haber corrido antes scripts/prepare_toloka.py.

Uso:
  python scripts/prepare_seg_combined.py
  python scripts/prepare_seg_combined.py --extra data/ceml_energia --out data/processed/seg_toloka_yfzw4_energia
"""

import argparse
import csv
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

MIN_DIGITS = 5
# Margen relativo del visor respecto del rectángulo de los dígitos (largo, alto).
MARGIN_LONG, MARGIN_SHORT = 1.04, 1.15
# Distancia de Hamming máxima entre hashes para considerar dos imágenes duplicadas.
DUPLICATE_DISTANCE = 12


def display_polygon(label_text: str, width: float, height: float) -> list[float] | None:
    """Devuelve el visor como polígono normalizado [x1, y1, ..., x4, y4] o None.

    Acepta dígitos como caja YOLO (cx cy w h) o como cuadrilátero (8 valores).
    """
    points, digits = [], 0
    for line in label_text.splitlines():
        values = list(map(float, line.split()[1:]))
        if len(values) == 4:
            cx, cy, w, h = values
            points += [[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2],
                       [cx + w / 2, cy + h / 2], [cx - w / 2, cy + h / 2]]
        elif len(values) >= 6:
            points += [values[i:i + 2] for i in range(0, len(values) - 1, 2)]
        else:
            continue
        digits += 1
    if digits < MIN_DIGITS:
        return None

    # Se trabaja en píxeles para que el rectángulo rotado respete la proporción real.
    pixels = (np.array(points) * [width, height]).astype(np.float32)
    center, (w, h), angle = cv2.minAreaRect(pixels)
    if w >= h:
        size = (w * MARGIN_LONG, h * MARGIN_SHORT)
    else:
        size = (w * MARGIN_SHORT, h * MARGIN_LONG)
    box = cv2.boxPoints((center, size, angle)) / [width, height]
    return [float(min(max(v, 0.0), 1.0)) for v in box.flatten()]


def image_hash(path: Path) -> int:
    """Hash promedio de 16x16 en escala de grises para detectar casi duplicados."""
    with Image.open(path) as im:
        pixels = np.asarray(im.convert("L").resize((16, 16))).flatten().tolist()
    mean = sum(pixels) / len(pixels)
    return sum(1 << i for i, p in enumerate(pixels) if p > mean)


def link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def add_extra(root: Path, train_images: Path, train_labels: Path) -> int:
    """Suma a train las fotos etiquetadas de root, salvo la muestra de test. Devuelve cuántas."""
    with (root / "readings.csv").open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("muestra", "").strip().lower() != "si"]
    added = 0
    for row in rows:
        label = root / "labels" / f"{Path(row['file']).stem}.txt"
        if not label.exists():
            continue
        link_or_copy(root / "images" / row["file"], train_images / row["file"])
        shutil.copy2(label, train_labels / label.name)
        added += 1
    return added


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--toloka", type=Path, default=Path("data/processed/toloka_seg"))
    parser.add_argument("--yfzw4", type=Path, default=Path("data/raw/roboflow_water_meter"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/seg_toloka_yfzw4"))
    parser.add_argument("--extra", type=Path, action="append", default=[],
                        help="conjunto con readings.csv, images/ y labels/ que se suma a train (repetible)")
    args = parser.parse_args()

    for sub in ("images", "labels"):
        shutil.rmtree(args.out / sub, ignore_errors=True)

    # 1. Toloka se copia tal cual, respetando sus splits.
    for split in ("train", "val", "test"):
        for sub in ("images", "labels"):
            (args.out / sub / split).mkdir(parents=True, exist_ok=True)
            for f in (args.toloka / sub / split).iterdir():
                link_or_copy(f, args.out / sub / split / f.name)

    eval_hashes = [image_hash(p) for split in ("val", "test")
                   for p in (args.toloka / "images" / split).iterdir()]

    # 2. yfzw4 (todos sus splits) se agrega a train con el visor derivado.
    added = too_few = duplicates = 0
    for label in sorted(args.yfzw4.glob("*/labels/*.txt")):
        image = label.parent.parent / "images" / f"{label.stem}.jpg"
        with Image.open(image) as im:
            width, height = im.size
        polygon = display_polygon(label.read_text(encoding="utf-8"), width, height)
        if polygon is None:
            too_few += 1
            continue
        h = image_hash(image)
        if any(bin(h ^ e).count("1") <= DUPLICATE_DISTANCE for e in eval_hashes):
            duplicates += 1
            continue
        name = f"yfzw4_{label.stem}"
        link_or_copy(image, args.out / "images" / "train" / f"{name}.jpg")
        coords = " ".join(f"{v:.6g}" for v in polygon)
        (args.out / "labels" / "train" / f"{name}.txt").write_text(f"0 {coords}\n", encoding="utf-8")
        added += 1

    # 3. Conjuntos extra con el visor etiquetado a mano (sin la muestra de test).
    for root in args.extra:
        n = add_extra(root, args.out / "images" / "train", args.out / "labels" / "train")
        print(f"{root.name}: {n} fotos agregadas a train")

    # Ruta absoluta: Ultralytics resuelve las relativas contra su propio directorio de datasets.
    (args.out / "data.yaml").write_text(
        f"path: {args.out.resolve()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        "names: {0: 'display'}\n",
        encoding="utf-8",
    )
    print(f"yfzw4: {added} agregadas, {too_few} con menos de {MIN_DIGITS} dígitos, "
          f"{duplicates} casi duplicadas de val/test de Toloka -> {args.out}")


if __name__ == "__main__":
    main()
