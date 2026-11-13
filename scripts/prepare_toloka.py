"""Convierte el dataset Toloka al formato YOLO-seg y arma los splits.

Lee data/raw/toloka/data.csv (polígono del visor por imagen) y genera:
  - <out>/images/{train,val,test}/*.jpg  (enlaces duros a las originales)
  - <out>/labels/{train,val,test}/*.txt  (formato YOLO-seg, clase 0 = display)
  - <out>/data.yaml                       (configuración para Ultralytics)
  - <splits>                              (CSV con el split y la lectura de cada imagen)

Cada imagen de Toloka corresponde a un medidor distinto, por lo que el split
aleatorio no filtra el mismo medidor entre train y test. La lectura se toma del
nombre de archivo (id_<n>_value_<entero>_<decimales>.jpg) porque la columna
"value" del CSV tiene errores de punto flotante (p. ej. 305.16200000000003).

Uso:
  python scripts/prepare_toloka.py
"""

import argparse
import ast
import csv
import os
import random
import re
import shutil
from pathlib import Path

SEED = 42
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
CLASS_NAMES = ["display"]
READING_RE = re.compile(r"value_(\d+)_(\d+)")


def reading_from_name(name: str) -> str:
    """Extrae la lectura exacta del nombre de archivo: ..._value_305_162 -> 305.162."""
    match = READING_RE.search(name)
    return f"{match[1]}.{match[2]}" if match else ""


def polygon_to_yolo(location: str) -> str:
    """Convierte el polígono normalizado de Toloka a una línea YOLO-seg."""
    points = ast.literal_eval(location)["data"]
    coords = []
    for p in points:
        # Se recorta a [0, 1] por seguridad ante puntos apenas fuera de la imagen.
        coords += [min(max(p["x"], 0.0), 1.0), min(max(p["y"], 0.0), 1.0)]
    return "0 " + " ".join(f"{c:.6g}" for c in coords)


def assign_splits(names: list[str]) -> dict[str, str]:
    """Asigna train/val/test de forma reproducible (70/15/15)."""
    shuffled = sorted(names)
    random.Random(SEED).shuffle(shuffled)
    n_val = round(len(shuffled) * VAL_FRACTION)
    n_test = round(len(shuffled) * TEST_FRACTION)
    splits = {}
    for i, name in enumerate(shuffled):
        splits[name] = "val" if i < n_val else "test" if i < n_val + n_test else "train"
    return splits


def link_or_copy(src: Path, dst: Path) -> None:
    """Usa un enlace duro para no duplicar espacio; si no se puede, copia."""
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, default=Path("data/raw/toloka"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/toloka_seg"))
    parser.add_argument("--splits", type=Path, default=Path("data/splits/toloka.csv"))
    args = parser.parse_args()

    with open(args.raw / "data.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    splits = assign_splits([r["photo_name"] for r in rows])

    # Se regenera desde cero para no dejar archivos de corridas anteriores.
    for sub in ("images", "labels"):
        shutil.rmtree(args.out / sub, ignore_errors=True)
        for split in ("train", "val", "test"):
            (args.out / sub / split).mkdir(parents=True, exist_ok=True)

    for r in rows:
        name, split = r["photo_name"], splits[r["photo_name"]]
        link_or_copy(args.raw / "images" / name, args.out / "images" / split / name)
        label = args.out / "labels" / split / f"{Path(name).stem}.txt"
        label.write_text(polygon_to_yolo(r["location"]) + "\n", encoding="utf-8")

    # Ruta absoluta: Ultralytics resuelve las relativas contra su propio directorio de datasets.
    (args.out / "data.yaml").write_text(
        f"path: {args.out.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"names: {dict(enumerate(CLASS_NAMES))}\n",
        encoding="utf-8",
    )

    args.splits.parent.mkdir(parents=True, exist_ok=True)
    with open(args.splits, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "split", "reading"])
        for name in sorted(splits):
            writer.writerow([name, splits[name], reading_from_name(name)])

    counts = {s: sum(v == s for v in splits.values()) for s in ("train", "val", "test")}
    print(f"{len(rows)} imágenes -> {args.out} {counts}")


if __name__ == "__main__":
    main()
