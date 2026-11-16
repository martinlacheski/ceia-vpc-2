"""Arma el dataset de detección de dígitos (etapa 2) a partir de water-meter-dataset (yfzw4).

Acepta varios datasets con dígitos anotados (--src repetido), por ejemplo yfzw4 junto con los de
medidores de energía LCD. Las clases se toman por nombre desde el data.yaml de cada uno: solo se
conservan '0' a '9' y se descartan las demás ('.', 'kWh', ...), así todos comparten las clases 0-9.
Con más de un origen, los archivos llevan el nombre de su carpeta como prefijo para no pisarse.

Para cada imagen con al menos 5 dígitos anotados:
  1. deriva el visor con el mismo criterio que la etapa 1 (prepare_seg_combined.display_polygon);
  2. lo endereza y recorta con meter_reader.rectify, igual que hará el pipeline con la máscara predicha;
  3. transforma las esquinas de cada dígito al recorte y escribe una caja YOLO (clase = dígito).

Así el detector de dígitos entrena con recortes parecidos a los que recibirá en inferencia.
Se respetan los splits originales de yfzw4 (train/valid/test → train/val/test).

Uso: python scripts/prepare_digits.py [--src data/raw/roboflow_water_meter] [--out data/processed/digits_yfzw4]
     python scripts/prepare_digits.py --src data/raw/roboflow_water_meter \
         --src data/raw/energia/meter-digits-5srka --src data/raw/energia/meter-digits-frvqn \
         --out data/processed/digits_mix
"""

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from meter_reader.rectify import rectify, transform_points  # noqa: E402
from prepare_seg_combined import display_polygon  # noqa: E402

SPLITS = {"train": "train", "valid": "val", "test": "test"}
PAD = 0.05  # margen del recorte; el visor derivado ya trae su propio margen


def digit_classes(src: Path) -> dict[int, int]:
    """Clase original -> dígito, según los nombres del data.yaml (sin data.yaml: 0-9 tal cual)."""
    config = src / "data.yaml"
    names = yaml.safe_load(config.read_text()).get("names", []) if config.exists() else [str(i) for i in range(10)]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    return {i: int(n) for i, n in enumerate(map(str, names)) if n.isdigit() and len(n) == 1}


def keep_digits(label_text: str, classes: dict[int, int]) -> str:
    """Deja solo las anotaciones de dígitos, con la clase renumerada a 0-9."""
    rows = []
    for line in label_text.splitlines():
        parts = line.split()
        if parts and int(parts[0]) in classes:
            rows.append(" ".join([str(classes[int(parts[0])]), *parts[1:]]))
    return "\n".join(rows) + "\n"


def digit_corners(values: list[float]) -> np.ndarray:
    """Esquinas normalizadas de un dígito anotado como caja (4 valores) o cuadrilátero (8)."""
    if len(values) == 4:
        cx, cy, w, h = values
        return np.array([[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2],
                         [cx + w / 2, cy + h / 2], [cx - w / 2, cy + h / 2]])
    return np.array(values[: len(values) // 2 * 2]).reshape(-1, 2)


def crop_labels(label_text: str, matrix: np.ndarray, size: tuple[int, int],
                width: int, height: int) -> list[str]:
    """Cajas YOLO de los dígitos en coordenadas del recorte (recortadas a sus bordes)."""
    cw, ch = size
    rows = []
    for line in label_text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        corners = digit_corners(list(map(float, parts[1:]))) * [width, height]
        pts = transform_points(corners, matrix)
        x0, y0 = np.clip(pts.min(axis=0), 0, [cw, ch])
        x1, y1 = np.clip(pts.max(axis=0), 0, [cw, ch])
        if x1 - x0 < 1 or y1 - y0 < 1:
            continue
        rows.append(f"{parts[0]} {(x0 + x1) / 2 / cw:.6f} {(y0 + y1) / 2 / ch:.6f} "
                    f"{(x1 - x0) / cw:.6f} {(y1 - y0) / ch:.6f}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", type=Path, action="append",
                        help="dataset con dígitos anotados (se puede repetir); por defecto yfzw4")
    parser.add_argument("--out", type=Path, default=ROOT / "data/processed/digits_yfzw4")
    args = parser.parse_args()
    sources = args.src or [ROOT / "data/raw/roboflow_water_meter"]
    out = args.out.resolve()
    shutil.rmtree(out, ignore_errors=True)

    counts = {}
    for src_split, split in SPLITS.items():
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)
        kept = 0
        for src in sources:
            kept += prepare_split(src, src_split, out, split, f"{src.name}_" if len(sources) > 1 else "")
        counts[split] = kept

    names = "\n".join(f"  {i}: '{i}'" for i in range(10))
    (out / "data.yaml").write_text(
        f"path: {out}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n{names}\n")
    print("Recortes por split:", counts)


def prepare_split(src: Path, src_split: str, out: Path, split: str, prefix: str) -> int:
    """Recorta el visor de cada imagen de un split de origen; devuelve cuántas se guardaron."""
    classes = digit_classes(src)
    kept = 0
    for image_path in sorted((src / src_split / "images").glob("*")):
        label_path = src / src_split / "labels" / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        text = keep_digits(label_path.read_text(), classes)
        polygon = display_polygon(text, width, height)
        if polygon is None:
            continue
        polygon_px = np.array(polygon).reshape(-1, 2) * [width, height]
        crop, matrix = rectify(image, polygon_px, pad=PAD)
        rows = crop_labels(text, matrix, (crop.shape[1], crop.shape[0]), width, height)
        if not rows:
            continue
        cv2.imwrite(str(out / "images" / split / f"{prefix}{image_path.stem}.jpg"), crop)
        (out / "labels" / split / f"{prefix}{image_path.stem}.txt").write_text("\n".join(rows) + "\n")
        kept += 1
    return kept


if __name__ == "__main__":
    main()
