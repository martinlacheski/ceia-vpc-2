"""Arma el dataset de detección de dígitos (etapa 2) a partir de water-meter-dataset (yfzw4).

Para cada imagen con al menos 5 dígitos anotados:
  1. deriva el visor con el mismo criterio que la etapa 1 (prepare_seg_combined.display_polygon);
  2. lo endereza y recorta con meter_reader.rectify, igual que hará el pipeline con la máscara predicha;
  3. transforma las esquinas de cada dígito al recorte y escribe una caja YOLO (clase = dígito).

Así el detector de dígitos entrena con recortes parecidos a los que recibirá en inferencia.
Se respetan los splits originales de yfzw4 (train/valid/test → train/val/test).

Uso: python scripts/prepare_digits.py [--src data/raw/roboflow_water_meter] [--out data/processed/digits_yfzw4]
"""

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from meter_reader.rectify import rectify, transform_points  # noqa: E402
from prepare_seg_combined import display_polygon  # noqa: E402

SPLITS = {"train": "train", "valid": "val", "test": "test"}
PAD = 0.05  # margen del recorte; el visor derivado ya trae su propio margen


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
    parser.add_argument("--src", type=Path, default=ROOT / "data/raw/roboflow_water_meter")
    parser.add_argument("--out", type=Path, default=ROOT / "data/processed/digits_yfzw4")
    args = parser.parse_args()
    out = args.out.resolve()
    shutil.rmtree(out, ignore_errors=True)

    counts = {}
    for src_split, split in SPLITS.items():
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)
        kept = 0
        for image_path in sorted((args.src / src_split / "images").glob("*")):
            label_path = args.src / src_split / "labels" / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            height, width = image.shape[:2]
            text = label_path.read_text()
            polygon = display_polygon(text, width, height)
            if polygon is None:
                continue
            polygon_px = np.array(polygon).reshape(-1, 2) * [width, height]
            crop, matrix = rectify(image, polygon_px, pad=PAD)
            rows = crop_labels(text, matrix, (crop.shape[1], crop.shape[0]), width, height)
            if not rows:
                continue
            cv2.imwrite(str(out / "images" / split / f"{image_path.stem}.jpg"), crop)
            (out / "labels" / split / f"{image_path.stem}.txt").write_text("\n".join(rows) + "\n")
            kept += 1
        counts[split] = kept

    names = "\n".join(f"  {i}: '{i}'" for i in range(10))
    (out / "data.yaml").write_text(
        f"path: {out}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n{names}\n")
    print("Recortes por split:", counts)


if __name__ == "__main__":
    main()
