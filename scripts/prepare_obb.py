"""Convierte un dataset de segmentación del visor (YOLO-seg) en uno de cajas orientadas (OBB).

Sirve para comparar dos formas de obtener el rectángulo rotado que usa la rectificación:
  - segmentación (pipeline actual): el modelo predice el contorno del visor y el pipeline
    le ajusta el rectángulo rotado mínimo (cv2.minAreaRect);
  - OBB: el modelo predice directamente ese rectángulo rotado.

Cada polígono se reemplaza por su rectángulo rotado mínimo, calculado en píxeles (en
coordenadas normalizadas una imagen no cuadrada deformaría el ángulo) y recortado al borde
de la imagen. Las imágenes no se copian: se enlazan (symlink) a las del dataset de origen.
Formato de salida (Ultralytics OBB): "clase x1 y1 x2 y2 x3 y3 x4 y4", normalizado.

Uso:
  python scripts/prepare_obb.py --src data/processed/seg_toloka_yfzw4 \
      --out data/processed/seg_toloka_yfzw4_obb
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
# Rotaciones EXIF de 90° o 270°: Ultralytics intercambia ancho y alto (utils.exif_size).
EXIF_SWAPS_SIDES = {5, 6, 7, 8}


def image_size(path: Path) -> tuple[int, int]:
    """Ancho y alto tal como los ve Ultralytics (con la rotación EXIF aplicada)."""
    with Image.open(path) as im:
        width, height = im.size
        if im.getexif().get(274) in EXIF_SWAPS_SIDES:  # 274: etiqueta Orientation
            width, height = height, width
    return width, height


def obb_line(seg_line: str, width: int, height: int) -> str:
    """Línea YOLO-seg ("clase x1 y1 x2 y2 ...") a línea OBB con el rectángulo rotado mínimo."""
    values = seg_line.split()
    polygon = np.array(values[1:], dtype=np.float32).reshape(-1, 2) * [width, height]
    box = cv2.boxPoints(cv2.minAreaRect(polygon.astype(np.float32)))
    box = np.clip(box / [width, height], 0.0, 1.0)
    return " ".join([values[0]] + [f"{v:.6f}" for v in box.flatten()])


def convert(src: Path, out: Path) -> int:
    """Arma el dataset OBB en out/ y devuelve la cantidad de imágenes."""
    total = 0
    for split in SPLITS:
        images = src / "images" / split
        if not images.is_dir():
            continue
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for image in sorted(p for p in images.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
            link = out / "images" / split / image.name
            if link.is_symlink() or link.exists():
                link.unlink()
            os.symlink(image.resolve(), link)
            label = src / "labels" / split / f"{image.stem}.txt"
            lines = label.read_text().splitlines() if label.exists() else []
            width, height = image_size(image)
            converted = [obb_line(line, width, height) for line in lines if len(line.split()) >= 7]
            (out / "labels" / split / f"{image.stem}.txt").write_text(
                "".join(line + "\n" for line in converted))
            total += 1
    names = next((line for line in (src / "data.yaml").read_text().splitlines()
                  if line.startswith("names:")), "names: {0: 'display'}")
    splits = "".join(f"{split}: images/{split}\n" for split in SPLITS if (out / "images" / split).is_dir())
    (out / "data.yaml").write_text(f"path: {out.resolve()}\n{splits}{names}\n")
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", type=Path, default=Path("data/processed/seg_toloka_yfzw4"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/seg_toloka_yfzw4_obb"))
    args = parser.parse_args()
    total = convert(args.src, args.out)
    print(f"{total} imágenes con cajas orientadas en {args.out}")


if __name__ == "__main__":
    main()
