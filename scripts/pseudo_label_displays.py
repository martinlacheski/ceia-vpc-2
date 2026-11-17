"""Genera pseudo-etiquetas del visor con el segmentador entrenado (formato YOLO-seg).

Para fotos que no tienen el visor dibujado a mano (por ejemplo, los medidores de energía de
data/ceml_energia), el segmentador actual propone el polígono del visor y se guarda como si fuera
una etiqueta. Sirve para entrenar el clasificador de formato, que recibe el visor con su entorno
y tolera un contorno aproximado. Son etiquetas derivadas: en el informe se describen como tales y
no se usan para medir el segmentador.

Las fotos donde no se encuentra el visor quedan sin etiqueta (el clasificador las saltea).

Uso (con el entorno que tiene ultralytics, por ejemplo .venv-rocm):
  python scripts/pseudo_label_displays.py --data data/ceml_energia
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def yolo_line(polygon: np.ndarray, width: int, height: int) -> str:
    """Línea YOLO-seg (clase 0) con el polígono normalizado y recortado a la imagen."""
    points = np.clip(np.asarray(polygon, dtype=float) / [width, height], 0.0, 1.0)
    return "0 " + " ".join(f"{v:.6f}" for v in points.reshape(-1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=ROOT / "data/ceml_energia",
                        help="carpeta con readings.csv e images/; escribe labels/")
    parser.add_argument("--seg-weights", type=Path, default=ROOT / "models/toloka_yfzw4_yolov8n_seg.pt")
    args = parser.parse_args()

    import cv2

    from meter_reader.pipeline import load_models

    segment, _ = load_models(str(args.seg_weights), str(args.seg_weights))
    labels_dir = args.data / "labels"
    labels_dir.mkdir(exist_ok=True)
    with (args.data / "readings.csv").open(newline="", encoding="utf-8") as f:
        files = [row["file"] for row in csv.DictReader(f)]
    found = 0
    for name in files:
        image = cv2.imread(str(args.data / "images" / name))
        polygon = segment(image) if image is not None else None
        label = labels_dir / f"{Path(name).stem}.txt"
        if polygon is None or len(polygon) < 3:
            label.unlink(missing_ok=True)
            continue
        height, width = image.shape[:2]
        label.write_text(yolo_line(polygon, width, height) + "\n", encoding="utf-8")
        found += 1
    print(f"visor encontrado en {found} de {len(files)} fotos -> {labels_dir}")


if __name__ == "__main__":
    main()
