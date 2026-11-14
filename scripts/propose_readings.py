"""Propone la lectura de cada foto CEML en data/ceml/readings.csv para verificarla a mano.

Corre el pipeline completo (segmentación → rectificación → dígitos) y completa las columnas
`propuesta` y `confianza`. La propuesta lleva un punto entre los m³ (dígitos negros) y la
fracción (rojos), el mismo formato que se usa en `reading`. No toca `reading`, `legible` ni `notes`: esas las completa la
persona que verifica, mirando cada foto. La propuesta acelera la carga pero puede sesgarla;
el valor de `reading` tiene que salir de la foto, no de copiar la propuesta.

Uso (con el entorno que tiene ultralytics, por ejemplo .venv-rocm):
  python scripts/propose_readings.py
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
PROPOSAL_COLUMNS = ["propuesta", "confianza"]

# Devuelve (lectura, confianza) o None si no se encontró el visor o ningún dígito.
Reader = Callable[[Path], tuple[str, float] | None]


def propose(sheet: Path, images_dir: Path, read: Reader) -> None:
    """Completa propuesta/confianza en la planilla, conservando las columnas manuales."""
    with sheet.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    fields += [c for c in PROPOSAL_COLUMNS if c not in fields]

    for row in rows:
        row["propuesta"], row["confianza"] = "", ""
        image = images_dir / row["file"]
        if not image.exists():
            continue
        result = read(image)
        if result is not None:
            row["propuesta"], row["confianza"] = result[0], f"{result[1]:.2f}"

    with sheet.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def pipeline_reader(seg_weights: Path, digit_weights: Path, meter_weights: Path | None = None) -> Reader:
    """Lector basado en el pipeline del proyecto (YOLO y, si hay pesos, el clasificador del modelo de medidor)."""
    import cv2

    sys.path.insert(0, str(ROOT / "src"))
    from meter_reader.pipeline import load_models, read_meter

    segment, detect = load_models(str(seg_weights), str(digit_weights))
    classify = None
    if meter_weights is not None and meter_weights.exists():
        from meter_reader.meter_model import load_classifier

        classify = load_classifier(meter_weights)

    def read(path: Path) -> tuple[str, float] | None:
        image = cv2.imread(str(path))
        if image is None:
            return None
        reading = read_meter(image, segment, detect, classify).reading
        return (reading.formatted, reading.confidence) if reading.text else None

    return read


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sheet", type=Path, default=ROOT / "data/ceml/readings.csv")
    parser.add_argument("--images", type=Path, default=ROOT / "data/ceml/images")
    parser.add_argument("--seg-weights", type=Path, default=ROOT / "models/toloka_yfzw4_yolov8n_seg.pt")
    parser.add_argument("--digit-weights", type=Path, default=ROOT / "models/digits_yolov8n.pt")
    parser.add_argument("--meter-weights", type=Path, default=ROOT / "models/meter_layout_mnv3s.pt",
                        help="clasificador del modelo de medidor; si no existe, la fracción sale del color")
    args = parser.parse_args()
    propose(args.sheet, args.images, pipeline_reader(args.seg_weights, args.digit_weights, args.meter_weights))
    print(f"Propuestas guardadas en {args.sheet}")


if __name__ == "__main__":
    main()
