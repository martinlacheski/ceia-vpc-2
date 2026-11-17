"""Dice e IoU por foto de un segmentador del visor, sobre las fotos de campo CEML.

Complementa a scripts/evaluate_seg.py (mAP y recall de Ultralytics) con las dos métricas de
superposición de máscaras más usadas en segmentación:
  - IoU = |A ∩ B| / |A ∪ B|
  - Dice = 2 |A ∩ B| / (|A| + |B|)
donde A es el visor que usa el pipeline (la instancia de mayor confianza, como en
meter_reader.pipeline.load_models) y B el polígono revisado de labels/. Se promedian por foto;
una foto sin visor predicho cuenta como 0 en las dos y se informa aparte (sin_visor).

Evalúa las 324 fotos de agua (data/ceml) y las 201 de la muestra de energía
(data/ceml_energia, muestra=si). Guarda experiments/<nombre>/dice.json.

Uso (con el entorno que tiene ultralytics, por ejemplo .venv-rocm):
  python scripts/evaluate_seg_dice.py --model models/toloka_yfzw4_yolov8n_seg.pt
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train_meter_model import read_polygon  # noqa: E402

WATER = ROOT / "data/ceml"
ENERGY = ROOT / "data/ceml_energia"


def overlap(predicted: np.ndarray | None, truth: np.ndarray, shape: tuple[int, int]) -> tuple[float, float]:
    """(Dice, IoU) entre dos polígonos en píxeles, rasterizados en una imagen de tamaño shape (alto, ancho)."""
    if predicted is None or len(predicted) < 3:
        return 0.0, 0.0
    masks = []
    for polygon in (predicted, truth):
        mask = np.zeros(shape[:2], np.uint8)
        cv2.fillPoly(mask, [np.round(polygon).astype(np.int32)], 1)
        masks.append(mask.astype(bool))
    intersection = np.logical_and(*masks).sum()
    total = masks[0].sum() + masks[1].sum()
    union = np.logical_or(*masks).sum()
    if total == 0:
        return 0.0, 0.0
    return float(2 * intersection / total), float(intersection / union)


def evaluate(segment: Callable[[np.ndarray], np.ndarray | None], root: Path, files: list[str]) -> dict:
    """Promedio por foto de Dice e IoU; las fotos sin polígono real se saltean."""
    dices, ious, missing = [], [], 0
    for name in files:
        image = cv2.imread(str(root / "images" / name))
        if image is None:
            continue
        truth = read_polygon(root / "labels" / f"{Path(name).stem}.txt", image.shape[1], image.shape[0])
        if truth is None:
            continue
        predicted = segment(image)
        missing += predicted is None or len(predicted) < 3
        dice, iou = overlap(predicted, truth, image.shape)
        dices.append(dice)
        ious.append(iou)
    n = len(dices)
    return {"fotos": n, "dice": round(float(np.mean(dices)), 4) if n else None,
            "iou": round(float(np.mean(ious)), 4) if n else None,
            "sin_visor": round(missing / n, 4) if n else None}


def energy_sample(root: Path = ENERGY) -> list[str]:
    with (root / "readings.csv").open(newline="", encoding="utf-8") as f:
        return [r["file"] for r in csv.DictReader(f) if r.get("muestra", "").strip().lower() == "si"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT / "src"))
    from meter_reader.pipeline import load_models

    # load_models pide un detector de dígitos; acá solo se usa el segmentador.
    segment, _ = load_models(str(args.model), str(ROOT / "models/digits_yolov8n.pt"), seg_imgsz=args.imgsz)
    with (WATER / "metadata.csv").open(newline="", encoding="utf-8") as f:
        water_files = [r["file"] for r in csv.DictReader(f)]
    result = {"ceml_agua": evaluate(segment, WATER, water_files),
              "ceml_energia_muestra": evaluate(segment, ENERGY, energy_sample()),
              "imgsz": args.imgsz}
    out = ROOT / "experiments" / args.model.stem / "dice.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
