"""Pipeline de lectura: segmentación del visor → rectificación → detección de dígitos → lectura.

read_meter recibe el segmentador y el detector como funciones, así la lógica no depende de
Ultralytics y se puede testear con dobles. load_models arma esas funciones con modelos YOLO.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .reading import Digit, Reading, assemble_reading, best_orientation
from .rectify import rectify

Segmenter = Callable[[np.ndarray], np.ndarray | None]
DigitDetector = Callable[[np.ndarray], list[Digit]]

CROP_PAD = 0.05  # el mismo margen que usa scripts/prepare_digits.py para entrenar


@dataclass(frozen=True)
class MeterResult:
    """Resultado de una foto: lectura, polígono del visor, recorte usado y si se giró 180°."""

    reading: Reading
    polygon: np.ndarray | None
    crop: np.ndarray | None
    flipped: bool


def read_meter(image: np.ndarray, segment: Segmenter, detect: DigitDetector) -> MeterResult:
    """Lee un medidor a partir de la foto (BGR)."""
    polygon = segment(image)
    if polygon is None or len(polygon) < 3:
        return MeterResult(assemble_reading([]), None, None, False)

    crop, _ = rectify(image, polygon, pad=CROP_PAD)
    turned = np.ascontiguousarray(crop[::-1, ::-1])
    reading, flipped = best_orientation(assemble_reading(detect(crop)),
                                        assemble_reading(detect(turned)))
    return MeterResult(reading, np.asarray(polygon), turned if flipped else crop, flipped)


def load_models(seg_weights: str, digit_weights: str, seg_imgsz: int = 640, digit_imgsz: int = 416,
                conf: float = 0.25, device: str | int | None = None) -> tuple[Segmenter, DigitDetector]:
    """Arma el segmentador y el detector a partir de pesos YOLO."""
    from ultralytics import YOLO

    seg_model, digit_model = YOLO(seg_weights), YOLO(digit_weights)

    def segment(image: np.ndarray) -> np.ndarray | None:
        result = seg_model.predict(image, imgsz=seg_imgsz, conf=conf, device=device, verbose=False)[0]
        if result.masks is None or len(result.masks) == 0:
            return None
        return result.masks.xy[int(result.boxes.conf.argmax())]

    def detect(crop: np.ndarray) -> list[Digit]:
        result = digit_model.predict(crop, imgsz=digit_imgsz, conf=conf, device=device, verbose=False)[0]
        boxes = result.boxes
        return [Digit(int(c), float(p), *map(float, xyxy))
                for c, p, xyxy in zip(boxes.cls, boxes.conf, boxes.xyxy)]

    return segment, detect
