"""Pipeline de lectura: segmentación del visor → rectificación → detección de dígitos → lectura
(con la separación de m³ y fracción por el color de los dígitos, el modelo de medidor o los
huecos entre dígitos).

read_meter recibe el segmentador y el detector como funciones, así la lógica no depende de
Ultralytics y se puede testear con dobles. load_models arma esas funciones con modelos YOLO.
"""

from dataclasses import dataclass, replace
from typing import Callable

import numpy as np

from .color import red_split, redness
from .gaps import decimal_gap
from .layout import fraction_start, layout_of, model_input
from .reading import Digit, Reading, assemble_reading, best_orientation
from .rectify import bbox_crop, rectify

Segmenter = Callable[[np.ndarray], np.ndarray | None]
DigitDetector = Callable[[np.ndarray], list[Digit]]
# Recibe el visor con su entorno (layout.model_input) y devuelve (modelo o formato, probabilidad)
MeterClassifier = Callable[[np.ndarray], tuple[str, float]]
# Recibe el recorte del visor y los dígitos leídos; devuelve la probabilidad de cada hueco (gaps.py)
GapClassifier = Callable[[np.ndarray, tuple[Digit, ...]], list[float]]

CROP_PAD = 0.05  # el mismo margen que usa scripts/prepare_digits.py para entrenar
MIN_MODEL_PROB = 0.5
# Con esta probabilidad en un hueco, el punto visto corrige a un modelo de medidor sin fracción
GAP_OVERRIDE_PROB = 0.9


@dataclass(frozen=True)
class MeterResult:
    """Resultado de una foto: lectura, polígono del visor, recorte usado, si se giró 180°, el
    modelo de medidor que decidió la fracción (None si no hubo clasificador o dudó) y si el
    decimal lo ubicó el clasificador de huecos."""

    reading: Reading
    polygon: np.ndarray | None
    crop: np.ndarray | None
    flipped: bool
    meter_model: str | None = None
    decimal_from_gaps: bool = False


def read_meter(image: np.ndarray, segment: Segmenter, detect: DigitDetector,
               classify: MeterClassifier | None = None, rectified: bool = True,
               allow_flip: bool = True, gaps: GapClassifier | None = None) -> MeterResult:
    """Lee un medidor a partir de la foto (BGR).

    Con `classify`, el modelo de medidor predicho decide cuántos dígitos son m³ y cuántos
    fracción; si el clasificador duda (probabilidad < MIN_MODEL_PROB), decide el color.
    Con `gaps`, cuando el clasificador duda decide el clasificador de huecos en lugar del color: mira
    cada hueco entre dígitos vecinos y ubica el punto donde lo ve (o no pone punto). No depende de
    conocer el modelo de medidor (ver gaps.py). También corrige al clasificador cuando este asegura
    que el medidor no tiene fracción pero un hueco muestra el punto con probabilidad ≥ GAP_OVERRIDE_PROB.
    Con `rectified=False` el visor se recorta por su caja alineada a los ejes, sin enderezar
    (línea de base del experimento principal).
    Con `allow_flip=False` no se prueba el visor girado 180°: se lee tal como quedó en la foto.
    Sirve para medidores que se fotografían siempre derechos (energía, en pared o pilar), donde
    los dígitos de 7 segmentos girados siguen pareciendo dígitos y la confianza no decide bien.
    """
    polygon = segment(image)
    if polygon is None or len(polygon) < 3:
        return MeterResult(assemble_reading([]), None, None, False)

    crop = rectify(image, polygon, pad=CROP_PAD)[0] if rectified else bbox_crop(image, polygon, pad=CROP_PAD)
    turned = np.ascontiguousarray(crop[::-1, ::-1])
    upright = _with_fraction(assemble_reading(detect(crop)), crop)
    if not allow_flip:
        reading, flipped = upright, False
    else:
        reading, flipped = _choose_orientation(upright, _with_fraction(assemble_reading(detect(turned)), turned))

    used = turned if flipped else crop
    meter_model = None
    if classify is not None:
        label, probability = classify(model_input(image, polygon))
        if probability >= MIN_MODEL_PROB and layout_of(label) is not None:
            meter_model = label
            reading = replace(reading, red_from=fraction_start(len(reading.digits), label))
    from_gaps = False
    without_fraction = meter_model is not None and layout_of(meter_model)[1] == 0
    if gaps is not None and (meter_model is None or without_fraction):
        probs = gaps(used, reading.digits)
        if meter_model is None:
            reading, from_gaps = replace(reading, red_from=decimal_gap(probs)), True
        elif (integer := decimal_gap(probs, GAP_OVERRIDE_PROB)) is not None:
            reading, from_gaps = replace(reading, red_from=integer), True
    return MeterResult(reading, np.asarray(polygon), used, flipped, meter_model, from_gaps)


def _choose_orientation(upright: Reading, flipped: Reading) -> tuple[Reading, bool]:
    """La fracción roja va siempre a la derecha: si solo una orientación la muestra así, es esa.
    Si no decide el color, decide la confianza del detector."""
    if (upright.red_from is None) != (flipped.red_from is None):
        return (upright, False) if upright.red_from is not None else (flipped, True)
    return best_orientation(upright, flipped)


def _with_fraction(reading: Reading, crop: np.ndarray) -> Reading:
    """Marca dónde empiezan los dígitos rojos (fracción de m³) según su color en el recorte."""
    return replace(reading, red_from=red_split([redness(crop, d) for d in reading.digits]))


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
