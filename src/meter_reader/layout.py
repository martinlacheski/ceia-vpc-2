"""Modelos de medidor y su disposición de dígitos (cuántos son m³ y cuántos fracción).

Cada modelo de medidor tiene una cantidad fija de rodillos negros (m³) y rojos (fracción),
y se reconoce a simple vista por el cuerpo, el logo y el marco del visor. Un clasificador
predice el modelo a partir del visor con su entorno (model_input) y su nombre dice dónde va
el punto, sin depender de que el rojo se vea en la foto (en algunos ABB la fracción es blanca).

El nombre de cada modelo (columna `modelo` de data/ceml/readings.csv) termina en
`_<dígitos de m³>_<dígitos de fracción>`: `generico_5_2`, `abb_4_3`, `qn15_5_0`, ...
"""

import re

import cv2
import numpy as np

from .rectify import rectify_matrix

_LAYOUT = re.compile(r"(?:^|_)(\d)_(\d)$")  # "generico_5_2" o solo el formato, "5_2"

# Entrada del clasificador: el visor horizontal, centrado, con entorno para ver logo y marco
INPUT_SIZE = (384, 192)  # ancho, alto
CONTEXT_WIDTH = 1.6  # ancho del recorte respecto del lado largo del visor


def model_input(image: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Recorte del visor enderezado con su entorno, en INPUT_SIZE (BGR).

    El visor queda horizontal (puede quedar girado 180°: el clasificador se entrena con
    ambas orientaciones) y centrado en un recorte 2:1 de CONTEXT_WIDTH veces su largo.
    """
    matrix, (width, height) = rectify_matrix(polygon, pad=0)
    out_w = max(1, round(width * CONTEXT_WIDTH))
    out_h = max(round(out_w * INPUT_SIZE[1] / INPUT_SIZE[0]), height)
    shifted = matrix.copy()
    shifted[:, 2] += [(out_w - width) / 2, (out_h - height) / 2]
    crop = cv2.warpAffine(image, shifted, (out_w, out_h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)
    return cv2.resize(crop, INPUT_SIZE, interpolation=cv2.INTER_AREA)


def layout_of(model: str) -> tuple[int, int] | None:
    """(dígitos de m³, dígitos de fracción) según el nombre del modelo, o None si no lo indica."""
    match = _LAYOUT.search(model or "")
    return (int(match[1]), int(match[2])) if match else None


def fraction_start(digit_count: int, model: str) -> int | None:
    """Índice del primer dígito de fracción según el modelo, o None si no hay fracción.

    Los rodillos se leen de izquierda a derecha: los primeros son los m³. Si se detectaron
    tantos dígitos como m³ o menos (faltan los de la derecha), no se marca fracción.
    """
    integer, fraction = layout_of(model)
    if fraction == 0 or digit_count <= integer:
        return None
    return integer
