"""Separador decimal por los huecos entre dígitos, sin conocer el modelo de medidor.

En agua el decimal se reconoce por el color de los rodillos (color.py), una señal que comparten
todas las marcas. En energía la señal cambia según el medidor: el punto de un LCD, el recuadro o
el color distinto del último rodillo. En vez de reconocer el modelo (layout.py), se mira cada
hueco entre dos dígitos vecinos y un clasificador estima si ahí hay un separador decimal.

Cada recorte va del centro de un dígito al centro del siguiente, con algo de margen vertical, así
incluye el espacio entre ambos y la mitad de cada uno (para ver un cambio de color o de marco).
"""

import cv2
import numpy as np

from .reading import Digit

PATCH_SIZE = (64, 96)  # ancho, alto
VERTICAL_PAD = 0.15  # margen arriba y abajo, como fracción de la altura de los dígitos
MIN_GAP_PROB = 0.5


def gap_patches(crop: np.ndarray, digits: list[Digit] | tuple[Digit, ...]) -> list[np.ndarray]:
    """Un recorte por cada par de dígitos vecinos, de izquierda a derecha, en PATCH_SIZE (BGR)."""
    ordered = sorted(digits, key=lambda d: (d.x1 + d.x2) / 2)
    height, width = crop.shape[:2]
    patches = []
    for left, right in zip(ordered, ordered[1:]):
        x1 = (left.x1 + left.x2) / 2
        x2 = (right.x1 + right.x2) / 2
        y1, y2 = min(left.y1, right.y1), max(left.y2, right.y2)
        pad = VERTICAL_PAD * (y2 - y1)
        cx1, cx2 = max(0, int(x1)), min(width, max(int(x1) + 1, int(round(x2))))
        cy1, cy2 = max(0, int(y1 - pad)), min(height, max(int(y1 - pad) + 1, int(round(y2 + pad))))
        patch = crop[cy1:cy2, cx1:cx2]
        patches.append(cv2.resize(patch, PATCH_SIZE, interpolation=cv2.INTER_AREA))
    return patches


def gap_target(digit_count: int, layout: tuple[int, int]) -> int | None:
    """Índice del hueco decimal según la disposición (enteros, fracción) del medidor.

    None si el medidor no tiene fracción; -1 si la cantidad de dígitos no coincide con la
    disposición (no se sabe qué hueco es: el ejemplo no sirve para entrenar ni evaluar).
    """
    integer, fraction = layout
    if digit_count != integer + fraction:
        return -1
    return integer - 1 if fraction else None


def decimal_gap(probs, threshold: float = MIN_GAP_PROB) -> int | None:
    """Cantidad de dígitos enteros: el punto va después del hueco más probable, si supera el umbral."""
    if len(probs) == 0:
        return None
    best = int(np.argmax(probs))
    return best + 1 if probs[best] >= threshold else None


def load_gap_classifier(weights, device="cpu"):
    """Clasificador de huecos listo para el pipeline: (recorte, dígitos) → probabilidad por hueco.

    Los pesos los guarda scripts/train_gap_model.py. torch se importa acá: el resto del módulo no lo necesita.
    """
    import torch

    from .meter_model import build_model, to_tensor

    checkpoint = torch.load(weights, map_location="cpu")
    net = build_model(checkpoint["arch"], 2, pretrained=False)
    net.load_state_dict(checkpoint["state_dict"])
    net.to(device).eval()

    def classify_gaps(crop: np.ndarray, digits) -> list[float]:
        patches = gap_patches(crop, digits)
        if not patches:
            return []
        with torch.no_grad():
            return net(to_tensor(patches).to(device)).softmax(1)[:, 1].cpu().tolist()

    return classify_gaps
