"""Separa los dígitos negros (metros cúbicos) de los rojos (fracción) por color.

El detector de dígitos no distingue colores: las 10 clases son 0–9. En los medidores de
rodillo, los m³ (lo que se factura) van en negro y la fracción en rojo, a la derecha. El
rojo aparece de dos formas: tinta roja sobre blanco (a menudo desteñida, casi rosada) o
números blancos sobre fondo rojo. Un umbral fijo de tono y saturación falla con la tinta
desteñida, así que se usa el exceso de rojo, R − max(G, B): es ~0 en negros, grises y
blancos, y positivo en cualquier rojo. Cada dígito se compara con los demás del mismo visor.
"""

from typing import Sequence

import numpy as np

from .reading import Digit

TOP_FRACTION = 0.2  # se promedia el 20% de píxeles más rojos: el trazo fino pesa aunque el fondo sea blanco
MIN_GAP = 4.0       # el rojo más débil tiene que superar al negro más fuerte por al menos esto
MIN_RED = 5.0       # y el grupo rojo tiene que tener un exceso medio mínimo (si no, es ruido)
MIN_RED_DIGITS = 2  # la fracción tiene al menos 2 dígitos; un solo "rojo" suele ser un borde del visor


def redness(crop: np.ndarray, digit: Digit) -> float:
    """Exceso de rojo del dígito (0–255) dentro de su caja en el recorte BGR."""
    height, width = crop.shape[:2]
    x1, x2 = max(0, int(round(digit.x1))), min(width, int(round(digit.x2)))
    y1, y2 = max(0, int(round(digit.y1))), min(height, int(round(digit.y2)))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    roi = crop[y1:y2, x1:x2].astype(np.int16)
    excess = np.clip(roi[..., 2] - np.maximum(roi[..., 0], roi[..., 1]), 0, None).ravel()
    top = np.sort(excess)[::-1][: max(1, int(len(excess) * TOP_FRACTION))]
    return float(top.mean())


def red_split(scores: Sequence[float]) -> int | None:
    """Índice del primer dígito rojo (los rojos van juntos a la derecha), o None si no hay.

    Elige el corte con mayor diferencia entre el rojo más débil y el negro más fuerte, y lo
    acepta solo si esa diferencia y el exceso medio del grupo rojo superan los mínimos y hay
    al menos MIN_RED_DIGITS dígitos rojos.
    """
    best, best_gap = None, MIN_GAP
    for k in range(1, len(scores) - MIN_RED_DIGITS + 1):
        black, red = scores[:k], scores[k:]
        gap = min(red) - max(black)
        if gap >= best_gap and float(np.mean(red)) >= MIN_RED:
            best, best_gap = k, gap
    return best
