"""Armado de la lectura a partir de los dígitos detectados en el visor rectificado.

El visor ya llega horizontal, así que la lectura es el orden de los dígitos de izquierda a
derecha. Quedan tres casos que resolver:
  - detecciones espurias angostas: el borde del visor o un rodillo cortado en el extremo del
    recorte pueden detectarse como dígito; todos los rodillos de un visor tienen casi el mismo
    ancho, así que se descartan las cajas mucho más angostas que la mediana;
  - rodillos a mitad de giro: el detector puede devolver dos dígitos en la misma posición;
    se conserva el de mayor confianza;
  - visor invertido (180°): la rectificación no lo distingue. Se lee el recorte en ambas
    orientaciones y se elige la de mayor confianza media (best_orientation); antes, el
    pipeline prefiere la orientación que deja la fracción roja a la derecha (ver color.py).
"""

from dataclasses import dataclass

# Dos cajas son la misma posición si se solapan horizontalmente más que esta fracción
# del ancho de la más angosta.
SAME_POSITION_OVERLAP = 0.5

# Se descarta una caja más angosta que esta fracción del ancho mediano de las demás.
MIN_WIDTH_RATIO = 0.6
# Con menos cajas que esto, la mediana no es representativa y no se filtra por ancho.
MIN_BOXES_FOR_WIDTH_FILTER = 3


@dataclass(frozen=True)
class Digit:
    """Un dígito detectado: clase, confianza y caja (x1, y1, x2, y2) en píxeles del recorte."""

    value: int
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class Reading:
    """Lectura armada. confidence es la del dígito más débil; mean_confidence, el promedio.

    red_from: posición del primer dígito rojo (fracción de m³), o None si no se distinguió.
    """

    text: str
    confidence: float
    mean_confidence: float
    digits: tuple[Digit, ...]
    red_from: int | None = None

    @property
    def formatted(self) -> str:
        """Lectura con un punto entre los m³ (negros) y la fracción (rojos): 0127.34."""
        if not self.red_from:
            return self.text
        return f"{self.text[:self.red_from]}.{self.text[self.red_from:]}"


def _same_position(a: Digit, b: Digit) -> bool:
    overlap = min(a.x2, b.x2) - max(a.x1, b.x1)
    narrowest = min(a.x2 - a.x1, b.x2 - b.x1)
    return narrowest > 0 and overlap / narrowest > SAME_POSITION_OVERLAP


def _drop_narrow(digits: list[Digit]) -> list[Digit]:
    if len(digits) < MIN_BOXES_FOR_WIDTH_FILTER:
        return digits
    widths = sorted(d.x2 - d.x1 for d in digits)
    median = widths[len(widths) // 2]
    return [d for d in digits if d.x2 - d.x1 >= MIN_WIDTH_RATIO * median]


def assemble_reading(digits: list[Digit]) -> Reading:
    """Ordena los dígitos de izquierda a derecha, uno por posición."""
    kept: list[Digit] = []
    for digit in sorted(digits, key=lambda d: d.confidence, reverse=True):
        if not any(_same_position(digit, other) for other in kept):
            kept.append(digit)
    kept = _drop_narrow(kept)
    kept.sort(key=lambda d: (d.x1 + d.x2) / 2)
    if not kept:
        return Reading("", 0.0, 0.0, ())
    confidences = [d.confidence for d in kept]
    return Reading("".join(str(d.value) for d in kept), min(confidences),
                   sum(confidences) / len(confidences), tuple(kept))


def best_orientation(upright: Reading, flipped: Reading) -> tuple[Reading, bool]:
    """Elige entre la lectura del recorte tal cual y la del recorte girado 180°.

    Devuelve la lectura elegida y si corresponde al recorte girado.
    """
    if flipped.mean_confidence > upright.mean_confidence:
        return flipped, True
    return upright, False
