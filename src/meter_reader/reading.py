"""Armado de la lectura a partir de los dígitos detectados en el visor rectificado.

El visor ya llega horizontal, así que la lectura es el orden de los dígitos de izquierda a
derecha. Quedan dos casos que resolver:
  - rodillos a mitad de giro: el detector puede devolver dos dígitos en la misma posición;
    se conserva el de mayor confianza;
  - visor invertido (180°): la rectificación no lo distingue. Se lee el recorte en ambas
    orientaciones y se elige la de mayor confianza media (best_orientation).
"""

from dataclasses import dataclass

# Dos cajas son la misma posición si se solapan horizontalmente más que esta fracción
# del ancho de la más angosta.
SAME_POSITION_OVERLAP = 0.5


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
    """Lectura armada. confidence es la del dígito más débil; mean_confidence, el promedio."""

    text: str
    confidence: float
    mean_confidence: float
    digits: tuple[Digit, ...]


def _same_position(a: Digit, b: Digit) -> bool:
    overlap = min(a.x2, b.x2) - max(a.x1, b.x1)
    narrowest = min(a.x2 - a.x1, b.x2 - b.x1)
    return narrowest > 0 and overlap / narrowest > SAME_POSITION_OVERLAP


def assemble_reading(digits: list[Digit]) -> Reading:
    """Ordena los dígitos de izquierda a derecha, uno por posición."""
    kept: list[Digit] = []
    for digit in sorted(digits, key=lambda d: d.confidence, reverse=True):
        if not any(_same_position(digit, other) for other in kept):
            kept.append(digit)
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
