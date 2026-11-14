"""Tests de src/meter_reader/color.py: separación de los dígitos negros (m³) y rojos (fracción)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from meter_reader.color import red_split, redness  # noqa: E402
from meter_reader.reading import Digit  # noqa: E402

WHITE = (240, 240, 240)  # BGR
BLACK_INK = (30, 30, 30)
FADED_RED_INK = (150, 150, 225)  # rojo desteñido, casi rosado
RED_BACKGROUND = (60, 60, 200)


def display(cells: list[tuple[tuple, tuple]]) -> tuple[np.ndarray, list[Digit]]:
    """Visor sintético: cada celda es (fondo, tinta) con un trazo vertical fino como dígito."""
    crop = np.zeros((40, 30 * len(cells), 3), dtype=np.uint8)
    digits = []
    for i, (background, ink) in enumerate(cells):
        x = 30 * i
        crop[:, x:x + 30] = background
        crop[8:32, x + 12:x + 17] = ink
        digits.append(Digit(i, 0.9, x + 2, 4, x + 28, 36))
    return crop, digits


def test_redness_is_low_for_black_and_high_for_red_ink_or_background():
    crop, digits = display([(WHITE, BLACK_INK), (WHITE, FADED_RED_INK), (RED_BACKGROUND, WHITE)])
    black, faded, background = (redness(crop, d) for d in digits)
    assert black < 3
    assert faded > 30 and background > 100


def test_redness_ignores_boxes_outside_the_crop():
    crop, _ = display([(WHITE, BLACK_INK)])
    assert redness(crop, Digit(1, 0.9, 500, 500, 520, 520)) == 0.0


def test_red_split_finds_where_red_digits_start():
    # puntajes medidos en fotos CEML reales
    assert red_split([0, 1, 2, 0, 56, 49]) == 4
    assert red_split([0, 0, 0, 0, 8, 8]) == 4  # rojo desteñido, pero claramente distinto del negro
    assert red_split([14, 7, 9, 25, 82, 82, 91, 102]) == 4  # fondo rojo
    assert red_split([1, 2, 2, 3, 9, 11]) == 4
    assert red_split([0, 0, 0, 10, 3, 27, 14]) == 5


def test_red_split_does_not_invent_a_fraction():
    assert red_split([0, 0, 0, 0]) is None  # todo negro
    assert red_split([7, 1, 0, 0, 1, 0]) is None  # ruido sin un bloque rojo a la derecha
    assert red_split([0, 0, 0, 0, 2, 3]) is None  # diferencia demasiado chica para decidir
    assert red_split([60, 0, 0, 0]) is None  # el rojo tiene que estar a la derecha
    # un solo dígito "rojo" es ruido (un borde del visor): la fracción tiene al menos 2 dígitos
    assert red_split([0, 1, 0, 0, 1, 6]) is None  # caso real CEML m0260, visor girado
    assert red_split([5]) is None and red_split([]) is None
