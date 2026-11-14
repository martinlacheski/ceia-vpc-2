"""Tests de src/meter_reader/reading.py: armar la lectura a partir de las cajas de dígitos."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from meter_reader.reading import Digit, assemble_reading, best_orientation  # noqa: E402


def row(text: str, conf: float = 0.9, step: float = 30.0) -> list[Digit]:
    """Dígitos en fila, de izquierda a derecha, con cajas de 20 px de ancho."""
    return [Digit(int(c), conf, i * step, 0.0, i * step + 20, 40.0) for i, c in enumerate(text)]


def test_digits_are_read_left_to_right():
    digits = row("00348127")
    assert assemble_reading(digits[::-1]).text == "00348127"


def test_overlapping_boxes_keep_the_most_confident_digit():
    # Rodillo a mitad de giro: dos detecciones en la misma posición.
    digits = row("0034") + [Digit(9, 0.95, 120, 0, 140, 40), Digit(0, 0.40, 122, 0, 142, 40)]
    assert assemble_reading(digits).text == "00349"


def test_confidence_is_the_weakest_digit():
    digits = row("123", conf=0.9)
    digits[1] = Digit(2, 0.5, 30, 0, 50, 40)
    assert assemble_reading(digits).confidence == 0.5


def test_empty_detection_gives_empty_reading():
    reading = assemble_reading([])
    assert reading.text == "" and reading.confidence == 0.0


def test_best_orientation_prefers_the_more_confident_reading():
    upright = assemble_reading(row("0278366", conf=0.9))
    flipped = assemble_reading(row("9927820", conf=0.3))
    assert best_orientation(upright, flipped) == (upright, False)
    assert best_orientation(flipped, upright) == (upright, True)
