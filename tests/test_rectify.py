"""Tests de src/meter_reader/rectify.py: enderezar y recortar el visor."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from meter_reader.rectify import rectify, rectify_matrix, transform_points  # noqa: E402


def digit_row(angle_deg: float, n: int = 6, center=(200.0, 150.0), step=20.0) -> np.ndarray:
    """Centros de n dígitos en fila, rotada angle_deg (positivo = sentido horario en la imagen)."""
    a = np.deg2rad(angle_deg)
    direction = np.array([np.cos(a), np.sin(a)])
    offsets = (np.arange(n) - (n - 1) / 2) * step
    return np.array(center) + offsets[:, None] * direction


def display_from(centers: np.ndarray, angle_deg: float) -> np.ndarray:
    """Cuadrilátero del visor alrededor de los dígitos (largo 140, alto 24)."""
    a = np.deg2rad(angle_deg)
    u = np.array([np.cos(a), np.sin(a)]) * 70
    v = np.array([-np.sin(a), np.cos(a)]) * 12
    c = centers.mean(axis=0)
    return np.array([c - u - v, c + u - v, c + u + v, c - u + v])


@pytest.mark.parametrize("angle", [0, 20, -20, 35, -35])
def test_rectified_digits_are_horizontal_and_keep_reading_order(angle):
    centers = digit_row(angle)
    matrix, size = rectify_matrix(display_from(centers, angle), pad=0.1)
    out = transform_points(centers, matrix)

    # Quedan en una misma fila horizontal...
    assert np.ptp(out[:, 1]) < 1.0
    # ...en el mismo orden de lectura (izquierda a derecha)...
    assert np.all(np.diff(out[:, 0]) > 0)
    # ...y dentro del recorte.
    w, h = size
    assert np.all((out >= 0) & (out <= [w, h]))


def test_crop_size_is_display_plus_padding():
    polygon = display_from(digit_row(0), 0)
    _, (w, h) = rectify_matrix(polygon, pad=0.1)
    assert (w, h) == (round(140 * 1.2), round(24 * 1.2))


def test_rectify_returns_crop_of_expected_size():
    image = np.zeros((300, 400, 3), dtype=np.uint8)
    crop, _ = rectify(image, display_from(digit_row(15), 15), pad=0.1)
    assert crop.shape == (round(24 * 1.2), round(140 * 1.2), 3)
