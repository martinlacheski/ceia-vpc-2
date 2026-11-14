"""Tests de meter_reader.layout: entrada del clasificador y posición de la fracción."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from meter_reader.layout import INPUT_SIZE, fraction_start, layout_of, model_input  # noqa: E402


def test_layout_comes_from_the_end_of_the_model_name():
    assert layout_of("generico_5_2") == (5, 2)
    assert layout_of("abb_4_3") == (4, 3)
    assert layout_of("cyble_5_2") == (5, 2)  # una marca nueva no necesita cambiar el código
    assert layout_of("qn15_5_0") == (5, 0)
    assert layout_of("5_2") == (5, 2)  # el clasificador por formato predice solo la disposición
    assert layout_of("desconocido") is None
    assert layout_of("") is None


def test_fraction_starts_after_the_cubic_meter_digits():
    assert fraction_start(7, "generico_5_2") == 5
    assert fraction_start(8, "azul_4_4") == 4
    # faltan rodillos a la derecha (no se ven): la fracción igual empieza en el quinto
    assert fraction_start(6, "azul_4_4") == 4


def test_no_fraction_for_models_without_red_rollers_or_missing_digits():
    assert fraction_start(5, "qn15_5_0") is None
    assert fraction_start(4, "iso4064_4_0") is None
    assert fraction_start(4, "actaris_4_2") is None  # solo se ven los m³


def test_model_input_centers_the_display_horizontally_with_context():
    image = np.zeros((600, 800, 3), dtype=np.uint8)
    image[280:320, 300:500] = 255  # visor blanco de 200 × 40
    polygon = np.array([[300, 280], [500, 280], [500, 320], [300, 320]])
    out = model_input(image, polygon)
    assert out.shape == (INPUT_SIZE[1], INPUT_SIZE[0], 3)
    white = np.argwhere(out[:, :, 0] > 128)
    center = white.mean(axis=0)
    np.testing.assert_allclose(center, [INPUT_SIZE[1] / 2, INPUT_SIZE[0] / 2], atol=3)
    # el visor ocupa 1/1.6 del ancho: queda entorno a los costados
    width = white[:, 1].max() - white[:, 1].min() + 1
    assert abs(width - INPUT_SIZE[0] / 1.6) < 4


def test_model_input_straightens_a_vertical_display():
    image = np.zeros((800, 600, 3), dtype=np.uint8)
    image[300:500, 280:320] = 255  # visor vertical
    polygon = np.array([[280, 300], [320, 300], [320, 500], [280, 500]])
    white = np.argwhere(model_input(image, polygon)[:, :, 0] > 128)
    height = white[:, 0].max() - white[:, 0].min() + 1
    width = white[:, 1].max() - white[:, 1].min() + 1
    assert width > 4 * height
