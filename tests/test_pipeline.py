"""Tests de src/meter_reader/pipeline.py con segmentador y detector simulados."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from meter_reader.pipeline import read_meter  # noqa: E402
from meter_reader.reading import Digit  # noqa: E402

POLYGON = np.array([[100, 100], [300, 100], [300, 140], [100, 140]], dtype=float)


def image_with_marker() -> np.ndarray:
    """Imagen gris con una marca blanca en el extremo izquierdo del visor."""
    image = np.full((300, 400, 3), 90, dtype=np.uint8)
    image[110:130, 105:120] = 255
    return image


def detector_reading(text: str):
    """Detector que solo lee bien si la marca blanca quedó a la izquierda del recorte."""
    def detect(crop: np.ndarray) -> list[Digit]:
        w = crop.shape[1]
        upright = crop[:, : w // 4].mean() > crop[:, -w // 4:].mean()
        conf = 0.9 if upright else 0.3
        return [Digit(int(c), conf, i * 25, 0, i * 25 + 20, 30) for i, c in enumerate(text)]
    return detect


def test_reads_meter_and_reports_polygon():
    result = read_meter(image_with_marker(), lambda img: POLYGON, detector_reading("00348"))
    assert result.reading.text == "00348"
    assert result.flipped is False
    assert np.allclose(result.polygon, POLYGON)


def test_upside_down_display_is_read_in_the_flipped_orientation():
    image = image_with_marker()[::-1, ::-1].copy()  # foto girada 180°
    polygon = np.array([[400, 300]]) - POLYGON
    result = read_meter(image, lambda img: polygon, detector_reading("00348"))
    assert result.flipped is True
    assert result.reading.mean_confidence == 0.9


def test_no_display_gives_empty_result():
    result = read_meter(image_with_marker(), lambda img: None, detector_reading("1"))
    assert result.reading.text == "" and result.polygon is None and result.crop is None
