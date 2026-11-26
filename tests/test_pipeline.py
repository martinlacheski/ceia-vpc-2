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


def test_without_flip_the_display_is_read_as_photographed():
    # Medidores que se fotografían siempre derechos (energía, en pared o pilar): los dígitos de
    # 7 segmentos girados 180° siguen pareciendo dígitos, así que la confianza no sirve para elegir.
    image = image_with_marker()[::-1, ::-1].copy()
    polygon = np.array([[400, 300]]) - POLYGON
    result = read_meter(image, lambda img: polygon, detector_reading("00348"), allow_flip=False)
    assert result.flipped is False
    assert result.reading.mean_confidence == 0.3


def test_no_display_gives_empty_result():
    result = read_meter(image_with_marker(), lambda img: None, detector_reading("1"))
    assert result.reading.text == "" and result.polygon is None and result.crop is None


def test_red_digits_on_the_right_become_the_fraction():
    image = image_with_marker()
    image[100:140, 250:300] = (60, 60, 200)  # los dos últimos rodillos sobre fondo rojo (BGR)

    def detect(crop):
        w = crop.shape[1]
        step = w / 6
        return [Digit(v, 0.9, i * step + 2, 2, (i + 1) * step - 2, crop.shape[0] - 2)
                for i, v in enumerate([0, 0, 3, 4, 8, 1])]

    result = read_meter(image, lambda img: POLYGON, detect)
    assert result.reading.text == "003481"
    assert result.reading.red_from == 4
    assert result.reading.formatted == "0034.81"


def test_red_fraction_on_the_right_decides_orientation_over_confidence():
    # Caso real (CEML m0174): la versión girada tenía apenas más confianza (0,837 contra 0,829)
    image = image_with_marker()
    image[100:140, 250:300] = (60, 60, 200)  # fracción roja a la derecha del visor derecho

    def detect(crop):
        w = crop.shape[1]
        red_on_left = crop[:, : w // 4, 2].mean() > crop[:, -w // 4:, 2].mean() + 30
        conf = 0.837 if red_on_left else 0.829  # el detector prefiere por poco la orientación equivocada
        step = w / 6
        return [Digit(v, conf, i * step + 2, 2, (i + 1) * step - 2, crop.shape[0] - 2)
                for i, v in enumerate([0, 0, 6, 4, 5, 7])]

    result = read_meter(image, lambda img: POLYGON, detect)
    assert result.flipped is False
    assert result.reading.formatted == "0064.57"


def six_digit_detector(values=(0, 0, 3, 4, 8, 1)):
    def detect(crop):
        step = crop.shape[1] / len(values)
        return [Digit(v, 0.9, i * step + 2, 2, (i + 1) * step - 2, crop.shape[0] - 2)
                for i, v in enumerate(values)]
    return detect


def test_meter_model_places_the_fraction_when_the_red_is_not_visible():
    seen = []

    def classify(image):
        seen.append(image.shape)
        return "4_2", 0.9

    result = read_meter(image_with_marker(), lambda img: POLYGON, six_digit_detector(), classify)
    assert result.reading.formatted == "0034.81"
    assert result.meter_model == "4_2"
    assert seen == [(192, 384, 3)]  # recibe el visor con su entorno (layout.model_input)


def test_meter_model_without_fraction_removes_a_color_split():
    image = image_with_marker()
    image[100:140, 250:300] = (60, 60, 200)  # rojo que no es fracción (por ejemplo, óxido)
    result = read_meter(image, lambda img: POLYGON, six_digit_detector(), lambda img: ("qn15_6_0", 0.9))
    assert result.reading.red_from is None
    assert result.reading.formatted == "003481"


def test_unsure_meter_model_falls_back_to_the_color_rule():
    image = image_with_marker()
    image[100:140, 250:300] = (60, 60, 200)
    result = read_meter(image, lambda img: POLYGON, six_digit_detector(), lambda img: ("5_1", 0.3))
    assert result.reading.formatted == "0034.81"  # decide el color
    assert result.meter_model is None


def test_unsure_meter_model_with_gap_classifier_lets_the_gaps_place_the_decimal():
    image = image_with_marker()
    image[100:140, 250:300] = (60, 60, 200)  # el color diría otra cosa: decide el clasificador de huecos
    calls = []

    def gaps(crop, digits):
        calls.append((crop.shape, len(digits)))
        return [0.1, 0.1, 0.1, 0.1, 0.8]  # el punto va en el último hueco: 5 enteros

    result = read_meter(image, lambda img: POLYGON, six_digit_detector(), lambda img: ("5_1", 0.3), gaps=gaps)
    assert result.reading.formatted == "00348.1"
    assert result.meter_model is None
    assert result.decimal_from_gaps is True
    assert calls == [(result.crop.shape, 6)]  # recibe el recorte usado y los dígitos leídos


def test_gap_classifier_without_a_likely_gap_leaves_no_decimal():
    result = read_meter(image_with_marker(), lambda img: POLYGON, six_digit_detector(),
                        lambda img: ("5_1", 0.3), gaps=lambda crop, digits: [0.2] * 5)
    assert result.reading.formatted == "003481"
    assert result.decimal_from_gaps is True


def test_confident_meter_model_does_not_use_the_gap_classifier():
    def gaps(crop, digits):
        raise AssertionError("no debería consultarse")

    result = read_meter(image_with_marker(), lambda img: POLYGON, six_digit_detector(),
                        lambda img: ("4_2", 0.9), gaps=gaps)
    assert result.reading.formatted == "0034.81"
    assert result.decimal_from_gaps is False


def test_a_clear_dot_overrides_a_meter_model_without_fraction():
    # El clasificador está seguro de que no hay decimal, pero el punto se ve en un hueco
    result = read_meter(image_with_marker(), lambda img: POLYGON, six_digit_detector(),
                        lambda img: ("6_0", 0.93), gaps=lambda crop, digits: [0.0, 0.0, 0.0, 0.0, 0.97])
    assert result.reading.formatted == "00348.1"
    assert result.meter_model == "6_0"
    assert result.decimal_from_gaps is True


def test_an_unclear_dot_does_not_override_a_meter_model_without_fraction():
    result = read_meter(image_with_marker(), lambda img: POLYGON, six_digit_detector(),
                        lambda img: ("6_0", 0.93), gaps=lambda crop, digits: [0.0, 0.0, 0.0, 0.0, 0.8])
    assert result.reading.formatted == "003481"
    assert result.decimal_from_gaps is False


def test_without_classifier_the_meter_model_is_unknown():
    result = read_meter(image_with_marker(), lambda img: POLYGON, six_digit_detector())
    assert result.meter_model is None


def test_bbox_crop_keeps_the_display_tilted_inside_its_axis_aligned_box():
    # visor de 200 × 40 girado 45°: rectificado queda alargado; por caja, casi cuadrado
    angle = np.deg2rad(45)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    corners = np.array([[-100, -20], [100, -20], [100, 20], [-100, 20]]) @ rotation.T + [200, 150]
    image = np.full((300, 400, 3), 90, dtype=np.uint8)
    rectified = read_meter(image, lambda img: corners, six_digit_detector())
    boxed = read_meter(image, lambda img: corners, six_digit_detector(), rectified=False)
    h, w = rectified.crop.shape[:2]
    assert w > 3 * h
    h, w = boxed.crop.shape[:2]
    assert abs(w - h) <= 2
    x0, y0 = corners.min(axis=0)
    x1, y1 = corners.max(axis=0)
    assert abs(w - (x1 - x0) * 1.1) <= 3  # la caja con el mismo margen relativo (CROP_PAD)


def test_best_polygon_takes_the_most_confident_segmentation_mask():
    from types import SimpleNamespace
    from meter_reader.pipeline import best_polygon
    low, high = np.zeros((5, 2)), np.ones((6, 2))
    result = SimpleNamespace(obb=None, masks=SimpleNamespace(xy=[low, high]),
                             boxes=SimpleNamespace(conf=np.array([0.3, 0.8])))
    assert np.array_equal(best_polygon(result), high)


def test_best_polygon_takes_the_corners_of_the_most_confident_oriented_box():
    # Un modelo OBB devuelve las 4 esquinas del rectángulo rotado: son el polígono del visor.
    from types import SimpleNamespace
    from meter_reader.pipeline import best_polygon
    corners = np.array([[[0, 0], [10, 0], [10, 2], [0, 2]], [[5, 5], [25, 5], [25, 9], [5, 9]]], dtype=float)
    result = SimpleNamespace(obb=SimpleNamespace(xyxyxyxy=corners, conf=np.array([0.4, 0.9])),
                             masks=None, boxes=None)
    assert np.array_equal(best_polygon(result), corners[1])


def test_best_polygon_is_none_without_detections():
    from types import SimpleNamespace
    from meter_reader.pipeline import best_polygon
    assert best_polygon(SimpleNamespace(obb=None, masks=None, boxes=None)) is None
    empty_obb = SimpleNamespace(xyxyxyxy=np.zeros((0, 4, 2)), conf=np.zeros(0))
    assert best_polygon(SimpleNamespace(obb=empty_obb, masks=None, boxes=None)) is None
