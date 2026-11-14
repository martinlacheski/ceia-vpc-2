"""Tests de scripts/evaluate_pipeline.py: verdad de campo y métricas (sin modelos)."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_pipeline.py"


def load_module():
    spec = importlib.util.spec_from_file_location("evaluate_pipeline", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ground_truth_keeps_legible_rows_and_marks_unreadable_fractions(tmp_path):
    module = load_module()
    sheet = tmp_path / "readings.csv"
    sheet.write_text(
        "file,reading,legible,notes,modelo\n"
        "a.jpg,00053.25,si,,generico_5_2\n"
        'b.jpg,"1328,97",si,,actaris_4_2\n'  # coma en lugar de punto
        "c.jpg,2178.,si,no se ven,abb_4_3\n"  # fracción ilegible
        "d.jpg,0061.52,no,no se puede descifrar,actaris_4_2\n"
        "e.jpg,,si,,\n"
    )
    truth = module.load_ground_truth(sheet)
    assert [(t["file"], t["reading"], t["m3"], t["fraction_known"]) for t in truth] == [
        ("a.jpg", "00053.25", "00053", True),
        ("b.jpg", "1328.97", "1328", True),
        ("c.jpg", "2178.", "2178", False),
    ]


def test_edit_distance():
    module = load_module()
    assert module.edit_distance("0005325", "0005325") == 0
    assert module.edit_distance("005325", "0005325") == 1  # falta un dígito
    assert module.edit_distance("0005335", "0005325") == 1  # un dígito mal
    assert module.edit_distance("", "123") == 3


def test_score_exact_m3_cer_and_missing():
    module = load_module()
    truth = [
        {"file": "a", "reading": "00053.25", "m3": "00053", "fraction_known": True},
        {"file": "b", "reading": "1328.97", "m3": "1328", "fraction_known": True},
        {"file": "c", "reading": "2178.", "m3": "2178", "fraction_known": False},
        {"file": "d", "reading": "0470", "m3": "0470", "fraction_known": True},
    ]
    predictions = {
        "a": ("00053.25", 0.9),  # exacta
        "b": ("1328.93", 0.4),  # m³ bien, fracción mal, baja confianza
        "c": ("2178.123", 0.8),  # solo cuenta para m³
        "d": None,  # sin lectura
    }
    metrics = module.score(truth, predictions)
    assert metrics["fotos"] == 4 and metrics["fotos_fraccion_legible"] == 3
    assert metrics["exacta"] == pytest.approx(1 / 3, abs=1e-4)
    assert metrics["m3_exactos"] == 0.75  # a, b y c
    assert metrics["sin_lectura"] == 0.25
    # dígitos reales: 7 + 6 + 4 = 17; errores: 0 + 1 + 4 (sin lectura)
    assert metrics["cer"] == pytest.approx(5 / 17, abs=1e-4)
    assert metrics["cobertura_confianza"] == pytest.approx(1 / 3, abs=1e-4)
    assert metrics["exacta_con_confianza"] == 1.0


def test_m3_compares_as_a_number_but_a_missing_dot_counts_all_digits():
    module = load_module()
    truth = [{"file": "a", "reading": "00053.25", "m3": "00053", "fraction_known": True}]
    assert module.score(truth, {"a": ("0053.25", 0.9)})["m3_exactos"] == 1.0  # cero de más a la izquierda
    assert module.score(truth, {"a": ("0005325", 0.9)})["m3_exactos"] == 0.0  # sin punto: 5325 m³


def test_ocr_digits_orders_boxes_left_to_right_and_keeps_only_digits():
    module = load_module()
    results = [
        ([[60, 0], [90, 0], [90, 20], [60, 20]], "25", 0.7),
        ([[0, 0], [50, 0], [50, 20], [0, 20]], "00 053", 0.9),
        ([[95, 0], [99, 0], [99, 20], [95, 20]], "m", 0.2),
    ]
    assert module.ocr_digits(results) == ("0005325", 0.7)
    assert module.ocr_digits([]) == ("", 0.0)


def test_with_dot():
    module = load_module()
    assert module.with_dot("0005325", 5) == "00053.25"
    assert module.with_dot("0470", None) == "0470"
