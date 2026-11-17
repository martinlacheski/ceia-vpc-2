"""Tests de scripts/evaluate_seg_dice.py: Dice e IoU por foto entre el visor predicho y el real."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_seg_dice.py"


def load_module():
    spec = importlib.util.spec_from_file_location("evaluate_seg_dice", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def square(x, y, side):
    return np.array([[x, y], [x + side, y], [x + side, y + side], [x, y + side]], dtype=np.float32)


def test_identical_masks_score_one():
    module = load_module()
    assert module.overlap(square(10, 10, 20), square(10, 10, 20), (100, 100)) == pytest.approx((1.0, 1.0), abs=0.01)


def test_half_overlap_gives_dice_and_iou():
    module = load_module()
    # dos rectángulos de 20x20 corridos 10 px: intersección 200, cada uno 400
    dice, iou = module.overlap(square(10, 10, 20), square(20, 10, 20), (100, 100))
    assert dice == pytest.approx(2 * 200 / 800, abs=0.03)
    assert iou == pytest.approx(200 / 600, abs=0.03)


def test_missing_prediction_scores_zero():
    module = load_module()
    assert module.overlap(None, square(10, 10, 20), (100, 100)) == (0.0, 0.0)


def test_evaluate_averages_per_photo_and_counts_misses(tmp_path):
    module = load_module()
    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    for name in ("a", "b"):
        Image.new("RGB", (100, 100)).save(tmp_path / "images" / f"{name}.jpg")
        (tmp_path / "labels" / f"{name}.txt").write_text("0 0.1 0.1 0.3 0.1 0.3 0.3 0.1 0.3\n")
    predictions = {"a.jpg": square(10, 10, 20), "b.jpg": None}  # a perfecta, b sin visor

    def segment(image, _calls=iter(["a.jpg", "b.jpg"])):
        return predictions[next(_calls)]

    result = module.evaluate(segment, tmp_path, ["a.jpg", "b.jpg"])

    assert result["fotos"] == 2
    assert result["dice"] == pytest.approx(0.5, abs=0.01)
    assert result["iou"] == pytest.approx(0.5, abs=0.01)
    assert result["sin_visor"] == 0.5
