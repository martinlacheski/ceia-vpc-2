"""Tests de meter_reader.gaps: recortes de los huecos entre dígitos y posición del decimal."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from meter_reader.gaps import PATCH_SIZE, decimal_gap, gap_patches, gap_target  # noqa: E402
from meter_reader.reading import Digit  # noqa: E402


def _digits(xs: list[float], y1: float = 10, y2: float = 50, width: float = 20) -> list[Digit]:
    return [Digit(i % 10, 0.9, x, y1, x + width, y2) for i, x in enumerate(xs)]


def test_one_patch_per_pair_of_neighbouring_digits_in_left_to_right_order():
    crop = np.zeros((60, 200, 3), dtype=np.uint8)
    crop[:, 75:85] = 255  # solo el hueco entre el segundo y el tercer dígito tiene blanco
    patches = gap_patches(crop, _digits([90, 10, 50]))  # desordenados: se ordenan por x
    assert len(patches) == 2
    assert all(p.shape == (PATCH_SIZE[1], PATCH_SIZE[0], 3) for p in patches)
    assert patches[0].max() == 0
    assert patches[1].max() == 255


def test_no_patches_with_fewer_than_two_digits():
    crop = np.zeros((60, 200, 3), dtype=np.uint8)
    assert gap_patches(crop, _digits([10])) == []
    assert gap_patches(crop, []) == []


def test_patches_stay_inside_the_crop_at_the_borders():
    crop = np.full((40, 60, 3), 7, dtype=np.uint8)
    patches = gap_patches(crop, _digits([0, 40], y1=-5, y2=45))
    assert len(patches) == 1 and patches[0].min() == 7


def test_target_gap_is_the_one_after_the_integer_digits():
    assert gap_target(6, (5, 1)) == 4  # 04872.3 → el hueco entre el quinto y el sexto dígito
    assert gap_target(7, (6, 1)) == 5
    assert gap_target(5, (5, 0)) is None  # sin fracción: ningún hueco es decimal


def test_target_is_unknown_when_the_digit_count_does_not_match_the_layout():
    assert gap_target(5, (5, 1)) == -1
    assert gap_target(8, (6, 1)) == -1


def test_decimal_goes_in_the_most_likely_gap_above_the_threshold():
    assert decimal_gap([0.1, 0.2, 0.9, 0.3]) == 3  # el punto va después del tercer dígito
    assert decimal_gap([0.1, 0.4, 0.2]) is None
    assert decimal_gap([0.1, 0.4, 0.2], threshold=0.3) == 2
    assert decimal_gap([]) is None
