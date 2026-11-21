"""Tests de scripts/train_gap_model.py: etiquetas de los huecos y datos externos (sin entrenar)."""

import importlib.util
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train_gap_model.py"
sys.path.insert(0, str(ROOT / "src"))
from meter_reader.reading import Digit  # noqa: E402


def load_module():
    spec = importlib.util.spec_from_file_location("train_gap_model", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _digits(xs):
    return [Digit(i % 10, 1.0, x, 10, x + 20, 50) for i, x in enumerate(xs)]


def test_dot_marks_the_gap_whose_digit_centers_surround_it():
    module = load_module()
    digits = _digits([10, 40, 70, 100])  # centros en 20, 50, 80, 110
    assert module.dot_target(digits, 85) == 3  # entre el tercer y el cuarto dígito: 3 enteros
    assert module.dot_target(digits, 30) == 1
    assert module.dot_target(digits, 120) == -1  # a la derecha de todo: no cae en un hueco
    assert module.dot_target(list(reversed(digits)), 85) == 3  # el orden de las cajas no importa


def test_external_dataset_keeps_images_with_one_dot_between_digits(tmp_path):
    module = load_module()
    split = tmp_path / "train"
    (split / "images").mkdir(parents=True)
    (split / "labels").mkdir()
    (tmp_path / "data.yaml").write_text("names:\n- .\n- '0'\n- '1'\n- '2'\n- '3'\n- kWh\nnc: 6\n")
    for name in ("a", "b", "c", "d"):
        Image.new("RGB", (200, 100), (90, 90, 90)).save(split / "images" / f"{name}.jpg")
    row = "1 0.1 0.5 0.1 0.4\n2 0.3 0.5 0.1 0.4\n3 0.5 0.5 0.1 0.4\n4 0.7 0.5 0.1 0.4\n"
    (split / "labels" / "a.txt").write_text(row + "0 0.6 0.65 0.02 0.04\n5 0.9 0.5 0.1 0.1\n")  # punto tras 3
    (split / "labels" / "b.txt").write_text(row)  # sin punto anotado: puede faltar la etiqueta
    (split / "labels" / "c.txt").write_text(row + "0 0.6 0.65 0.02 0.04\n0 0.2 0.65 0.02 0.04\n")  # dos puntos
    (split / "labels" / "d.txt").write_text("1 0.1 0.5 0.1 0.4\n0 0.6 0.65 0.02 0.04\n")  # un solo dígito
    photos = module.load_dot_dataset(tmp_path)
    assert [(p["file"], p["target"], len(p["patches"])) for p in photos] == [("a.jpg", 3, 3)]
    assert module.gap_labels(photos[0]) == [0, 0, 1]
