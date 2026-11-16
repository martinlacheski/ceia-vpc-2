"""Tests de scripts/pseudo_label_displays.py: polígono del visor en formato YOLO-seg."""

import importlib.util
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_yolo_line_normalizes_and_round_trips_with_the_training_reader(tmp_path):
    polygon = np.array([[100, 50], [300, 50], [300, 150], [100, 150]], dtype=float)
    line = load("pseudo_label_displays").yolo_line(polygon, width=400, height=200)
    assert line == "0 0.250000 0.250000 0.750000 0.250000 0.750000 0.750000 0.250000 0.750000"
    label = tmp_path / "a.txt"
    label.write_text(line + "\n")
    read = load("train_meter_model").read_polygon(label, 400, 200)
    assert np.allclose(read, polygon)


def test_yolo_line_clips_points_outside_the_image():
    polygon = np.array([[-10, 0], [410, 0], [410, 250]], dtype=float)
    line = load("pseudo_label_displays").yolo_line(polygon, width=400, height=200)
    assert line == "0 0.000000 0.000000 1.000000 0.000000 1.000000 1.000000"
