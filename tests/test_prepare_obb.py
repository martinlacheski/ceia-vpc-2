"""Tests de scripts/prepare_obb.py con datos sintéticos."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_obb.py"


def load_module():
    spec = importlib.util.spec_from_file_location("prepare_obb", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def corners(line: str, width: int, height: int) -> tuple[int, np.ndarray]:
    """Clase y esquinas en píxeles de una línea OBB de Ultralytics."""
    values = line.split()
    points = np.array(values[1:], dtype=float).reshape(4, 2) * [width, height]
    return int(values[0]), points


def same_points(a: np.ndarray, b: np.ndarray, tol: float = 1.0) -> bool:
    """Mismo conjunto de puntos, sin importar el orden."""
    return all(np.min(np.linalg.norm(b - p, axis=1)) < tol for p in a)


def test_axis_aligned_polygon_keeps_its_rectangle():
    width, height = 400, 200
    rect = np.array([[100, 50], [300, 50], [300, 90], [100, 90]], dtype=float)
    polygon = " ".join(f"{x / width:.6f} {y / height:.6f}" for x, y in rect)
    cls, points = corners(load_module().obb_line(f"0 {polygon}", width, height), width, height)
    assert cls == 0
    assert same_points(points, rect)


def test_rotated_polygon_becomes_its_minimum_rotated_rectangle_in_pixels():
    # Visor girado 30° en una imagen no cuadrada: el rectángulo se calcula en píxeles,
    # no en coordenadas normalizadas (que deformarían el ángulo).
    width, height = 1000, 600
    rect = cv2.boxPoints(((500.0, 300.0), (300.0, 60.0), 30.0))
    # Polígono con más vértices que el rectángulo (puntos intermedios en cada lado).
    dense = np.concatenate([np.linspace(rect[i], rect[(i + 1) % 4], 5, endpoint=False) for i in range(4)])
    polygon = " ".join(f"{x / width:.6f} {y / height:.6f}" for x, y in dense)
    _, points = corners(load_module().obb_line(f"0 {polygon}", width, height), width, height)
    assert same_points(points, rect, tol=1.5)


def test_corners_outside_the_image_are_clipped():
    width, height = 400, 200
    rect = cv2.boxPoints(((20.0, 100.0), (120.0, 30.0), 45.0))  # sale por el borde izquierdo
    polygon = " ".join(f"{x / width:.6f} {y / height:.6f}" for x, y in rect)
    values = np.array(load_module().obb_line(f"0 {polygon}", width, height).split()[1:], dtype=float)
    assert values.min() >= 0.0 and values.max() <= 1.0


def test_cli_builds_an_obb_dataset_next_to_the_same_images(tmp_path):
    src, out = tmp_path / "seg", tmp_path / "obb"
    for split in ("train", "val", "test"):
        (src / "images" / split).mkdir(parents=True)
        (src / "labels" / split).mkdir(parents=True)
        Image.new("RGB", (400, 200)).save(src / "images" / split / f"{split}.jpg")
        (src / "labels" / split / f"{split}.txt").write_text("0 0.25 0.25 0.75 0.25 0.75 0.45 0.25 0.45\n")
    (src / "data.yaml").write_text(f"path: {src}\ntrain: images/train\nval: images/val\n"
                                   "test: images/test\nnames: {0: 'display'}\n")

    subprocess.run([sys.executable, str(SCRIPT), "--src", str(src), "--out", str(out)], check=True)

    for split in ("train", "val", "test"):
        image = out / "images" / split / f"{split}.jpg"
        assert image.exists() and image.resolve() == (src / "images" / split / f"{split}.jpg").resolve()
        _, points = corners((out / "labels" / split / f"{split}.txt").read_text().strip(), 400, 200)
        assert same_points(points, np.array([[100, 50], [300, 50], [300, 90], [100, 90]], dtype=float))
    data = (out / "data.yaml").read_text()
    assert f"path: {out.resolve()}" in data
    assert "names: {0: 'display'}" in data
