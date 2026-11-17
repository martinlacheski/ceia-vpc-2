"""Tests de scripts/export_digit_prelabels.py (sin modelos: el detector es un doble)."""

import importlib.util
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "export_digit_prelabels.py"
sys.path.insert(0, str(ROOT / "src"))
from meter_reader.reading import Digit  # noqa: E402


def load_module():
    spec = importlib.util.spec_from_file_location("export_digit_prelabels", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_dataset(root: Path) -> None:
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    for name in ("a", "b", "c", "d"):
        Image.new("RGB", (400, 300), (90, 90, 90)).save(root / "images" / f"{name}.jpg")
    for name in ("a", "b", "c"):  # d no tiene visor etiquetado
        (root / "labels" / f"{name}.txt").write_text("0 0.25 0.4 0.75 0.4 0.75 0.6 0.25 0.6\n")
    (root / "readings.csv").write_text(
        "file,muestra,reading,legible\n"
        "a.jpg,,,\n"
        "b.jpg,si,012345.6,si\n"  # muestra de prueba: no se exporta
        "c.jpg,no,,\n"
        "d.jpg,,,\n",
        encoding="utf-8",
    )


def test_yolo_lines_normalize_digit_boxes_to_the_crop():
    module = load_module()
    lines = module.yolo_lines([Digit(7, 0.9, 10, 20, 30, 60)], width=100, height=80)
    assert lines == ["7 0.200000 0.500000 0.200000 0.500000"]


def test_yolo_lines_clip_boxes_that_leave_the_crop():
    module = load_module()
    lines = module.yolo_lines([Digit(3, 0.9, -10, 0, 20, 80)], width=100, height=80)
    assert lines == ["3 0.100000 0.500000 0.200000 1.000000"]


def test_export_skips_the_test_sample_and_photos_without_display(tmp_path):
    module = load_module()
    make_dataset(tmp_path / "src")
    out = tmp_path / "out"
    calls = []

    def detect(crop):
        calls.append(crop.shape)
        return [Digit(1, 0.9, 5, 5, 25, 35), Digit(2, 0.8, 30, 5, 50, 35)]

    photos, boxes = module.export(tmp_path / "src", out, detect)

    assert (photos, boxes) == (2, 4)
    assert sorted(p.name for p in (out / "images").iterdir()) == ["a.jpg", "c.jpg"]
    assert sorted(p.name for p in (out / "labels").iterdir()) == ["a.txt", "c.txt"]
    first = (out / "labels" / "a.txt").read_text().splitlines()
    assert [line.split()[0] for line in first] == ["1", "2"]
    # el recorte es el visor enderezado (más ancho que alto), no la foto entera
    height, width = calls[0][:2]
    assert width > height and width < 400
    assert "names" in (out / "data.yaml").read_text()


def test_export_keeps_digits_in_reading_order_without_duplicates(tmp_path):
    module = load_module()
    make_dataset(tmp_path / "src")
    out = tmp_path / "out"
    # dos cajas casi en la misma posición: queda la de mayor confianza, como en la lectura
    detect = lambda crop: [Digit(4, 0.9, 40, 5, 60, 35), Digit(9, 0.5, 41, 5, 61, 35),  # noqa: E731
                           Digit(0, 0.9, 5, 5, 25, 35)]
    module.export(tmp_path / "src", out, detect)
    values = [line.split()[0] for line in (out / "labels" / "a.txt").read_text().splitlines()]
    assert values == ["0", "4"]


def test_export_refuses_a_non_empty_output(tmp_path):
    module = load_module()
    make_dataset(tmp_path / "src")
    out = tmp_path / "out"
    (out / "images").mkdir(parents=True)
    (out / "images" / "old.jpg").write_bytes(b"x")
    with pytest.raises(SystemExit):
        module.export(tmp_path / "src", out, lambda crop: [])
