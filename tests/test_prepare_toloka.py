"""Tests de scripts/prepare_toloka.py usando un mini dataset sintético."""

import csv
import subprocess
import sys
from pathlib import Path

from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_toloka.py"

POLYGON = "{'type': 'polygon', 'data': [{'x': 0.1, 'y': 0.2}, {'x': 0.5, 'y': 0.2}, {'x': 0.5, 'y': 0.3}, {'x': 0.1, 'y': 0.3}]}"


def make_raw(root: Path, n: int) -> Path:
    """Crea un dataset Toloka falso con n imágenes y su data.csv."""
    raw = root / "toloka"
    (raw / "images").mkdir(parents=True)
    with open(raw / "data.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["photo_name", "value", "location"])
        for i in range(n):
            name = f"id_{i}_value_305_162.jpg"
            Image.new("RGB", (40, 30)).save(raw / "images" / name)
            # El valor del CSV trae error de punto flotante a propósito.
            writer.writerow([name, "305.16200000000003", POLYGON])
    return raw


def run(raw: Path, out: Path, splits: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--raw", str(raw), "--out", str(out), "--splits", str(splits)],
        capture_output=True, text=True, check=True,
    )


def test_converts_polygons_to_yolo_seg(tmp_path):
    raw = make_raw(tmp_path, 20)
    out, splits = tmp_path / "out", tmp_path / "splits.csv"
    run(raw, out, splits)

    labels = sorted(out.glob("labels/*/*.txt"))
    assert len(labels) == 20
    assert labels[0].read_text().strip() == "0 0.1 0.2 0.5 0.2 0.5 0.3 0.1 0.3"
    assert len(list(out.glob("images/*/*.jpg"))) == 20

    data_yaml = (out / "data.yaml").read_text()
    assert "names:" in data_yaml and "display" in data_yaml
    assert f"path: {out.resolve()}" in data_yaml


def test_splits_are_70_15_15_and_reproducible(tmp_path):
    raw = make_raw(tmp_path, 20)
    run(raw, tmp_path / "a", tmp_path / "a.csv")
    run(raw, tmp_path / "b", tmp_path / "b.csv")

    rows = list(csv.DictReader(open(tmp_path / "a.csv", encoding="utf-8")))
    counts = {s: sum(r["split"] == s for r in rows) for s in ("train", "val", "test")}
    assert counts == {"train": 14, "val": 3, "test": 3}
    assert (tmp_path / "a.csv").read_text() == (tmp_path / "b.csv").read_text()


def test_reading_comes_from_filename_not_float_column(tmp_path):
    raw = make_raw(tmp_path, 5)
    run(raw, tmp_path / "out", tmp_path / "splits.csv")

    rows = list(csv.DictReader(open(tmp_path / "splits.csv", encoding="utf-8")))
    assert {r["reading"] for r in rows} == {"305.162"}
