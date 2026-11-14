"""Tests de scripts/propose_readings.py con un lector falso (sin modelos)."""

import csv
import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "propose_readings.py"


def load_module():
    spec = importlib.util.spec_from_file_location("propose_readings", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "reading", "legible", "notes"])
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_fills_proposal_and_confidence_without_touching_manual_columns(tmp_path):
    module = load_module()
    images = tmp_path / "images"
    images.mkdir()
    for name in ("a.jpg", "b.jpg"):
        (images / name).write_bytes(b"x")
    sheet = tmp_path / "readings.csv"
    write_csv(sheet, [
        {"file": "a.jpg", "reading": "", "legible": "", "notes": ""},
        {"file": "b.jpg", "reading": "00042", "legible": "si", "notes": "verificada"},
        {"file": "c.jpg", "reading": "", "legible": "", "notes": ""},  # sin foto
    ])
    proposals = {"a.jpg": ("01234", 0.8765), "b.jpg": None}  # b: no se encontró el visor
    seen = []

    def read(path: Path):
        seen.append(path.name)
        return proposals[path.name]

    module.propose(sheet, images, read)
    rows = read_csv(sheet)
    assert list(rows[0]) == ["file", "reading", "legible", "notes", "propuesta", "confianza"]
    assert rows[0] == {"file": "a.jpg", "reading": "", "legible": "", "notes": "",
                       "propuesta": "01234", "confianza": "0.88"}
    assert rows[1] == {"file": "b.jpg", "reading": "00042", "legible": "si", "notes": "verificada",
                       "propuesta": "", "confianza": ""}
    assert rows[2]["propuesta"] == "" and rows[2]["confianza"] == ""
    assert seen == ["a.jpg", "b.jpg"]
    assert b"\r\n" not in sheet.read_bytes()  # conserva los finales de línea LF del repo


def test_rerun_replaces_proposals_and_keeps_manual_columns(tmp_path):
    module = load_module()
    images = tmp_path / "images"
    images.mkdir()
    (images / "a.jpg").write_bytes(b"x")
    sheet = tmp_path / "readings.csv"
    write_csv(sheet, [{"file": "a.jpg", "reading": "", "legible": "", "notes": ""}])
    module.propose(sheet, images, lambda p: ("111", 0.5))
    rows = read_csv(sheet)
    rows[0]["reading"] = "112"
    with sheet.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    module.propose(sheet, images, lambda p: ("113", 0.9))
    assert read_csv(sheet) == [{"file": "a.jpg", "reading": "112", "legible": "", "notes": "",
                                "propuesta": "113", "confianza": "0.90"}]
