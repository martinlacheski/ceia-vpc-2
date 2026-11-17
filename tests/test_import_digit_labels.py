"""Tests de scripts/import_digit_labels.py: export YOLO de Roboflow → data/ceml_energia/digits."""

import importlib.util
from pathlib import Path

from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "import_digit_labels.py"


def load_module():
    spec = importlib.util.spec_from_file_location("import_digit_labels", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_export(root: Path) -> None:
    """Export de Roboflow: nombres cambiados, clases por nombre y una imagen marcada Null (sin cajas)."""
    for split, names in {"train": ["energia_a_jpg.rf.abc123.jpg", "energia_b_jpg.rf.def456.jpg"],
                         "valid": ["energia_c_jpg.rf.987fed.jpg"]}.items():
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        for name in names:
            Image.new("RGB", (120, 40), (90, 90, 90)).save(root / split / "images" / name)
    # Roboflow ordena las clases por nombre; acá se cambia el orden para probar que se mapea por nombre
    (root / "data.yaml").write_text("nc: 10\nnames: ['1', '0', '2', '3', '4', '5', '6', '7', '8', '9']\n")
    (root / "train" / "labels" / "energia_a_jpg.rf.abc123.txt").write_text("0 0.2 0.5 0.1 0.8\n1 0.4 0.5 0.1 0.8\n")
    (root / "train" / "labels" / "energia_b_jpg.rf.def456.txt").write_text("")  # Null: ilegible
    (root / "valid" / "labels" / "energia_c_jpg.rf.987fed.txt").write_text("9 0.5 0.5 0.1 0.8\n")


def test_original_name_drops_the_roboflow_suffix():
    module = load_module()
    assert module.original_name("energia_2018-07_r12_m0229_jpg.rf.0a1b2c.jpg") == "energia_2018-07_r12_m0229.jpg"
    assert module.original_name("energia_x.jpg") == "energia_x.jpg"


def test_import_restores_names_maps_classes_by_name_and_skips_null_images(tmp_path):
    module = load_module()
    make_export(tmp_path / "export")
    out = tmp_path / "digits"

    imported, skipped = module.import_export(tmp_path / "export", out)

    assert (imported, skipped) == (2, 1)
    assert sorted(p.name for p in (out / "images").iterdir()) == ["energia_a.jpg", "energia_c.jpg"]
    # la clase 0 del export se llama '1' y la 1 se llama '0'
    assert (out / "labels" / "energia_a.txt").read_text() == "1 0.2 0.5 0.1 0.8\n0 0.4 0.5 0.1 0.8\n"
    assert (out / "labels" / "energia_c.txt").read_text() == "9 0.5 0.5 0.1 0.8\n"
    assert not (out / "labels" / "energia_b.txt").exists()


def test_import_replaces_a_previous_import(tmp_path):
    module = load_module()
    make_export(tmp_path / "export")
    out = tmp_path / "digits"
    (out / "images").mkdir(parents=True)
    (out / "images" / "vieja.jpg").write_bytes(b"x")
    module.import_export(tmp_path / "export", out)
    assert not (out / "images" / "vieja.jpg").exists()


def test_polygons_drawn_in_roboflow_become_boxes(tmp_path):
    module = load_module()
    make_export(tmp_path / "export")
    label = tmp_path / "export" / "valid" / "labels" / "energia_c_jpg.rf.987fed.txt"
    label.write_text("9 0.4 0.2 0.6 0.2 0.6 0.8 0.4 0.8\n")  # dígito dibujado como polígono
    module.import_export(tmp_path / "export", tmp_path / "digits")
    assert (tmp_path / "digits" / "labels" / "energia_c.txt").read_text() == "9 0.500000 0.500000 0.200000 0.600000\n"
