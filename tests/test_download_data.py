"""Tests de la búsqueda de Toloka adjuntado como input de Kaggle."""

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "download_data.py"


def load_module():
    spec = importlib.util.spec_from_file_location("download_data", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_finds_toloka_folder_at_any_depth_under_kaggle_input(tmp_path):
    module = load_module()
    toloka = tmp_path / "datasets" / "tapakah68" / "x" / "WaterMeters"
    (toloka / "images").mkdir(parents=True)
    (toloka / "data.csv").write_text("photo_name,value,location\n")
    # Otro input sin images/ al lado del CSV: no debe confundirse con Toloka.
    (tmp_path / "otro").mkdir()
    (tmp_path / "otro" / "data.csv").write_text("a\n")

    assert module.find_attached_toloka(tmp_path) == toloka


def test_returns_none_when_not_attached(tmp_path):
    module = load_module()
    assert module.find_attached_toloka(tmp_path) is None
    assert module.find_attached_toloka(tmp_path / "no_existe") is None


def test_energy_datasets_are_skipped_when_already_downloaded(tmp_path, monkeypatch):
    module = load_module()
    monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)
    for _, project, _ in module.ENERGY_DATASETS:
        (tmp_path / project).mkdir()
        (tmp_path / project / "data.yaml").write_text("names: []\n")
    module.download_energy(tmp_path)  # no pide la clave: ya están todos
