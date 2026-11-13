"""Tests de las partes de scripts/evaluate_seg.py que no requieren un modelo."""

import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_seg.py"


def load_module():
    spec = importlib.util.spec_from_file_location("evaluate_seg", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ceml_yaml_points_to_images_with_absolute_path(tmp_path):
    module = load_module()
    ceml = tmp_path / "ceml"
    (ceml / "images").mkdir(parents=True)

    yaml_path = module.write_ceml_yaml(ceml, tmp_path / "ceml.yaml")

    text = yaml_path.read_text()
    assert f"path: {ceml.resolve()}" in text
    assert "val: images" in text
    assert "display" in text


def test_save_results_writes_json_per_experiment(tmp_path):
    module = load_module()
    results = {"toloka_test": {"mask_map50": 0.99}, "ceml": {"mask_map50": 0.86}}

    module.save_results(results, "demo", tmp_path)

    saved = json.loads((tmp_path / "demo" / "eval.json").read_text())
    assert saved == results
