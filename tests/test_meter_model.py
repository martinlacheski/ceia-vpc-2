"""Tests de src/meter_reader/meter_model.py: guardar y cargar el clasificador (necesita torch)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
from meter_reader.meter_model import build_model, load_classifier  # noqa: E402


def test_load_classifier_returns_a_trained_class_and_its_probability(tmp_path):
    classes = ["4_2", "5_0", "5_2"]
    net = build_model("mobilenet_v3_small", len(classes), pretrained=False)
    weights = tmp_path / "clasificador.pt"
    torch.save({"arch": "mobilenet_v3_small", "target": "formato", "classes": classes,
                "state_dict": net.state_dict()}, weights)
    label, probability = load_classifier(weights)(np.zeros((192, 384, 3), dtype=np.uint8))
    assert label in classes
    assert 0 < probability <= 1
