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


def test_load_gap_classifier_gives_one_probability_per_gap(tmp_path):
    from meter_reader.gaps import load_gap_classifier
    from meter_reader.reading import Digit

    weights = tmp_path / "huecos.pt"
    torch.save({"arch": "mobilenet_v3_small", "patch_size": [64, 96],
                "state_dict": build_model("mobilenet_v3_small", 2, pretrained=False).state_dict()}, weights)
    gaps = load_gap_classifier(weights)
    crop = np.zeros((60, 200, 3), dtype=np.uint8)
    digits = tuple(Digit(i, 0.9, x, 10, x + 20, 50) for i, x in enumerate([10, 50, 90, 130]))
    probs = gaps(crop, digits)
    assert len(probs) == 3 and all(0 <= p <= 1 for p in probs)
    assert list(gaps(crop, digits[:1])) == []  # un solo dígito: no hay huecos
