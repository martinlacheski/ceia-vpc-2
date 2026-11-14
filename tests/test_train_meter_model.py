"""Tests de scripts/train_meter_model.py: datos, folds y métricas (sin entrenar)."""

import importlib.util
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_meter_model.py"


def load_module():
    spec = importlib.util.spec_from_file_location("train_meter_model", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_examples_keeps_rows_with_known_model_and_display_label(tmp_path):
    module = load_module()
    images, labels = tmp_path / "images", tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    for name in ("a", "b", "c", "d"):
        Image.new("RGB", (400, 300), (90, 90, 90)).save(images / f"{name}.jpg")
    for name in ("a", "b", "c"):
        (labels / f"{name}.txt").write_text("0 0.3 0.4 0.7 0.4 0.7 0.5 0.3 0.5\n")
    sheet = tmp_path / "readings.csv"
    sheet.write_text(
        "file,reading,legible,notes,modelo\n"
        "a.jpg,00053.25,si,,generico_5_2\n"
        "b.jpg,1203,si,,iso4064_4_0\n"
        "c.jpg,1203,si,,desconocido\n"  # modelo fuera de la tabla
        "d.jpg,1203,si,,iso4064_4_0\n"  # sin etiqueta del visor
    )
    examples = module.load_examples(sheet, images, labels)
    assert [(e["file"], e["model"]) for e in examples] == [("a.jpg", "generico_5_2"), ("b.jpg", "iso4064_4_0")]
    assert examples[0]["image"].shape == (192, 384, 3)


def test_target_formato_groups_models_with_the_same_layout():
    module = load_module()
    assert module.target_of("abb_4_1", "formato") == module.target_of("recuadro_4_1", "formato") == "4_1"
    assert module.target_of("abb_4_1", "modelo") == "abb_4_1"


def test_classes_below_min_samples_are_not_trained():
    module = load_module()
    labels = ["a"] * 6 + ["b"] * 5 + ["c"] * 2
    assert module.trainable_classes(labels, min_samples=5) == ["a", "b"]


def test_stratified_folds_spread_every_class_evenly():
    module = load_module()
    labels = ["a"] * 10 + ["b"] * 5 + ["c"] * 3
    folds = module.stratified_folds(labels, k=5, seed=0)
    for label in "ab":
        per_fold = Counter(f for f, l in zip(folds, labels) if l == label)
        assert max(per_fold.values()) - min(per_fold.values()) <= 1 and len(per_fold) == 5
    assert len({f for f, l in zip(folds, labels) if l == "c"}) == 3  # 3 fotos, 3 folds distintos
    assert folds == module.stratified_folds(labels, k=5, seed=0)  # reproducible


def test_classification_metrics_accuracy_macro_f1_and_confusion():
    module = load_module()
    metrics = module.classification_metrics(["a", "a", "b", "b"], ["a", "b", "b", "b"], ["a", "b"])
    assert metrics["accuracy"] == 0.75
    assert metrics["confusion"] == [[1, 1], [0, 2]]
    # F1(a) = 2·1·0.5/1.5 = 0.667; F1(b) = 2·0.667·1/1.667 = 0.8
    assert metrics["macro_f1"] == pytest.approx(0.7333, abs=1e-3)


def test_augment_keeps_shape_and_type():
    module = load_module()
    image = np.random.default_rng(0).integers(0, 255, (192, 384, 3), dtype=np.uint8)
    out = module.augment(image, np.random.default_rng(1))
    assert out.shape == image.shape and out.dtype == np.uint8


def test_build_model_outputs_one_score_per_class():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    module = load_module()
    net = module.build_model("mobilenet_v3_small", num_classes=6, pretrained=False).eval()
    x = module.to_tensor([np.zeros((192, 384, 3), dtype=np.uint8)] * 2)
    with torch.no_grad():
        assert net(x).shape == (2, 6)
