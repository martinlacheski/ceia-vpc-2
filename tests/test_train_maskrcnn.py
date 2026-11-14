"""Tests de scripts/train_maskrcnn.py: carga de muestras y armado del modelo."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_maskrcnn.py"


def load_module():
    spec = importlib.util.spec_from_file_location("train_maskrcnn", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_sample(root: Path, size=(1280, 960)) -> tuple[Path, Path]:
    image, label = root / "a.jpg", root / "a.txt"
    Image.new("RGB", size, (200, 200, 200)).save(image)
    label.write_text("0 0.25 0.25 0.75 0.25 0.75 0.5 0.25 0.5\n")
    return image, label


def test_load_sample_resizes_longest_side_and_builds_targets(tmp_path):
    module = load_module()
    image, label = make_sample(tmp_path)
    rgb, boxes, masks = module.load_sample(image, label, max_side=640)
    assert rgb.shape == (480, 640, 3) and rgb.dtype == np.uint8
    np.testing.assert_allclose(boxes, [[160, 120, 480, 240]], atol=1)
    assert masks.shape == (1, 480, 640) and masks.dtype == np.uint8
    assert masks[0, 180, 320] == 1 and masks[0, 400, 600] == 0


def test_load_sample_keeps_small_images_and_skips_missing_labels(tmp_path):
    module = load_module()
    image, _ = make_sample(tmp_path, size=(320, 200))
    rgb, boxes, masks = module.load_sample(image, tmp_path / "no-existe.txt", max_side=640)
    assert rgb.shape == (200, 320, 3)
    assert boxes.shape == (0, 4) and masks.shape == (0, 200, 320)


def test_hflip_mirrors_image_boxes_and_masks():
    module = load_module()
    rgb = np.zeros((10, 20, 3), dtype=np.uint8)
    rgb[:, :5] = 255
    masks = np.zeros((1, 10, 20), dtype=np.uint8)
    masks[0, :, :5] = 1
    boxes = np.array([[0, 0, 5, 10]], dtype=np.float32)
    f_rgb, f_boxes, f_masks = module.hflip(rgb, boxes, masks)
    assert f_rgb[0, 19, 0] == 255 and f_rgb[0, 0, 0] == 0
    np.testing.assert_allclose(f_boxes, [[15, 0, 20, 10]])
    assert f_masks[0, 0, 19] == 1 and f_masks[0, 0, 0] == 0


def test_build_model_has_two_classes_display_and_background():
    pytest.importorskip("torchvision")
    module = load_module()
    model = module.build_model(pretrained=False)
    assert model.roi_heads.box_predictor.cls_score.out_features == 2
    assert model.roi_heads.mask_predictor.mask_fcn_logits.out_channels == 2
    assert model.transform.min_size == (640,) and model.transform.max_size == 640
