"""Tests de scripts/evaluate_seg_coco.py: verdad de campo COCO desde etiquetas YOLO y métricas."""

import importlib.util
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_seg_coco.py"


def load_module():
    spec = importlib.util.spec_from_file_location("evaluate_seg_coco", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_split(root: Path) -> tuple[Path, Path]:
    images, labels = root / "images", root / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    Image.new("RGB", (200, 100)).save(images / "a.jpg")
    Image.new("RGB", (100, 100)).save(images / "b.jpg")  # sin etiqueta: imagen sin visor
    # rectángulo de x 0.1–0.5, y 0.2–0.6 en una imagen de 200x100
    (labels / "a.txt").write_text("0 0.1 0.2 0.5 0.2 0.5 0.6 0.1 0.6\n")
    return images, labels


def test_read_polygons_scales_normalized_coordinates(tmp_path):
    module = load_module()
    label = tmp_path / "x.txt"
    label.write_text("0 0.1 0.2 0.5 0.2 0.5 0.6\n0 0.5\n")  # la segunda línea no es un polígono
    polygons = module.read_polygons(label, 200, 100)
    assert len(polygons) == 1
    np.testing.assert_allclose(polygons[0], [[20, 20], [100, 20], [100, 60]])
    assert module.read_polygons(tmp_path / "no-existe.txt", 10, 10) == []


def test_coco_ground_truth_from_yolo_labels(tmp_path):
    module = load_module()
    images, labels = make_split(tmp_path)
    gt = module.coco_ground_truth(images, labels)
    assert [(i["file_name"], i["width"], i["height"]) for i in gt["images"]] == [("a.jpg", 200, 100), ("b.jpg", 100, 100)]
    [ann] = gt["annotations"]
    assert ann["image_id"] == gt["images"][0]["id"] and ann["category_id"] == 1 and ann["iscrowd"] == 0
    np.testing.assert_allclose(ann["bbox"], [20, 20, 80, 40])
    assert ann["area"] == 80 * 40
    assert gt["categories"] == [{"id": 1, "name": "display"}]


def rect_mask(height: int, width: int, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def test_perfect_prediction_scores_ap_one_and_missing_scores_zero(tmp_path):
    module = load_module()
    images, labels = make_split(tmp_path)
    gt = module.coco_ground_truth(images, labels)
    image_id = gt["images"][0]["id"]
    perfect = module.encode_masks(image_id, [rect_mask(100, 200, 20, 20, 100, 60)], [0.9])
    metrics = module.coco_metrics(gt, perfect)
    assert metrics["mask_ap50"] == 1.0 and metrics["mask_ap50_95"] > 0.9
    assert metrics["box_ap50"] == 1.0
    assert module.coco_metrics(gt, [])["mask_ap50"] == 0.0
    # una máscara en el lugar equivocado no cuenta
    wrong = module.encode_masks(image_id, [rect_mask(100, 200, 150, 70, 190, 95)], [0.9])
    assert module.coco_metrics(gt, wrong)["mask_ap50"] == 0.0


def test_polygons_to_mask_rasterizes_at_image_size():
    module = load_module()
    mask = module.polygons_to_mask([np.array([[20, 20], [100, 20], [100, 60], [20, 60]], dtype=float)], 100, 200)
    assert mask.shape == (100, 200) and mask.dtype == bool
    assert mask[40, 60] and not mask[80, 150]
    assert abs(int(mask.sum()) - cv2.contourArea(np.array([[20, 20], [100, 20], [100, 60], [20, 60]], np.float32))) < 200


def test_ground_truth_uses_exif_oriented_size_like_cv2(tmp_path):
    # cv2 (y Ultralytics) aplican la rotación EXIF al leer; la verdad de campo tiene que usar ese tamaño
    module = load_module()
    images, labels = tmp_path / "images", tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    exif = Image.Exif()
    exif[0x0112] = 6  # rotar 90°: la foto cruda de 200x100 se ve de 100x200
    Image.new("RGB", (200, 100)).save(images / "r.jpg", exif=exif)
    (labels / "r.txt").write_text("0 0.1 0.2 0.5 0.2 0.5 0.6 0.1 0.6\n")
    gt = module.coco_ground_truth(images, labels)
    assert cv2.imread(str(images / "r.jpg")).shape[:2] == (200, 100)
    assert (gt["images"][0]["width"], gt["images"][0]["height"]) == (100, 200)
    np.testing.assert_allclose(gt["annotations"][0]["bbox"], [10, 40, 40, 80])
