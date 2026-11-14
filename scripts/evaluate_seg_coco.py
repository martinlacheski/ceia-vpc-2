"""Evalúa la segmentación del visor con métricas COCO, igual para YOLO y Mask R-CNN.

Ultralytics y torchvision calculan el mAP de formas parecidas pero no idénticas; para comparar
arquitecturas se evalúan las dos con pycocotools sobre las mismas imágenes y etiquetas:
  - toloka_test: split de prueba de Toloka;
  - ceml: las fotos de campo de la cooperativa (dominio real, sin usar en entrenamiento).
Reporta AP50, AP50-95 y AR@100 de máscara y de caja, y la latencia media en CPU.

Guarda el resultado en experiments/<nombre>/eval_coco.json.

Uso:
  python scripts/evaluate_seg_coco.py --arch yolo --weights models/toloka_yfzw4_yolov8n_seg.pt
  python scripts/evaluate_seg_coco.py --arch maskrcnn --weights models/toloka_yfzw4_maskrcnn.pth
"""

import argparse
import json
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

SPLITS = {
    "toloka_test": (Path("data/processed/toloka_seg/images/test"), Path("data/processed/toloka_seg/labels/test")),
    "ceml": (Path("data/ceml/images"), Path("data/ceml/labels")),
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
LATENCY_IMAGES = 20
# Umbral bajo para recorrer toda la curva precisión-recall, como hace la validación de Ultralytics
SCORE_THRESHOLD = 0.001
EXIF_ORIENTATION = 0x0112

# Recibe una imagen BGR y devuelve (máscaras booleanas del tamaño de la imagen, puntajes)
Predictor = Callable[[np.ndarray], tuple[list[np.ndarray], list[float]]]


def read_polygons(label_path: Path, width: int, height: int) -> list[np.ndarray]:
    """Polígonos de una etiqueta YOLO de segmentación, en píxeles."""
    if not label_path.exists():
        return []
    polygons = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) < 7:  # clase + al menos 3 puntos
            continue
        polygons.append(np.array(values[1:], dtype=float).reshape(-1, 2) * [width, height])
    return polygons


def polygons_to_mask(polygons: list[np.ndarray], height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in polygons:
        cv2.fillPoly(mask, [np.round(polygon).astype(np.int32)], 1)
    return mask.astype(bool)


def _polygon_area(xy: np.ndarray) -> float:
    x, y = xy[:, 0], xy[:, 1]
    return float(abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2)


def list_images(images_dir: Path) -> list[Path]:
    return sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def coco_ground_truth(images_dir: Path, labels_dir: Path) -> dict:
    """Arma el dataset COCO (imágenes y anotaciones) a partir de imágenes y etiquetas YOLO."""
    from PIL import Image

    images, annotations = [], []
    for image_id, path in enumerate(list_images(images_dir), start=1):
        with Image.open(path) as img:
            width, height = img.size
            # cv2 y Ultralytics leen la foto ya rotada según EXIF: las etiquetas son relativas a esa vista
            if img.getexif().get(EXIF_ORIENTATION, 1) in (5, 6, 7, 8):
                width, height = height, width
        images.append({"id": image_id, "file_name": path.name, "width": width, "height": height})
        for xy in read_polygons(labels_dir / f"{path.stem}.txt", width, height):
            x0, y0 = xy.min(axis=0)
            x1, y1 = xy.max(axis=0)
            annotations.append({
                "id": len(annotations) + 1, "image_id": image_id, "category_id": 1, "iscrowd": 0,
                "segmentation": [xy.flatten().tolist()],
                "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
                "area": _polygon_area(xy),
            })
    return {"images": images, "annotations": annotations, "categories": [{"id": 1, "name": "display"}]}


def encode_masks(image_id: int, masks: list[np.ndarray], scores: list[float]) -> list[dict]:
    """Predicciones en formato de resultados COCO (máscaras en RLE)."""
    from pycocotools import mask as mask_utils

    results = []
    for mask, score in zip(masks, scores):
        rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
        bbox = mask_utils.toBbox(rle).tolist()
        rle["counts"] = rle["counts"].decode("ascii")
        results.append({"image_id": image_id, "category_id": 1, "segmentation": rle,
                        "bbox": bbox, "score": float(score)})
    return results


def coco_metrics(gt: dict, results: list[dict]) -> dict:
    """AP50, AP50-95 y AR@100 de máscara y de caja con pycocotools."""
    import contextlib
    import io

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    metrics = {}
    for iou_type, prefix in (("segm", "mask"), ("bbox", "box")):
        if not results:
            metrics.update({f"{prefix}_ap50": 0.0, f"{prefix}_ap50_95": 0.0, f"{prefix}_ar100": 0.0})
            continue
        with contextlib.redirect_stdout(io.StringIO()):  # pycocotools imprime tablas largas
            coco_gt = COCO()
            coco_gt.dataset = gt
            coco_gt.createIndex()
            evaluation = COCOeval(coco_gt, coco_gt.loadRes(results), iou_type)
            evaluation.evaluate()
            evaluation.accumulate()
            evaluation.summarize()
        stats = [max(0.0, float(s)) for s in evaluation.stats]  # -1 = sin datos
        metrics.update({f"{prefix}_ap50": round(stats[1], 4), f"{prefix}_ap50_95": round(stats[0], 4),
                        f"{prefix}_ar100": round(stats[8], 4)})
    return metrics


def evaluate_split(predict: Predictor, images_dir: Path, labels_dir: Path) -> dict:
    gt = coco_ground_truth(images_dir, labels_dir)
    results = []
    for image in gt["images"]:
        masks, scores = predict(cv2.imread(str(images_dir / image["file_name"])))
        results += encode_masks(image["id"], masks, scores)
    return {**coco_metrics(gt, results), "images": len(gt["images"])}


def cpu_latency_ms(predict: Predictor, images: list[Path]) -> float:
    """Latencia media por imagen en CPU (inferencia completa), tras 3 imágenes de calentamiento."""
    loaded = [cv2.imread(str(p)) for p in images]
    for image in loaded[:3]:
        predict(image)
    start = time.perf_counter()
    for image in loaded:
        predict(image)
    return round((time.perf_counter() - start) / len(loaded) * 1000, 1)


def yolo_predictor(weights: Path, imgsz: int, device) -> Predictor:
    from ultralytics import YOLO

    model = YOLO(str(weights))

    def predict(image: np.ndarray) -> tuple[list[np.ndarray], list[float]]:
        result = model.predict(image, imgsz=imgsz, conf=SCORE_THRESHOLD, device=device, verbose=False)[0]
        if result.masks is None:
            return [], []
        height, width = image.shape[:2]
        masks = [polygons_to_mask([xy], height, width) for xy in result.masks.xy]
        return masks, [float(s) for s in result.boxes.conf]

    return predict


def maskrcnn_predictor(weights: Path, device) -> Predictor:
    import torch

    from train_maskrcnn import build_model

    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    model.roi_heads.score_thresh = SCORE_THRESHOLD
    model.to(device).eval()

    @torch.no_grad()
    def predict(image: np.ndarray) -> tuple[list[np.ndarray], list[float]]:
        rgb = torch.from_numpy(np.ascontiguousarray(image[:, :, ::-1])).permute(2, 0, 1).float() / 255
        output = model([rgb.to(device)])[0]
        masks = (output["masks"][:, 0] > 0.5).cpu().numpy()
        return list(masks), output["scores"].cpu().tolist()

    return predict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arch", choices=["yolo", "maskrcnn"], required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640, help="solo YOLO; Mask R-CNN usa el tamaño con el que se entrenó")
    parser.add_argument("--name", help="por defecto, el nombre del archivo de pesos")
    args = parser.parse_args()

    if args.arch == "yolo":
        # ultralytics antes que torch: configura los hilos de OpenMP (ver evaluate_seg.py)
        import ultralytics  # noqa: F401
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def make(target_device) -> Predictor:
        if args.arch == "yolo":
            return yolo_predictor(args.weights, args.imgsz, 0 if target_device == "cuda" else "cpu")
        return maskrcnn_predictor(args.weights, target_device)

    latency_images = list_images(SPLITS["ceml"][0])[:LATENCY_IMAGES]
    latency = cpu_latency_ms(make("cpu"), latency_images)  # modelo recién cargado, en CPU
    predict = make(device)
    results = {split: evaluate_split(predict, *dirs) for split, dirs in SPLITS.items()}
    results.update({"cpu_latency_ms": latency, "arch": args.arch, "imgsz": args.imgsz if args.arch == "yolo" else None})

    name = args.name or args.weights.stem
    out = Path("experiments") / name / "eval_coco.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Guardado en {out}")


if __name__ == "__main__":
    main()
