"""Entrena Mask R-CNN (torchvision, ResNet-50 FPN preentrenado en COCO) para segmentar el visor.

Es la arquitectura de dos etapas contra la que se compara YOLOv8n-seg (una etapa). Usa el
mismo dataset que el modelo B (Toloka + yfzw4, formato YOLO) y el mismo tamaño de entrada:
las imágenes se achican a 640 px de lado mayor y la única aumentación es el espejado
horizontal (YOLO además usa mosaico y variaciones de color; queda anotado en el informe).

Al terminar:
  - models/<nombre>.pth: pesos del mejor epoch según mask AP50-95 en validación
    (~170 MB: no se versiona; queda en el zip de resultados de Kaggle);
  - experiments/<nombre>/results.csv y args.yaml: métricas por epoch e hiperparámetros.

Uso:
  python scripts/train_maskrcnn.py --data data/processed/seg_toloka_yfzw4 --name toloka_yfzw4_maskrcnn
"""

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_seg_coco import coco_metrics, coco_ground_truth, encode_masks, list_images, read_polygons  # noqa: E402

MAX_SIDE = 640  # el imgsz del modelo B
WARMUP_ITERS = 300


def load_sample(image_path: Path, label_path: Path, max_side: int = MAX_SIDE):
    """Imagen RGB achicada a max_side y sus cajas (x1, y1, x2, y2) y máscaras."""
    image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale < 1:
        width, height = round(width * scale), round(height * scale)
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    boxes, masks = [], []
    for xy in read_polygons(label_path, width, height):
        x0, y0 = np.clip(xy.min(axis=0), 0, [width, height])
        x1, y1 = np.clip(xy.max(axis=0), 0, [width, height])
        if x1 - x0 < 1 or y1 - y0 < 1:
            continue
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [np.round(xy).astype(np.int32)], 1)
        boxes.append([x0, y0, x1, y1])
        masks.append(mask)
    return (image, np.array(boxes, dtype=np.float32).reshape(-1, 4),
            np.array(masks, dtype=np.uint8).reshape(-1, height, width))


def hflip(image: np.ndarray, boxes: np.ndarray, masks: np.ndarray):
    width = image.shape[1]
    flipped = boxes.copy()
    flipped[:, [0, 2]] = width - boxes[:, [2, 0]]
    return image[:, ::-1].copy(), flipped, masks[:, :, ::-1].copy()


class YoloSegDataset:
    """Dataset de torch (estilo mapa) sobre imágenes y etiquetas YOLO de segmentación."""

    def __init__(self, images_dir: Path, labels_dir: Path, train: bool, seed: int = 0):
        self.images = list_images(images_dir)
        self.labels_dir = labels_dir
        self.train = train
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        import torch

        path = self.images[index]
        image, boxes, masks = load_sample(path, self.labels_dir / f"{path.stem}.txt")
        if self.train and self.rng.random() < 0.5:
            image, boxes, masks = hflip(image, boxes, masks)
        target = {
            "boxes": torch.from_numpy(boxes),
            "labels": torch.ones(len(boxes), dtype=torch.int64),  # 1 = visor (0 es fondo)
            "masks": torch.from_numpy(masks),
        }
        return torch.from_numpy(image).permute(2, 0, 1).float() / 255, target


def collate(batch):
    return tuple(zip(*batch))


def build_model(num_classes: int = 2, pretrained: bool = True, size: int = MAX_SIDE):
    """Mask R-CNN R50-FPN con las cabezas reemplazadas para fondo + visor."""
    from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights, maskrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = maskrcnn_resnet50_fpn(weights=weights, weights_backbone=None, min_size=size, max_size=size)
    box_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(box_features, num_classes)
    mask_channels = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_channels, 256, num_classes)
    return model


def validate(model, images_dir: Path, labels_dir: Path, device) -> dict:
    """Métricas COCO en validación, con las imágenes en tamaño original (como en la evaluación final)."""
    import torch

    gt = coco_ground_truth(images_dir, labels_dir)
    model.eval()
    model.roi_heads.score_thresh = 0.001
    results = []
    with torch.no_grad():
        for image in gt["images"]:
            bgr = cv2.imread(str(images_dir / image["file_name"]))
            rgb = torch.from_numpy(np.ascontiguousarray(bgr[:, :, ::-1])).permute(2, 0, 1).float() / 255
            output = model([rgb.to(device)])[0]
            masks = (output["masks"][:, 0] > 0.5).cpu().numpy()
            results += encode_masks(image["id"], list(masks), output["scores"].cpu().tolist())
    model.roi_heads.score_thresh = 0.05
    return coco_metrics(gt, results)


def train(data: Path, name: str, epochs: int, batch: int, lr: float, workers: int, seed: int,
          pretrained: bool = True, models_dir: Path = Path("models"),
          experiments_dir: Path = Path("experiments")) -> Path:
    import torch

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set = YoloSegDataset(data / "images/train", data / "labels/train", train=True, seed=seed)
    loader = torch.utils.data.DataLoader(train_set, batch_size=batch, shuffle=True, num_workers=workers,
                                         collate_fn=collate, drop_last=True)
    model = build_model(pretrained=pretrained).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4)
    total = epochs * len(loader)

    def schedule(step: int) -> float:  # calentamiento lineal y luego coseno
        if step < WARMUP_ITERS:
            return (step + 1) / WARMUP_ITERS
        return 0.5 * (1 + math.cos(math.pi * (step - WARMUP_ITERS) / max(1, total - WARMUP_ITERS)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    exp_dir = experiments_dir / name
    exp_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "args.yaml").write_text(
        f"arch: maskrcnn_resnet50_fpn\npretrained: {pretrained}\ndata: {data}\nepochs: {epochs}\n"
        f"batch: {batch}\nlr: {lr}\nmax_side: {MAX_SIDE}\nfliplr: 0.5\nseed: {seed}\n", encoding="utf-8")
    weights_path = models_dir / f"{name}.pth"
    best = -1.0
    with (exp_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["epoch", "train_loss", "mask_ap50", "mask_ap50_95", "box_ap50", "box_ap50_95", "seconds"])
        for epoch in range(1, epochs + 1):
            start, losses = time.time(), []
            model.train()
            for step, (images, targets) in enumerate(loader):
                images = [i.to(device) for i in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                with torch.autocast(device.type, enabled=device.type == "cuda"):
                    loss = sum(model(images, targets).values())
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                losses.append(loss.item())
                if step % 200 == 0:
                    print(f"epoch {epoch} paso {step}/{len(loader)} loss {loss.item():.3f}", flush=True)
            metrics = validate(model, data / "images/val", data / "labels/val", device)
            if metrics["mask_ap50_95"] > best:
                best = metrics["mask_ap50_95"]
                torch.save(model.state_dict(), weights_path)
            row = [epoch, round(float(np.mean(losses)), 4), metrics["mask_ap50"], metrics["mask_ap50_95"],
                   metrics["box_ap50"], metrics["box_ap50_95"], round(time.time() - start)]
            writer.writerow(row)
            f.flush()
            print("epoch", epoch, dict(zip(["train_loss", "mask_ap50", "mask_ap50_95"], row[1:4])), flush=True)
    return weights_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True, help="carpeta con images/ y labels/ (train, val)")
    parser.add_argument("--name", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-pretrained", action="store_true", help="sin pesos COCO (solo para pruebas)")
    args = parser.parse_args()
    weights = train(args.data, args.name, args.epochs, args.batch, args.lr, args.workers, args.seed,
                    pretrained=not args.no_pretrained)
    print(f"Mejores pesos en {weights}")


if __name__ == "__main__":
    main()
