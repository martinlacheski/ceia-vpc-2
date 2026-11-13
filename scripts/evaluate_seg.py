"""Evalúa un modelo de segmentación del visor en Toloka (test) y en las fotos CEML.

Mide métricas de caja y de máscara (mAP50, mAP50-95, precisión, recall) y la
latencia de inferencia en CPU. La latencia se mide aparte, con calentamiento
previo y tamaño fijo, porque la primera pasada en GPU (ROCm) compila kernels.

Guarda el resultado en experiments/<nombre>/eval.json.

Uso:
  python scripts/evaluate_seg.py --model models/toloka_yolov8n_seg.pt
"""

import argparse
import json
import time
from pathlib import Path

CEML = Path("data/ceml")
TOLOKA_YAML = Path("data/processed/toloka_seg/data.yaml")
LATENCY_IMAGES = 50


def write_ceml_yaml(ceml_dir: Path, yaml_path: Path) -> Path:
    """Crea el data.yaml de CEML: todas las fotos se usan como conjunto de evaluación."""
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(
        f"path: {ceml_dir.resolve()}\ntrain: images\nval: images\nnames: {{0: 'display'}}\n",
        encoding="utf-8",
    )
    return yaml_path


def save_results(results: dict, name: str, experiments_dir: Path) -> Path:
    out = experiments_dir / name / "eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return out


def metrics(model, data: Path, split: str, imgsz: int, device) -> dict:
    r = model.val(data=str(data), split=split, imgsz=imgsz, device=device, plots=False,
                  verbose=False, project=str(Path("runs/val").resolve()), name="eval", exist_ok=True)
    return {
        "box_map50": round(r.box.map50, 4), "box_map50_95": round(r.box.map, 4),
        "mask_map50": round(r.seg.map50, 4), "mask_map50_95": round(r.seg.map, 4),
        "mask_precision": round(r.seg.mp, 4), "mask_recall": round(r.seg.mr, 4),
    }


def cpu_latency_ms(model, images: list[Path], imgsz: int) -> float:
    """Latencia media por imagen en CPU (inferencia completa), tras 5 imágenes de calentamiento."""
    for img in images[:5]:
        model.predict(str(img), imgsz=imgsz, device="cpu", verbose=False)
    start = time.perf_counter()
    for img in images:
        model.predict(str(img), imgsz=imgsz, device="cpu", verbose=False)
    return round((time.perf_counter() - start) / len(images) * 1000, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    import torch
    from ultralytics import YOLO

    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(str(args.model))
    ceml_yaml = write_ceml_yaml(CEML, Path("runs/val/ceml.yaml"))
    images = sorted((CEML / "images").glob("*.jpg"))[:LATENCY_IMAGES]

    results = {
        "toloka_test": metrics(model, TOLOKA_YAML, "test", args.imgsz, device),
        "ceml": metrics(model, ceml_yaml, "val", args.imgsz, device),
        "cpu_latency_ms": cpu_latency_ms(model, images, args.imgsz),
        "imgsz": args.imgsz,
    }
    out = save_results(results, args.model.stem, Path("experiments"))
    print(json.dumps(results, indent=2))
    print(f"Guardado en {out}")


if __name__ == "__main__":
    main()
