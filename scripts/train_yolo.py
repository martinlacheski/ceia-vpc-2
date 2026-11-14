"""Entrena un modelo YOLO (Ultralytics) y guarda los artefactos finales.

Funciona igual en local (GPU AMD con ROCm, que PyTorch expone como "cuda"),
en Google Colab (GPU NVIDIA) o en CPU. Al terminar:
  - copia el mejor peso a models/<nombre>.pt (versionado en el repositorio);
  - copia métricas, curvas e hiperparámetros a experiments/<nombre>/.

Uso (para retomar una corrida interrumpida: --task segment --name <nombre> --resume):
  python scripts/train_yolo.py --task segment --model yolov8n-seg.pt \
      --data data/processed/toloka_seg/data.yaml --name toloka_yolov8n_seg
"""

import argparse
import shutil
from pathlib import Path

# Artefactos livianos que se versionan (el resto de la corrida queda en runs/).
LIGHT_SUFFIXES = {".csv", ".yaml", ".png", ".jpg"}


def collect_artifacts(run_dir: Path, name: str, models_dir: Path, experiments_dir: Path) -> None:
    """Copia best.pt a models/ y los archivos livianos de la corrida a experiments/."""
    models_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(run_dir / "weights" / "best.pt", models_dir / f"{name}.pt")

    exp_dir = experiments_dir / name
    shutil.rmtree(exp_dir, ignore_errors=True)
    exp_dir.mkdir(parents=True)
    for path in run_dir.iterdir():
        if path.is_file() and path.suffix in LIGHT_SUFFIXES:
            shutil.copy2(path, exp_dir / path.name)


def train(task: str, model: str, data: str, name: str, epochs: int, imgsz: int,
          batch: int, patience: int, seed: int, fliplr: float = 0.5) -> Path:
    """Entrena con Ultralytics y devuelve la carpeta de la corrida.

    fliplr: probabilidad de espejado horizontal (0 para dígitos: un 2 espejado parece un 5).
    """
    # Se importan acá para que los tests no requieran PyTorch. ultralytics va primero: configura
    # los hilos de OpenMP al importarse y eso solo tiene efecto si torch todavía no está cargado.
    from ultralytics import YOLO
    import torch

    device = 0 if torch.cuda.is_available() else "cpu"
    # Los pesos preentrenados se descargan en weights/ (ignorado por git).
    weights = Path("weights") / model
    weights.parent.mkdir(exist_ok=True)

    yolo = YOLO(str(weights))
    yolo.train(
        task=task, data=data, epochs=epochs, imgsz=imgsz, batch=batch,
        patience=patience, seed=seed, device=device, fliplr=fliplr,
        # Ruta absoluta: con una relativa, Ultralytics la anida dentro de su runs_dir.
        project=str((Path("runs") / task).resolve()), name=name, exist_ok=True,
    )
    return Path(yolo.trainer.save_dir)


def resume(task: str, name: str) -> Path:
    """Retoma una corrida interrumpida desde su último punto de control (last.pt)."""
    from ultralytics import YOLO

    last = (Path("runs") / task / name / "weights" / "last.pt").resolve()
    if not last.exists():
        raise SystemExit(f"No hay punto de control para retomar en {last}")
    yolo = YOLO(str(last))
    # Ultralytics recupera datos, épocas e hiperparámetros guardados en last.pt.
    yolo.train(resume=True)
    return Path(yolo.trainer.save_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", choices=["segment", "detect"], required=True)
    parser.add_argument("--name", required=True, help="nombre de la corrida y del peso final")
    parser.add_argument("--resume", action="store_true",
                        help="retoma la corrida --name interrumpida (no requiere --model ni --data)")
    parser.add_argument("--model", help="p. ej. yolov8n-seg.pt")
    parser.add_argument("--data", help="ruta al data.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fliplr", type=float, default=0.5,
                        help="probabilidad de espejado horizontal; usar 0 para dígitos")
    args = parser.parse_args()

    if args.resume:
        run_dir = resume(args.task, args.name)
    else:
        if not (args.model and args.data):
            parser.error("--model y --data son obligatorios salvo con --resume")
        run_dir = train(args.task, args.model, args.data, args.name, args.epochs,
                        args.imgsz, args.batch, args.patience, args.seed, args.fliplr)
    collect_artifacts(run_dir, args.name, Path("models"), Path("experiments"))
    print(f"Pesos: models/{args.name}.pt | métricas: experiments/{args.name}/")


if __name__ == "__main__":
    main()
