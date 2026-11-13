"""Entrena un modelo YOLO (Ultralytics) y guarda los artefactos finales.

Funciona igual en local (GPU AMD con ROCm, que PyTorch expone como "cuda"),
en Google Colab (GPU NVIDIA) o en CPU. Al terminar:
  - copia el mejor peso a models/<nombre>.pt (versionado en el repositorio);
  - copia métricas, curvas e hiperparámetros a experiments/<nombre>/.

Uso:
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
          batch: int, patience: int, seed: int) -> Path:
    """Entrena con Ultralytics y devuelve la carpeta de la corrida."""
    import torch  # se importa acá para que los tests no requieran PyTorch
    from ultralytics import YOLO

    device = 0 if torch.cuda.is_available() else "cpu"
    # Los pesos preentrenados se descargan en weights/ (ignorado por git).
    weights = Path("weights") / model
    weights.parent.mkdir(exist_ok=True)

    yolo = YOLO(str(weights))
    yolo.train(
        task=task, data=data, epochs=epochs, imgsz=imgsz, batch=batch,
        patience=patience, seed=seed, device=device,
        project=str(Path("runs") / task), name=name, exist_ok=True,
    )
    return Path(yolo.trainer.save_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", choices=["segment", "detect"], required=True)
    parser.add_argument("--model", required=True, help="p. ej. yolov8n-seg.pt")
    parser.add_argument("--data", required=True, help="ruta al data.yaml")
    parser.add_argument("--name", required=True, help="nombre de la corrida y del peso final")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_dir = train(args.task, args.model, args.data, args.name, args.epochs,
                    args.imgsz, args.batch, args.patience, args.seed)
    collect_artifacts(run_dir, args.name, Path("models"), Path("experiments"))
    print(f"Pesos: models/{args.name}.pt | métricas: experiments/{args.name}/")


if __name__ == "__main__":
    main()
