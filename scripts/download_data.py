"""Descarga los datasets públicos en data/raw/ si todavía no están.

  - Toloka Water Meters (Kaggle): descarga anónima con kagglehub.
  - water-meter-dataset v3 (Roboflow, formato YOLOv8): requiere la variable de
    entorno ROBOFLOW_API_KEY (en Colab se carga desde los "Secrets").

Si la carpeta de destino ya tiene los datos, no se descarga nada.

Uso:
  python scripts/download_data.py              # ambos datasets
  python scripts/download_data.py --only toloka
"""

import argparse
import os
import shutil
from pathlib import Path

RAW = Path("data/raw")
TOLOKA_HANDLE = "tapakah68/yandextoloka-water-meters-dataset"
ROBOFLOW_WORKSPACE = "water-meter-yfzw4"
ROBOFLOW_PROJECT = "water-meter-dataset"
ROBOFLOW_VERSION = 3


def download_toloka(dest: Path = RAW / "toloka") -> None:
    if (dest / "data.csv").exists():
        print(f"Toloka ya está en {dest}")
        return
    import kagglehub

    cache = Path(kagglehub.dataset_download(TOLOKA_HANDLE)) / "WaterMeters"
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cache / "data.csv", dest / "data.csv")
    # Se omiten los collages (~700 MB): no se usan en el pipeline.
    for sub in ("images", "masks"):
        shutil.copytree(cache / sub, dest / sub, dirs_exist_ok=True)
    print(f"Toloka descargado en {dest}")


def download_roboflow(dest: Path = RAW / "roboflow_water_meter") -> None:
    if (dest / "data.yaml").exists():
        print(f"water-meter-dataset ya está en {dest}")
        return
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit("Falta ROBOFLOW_API_KEY: definila como variable de entorno o como Secret de Colab.")
    from roboflow import Roboflow

    project = Roboflow(api_key=api_key).workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    project.version(ROBOFLOW_VERSION).download("yolov8", location=str(dest))
    print(f"water-meter-dataset v{ROBOFLOW_VERSION} descargado en {dest}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", choices=["toloka", "roboflow"])
    args = parser.parse_args()

    if args.only in (None, "toloka"):
        download_toloka()
    if args.only in (None, "roboflow"):
        download_roboflow()


if __name__ == "__main__":
    main()
