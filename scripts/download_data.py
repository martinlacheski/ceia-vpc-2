"""Descarga los datasets públicos en data/raw/ si todavía no están.

  - Toloka Water Meters (Kaggle): en Kaggle se toma del input adjuntado al
    notebook (/kaggle/input); fuera de Kaggle, descarga anónima con kagglehub.
  - water-meter-dataset v3 (Roboflow, formato YOLOv8): requiere la variable de
    entorno ROBOFLOW_API_KEY (en Colab se carga desde los "Secrets").
  - Dígitos de medidores de energía LCD (Roboflow, solo con --only energia):
    meter-digits-5srka v7 y meter-digits-frvqn v2, en data/raw/energia/.

Si la carpeta de destino ya tiene los datos, no se descarga nada.

Uso:
  python scripts/download_data.py              # ambos datasets
  python scripts/download_data.py --only toloka
  python scripts/download_data.py --only energia
"""

import argparse
import os
import shutil
from pathlib import Path

RAW = Path("data/raw")
KAGGLE_INPUT = Path("/kaggle/input")
TOLOKA_HANDLE = "tapakah68/yandextoloka-water-meters-dataset"
ROBOFLOW_WORKSPACE = "water-meter-yfzw4"
ROBOFLOW_PROJECT = "water-meter-dataset"
ROBOFLOW_VERSION = 3
# (workspace, proyecto, versión) de los datasets de dígitos LCD de energía (CC BY 4.0)
ENERGY_DATASETS = [
    ("mohammad-annus", "meter-digits-5srka", 7),
    ("meter-i5vud", "meter-digits-frvqn", 2),
]


def find_attached_toloka(root: Path) -> Path | None:
    """Busca la carpeta de Toloka (data.csv junto a images/) dentro de los inputs de Kaggle."""
    if not root.exists():
        return None
    for csv_path in sorted(root.rglob("data.csv")):
        if (csv_path.parent / "images").is_dir():
            return csv_path.parent
    return None


def download_toloka(dest: Path = RAW / "toloka") -> None:
    if (dest / "data.csv").exists():
        print(f"Toloka ya está en {dest}")
        return

    # En Kaggle, una ejecución en segundo plano no puede adjuntar datasets nuevos:
    # hay que adjuntarlo desde la interfaz (Add Input) y se lo toma de /kaggle/input.
    cache = find_attached_toloka(KAGGLE_INPUT)
    if cache is None:
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


def download_energy(dest: Path = RAW / "energia") -> None:
    pending = [d for d in ENERGY_DATASETS if not (dest / d[1] / "data.yaml").exists()]
    for _, project, _ in ENERGY_DATASETS:
        if (dest / project / "data.yaml").exists():
            print(f"{project} ya está en {dest / project}")
    if not pending:
        return
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit("Falta ROBOFLOW_API_KEY: definila como variable de entorno o como Secret de Colab.")
    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    for workspace, project, version in pending:
        rf.workspace(workspace).project(project).version(version).download("yolov8", location=str(dest / project))
        print(f"{project} v{version} descargado en {dest / project}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", choices=["toloka", "roboflow", "energia"])
    args = parser.parse_args()

    if args.only in (None, "toloka"):
        download_toloka()
    if args.only in (None, "roboflow"):
        download_roboflow()
    if args.only == "energia":
        download_energy()


if __name__ == "__main__":
    main()
