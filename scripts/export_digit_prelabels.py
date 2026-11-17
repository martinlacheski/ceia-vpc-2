"""Exporta los visores de energía enderezados con pre-etiquetas de dígitos, para corregirlas en Roboflow.

Para cada foto de data/ceml_energia con el visor etiquetado que no es de la muestra de prueba
(muestra=si queda afuera, porque es el test de punta a punta):
  - endereza y recorta el visor a partir de su polígono revisado (labels/), con el mismo margen
    que usa el pipeline (CROP_PAD) y sin girarlo 180°: los medidores de energía se fotografían
    derechos, igual que en `read_meter(..., allow_flip=False)`;
  - propone las cajas de dígitos con el detector actual, una por posición y de izquierda a
    derecha (mismo criterio que la lectura), en formato YOLO con clases 0-9.

El detector se entrena después con estos mismos recortes, así que ve el mismo encuadre que en el
pipeline. La salida (images/, labels/ y data.yaml) se comprime en <out>.zip para subirla a
Roboflow como dataset YOLO y corregir las cajas y los números.

Uso (con el entorno que tiene ultralytics, por ejemplo .venv-rocm):
  python scripts/export_digit_prelabels.py --out data/interim/energia_digitos
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from meter_reader.pipeline import CROP_PAD, load_models  # noqa: E402
from meter_reader.reading import Digit, assemble_reading  # noqa: E402
from meter_reader.rectify import rectify  # noqa: E402
from train_meter_model import read_polygon  # noqa: E402

DigitDetector = Callable[[np.ndarray], list[Digit]]


def yolo_lines(digits: list[Digit], width: int, height: int) -> list[str]:
    """Cajas de dígitos en píxeles del recorte → líneas YOLO (clase cx cy w h, normalizadas)."""
    lines = []
    for d in digits:
        x1, x2 = max(0.0, d.x1), min(float(width), d.x2)
        y1, y2 = max(0.0, d.y1), min(float(height), d.y2)
        if x2 <= x1 or y2 <= y1:
            continue
        lines.append(f"{d.value} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} "
                     f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}")
    return lines


def export(root: Path, out: Path, detect: DigitDetector) -> tuple[int, int]:
    """Escribe los recortes y sus pre-etiquetas en out. Devuelve (fotos, cajas)."""
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} no está vacía: borrarla o elegir otra salida")
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(exist_ok=True)
    with (root / "readings.csv").open(newline="", encoding="utf-8") as f:
        files = [r["file"] for r in csv.DictReader(f) if r.get("muestra", "").strip().lower() != "si"]

    photos = boxes = 0
    for name in files:
        image = cv2.imread(str(root / "images" / name))
        if image is None:
            continue
        polygon = read_polygon(root / "labels" / f"{Path(name).stem}.txt", image.shape[1], image.shape[0])
        if polygon is None:
            continue
        crop = rectify(image, polygon, pad=CROP_PAD)[0]
        lines = yolo_lines(list(assemble_reading(detect(crop)).digits), crop.shape[1], crop.shape[0])
        cv2.imwrite(str(out / "images" / name), crop)
        (out / "labels" / f"{Path(name).stem}.txt").write_text("".join(f"{line}\n" for line in lines),
                                                               encoding="utf-8")
        photos += 1
        boxes += len(lines)
    names = ", ".join(f"'{i}'" for i in range(10))
    (out / "data.yaml").write_text(f"train: images\nval: images\nnc: 10\nnames: [{names}]\n", encoding="utf-8")
    return photos, boxes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=ROOT / "data/ceml_energia")
    parser.add_argument("--out", type=Path, default=ROOT / "data/interim/energia_digitos")
    parser.add_argument("--digits", type=Path, default=ROOT / "models/digits_mix_yolov8n.pt")
    args = parser.parse_args()

    # El mismo detector que usa la lectura (load_models arma también el segmentador, que acá no se usa:
    # el visor sale del polígono revisado).
    _, detect = load_models(str(ROOT / "models/toloka_yfzw4_energia_yolov8n_seg.pt"), str(args.digits))
    photos, boxes = export(args.root, args.out, detect)
    archive = shutil.make_archive(str(args.out), "zip", root_dir=args.out)
    print(f"{photos} recortes y {boxes} cajas de dígitos en {args.out}; para Roboflow: {archive}")


if __name__ == "__main__":
    main()
