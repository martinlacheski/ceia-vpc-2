"""Importa los dígitos de energía corregidos en Roboflow (export YOLOv8) a data/ceml_energia/digits/.

El export trae los recortes del visor que generó scripts/export_digit_prelabels.py, ya revisados:
  - recupera el nombre original de cada recorte (Roboflow lo cambia a <nombre>_jpg.rf.<hash>.jpg);
  - mapea las clases por nombre a 0-9 (prepare_digits.digit_classes), sin depender del orden del export;
  - convierte a caja los dígitos dibujados como polígono;
  - descarta las imágenes sin cajas (marcadas Null por ilegibles);
  - junta los splits del export: la división en train y val se hace después, por medidor
    (scripts/prepare_digits.py --crops).
Reemplaza una importación anterior.

Uso:
  python scripts/import_digit_labels.py --export data/raw/ceml_energia_digitos_roboflow
"""

import argparse
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_digits import digit_classes, keep_digits  # noqa: E402

ROBOFLOW_SUFFIX = re.compile(r"_(jpe?g|png)\.rf\.[0-9a-zA-Z]+\.(jpe?g|png)$")


def original_name(name: str) -> str:
    """energia_..._jpg.rf.<hash>.jpg → energia_....jpg (los nombres sin sufijo quedan igual)."""
    match = ROBOFLOW_SUFFIX.search(name)
    return f"{name[:match.start()]}.{match.group(1)}" if match else name


def as_box(line: str) -> str:
    """Línea YOLO de caja tal cual; la de un polígono (clase x1 y1 x2 y2 ...) pasa a su caja."""
    parts = line.split()
    if len(parts) <= 5:
        return line
    xs, ys = [float(v) for v in parts[1::2]], [float(v) for v in parts[2::2]]
    x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
    return f"{parts[0]} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}"


def import_export(export: Path, out: Path) -> tuple[int, int]:
    """Copia los recortes con cajas a out/images y out/labels. Devuelve (importadas, sin cajas)."""
    classes = digit_classes(export)
    shutil.rmtree(out, ignore_errors=True)
    (out / "images").mkdir(parents=True)
    (out / "labels").mkdir()
    imported = skipped = 0
    for image_path in sorted(export.glob("*/images/*")):
        label_path = image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
        text = keep_digits(label_path.read_text(encoding="utf-8"), classes) if label_path.exists() else "\n"
        if not text.strip():
            skipped += 1
            continue
        text = "".join(f"{as_box(line)}\n" for line in text.splitlines() if line.strip())
        name = original_name(image_path.name)
        shutil.copy2(image_path, out / "images" / name)
        (out / "labels" / f"{Path(name).stem}.txt").write_text(text, encoding="utf-8")
        imported += 1
    return imported, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--export", type=Path, required=True, help="carpeta del export YOLOv8 descomprimido")
    parser.add_argument("--out", type=Path, default=ROOT / "data/ceml_energia/digits")
    args = parser.parse_args()
    imported, skipped = import_export(args.export, args.out)
    print(f"{imported} recortes con dígitos en {args.out}; {skipped} sin cajas (Null) descartados")


if __name__ == "__main__":
    main()
