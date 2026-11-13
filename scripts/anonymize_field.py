"""Anonimiza las fotos de campo de medidores de agua.

Copia las fotos de toma_estado/ a data/ceml/images/ con nombres neutros
y sin metadatos EXIF. El nombre del cliente se reemplaza por un identificador
de medidor (m0001, m0002, ...) estable entre meses, para poder agrupar las
fotos del mismo medidor al armar los splits sin filtrar datos personales.

Salidas:
  - data/ceml/images/agua_<AAAA-MM>_r<ruta>_<medidor>.jpg
  - data/ceml/metadata.csv  (sin datos personales, se puede compartir)
  - <origen>/mapa_anonimizacion.csv  (privado: cliente -> medidor, no subir)

Uso:
  python scripts/anonymize_field.py
  python scripts/anonymize_field.py --src "toma_estado/AGUA EL ALCAZAR (PEPE)" --dst data/ceml
"""

import argparse
import csv
import re
from pathlib import Path

from PIL import Image, ImageOps

MONTH_RE = re.compile(r"^(\d{2})-(\d{4})$")
ROUTE_RE = re.compile(r"RUTAS?\s*(\d+)", re.IGNORECASE)
MAX_SIDE = 1600


def parse_photo(path: Path, src: Path) -> dict:
    """Extrae mes, ruta y cliente a partir de la ruta del archivo."""
    rel = path.relative_to(src)
    # El mes se toma de la carpeta de primer nivel (MM-AAAA); las subcarpetas
    # a veces tienen fechas inconsistentes.
    month_match = MONTH_RE.match(rel.parts[0])
    month = f"{month_match[2]}-{month_match[1]}" if month_match else "unknown"
    route_match = ROUTE_RE.search(str(rel))
    route = route_match[1] if route_match else "0"
    # El cliente es la parte del nombre de archivo antes de " - RUTA".
    customer = re.split(r"\s+-\s+RUTA", path.stem, flags=re.IGNORECASE)[0]
    customer = " ".join(customer.upper().split())
    return {"path": path, "month": month, "route": route, "customer": customer}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", type=Path, default=Path("toma_estado/AGUA EL ALCAZAR (PEPE)"))
    parser.add_argument("--dst", type=Path, default=Path("data/ceml"))
    args = parser.parse_args()

    photos = sorted(
        (p for p in args.src.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg"}),
        key=lambda p: str(p).lower(),
    )
    records = [parse_photo(p, args.src) for p in photos]

    # Identificador de medidor estable: orden alfabético de clientes.
    customers = sorted({r["customer"] for r in records})
    meter_ids = {c: f"m{i:04d}" for i, c in enumerate(customers, start=1)}

    images_dir = args.dst / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    used_names: set[str] = set()
    rows = []
    for r in records:
        meter = meter_ids[r["customer"]]
        base = f"agua_{r['month']}_r{r['route']}_{meter}"
        name, n = f"{base}.jpg", 1
        while name in used_names:  # mismo medidor repetido en el mismo mes
            n += 1
            name = f"{base}_{n}.jpg"
        used_names.add(name)

        # exif_transpose aplica la orientación antes de descartar el EXIF.
        with Image.open(r["path"]) as im:
            clean = ImageOps.exif_transpose(im).convert("RGB")
        # Se limita el lado mayor para que el repositorio no pese de más;
        # 1600 px conserva detalle de sobra para modelos que entrenan a 640 px.
        clean.thumbnail((MAX_SIDE, MAX_SIDE))
        clean.save(images_dir / name, "JPEG", quality=90)  # se guarda sin EXIF

        rows.append({**r, "file": name, "meter_id": meter})

    with open(args.dst / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "month", "route", "meter_id"])
        writer.writeheader()
        writer.writerows({k: row[k] for k in writer.fieldnames} for row in rows)

    with open(args.src / "mapa_anonimizacion.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["original", "file", "meter_id"])
        writer.writerows([str(row["path"].relative_to(args.src)), row["file"], row["meter_id"]] for row in rows)

    print(f"{len(rows)} fotos anonimizadas, {len(customers)} medidores -> {images_dir}")


if __name__ == "__main__":
    main()
