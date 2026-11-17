"""Importa las anotaciones del visor exportadas de Roboflow (COCO Segmentation).

Roboflow renombra las imágenes (agua_..._jpg.rf.<hash>.jpg) y el auto-label a
veces agrega fragmentos espurios además del visor. Este script:
  - recupera el nombre original de cada imagen (campo extra.name);
  - conserva un solo polígono por imagen, el de mayor área (el visor);
    con --merge, en cambio, une todos los polígonos en su envolvente convexa
    (medidores de rodillos con las ruedas separadas, donde el auto-label da un
    polígono por rueda; los polígonos fuera del visor se borran al revisar);
  - escribe las etiquetas en formato YOLO-seg en <out>/labels/<nombre>.txt;
  - escribe un COCO limpio en <out>/annotations.coco.json (para Mask R-CNN).

Las coordenadas son válidas para data/ceml/images/ porque la versión de
Roboflow se exporta sin redimensionar.

Uso:
  python scripts/import_ceml_labels.py --coco data/raw/ceml_roboflow/train/_annotations.coco.json
  python scripts/import_ceml_labels.py --coco data/raw/ceml_energia_roboflow/train/_annotations.coco.json \
      --out data/ceml_energia --merge
"""

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

DISPLAY = "display"


def to_yolo(polygon: list[float], width: int, height: int) -> str:
    """Normaliza un polígono COCO [x1, y1, x2, y2, ...] a una línea YOLO-seg."""
    coords = []
    for x, y in zip(polygon[::2], polygon[1::2]):
        coords += [min(max(x / width, 0.0), 1.0), min(max(y / height, 0.0), 1.0)]
    return "0 " + " ".join(f"{c:.6g}" for c in coords)


def convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Envolvente convexa en sentido antihorario (cadena monótona de Andrew)."""
    points = sorted(set(points))
    if len(points) < 3:
        return points

    def half(seq):
        chain = []
        for p in seq:
            while len(chain) >= 2 and (
                (chain[-1][0] - chain[-2][0]) * (p[1] - chain[-2][1])
                - (chain[-1][1] - chain[-2][1]) * (p[0] - chain[-2][0])
            ) <= 0:
                chain.pop()
            chain.append(p)
        return chain[:-1]

    return half(points) + half(reversed(points))


def merge_annotations(anns: list[dict]) -> dict:
    """Une los polígonos de una imagen en uno solo (su envolvente convexa), con área y caja."""
    points = [(x, y) for a in anns for seg in a["segmentation"] for x, y in zip(seg[::2], seg[1::2])]
    hull = convex_hull(points)
    xs, ys = [p[0] for p in hull], [p[1] for p in hull]
    area = abs(sum(xs[i] * ys[i - 1] - xs[i - 1] * ys[i] for i in range(len(hull)))) / 2
    first = min(anns, key=lambda a: a["id"])
    return {**first, "segmentation": [[c for p in hull for c in p]], "area": area,
            "bbox": [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--coco", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("data/ceml"))
    parser.add_argument("--merge", action="store_true",
                        help="unir todos los polígonos de cada imagen en vez de quedarse con el mayor")
    args = parser.parse_args()

    coco = json.loads(args.coco.read_text(encoding="utf-8"))
    display_id = next(c["id"] for c in coco["categories"] if c["name"] == DISPLAY)

    by_image = defaultdict(list)
    for ann in coco["annotations"]:
        if ann["category_id"] == display_id and ann["segmentation"]:
            by_image[ann["image_id"]].append(ann)

    # Se regeneran las etiquetas desde cero para no mezclar versiones.
    labels_dir = args.out / "labels"
    shutil.rmtree(labels_dir, ignore_errors=True)
    labels_dir.mkdir(parents=True)

    images, annotations, dropped = [], [], 0
    for image in coco["images"]:
        anns = by_image.get(image["id"])
        if not anns:
            continue
        if args.merge:
            best = merge_annotations(anns)
        else:
            best = max(anns, key=lambda a: a["area"])
            dropped += len(anns) - 1
        name = image.get("extra", {}).get("name", image["file_name"])

        line = to_yolo(best["segmentation"][0], image["width"], image["height"])
        (labels_dir / f"{Path(name).stem}.txt").write_text(line + "\n", encoding="utf-8")

        images.append({"id": image["id"], "file_name": name, "width": image["width"], "height": image["height"]})
        annotations.append({**best, "segmentation": [best["segmentation"][0]]})

    clean = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": display_id, "name": DISPLAY}],
    }
    (args.out / "annotations.coco.json").write_text(json.dumps(clean, indent=1), encoding="utf-8")

    print(f"{len(images)} imágenes etiquetadas, {dropped} polígonos espurios descartados -> {labels_dir}")


if __name__ == "__main__":
    main()
