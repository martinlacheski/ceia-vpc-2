"""Tests de scripts/import_ceml_labels.py usando una exportación COCO sintética."""

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "import_ceml_labels.py"


def square(x: float, y: float, side: float) -> list[float]:
    return [x, y, x + side, y, x + side, y + side, x, y + side]


def make_export(root: Path) -> Path:
    """Exportación COCO de Roboflow con una imagen de 100x200 y dos polígonos."""
    coco = {
        "categories": [{"id": 0, "name": "ceml-water-meters"}, {"id": 1, "name": "display"}],
        "images": [{
            "id": 7, "width": 100, "height": 200,
            "file_name": "agua_2019-01_r80_m0011_jpg.rf.abc123.jpg",
            "extra": {"name": "agua_2019-01_r80_m0011.jpg"},
        }],
        "annotations": [
            # Polígono principal (el visor) y un fragmento espurio del auto-label.
            {"id": 1, "image_id": 7, "category_id": 1, "segmentation": [square(10, 20, 50)], "area": 2500},
            {"id": 2, "image_id": 7, "category_id": 1, "segmentation": [square(90, 190, 2)], "area": 4},
        ],
    }
    path = root / "_annotations.coco.json"
    path.write_text(json.dumps(coco), encoding="utf-8")
    return path


def run(coco: Path, out: Path) -> None:
    subprocess.run(
        [sys.executable, str(SCRIPT), "--coco", str(coco), "--out", str(out)],
        capture_output=True, text=True, check=True,
    )


def test_writes_largest_polygon_as_yolo_seg_with_original_name(tmp_path):
    out = tmp_path / "ceml"
    run(make_export(tmp_path), out)

    labels = list((out / "labels").glob("*.txt"))
    assert [p.name for p in labels] == ["agua_2019-01_r80_m0011.txt"]
    # Solo el polígono grande, normalizado por ancho (100) y alto (200).
    assert labels[0].read_text().strip() == "0 0.1 0.1 0.6 0.1 0.6 0.35 0.1 0.35"


def test_writes_clean_coco_with_original_names(tmp_path):
    out = tmp_path / "ceml"
    run(make_export(tmp_path), out)

    coco = json.loads((out / "annotations.coco.json").read_text(encoding="utf-8"))
    assert [i["file_name"] for i in coco["images"]] == ["agua_2019-01_r80_m0011.jpg"]
    assert [a["id"] for a in coco["annotations"]] == [1]
    assert coco["categories"] == [{"id": 1, "name": "display"}]


def test_merge_joins_separate_wheels_into_one_hull(tmp_path):
    # Medidor de rodillos con las ruedas separadas: el auto-label da un polígono por rueda.
    coco_path = make_export(tmp_path)
    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    coco["annotations"] = [
        {"id": 1, "image_id": 7, "category_id": 1, "segmentation": [square(10, 20, 10)], "area": 100},
        {"id": 2, "image_id": 7, "category_id": 1, "segmentation": [square(40, 20, 10)], "area": 100},
    ]
    coco_path.write_text(json.dumps(coco), encoding="utf-8")
    out = tmp_path / "ceml"
    subprocess.run([sys.executable, str(SCRIPT), "--coco", str(coco_path), "--out", str(out), "--merge"],
                   capture_output=True, text=True, check=True)

    label = (out / "labels" / "agua_2019-01_r80_m0011.txt").read_text().split()
    xs, ys = [float(v) for v in label[1::2]], [float(v) for v in label[2::2]]
    # Un solo polígono que va de la primera rueda (x=10) a la última (x=50).
    assert (min(xs), max(xs), min(ys), max(ys)) == (0.1, 0.5, 0.1, 0.15)
    merged = json.loads((out / "annotations.coco.json").read_text(encoding="utf-8"))["annotations"]
    assert len(merged) == 1 and merged[0]["area"] == 400
