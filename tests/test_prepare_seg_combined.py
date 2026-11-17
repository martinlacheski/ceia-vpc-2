"""Tests de scripts/prepare_seg_combined.py con datos sintéticos."""

import importlib.util
import subprocess
import sys
from pathlib import Path

from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_seg_combined.py"


def load_module():
    spec = importlib.util.spec_from_file_location("prepare_seg_combined", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digit_boxes(n: int) -> str:
    """n dígitos en fila horizontal (formato caja YOLO), centrados en y=0.5."""
    return "\n".join(f"{i % 10} {0.2 + 0.1 * i:.2f} 0.5 0.08 0.1" for i in range(n))


def test_display_polygon_wraps_all_digits_with_margin():
    module = load_module()
    poly = module.display_polygon(digit_boxes(6), 1.0, 1.0)

    xs, ys = poly[0::2], poly[1::2]
    # Los dígitos van de x=0.16 a x=0.74 y de y=0.45 a y=0.55: el visor los contiene.
    assert min(xs) < 0.16 and max(xs) > 0.74
    assert min(ys) < 0.45 and max(ys) > 0.55
    assert len(poly) == 8


def test_display_polygon_accepts_quadrilateral_digits():
    module = load_module()
    quad = "3 0.2 0.4 0.3 0.4 0.3 0.6 0.2 0.6"
    lines = "\n".join([quad] * 4 + [digit_boxes(1)])
    assert len(module.display_polygon(lines, 1.0, 1.0)) == 8


def test_display_polygon_rejects_images_with_few_digits():
    module = load_module()
    assert module.display_polygon(digit_boxes(4), 1.0, 1.0) is None


def make_tree(root: Path) -> tuple[Path, Path]:
    """Toloka ya procesado (1 imagen por split) y un yfzw4 con 2 imágenes válidas y 1 descartable."""
    toloka = root / "toloka_seg"
    for split, color in (("train", 10), ("val", 80), ("test", 160)):
        (toloka / "images" / split).mkdir(parents=True)
        (toloka / "labels" / split).mkdir(parents=True)
        Image.new("RGB", (64, 48), (color, 0, 0)).save(toloka / "images" / split / f"t_{split}.jpg")
        (toloka / "labels" / split / f"t_{split}.txt").write_text("0 0.1 0.1 0.5 0.1 0.5 0.3\n")

    yfzw4 = root / "yfzw4"
    (yfzw4 / "train" / "images").mkdir(parents=True)
    (yfzw4 / "train" / "labels").mkdir(parents=True)
    for i, n_digits in enumerate([6, 7, 3]):
        img = Image.new("RGB", (64, 48))
        img.putpixel((i, i), (255, 255, 255))  # imágenes distintas entre sí
        img.paste((0, 200 + i * 20, 0), (0, 0, 32, 24))
        img.save(yfzw4 / "train" / "images" / f"y{i}.jpg")
        (yfzw4 / "train" / "labels" / f"y{i}.txt").write_text(digit_boxes(n_digits) + "\n")
    return toloka, yfzw4


def test_builds_combined_dataset_keeping_toloka_val_and_test(tmp_path):
    toloka, yfzw4 = make_tree(tmp_path)
    out = tmp_path / "out"
    subprocess.run(
        [sys.executable, str(SCRIPT), "--toloka", str(toloka), "--yfzw4", str(yfzw4), "--out", str(out)],
        capture_output=True, text=True, check=True,
    )

    train = sorted(p.name for p in (out / "images" / "train").iterdir())
    assert train == ["t_train.jpg", "yfzw4_y0.jpg", "yfzw4_y1.jpg"]
    assert [p.name for p in (out / "images" / "val").iterdir()] == ["t_val.jpg"]
    assert [p.name for p in (out / "images" / "test").iterdir()] == ["t_test.jpg"]
    label = (out / "labels" / "train" / "yfzw4_y0.txt").read_text().split()
    assert label[0] == "0" and len(label) == 9
    assert f"path: {out.resolve()}" in (out / "data.yaml").read_text()


def test_extra_dataset_goes_to_train_except_the_test_sample(tmp_path):
    toloka, yfzw4 = make_tree(tmp_path)
    energia = tmp_path / "ceml_energia"
    (energia / "images").mkdir(parents=True)
    (energia / "labels").mkdir()
    (energia / "readings.csv").write_text(
        "file,muestra,reading\nenergia_a.jpg,,\nenergia_b.jpg,si,12345.6\nenergia_c.jpg,,\n", encoding="utf-8")
    for name in ("energia_a", "energia_b", "energia_c"):
        Image.new("RGB", (64, 48)).save(energia / "images" / f"{name}.jpg")
    for name in ("energia_a", "energia_b"):  # energia_c no tiene visor etiquetado
        (energia / "labels" / f"{name}.txt").write_text("0 0.1 0.1 0.9 0.1 0.9 0.4\n")
    out = tmp_path / "out"
    subprocess.run(
        [sys.executable, str(SCRIPT), "--toloka", str(toloka), "--yfzw4", str(yfzw4), "--out", str(out),
         "--extra", str(energia)],
        capture_output=True, text=True, check=True,
    )

    train = sorted(p.name for p in (out / "images" / "train").iterdir())
    # Solo la foto etiquetada que no es de la muestra; val y test siguen siendo los de Toloka.
    assert train == ["energia_a.jpg", "t_train.jpg", "yfzw4_y0.jpg", "yfzw4_y1.jpg"]
    assert (out / "labels" / "train" / "energia_a.txt").read_text() == "0 0.1 0.1 0.9 0.1 0.9 0.4\n"
    assert [p.name for p in (out / "images" / "test").iterdir()] == ["t_test.jpg"]
