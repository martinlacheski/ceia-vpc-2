"""Tests de scripts/prepare_digits.py con un medidor sintético inclinado."""

import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_digits.py"
W, H = 400, 300


def quad(cx: float, cy: float, angle_deg: float, half_w=7.0, half_h=10.0) -> list[float]:
    """Cuadrilátero de un dígito rotado, normalizado por el tamaño de la imagen."""
    a = np.deg2rad(angle_deg)
    u = np.array([np.cos(a), np.sin(a)]) * half_w
    v = np.array([-np.sin(a), np.cos(a)]) * half_h
    c = np.array([cx, cy])
    pts = [c - u - v, c + u - v, c + u + v, c - u + v]
    return [coord for p in pts for coord in (p[0] / W, p[1] / H)]


def make_yfzw4(root: Path, angle: float = 20.0) -> Path:
    """Una imagen en train con la lectura 012345 inclinada y otra en valid con pocos dígitos."""
    a = np.deg2rad(angle)
    for split, digits in (("train", "012345"), ("valid", "12")):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        Image.new("RGB", (W, H), (90, 90, 90)).save(root / split / "images" / "m.jpg")
        lines = []
        for i, d in enumerate(digits):
            offset = (i - (len(digits) - 1) / 2) * 20
            cx, cy = 200 + offset * np.cos(a), 150 + offset * np.sin(a)
            lines.append(d + " " + " ".join(f"{v:.6f}" for v in quad(cx, cy, angle)))
        (root / split / "labels" / "m.txt").write_text("\n".join(lines) + "\n")
    return root


def run(src: Path, out: Path) -> None:
    subprocess.run([sys.executable, str(SCRIPT), "--src", str(src), "--out", str(out)],
                   capture_output=True, text=True, check=True)


def test_crops_rectified_display_with_digit_boxes_in_reading_order(tmp_path):
    out = tmp_path / "out"
    run(make_yfzw4(tmp_path / "yfzw4"), out)

    crop = Image.open(out / "images" / "train" / "m.jpg")
    assert crop.width > crop.height  # visor horizontal

    rows = [list(map(float, l.split())) for l in (out / "labels" / "train" / "m.txt").read_text().splitlines()]
    assert len(rows) == 6
    rows.sort(key=lambda r: r[1])  # de izquierda a derecha
    assert [int(r[0]) for r in rows] == [0, 1, 2, 3, 4, 5]
    centers_y = [r[2] for r in rows]
    assert max(centers_y) - min(centers_y) < 0.05  # todos en la misma fila
    assert all(0 <= v <= 1 for r in rows for v in r[1:])


def test_skips_images_with_few_digits_and_writes_data_yaml(tmp_path):
    out = tmp_path / "out"
    run(make_yfzw4(tmp_path / "yfzw4"), out)

    assert not list((out / "images" / "val").glob("*.jpg"))
    yaml = (out / "data.yaml").read_text()
    assert f"path: {out.resolve()}" in yaml
    assert "names:" in yaml and "'9'" in yaml


def run_many(sources: list[Path], out: Path) -> None:
    cmd = [sys.executable, str(SCRIPT), "--out", str(out)]
    for src in sources:
        cmd += ["--src", str(src)]
    subprocess.run(cmd, capture_output=True, text=True, check=True)


def test_maps_classes_by_name_and_drops_non_digit_classes(tmp_path):
    # Dataset con clases extra: '.' (id 0) y 'kWh' (id 11); los dígitos van de 1 a 10.
    src = make_yfzw4(tmp_path / "lcd")
    label = src / "train" / "labels" / "m.txt"
    lines = [f"{int(l.split()[0]) + 1} {' '.join(l.split()[1:])}" for l in label.read_text().splitlines()]
    lines.append("0 " + " ".join(f"{v:.6f}" for v in quad(260, 160, 20.0, 2, 2)))   # el punto
    lines.append("11 " + " ".join(f"{v:.6f}" for v in quad(380, 290, 0.0, 15, 8)))  # 'kWh' lejos del visor
    label.write_text("\n".join(lines) + "\n")
    names = ["."] + [str(i) for i in range(10)] + ["kWh"]
    (src / "data.yaml").write_text("names:\n" + "".join(f"- '{n}'\n" for n in names))

    out = tmp_path / "out"
    run(src, out)
    rows = [list(map(float, l.split())) for l in (out / "labels" / "train" / "m.txt").read_text().splitlines()]
    rows.sort(key=lambda r: r[1])
    assert [int(r[0]) for r in rows] == [0, 1, 2, 3, 4, 5]  # sin '.' ni 'kWh'
    crop = Image.open(out / "images" / "train" / "m.jpg")
    assert crop.width > 2 * crop.height  # el visor no se estiró hasta la caja de 'kWh'


def test_several_sources_go_to_one_dataset_with_prefixed_names(tmp_path):
    out = tmp_path / "out"
    run_many([make_yfzw4(tmp_path / "agua"), make_yfzw4(tmp_path / "lcd")], out)
    assert sorted(p.name for p in (out / "images" / "train").glob("*.jpg")) == ["agua_m.jpg", "lcd_m.jpg"]
    assert sorted(p.name for p in (out / "labels" / "train").glob("*.txt")) == ["agua_m.txt", "lcd_m.txt"]
