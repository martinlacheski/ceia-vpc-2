"""Tests de la recolección de artefactos de scripts/train_yolo.py."""

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_yolo.py"


def load_module():
    spec = importlib.util.spec_from_file_location("train_yolo", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_run(root: Path) -> Path:
    """Simula la carpeta que deja Ultralytics al terminar un entrenamiento."""
    run = root / "runs" / "segment" / "demo"
    (run / "weights").mkdir(parents=True)
    (run / "weights" / "best.pt").write_bytes(b"best")
    (run / "weights" / "last.pt").write_bytes(b"last")
    (run / "results.csv").write_text("epoch,loss\n1,0.5\n")
    (run / "args.yaml").write_text("epochs: 1\n")
    (run / "results.png").write_bytes(b"png")
    (run / "val_batch0_pred.jpg").write_bytes(b"jpg")
    return run


def test_collect_copies_best_weights_and_light_artifacts(tmp_path):
    module = load_module()
    run = make_run(tmp_path)

    module.collect_artifacts(run, "demo", tmp_path / "models", tmp_path / "experiments")

    assert (tmp_path / "models" / "demo.pt").read_bytes() == b"best"
    exp = tmp_path / "experiments" / "demo"
    assert sorted(p.name for p in exp.iterdir()) == ["args.yaml", "results.csv", "results.png", "val_batch0_pred.jpg"]


def fake_training(monkeypatch, tmp_path):
    """Reemplaza torch y ultralytics por dobles que registran los argumentos de entrenamiento."""
    import sys
    import types

    calls = {}

    class FakeYOLO:
        def __init__(self, weights):
            self.trainer = types.SimpleNamespace(save_dir=str(tmp_path / "run"))

        def train(self, **kwargs):
            calls.update(kwargs)

    torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))
    monkeypatch.chdir(tmp_path)
    return calls


def test_train_passes_horizontal_flip_probability(monkeypatch, tmp_path):
    calls = fake_training(monkeypatch, tmp_path)
    load_module().train("detect", "yolov8n.pt", "d.yaml", "digits", 1, 416, 8, 5, 42, fliplr=0.0)
    assert calls["fliplr"] == 0.0


def test_train_keeps_ultralytics_default_flip(monkeypatch, tmp_path):
    calls = fake_training(monkeypatch, tmp_path)
    load_module().train("segment", "yolov8n-seg.pt", "d.yaml", "seg", 1, 640, 8, 5, 42)
    assert calls["fliplr"] == 0.5


def test_train_forwards_extra_hyperparameters(monkeypatch, tmp_path):
    calls = fake_training(monkeypatch, tmp_path)
    load_module().train("detect", "yolov8n.pt", "d.yaml", "digits", 1, 416, 8, 5, 42, fliplr=0.0,
                        extra={"hsv_v": 0.5, "degrees": 5.0})
    assert calls["hsv_v"] == 0.5 and calls["degrees"] == 5.0
