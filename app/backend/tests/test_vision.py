"""MeterReader elige los modelos y la orientación según el servicio del medidor (sin modelos reales)."""

import sys

import cv2
import numpy as np

from app.config import REPO_ROOT, Settings
from app.vision import MeterReader

sys.path.insert(0, str(REPO_ROOT / "src"))
import meter_reader.pipeline as pipeline  # noqa: E402
from meter_reader.reading import Digit  # noqa: E402


def photo_bytes() -> bytes:
    image = np.full((300, 400, 3), 90, np.uint8)
    ok, buffer = cv2.imencode(".jpg", image)
    return buffer.tobytes()


def make_settings(tmp_path) -> Settings:
    paths = {name: tmp_path / f"{name}.pt" for name in ("seg", "digits", "energy_seg", "energy_digits")}
    for path in paths.values():
        path.write_bytes(b"")
    return Settings(
        src_dir=REPO_ROOT / "src",
        seg_weights=paths["seg"], digit_weights=paths["digits"],
        meter_model_weights=tmp_path / "sin_clasificador.pt",
        energy_seg_weights=paths["energy_seg"], energy_digit_weights=paths["energy_digits"],
        energy_meter_model_weights=tmp_path / "sin_clasificador.pt",
        energy_gap_weights=tmp_path / "sin_huecos.pt",
    )


def fake_loader(loaded, detect_calls):
    """Reemplazo de meter_reader.pipeline.load_models: registra los pesos y cuenta las detecciones."""
    def load_models(seg_weights, digit_weights):
        loaded.append((seg_weights, digit_weights))

        def segment(image):
            return np.array([[100, 120], [300, 120], [300, 180], [100, 180]], np.float32)

        def detect(crop):
            detect_calls[digit_weights] = detect_calls.get(digit_weights, 0) + 1
            return [Digit(1, 0.9, 5, 5, 25, 40), Digit(2, 0.9, 30, 5, 50, 40)]

        return segment, detect

    return load_models


def test_each_service_loads_its_own_weights_once(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    loaded, calls = [], {}
    monkeypatch.setattr(pipeline, "load_models", fake_loader(loaded, calls))
    reader = MeterReader(settings)
    for service in ("water", "energy", "water", "energy"):
        assert reader.analyze(photo_bytes(), service).reading == "12"
    assert loaded == [(str(settings.seg_weights), str(settings.digit_weights)),
                      (str(settings.energy_seg_weights), str(settings.energy_digit_weights))]


def test_energy_is_read_as_photographed_and_water_tries_the_flip(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    loaded, calls = [], {}
    monkeypatch.setattr(pipeline, "load_models", fake_loader(loaded, calls))
    reader = MeterReader(settings)
    water, energy = reader.analyze(photo_bytes(), "water"), reader.analyze(photo_bytes(), "energy")
    # agua prueba el visor derecho y girado 180°; energía lo lee una sola vez, tal como está
    assert calls == {str(settings.digit_weights): 2, str(settings.energy_digit_weights): 1}
    assert energy.flipped is False and energy.display_found and water.display_found


def test_energy_uses_the_gap_classifier_and_water_does_not(tmp_path, monkeypatch):
    import dataclasses

    import meter_reader.gaps as gaps_module

    gap_weights = tmp_path / "huecos.pt"
    gap_weights.write_bytes(b"")
    settings = dataclasses.replace(make_settings(tmp_path), energy_gap_weights=gap_weights)
    loaded, calls, gap_calls = [], {}, []
    monkeypatch.setattr(pipeline, "load_models", fake_loader(loaded, calls))

    def load_gap_classifier(weights):
        gap_calls.append(("cargar", str(weights)))
        return lambda crop, digits: gap_calls.append(("leer", len(digits))) or [0.95]

    monkeypatch.setattr(gaps_module, "load_gap_classifier", load_gap_classifier)
    reader = MeterReader(settings)
    # sin clasificador de formato, en energía el punto lo ubica el clasificador de huecos
    assert reader.analyze(photo_bytes(), "energy").reading == "1.2"
    assert reader.analyze(photo_bytes(), "water").reading == "12"
    assert gap_calls == [("cargar", str(gap_weights)), ("leer", 2)]
