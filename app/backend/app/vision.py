"""Lector de medidores: adaptador entre la app y el pipeline `meter_reader`.

El lector es una dependencia de FastAPI (`get_reader`) para poder reemplazarlo en los tests.
Torch, Ultralytics y OpenCV se importan recién al leer la primera foto.
"""

import sys
import threading
from dataclasses import dataclass

from fastapi import Request

from .config import Settings

OVERLAY_MAX_SIDE = 1024


@dataclass(frozen=True)
class ReaderOutput:
    """Resultado de analizar una foto. Las imágenes son JPEG en bytes (None si no hay visor)."""

    display_found: bool
    reading: str | None = None
    confidence: float | None = None
    flipped: bool = False
    digits_model_available: bool = True
    overlay_jpeg: bytes | None = None
    crop_jpeg: bytes | None = None
    digits_jpeg: bytes | None = None


@dataclass
class _Models:
    """Modelos de un servicio. Sin detector de dígitos solo se segmenta el visor."""

    segment: object
    detect: object = None
    classify: object = None


class MeterReader:
    """Lector real, con un juego de modelos por servicio ("water" o "energy") que se carga recién
    al leer la primera foto de ese servicio. Si faltan los pesos de dígitos solo segmenta el visor
    (modo sin lectura); si falta el clasificador del modelo de medidor, la fracción se separa por
    el color. Energía se lee sin probar el visor girado 180° (ver read_meter, allow_flip)."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._lock = threading.Lock()  # una inferencia a la vez
        self._models: dict[str, _Models] = {}

    def digits_model_available(self, service: str = "water") -> bool:
        return self.settings.weights(service)[1].exists()

    def _load(self, service: str) -> _Models:
        if service in self._models:
            return self._models[service]
        src = str(self.settings.src_dir)
        if src not in sys.path:
            sys.path.insert(0, src)
        seg_weights, digit_weights, meter_model_weights = self.settings.weights(service)
        if digit_weights.exists():
            from meter_reader import pipeline

            models = _Models(*pipeline.load_models(str(seg_weights), str(digit_weights)))
            if meter_model_weights.exists():
                from meter_reader.meter_model import load_classifier

                models.classify = load_classifier(meter_model_weights)
        else:
            models = _Models(self._load_segmenter_only(seg_weights))
        self._models[service] = models
        return models

    @staticmethod
    def _load_segmenter_only(seg_weights):
        """Segmentador equivalente al de `load_models`, sin exigir los pesos de dígitos."""
        from ultralytics import YOLO

        model = YOLO(str(seg_weights))

        def segment(image):
            result = model.predict(image, imgsz=640, conf=0.25, verbose=False)[0]
            if result.masks is None or len(result.masks) == 0:
                return None
            return result.masks.xy[int(result.boxes.conf.argmax())]

        return segment

    def analyze(self, photo: bytes, service: str = "water") -> ReaderOutput:
        import cv2
        import numpy as np

        image = cv2.imdecode(np.frombuffer(photo, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("no es una imagen")
        with self._lock:
            return self._run(image, self._load(service), allow_flip=service != "energy")

    def _run(self, image, models: _Models, allow_flip: bool) -> ReaderOutput:
        from meter_reader import pipeline
        from meter_reader.rectify import rectify

        available = models.detect is not None
        if available:
            result = pipeline.read_meter(image, models.segment, models.detect, models.classify,
                                         allow_flip=allow_flip)
            polygon, crop, flipped, reading = result.polygon, result.crop, result.flipped, result.reading
        else:
            polygon = models.segment(image)
            if polygon is not None and len(polygon) < 3:
                polygon = None
            crop = rectify(image, polygon, pad=pipeline.CROP_PAD)[0] if polygon is not None else None
            flipped, reading = False, None

        if polygon is None or crop is None:
            return ReaderOutput(display_found=False, digits_model_available=available)

        has_digits = available and reading is not None and bool(reading.text)
        return ReaderOutput(
            display_found=True,
            reading=reading.formatted if has_digits else None,  # con punto entre m³ y fracción
            confidence=round(reading.confidence, 4) if has_digits else None,
            flipped=flipped,
            digits_model_available=available,
            overlay_jpeg=_encode(_draw_overlay(image, polygon)),
            crop_jpeg=_encode(crop),
            digits_jpeg=_encode(_draw_digits(crop, reading.digits, reading.red_from)) if has_digits else None,
        )


def _encode(image) -> bytes:
    import cv2

    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buffer.tobytes() if ok else b""


def _draw_overlay(image, polygon):
    """Foto con el polígono del visor resaltado, reducida a OVERLAY_MAX_SIDE."""
    import cv2
    import numpy as np

    points = np.asarray(polygon, dtype=np.int32).reshape(-1, 1, 2)
    canvas = image.copy()
    filled = canvas.copy()
    cv2.fillPoly(filled, [points], (0, 200, 0))
    canvas = cv2.addWeighted(filled, 0.35, canvas, 0.65, 0)
    thickness = max(2, round(max(image.shape[:2]) / 400))
    cv2.polylines(canvas, [points], True, (0, 255, 0), thickness)
    scale = OVERLAY_MAX_SIDE / max(canvas.shape[:2])
    if scale < 1:
        canvas = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return canvas


def _draw_digits(crop, digits, red_from=None):
    """Recorte con la caja y la clase de cada dígito: verde los m³, naranja la fracción (rojos)."""
    import cv2

    canvas = crop.copy()
    scale = max(0.4, canvas.shape[0] / 120)
    for i, d in enumerate(digits):
        p1, p2 = (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2))
        fraction = red_from is not None and i >= red_from
        cv2.rectangle(canvas, p1, p2, (0, 140, 255) if fraction else (0, 200, 0), 2)
        cv2.putText(canvas, str(d.value), (p1[0], max(p1[1] - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, scale * 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    return canvas


def get_reader(request: Request) -> MeterReader:
    """Dependencia: lector compartido por todos los pedidos (se crea una sola vez)."""
    state = request.app.state
    if getattr(state, "reader", None) is None:
        state.reader = MeterReader(state.settings)
    return state.reader
