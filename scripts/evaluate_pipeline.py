"""Evaluación de punta a punta sobre las fotos CEML verificadas (data/ceml/readings.csv).

Compara configuraciones del pipeline sobre las mismas fotos y la misma segmentación del visor:
  A. rectificado + YOLOv8n + clasificador de formato (pipeline final)
  B. rectificado + YOLOv8n + solo color (sin clasificador)
  C. recorte por caja + YOLOv8n + clasificador (experimento principal: ¿rectificar sirve?)
  D. rectificado + YOLO11n + clasificador
  E. rectificado + EasyOCR + clasificador (OCR genérico como línea de base)

El clasificador de formato se entrenó con estas mismas fotos, así que no se usa el modelo final:
cada foto recibe su predicción out-of-fold (experiments/<clasificador>/oof.csv), hecha por un
modelo que no la vio. Las fotos sin predicción (modelo raro o vacío) caen en la regla de color,
igual que un medidor desconocido en la app.

Métricas (ver score):
  - exacta: la lectura completa coincide, con el punto y los ceros a la izquierda;
  - m³ exactos: coincide la parte entera (lo que se factura), comparada como número;
  - CER: distancia de edición entre dígitos predichos y reales / cantidad de dígitos reales;
  - sin lectura: no se encontró el visor o ningún dígito.
La lectura exacta y el CER se calculan sobre las fotos con la fracción legible; las que terminan
en punto (fracción sin números o no visible) cuentan solo para m³.

Guarda experiments/pipeline_ceml/results.json y predictions.csv (una columna por configuración).

Uso (con el entorno que tiene ultralytics, torch y easyocr, por ejemplo .venv-rocm):
  python scripts/evaluate_pipeline.py
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

REVIEW_CONFIDENCE = 0.5  # por debajo, la app marca la lectura para revisar
LATENCY_IMAGES = 20
CONFIGS = {
    "A": "rectificado + YOLOv8n + clasificador",
    "B": "rectificado + YOLOv8n + color",
    "C": "caja + YOLOv8n + clasificador",
    "D": "rectificado + YOLO11n + clasificador",
    "E": "rectificado + EasyOCR + clasificador",
}


def normalize(value: str) -> str:
    return value.strip().replace(",", ".")


def split_reading(value: str) -> tuple[str, str | None]:
    """(dígitos de m³, dígitos de fracción o None si no hay punto)."""
    if "." in value:
        integer, fraction = value.split(".", 1)
        return integer, fraction
    return value, None


def load_ground_truth(sheet: Path) -> list[dict]:
    """Filas verificadas y legibles: {file, reading, m3, fraction_known}."""
    rows = []
    with sheet.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            reading = normalize(row.get("reading", ""))
            if row.get("legible") != "si" or not reading:
                continue
            integer, fraction = split_reading(reading)
            rows.append({"file": row["file"], "reading": reading, "m3": integer,
                         "fraction_known": fraction != ""})  # "2178." → fracción ilegible
    return rows


def edit_distance(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def same_number(a: str, b: str) -> bool:
    return a.isdigit() and b.isdigit() and int(a) == int(b)


def score(truth: list[dict], predictions: dict[str, tuple[str, float] | None]) -> dict:
    """Métricas de una configuración. predictions: archivo → (lectura con punto, confianza) o None."""
    full = [t for t in truth if t["fraction_known"]]
    exact = m3 = missing = edits = chars = 0
    confident = confident_exact = 0
    for t in truth:
        prediction = predictions.get(t["file"])
        if prediction is None or not prediction[0]:
            missing += 1
            continue
        text, confidence = prediction
        m3 += same_number(split_reading(text)[0], t["m3"])
    for t in full:
        prediction = predictions.get(t["file"])
        text, confidence = prediction if prediction and prediction[0] else ("", 0.0)
        hit = text == t["reading"]
        exact += hit
        truth_digits = t["reading"].replace(".", "")
        edits += edit_distance(text.replace(".", ""), truth_digits)
        chars += len(truth_digits)
        if confidence >= REVIEW_CONFIDENCE:
            confident += 1
            confident_exact += hit
    return {
        "fotos": len(truth),
        "fotos_fraccion_legible": len(full),
        "exacta": round(exact / max(1, len(full)), 4),
        "m3_exactos": round(m3 / max(1, len(truth)), 4),
        "cer": round(edits / max(1, chars), 4),
        "sin_lectura": round(missing / max(1, len(truth)), 4),
        # lo que la app acepta sin pedir revisión: cuántas son y cuántas de ellas están bien
        "cobertura_confianza": round(confident / max(1, len(full)), 4),
        "exacta_con_confianza": round(confident_exact / max(1, confident), 4),
    }


def ocr_digits(results: list) -> tuple[str, float]:
    """Dígitos de la salida de EasyOCR (caja, texto, confianza), de izquierda a derecha."""
    pieces = sorted(results, key=lambda r: min(p[0] for p in r[0]))
    text = "".join(c for _, t, _ in pieces for c in t if c.isdigit())
    confidences = [float(c) for _, t, c in pieces if any(ch.isdigit() for ch in t)]
    return text, min(confidences) if confidences else 0.0


def with_dot(digits: str, red_from: int | None) -> str:
    return digits if red_from is None else f"{digits[:red_from]}.{digits[red_from:]}"


def load_oof(path: Path) -> dict[str, tuple[str, float]]:
    with path.open(newline="", encoding="utf-8") as f:
        return {r["file"]: (r["predicho"], float(r["probabilidad"])) for r in csv.DictReader(f)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sheet", type=Path, default=ROOT / "data/ceml/readings.csv")
    parser.add_argument("--images", type=Path, default=ROOT / "data/ceml/images")
    parser.add_argument("--seg-weights", type=Path, default=ROOT / "models/toloka_yfzw4_yolov8n_seg.pt")
    parser.add_argument("--digits-v8", type=Path, default=ROOT / "models/digits_yolov8n.pt")
    parser.add_argument("--digits-11", type=Path, default=ROOT / "models/digits_yolo11n.pt")
    parser.add_argument("--classifier", default="meter_layout_mnv3s",
                        help="experimento del clasificador: usa su oof.csv y models/<nombre>.pt (latencia)")
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/pipeline_ceml")
    args = parser.parse_args()

    import ultralytics  # noqa: F401  (antes que torch: configura los hilos de OpenMP)
    import cv2
    import easyocr
    import torch

    from meter_reader.layout import fraction_start
    from meter_reader.meter_model import load_classifier
    from meter_reader.pipeline import MIN_MODEL_PROB, load_models, read_meter

    gpu = torch.cuda.is_available()
    device = 0 if gpu else "cpu"
    segment, detect_v8 = load_models(str(args.seg_weights), str(args.digits_v8), device=device)
    _, detect_11 = load_models(str(args.seg_weights), str(args.digits_11), device=device)
    ocr = easyocr.Reader(["en"], gpu=gpu, verbose=False)
    oof = load_oof(ROOT / "experiments" / args.classifier / "oof.csv")
    truth = load_ground_truth(args.sheet)

    predictions = {key: {} for key in CONFIGS}
    for i, t in enumerate(truth, 1):
        image = cv2.imread(str(args.images / t["file"]))
        polygon = segment(image)  # la misma segmentación para todas las configuraciones
        seg = lambda _img, polygon=polygon: polygon  # noqa: E731
        label, probability = oof.get(t["file"], ("", 0.0))
        classify = lambda _input: (label, probability)  # noqa: E731  predicción out-of-fold

        def run(key, **kwargs):
            reading = read_meter(image, seg, **kwargs).reading
            predictions[key][t["file"]] = (reading.formatted, reading.confidence) if reading.text else None

        final = read_meter(image, seg, detect_v8, classify)
        predictions["A"][t["file"]] = (final.reading.formatted, final.reading.confidence) if final.reading.text else None
        run("B", detect=detect_v8)
        run("C", detect=detect_v8, classify=classify, rectified=False)
        run("D", detect=detect_11, classify=classify)
        # E: EasyOCR sobre el mismo recorte (y orientación) que eligió el pipeline final
        if final.crop is None:
            predictions["E"][t["file"]] = None
        else:
            digits, confidence = ocr_digits(ocr.readtext(final.crop, allowlist="0123456789"))
            red_from = fraction_start(len(digits), label) if probability >= MIN_MODEL_PROB and digits else None
            predictions["E"][t["file"]] = (with_dot(digits, red_from), confidence) if digits else None
        if i % 50 == 0:
            print(f"{i}/{len(truth)}", flush=True)

    results = {key: {"configuracion": name, **score(truth, predictions[key])} for key, name in CONFIGS.items()}

    # Latencia del pipeline final en CPU (segmentación + dígitos en dos orientaciones + clasificador)
    cpu_segment, cpu_detect = load_models(str(args.seg_weights), str(args.digits_v8), device="cpu")
    cpu_classify = load_classifier(ROOT / "models" / f"{args.classifier}.pt")
    images = [cv2.imread(str(args.images / t["file"])) for t in truth[:LATENCY_IMAGES]]
    for image in images[:3]:
        read_meter(image, cpu_segment, cpu_detect, cpu_classify)
    start = time.perf_counter()
    for image in images:
        read_meter(image, cpu_segment, cpu_detect, cpu_classify)
    results["latencia_cpu_ms"] = round((time.perf_counter() - start) / len(images) * 1000, 1)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    with (args.out / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["file", "reading", *CONFIGS])
        for t in truth:
            writer.writerow([t["file"], t["reading"],
                             *((predictions[k].get(t["file"]) or ("",))[0] for k in CONFIGS)])
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
