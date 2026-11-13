# Lectura automática de medidores de agua con visión por computadora

Trabajo práctico final de **Visión por Computadora II** — CEIA, FIUBA (2026).

## Objetivo

Leer automáticamente el estado de medidores de agua de tipo rodillo a partir de fotos tomadas en campo, con un pipeline basado en redes neuronales convolucionales:

1. **Segmentación de instancia del visor**: YOLOv8n-seg vs Mask R-CNN.
2. **Rectificación**: se usa la máscara para enderezar el visor mediante una homografía.
3. **Detección de dígitos**: YOLOv8 con una clase por dígito (0–9); la lectura se obtiene ordenando las detecciones.
4. **Baseline**: OCR genérico (EasyOCR) sobre el mismo recorte.

**Experimento principal:** evaluar si segmentar y rectificar el visor mejora la lectura frente a un recorte por bounding box. Métricas: exactitud de lectura completa y CER.

## Datos

Ver [data/README.md](data/README.md).

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Integrantes

- Martín Lacheski
