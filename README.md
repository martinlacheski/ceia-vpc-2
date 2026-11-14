# Lectura automática de medidores de agua con visión por computadora

Trabajo práctico final de **Visión por Computadora II** — CEIA, FIUBA (2026).

## Objetivo

Leer automáticamente el estado de medidores de agua de tipo rodillo a partir de fotos tomadas en campo, con un pipeline basado en redes neuronales convolucionales:

1. **Segmentación de instancia del visor**: YOLOv8n-seg vs Mask R-CNN.
2. **Rectificación**: se ajusta un rectángulo rotado mínimo a la máscara y se endereza el visor con una transformación afín (`src/meter_reader/rectify.py`).
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

## App web de toma de lecturas

Prototipo para el uso real del sistema: el lecturista inicia sesión, recorre su ruta (listado y mapa), saca la foto del medidor, el pipeline propone la lectura y él la confirma o corrige; se guardan la foto, las imágenes procesadas y la ubicación GPS donde se tomó la foto. El administrador ve el historial y el porcentaje de lecturas aceptadas sin corrección.

```bash
docker compose up -d --build
```

Abrir http://localhost:8080. Usuarios de demo: `lector1`, `lector2` y `admin`, con contraseña `demo1234` (solo para desarrollo local). Los medidores son los de las fotos CEML, con direcciones y coordenadas ficticias; el botón *Usar foto de ejemplo* analiza la foto real de cada medidor.

| Servicio | Qué es |
|---|---|
| `web` | Frontend React + Vite servido por nginx ([app/frontend](app/frontend/)) |
| `api` | FastAPI con el pipeline de visión ([app/backend](app/backend/)) |
| `migrate` | Aplica las migraciones Alembic de [app/db](app/db/) y termina |
| `db` | PostgreSQL |
| `storage` | SeaweedFS (API S3) para las fotos y las imágenes procesadas ([app/storage](app/storage/)) |

`docker compose down` conserva los datos; `docker compose down -v` borra la base y las imágenes.

## Integrantes

- Lacheski, Martin
- Lastra, Nicolas
- Scheidl, Hernando
- Vazquez, Diego

