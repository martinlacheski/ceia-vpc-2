# Backend de la app de lecturas

API REST (FastAPI + SQLAlchemy) para que un lecturista recorra su ruta, fotografíe el medidor,
reciba la lectura propuesta por el pipeline de visión (`src/meter_reader`) y la confirme o corrija.

## Ejecutar en local (sin Docker)

Desde la raíz del repo, con un entorno que tenga torch y ultralytics (p. ej. `.venv-rocm`) más las
dependencias de la API:

```bash
pip install -r app/backend/requirements-dev.txt   # incluye las dependencias de tests
cd app/backend
mkdir -p ../data
export DATABASE_URL=sqlite:///../data/app.db APP_DATA_DIR=../data
alembic -c ../db/alembic.ini upgrade head   # la API no crea tablas (ver app/db/README.md)
PYTHONPATH=../../src \
uvicorn app.main:app --reload --port 8000
```

`PYTHONPATH=../../src` (o `METER_READER_SRC`) permite importar el paquete `meter_reader`; si no se
define, el backend agrega solo `<repo>/src` al `sys.path` al cargar los modelos. Documentación
interactiva en `http://localhost:8000/docs`. Los tests no necesitan torch ni ultralytics.

## Variables de entorno

| Variable | Por defecto | Uso |
|---|---|---|
| `DATABASE_URL` | `sqlite:///app/data/app.db` | PostgreSQL en compose: `postgresql+psycopg://usuario:clave@db:5432/lecturas` |
| `APP_DATA_DIR` | `app/data` | Fotos e imágenes de los análisis cuando `S3_ENDPOINT` no está definido (ignorado por git) |
| `S3_ENDPOINT` | vacío | Si se define (compose: `http://storage:8333`), las imágenes van a S3/SeaweedFS en vez de al disco |
| `S3_ACCESS_KEY` / `S3_SECRET_KEY` | claves locales de desarrollo | Credenciales S3. **Cambiarlas fuera de la demo local.** |
| `S3_BUCKET` | `lecturas` | Bucket; se crea al arrancar si falta |
| `SECRET_KEY` | clave de desarrollo | Firma de la cookie de sesión. **Cambiarla fuera de la demo local.** |
| `COOKIE_SECURE` | `false` | `true` si se sirve por HTTPS (cookie solo por canal seguro) |
| `DEMO_PASSWORD` | `demo1234` | Contraseña inicial de los usuarios de demo |
| `CEML_DIR` | `<repo>/data/ceml` | `metadata.csv` e `images/` de CEML (solo lectura) |
| `SEG_WEIGHTS` | `<repo>/models/toloka_yfzw4_yolov8n_seg.pt` | Pesos de segmentación del visor |
| `DIGIT_WEIGHTS` | `<repo>/models/digits_yolov8n.pt` | Pesos del detector de dígitos |
| `METER_MODEL_WEIGHTS` | `<repo>/models/meter_layout_mnv3s.pt` | Clasificador del modelo de medidor (dónde va el punto) |
| `CEML_ENERGY_DIR` | `<repo>/data/ceml_energia` | `metadata.csv` e `images/` de los medidores de energía |
| `ENERGY_SEG_WEIGHTS` | `<repo>/models/toloka_yfzw4_energia_yolov8n_seg.pt` | Segmentador para medidores de energía |
| `ENERGY_DIGIT_WEIGHTS` | `<repo>/models/digits_mix_energia_yolov8n.pt` | Detector de dígitos para energía |
| `ENERGY_METER_MODEL_WEIGHTS` | `<repo>/models/meter_layout_agua_energia.pt` | Clasificador del modelo de medidor para energía |
| `METER_READER_SRC` | `<repo>/src` | Carpeta que contiene el paquete `meter_reader` |

## Datos de demo

Al arrancar (con el esquema ya migrado) se cargan (de forma idempotente) las rutas y medidores a partir
de `data/ceml/metadata.csv` (agua: "Ruta 80" y "Ruta 90") y de `data/ceml_energia/metadata.csv`
(energía: "Ruta 1 (energía)"), un medidor por `meter_id`. Los códigos de energía llevan el prefijo
`E-` porque CEML repite ids entre los dos servicios. Las direcciones y
las coordenadas son **ficticias** (generadas con semilla fija alrededor de un punto genérico) y no
corresponden a ubicaciones reales. La foto de muestra de cada medidor es su primera foto CEML.

Usuarios (solo para demo; la contraseña es `DEMO_PASSWORD`, por defecto `demo1234`):

- `admin`: administrador, ve todo y las estadísticas.
- `lector1`: lecturista de la Ruta 80.
- `lector2`: lecturista de la Ruta 90.
- `lector3`: lecturista de la Ruta 1 (energía).

## Agua y energía

Cada medidor tiene un servicio (`service`: `water` o `energy`) y sus fotos se leen con los modelos
de ese servicio, que se cargan recién con la primera foto. Los de energía se leen tal como se
fotografiaron, sin probar el visor girado 180°: están en pared o pilar y se fotografían derechos,
y los dígitos de 7 segmentos girados siguen pareciendo dígitos.

## Sin modelo de dígitos

Si no existe el detector de dígitos de un servicio (`models/digits_yolov8n.pt` para agua), el
backend carga solo el segmentador de ese servicio: `analyze` devuelve
el visor (`overlay`, `crop`) con `reading: null` y `digits_model_available: false`, y la lectura se
tipea a mano. Al copiar los pesos se usa el pipeline completo (reiniciar el servidor).

## Contrato

Ver "Contrato de la API" en `odd/tasks/app-lecturas.md`. Notas de implementación:

- `GET /api/auth/me` y `POST /api/auth/login` devuelven `{"user": {...}}`.
- `corrected` es `true` si el valor final difiere del propuesto o si no hubo propuesta.
- Un `analysis_id` solo sirve para el medidor y el usuario que lo generó (404 si no); reutilizarlo da 409.
- `lat`, `lon` y `accuracy_m` guardan dónde se tomó la foto; son `null` si el navegador no informó la ubicación.
- Los análisis que no llegan a confirmarse dejan sus imágenes en el almacenamiento (`analyses/<id>/photo.jpg`, `overlay.jpg`, `crop.jpg`, `digits.jpg`).

## Tests

```bash
.venv/bin/python -m pytest -q app/backend/tests
```

## Docker

El contexto de build es la raíz del repo y el Dockerfile está en `app/backend/Dockerfile`:

```yaml
api:
  build:
    context: .
    dockerfile: app/backend/Dockerfile
```

La imagen instala torch solo CPU y copia `src/meter_reader`, `models/*.pt` y `data/ceml`.
`app/backend/Dockerfile.dockerignore` limita el contexto a esos archivos (requiere BuildKit).
