# Lectura automática de medidores de agua y energía con visión por computadora

Trabajo práctico final de **Visión por Computadora II** — CEIA, FIUBA (2026).

## Objetivo

Leer automáticamente el estado de medidores de agua (rodillos) y de energía (rodillos y LCD de 7 segmentos) a partir de fotos tomadas en campo, con un pipeline basado en redes neuronales convolucionales. El pipeline es el mismo para los dos servicios; cambian los modelos entrenados y la orientación (ver *Medidores de energía*):

1. **Segmentación de instancia del visor**: YOLOv8n-seg vs Mask R-CNN.
2. **Rectificación**: se ajusta un rectángulo rotado mínimo a la máscara y se endereza el visor con una transformación afín (`src/meter_reader/rectify.py`).
3. **Detección de dígitos**: YOLOv8 con una clase por dígito (0–9); la lectura se obtiene ordenando las detecciones.
4. **m³ y fracción**: cada modelo de medidor tiene una cantidad fija de rodillos de m³ (lo que se factura) y de fracción. Una CNN (MobileNetV3, transfer learning) clasifica el formato a partir del visor con su entorno (logo, marco) y fija dónde va el punto (`src/meter_reader/layout.py`, `scripts/train_meter_model.py`); si duda, los dígitos rojos se separan de los negros por el exceso de rojo de cada caja (`src/meter_reader/color.py`). La lectura queda como `00053.25`.
5. **Baseline**: OCR genérico (EasyOCR) sobre el mismo recorte.

**Experimento principal:** evaluar si segmentar y rectificar el visor mejora la lectura frente a un recorte por bounding box. Métricas: exactitud de lectura completa y CER.

## Resultados

Lectura de punta a punta sobre 323 fotos de campo CEML verificadas a mano (`scripts/evaluate_pipeline.py`). Todas las configuraciones usan la misma segmentación del visor (YOLOv8n-seg).

| Configuración | Lectura exacta | m³ exactos | CER | Sin lectura |
|---|---|---|---|---|
| **A. Rectificado + YOLOv8n + clasificador de formato (final)** | **71,2 %** | **78,0 %** | **0,121** | **8,0 %** |
| B. Rectificado + YOLOv8n + solo color | 54,7 % | 58,2 % | 0,121 | 8,0 % |
| C. Recorte por caja + YOLOv8n + clasificador | 63,6 % | 71,8 % | 0,177 | 12,4 % |
| D. Rectificado + YOLO11n + clasificador | 61,4 % | 73,4 % | 0,151 | 8,0 % |
| E. Rectificado + EasyOCR + clasificador (baseline) | 0,3 % | 0,6 % | 0,899 | 16,4 % |

- **Rectificar sirve** (A contra C): +7,6 puntos de lectura exacta. La mejora sale de los visores inclinados: con más de 15°, el recorte por caja no lee ninguna foto y rectificando se lee el 62 % (en energía, 75 %).
- **El clasificador de formato** (MobileNetV3-small, 98,4 % de exactitud con validación cruzada) decide dónde va el punto: +16,5 puntos frente al color solo (A contra B).
- **Segmentación del visor**: YOLOv8n-seg supera a Mask R-CNN en las fotos de campo (máscara AP50-95 0,359 contra 0,262) y es 21 veces más rápido en CPU.
- El pipeline final tarda 74 ms por foto en CPU.

El resumen con gráficos y el pipeline paso a paso está en [notebooks/03_resultados.ipynb](notebooks/03_resultados.ipynb); el detalle de cada experimento, en [experiments/README.md](experiments/README.md); y el motivo de cada decisión, en [docs/decisiones.md](docs/decisiones.md).

### Medidores de energía

El mismo pipeline se extendió a los medidores de energía de la cooperativa (rodillos y LCD de 7 segmentos), con el segmentador, el clasificador de formato y el detector de dígitos reentrenados con fotos de energía. Evaluación sobre una muestra aparte de 201 fotos verificadas a mano, que no se usaron para entrenar (`experiments/pipeline_ceml_energia/`):

| Configuración (energía) | Lectura exacta | m³ exactos | CER | Sin lectura |
|---|---|---|---|---|
| **A. Rectificado + YOLOv8n + clasificador de formato** | **66,7 %** | **79,1 %** | **0,082** | **0 %** |
| B. Rectificado + YOLOv8n + solo color | 3,0 % | 5,5 % | 0,082 | 0 % |
| C. Recorte por caja + YOLOv8n + clasificador | 51,7 % | 62,2 % | 0,226 | 5,0 % |
| E. Rectificado + EasyOCR + clasificador (baseline) | 3,0 % | 8,5 % | 0,577 | 11,9 % |

- **Rectificar también sirve en energía** (A contra C): +15,0 puntos de lectura exacta.
- Sin fracción roja, el color no ubica el punto: el clasificador de formato es imprescindible (A contra B).
- Partiendo del pipeline de agua, la lectura exacta en energía pasó de 18,9 % a 66,7 %: segmentador con energía, leer sin giro de 180° (se fotografían derechos), clasificador reentrenado y, el salto mayor, el detector de dígitos con 483 recortes de energía etiquetados.
- La app elige los modelos según el servicio de cada medidor; agua conserva los de la tabla anterior.

## Estructura del repositorio

| Carpeta | Contenido |
|---|---|
| [src/meter_reader/](src/meter_reader/) | Pipeline de lectura: segmentación, rectificación, dígitos, formato y armado de la lectura |
| [scripts/](scripts/) | Descarga y preparación de datos, entrenamiento y evaluación (cada script explica su uso al inicio) |
| [notebooks/](notebooks/) | `01_eda` (análisis exploratorio), `02_entrenamiento` (Kaggle/Colab) y `03_resultados` |
| [models/](models/) | Pesos finales de cada experimento |
| [experiments/](experiments/) | Métricas, curvas e hiperparámetros de cada experimento |
| [data/](data/) | Fotos de campo CEML anonimizadas, de agua (`ceml/`) y de energía (`ceml_energia/`), con su verdad de campo; los datasets públicos se descargan en `data/raw/` |
| [tests/](tests/) | Tests con pytest |
| [app/](app/) | App web de toma de lecturas (ver abajo) |
| [docs/](docs/) | Decisiones del proyecto y su motivo |

## Datos

Fuentes, versiones, licencias y cómo se armó cada conjunto: [data/README.md](data/README.md). Resumen:

| Dataset | Servicio | Uso | Licencia |
|---|---|---|---|
| [Toloka Water Meters](https://www.kaggle.com/datasets/tapakah68/yandextoloka-water-meters-dataset) | agua | segmentación del visor | CC BY-NC-ND 4.0 |
| [water-meter-dataset v3](https://universe.roboflow.com/water-meter-yfzw4/water-meter-dataset) | agua | dígitos y visor derivado | CC BY 4.0 |
| [meter-digits-5srka v7](https://universe.roboflow.com/mohammad-annus/meter-digits-5srka) y [meter-digits-frvqn v2](https://universe.roboflow.com/meter-i5vud/meter-digits-frvqn) | energía | dígitos LCD | CC BY 4.0 |
| Fotos de campo CEML (`data/ceml/`, `data/ceml_energia/`) | agua y energía | prueba y clasificador de formato; en energía también segmentador y dígitos | uso académico |

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt   # entorno exacto verificado (torch CPU)
pytest                      # tests del pipeline, de los scripts y de la API
```

`requirements-lock.txt` fija todas las dependencias, también las indirectas, con las versiones con las que se verificó la reproducción (Python 3.12, Linux, torch para CPU). `requirements.txt` lista solo las dependencias directas de la investigación y `app/backend/requirements-dev.txt`, las de la API y sus tests. Para entrenar con GPU, instalar `torch` y `torchvision` desde el índice de PyTorch que corresponda (CUDA o ROCm) y después `pip install -r requirements.txt -r app/backend/requirements-dev.txt`.

## Reproducir los resultados

Los pesos ya están en `models/`, así que para ver los resultados alcanza con abrir `notebooks/03_resultados.ipynb`. Para rehacer todo desde cero:

1. **Datos**: `python scripts/download_data.py` y, para energía, `python scripts/download_data.py --only energia` (requieren `ROBOFLOW_API_KEY`; ver [data/README.md](data/README.md)). Las fotos CEML ya están en el repositorio.
2. **Preparación**: `python scripts/prepare_toloka.py`, `python scripts/prepare_seg_combined.py` y `python scripts/prepare_digits.py`. Para energía, `prepare_seg_combined.py --extra data/ceml_energia` y `prepare_digits.py` con los LCD públicos y `--crops data/ceml_energia` (los comandos exactos están en `notebooks/02_entrenamiento.ipynb`, sección 3).
3. **Segmentación del visor y dígitos**: `notebooks/02_entrenamiento.ipynb` en Kaggle o Colab (GPU), o `scripts/train_yolo.py` y `scripts/train_maskrcnn.py` en local. Se evalúan con `scripts/evaluate_seg.py` y `scripts/evaluate_seg_coco.py`.
4. **Clasificador de formato**: `python scripts/train_meter_model.py --target formato --name meter_layout_mnv3s` para agua y `--name meter_layout_agua_energia --extra-data data/ceml_energia` para agua y energía (validación cruzada de 5 folds y modelo final).
5. **Punta a punta**: `python scripts/evaluate_pipeline.py` (agua, guarda `experiments/pipeline_ceml/`) y, para energía, con `--no-flip` y los modelos de energía (comando completo en el docstring del script; guarda `experiments/pipeline_ceml_energia/`).

Los notebooks se pueden ejecutar desde la terminal con `jupyter nbconvert --to notebook --execute --inplace notebooks/03_resultados.ipynb`.

### Verificación de la reproducibilidad

Se verificó desde un clon limpio del repositorio, con un entorno nuevo instalado desde `requirements-lock.txt` y sin GPU:

| Qué | Resultado |
|---|---|
| `pytest` | 171 tests pasan |
| `notebooks/01_eda.ipynb` (con los datos descargados por `download_data.py`) | misma salida en todas las celdas |
| `notebooks/03_resultados.ipynb` | misma salida en todas las celdas, incluidas las validaciones que se calculan en vivo (1,5 min en CPU) |
| `evaluate_pipeline.py`, agua y energía | mismas predicciones foto por foto en las configuraciones A–D. Cambian la latencia (depende de la máquina) y algunas lecturas de la referencia EasyOCR (E), cuyos pesos se descargan aparte y no están versionados; su exactitud no cambia |
| `prepare_toloka.py`, `prepare_seg_combined.py`, `prepare_digits.py` | splits y etiquetas idénticos |
| `train_yolo.py` (1 época, prueba de humo) | entrena y guarda pesos y métricas. Un entrenamiento completo en GPU no es idéntico bit a bit entre corridas (operaciones no deterministas de CUDA/ROCm), por eso los resultados se reportan con los pesos versionados en `models/` |
| App web (`docker compose up -d --build` y `python scripts/smoke_app.py`) | login, análisis de la foto de ejemplo, lectura confirmada e imágenes guardadas, para agua y energía |

## App web de toma de lecturas

Prototipo para el uso real del sistema: el lecturista inicia sesión, recorre su ruta (listado y mapa), saca la foto del medidor, el pipeline propone la lectura y él la confirma o corrige; se guardan la foto, las imágenes procesadas y la ubicación GPS donde se tomó la foto. El administrador ve el historial y el porcentaje de lecturas aceptadas sin corrección.

```bash
docker compose up -d --build
```

Abrir http://localhost:8080. Usuarios de demo: `lector1` y `lector2` (agua), `lector3` (energía) y `admin`, con contraseña `demo1234` (solo para desarrollo local). Los medidores son los de las fotos CEML, con direcciones y coordenadas ficticias; el botón *Usar foto de ejemplo* analiza la foto real de cada medidor.

| Servicio | Qué es |
|---|---|
| `web` | Frontend React + Vite servido por nginx ([app/frontend](app/frontend/)) |
| `api` | FastAPI con el pipeline de visión ([app/backend](app/backend/)) |
| `migrate` | Aplica las migraciones Alembic de [app/db](app/db/) y termina |
| `db` | PostgreSQL |
| `storage` | SeaweedFS (API S3) para las fotos y las imágenes procesadas ([app/storage](app/storage/)) |

`docker compose down` conserva los datos; `docker compose down -v` borra la base y las imágenes.

### Probar desde el celular con ngrok

El celular solo permite usar la cámara y el GPS en páginas servidas por HTTPS. [ngrok](https://ngrok.com) publica la app local con una URL HTTPS; alcanza con un único túnel (plan gratuito) hacia el servicio `web`, porque su nginx reenvía `/api` a la API.

**1. Cuenta e instalación (una sola vez)**

1. Crear una cuenta gratuita en https://dashboard.ngrok.com/signup.
2. Instalar ngrok siguiendo https://ngrok.com/download (en Linux: `sudo snap install ngrok` o el paquete de la página).
3. Copiar el *authtoken* desde *Your Authtoken* en el panel y registrarlo:

   ```bash
   ngrok config add-authtoken <TU_AUTHTOKEN>
   ```

   El token queda en la configuración local de ngrok (`~/.config/ngrok/ngrok.yml`); no debe copiarse en el repositorio.
4. Opcional: en *Domains* del panel, reclamar el dominio estático gratuito (`<algo>.ngrok-free.app`), para que la URL no cambie entre sesiones.

**2. Levantar la app con una contraseña propia**

La URL de ngrok es pública: cualquiera que la tenga llega al login. Antes de exponerla, conviene reemplazar la contraseña de demo. Los usuarios se crean en el primer arranque, así que la contraseña se fija con la base vacía:

```bash
docker compose down -v      # solo si ya se había levantado con otra contraseña (borra lecturas e imágenes)
DEMO_PASSWORD='<una-contraseña>' COOKIE_SECURE=true docker compose up -d --build
```

`COOKIE_SECURE=true` marca la cookie de sesión como segura, porque el acceso es por HTTPS.

**3. Abrir el túnel**

```bash
ngrok http 8080                                   # URL nueva en cada sesión
ngrok http --url=<tu-dominio>.ngrok-free.app 8080 # con el dominio estático
```

ngrok muestra la URL pública en la línea *Forwarding* (`https://....ngrok-free.app`). El túnel queda abierto mientras la terminal siga corriendo; se cierra con `Ctrl+C`.

**4. Tomar una lectura real desde el celular**

1. Abrir la URL `https://...` en el navegador del celular. El plan gratuito muestra primero un aviso de ngrok: tocar **Visit Site**.
2. Ingresar con `lector1` (o `lector2`, o `lector3` para energía) y la contraseña elegida.
3. Entrar a la ruta y elegir un medidor (son medidores de prueba; se puede fotografiar cualquier medidor de agua de rodillos).
4. Tocar el campo **Foto del medidor**: el celular ofrece abrir la cámara. Encuadrar el visor con buena luz y sin reflejos, y sacar la foto.
5. La app muestra la segmentación del visor, el visor rectificado, los dígitos detectados y la lectura propuesta. Corregirla si hace falta y tocar **Confirmar lectura**.
6. Aceptar el permiso de **ubicación** cuando el navegador lo pida: la lectura se guarda con el punto GPS donde se tomó la foto.
7. Con `admin`, la pestaña **Mapa de lecturas** muestra dónde se tomó cada foto (satélite o calles) y **Historial de lecturas** lista todas las lecturas; cada una abre un detalle con la foto, la segmentación del visor, el visor rectificado, los dígitos y el mapa de la ubicación.

**Precisión del GPS**

Una página web no elige entre GPS y Wi-Fi: pide la máxima precisión y el sistema operativo combina las fuentes disponibles. La app empieza a seguir la ubicación apenas se abre la pantalla del medidor, conserva la posición más precisa y la muestra en vivo (*GPS: ±6 m*; verde hasta 15 m, ámbar hasta 50 m, rojo por encima). El GPS tarda unos segundos en "engancharse": si el indicador no está en verde, conviene esperar antes de confirmar. Para que el celular use el GPS:

- **Android**: *Ajustes → Ubicación* activada, con *Precisión de la ubicación de Google* encendida. En Chrome, el permiso del sitio debe ser *ubicación precisa* (no *aproximada*): candado de la barra de direcciones → *Permisos → Ubicación*.
- **iPhone**: *Ajustes → Privacidad y seguridad → Localización → Safari* (o el navegador que se use) → *Al usar la app* y **Ubicación exacta** activada.
- A cielo abierto el GPS es mucho más preciso; bajo techo el celular se apoya en Wi-Fi y antenas (decenas o cientos de metros).

**Si algo falla**

- No pide ubicación o no la guarda: verificar que la URL empiece con `https://`, que el GPS del celular esté activo y que el navegador tenga permiso de ubicación para ese sitio (se puede revisar desde el candado de la barra de direcciones).
- Error al analizar la foto: revisar `docker compose logs api`.
- La URL dejó de andar: el túnel se cerró o cambió (sin dominio estático cambia en cada `ngrok http`).

## Integrantes

- Lacheski, Martin
- Lastra, Nicolas
- Scheidl, Hernando
- Vazquez, Diego


## Licencia

El código se distribuye bajo la licencia [AGPL-3.0](LICENSE), la misma de Ultralytics (YOLO), de la que depende el pipeline. Los datasets conservan sus propias licencias (ver [data/README.md](data/README.md)); las fotos de campo CEML son de uso académico.
