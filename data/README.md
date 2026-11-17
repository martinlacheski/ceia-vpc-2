# Datasets

Los datasets públicos no se versionan en el repositorio: se descargan en `data/raw/`. Las fotos de campo propias sí se versionan, en `data/ceml/`.

| Dataset | Uso | Versión | Contenido | Licencia | Carpeta |
|---|---|---|---|---|---|
| [Water Meters Dataset (Toloka)](https://www.kaggle.com/datasets/tapakah68/yandextoloka-water-meters-dataset) | Etapa 1: segmentación del visor | Kaggle | 1.244 imágenes; polígono del visor y lectura en `data.csv`, máscaras y collages | CC BY-NC-ND 4.0 | `data/raw/toloka/` |
| [water-meter-dataset (Roboflow)](https://universe.roboflow.com/water-meter-yfzw4/water-meter-dataset) | Etapa 2: detección de dígitos. Etapa 1: entrenamiento adicional de la segmentación con el visor derivado de los dígitos (ver abajo) | v3 (exportada el 17/01/2024) | 8.978 imágenes sin augmentation; solo clases `0`–`9` (cajas o cuadriláteros por dígito) | CC BY 4.0 | `data/raw/roboflow_water_meter/` |
| [Fotos de campo CEML (Misiones)](ceml/) | Conjunto de prueba en condiciones reales | — | 324 imágenes de 314 medidores; polígono del visor (auto-etiquetado con SAM 3 y revisión manual) | Uso académico | `data/ceml/` |

## Descarga

Automática: `python scripts/download_data.py` (Toloka sin credenciales; water-meter-dataset requiere la variable de entorno `ROBOFLOW_API_KEY`). En Kaggle, Toloka debe adjuntarse al notebook con *Add Input*.

Manual:

1. **Toloka**: en Kaggle, botón *Download*. Descomprimir el zip en `data/raw/toloka/` (deben quedar `images/`, `masks/`, `collage/` y el CSV).
2. **water-meter-dataset**: en Roboflow Universe, *Download Dataset* → versión 3 → formato **YOLOv8** → *Download zip to computer*. Descomprimir en `data/raw/roboflow_water_meter/` (deben quedar `train/`, `valid/`, `test/` y `data.yaml`).

## Datasets evaluados y descartados

- [meter-digits](https://universe.roboflow.com/meter-i5vud/meter-digits-frvqn) (v2): corresponde a medidores eléctricos y la mayoría de las imágenes muestra un único medidor.
- [water-meter-reading](https://universe.roboflow.com/new-workspace-j7ezo/water-meter-reading) (v1): 549 de sus 774 imágenes de origen (71%) ya están incluidas en water-meter-dataset; además trae augmentation aplicada (rotaciones ×3) e imágenes estiradas a 608×608.

El solapamiento entre Toloka y water-meter-dataset es mínimo (16 imágenes similares sobre 8.978, por hash perceptual).

## Visor derivado en water-meter-dataset (etapa 1)

La versión 3 de water-meter-dataset no anota el visor, pero sí cada dígito de la lectura. Para ampliar y diversificar el entrenamiento de la segmentación (experimento B en `experiments/README.md`), `scripts/prepare_seg_combined.py` deriva el visor como el rectángulo rotado mínimo que contiene todos los dígitos, con un margen de 4% en el lado largo y 15% en el corto. Reglas:

- solo imágenes con al menos 5 dígitos anotados (8.217 de 8.978; con menos, el rectángulo quedaría corto);
- se descartan 6 imágenes casi idénticas a alguna de validación o prueba de Toloka (hash perceptual);
- todas van a entrenamiento: validación y prueba siguen siendo las de Toloka, para comparar con el entrenamiento solo con Toloka.

Son **etiquetas derivadas**, no anotaciones originales del visor: siguen el contorno de los dígitos y no necesariamente el marco de la ventana.

## Dígitos sobre el visor rectificado (etapa 2)

`scripts/prepare_digits.py` arma el dataset de detección de dígitos con los mismos recortes que verá el pipeline. Para cada imagen de water-meter-dataset con al menos 5 dígitos, deriva el visor con el criterio anterior, lo endereza y recorta con `src/meter_reader/rectify.py` y transforma cada dígito a una caja YOLO dentro del recorte. Se conservan los splits originales: 6.848 recortes en train, 719 en val y 656 en test. Salida: `data/processed/digits_yfzw4/`.

## Fotos de campo CEML

Las fotos están en `data/ceml/images/`, anonimizadas con `scripts/anonymize_field.py`: nombres neutros (`agua_<AAAA-MM>_r<ruta>_<medidor>.jpg`), sin EXIF y con el lado mayor limitado a 1600 px. `data/ceml/metadata.csv` contiene mes, ruta e identificador de medidor de cada foto. Los originales y la correspondencia cliente → medidor no se publican.

`data/ceml/readings.csv` es la verdad de campo para medir la lectura completa: `reading` es el valor que se ve en la foto, con todos los dígitos y un punto entre los negros (m³) y los rojos (fracción), por ejemplo `00053.25`, `legible` indica si se puede leer, `notes` deja observaciones y `modelo` identifica el modelo de medidor, que fija cuántos dígitos son m³ y cuántos fracción (el nombre termina en `_<dígitos de m³>_<dígitos de fracción>`, por ejemplo `generico_5_2`, `abb_4_3` o `qn15_5_0`; vacío si no se reconoce, y esa foto no se usa para entrenar el clasificador de modelo). Se completa a mano mirando cada foto. `scripts/propose_readings.py` agrega las columnas `propuesta` y `confianza` con lo que lee el pipeline, para acelerar la carga sin tocar las columnas manuales. La propuesta puede sesgar al que verifica: `reading` tiene que salir de la foto, no de copiar la propuesta.

Cómo se armó (noviembre de 2026):

1. El pipeline propuso una lectura para cada foto y una persona la verificó mirando la foto.
2. En una primera pasada, 262 filas quedaron iguales a la propuesta, sin punto y sin `legible`: se habían copiado como punto de partida. Medir contra ellas habría comparado el modelo consigo mismo, así que se rehicieron mirando cada foto.
3. Al agregar la columna `modelo` se revisaron las fotos agrupadas por modelo (hojas de contacto) y las que el clasificador predecía distinto; así aparecieron lecturas con el punto corrido, comas en lugar de puntos y un nombre de archivo pisado al editar la planilla.

Convenciones:

- Un punto al final (`2178.`) indica que el medidor tiene fracción pero sus rodillos no se ven o no tienen números; los m³ sí se leen.
- Los medidores sin rodillos de fracción (5+0, 4+0) van sin punto.
- `modelo` vacío: no se distingue la marca ni el formato (foto muy movida o sucia).

Modelos más frecuentes y cómo se reconocen:

| `modelo` | Fotos | Rasgos |
|---|---|---|
| `generico_5_2` | 141 | Cuerpo blanco, 5 rodillos negros y 2 rojos, "m³" a la derecha |
| `actaris_4_2` | 80 | Actaris/Unimag ("Indústria Brasileira"), marco redondeado |
| `azul_4_4` | 31 | Cuerpo azul con ventana ovalada; la fracción es blanca sobre fondo rojo |
| `qn15_5_0` | 23 | "Qn1.5 H.B", sin rodillos de fracción |
| `iso4064_4_0` | 17 | "ISO 4064 Class B", marco escalonado, sin fracción |
| `elster_4_3` | 10 | Logo ELSTER, 3 rodillos rojos |
| `abb_4_1`, `abb_4_3` | 5 y 6 | "Asea Brown Boveri"; en un modelo 4+3 la fracción es blanca |

El resto (Cyble, SIAP, Aquarius, Elster S120-V y otros) tiene 1 o 2 fotos cada uno.

## Fotos de campo CEML de energía

`data/ceml_energia/` tiene 788 fotos de medidores de energía (rodillos y LCD), anonimizadas con el mismo criterio que las de agua. Es una extensión del trabajo: el eje sigue siendo agua.

- `readings.csv`: `muestra=si` marca las 201 fotos de la **muestra de prueba** (100 LCD y 101 de rodillos), con la lectura verificada a mano. Se usan solo para evaluar de punta a punta (`scripts/evaluate_pipeline.py --no-flip`) y nunca para entrenar. `tipo` es `lcd` o `rodillos`; `modelo` termina en `_<dígitos enteros>_<dígitos de fracción>` (por ejemplo `elster_a100_6_1`).
- `labels/` y `annotations.coco.json`: polígono del visor. Auto-etiquetado con SAM 3 en Roboflow, uniendo los polígonos de cada foto en su envolvente convexa (`scripts/import_ceml_labels.py --merge`) y revisado a mano.
- `digits/`: visor enderezado de las fotos fuera de la muestra, con una caja por dígito de la lectura. Cómo se armó:
  1. `scripts/export_digit_prelabels.py` recortó el visor con el polígono revisado y propuso las cajas con el detector `digits_mix_yolov8n`.
  2. Se corrigieron en Roboflow (proyecto privado) y se importaron con `scripts/import_digit_labels.py`.
  3. Se descartaron 83 recortes cuya cantidad de cajas no coincide con el formato del modelo de medidor (fotos ilegibles, fracción sin caja o indicadores marcados como dígito).
  4. Los 499 restantes se revisaron con un modelo de lenguaje visual sobre hojas de contacto: 190 correctos, 28 con algún número corregido y 281 marcados `ilegible` (algún dígito borroso o a medio girar). Las 10 correcciones controladas a mano estaban bien; el criterio de `ilegible` resultó más estricto que el de una persona.
- `digits/revision.csv`: resultado de esa revisión por recorte (`estado`: `ok`, `corregida` o `ilegible`; `etiqueta_roboflow` y `lectura` de izquierda a derecha).

`scripts/prepare_digits.py --crops data/ceml_energia` suma estos recortes al dataset de dígitos sin las fotos de los medidores de la muestra, dividido por medidor (`metadata.csv`); con `--skip-unreadable` deja afuera los marcados `ilegible`.
