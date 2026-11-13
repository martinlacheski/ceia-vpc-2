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
