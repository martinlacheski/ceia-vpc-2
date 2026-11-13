# Datasets

Los datasets públicos no se versionan en el repositorio: se descargan en `data/raw/`. Las fotos de campo propias sí se versionan, en `data/ceml/`.

| Dataset | Uso | Versión | Contenido | Licencia | Carpeta |
|---|---|---|---|---|---|
| [Water Meters Dataset (Toloka)](https://www.kaggle.com/datasets/tapakah68/yandextoloka-water-meters-dataset) | Etapa 1: segmentación del visor | Kaggle | 1.244 imágenes; polígono del visor y lectura en `data.csv`, máscaras y collages | CC BY-NC-ND 4.0 | `data/raw/toloka/` |
| [water-meter-dataset (Roboflow)](https://universe.roboflow.com/water-meter-yfzw4/water-meter-dataset) | Etapa 2: detección de dígitos | v3 (exportada el 17/01/2024) | 8.978 imágenes sin augmentation; clases `0`–`9` y `border_water_meter_number` | CC BY 4.0 | `data/raw/roboflow_water_meter/` |
| [Fotos de campo CEML (Misiones)](ceml/) | Conjunto de prueba en condiciones reales | — | 330 imágenes de 319 medidores; etiquetado propio (150–200 imágenes) | Uso académico | `data/ceml/` |

## Descarga

1. **Toloka**: en Kaggle, botón *Download*. Descomprimir el zip en `data/raw/toloka/` (deben quedar `images/`, `masks/`, `collage/` y el CSV).
2. **water-meter-dataset**: en Roboflow Universe, *Download Dataset* → versión 3 → formato **YOLOv8** → *Download zip to computer*. Descomprimir en `data/raw/roboflow_water_meter/` (deben quedar `train/`, `valid/`, `test/` y `data.yaml`).

## Datasets evaluados y descartados

- [meter-digits](https://universe.roboflow.com/meter-i5vud/meter-digits-frvqn) (v2): corresponde a medidores eléctricos y la mayoría de las imágenes muestra un único medidor.
- [water-meter-reading](https://universe.roboflow.com/new-workspace-j7ezo/water-meter-reading) (v1): 549 de sus 774 imágenes de origen (71%) ya están incluidas en water-meter-dataset; además trae augmentation aplicada (rotaciones ×3) e imágenes estiradas a 608×608.

El solapamiento entre Toloka y water-meter-dataset es mínimo (16 imágenes similares sobre 8.978, por hash perceptual).

## Fotos de campo CEML

Las fotos están en `data/ceml/images/`, anonimizadas con `scripts/anonymize_field.py`: nombres neutros (`agua_<AAAA-MM>_r<ruta>_<medidor>.jpg`), sin EXIF y con el lado mayor limitado a 1600 px. `data/ceml/metadata.csv` contiene mes, ruta e identificador de medidor de cada foto. Los originales y la correspondencia cliente → medidor no se publican.
