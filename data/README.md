# Datasets

Los datasets públicos no se versionan en el repositorio: se descargan en `data/raw/`. Las fotos de campo propias sí se versionan, en `data/ceml/`.

| Dataset | Uso | Anotaciones | Licencia |
|---|---|---|---|
| [Water Meters Dataset (Toloka)](https://www.kaggle.com/datasets/tapakah68/yandextoloka-water-meters-dataset) | Entrenamiento de segmentación | 1.244 imágenes, máscara del visor y lectura en el nombre de archivo | CC BY-NC-ND 4.0 |
| [meter-digits (Roboflow)](https://universe.roboflow.com/meter-i5vud/meter-digits-frvqn) | Entrenamiento de detección de dígitos | En evaluación | En evaluación |
| [water-meter-dataset (Roboflow)](https://universe.roboflow.com/water-meter-yfzw4/water-meter-dataset) | Entrenamiento de detección de dígitos | En evaluación | En evaluación |
| [Fotos de campo CEML (Misiones)](ceml/) | Conjunto de prueba en condiciones reales | 330 imágenes de 319 medidores; etiquetado propio (150–200 imágenes) | Uso académico |

Las fotos de campo están en `data/ceml/images/`, anonimizadas con `scripts/anonymize_field.py`: nombres neutros (`agua_<AAAA-MM>_r<ruta>_<medidor>.jpg`), sin EXIF y con el lado mayor limitado a 1600 px. `data/ceml/metadata.csv` contiene mes, ruta e identificador de medidor de cada foto. Los originales y la correspondencia cliente → medidor no se publican.
