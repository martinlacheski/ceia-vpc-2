# Experimentos

Cada experimento guarda en `experiments/<nombre>/` las métricas (`results.csv`), las curvas y los hiperparámetros (`args.yaml`) de su entrenamiento. Los pesos finales están en `models/<nombre>.pt`.

## Etapa 1: segmentación del visor (ablación)

Objetivo: reducir la caída de desempeño entre Toloka y las fotos de campo CEML. El EDA (`notebooks/01_eda.ipynb`) mostró que la geometría del visor es similar en ambos conjuntos; la diferencia está en la apariencia (barro, reflejos, poca luz).

Métricas de máscara; en CEML se evalúan las 324 fotos salvo en E (validación cruzada).

| # | Configuración | Experimento | Toloka test mAP50 / mAP50-95 | CEML mAP50 / mAP50-95 | CEML recall | Estado |
|---|---|---|---|---|---|---|
| A | YOLOv8n-seg, Toloka, 640 px | `toloka_yolov8n_seg` | 0,995 / 0,877 | 0,858 / 0,429 | 0,867 | Hecho |
| B | A + 8.217 imágenes de water-meter-dataset con visor derivado de los dígitos (rectángulo rotado mínimo + margen; solo imágenes con ≥ 5 dígitos); 40 épocas | `toloka_yfzw4_yolov8n_seg` | — | — | — | En curso |
| C | B + `imgsz=1024` | — | — | — | — | Pendiente |
| D | La mejor + augmentation de condiciones de campo (brillo, desenfoque, ruido) | — | — | — | — | Pendiente |
| E | La mejor + fine-tuning con CEML (validación cruzada de 5 particiones por `meter_id`) | — | — | — | — | Pendiente |
| F | Mask R-CNN (ResNet-50) con los datos de la mejor configuración | — | — | — | — | Pendiente |

Latencia de A en CPU (Ryzen AI, 20 hilos): 19 ms por imagen.

## Notas

- La métrica que más importa para el pipeline es el **recall** en CEML: un visor no detectado es una lectura perdida.
- Parte de la caída en mAP50-95 puede deberse a criterios de borde distintos entre los polígonos de Toloka (manuales) y los de CEML (SAM 3 con revisión manual); la caída en mAP50 es menos sensible a ese efecto.
