# Experimentos

Cada experimento guarda en `experiments/<nombre>/` las métricas (`results.csv`), las curvas y los hiperparámetros (`args.yaml`) de su entrenamiento. Los pesos finales están en `models/<nombre>.pt`.

## Etapa 1: segmentación del visor (ablación)

Objetivo: reducir la caída de desempeño entre Toloka y las fotos de campo CEML. El EDA (`notebooks/01_eda.ipynb`) mostró que la geometría del visor es similar en ambos conjuntos; la diferencia está en la apariencia (barro, reflejos, poca luz).

Métricas de máscara; en CEML se evalúan las 324 fotos salvo en E (validación cruzada).

| # | Configuración | Experimento | Toloka test mAP50 / mAP50-95 | CEML mAP50 / mAP50-95 | CEML recall | Estado |
|---|---|---|---|---|---|---|
| A | YOLOv8n-seg, Toloka, 640 px | `toloka_yolov8n_seg` | 0,995 / 0,877 | 0,858 / 0,429 | 0,867 | Hecho |
| B | A + 8.217 imágenes de water-meter-dataset con visor derivado de los dígitos (rectángulo rotado mínimo + margen; solo imágenes con ≥ 5 dígitos); 40 épocas | `toloka_yfzw4_yolov8n_seg` | 0,995 / 0,847 | 0,937 / 0,434 | 0,911 | Hecho |
| C | B + `imgsz=1024`; 40 épocas | `toloka_yfzw4_yolov8n_seg_1024` | 0,995 / 0,901 | 0,940 / 0,401 | 0,931 | Hecho |
| D | La mejor + augmentation de condiciones de campo (brillo, desenfoque, ruido) | — | — | — | — | Pendiente |
| E | La mejor + fine-tuning con CEML (validación cruzada de 5 particiones por `meter_id`) | — | — | — | — | Pendiente |
| F | Mask R-CNN (ResNet-50) con los datos de la mejor configuración | — | — | — | — | Pendiente |

Latencia en CPU (Ryzen AI, inferencia completa por imagen, media de 50 fotos CEML): A 24 ms, B 24 ms, C 51 ms. B y C se entrenaron en Kaggle (2× T4); las métricas se recalcularon localmente con `scripts/evaluate_seg.py` y coinciden con las de Kaggle.

## Etapa 2: detección de dígitos

Dataset: recortes del visor rectificado de water-meter-dataset (`scripts/prepare_digits.py`; 6.848 train, 719 val, 656 test), clases `0`–`9`. Sin espejado horizontal (`fliplr=0`): un dígito espejado cambia de clase o deja de ser un dígito.

| # | Configuración | Experimento | Test mAP50 / mAP50-95 | Precisión / recall | Estado |
|---|---|---|---|---|---|
| D1 | YOLOv8n, 416 px, hasta 100 épocas (paciencia 20) | `digits_yolov8n` | 0,983 / 0,687 | 0,962 / 0,961 | Hecho (Kaggle, 2× T4; paró en la época 91, mejor la 71) |

El mAP50-95 por dígito va de 0,672 (`1` y `4`) a 0,707 (`5`): no hay un dígito claramente más difícil.

Primera prueba del pipeline completo sobre 12 fotos CEML al azar (inspección visual, sin lecturas de referencia todavía): los visores nítidos se leen bien salvo los rodillos rojos a mitad de giro; un visor invertido se resolvió con la elección de orientación; las fotos borrosas o empañadas fallan, pero con confianza mínima baja (0,25–0,51 frente a 0,69–0,85), lo que permite marcarlas para revisión manual.

## Notas

- **B frente a A**: el visor derivado de water-meter-dataset sube el recall en CEML de 0,867 a 0,911 y el mAP50 de 0,858 a 0,937. En Toloka baja algo el mAP50-95 (0,877 → 0,847), esperable porque el visor derivado sigue el contorno de los dígitos y no el marco anotado en Toloka.
- **C frente a B**: 1024 px sube el recall en CEML a 0,931, pero duplica la latencia y baja el mAP50-95 en CEML (0,434 → 0,401).

- La métrica que más importa para el pipeline es el **recall** en CEML: un visor no detectado es una lectura perdida.
- Parte de la caída en mAP50-95 puede deberse a criterios de borde distintos entre los polígonos de Toloka (manuales) y los de CEML (SAM 3 con revisión manual); la caída en mAP50 es menos sensible a ese efecto.
