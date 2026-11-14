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
| F | Mask R-CNN (ResNet-50 FPN, COCO) con los datos de B, 640 px, 10 épocas | `toloka_yfzw4_maskrcnn` | 0,9997 / 0,843 * | 0,864 / 0,262 * | — | Hecho (Kaggle, `scripts/train_maskrcnn.py`; mejor época por mask AP50-95 en validación: la 7) |

Latencia en CPU (Ryzen AI, inferencia completa por imagen, media de 50 fotos CEML): A 24 ms, B 24 ms, C 51 ms. B y C se entrenaron en Kaggle (2× T4); las métricas se recalcularon localmente con `scripts/evaluate_seg.py` y coinciden con las de Kaggle.

**Comparación entre arquitecturas (B contra F).** Ultralytics y torchvision no calculan el mAP exactamente igual, así que las dos se evalúan con `scripts/evaluate_seg_coco.py` (pycocotools, máscaras a resolución completa) y se guarda `eval_coco.json`. Con ese criterio B da en Toloka test 0,9999 / 0,847 de AP50 / AP50-95 de máscara (igual que Ultralytics) y en CEML 0,908 / 0,359 (más bajo que 0,937 / 0,434 porque pycocotools compara máscaras a resolución completa: es más exigente con los bordes). La latencia de las dos se mide en la misma máquina de Kaggle. (*) F solo tiene métricas COCO.

| Métrica (pycocotools) | B: YOLOv8n-seg | F: Mask R-CNN R50-FPN |
|---|---|---|
| Toloka test, máscara AP50 / AP50-95 | 0,9999 / 0,847 | 0,9997 / 0,843 |
| Toloka test, caja AP50 / AP50-95 | 0,9999 / 0,892 | 0,9997 / 0,817 |
| CEML, máscara AP50 / AP50-95 / AR@100 | **0,908 / 0,359 / 0,464** | 0,864 / 0,262 / 0,402 |
| CEML, caja AP50 / AP50-95 | 0,948 / **0,443** | **0,960** / 0,350 |
| Latencia CPU (Kaggle, media de 20 fotos CEML) | **85 ms** | 1.771 ms |
| Tamaño de los pesos | **6,7 MB** | 168 MB |

En Toloka empatan. En CEML, Mask R-CNN ubica el visor igual o algo mejor a IoU 0,5 (caja AP50 0,960), pero sus máscaras son menos precisas (máscara AP50-95 0,262 contra 0,359): la cabeza de máscara predice 28 × 28 por objeto y al escalarla a un visor alargado los bordes quedan gruesos. Además es 21 veces más lenta en CPU y 25 veces más pesada. Para la rectificación importa el contorno del visor, así que se mantiene YOLOv8n-seg. Diferencias de entrenamiento a tener en cuenta: F se entrenó 10 épocas con solo espejado horizontal; B, 40 épocas con las aumentaciones de Ultralytics (mosaico, color, escala).

## Etapa 2: detección de dígitos

Dataset: recortes del visor rectificado de water-meter-dataset (`scripts/prepare_digits.py`; 6.848 train, 719 val, 656 test), clases `0`–`9`. Sin espejado horizontal (`fliplr=0`): un dígito espejado cambia de clase o deja de ser un dígito.

| # | Configuración | Experimento | Test mAP50 / mAP50-95 | Precisión / recall | Estado |
|---|---|---|---|---|---|
| D1 | YOLOv8n, 416 px, hasta 100 épocas (paciencia 20) | `digits_yolov8n` | 0,983 / 0,687 | 0,962 / 0,961 | Hecho (Kaggle, 2× T4; paró en la época 91, mejor la 71) |
| D2 | YOLO11n, mismos hiperparámetros que D1 | `digits_yolo11n` | 0,984 / 0,694 | 0,968 / 0,955 | Hecho (Kaggle, 2× T4) |

El mAP50-95 por dígito va de 0,672 (`1` y `4`) a 0,707 (`5`): no hay un dígito claramente más difícil.

**D2 frente a D1**: YOLO11n empata con YOLOv8n (mAP50-95 0,694 contra 0,687; más precisión y algo menos de recall) con pesos un 16 % más chicos (5,2 MB contra 6,2 MB). La diferencia está dentro de lo que cambia entre corridas; se elige entre las dos por la exactitud de lectura de punta a punta en CEML.

Primera prueba del pipeline completo sobre 12 fotos CEML al azar (inspección visual, sin lecturas de referencia todavía): los visores nítidos se leen bien salvo los rodillos rojos a mitad de giro; un visor invertido se resolvió con la elección de orientación; las fotos borrosas o empañadas fallan, pero con confianza mínima baja (0,25–0,51 frente a 0,69–0,85), lo que permite marcarlas para revisión manual.

## Notas

- **B frente a A**: el visor derivado de water-meter-dataset sube el recall en CEML de 0,867 a 0,911 y el mAP50 de 0,858 a 0,937. En Toloka baja algo el mAP50-95 (0,877 → 0,847), esperable porque el visor derivado sigue el contorno de los dígitos y no el marco anotado en Toloka.
- **C frente a B**: 1024 px sube el recall en CEML a 0,931, pero duplica la latencia y baja el mAP50-95 en CEML (0,434 → 0,401).

- La métrica que más importa para el pipeline es el **recall** en CEML: un visor no detectado es una lectura perdida.
- Parte de la caída en mAP50-95 puede deberse a criterios de borde distintos entre los polígonos de Toloka (manuales) y los de CEML (SAM 3 con revisión manual); la caída en mAP50 es menos sensible a ese efecto.
