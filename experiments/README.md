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
| D3 | YOLOv8s, 416 px | `digits_yolov8s` | 0,981 / 0,681 | 0,962 / 0,957 | Hecho (Kaggle, T4; paró en la época 66, mejor la 46) |
| D4 | YOLOv8m, 416 px | `digits_yolov8m` | 0,985 / 0,684 | 0,964 / 0,962 | Hecho (Kaggle, T4; paró en la época 51, mejor la 31) |
| D5 | YOLOv8s, 640 px | `digits_yolov8s_640` | 0,982 / 0,680 | 0,960 / 0,964 | Hecho (Kaggle, T4; paró en la época 52, mejor la 32) |
| D6 | D1 + más aumentaciones de color y geometría (`hsv_h` 0,02, `hsv_s` 0,7, `hsv_v` 0,5, `degrees` 5, `perspective` 0,0005) | `digits_yolov8n_aug` | 0,985 / 0,682 | 0,961 / 0,962 | Hecho (Kaggle, T4; paró en la época 96, mejor la 76) |

El mAP50-95 por dígito va de 0,672 (`1` y `4`) a 0,707 (`5`): no hay un dígito claramente más difícil.

**D2 frente a D1**: YOLO11n empata con YOLOv8n (mAP50-95 0,694 contra 0,687; más precisión y algo menos de recall) con pesos un 15 % más chicos (5,5 MB contra 6,5 MB). La diferencia está dentro de lo que cambia entre corridas; se elige entre las dos por la exactitud de lectura de punta a punta en CEML.

**Ronda de modelos más grandes y más aumentaciones (D3–D6).** Se probó si más capacidad, más resolución o más aumentaciones mejoran el detector. Se compara en validación (la selección no usa CEML ni el test):

| Modelo | Val mAP50-95 | Test mAP50-95 | s/época | Latencia CPU | Pesos |
|---|---|---|---|---|---|
| D1 YOLOv8n, 416 | **0,709** | **0,687** | 28 | — | 6,5 MB |
| D6 YOLOv8n + aug, 416 | 0,705 | 0,682 | 31 | 40 ms | 6,5 MB |
| D3 YOLOv8s, 416 | 0,706 | 0,681 | 50 | 96 ms | 22,5 MB |
| D4 YOLOv8m, 416 | 0,706 | 0,684 | 109 | 220 ms | 52 MB |
| D5 YOLOv8s, 640 | 0,707 | 0,680 | 103 | 235 ms | 22,5 MB |

Ninguna variante supera a D1 por más de 0,01 de mAP50-95 en validación (la regla fijada de antemano para pasar a evaluar en CEML): todas quedan entre 0,705 y 0,709, dentro del ruido entre corridas. Hay un techo común, probablemente por el ruido de las etiquetas más que por la capacidad. Como más tamaño y más resolución solo agregan latencia (hasta 5–6 veces) y peso, se mantiene YOLOv8n. Las latencias de D3–D6 se midieron en la CPU de Kaggle; la de D1 no se midió allí, por lo que no es comparable con las de las otras filas. No se probó RT-DETR.

Primera prueba del pipeline completo sobre 12 fotos CEML al azar (inspección visual, sin lecturas de referencia todavía): los visores nítidos se leen bien salvo los rodillos rojos a mitad de giro; un visor invertido se resolvió con la elección de orientación; las fotos borrosas o empañadas fallan, pero con confianza mínima baja (0,25–0,51 frente a 0,69–0,85), lo que permite marcarlas para revisión manual.

## Etapa 3: formato del medidor (m³ y fracción)

Cada modelo de medidor tiene una cantidad fija de rodillos de m³ (lo que se factura) y de fracción. En las 324 fotos CEML aparecen al menos 15 modelos, con 7 formatos: 5+2 (genérico), 4+2 (Actaris/Unimag), 4+4 (cuerpo azul, fracción blanca sobre fondo rojo), 4+3 (Elster, ABB; en un ABB la fracción es blanca), 4+1 (ABB, SIAP), 5+0 (Qn1.5 H.B, sin fracción) y 4+0 (ISO 4064 Class B, sin fracción), más dos 5+3 aislados. El modelo de cada foto está en la columna `modelo` de `data/ceml/readings.csv`.

**Solo color (`src/meter_reader/color.py`)**: separa los dígitos rojos por el exceso de rojo de cada caja. Acierta si hay fracción y cuántos dígitos son m³ en 206 de 291 fotos (70,8 %). Falla sobre todo por omisión: con humedad, reflejos o poca luz el rojo no se distingue, y no puede separar una fracción blanca.

**Clasificador de formato (`scripts/train_meter_model.py`)**: MobileNetV3-small preentrenada en ImageNet con fine-tuning. La entrada es el visor enderezado con su entorno, en 384 × 192 px (`layout.model_input`), para que se vean el logo, el marco y el "m³". Aumentaciones: giro de 180°, rotación de ±6°, escala, brillo, contraste y desenfoque; en la predicción se promedia la imagen y su giro de 180°. 30 épocas, AdamW con OneCycle (lr 1e-3), pérdida ponderada por clase con suavizado de etiquetas 0,1. Evaluación con validación cruzada estratificada de 5 folds: cada foto se predice con un modelo que no la vio (`oof.csv`). Se excluyen las clases con menos de 5 fotos.

| Experimento | Clases | Fotos | Exactitud | F1 macro | Formato correcto |
|---|---|---|---|---|---|
| Solo color (propuesta del pipeline) | — | 291 | — | — | 70,8 % |
| `meter_layout_mnv3s` (`--target formato`) | 7 formatos | 320 | **98,4 %** | **0,954** | **98,4 %** |
| `meter_model_mnv3s` (`--target modelo`) | 8 modelos | 313 | 97,8 % | 0,891 | 99,0 % |

La variante por modelo deja afuera 9 fotos de modelos raros (Cyble, SIAP, Aquarius, …), por eso no es directamente comparable; la app usa la variante por formato. Entrenamiento local (Radeon 880M, ROCm): unos 6 minutos los 5 folds más el modelo final.

En la primera corrida (con la planilla sin revisar) la variante por formato daba 95,7 %. Los errores del clasificador marcaron fotos con el modelo mal cargado (4+3 anotados como 4+0, marcas distintas a las anotadas): se revisaron una por una mirando la foto y se corrigieron en la planilla (modelo y, en algunos casos, la lectura).

En el pipeline (`read_meter`), si el clasificador predice con probabilidad ≥ 0,5, su formato decide dónde va el punto (y lo saca en los medidores sin fracción); si duda, decide el color. Limitación: solo conoce los formatos que vio; un modelo nuevo con otra disposición se resuelve por el color.

## Etapa 4: lectura de punta a punta en CEML

`scripts/evaluate_pipeline.py` sobre las 323 fotos verificadas y legibles de `data/ceml/readings.csv` (316 con la fracción legible). Todas las configuraciones usan la misma segmentación del visor (B de la etapa 1). El clasificador de formato aporta su predicción out-of-fold (`experiments/meter_layout_mnv3s/oof.csv`): ninguna foto se lee con un clasificador que la haya visto. Resultados en `experiments/pipeline_ceml/` (`results.json` y `predictions.csv`, una columna por configuración).

- **Exacta**: la lectura completa coincide, con el punto y los ceros (sobre las 316 con fracción legible).
- **m³ exactos**: coincide la parte entera, lo que se factura (sobre las 323).
- **CER**: errores de edición entre dígitos / dígitos reales.
- **Sin lectura**: no se encontró el visor o ningún dígito.
- **Sin revisión**: lecturas con confianza ≥ 0,5, que la app acepta sin marcarlas para revisar; entre paréntesis, cuántas de ellas son exactas.

| # | Configuración | Exacta | m³ exactos | CER | Sin lectura | Sin revisión (exactas) |
|---|---|---|---|---|---|---|
| A | Rectificado + YOLOv8n + clasificador (pipeline final) | **71,2 %** | **78,0 %** | **0,121** | **8,0 %** | 80,7 % (83,1 %) |
| B | Rectificado + YOLOv8n + solo color | 54,8 % | 58,2 % | 0,121 | 8,0 % | 80,7 % (63,1 %) |
| C | Recorte por caja + YOLOv8n + clasificador | 63,6 % | 71,8 % | 0,177 | 12,4 % | 73,7 % (82,8 %) |
| D | Rectificado + YOLO11n + clasificador | 61,4 % | 73,4 % | 0,151 | 8,0 % | 81,0 % (74,2 %) |
| E | Rectificado + EasyOCR + clasificador | 0,3 % | 0,6 % | 0,899 | 16,4 % | 7,6 % (4,2 %) |

Latencia del pipeline final en CPU (Ryzen AI, media de 20 fotos, segmentación + dígitos en las dos orientaciones + clasificador): 74 ms.

- **Rectificar sirve (A frente a C)**: +7,6 puntos de lectura exacta y +6,2 de m³; el CER baja de 0,177 a 0,121 y las fotos sin lectura de 12,4 % a 8,0 %. Con el recorte por caja, un visor inclinado o vertical llega torcido al detector de dígitos, que se entrenó con visores horizontales.
- **El clasificador de formato (A frente a B)**: +16,5 puntos de lectura exacta y +19,8 de m³ con los mismos dígitos (el CER es idéntico): todo lo que cambia es dónde va el punto. Sin él, un punto mal puesto cambia los m³ facturados.
- **YOLOv8n sobre YOLO11n (A frente a D)**: empataban en el test de yfzw4 (mAP50-95 0,687 contra 0,694), pero en CEML YOLOv8n lee mejor (71,2 % contra 61,4 %). Se queda YOLOv8n: la métrica de detección en el dominio de entrenamiento no anticipó la diferencia en las fotos de campo.
- **EasyOCR no sirve como lector de rodillos (E)**: lee las barras entre rodillos y los números a medio girar (para `00053.25` devuelve `64055837215`). Se probó agrandar el recorte 3 veces y pasarlo a gris en una muestra de 12 fotos, sin mejora. Muestra por qué hace falta un detector entrenado para el dominio.
- **Confianza como filtro**: el 80,7 % de las lecturas pasa sin revisión y de esas el 83,1 % es exacta; el resto va a revisión manual en la app.

## Etapa 5: extensión a medidores de energía

Muestra de prueba: 201 fotos de energía verificadas a mano (`muestra=si` en `data/ceml_energia/readings.csv`; 101 de rodillos y 100 LCD). Para entrenar se excluyen todas las fotos de los medidores de la muestra.

**Segmentador** `toloka_yfzw4_energia_yolov8n_seg`: B de la etapa 1 con 587 fotos de energía de más en train (visores etiquetados en Roboflow con SAM 3 y revisados), mismos hiperparámetros.

| Métrica | B (agua) | Con energía |
|---|---|---|
| Toloka test, máscara mAP50-95 | 0,847 | 0,851 |
| CEML agua, recall / mAP50-95 | 0,911 / 0,434 | 0,921 / 0,404 |
| Muestra de energía, máscara mAP50 | 0,776 | 0,995 |
| Muestra de energía, Dice por foto / sin visor | 0,621 / 23 % | 0,951 / 0 % |

**Clasificador de formato** `meter_layout_agua_energia`: MobileNetV3-small con agua y energía juntos (1107 fotos, 10 formatos), validación cruzada de 5 folds: exactitud 99,37 %, F1 macro 0,967.

**Detector de dígitos**: pre-etiquetas del detector de agua sobre los visores rectificados, corregidas en Roboflow y revisadas contra el modelo. Split train/val por medidor.

| Experimento | Recortes de energía | Test público mAP50-95 | Muestra: exacta | m³ | CER |
|---|---|---|---|---|---|
| `digits_mix_yolov8n` (sin energía) | 0 | 0,7116 | 32,3 % | 46,3 % | 0,175 |
| **`digits_mix_energia_yolov8n` (E3)** | **483** | **0,7126** | **66,7 %** | **79,1 %** | **0,082** |
| `digits_mix_energia_rev_yolov8n` (E4, solo legibles) | 209 | 0,7134 | 65,7 % | 76,1 % | 0,082 |

Se elige E3. Por tipo: rodillos 58/101 exactas, LCD 76/100 (E4: 62/101 y 70/100).

**Lectura de punta a punta** (`scripts/evaluate_pipeline.py --no-flip`, ver el docstring; resultados en `experiments/pipeline_ceml_energia/`):

| # | Configuración | Exacta | m³ exactos | CER | Sin lectura | Sin revisión (exactas) |
|---|---|---|---|---|---|---|
| A | Rectificado + YOLOv8n (E3) + clasificador | **66,7 %** | **79,1 %** | **0,082** | **0 %** | 88,1 % (74,0 %) |
| B | Rectificado + YOLOv8n (E3) + solo color | 3,0 % | 5,5 % | 0,082 | 0 % | 88,1 % (3,4 %) |
| C | Recorte por caja + YOLOv8n (E3) + clasificador | 51,7 % | 62,2 % | 0,226 | 5,0 % | 66,7 % (70,9 %) |
| D | Rectificado + YOLO11n (agua) + clasificador | 21,9 % | 34,8 % | 0,446 | 3,5 % | 36,8 % (52,7 %) |
| E | Rectificado + EasyOCR + clasificador | 3,0 % | 8,5 % | 0,577 | 11,9 % | 30,3 % (9,8 %) |

D usa el YOLO11n de agua, que no vio energía: no es comparable con A. Latencia en CPU: 61 ms (una sola orientación).

Camino desde el pipeline de agua (lectura exacta en la muestra): 18,9 % → 16,9 % con el segmentador nuevo (pero sin lectura de 22,9 % a 0 %) → 27,4 % sin giro de 180° → 32,3 % con el clasificador reentrenado → 66,7 % con los dígitos de energía.

- **Orientación**: con giro, 78 de 162 errores eran lecturas al revés. En energía no hay fracción roja que decida y los 7 segmentos girados siguen pareciendo dígitos; leer siempre sin girar iguala al oráculo. En agua sí hace falta (medidores en el piso).
- **Agua con E3**: 65,2 % contra 71,2 % con `digits_yolov8n`, así que el detector es por servicio.

## Etapa 6: separador decimal por los huecos entre dígitos

El clasificador de formato (etapa 3) reconoce el modelo de medidor y deduce dónde va el punto, así que solo sirve para los modelos con los que se entrenó. Para no depender del modelo, `scripts/train_gap_model.py` entrena una CNN binaria (MobileNetV3-small) sobre cada hueco entre dos dígitos vecinos del visor rectificado: el recorte va del centro de un dígito al del siguiente y la pregunta es si ahí hay un separador decimal (el punto de un LCD, el recuadro o el color del último rodillo). Las etiquetas salen del formato ya anotado (`modelo` en `readings.csv`) y de las cajas de los dígitos de `data/ceml_energia/digits/`; no hace falta marcar el punto a mano. Quedan afuera los medidores de la muestra de prueba.

**Generalización a modelos no vistos**: validación dejando afuera un modelo de medidor entero por vez (12 modelos, 483 fotos). Acierto = posición del decimal correcta (o ninguno, si no hay).

| Modelo dejado afuera (fotos) | Solo CEML (`gap_energia_mnv3s`) | + LCD públicos con el punto (`gap_energia_lcd_mnv3s`) |
|---|---|---|
| elster_a100_6_1, LCD (139) | 29,5 % | 74,8 % |
| abb_5_1, rodillos (101) | 80,2 % | 85,1 % |
| generico_5_1, rodillos (107) | 88,8 % | 93,5 % |
| galileo_5_1, rodillos (46) | 91,3 % | 93,5 % |
| ampy_mono_6_1, LCD (23) | 60,9 % | 87,0 % |
| elster_m0507_5_1, rodillos (14) | 57,1 % | 35,7 % |
| **Total** | **64,8 %** | **81,4 %** |

Los LCD públicos son 1713 fotos de `meter-digits-5srka` con un único punto anotado entre dos dígitos (`--dot-data`); solo entrenan. Sin ellos, al dejar afuera el Elster A100 casi no quedan LCD en el entrenamiento y el modelo no aprende el punto: la mejora viene de la diversidad de los datos, no de la arquitectura. Por tecnología: rodillos 84,9 %, LCD 74,5 %. Casi todo el error es no poner punto (71 fotos); el punto en el lugar equivocado ocurre en 19.

Referencia: la regla fija "el último dígito es el decimal" acierta el 96,9 % en esta flota (casi todos los modelos tienen un decimal), pero no es visión y falla en cualquier modelo con otra disposición.

**En el pipeline** (`read_meter(..., gaps=...)`), como respaldo del clasificador de formato: decide cuando este duda (probabilidad < 0,5) y corrige un formato sin fracción cuando un hueco muestra el punto con probabilidad ≥ 0,9 (`GAP_OVERRIDE_PROB`). En la muestra de prueba los huecos decidieron en 2 fotos y la exactitud queda en 66,7 % (configuración F de `scripts/evaluate_pipeline.py --gap-model`, en `experiments/pipeline_ceml_energia_huecos/`); usar siempre los huecos la baja a 60,2 %. En fotos tomadas con la app de medidores nuevos, el respaldo ubicó el decimal en dos casos donde el clasificador no lo hacía: uno con el formato dudoso (p = 0,40) y otro con un formato sin fracción asignado con seguridad (p = 0,93).

## Notas

- **B frente a A**: el visor derivado de water-meter-dataset sube el recall en CEML de 0,867 a 0,911 y el mAP50 de 0,858 a 0,937. En Toloka baja algo el mAP50-95 (0,877 → 0,847), esperable porque el visor derivado sigue el contorno de los dígitos y no el marco anotado en Toloka.
- **C frente a B**: 1024 px sube el recall en CEML a 0,931, pero duplica la latencia y baja el mAP50-95 en CEML (0,434 → 0,401).

- La métrica que más importa para el pipeline es el **recall** en CEML: un visor no detectado es una lectura perdida.
- **Dice e IoU por foto** (`scripts/evaluate_seg_dice.py`, `dice.json` de cada experimento): en CEML agua, A 0,786 / B 0,765 / C 0,763 de Dice, con 11,1 %, 8,0 % y 7,7 % de fotos sin visor. A tiene mejor contorno pero pierde más visores; por eso se mantiene B.
- Parte de la caída en mAP50-95 puede deberse a criterios de borde distintos entre los polígonos de Toloka (manuales) y los de CEML (SAM 3 con revisión manual); la caída en mAP50 es menos sensible a ese efecto.
