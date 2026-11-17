# Decisiones del proyecto

Registro breve de las decisiones técnicas y su motivo. Los resultados numéricos están en [experiments/README.md](../experiments/README.md) y el armado de los datos en [data/README.md](../data/README.md).

## Alcance

- **Solo medidores de agua de rodillos.** La devolución pedía un único tipo de medidor; los de energía mezclan rodillos y LCD, y casi todos los datasets públicos son de agua.
- **CNN como núcleo, OCR como baseline.** La lectura sale de un detector de dígitos (clases 0–9), no de OCR; EasyOCR queda como comparación.
- **Segmentar y rectificar el visor** antes de leer, y medirlo contra un recorte por caja: es el experimento principal.

## Datos

- **Toloka** (polígono del visor) para segmentar y **water-meter-dataset (yfzw4)** (cajas de dígitos) para leer. Se descartaron `water-meter-reading` (71 % de sus fotos ya está en yfzw4: riesgo de fuga) y `meter-digits` (medidores eléctricos, casi un solo equipo).
- **Fotos CEML como conjunto de prueba real**, anonimizadas (nombres neutros, sin EXIF). Cualquier partición de CEML se hace por medidor, no por foto.
- **Visor derivado de los dígitos en yfzw4** (rectángulo rotado mínimo con margen), porque ese dataset no anota el visor: en el informe se describe como etiqueta derivada.
- **Verdad de campo verificada foto por foto.** Las lecturas propuestas por el pipeline solo aceleran la carga; filas copiadas de la propuesta se rehicieron para no medir el modelo contra sí mismo.

## Modelos

- **YOLOv8n-seg sobre Mask R-CNN** para el visor: en CEML su máscara es más precisa (AP50-95 0,359 contra 0,262), es 21 veces más rápido en CPU y pesa 25 veces menos. La rectificación depende del contorno.
- **Sin espejado horizontal en dígitos** (`fliplr=0`): un dígito espejado cambia de clase.
- **YOLOv8n sobre YOLO11n para dígitos**: empatan en el test de yfzw4, pero en CEML YOLOv8n lee mejor (71,2 % contra 61,4 % de lectura exacta). Se decide por la métrica de punta a punta en el dominio real, no por el mAP.
- **Rectificar el visor** antes de leer: +7,6 puntos de lectura exacta frente al recorte por caja.
- **EasyOCR descartado** como lector: 0,3 % de lectura exacta en CEML; queda como línea de base.
- **Orientación del visor**: se lee en las dos orientaciones (0° y 180°); decide la fracción a la derecha y, si no, la confianza del detector.

## m³ y fracción

- **El formato depende del modelo de medidor** (5+2, 4+2, 4+4, 4+3, 4+1, 5+0, 4+0), y el modelo se reconoce a simple vista (cuerpo, logo, marco).
- **Clasificador de formato con transfer learning** (MobileNetV3-small) en lugar de solo el color: el color falla con humedad o reflejos y no separa fracciones blancas (70,8 % contra 98,4 % de formato correcto). El color queda como respaldo cuando el clasificador duda.
- **Clasificar y no segmentar**: el formato es una etiqueta por foto, y segmentar la zona de fracción exigía dibujar polígonos nuevos.
- **Se clasifica el formato, no la marca**: junta modelos distintos con la misma disposición y tiene más fotos por clase.
- **Validación cruzada de 5 folds**: con 324 fotos no alcanza un único split; las predicciones out-of-fold se usan para evaluar el pipeline completo sin fuga.

## Extensión a medidores de energía

- **Mismo pipeline, modelos por servicio**: la cooperativa también lee medidores de energía (rodillos y LCD de 7 segmentos). Se reentrenan el segmentador, el clasificador de formato y el detector de dígitos con fotos de energía, y la app elige los modelos según el servicio del medidor. Agua conserva los modelos de la entrega: el detector entrenado con energía baja la lectura exacta de agua de 71,2 % a 65,2 %.
- **Sin giro de 180° en energía**: están en pared o pilar y se fotografían derechos, no hay fracción roja que decida la orientación y los 7 segmentos girados siguen pareciendo dígitos. Con giro, 78 de 162 errores eran lecturas al revés; sin giro, la orientación iguala al oráculo (`read_meter(..., allow_flip=False)`).
- **Muestra de prueba aparte**: 201 fotos verificadas (`muestra=si`) solo para la evaluación de punta a punta. Para entrenar se excluyen todas las fotos de los medidores de la muestra, y el split de dígitos es por medidor.
- **Dígitos de energía con pre-etiquetas**: el detector de agua propone las cajas sobre los visores rectificados, se corrigen en Roboflow y se revisan contra el modelo. Se probaron dos conjuntos: todos los recortes (483, con algo de ruido) y solo los revisados como legibles (209). Gana el primero por lectura completa (66,7 % contra 65,7 %): más datos pesan más que el ruido.

## Limitaciones conocidas y trabajo futuro

- El clasificador solo conoce los formatos que vio; un modelo nuevo se resuelve por el color. Trabajo futuro: detectar la zona de fracción, con pseudo-etiquetas generadas sobre Toloka (todo 5+3) y yfzw4.
- Fotos muy movidas, empañadas o con reflejo fuerte fallan en la segmentación o en los dígitos; su confianza baja permite marcarlas para revisión.
- Si la fracción no se ve (por ejemplo, en m0174), la orientación puede elegirse mal.
- Energía: los LCD se leen mejor que los rodillos (76/100 contra 58/101 exactas): los rodillos son lo que más queda por mejorar. El modelo se entrenó con fotos de una sola cooperativa.
