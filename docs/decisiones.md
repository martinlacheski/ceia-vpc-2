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
- **YOLOv8n y YOLO11n empatan en dígitos**; se elige por la exactitud de lectura completa en CEML.
- **Orientación del visor**: se lee en las dos orientaciones (0° y 180°); decide la fracción a la derecha y, si no, la confianza del detector.

## m³ y fracción

- **El formato depende del modelo de medidor** (5+2, 4+2, 4+4, 4+3, 4+1, 5+0, 4+0), y el modelo se reconoce a simple vista (cuerpo, logo, marco).
- **Clasificador de formato con transfer learning** (MobileNetV3-small) en lugar de solo el color: el color falla con humedad o reflejos y no separa fracciones blancas (70,8 % contra 98,4 % de formato correcto). El color queda como respaldo cuando el clasificador duda.
- **Clasificar y no segmentar**: el formato es una etiqueta por foto, y segmentar la zona de fracción exigía dibujar polígonos nuevos.
- **Se clasifica el formato, no la marca**: junta modelos distintos con la misma disposición y tiene más fotos por clase.
- **Validación cruzada de 5 folds**: con 324 fotos no alcanza un único split; las predicciones out-of-fold se usan para evaluar el pipeline completo sin fuga.

## Limitaciones conocidas y trabajo futuro

- El clasificador solo conoce los formatos que vio; un modelo nuevo se resuelve por el color. Trabajo futuro: detectar la zona de fracción, con pseudo-etiquetas generadas sobre Toloka (todo 5+3) y yfzw4.
- Fotos muy movidas, empañadas o con reflejo fuerte fallan en la segmentación o en los dígitos; su confianza baja permite marcarlas para revisión.
- Si la fracción no se ve (por ejemplo, en m0174), la orientación puede elegirse mal.
