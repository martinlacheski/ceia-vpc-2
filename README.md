# Lectura automática de medidores de agua con visión por computadora

Trabajo práctico final de **Visión por Computadora II** — CEIA, FIUBA (2026).

## Objetivo

Leer automáticamente el estado de medidores de agua de tipo rodillo a partir de fotos tomadas en campo, con un pipeline basado en redes neuronales convolucionales:

1. **Segmentación de instancia del visor**: YOLOv8n-seg vs Mask R-CNN.
2. **Rectificación**: se ajusta un rectángulo rotado mínimo a la máscara y se endereza el visor con una transformación afín (`src/meter_reader/rectify.py`).
3. **Detección de dígitos**: YOLOv8 con una clase por dígito (0–9); la lectura se obtiene ordenando las detecciones.
4. **Baseline**: OCR genérico (EasyOCR) sobre el mismo recorte.

**Experimento principal:** evaluar si segmentar y rectificar el visor mejora la lectura frente a un recorte por bounding box. Métricas: exactitud de lectura completa y CER.

## Datos

Ver [data/README.md](data/README.md).

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## App web de toma de lecturas

Prototipo para el uso real del sistema: el lecturista inicia sesión, recorre su ruta (listado y mapa), saca la foto del medidor, el pipeline propone la lectura y él la confirma o corrige; se guardan la foto, las imágenes procesadas y la ubicación GPS donde se tomó la foto. El administrador ve el historial y el porcentaje de lecturas aceptadas sin corrección.

```bash
docker compose up -d --build
```

Abrir http://localhost:8080. Usuarios de demo: `lector1`, `lector2` y `admin`, con contraseña `demo1234` (solo para desarrollo local). Los medidores son los de las fotos CEML, con direcciones y coordenadas ficticias; el botón *Usar foto de ejemplo* analiza la foto real de cada medidor.

| Servicio | Qué es |
|---|---|
| `web` | Frontend React + Vite servido por nginx ([app/frontend](app/frontend/)) |
| `api` | FastAPI con el pipeline de visión ([app/backend](app/backend/)) |
| `migrate` | Aplica las migraciones Alembic de [app/db](app/db/) y termina |
| `db` | PostgreSQL |
| `storage` | SeaweedFS (API S3) para las fotos y las imágenes procesadas ([app/storage](app/storage/)) |

`docker compose down` conserva los datos; `docker compose down -v` borra la base y las imágenes.

### Probar desde el celular con ngrok

El celular solo permite usar la cámara y el GPS en páginas servidas por HTTPS. [ngrok](https://ngrok.com) publica la app local con una URL HTTPS; alcanza con un único túnel (plan gratuito) hacia el servicio `web`, porque su nginx reenvía `/api` a la API.

**1. Cuenta e instalación (una sola vez)**

1. Crear una cuenta gratuita en https://dashboard.ngrok.com/signup.
2. Instalar ngrok siguiendo https://ngrok.com/download (en Linux: `sudo snap install ngrok` o el paquete de la página).
3. Copiar el *authtoken* desde *Your Authtoken* en el panel y registrarlo:

   ```bash
   ngrok config add-authtoken <TU_AUTHTOKEN>
   ```

   El token queda en la configuración local de ngrok (`~/.config/ngrok/ngrok.yml`); no debe copiarse en el repositorio.
4. Opcional: en *Domains* del panel, reclamar el dominio estático gratuito (`<algo>.ngrok-free.app`), para que la URL no cambie entre sesiones.

**2. Levantar la app con una contraseña propia**

La URL de ngrok es pública: cualquiera que la tenga llega al login. Antes de exponerla, conviene reemplazar la contraseña de demo. Los usuarios se crean en el primer arranque, así que la contraseña se fija con la base vacía:

```bash
docker compose down -v      # solo si ya se había levantado con otra contraseña (borra lecturas e imágenes)
DEMO_PASSWORD='<una-contraseña>' COOKIE_SECURE=true docker compose up -d --build
```

`COOKIE_SECURE=true` marca la cookie de sesión como segura, porque el acceso es por HTTPS.

**3. Abrir el túnel**

```bash
ngrok http 8080                                   # URL nueva en cada sesión
ngrok http --url=<tu-dominio>.ngrok-free.app 8080 # con el dominio estático
```

ngrok muestra la URL pública en la línea *Forwarding* (`https://....ngrok-free.app`). El túnel queda abierto mientras la terminal siga corriendo; se cierra con `Ctrl+C`.

**4. Tomar una lectura real desde el celular**

1. Abrir la URL `https://...` en el navegador del celular. El plan gratuito muestra primero un aviso de ngrok: tocar **Visit Site**.
2. Ingresar con `lector1` (o `lector2`) y la contraseña elegida.
3. Entrar a la ruta y elegir un medidor (son medidores de prueba; se puede fotografiar cualquier medidor de agua de rodillos).
4. Tocar el campo **Foto del medidor**: el celular ofrece abrir la cámara. Encuadrar el visor con buena luz y sin reflejos, y sacar la foto.
5. La app muestra la segmentación del visor, el visor rectificado, los dígitos detectados y la lectura propuesta. Corregirla si hace falta y tocar **Confirmar lectura**.
6. Aceptar el permiso de **ubicación** cuando el navegador lo pida: la lectura se guarda con el punto GPS donde se tomó la foto.
7. Con `admin`, la pestaña **Mapa de lecturas** muestra dónde se tomó cada foto (satélite o calles) y **Historial de lecturas** lista todas las lecturas; cada una abre un detalle con la foto, la segmentación del visor, el visor rectificado, los dígitos y el mapa de la ubicación.

**Precisión del GPS**

Una página web no elige entre GPS y Wi-Fi: pide la máxima precisión y el sistema operativo combina las fuentes disponibles. La app empieza a seguir la ubicación apenas se abre la pantalla del medidor, conserva la posición más precisa y la muestra en vivo (*GPS: ±6 m*; verde hasta 15 m, ámbar hasta 50 m, rojo por encima). El GPS tarda unos segundos en "engancharse": si el indicador no está en verde, conviene esperar antes de confirmar. Para que el celular use el GPS:

- **Android**: *Ajustes → Ubicación* activada, con *Precisión de la ubicación de Google* encendida. En Chrome, el permiso del sitio debe ser *ubicación precisa* (no *aproximada*): candado de la barra de direcciones → *Permisos → Ubicación*.
- **iPhone**: *Ajustes → Privacidad y seguridad → Localización → Safari* (o el navegador que se use) → *Al usar la app* y **Ubicación exacta** activada.
- A cielo abierto el GPS es mucho más preciso; bajo techo el celular se apoya en Wi-Fi y antenas (decenas o cientos de metros).

**Si algo falla**

- No pide ubicación o no la guarda: verificar que la URL empiece con `https://`, que el GPS del celular esté activo y que el navegador tenga permiso de ubicación para ese sitio (se puede revisar desde el candado de la barra de direcciones).
- Error al analizar la foto: revisar `docker compose logs api`.
- La URL dejó de andar: el túnel se cerró o cambió (sin dominio estático cambia en cada `ngrok http`).

## Integrantes

- Lacheski, Martin
- Lastra, Nicolas
- Scheidl, Hernando
- Vazquez, Diego

