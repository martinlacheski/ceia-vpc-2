# Base de datos: migraciones con Alembic

El esquema de PostgreSQL lo crean las migraciones de esta carpeta. La API **no** crea tablas:
solo siembra los datos de demo (de forma idempotente) al arrancar, así que el esquema debe estar
aplicado antes.

- `alembic.ini`: configuración. La URL de la base se toma de `DATABASE_URL`.
- `migrations/env.py`: usa los modelos del backend (`app/backend/app/models.py`) como referencia.
- `migrations/versions/`: una revisión por cambio de esquema (`0001_initial` es el esquema inicial).

## Con docker compose

El servicio `migrate` ejecuta `alembic upgrade head` una vez y termina; `api` arranca solo si
terminó bien. No hace falta hacer nada a mano: `docker compose up -d --build`.

## Crear una migración nueva

1. Cambiar los modelos en `app/backend/app/models.py`.
2. Generar la revisión contra una base con el esquema actual, por ejemplo una SQLite local:

   ```bash
   cd app/backend
   export DATABASE_URL=sqlite:////tmp/lecturas.db
   ../../.venv/bin/alembic -c ../db/alembic.ini upgrade head
   ../../.venv/bin/alembic -c ../db/alembic.ini revision --autogenerate -m "describir el cambio"
   ```

3. Revisar el archivo generado en `migrations/versions/` (Alembic no detecta todo, por ejemplo
   renombres) y confirmar que `downgrade` revierte el cambio.
4. Aplicarla: `alembic -c ../db/alembic.ini upgrade head`.
   En compose: `docker compose run --rm migrate` (o simplemente `docker compose up -d --build`).

Los tests (`app/backend/tests`) aplican las migraciones a una base SQLite vacía y comparan el
resultado con los modelos: si un modelo cambia sin su migración, `test_migrations` falla.

## Desarrollo local sin Docker

Con `uvicorn` y SQLite, aplicar primero las migraciones (`alembic ... upgrade head` con
`DATABASE_URL` apuntando a la misma base que usará la API).
