# Contexto de build: la RAÍZ del repo (necesita src/, models/, data/ceml y data/ceml_energia).
#   docker build -f app/backend/Dockerfile .
# En docker compose: build: { context: ., dockerfile: app/backend/Dockerfile }
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

WORKDIR /srv

# Torch solo CPU: evita la imagen de varios GB con CUDA.
RUN pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu

COPY app/backend/requirements.txt /tmp/requirements.txt
# ultralytics arrastra opencv-python (con GUI), que pide libxcb/libGL ausentes en la imagen slim:
# se reemplaza por la variante headless.
RUN pip install -r /tmp/requirements.txt \
    && pip uninstall -y opencv-python \
    && pip install --force-reinstall --no-deps opencv-python-headless==5.0.0.93

COPY app/backend/app /srv/backend/app
COPY app/db /srv/db
COPY src/meter_reader /srv/src/meter_reader
COPY models/*.pt /srv/models/
COPY data/ceml /srv/data/ceml
# De energía solo hace falta la metadata y las fotos (fotos de muestra de los medidores).
COPY data/ceml_energia/metadata.csv /srv/data/ceml_energia/metadata.csv
COPY data/ceml_energia/images /srv/data/ceml_energia/images

ENV METER_READER_SRC=/srv/src \
    SEG_WEIGHTS=/srv/models/toloka_yfzw4_yolov8n_seg.pt \
    DIGIT_WEIGHTS=/srv/models/digits_yolov8n.pt \
    METER_MODEL_WEIGHTS=/srv/models/meter_layout_mnv3s.pt \
    CEML_DIR=/srv/data/ceml \
    ENERGY_SEG_WEIGHTS=/srv/models/toloka_yfzw4_energia_yolov8n_seg.pt \
    ENERGY_DIGIT_WEIGHTS=/srv/models/digits_mix_energia_yolov8n.pt \
    ENERGY_METER_MODEL_WEIGHTS=/srv/models/meter_layout_agua_energia.pt \
    CEML_ENERGY_DIR=/srv/data/ceml_energia

WORKDIR /srv/backend
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
