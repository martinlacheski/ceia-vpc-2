"""Fixtures: app con SQLite en un directorio temporal, CEML de juguete y lector falso."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # permite `import app` sin instalar

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from helpers import login, migrate
from app.main import create_app
from app.vision import ReaderOutput, get_reader

METADATA = """file,month,route,meter_id
agua_2019-02_r80_m0001.jpg,2019-02,80,m0001
agua_2019-01_r80_m0001.jpg,2019-01,80,m0001
agua_2019-01_r80_m0002.jpg,2019-01,80,m0002
agua_2019-01_r90_m0003.jpg,2019-01,90,m0003
"""

# Energía: el m0001 repite el id de un medidor de agua (pasa en CEML); la ruta 7 no se siembra.
ENERGY_METADATA = """file,month,route,meter_id
energia_2019-02_r1_m0001.jpg,2019-02,1,m0001
energia_2019-03_r7_m0050.jpg,2019-03,7,m0050
"""


class FakeReader:
    """Lector falso: devuelve la salida configurada sin usar modelos."""

    def __init__(self):
        self.output = ReaderOutput(
            display_found=True, reading="01234", confidence=0.9, flipped=False,
            digits_model_available=True, overlay_jpeg=b"overlay", crop_jpeg=b"crop",
            digits_jpeg=b"digits",
        )
        self.calls = 0
        self.services: list[str] = []

    def analyze(self, photo: bytes, service: str = "water") -> ReaderOutput:
        self.calls += 1
        self.services.append(service)
        return self.output


@pytest.fixture
def settings(tmp_path):
    ceml = tmp_path / "ceml"
    (ceml / "images").mkdir(parents=True)
    (ceml / "metadata.csv").write_text(METADATA)
    for name in ("agua_2019-02_r80_m0001.jpg", "agua_2019-01_r80_m0001.jpg",
                 "agua_2019-01_r80_m0002.jpg", "agua_2019-01_r90_m0003.jpg"):
        cv2.imwrite(str(ceml / "images" / name), np.full((20, 30, 3), 120, np.uint8))
    energy = tmp_path / "ceml_energia"
    (energy / "images").mkdir(parents=True)
    (energy / "metadata.csv").write_text(ENERGY_METADATA)
    for name in ("energia_2019-02_r1_m0001.jpg", "energia_2019-03_r7_m0050.jpg"):
        cv2.imwrite(str(energy / "images" / name), np.full((20, 30, 3), 200, np.uint8))
    database_url = f"sqlite:///{tmp_path / 'test.db'}"
    migrate(database_url)  # el esquema lo crea Alembic, no la API
    return Settings(
        database_url=database_url,
        data_dir=tmp_path / "data",
        ceml_dir=ceml,
        ceml_energy_dir=energy,
        secret_key="test-secret",
        demo_password="demo1234",
    )


@pytest.fixture
def fake_reader():
    return FakeReader()


@pytest.fixture
def app(settings, fake_reader):
    application = create_app(settings)
    application.dependency_overrides[get_reader] = lambda: fake_reader
    return application


@pytest.fixture
def client(app):
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def reader1(client):
    login(client, "lector1")
    return client


@pytest.fixture
def photo():
    return {"photo": ("foto.jpg", b"\xff\xd8fake", "image/jpeg")}
