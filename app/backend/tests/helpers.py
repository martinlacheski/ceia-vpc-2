"""Utilidades compartidas por los tests."""

from pathlib import Path

from alembic import command
from alembic.config import Config

ALEMBIC_INI = Path(__file__).resolve().parents[2] / "db" / "alembic.ini"


def alembic_config(database_url: str) -> Config:
    config = Config(str(ALEMBIC_INI))
    config.set_main_option("sqlalchemy.url", database_url)
    return config


def migrate(database_url: str) -> None:
    """Aplica las migraciones de app/db a la base indicada (como hace el servicio `migrate`)."""
    command.upgrade(alembic_config(database_url), "head")


def login(client, username="lector1", password="demo1234"):
    response = client.post("/api/auth/login", json={"username": username, "password": password})
    assert response.status_code == 200, response.text
    return response
