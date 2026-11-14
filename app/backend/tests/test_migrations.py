"""Las migraciones de app/db deben crear el esquema y coincidir con los modelos."""

from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect

from app.config import Settings
from app.db import Base
from app.main import create_app
from app.vision import get_reader
from helpers import alembic_config, login, migrate
import app.models  # noqa: F401  registra las tablas en Base.metadata


def test_migrations_create_schema_matching_models(tmp_path):
    url = f"sqlite:///{tmp_path / 'empty.db'}"
    migrate(url)
    engine = create_engine(url)
    assert {"users", "routes", "meters", "analyses", "readings"} <= set(inspect(engine).get_table_names())
    with engine.connect() as connection:
        diff = compare_metadata(MigrationContext.configure(connection), Base.metadata)
    assert diff == []  # modelo y migración sin deriva


def test_app_works_on_a_migrated_empty_database(settings, fake_reader):
    # `settings` ya migró una base vacía; la API solo siembra, no crea tablas.
    application = create_app(settings)
    application.dependency_overrides[get_reader] = lambda: fake_reader
    with TestClient(application) as client:
        login(client, "lector1")
        assert client.get("/api/routes").json()


def test_api_does_not_create_tables(tmp_path, fake_reader, settings):
    bare = Settings(database_url=f"sqlite:///{tmp_path / 'bare.db'}", data_dir=settings.data_dir,
                    ceml_dir=settings.ceml_dir)
    try:
        with TestClient(create_app(bare)):
            raised = False
    except Exception:
        raised = True
    assert raised  # sin migrar, el arranque falla en vez de crear tablas a escondidas


def test_review_migration_downgrades_and_upgrades(tmp_path):
    url = f"sqlite:///{tmp_path / 'roundtrip.db'}"
    migrate(url)
    review_columns = {"reviewed_value", "reviewed_by_id", "reviewed_at"}

    def columns():
        return {c["name"] for c in inspect(create_engine(url)).get_columns("readings")}

    assert review_columns <= columns()
    command.downgrade(alembic_config(url), "0001_initial")
    assert not review_columns & columns()
    command.upgrade(alembic_config(url), "head")
    assert review_columns <= columns()
