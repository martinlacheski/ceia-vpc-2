"""Entorno de Alembic: usa los modelos del backend como fuente de verdad del esquema."""

import os
import sys
from pathlib import Path

from alembic import context
from sqlalchemy import create_engine

# El backend vive en app/backend (repo) o /srv/backend (imagen Docker).
_here = Path(__file__).resolve()
for candidate in (_here.parents[2] / "backend", Path("/srv/backend")):
    if (candidate / "app" / "models.py").is_file():
        sys.path.insert(0, str(candidate))
        break

from app import models  # noqa: E402,F401  registra las tablas en Base.metadata
from app.db import Base, make_engine  # noqa: E402

config = context.config
target_metadata = Base.metadata


def _url() -> str:
    # DATABASE_URL gana; sqlalchemy.url del .ini permite fijarla desde código (tests).
    return os.environ.get("DATABASE_URL") or config.get_main_option("sqlalchemy.url")


def run_migrations_offline() -> None:
    context.configure(url=_url(), target_metadata=target_metadata, literal_binds=True,
                      render_as_batch=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    # Programático (tests): la URL explícita en la config tiene prioridad sobre el entorno.
    explicit = config.get_main_option("sqlalchemy.url")
    engine = make_engine(explicit or _url())
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata,
                          render_as_batch=True)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
