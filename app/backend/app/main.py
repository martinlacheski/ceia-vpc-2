"""Punto de entrada: `uvicorn app.main:app` (desde app/backend)."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from .config import Settings
from .db import make_engine, make_session_factory
from .routers import auth, catalog, readings
from .seed import seed
from .storage import make_storage


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        engine = make_engine(settings.database_url)
        # El esquema lo crea Alembic (servicio `migrate`, ver app/db); aquí solo se siembra.
        app.state.session_factory = make_session_factory(engine)
        app.state.storage = make_storage(settings)
        with app.state.session_factory() as session:
            seed(session, settings)
        yield
        engine.dispose()

    app = FastAPI(title="Lecturas de medidores", lifespan=lifespan)
    app.state.settings = settings
    app.state.reader = None

    @app.get("/health")
    def health():
        return {"status": "ok"}

    for router in (auth.router, catalog.router, readings.router):
        app.include_router(router)
    return app


app = create_app() if __name__ != "__main__" else None
