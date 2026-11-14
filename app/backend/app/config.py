"""Configuración de la app, leída de variables de entorno."""

import os
from dataclasses import dataclass
from pathlib import Path

# Raíz del repo: app/backend/app/config.py -> parents[3]
REPO_ROOT = Path(__file__).resolve().parents[3]


def _flag(name: str, default: bool = False) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in {"1", "true", "yes", "si", "sí"}


@dataclass(frozen=True)
class Settings:
    """Parámetros de ejecución. Los valores por defecto sirven para correr en local."""

    database_url: str = "sqlite:///app/data/app.db"
    data_dir: Path = Path("app/data")  # almacenamiento en disco cuando S3_ENDPOINT no está definido
    ceml_dir: Path = REPO_ROOT / "data" / "ceml"  # metadata.csv + images/ (solo lectura)
    src_dir: Path = REPO_ROOT / "src"  # contiene el paquete meter_reader
    seg_weights: Path = REPO_ROOT / "models" / "toloka_yfzw4_yolov8n_seg.pt"
    digit_weights: Path = REPO_ROOT / "models" / "digits_yolov8n.pt"
    # Solo demo: cambiar SECRET_KEY y DEMO_PASSWORD en cualquier despliegue público.
    # Almacenamiento S3 (SeaweedFS en compose). Sin endpoint se usa el disco (data_dir).
    s3_endpoint: str | None = None
    s3_access_key: str = "lecturas-local"
    s3_secret_key: str = "lecturas-local-secret"
    s3_bucket: str = "lecturas"
    secret_key: str = "clave-de-desarrollo-no-usar-en-produccion"
    demo_password: str = "demo1234"
    cookie_secure: bool = False
    session_hours: int = 12
    max_upload_bytes: int = 15 * 1024 * 1024

    @classmethod
    def from_env(cls) -> "Settings":
        defaults = cls()
        env = os.environ.get
        return cls(
            database_url=env("DATABASE_URL", defaults.database_url),
            data_dir=Path(env("APP_DATA_DIR", str(defaults.data_dir))),
            ceml_dir=Path(env("CEML_DIR", str(defaults.ceml_dir))),
            src_dir=Path(env("METER_READER_SRC", str(defaults.src_dir))),
            seg_weights=Path(env("SEG_WEIGHTS", str(defaults.seg_weights))),
            digit_weights=Path(env("DIGIT_WEIGHTS", str(defaults.digit_weights))),
            s3_endpoint=env("S3_ENDPOINT") or None,
            s3_access_key=env("S3_ACCESS_KEY", defaults.s3_access_key),
            s3_secret_key=env("S3_SECRET_KEY", defaults.s3_secret_key),
            s3_bucket=env("S3_BUCKET", defaults.s3_bucket),
            secret_key=env("SECRET_KEY", defaults.secret_key),
            demo_password=env("DEMO_PASSWORD", defaults.demo_password),
            cookie_secure=_flag("COOKIE_SECURE"),
        )
