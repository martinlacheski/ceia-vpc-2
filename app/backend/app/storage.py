"""Almacenamiento de imágenes de los análisis: S3 (SeaweedFS) o disco para desarrollo local."""

import time
from pathlib import Path
from typing import Protocol

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from .config import Settings


class Storage(Protocol):
    def put(self, key: str, data: bytes, content_type: str = "image/jpeg") -> None: ...

    def get(self, key: str) -> bytes | None:
        """Devuelve el contenido del objeto, o None si no existe."""


class FilesystemStorage:
    """Guarda los objetos como archivos bajo un directorio raíz."""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()

    def _path(self, key: str) -> Path:
        path = (self.root / key).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"Clave fuera del almacenamiento: {key}")
        return path

    def put(self, key: str, data: bytes, content_type: str = "image/jpeg") -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def get(self, key: str) -> bytes | None:
        try:
            path = self._path(key)
        except ValueError:
            return None
        return path.read_bytes() if path.is_file() else None


class S3Storage:
    """Cliente S3 sobre un bucket; lo crea al iniciar y reintenta mientras el servicio arranca."""

    def __init__(self, client, bucket: str, attempts: int = 30, delay: float = 1.0):
        self.client = client
        self.bucket = bucket
        self._ensure_bucket(attempts, delay)

    def _ensure_bucket(self, attempts: int, delay: float) -> None:
        for attempt in range(attempts):
            try:
                try:
                    self.client.head_bucket(Bucket=self.bucket)
                except ClientError as error:
                    if error.response.get("Error", {}).get("Code") not in {"404", "NoSuchBucket", "NotFound"}:
                        raise
                    self.client.create_bucket(Bucket=self.bucket)
                return
            except (BotoCoreError, ClientError):
                if attempt == attempts - 1:
                    raise
                time.sleep(delay)

    def put(self, key: str, data: bytes, content_type: str = "image/jpeg") -> None:
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data, ContentType=content_type)

    def get(self, key: str) -> bytes | None:
        try:
            return self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        except ClientError as error:
            if error.response.get("Error", {}).get("Code") in {"NoSuchKey", "404", "NotFound"}:
                return None
            raise


def make_storage(settings: Settings) -> Storage:
    if not settings.s3_endpoint:
        return FilesystemStorage(settings.data_dir)
    client = boto3.client(
        "s3", endpoint_url=settings.s3_endpoint, aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key, region_name="us-east-1",
        config=Config(s3={"addressing_style": "path"}, retries={"max_attempts": 2}))
    return S3Storage(client, settings.s3_bucket)
