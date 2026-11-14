"""Almacenamiento de imágenes: implementación en disco y en S3 (moto)."""

import boto3
import pytest
from moto import mock_aws

from app.config import Settings
from app.storage import FilesystemStorage, S3Storage, make_storage


def test_filesystem_roundtrip(tmp_path):
    storage = FilesystemStorage(tmp_path)
    storage.put("analyses/a1/photo.jpg", b"datos")
    assert storage.get("analyses/a1/photo.jpg") == b"datos"
    assert storage.get("analyses/a1/otra.jpg") is None


def test_filesystem_rejects_path_traversal(tmp_path):
    storage = FilesystemStorage(tmp_path)
    with pytest.raises(ValueError):
        storage.put("../fuera.jpg", b"x")
    assert storage.get("../fuera.jpg") is None


def test_make_storage_uses_filesystem_without_endpoint(tmp_path):
    assert isinstance(make_storage(Settings(data_dir=tmp_path)), FilesystemStorage)


@mock_aws
def test_s3_creates_bucket_and_roundtrips():
    client = boto3.client("s3", region_name="us-east-1")
    storage = S3Storage(client, "lecturas", attempts=1)
    storage.put("analyses/a1/photo.jpg", b"datos")
    assert storage.get("analyses/a1/photo.jpg") == b"datos"
    assert storage.get("analyses/a1/overlay.jpg") is None
    S3Storage(client, "lecturas", attempts=1)  # bucket ya existente: no falla
