"""Contraseñas (scrypt) y sesión por cookie firmada."""

import base64
import hashlib
import hmac
import os

from fastapi import Depends, HTTPException, Request
from itsdangerous import BadSignature, URLSafeTimedSerializer
from sqlalchemy.orm import Session

from .config import Settings
from .db import get_db
from .models import User

COOKIE_NAME = "session"
_SCRYPT = {"n": 2**14, "r": 8, "p": 1}


def hash_password(password: str) -> str:
    """Devuelve `salt$hash` en base64, con sal propia por usuario."""
    salt = os.urandom(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, **_SCRYPT)
    return base64.b64encode(salt).decode() + "$" + base64.b64encode(digest).decode()


def verify_password(password: str, stored: str) -> bool:
    try:
        salt_b64, digest_b64 = stored.split("$")
        salt, expected = base64.b64decode(salt_b64), base64.b64decode(digest_b64)
    except ValueError:
        return False
    digest = hashlib.scrypt(password.encode(), salt=salt, **_SCRYPT)
    return hmac.compare_digest(digest, expected)


def _serializer(settings: Settings) -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(settings.secret_key, salt="session-cookie")


def make_session_token(settings: Settings, user_id: int) -> str:
    return _serializer(settings).dumps({"uid": user_id})


def read_session_token(settings: Settings, token: str) -> int | None:
    try:
        data = _serializer(settings).loads(token, max_age=settings.session_hours * 3600)
    except BadSignature:
        return None
    return data.get("uid") if isinstance(data, dict) else None


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def current_user(request: Request, db: Session = Depends(get_db),
                 settings: Settings = Depends(get_settings)) -> User:
    token = request.cookies.get(COOKIE_NAME)
    user_id = read_session_token(settings, token) if token else None
    user = db.get(User, user_id) if user_id is not None else None
    if user is None:
        raise HTTPException(401, "No autenticado")
    return user


def admin_user(user: User = Depends(current_user)) -> User:
    if user.role != "admin":
        raise HTTPException(403, "Solo para administradores")
    return user
