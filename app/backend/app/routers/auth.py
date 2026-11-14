"""Login, logout y usuario actual."""

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..auth import COOKIE_NAME, current_user, get_settings, make_session_token, verify_password
from ..config import Settings
from ..db import get_db
from ..models import User
from ..serializers import user_out

router = APIRouter(prefix="/api/auth", tags=["auth"])


class LoginIn(BaseModel):
    username: str
    password: str


@router.post("/login")
def login(body: LoginIn, response: Response, db: Session = Depends(get_db),
          settings: Settings = Depends(get_settings)):
    user = db.scalar(select(User).where(User.username == body.username))
    if user is None or not verify_password(body.password, user.password_hash):
        raise HTTPException(401, "Usuario o contraseña incorrectos")
    response.set_cookie(COOKIE_NAME, make_session_token(settings, user.id),
                        max_age=settings.session_hours * 3600, httponly=True,
                        samesite="lax", secure=settings.cookie_secure)
    return {"user": user_out(user)}


@router.post("/logout", status_code=204, response_class=Response)
def logout():
    response = Response(status_code=204)
    response.delete_cookie(COOKIE_NAME)
    return response


@router.get("/me")
def me(user: User = Depends(current_user)):
    return {"user": user_out(user)}
