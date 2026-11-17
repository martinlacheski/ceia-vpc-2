"""Tablas de la base."""

from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(50), unique=True)
    full_name: Mapped[str] = mapped_column(String(100))
    role: Mapped[str] = mapped_column(String(10))  # "reader" | "admin"
    password_hash: Mapped[str] = mapped_column(String(200))


class Route(Base):
    __tablename__ = "routes"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(50), unique=True)
    assignee_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    assignee: Mapped[User] = relationship()
    meters: Mapped[list["Meter"]] = relationship(back_populates="route", order_by="Meter.code")


class Meter(Base):
    __tablename__ = "meters"

    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(String(50), unique=True)
    service: Mapped[str] = mapped_column(String(10), default="water", server_default="water")  # "water" | "energy"
    address: Mapped[str] = mapped_column(String(200))
    lat: Mapped[float] = mapped_column(Float)  # coordenadas ficticias (ver seed.py)
    lon: Mapped[float] = mapped_column(Float)
    route_id: Mapped[int] = mapped_column(ForeignKey("routes.id"))
    route: Mapped[Route] = relationship(back_populates="meters")
    sample_photo: Mapped[str | None] = mapped_column(String(200))  # archivo dentro de CEML_DIR/images


class Analysis(Base):
    """Foto analizada a la espera de que el lecturista confirme la lectura."""

    __tablename__ = "analyses"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)  # uuid
    meter_id: Mapped[int] = mapped_column(ForeignKey("meters.id"))
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    photo_path: Mapped[str] = mapped_column(String(200))  # clave del objeto en el almacenamiento (analyses/<id>/photo.jpg)
    proposed_value: Mapped[str | None] = mapped_column(String(50))
    confidence: Mapped[float | None] = mapped_column(Float)
    used: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Reading(Base):
    __tablename__ = "readings"

    id: Mapped[int] = mapped_column(primary_key=True)
    meter_id: Mapped[int] = mapped_column(ForeignKey("meters.id"), index=True)
    meter: Mapped[Meter] = relationship()
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    user: Mapped[User] = relationship(foreign_keys=[user_id])
    analysis_id: Mapped[str] = mapped_column(ForeignKey("analyses.id"), unique=True)
    analysis: Mapped[Analysis] = relationship()
    value: Mapped[str] = mapped_column(String(50))
    proposed_value: Mapped[str | None] = mapped_column(String(50))
    corrected: Mapped[bool] = mapped_column(Boolean)
    confidence: Mapped[float | None] = mapped_column(Float)
    lat: Mapped[float | None] = mapped_column(Float)
    lon: Mapped[float | None] = mapped_column(Float)
    accuracy_m: Mapped[float | None] = mapped_column(Float)
    taken_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    # Revisión del administrador: no pisa lo que confirmó el lecturista (value, corrected),
    # para que la métrica de aceptación del modelo en campo siga siendo la original.
    reviewed_value: Mapped[str | None] = mapped_column(String(50))
    reviewed_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    reviewed_by: Mapped[User | None] = relationship(foreign_keys=[reviewed_by_id])
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


__all__ = ["User", "Route", "Meter", "Analysis", "Reading"]
