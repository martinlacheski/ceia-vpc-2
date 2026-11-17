"""Conversión de filas a las estructuras JSON del contrato de la API."""

from datetime import datetime, timezone

from fastapi import Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from .auth import current_user
from .db import get_db
from .models import Meter, Reading, User


def iso(value: datetime) -> str:
    if value.tzinfo is None:  # SQLite no guarda la zona horaria; siempre es UTC
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()


def user_out(user: User) -> dict:
    return {"id": user.id, "username": user.username, "full_name": user.full_name, "role": user.role}


def last_reading(db: Session, meter_id: int) -> Reading | None:
    return db.scalar(select(Reading).where(Reading.meter_id == meter_id)
                     .order_by(Reading.taken_at.desc(), Reading.id.desc()).limit(1))


def final_value(reading: Reading) -> str:
    """Valor vigente: el de la revisión del administrador si la hubo, si no el del lecturista."""
    return reading.reviewed_value if reading.reviewed_value is not None else reading.value


def meter_out(db: Session, meter: Meter) -> dict:
    last = last_reading(db, meter.id)
    return {
        "id": meter.id, "code": meter.code, "service": meter.service, "address": meter.address,
        "lat": meter.lat, "lon": meter.lon,
        "status": "read" if last else "pending",
        "last_reading": {"value": final_value(last), "taken_at": iso(last.taken_at)} if last else None,
    }


def reading_out(reading: Reading) -> dict:
    return {
        "id": reading.id, "meter_id": reading.meter_id, "value": reading.value,
        "proposed_value": reading.proposed_value, "corrected": reading.corrected,
        "confidence": reading.confidence, "lat": reading.lat, "lon": reading.lon,
        "accuracy_m": reading.accuracy_m,
        "taken_at": iso(reading.taken_at),
        "user": {"id": reading.user.id, "full_name": reading.user.full_name},
        "route": {"id": reading.meter.route.id, "name": reading.meter.route.name},
        "final_value": final_value(reading),
        "review": None if reading.reviewed_value is None else {
            "value": reading.reviewed_value,
            "reviewed_by": {"id": reading.reviewed_by.id, "full_name": reading.reviewed_by.full_name},
            "reviewed_at": iso(reading.reviewed_at),
        },
    }


def accessible_meter(meter_id: int, db: Session = Depends(get_db),
                     user: User = Depends(current_user)) -> Meter:
    """Medidor pedido en la URL, si existe y el usuario puede verlo."""
    meter = db.get(Meter, meter_id)
    if meter is None:
        raise HTTPException(404, "Medidor no encontrado")
    if user.role != "admin" and meter.route.assignee_id != user.id:
        raise HTTPException(403, "No tenés acceso a este medidor")
    return meter
