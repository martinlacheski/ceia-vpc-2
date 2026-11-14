"""Análisis de fotos, registro de lecturas, historial y estadísticas."""

import base64
import uuid

from fastapi import APIRouter, Depends, File, HTTPException, Request, Response, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..auth import admin_user, current_user, get_settings
from ..config import Settings
from ..db import get_db
from ..models import Analysis, Meter, Reading, Route, User
from ..serializers import accessible_meter, reading_out
from ..storage import Storage
from ..vision import MeterReader, get_reader

router = APIRouter(prefix="/api", tags=["readings"])

IMAGE_KINDS = ("overlay", "crop", "digits")


def get_storage(request: Request) -> Storage:
    return request.app.state.storage




def _data_url(jpeg: bytes | None) -> str | None:
    if not jpeg:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()


@router.post("/meters/{meter_id}/analyze")
def analyze(photo: UploadFile = File(...), meter: Meter = Depends(accessible_meter),
            user: User = Depends(current_user), db: Session = Depends(get_db),
            settings: Settings = Depends(get_settings), storage: Storage = Depends(get_storage),
            reader: MeterReader = Depends(get_reader)):
    data = photo.file.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(413, "La foto es demasiado grande")
    try:
        output = reader.analyze(data)
    except ValueError:
        raise HTTPException(422, "La foto no es una imagen válida")

    analysis_id = str(uuid.uuid4())
    prefix = f"analyses/{analysis_id}"
    storage.put(f"{prefix}/photo.jpg", data)
    for kind, jpeg in (("overlay", output.overlay_jpeg), ("crop", output.crop_jpeg),
                       ("digits", output.digits_jpeg)):
        if jpeg:
            storage.put(f"{prefix}/{kind}.jpg", jpeg)
    db.add(Analysis(id=analysis_id, meter_id=meter.id, user_id=user.id, photo_path=f"{prefix}/photo.jpg",
                    proposed_value=output.reading, confidence=output.confidence))
    db.commit()
    return {
        "analysis_id": analysis_id,
        "display_found": output.display_found,
        "reading": output.reading,
        "confidence": output.confidence,
        "flipped": output.flipped,
        "digits_model_available": output.digits_model_available,
        "images": {"overlay": _data_url(output.overlay_jpeg), "crop": _data_url(output.crop_jpeg),
                   "digits": _data_url(output.digits_jpeg)},
    }


class ReadingIn(BaseModel):
    analysis_id: str
    value: str = Field(pattern=r"^\d+(\.\d+)?$")
    lat: float | None = Field(default=None, ge=-90, le=90)
    lon: float | None = Field(default=None, ge=-180, le=180)
    accuracy_m: float | None = Field(default=None, ge=0)


@router.post("/meters/{meter_id}/readings", status_code=201)
def create_reading(body: ReadingIn, meter: Meter = Depends(accessible_meter),
                   user: User = Depends(current_user), db: Session = Depends(get_db),
                   settings: Settings = Depends(get_settings)):
    analysis = db.get(Analysis, body.analysis_id)
    if analysis is None or analysis.meter_id != meter.id or analysis.user_id != user.id:
        raise HTTPException(404, "Análisis no encontrado")
    if analysis.used:
        raise HTTPException(409, "El análisis ya fue utilizado en una lectura")

    # Se registra dónde se tomó la foto; solo si llegan las dos coordenadas.
    has_location = body.lat is not None and body.lon is not None
    lat, lon = (body.lat, body.lon) if has_location else (None, None)
    accuracy = body.accuracy_m if has_location else None

    reading = Reading(
        meter_id=meter.id, user_id=user.id, analysis_id=analysis.id, value=body.value,
        proposed_value=analysis.proposed_value,
        corrected=analysis.proposed_value is None or body.value != analysis.proposed_value,
        confidence=analysis.confidence, lat=lat, lon=lon, accuracy_m=accuracy)
    analysis.used = True
    db.add(reading)
    try:
        db.commit()
    except IntegrityError:  # dos pedidos simultáneos con el mismo análisis
        db.rollback()
        raise HTTPException(409, "El análisis ya fue utilizado en una lectura")
    return reading_out(reading)


@router.get("/readings")
def list_readings(route_id: int | None = None, meter_id: int | None = None,
                  db: Session = Depends(get_db), user: User = Depends(current_user)):
    query = select(Reading).order_by(Reading.taken_at.desc(), Reading.id.desc())
    if user.role != "admin":
        query = query.where(Reading.user_id == user.id)
    if route_id is not None:
        route = db.get(Route, route_id)
        if route is not None and user.role != "admin" and route.assignee_id != user.id:
            raise HTTPException(403, "No tenés acceso a esta ruta")
        query = query.join(Meter, Meter.id == Reading.meter_id).where(Meter.route_id == route_id)
    if meter_id is not None:
        query = query.where(Reading.meter_id == meter_id)
    return [reading_out(r) for r in db.scalars(query)]


def _accessible_reading(reading_id: int, db: Session, user: User) -> Reading:
    reading = db.get(Reading, reading_id)
    if reading is None:
        raise HTTPException(404, "Lectura no encontrada")
    if user.role != "admin" and reading.user_id != user.id:
        raise HTTPException(403, "No tenés acceso a esta lectura")
    return reading


@router.get("/readings/{reading_id}/photo")
def reading_photo(reading_id: int, db: Session = Depends(get_db),
                  user: User = Depends(current_user), storage: Storage = Depends(get_storage)):
    reading = _accessible_reading(reading_id, db, user)
    data = storage.get(reading.analysis.photo_path)
    if data is None:
        raise HTTPException(404, "La foto de la lectura no está disponible")
    return Response(data, media_type="image/jpeg")


@router.get("/readings/{reading_id}/images/{kind}")
def reading_image(reading_id: int, kind: str, db: Session = Depends(get_db),
                  user: User = Depends(current_user), storage: Storage = Depends(get_storage)):
    """Imagen procesada del análisis (overlay, crop o digits), con la misma autorización que la foto."""
    reading = _accessible_reading(reading_id, db, user)
    data = storage.get(f"analyses/{reading.analysis_id}/{kind}.jpg") if kind in IMAGE_KINDS else None
    if data is None:
        raise HTTPException(404, "La imagen no está disponible")
    return Response(data, media_type="image/jpeg")


@router.get("/stats")
def stats(db: Session = Depends(get_db), _: User = Depends(admin_user)):
    total = db.scalar(select(func.count()).select_from(Reading))
    accepted = db.scalar(select(func.count()).select_from(Reading).where(Reading.corrected.is_(False)))
    return {
        "readings_total": total,
        "accepted_without_correction_pct": round(100 * accepted / total, 1) if total else 0.0,
        "meters_total": db.scalar(select(func.count()).select_from(Meter)),
        "meters_read": db.scalar(select(func.count(func.distinct(Reading.meter_id)))),
    }
