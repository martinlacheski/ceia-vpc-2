"""Rutas y medidores."""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..auth import current_user, get_settings
from ..config import Settings
from ..db import get_db
from ..models import Meter, Reading, Route, User
from ..serializers import accessible_meter, meter_out

router = APIRouter(prefix="/api", tags=["catalog"])


def _route_out(db: Session, route: Route) -> dict:
    total = db.scalar(select(func.count()).select_from(Meter).where(Meter.route_id == route.id))
    read = db.scalar(select(func.count(func.distinct(Reading.meter_id)))
                     .join(Meter, Meter.id == Reading.meter_id).where(Meter.route_id == route.id))
    return {"id": route.id, "name": route.name,
            "assignee": {"id": route.assignee.id, "full_name": route.assignee.full_name},
            "meters_total": total, "meters_read": read}


@router.get("/routes")
def list_routes(db: Session = Depends(get_db), user: User = Depends(current_user)):
    query = select(Route).order_by(Route.name)
    if user.role != "admin":
        query = query.where(Route.assignee_id == user.id)
    return [_route_out(db, route) for route in db.scalars(query)]


@router.get("/routes/{route_id}/meters")
def route_meters(route_id: int, db: Session = Depends(get_db), user: User = Depends(current_user)):
    route = db.get(Route, route_id)
    if route is None:
        raise HTTPException(404, "Ruta no encontrada")
    if user.role != "admin" and route.assignee_id != user.id:
        raise HTTPException(403, "No tenés acceso a esta ruta")
    return [meter_out(db, meter) for meter in route.meters]


@router.get("/meters/{meter_id}")
def meter_detail(meter: Meter = Depends(accessible_meter), db: Session = Depends(get_db),
                 settings: Settings = Depends(get_settings)):
    return {**meter_out(db, meter), "route_id": meter.route_id,
            "has_sample_photo": _sample_path(settings, meter) is not None}


@router.get("/meters/{meter_id}/sample-photo")
def sample_photo(meter: Meter = Depends(accessible_meter),
                 settings: Settings = Depends(get_settings)):
    path = _sample_path(settings, meter)
    if path is None:
        raise HTTPException(404, "El medidor no tiene foto de muestra")
    return FileResponse(path, media_type="image/jpeg")


def _sample_path(settings: Settings, meter: Meter):
    if not meter.sample_photo:
        return None
    path = settings.ceml_dir / "images" / meter.sample_photo
    return path if path.is_file() else None
