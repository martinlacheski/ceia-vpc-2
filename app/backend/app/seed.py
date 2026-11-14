"""Datos de demo: usuarios, rutas y medidores a partir de la metadata de CEML.

Es idempotente: se puede correr en cada arranque; solo crea lo que falta.
Las direcciones y coordenadas son FICTICIAS: se generan con un RNG con semilla alrededor de un
punto fijo genérico y no corresponden a ubicaciones reales de los medidores.
"""

import csv
import random

from sqlalchemy import select
from sqlalchemy.orm import Session

from .auth import hash_password
from .config import Settings
from .models import Meter, Route, User

# Punto genérico (sin significado real) y radio del área de dispersión, en grados (~1 km).
CENTER_LAT, CENTER_LON = -31.4200, -64.1900
SPREAD_DEG = 0.009
RNG_SEED = 20240

USERS = [  # (username, nombre, rol, ruta CEML asignada)
    ("admin", "Administrador", "admin", None),
    ("lector1", "Lector Uno", "reader", "80"),
    ("lector2", "Lector Dos", "reader", "90"),
]


def read_ceml(settings: Settings) -> dict[str, tuple[str, list[str]]]:
    """Devuelve {meter_id: (ruta, [fotos ordenadas])} a partir de metadata.csv."""
    path = settings.ceml_dir / "metadata.csv"
    if not path.exists():
        return {}
    meters: dict[str, tuple[str, list[str]]] = {}
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            route, files = meters.setdefault(row["meter_id"], (row["route"], []))
            files.append(row["file"])
    return {mid: (route, sorted(files)) for mid, (route, files) in meters.items()}


def seed(db: Session, settings: Settings) -> None:
    users: dict[str, User] = {}
    for username, full_name, role, _ in USERS:
        user = db.scalar(select(User).where(User.username == username))
        if user is None:
            user = User(username=username, full_name=full_name, role=role,
                        password_hash=hash_password(settings.demo_password))
            db.add(user)
        users[username] = user
    db.flush()

    ceml = read_ceml(settings)
    routes: dict[str, Route] = {}
    for username, _, _, route_key in USERS:
        if route_key is None:
            continue
        name = f"Ruta {route_key}"
        route = db.scalar(select(Route).where(Route.name == name))
        if route is None:
            route = Route(name=name, assignee_id=users[username].id)
            db.add(route)
        routes[route_key] = route
    db.flush()

    rng = random.Random(RNG_SEED)
    for meter_id in sorted(ceml):
        route_key, files = ceml[meter_id]
        # Se sortea siempre, exista o no el medidor, para que los valores no dependan de lo ya cargado.
        lat = CENTER_LAT + rng.uniform(-SPREAD_DEG, SPREAD_DEG)
        lon = CENTER_LON + rng.uniform(-SPREAD_DEG, SPREAD_DEG)
        address = f"Calle {rng.randint(1, 60)} Nº {rng.randint(100, 999)}"
        if route_key not in routes or db.scalar(select(Meter).where(Meter.code == meter_id)):
            continue
        db.add(Meter(code=meter_id, address=address, lat=round(lat, 6), lon=round(lon, 6),
                     route_id=routes[route_key].id, sample_photo=files[0]))
    db.commit()
