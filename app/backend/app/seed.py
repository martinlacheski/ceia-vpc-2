"""Datos de demo: usuarios, rutas y medidores a partir de la metadata de CEML (agua y energía).

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
# Energía usa su propia secuencia: agregarla no cambia las coordenadas de agua ya sembradas.
ENERGY_RNG_SEED = RNG_SEED + 1
# CEML repite ids entre agua y energía; el código de energía lleva prefijo para que sea único.
ENERGY_CODE_PREFIX = "E-"

USERS = [  # (username, nombre, rol)
    ("admin", "Administrador", "admin"),
    ("lector1", "Lector Uno", "reader"),
    ("lector2", "Lector Dos", "reader"),
    ("lector3", "Lector Tres", "reader"),
]

ROUTES = [  # (servicio, ruta CEML, nombre en la app, lecturista)
    ("water", "80", "Ruta 80", "lector1"),
    ("water", "90", "Ruta 90", "lector2"),
    ("energy", "1", "Ruta 1 (energía)", "lector3"),
]


def read_ceml(settings: Settings, service: str = "water") -> dict[str, tuple[str, list[str]]]:
    """Devuelve {meter_id: (ruta, [fotos ordenadas])} a partir del metadata.csv del servicio."""
    path = settings.photos_dir(service) / "metadata.csv"
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
    for username, full_name, role in USERS:
        user = db.scalar(select(User).where(User.username == username))
        if user is None:
            user = User(username=username, full_name=full_name, role=role,
                        password_hash=hash_password(settings.demo_password))
            db.add(user)
        users[username] = user
    db.flush()

    routes: dict[tuple[str, str], Route] = {}
    for service, route_key, name, username in ROUTES:
        route = db.scalar(select(Route).where(Route.name == name))
        if route is None:
            route = Route(name=name, assignee_id=users[username].id)
            db.add(route)
        routes[(service, route_key)] = route
    db.flush()

    for service, rng_seed, prefix in (("water", RNG_SEED, ""), ("energy", ENERGY_RNG_SEED, ENERGY_CODE_PREFIX)):
        ceml = read_ceml(settings, service)
        rng = random.Random(rng_seed)
        for meter_id in sorted(ceml):
            route_key, files = ceml[meter_id]
            # Se sortea siempre, exista o no el medidor, para que los valores no dependan de lo ya cargado.
            lat = CENTER_LAT + rng.uniform(-SPREAD_DEG, SPREAD_DEG)
            lon = CENTER_LON + rng.uniform(-SPREAD_DEG, SPREAD_DEG)
            address = f"Calle {rng.randint(1, 60)} Nº {rng.randint(100, 999)}"
            code = prefix + meter_id
            route = routes.get((service, route_key))
            if route is None or db.scalar(select(Meter).where(Meter.code == code)):
                continue
            db.add(Meter(code=code, service=service, address=address, lat=round(lat, 6), lon=round(lon, 6),
                         route_id=route.id, sample_photo=files[0]))
    db.commit()
