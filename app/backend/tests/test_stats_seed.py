from sqlalchemy import func, select

from app import models
from app.seed import seed
from helpers import login


def add_reading(client, meter, photo, value):
    analysis = client.post(f"/api/meters/{meter['id']}/analyze", files=photo).json()
    return client.post(f"/api/meters/{meter['id']}/readings", json={
        "analysis_id": analysis["analysis_id"], "value": value,
        "lat": meter["lat"] + 0.001, "lon": meter["lon"]})


def test_stats_admin_only(client):
    assert client.get("/api/stats").status_code == 401
    login(client, "lector1")
    assert client.get("/api/stats").status_code == 403
    login(client, "admin")
    assert client.get("/api/stats").status_code == 200


def test_stats_values(client, photo):
    login(client, "lector1")
    route_id = client.get("/api/routes").json()[0]["id"]
    m1, m2 = client.get(f"/api/routes/{route_id}/meters").json()
    add_reading(client, m1, photo, "01234")   # aceptada
    add_reading(client, m1, photo, "99999")   # corregida
    add_reading(client, m2, photo, "01234")   # aceptada
    add_reading(client, m2, photo, "01234")   # aceptada
    login(client, "admin")
    assert client.get("/api/stats").json() == {
        "readings_total": 4, "accepted_without_correction_pct": 75.0,
        "meters_total": 4, "meters_read": 2}


def test_stats_empty(client):
    login(client, "admin")
    body = client.get("/api/stats").json()
    assert body["readings_total"] == 0 and body["accepted_without_correction_pct"] == 0
    assert body["meters_total"] == 4 and body["meters_read"] == 0


def test_seed_is_idempotent_and_deterministic(app, client, settings):
    session_factory = app.state.session_factory

    def snapshot():
        with session_factory() as session:
            return (
                session.scalar(select(func.count()).select_from(models.User)),
                session.scalar(select(func.count()).select_from(models.Route)),
                session.scalars(select(models.Meter.code, models.Meter.lat, models.Meter.lon)
                                .order_by(models.Meter.code)).all(),
            )

    before = snapshot()
    with session_factory() as session:
        seed(session, settings)
    after = snapshot()
    assert before[0] == 4 and before[1] == 3 and len(before[2]) == 4
    assert [tuple(r) for r in before[2]] == [tuple(r) for r in after[2]] and before[:2] == after[:2]


def test_seed_assigns_routes_and_sample_photo(app, client):
    with app.state.session_factory() as session:
        route80 = session.scalar(select(models.Route).where(models.Route.name == "Ruta 80"))
        assert route80.assignee.username == "lector1"
        meter = session.scalar(select(models.Meter).where(models.Meter.code == "m0001"))
        assert meter.sample_photo == "agua_2019-01_r80_m0001.jpg"  # la primera del medidor
        assert meter.address and meter.lat and meter.lon


def test_seed_adds_energy_meters_without_moving_the_water_ones(app, client, settings):
    with app.state.session_factory() as session:
        water = {m.code: (m.lat, m.lon, m.service) for m in session.scalars(
            select(models.Meter).where(models.Meter.service == "water"))}
        energy = session.scalar(select(models.Meter).where(models.Meter.code == "E-m0001"))
        assert energy.service == "energy" and energy.route.name == "Ruta 1 (energía)"
        assert energy.route.assignee.username == "lector3"
        assert energy.sample_photo == "energia_2019-02_r1_m0001.jpg"
        # solo se siembra la ruta de energía asignada (la 7 queda afuera)
        assert session.scalar(select(models.Meter).where(models.Meter.code == "E-m0050")) is None

    # las coordenadas de agua son las mismas que sin datos de energía
    from dataclasses import replace
    from app.config import Settings
    from helpers import migrate
    url = f"sqlite:///{settings.data_dir.parent / 'water_only.db'}"
    migrate(url)
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    with sessionmaker(create_engine(url))() as session:
        seed(session, replace(settings, ceml_energy_dir=settings.data_dir.parent / "no_existe"))
        water_only = {m.code: (m.lat, m.lon, m.service) for m in session.scalars(select(models.Meter))}
    assert water_only == water
