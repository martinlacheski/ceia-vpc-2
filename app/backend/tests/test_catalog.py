from helpers import login


def test_reader_sees_only_own_routes(reader1):
    routes = reader1.get("/api/routes").json()
    assert [r["name"] for r in routes] == ["Ruta 80"]
    route = routes[0]
    assert route["meters_total"] == 2 and route["meters_read"] == 0
    assert set(route["assignee"]) == {"id", "full_name"}


def test_admin_sees_all_routes(client):
    login(client, "admin")
    assert [r["name"] for r in client.get("/api/routes").json()] == ["Ruta 80", "Ruta 90"]


def test_reader_forbidden_on_other_route_meters(reader1, client):
    login(client, "admin")
    other = [r for r in client.get("/api/routes").json() if r["name"] == "Ruta 90"][0]
    login(client, "lector1")
    assert reader1.get(f"/api/routes/{other['id']}/meters").status_code == 403


def test_meters_listing_shape_and_pending_to_read(reader1, photo):
    route_id = reader1.get("/api/routes").json()[0]["id"]
    meters = reader1.get(f"/api/routes/{route_id}/meters").json()
    assert [m["code"] for m in meters] == ["m0001", "m0002"]
    first = meters[0]
    assert set(first) == {"id", "code", "address", "lat", "lon", "status", "last_reading"}
    assert first["status"] == "pending" and first["last_reading"] is None

    analysis = reader1.post(f"/api/meters/{first['id']}/analyze", files=photo).json()
    created = reader1.post(f"/api/meters/{first['id']}/readings",
                           json={"analysis_id": analysis["analysis_id"], "value": "01234"})
    assert created.status_code == 201

    meters = reader1.get(f"/api/routes/{route_id}/meters").json()
    assert meters[0]["status"] == "read"
    assert meters[0]["last_reading"]["value"] == "01234" and meters[0]["last_reading"]["taken_at"]
    assert meters[1]["status"] == "pending"
    assert reader1.get("/api/routes").json()[0]["meters_read"] == 1


def test_meter_detail_and_sample_photo(reader1):
    route_id = reader1.get("/api/routes").json()[0]["id"]
    meter_id = reader1.get(f"/api/routes/{route_id}/meters").json()[0]["id"]
    detail = reader1.get(f"/api/meters/{meter_id}").json()
    assert set(detail) == {"id", "code", "address", "lat", "lon", "route_id", "status",
                           "last_reading", "has_sample_photo"}
    assert detail["has_sample_photo"] is True and detail["route_id"] == route_id
    photo = reader1.get(f"/api/meters/{meter_id}/sample-photo")
    assert photo.status_code == 200 and photo.headers["content-type"] == "image/jpeg"


def test_meter_of_other_route_is_403_and_missing_is_404(reader1, client):
    login(client, "admin")
    other = [r for r in client.get("/api/routes").json() if r["name"] == "Ruta 90"][0]
    other_meter = client.get(f"/api/routes/{other['id']}/meters").json()[0]["id"]
    login(client, "lector1")
    assert reader1.get(f"/api/meters/{other_meter}").status_code == 403
    assert reader1.get("/api/meters/9999").status_code == 404


def test_requires_login(client):
    assert client.get("/api/routes").status_code == 401
