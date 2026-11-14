import cv2
import numpy as np

from app.vision import ReaderOutput
from helpers import login


def first_meter(client):
    route_id = client.get("/api/routes").json()[0]["id"]
    return client.get(f"/api/routes/{route_id}/meters").json()[0]


def analyze(client, meter_id, photo):
    response = client.post(f"/api/meters/{meter_id}/analyze", files=photo)
    assert response.status_code == 200, response.text
    return response.json()


def test_analyze_contract_shape(reader1, photo):
    meter = first_meter(reader1)
    body = analyze(reader1, meter["id"], photo)
    assert set(body) == {"analysis_id", "display_found", "reading", "confidence", "flipped",
                         "digits_model_available", "images"}
    assert body["reading"] == "01234" and body["confidence"] == 0.9
    assert body["display_found"] is True and body["digits_model_available"] is True
    assert body["images"]["overlay"].startswith("data:image/jpeg;base64,")
    assert set(body["images"]) == {"overlay", "crop", "digits"}


def test_analyze_without_digits_model(reader1, fake_reader, photo):
    fake_reader.output = ReaderOutput(
        display_found=True, reading=None, confidence=None, flipped=False,
        digits_model_available=False, overlay_jpeg=b"o", crop_jpeg=b"c", digits_jpeg=None)
    body = analyze(reader1, first_meter(reader1)["id"], photo)
    assert body["reading"] is None and body["confidence"] is None
    assert body["digits_model_available"] is False and body["display_found"] is True
    assert body["images"]["crop"] and body["images"]["digits"] is None


def test_analyze_without_display(reader1, fake_reader, photo):
    fake_reader.output = ReaderOutput(display_found=False, digits_model_available=True)
    body = analyze(reader1, first_meter(reader1)["id"], photo)
    assert body["display_found"] is False
    assert body["images"] == {"overlay": None, "crop": None, "digits": None}


def test_analyze_forbidden_for_other_route(client, photo):
    login(client, "lector2")
    meter = first_meter(client)
    login(client, "lector1")
    assert client.post(f"/api/meters/{meter['id']}/analyze", files=photo).status_code == 403


def test_analyze_requires_photo(reader1):
    meter = first_meter(reader1)
    assert reader1.post(f"/api/meters/{meter['id']}/analyze").status_code == 422


def test_analyze_invalid_image_is_422(reader1, fake_reader, photo):
    def boom(_):
        raise ValueError("no es imagen")
    fake_reader.analyze = boom
    meter = first_meter(reader1)
    response = reader1.post(f"/api/meters/{meter['id']}/analyze", files=photo)
    assert response.status_code == 422
    assert response.json() == {"detail": "La foto no es una imagen válida"}


def test_reading_accepted_without_correction(reader1, photo):
    meter = first_meter(reader1)
    analysis = analyze(reader1, meter["id"], photo)
    response = reader1.post(f"/api/meters/{meter['id']}/readings",
                            json={"analysis_id": analysis["analysis_id"], "value": "01234"})
    assert response.status_code == 201
    body = response.json()
    assert set(body) == {"id", "meter_id", "value", "proposed_value", "corrected", "confidence",
                         "lat", "lon", "accuracy_m", "taken_at", "user"}
    assert body["proposed_value"] == "01234" and body["corrected"] is False
    assert body["confidence"] == 0.9
    assert body["lat"] is None and body["lon"] is None and body["accuracy_m"] is None
    assert body["user"]["full_name"]


def test_reading_corrected_keeps_proposed_and_value(reader1, photo):
    meter = first_meter(reader1)
    analysis = analyze(reader1, meter["id"], photo)
    body = reader1.post(f"/api/meters/{meter['id']}/readings",
                        json={"analysis_id": analysis["analysis_id"], "value": "01299.5"}).json()
    assert body["value"] == "01299.5" and body["proposed_value"] == "01234"
    assert body["corrected"] is True


def test_reading_without_proposal_counts_as_corrected(reader1, fake_reader, photo):
    fake_reader.output = ReaderOutput(display_found=True, digits_model_available=False)
    meter = first_meter(reader1)
    analysis = analyze(reader1, meter["id"], photo)
    body = reader1.post(f"/api/meters/{meter['id']}/readings",
                        json={"analysis_id": analysis["analysis_id"], "value": "42"}).json()
    assert body["proposed_value"] is None and body["corrected"] is True


def test_reading_stores_photo_location_as_sent(reader1, photo):
    # Se registra dónde se tomó la foto, sin compararla con la ubicación del medidor.
    meter = first_meter(reader1)
    analysis = analyze(reader1, meter["id"], photo)
    body = reader1.post(f"/api/meters/{meter['id']}/readings", json={
        "analysis_id": analysis["analysis_id"], "value": "1",
        "lat": -27.3671, "lon": -55.8961, "accuracy_m": 8.0}).json()
    assert (body["lat"], body["lon"], body["accuracy_m"]) == (-27.3671, -55.8961, 8.0)
    listed = reader1.get("/api/readings").json()[0]
    assert (listed["lat"], listed["lon"], listed["accuracy_m"]) == (-27.3671, -55.8961, 8.0)


def test_reading_invalid_value_is_422(reader1, photo):
    meter = first_meter(reader1)
    analysis = analyze(reader1, meter["id"], photo)
    for bad in ("12a", "", "-5", "1,5", "1."):
        response = reader1.post(f"/api/meters/{meter['id']}/readings",
                                json={"analysis_id": analysis["analysis_id"], "value": bad})
        assert response.status_code == 422, bad


def test_reading_unknown_analysis_is_404(reader1):
    meter = first_meter(reader1)
    response = reader1.post(f"/api/meters/{meter['id']}/readings",
                            json={"analysis_id": "no-existe", "value": "1"})
    assert response.status_code == 404


def test_reading_reused_analysis_is_409(reader1, photo):
    meter = first_meter(reader1)
    analysis = analyze(reader1, meter["id"], photo)
    payload = {"analysis_id": analysis["analysis_id"], "value": "1"}
    assert reader1.post(f"/api/meters/{meter['id']}/readings", json=payload).status_code == 201
    assert reader1.post(f"/api/meters/{meter['id']}/readings", json=payload).status_code == 409


def test_analysis_of_other_meter_is_404(reader1, photo):
    route_id = reader1.get("/api/routes").json()[0]["id"]
    m1, m2 = reader1.get(f"/api/routes/{route_id}/meters").json()
    analysis = analyze(reader1, m1["id"], photo)
    response = reader1.post(f"/api/meters/{m2['id']}/readings",
                            json={"analysis_id": analysis["analysis_id"], "value": "1"})
    assert response.status_code == 404


def test_readings_list_photo_and_permissions(client, photo):
    login(client, "lector1")
    meter = first_meter(client)
    analysis = analyze(client, meter["id"], photo)
    client.post(f"/api/meters/{meter['id']}/readings",
                json={"analysis_id": analysis["analysis_id"], "value": "1"})
    mine = client.get("/api/readings").json()
    assert len(mine) == 1
    reading_id = mine[0]["id"]
    assert client.get(f"/api/readings?meter_id={meter['id']}").json()[0]["id"] == reading_id
    assert client.get("/api/readings?meter_id=9999").json() == []
    photo_response = client.get(f"/api/readings/{reading_id}/photo")
    assert photo_response.status_code == 200
    assert photo_response.headers["content-type"] == "image/jpeg"
    assert photo_response.content == b"\xff\xd8fake"

    login(client, "lector2")
    assert client.get("/api/readings").json() == []
    assert client.get(f"/api/readings/{reading_id}/photo").status_code == 403
    login(client, "admin")
    assert len(client.get("/api/readings").json()) == 1
    assert client.get(f"/api/readings/{reading_id}/photo").status_code == 200
    assert client.get("/api/readings/9999/photo").status_code == 404


def test_readings_list_newest_first(reader1, photo):
    meter = first_meter(reader1)
    for value in ("1", "2"):
        analysis = analyze(reader1, meter["id"], photo)
        reader1.post(f"/api/meters/{meter['id']}/readings",
                     json={"analysis_id": analysis["analysis_id"], "value": value})
    assert [r["value"] for r in reader1.get("/api/readings").json()] == ["2", "1"]


def _create_reading(client, photo):
    meter = first_meter(client)
    analysis = analyze(client, meter["id"], photo)
    response = client.post(f"/api/meters/{meter['id']}/readings",
                           json={"analysis_id": analysis["analysis_id"], "value": "01234"})
    return response.json()["id"]


def test_analysis_images_are_stored_under_analysis_prefix(reader1, photo, app):
    analysis = analyze(reader1, first_meter(reader1)["id"], photo)
    storage = app.state.storage
    prefix = f"analyses/{analysis['analysis_id']}"
    assert storage.get(f"{prefix}/photo.jpg") == photo["photo"][1]
    assert storage.get(f"{prefix}/overlay.jpg") == b"overlay"
    assert storage.get(f"{prefix}/crop.jpg") == b"crop"
    assert storage.get(f"{prefix}/digits.jpg") == b"digits"


def test_reading_images_endpoint(client, photo, fake_reader):
    login(client, "lector1")
    fake_reader.output = ReaderOutput(display_found=True, reading="01234", confidence=0.9,
                                      flipped=False, digits_model_available=False,
                                      overlay_jpeg=b"overlay", crop_jpeg=b"crop", digits_jpeg=None)
    reading_id = _create_reading(client, photo)
    overlay = client.get(f"/api/readings/{reading_id}/images/overlay")
    assert overlay.status_code == 200
    assert overlay.headers["content-type"] == "image/jpeg"
    assert overlay.content == b"overlay"
    assert client.get(f"/api/readings/{reading_id}/images/crop").content == b"crop"
    assert client.get(f"/api/readings/{reading_id}/images/digits").status_code == 404  # no hubo dígitos
    assert client.get(f"/api/readings/{reading_id}/images/otra").status_code == 404
    assert client.get("/api/readings/9999/images/overlay").status_code == 404


def test_reading_images_permissions(client, photo):
    login(client, "lector1")
    reading_id = _create_reading(client, photo)
    client.post("/api/auth/logout")
    assert client.get(f"/api/readings/{reading_id}/images/overlay").status_code == 401
    login(client, "lector2")
    assert client.get(f"/api/readings/{reading_id}/images/overlay").status_code == 403
    client.post("/api/auth/logout")
    login(client, "admin")
    assert client.get(f"/api/readings/{reading_id}/images/overlay").status_code == 200
