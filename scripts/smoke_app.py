"""Prueba de humo de punta a punta de la app web levantada con docker compose.

Para agua (lector1) y energía (lector3): inicia sesión, toma el primer medidor de su ruta, analiza
su foto de ejemplo con el pipeline, confirma la lectura propuesta y verifica que el administrador la
ve con la foto y las tres imágenes procesadas. Crea una lectura por servicio en la base.

Uso (con la app levantada; solo biblioteca estándar):
  python scripts/smoke_app.py                         # http://localhost:8080, contraseña demo1234
  python scripts/smoke_app.py --url http://localhost:8092
La contraseña se toma de DEMO_PASSWORD si está definida.
"""

import argparse
import http.cookiejar
import json
import os
import urllib.request
import uuid

JPEG = b"\xff\xd8"


class Client:
    def __init__(self, base: str):
        self.base = base.rstrip("/")
        cookies = urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())
        self.opener = urllib.request.build_opener(cookies)

    def call(self, method: str, path: str, body: bytes | None = None, headers: dict | None = None,
             raw: bool = False):
        request = urllib.request.Request(self.base + path, data=body, method=method, headers=headers or {})
        with self.opener.open(request, timeout=120) as response:
            data = response.read()
            return response.status, data if raw else json.loads(data or b"null")

    def post_json(self, path: str, payload: dict):
        return self.call("POST", path, json.dumps(payload).encode(), {"content-type": "application/json"})


def login(base: str, user: str, password: str) -> Client:
    client = Client(base)
    status, _ = client.post_json("/api/auth/login", {"username": user, "password": password})
    assert status == 200, f"login de {user}: {status}"
    return client


def photo_form(content: bytes) -> tuple[bytes, dict]:
    boundary = uuid.uuid4().hex
    body = (f'--{boundary}\r\nContent-Disposition: form-data; name="photo"; filename="foto.jpg"\r\n'
            "Content-Type: image/jpeg\r\n\r\n").encode() + content + f"\r\n--{boundary}--\r\n".encode()
    return body, {"content-type": f"multipart/form-data; boundary={boundary}"}


def read_first_meter(base: str, user: str, password: str) -> dict:
    """Recorre el flujo del lecturista sobre el primer medidor de su primera ruta."""
    client = login(base, user, password)
    _, routes = client.call("GET", "/api/routes")
    _, meters = client.call("GET", f"/api/routes/{routes[0]['id']}/meters")
    meter = meters[0]
    _, photo = client.call("GET", f"/api/meters/{meter['id']}/sample-photo", raw=True)
    body, headers = photo_form(photo)
    _, analysis = client.call("POST", f"/api/meters/{meter['id']}/analyze", body, headers)
    assert analysis["display_found"], f"{user}: no se encontró el visor"
    status, reading = client.post_json(f"/api/meters/{meter['id']}/readings", {
        "analysis_id": analysis["analysis_id"], "value": analysis["reading"] or "0",
        "lat": -27.0, "lon": -55.0, "accuracy_m": 8})
    assert status == 201, f"{user}: la lectura no se guardó ({status})"
    return {"usuario": user, "ruta": routes[0]["name"], "medidor": meter.get("code", meter["id"]),
            "lectura": analysis["reading"], "confianza": analysis["confidence"], "id": reading["id"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument("--password", default=os.environ.get("DEMO_PASSWORD", "demo1234"))
    args = parser.parse_args()

    results = [read_first_meter(args.url, user, args.password) for user in ("lector1", "lector3")]

    admin = login(args.url, "admin", args.password)
    _, readings = admin.call("GET", "/api/readings")
    items = readings.get("items", readings) if isinstance(readings, dict) else readings
    ids = {r["id"] for r in items}
    for result in results:
        assert result["id"] in ids, f"el administrador no ve la lectura {result['id']}"
        for path in [f"/api/readings/{result['id']}/photo"] + [
                f"/api/readings/{result['id']}/images/{kind}" for kind in ("overlay", "crop", "digits")]:
            status, image = admin.call("GET", path, raw=True)
            assert status == 200 and image[:2] == JPEG, f"imagen faltante: {path}"
        print("ok", json.dumps(result, ensure_ascii=False))
    _, stats = admin.call("GET", "/api/stats")
    print("estadísticas", json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
