from helpers import login


def test_login_ok_sets_httponly_cookie_and_me(client):
    response = login(client, "lector1")
    user = response.json()["user"]
    assert user["username"] == "lector1" and user["role"] == "reader"
    assert set(user) == {"id", "username", "full_name", "role"}
    cookie = response.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=lax" in cookie and "secure" not in cookie
    assert client.get("/api/auth/me").json()["user"]["username"] == "lector1"


def test_login_wrong_password_is_401(client):
    response = client.post("/api/auth/login", json={"username": "lector1", "password": "mal"})
    assert response.status_code == 401
    assert response.json() == {"detail": "Usuario o contraseña incorrectos"}


def test_login_unknown_user_is_401(client):
    response = client.post("/api/auth/login", json={"username": "nadie", "password": "demo1234"})
    assert response.status_code == 401


def test_me_without_session_is_401(client):
    assert client.get("/api/auth/me").status_code == 401


def test_me_with_tampered_cookie_is_401(client):
    client.cookies.set("session", "basura")
    assert client.get("/api/auth/me").status_code == 401


def test_logout_closes_session(client):
    login(client)
    assert client.post("/api/auth/logout").status_code == 204
    assert client.get("/api/auth/me").status_code == 401


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}
