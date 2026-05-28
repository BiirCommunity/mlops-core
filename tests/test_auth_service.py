import pytest
from fastapi.testclient import TestClient

from auth_service.config import Settings
from auth_service.routes import create_app


@pytest.fixture
def client(tmp_path) -> TestClient:
    settings = Settings(
        database_path=str(tmp_path / "auth.db"),
        bootstrap_username="admin",
        bootstrap_password="secret123",
        token_prefix="test_",
        host="127.0.0.1",
        port=8090,
    )
    app = create_app(settings)
    return TestClient(app)


def test_login_and_verify(client: TestClient) -> None:
    login = client.post(
        "/auth/login", json={"username": "admin", "password": "secret123"}
    )
    assert login.status_code == 200
    token = login.json()["token"]

    verify = client.post(
        "/auth/verify",
        headers={"Authorization": f"Bearer {token}"},
        json={"scope": "admin"},
    )
    assert verify.status_code == 200
    assert verify.json()["valid"] is True


def test_users_crud(client: TestClient) -> None:
    login = client.post(
        "/auth/login", json={"username": "admin", "password": "secret123"}
    )
    token = login.json()["token"]
    headers = {"Authorization": f"Bearer {token}"}

    created = client.post(
        "/users",
        headers=headers,
        json={"username": "demo", "password": "demo1234", "active": True},
    )
    assert created.status_code == 200
    user_id = created.json()["id"]

    listed = client.get("/users", headers=headers)
    assert listed.status_code == 200
    assert listed.json()["count"] >= 2

    updated = client.patch(
        f"/users/{user_id}",
        headers=headers,
        json={"active": False},
    )
    assert updated.status_code == 200
    assert updated.json()["active"] is False

    deleted = client.delete(f"/users/{user_id}", headers=headers)
    assert deleted.status_code == 200
