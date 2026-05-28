from app.core.auth_client import verify_with_auth_service


def test_verify_with_auth_service_without_url(monkeypatch) -> None:
    monkeypatch.delenv("AUTH_SERVICE_URL", raising=False)
    assert verify_with_auth_service("token", "admin") is None
