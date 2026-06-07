import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_CACHE_TTL_SEC = 30.0


def auth_service_url() -> str | None:
    url = (os.environ.get("AUTH_SERVICE_URL") or "").strip().rstrip("/")
    return url or None


def verify_with_auth_service(token: str, scope: str) -> dict[str, Any] | None:
    base = auth_service_url()
    if not base:
        return None

    cache_key = f"{token}:{scope}"
    now = time.time()
    cached = _CACHE.get(cache_key)
    if cached and cached[0] > now:
        return cached[1]

    request = urllib.request.Request(
        f"{base}/auth/verify",
        data=json.dumps({"scope": scope}).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None

    if not payload.get("valid"):
        return None

    _CACHE[cache_key] = (now + _CACHE_TTL_SEC, payload)
    return payload
