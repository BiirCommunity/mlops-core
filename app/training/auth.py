import os
import secrets

from fastapi import HTTPException, Request, status

from app.core.auth_client import auth_service_url, verify_with_auth_service

ACCESS_TOKEN_HEADER = "Authorization"
ACCESS_TOKEN_ALT_HEADER = "X-Access-Token"


def get_configured_access_token() -> str:
    token = (os.environ.get("ACCESS_TOKEN") or "").strip()
    if not token:
        raise RuntimeError(
            "ACCESS_TOKEN is not configured. Set ACCESS_TOKEN in environment "
            "before using training/admin API."
        )
    return token


def access_token_configured() -> bool:
    if auth_service_url():
        return True
    return bool((os.environ.get("ACCESS_TOKEN") or "").strip())


def extract_bearer_token(request: Request) -> str | None:
    auth_header = request.headers.get(ACCESS_TOKEN_HEADER, "")
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip()
    alt = request.headers.get(ACCESS_TOKEN_ALT_HEADER)
    if alt:
        return alt.strip()
    return None


def verify_access_token(request: Request) -> None:
    provided = extract_bearer_token(request)
    if not provided:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing access token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if auth_service_url():
        if verify_with_auth_service(provided, "admin"):
            return

    if access_token_configured() and secrets.compare_digest(
        provided, get_configured_access_token()
    ):
        return

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing access token",
        headers={"WWW-Authenticate": "Bearer"},
    )


def verify_token_value(token: str | None) -> None:
    if not access_token_configured():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Access auth is not configured on the server",
        )
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing access token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if auth_service_url() and verify_with_auth_service(token.strip(), "admin"):
        return

    if access_token_configured() and secrets.compare_digest(
        token.strip(), get_configured_access_token()
    ):
        return

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing access token",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def require_access_token(request: Request) -> None:
    if not access_token_configured():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Access auth is not configured on the server",
        )
    verify_access_token(request)
