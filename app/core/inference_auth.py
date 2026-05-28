"""API key access control for inference endpoints."""

import os
import secrets

from fastapi import HTTPException, Request, status

from app.core.auth_client import auth_service_url, verify_with_auth_service

API_KEY_HEADER = "X-API-Key"
AUTHORIZATION_HEADER = "Authorization"


def get_configured_inference_api_key() -> str:
    key = (os.environ.get("INFERENCE_API_KEY") or "").strip()
    if not key:
        raise RuntimeError(
            "INFERENCE_API_KEY is not configured. Set INFERENCE_API_KEY in environment "
            "before using inference API."
        )
    return key


def inference_api_key_configured() -> bool:
    if auth_service_url():
        return True
    return bool((os.environ.get("INFERENCE_API_KEY") or "").strip())


def extract_inference_api_key(request: Request) -> str | None:
    direct = request.headers.get(API_KEY_HEADER)
    if direct:
        return direct.strip()

    auth_header = request.headers.get(AUTHORIZATION_HEADER, "")
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip()
    return None


def verify_inference_api_key(request: Request) -> None:
    provided = extract_inference_api_key(request)
    if not provided:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing inference API key",
        )

    if auth_service_url():
        if verify_with_auth_service(provided, "inference"):
            return

    if inference_api_key_configured() and secrets.compare_digest(
        provided, get_configured_inference_api_key()
    ):
        return

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing inference API key",
    )


async def require_inference_api_key(request: Request) -> None:
    if not inference_api_key_configured():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference auth is not configured on the server",
        )
    verify_inference_api_key(request)
