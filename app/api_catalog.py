"""Shared API index for root and versioned entrypoints."""

from __future__ import annotations

API_VERSION = "v1"


def build_api_index(*, via_ingress: bool = False) -> dict[str, object]:
    """Describe public API layout."""
    _ = via_ingress
    return {
        "service": "mlops-core-app",
        "api_version": API_VERSION,
        "docs": "/v1/docs",
        "openapi": "/v1/openapi.json",
        "groups": {
            "auth": {
                "prefix": "/v1/auth",
                "service": "auth-service",
                "examples": ["POST /v1/auth/login", "GET /v1/users"],
            },
            "inference": {
                "prefix": "/v1",
                "examples": [
                    "POST /v1/chat/completions",
                    "POST /v1/feedback",
                ],
            },
            "training": {
                "prefix": "/v1/training",
                "examples": [
                    "GET /v1/training/health",
                    "GET /v1/training/jobs",
                    "GET /v1/training/admin/overview",
                ],
            },
            "monitoring": {
                "prefix": "/",
                "examples": [
                    "GET /health",
                    "GET /metrics",
                    "GET /v1/drift/reports",
                ],
            },
        },
    }
