"""Shared API index for root and versioned entrypoints."""

from __future__ import annotations

API_VERSION = "v1"


def build_api_index(*, via_ingress: bool = False) -> dict[str, object]:
    """Describe public API layout (internal paths or /api aliases via Ingress)."""
    if via_ingress:
        return {
            "service": "mlops-core-app",
            "api_version": API_VERSION,
            "docs": "/api/docs",
            "openapi": "/api/openapi.json",
            "groups": {
                "auth": {
                    "prefix": "/api/auth",
                    "service": "auth-service",
                    "examples": ["POST /api/auth/login", "GET /api/users"],
                },
                "inference": {
                    "examples": [
                        "POST /api/chat/completions",
                        "POST /api/chat/feedback",
                    ],
                },
                "training": {
                    "prefix": "/api",
                    "examples": [
                        "GET /api/health",
                        "GET /api/jobs",
                        "GET /api/admin/overview",
                    ],
                },
            },
            "monitoring": {
                "note": "Probe endpoints are on app root, not exposed via /api",
                "examples": ["GET /health", "GET /metrics"],
            },
        }

    return {
        "service": "mlops-core-app",
        "api_version": API_VERSION,
        "docs": "/docs",
        "openapi": "/openapi.json",
        "groups": {
            "inference": {
                "prefix": f"/{API_VERSION}",
                "examples": [
                    f"POST /{API_VERSION}/chat/completions",
                    f"POST /{API_VERSION}/feedback",
                    f"GET /{API_VERSION}/drift/reports",
                ],
            },
            "training": {
                "prefix": f"/{API_VERSION}/training",
                "examples": [
                    f"GET /{API_VERSION}/training/health",
                    f"GET /{API_VERSION}/training/jobs",
                    f"GET /{API_VERSION}/training/admin/overview",
                ],
            },
            "monitoring": {
                "prefix": "/",
                "examples": ["GET /health", "GET /metrics"],
            },
            "auth": {
                "note": "Separate auth-service; via Ingress use /api/auth, /api/users",
            },
        },
    }
