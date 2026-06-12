"""OpenAPI schema, Swagger UI routes and security schemes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.responses import HTMLResponse, JSONResponse

if TYPE_CHECKING:
    from fastapi import FastAPI

SECURITY_SCHEMES: dict[str, dict[str, str]] = {
    "InferenceApiKey": {
        "type": "apiKey",
        "in": "header",
        "name": "X-API-Key",
        "description": (
            "Ключ inference (Chat UI / API keys со scope inference). "
            "Создаётся в Admin UI или auth-service."
        ),
    },
    "InferenceBearer": {
        "type": "http",
        "scheme": "bearer",
        "description": (
            "Альтернатива X-API-Key: Authorization: Bearer <token> "
            "с тем же inference-ключом или mlops_* токеном."
        ),
    },
    "AccessTokenBearer": {
        "type": "http",
        "scheme": "bearer",
        "description": (
            "Admin/training токен после POST /api/auth/login "
            "(scope admin). Используется для LoRA, моделей, датасетов."
        ),
    },
    "AccessTokenHeader": {
        "type": "apiKey",
        "in": "header",
        "name": "X-Access-Token",
        "description": "Альтернатива Bearer для training/admin API.",
    },
}

INFERENCE_SECURITY = [{"InferenceApiKey": []}, {"InferenceBearer": []}]
ACCESS_TOKEN_SECURITY = [{"AccessTokenBearer": []}, {"AccessTokenHeader": []}]

PUBLIC_PATHS = {
    "/",
    "/v1",
    "/health",
    "/metrics",
    "/v1/training",
    "/v1/training/",
}

PUBLIC_PREFIXES = (
    "/v1/drift/",
    "/v1/training/auth/status",
    "/v1/training/auth/verify",
)


def _needs_inference_security(path: str, method: str) -> bool:
    if method not in {"get", "post", "put", "patch", "delete"}:
        return False
    return path in {"/v1/chat/completions", "/v1/feedback"} or path.startswith(
        "/v1/chat/"
    )


def _needs_access_token_security(path: str, method: str) -> bool:
    if method not in {"get", "post", "put", "patch", "delete"}:
        return False
    if path in PUBLIC_PATHS or path.startswith(PUBLIC_PREFIXES):
        return False
    return path.startswith("/v1/training/")


def _apply_security(schema: dict) -> None:
    components = schema.setdefault("components", {})
    components["securitySchemes"] = SECURITY_SCHEMES

    for path, path_item in schema.get("paths", {}).items():
        for method, operation in path_item.items():
            if not isinstance(operation, dict):
                continue
            if _needs_inference_security(path, method):
                operation["security"] = INFERENCE_SECURITY
            elif _needs_access_token_security(path, method):
                operation["security"] = ACCESS_TOKEN_SECURITY


def _apply_servers(schema: dict) -> None:
    schema["servers"] = [
        {
            "url": "/",
            "description": "Прямой доступ к app (NodePort :30800 или внутри кластера)",
        },
        {
            "url": "https://adaptive-llm.ru",
            "description": "Внешний домен: inference — /v1/*, training — /api/* (см. описание)",
        },
    ]


def configure_openapi(app: FastAPI) -> None:
    def custom_openapi() -> dict:
        if app.openapi_schema:
            return app.openapi_schema

        schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
            tags=app.openapi_tags,
        )
        _apply_security(schema)
        _apply_servers(schema)
        app.openapi_schema = schema
        return schema

    app.openapi = custom_openapi  # type: ignore[method-assign]


def _swagger_html(openapi_url: str, title: str) -> HTMLResponse:
    return get_swagger_ui_html(
        openapi_url=openapi_url,
        title=title,
        swagger_ui_parameters={"docExpansion": "list", "defaultModelsExpandDepth": 1},
    )


def register_docs_routes(app: FastAPI) -> None:
    title = f"{app.title} — Swagger UI"

    @app.get("/openapi.json", include_in_schema=False)
    async def openapi_json() -> JSONResponse:
        return JSONResponse(app.openapi())

    @app.get("/docs", include_in_schema=False)
    async def swagger_docs() -> HTMLResponse:
        return _swagger_html("/openapi.json", title)

    @app.get("/api/openapi.json", include_in_schema=False)
    async def api_openapi_json() -> JSONResponse:
        return JSONResponse(app.openapi())

    @app.get("/api/docs", include_in_schema=False)
    async def api_swagger_docs() -> HTMLResponse:
        return _swagger_html("/api/openapi.json", title)

    # Fallback when Ingress rewrite sends /api/docs → /v1/training/docs
    @app.get("/v1/training/openapi.json", include_in_schema=False)
    async def training_openapi_json() -> JSONResponse:
        return JSONResponse(app.openapi())

    @app.get("/v1/training/docs", include_in_schema=False)
    async def training_swagger_docs() -> HTMLResponse:
        return _swagger_html("/v1/training/openapi.json", title)
