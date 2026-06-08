from typing import Annotated, Any

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from auth_service.config import Settings
from auth_service.database import AuthDatabase
from auth_service.security import (
    generate_token,
    hash_password,
    hash_token,
    verify_password,
)

router = APIRouter(prefix="/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str
    password: str


class VerifyRequest(BaseModel):
    scope: str | None = None


class UserCreateRequest(BaseModel):
    username: str = Field(min_length=2, max_length=64)
    password: str = Field(min_length=6, max_length=128)
    active: bool = True


class UserUpdateRequest(BaseModel):
    username: str | None = Field(default=None, min_length=2, max_length=64)
    password: str | None = Field(default=None, min_length=6, max_length=128)
    active: bool | None = None


class ApiKeyCreateRequest(BaseModel):
    user_id: int
    name: str = Field(min_length=1, max_length=128)
    scopes: list[str] = Field(default_factory=lambda: ["inference"])


class ApiKeyUpdateRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=128)
    scopes: list[str] | None = None
    active: bool | None = None


def get_db(request: Request) -> AuthDatabase:
    return request.app.state.db


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def extract_bearer_token(request: Request) -> str | None:
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    for header in ("X-API-Key", "X-Access-Token"):
        alt = request.headers.get(header)
        if alt:
            return alt.strip()
    return None


def resolve_auth_context(
    request: Request,
    db: AuthDatabase,
) -> dict[str, Any]:
    token = extract_bearer_token(request)
    if not token:
        raise HTTPException(status_code=401, detail="Missing bearer token")
    record = db.find_api_key_by_hash(hash_token(token))
    if record is None or not record["active"]:
        raise HTTPException(status_code=401, detail="Invalid or inactive token")
    user = db.get_user(record["user_id"])
    if not user["active"]:
        raise HTTPException(status_code=401, detail="User is inactive")
    return {
        "token": token,
        "user": user,
        "api_key": record,
        "scopes": record["scopes"],
    }


def require_scope(scope: str):
    def dependency(
        request: Request,
        db: Annotated[AuthDatabase, Depends(get_db)],
    ) -> dict[str, Any]:
        ctx = resolve_auth_context(request, db)
        if scope not in ctx["scopes"] and "admin" not in ctx["scopes"]:
            raise HTTPException(status_code=403, detail=f"Scope '{scope}' required")
        return ctx

    return dependency


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/login")
async def login(
    req: LoginRequest,
    request: Request,
    db: Annotated[AuthDatabase, Depends(get_db)],
) -> dict[str, Any]:
    settings: Settings = request.app.state.settings
    user = db.get_user_by_username(req.username.strip())
    if user is None or not user["active"]:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not verify_password(req.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    raw_token = generate_token(settings.token_prefix)
    api_key = db.create_api_key(
        user_id=user["id"],
        name="login-session",
        key_hash=hash_token(raw_token),
        scopes=["inference", "admin"],
    )
    return {
        "token": raw_token,
        "token_type": "bearer",
        "user": db.get_user(user["id"]),
        "scopes": api_key["scopes"],
        "api_key_id": api_key["id"],
    }


@router.post("/verify")
async def verify_token(
    request: Request,
    db: Annotated[AuthDatabase, Depends(get_db)],
    req: VerifyRequest | None = None,
) -> dict[str, Any]:
    ctx = resolve_auth_context(request, db)
    required = req.scope if req else None
    if required and required not in ctx["scopes"] and "admin" not in ctx["scopes"]:
        return {"valid": False, "reason": f"missing scope: {required}"}
    return {
        "valid": True,
        "user_id": ctx["user"]["id"],
        "username": ctx["user"]["username"],
        "scopes": ctx["scopes"],
        "api_key_id": ctx["api_key"]["id"],
    }


@router.get("/me")
async def me(
    ctx: Annotated[dict[str, Any], Depends(require_scope("inference"))],
) -> dict[str, Any]:
    return {
        "user": ctx["user"],
        "scopes": ctx["scopes"],
        "api_key": {
            "id": ctx["api_key"]["id"],
            "name": ctx["api_key"]["name"],
        },
    }


users_router = APIRouter(prefix="/users", tags=["users"])


@users_router.get("")
async def list_users(
    _: Annotated[dict[str, Any], Depends(require_scope("admin"))],
    db: Annotated[AuthDatabase, Depends(get_db)],
) -> dict[str, Any]:
    users = db.list_users()
    return {"count": len(users), "users": users}


@users_router.post("")
async def create_user(
    req: UserCreateRequest,
    _: Annotated[dict[str, Any], Depends(require_scope("admin"))],
    db: Annotated[AuthDatabase, Depends(get_db)],
) -> dict[str, Any]:
    if db.get_user_by_username(req.username):
        raise HTTPException(status_code=409, detail="Username already exists")
    user = db.create_user(
        username=req.username.strip(),
        password_hash=hash_password(req.password),
    )
    if not req.active:
        user = db.update_user(user["id"], active=False)
    return user


@users_router.get("/{user_id}")
async def get_user(
    user_id: int,
    _: Annotated[dict[str, Any], Depends(require_scope("admin"))],
    db: Annotated[AuthDatabase, Depends(get_db)],
) -> dict[str, Any]:
    try:
        return db.get_user(user_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="User not found") from exc


@users_router.patch("/{user_id}")
async def update_user(
    user_id: int,
    req: UserUpdateRequest,
    _: Annotated[dict[str, Any], Depends(require_scope("admin"))],
    db: Annotated[AuthDatabase, Depends(get_db)],
) -> dict[str, Any]:
    try:
        password_hash = hash_password(req.password) if req.password else None
        return db.update_user(
            user_id,
            username=req.username.strip() if req.username else None,
            password_hash=password_hash,
            active=req.active,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="User not found") from exc


@users_router.delete("/{user_id}")
async def delete_user(
    user_id: int,
    ctx: Annotated[dict[str, Any], Depends(require_scope("admin"))],
    db: Annotated[AuthDatabase, Depends(get_db)],
) -> dict[str, str]:
    if ctx["user"]["id"] == user_id:
        raise HTTPException(status_code=400, detail="Cannot delete current user")
    try:
        db.delete_user(user_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="User not found") from exc
    return {"status": "deleted", "id": str(user_id)}


keys_router = APIRouter(prefix="/api-keys", tags=["api-keys"])


@keys_router.get("")
async def list_api_keys(
    _: Annotated[dict[str, Any], Depends(require_scope("admin"))],
    db: Annotated[AuthDatabase, Depends(get_db)],
) -> dict[str, Any]:
    keys = db.list_api_keys()
    return {"count": len(keys), "api_keys": keys}


@keys_router.post("")
async def create_api_key(
    req: ApiKeyCreateRequest,
    _: Annotated[dict[str, Any], Depends(require_scope("admin"))],
    request: Request,
    db: Annotated[AuthDatabase, Depends(get_db)],
) -> dict[str, Any]:
    settings: Settings = request.app.state.settings
    try:
        db.get_user(req.user_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="User not found") from exc
    allowed = {"inference", "admin"}
    scopes = [scope for scope in req.scopes if scope in allowed]
    if not scopes:
        raise HTTPException(status_code=400, detail="At least one valid scope required")
    raw_token = generate_token(settings.token_prefix)
    record = db.create_api_key(
        user_id=req.user_id,
        name=req.name.strip(),
        key_hash=hash_token(raw_token),
        scopes=scopes,
    )
    return {"token": raw_token, "api_key": record}


@keys_router.patch("/{key_id}")
async def update_api_key(
    key_id: int,
    req: ApiKeyUpdateRequest,
    _: Annotated[dict[str, Any], Depends(require_scope("admin"))],
    db: Annotated[AuthDatabase, Depends(get_db)],
) -> dict[str, Any]:
    try:
        return db.update_api_key(
            key_id,
            name=req.name.strip() if req.name else None,
            scopes=req.scopes,
            active=req.active,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="API key not found") from exc


@keys_router.delete("/{key_id}")
async def delete_api_key(
    key_id: int,
    _: Annotated[dict[str, Any], Depends(require_scope("admin"))],
    db: Annotated[AuthDatabase, Depends(get_db)],
) -> dict[str, str]:
    try:
        db.delete_api_key(key_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="API key not found") from exc
    return {"status": "deleted", "id": str(key_id)}


def bootstrap_admin(db: AuthDatabase, settings: Settings) -> None:
    if db.list_users():
        return
    user = db.create_user(
        username=settings.bootstrap_username,
        password_hash=hash_password(settings.bootstrap_password),
    )
    raw_token = generate_token(settings.token_prefix)
    db.create_api_key(
        user_id=user["id"],
        name="bootstrap-admin",
        key_hash=hash_token(raw_token),
        scopes=["admin", "inference"],
    )
    print(
        f"[auth-service] bootstrap admin user={settings.bootstrap_username!r} "
        f"(change AUTH_BOOTSTRAP_PASSWORD and create dedicated keys)"
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    db = AuthDatabase(settings.database_path)
    bootstrap_admin(db, settings)

    app = FastAPI(title="MLOps Auth Service", version="0.1.0")
    app.state.db = db
    app.state.settings = settings
    app.include_router(router)
    app.include_router(users_router)
    app.include_router(keys_router)
    return app
