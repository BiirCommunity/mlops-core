import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    database_path: str
    bootstrap_username: str
    bootstrap_password: str
    token_prefix: str
    host: str
    port: int

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            database_path=os.environ.get("AUTH_DATABASE_PATH", "/data/auth.db"),
            bootstrap_username=os.environ.get("AUTH_BOOTSTRAP_USERNAME", "admin"),
            bootstrap_password=os.environ.get("AUTH_BOOTSTRAP_PASSWORD", "changeme"),
            token_prefix=os.environ.get("AUTH_TOKEN_PREFIX", "mlops_"),
            host=os.environ.get("AUTH_HOST", "0.0.0.0"),
            port=int(os.environ.get("AUTH_PORT", "8090")),
        )
