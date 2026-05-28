import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


class AuthDatabase:
    def __init__(self, path: str) -> None:
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    active INTEGER NOT NULL DEFAULT 1,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS api_keys (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    key_hash TEXT NOT NULL UNIQUE,
                    scopes TEXT NOT NULL,
                    active INTEGER NOT NULL DEFAULT 1,
                    created_at REAL NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                );
                """)

    def create_user(self, *, username: str, password_hash: str) -> dict[str, Any]:
        now = time.time()
        with self.connect() as conn:
            cur = conn.execute(
                "INSERT INTO users (username, password_hash, active, created_at) VALUES (?, ?, 1, ?)",
                (username, password_hash, now),
            )
            user_id = int(cur.lastrowid)
        return self.get_user(user_id)

    def list_users(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT id, username, active, created_at FROM users ORDER BY id"
            ).fetchall()
        return [self._user_row(row) for row in rows]

    def get_user(self, user_id: int) -> dict[str, Any]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT id, username, active, created_at FROM users WHERE id = ?",
                (user_id,),
            ).fetchone()
        if row is None:
            raise KeyError(user_id)
        return self._user_row(row)

    def get_user_by_username(self, username: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT id, username, password_hash, active, created_at FROM users WHERE username = ?",
                (username,),
            ).fetchone()
        if row is None:
            return None
        return {
            **self._user_row(row),
            "password_hash": row["password_hash"],
        }

    def update_user(
        self,
        user_id: int,
        *,
        username: str | None = None,
        password_hash: str | None = None,
        active: bool | None = None,
    ) -> dict[str, Any]:
        fields: list[str] = []
        values: list[Any] = []
        if username is not None:
            fields.append("username = ?")
            values.append(username)
        if password_hash is not None:
            fields.append("password_hash = ?")
            values.append(password_hash)
        if active is not None:
            fields.append("active = ?")
            values.append(1 if active else 0)
        if not fields:
            return self.get_user(user_id)
        values.append(user_id)
        with self.connect() as conn:
            conn.execute(
                f"UPDATE users SET {', '.join(fields)} WHERE id = ?",
                values,
            )
        return self.get_user(user_id)

    def delete_user(self, user_id: int) -> None:
        with self.connect() as conn:
            cur = conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        if cur.rowcount == 0:
            raise KeyError(user_id)

    def create_api_key(
        self,
        *,
        user_id: int,
        name: str,
        key_hash: str,
        scopes: list[str],
    ) -> dict[str, Any]:
        now = time.time()
        with self.connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO api_keys (user_id, name, key_hash, scopes, active, created_at)
                VALUES (?, ?, ?, ?, 1, ?)
                """,
                (user_id, name, key_hash, json.dumps(scopes), now),
            )
            key_id = int(cur.lastrowid)
        return self.get_api_key(key_id)

    def list_api_keys(self, *, user_id: int | None = None) -> list[dict[str, Any]]:
        query = """
            SELECT k.id, k.user_id, u.username, k.name, k.scopes, k.active, k.created_at
            FROM api_keys k
            JOIN users u ON u.id = k.user_id
        """
        params: tuple[Any, ...] = ()
        if user_id is not None:
            query += " WHERE k.user_id = ?"
            params = (user_id,)
        query += " ORDER BY k.id DESC"
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._api_key_row(row) for row in rows]

    def get_api_key(self, key_id: int) -> dict[str, Any]:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT k.id, k.user_id, u.username, k.name, k.scopes, k.active, k.created_at
                FROM api_keys k
                JOIN users u ON u.id = k.user_id
                WHERE k.id = ?
                """,
                (key_id,),
            ).fetchone()
        if row is None:
            raise KeyError(key_id)
        return self._api_key_row(row)

    def find_api_key_by_hash(self, key_hash: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT k.id, k.user_id, u.username, k.name, k.scopes, k.active, k.created_at
                FROM api_keys k
                JOIN users u ON u.id = k.user_id
                WHERE k.key_hash = ?
                """,
                (key_hash,),
            ).fetchone()
        if row is None:
            return None
        return self._api_key_row(row)

    def update_api_key(
        self,
        key_id: int,
        *,
        name: str | None = None,
        scopes: list[str] | None = None,
        active: bool | None = None,
    ) -> dict[str, Any]:
        fields: list[str] = []
        values: list[Any] = []
        if name is not None:
            fields.append("name = ?")
            values.append(name)
        if scopes is not None:
            fields.append("scopes = ?")
            values.append(json.dumps(scopes))
        if active is not None:
            fields.append("active = ?")
            values.append(1 if active else 0)
        if not fields:
            return self.get_api_key(key_id)
        values.append(key_id)
        with self.connect() as conn:
            conn.execute(
                f"UPDATE api_keys SET {', '.join(fields)} WHERE id = ?",
                values,
            )
        return self.get_api_key(key_id)

    def delete_api_key(self, key_id: int) -> None:
        with self.connect() as conn:
            cur = conn.execute("DELETE FROM api_keys WHERE id = ?", (key_id,))
        if cur.rowcount == 0:
            raise KeyError(key_id)

    @staticmethod
    def _user_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "username": row["username"],
            "active": bool(row["active"]),
            "created_at": row["created_at"],
        }

    @staticmethod
    def _api_key_row(row: sqlite3.Row) -> dict[str, Any]:
        scopes = json.loads(row["scopes"])
        return {
            "id": row["id"],
            "user_id": row["user_id"],
            "username": row["username"],
            "name": row["name"],
            "scopes": scopes,
            "active": bool(row["active"]),
            "created_at": row["created_at"],
        }
