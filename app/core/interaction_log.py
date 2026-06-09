import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any


@dataclass
class InteractionRecord:
    id: str
    created_at: float
    prompt: str
    response: str
    prompt_lang: str
    response_lang: str
    toxicity: float
    json_valid: bool
    status: str
    anomaly_flags: list[str] = field(default_factory=list)
    session_id: str | None = None
    conversation_id: str | None = None
    user_rating: int | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = "local"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class InteractionLog:
    def __init__(
        self,
        *,
        log_dir: str | Path | None = None,
        max_memory: int = 5000,
    ) -> None:
        default_dir = os.environ.get("INTERACTION_LOG_DIR", "/app/reports/interactions")
        self.log_dir = Path(log_dir or default_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / "interactions.jsonl"
        self.max_memory = max_memory
        self._lock = Lock()
        self._records: list[InteractionRecord] = []
        self._index: dict[str, InteractionRecord] = {}
        self._load_existing()

    def _load_existing(self) -> None:
        if not self.log_path.exists():
            return
        rows: list[InteractionRecord] = []
        for line in self.log_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if payload.get("event") == "rating_update":
                continue
            if "prompt" not in payload or "response" not in payload:
                continue
            record = InteractionRecord(**payload)
            rows.append(record)
        if len(rows) > self.max_memory:
            rows = rows[-self.max_memory :]
        self._records = rows
        self._index = {record.id: record for record in rows}

    def append(self, record: InteractionRecord) -> InteractionRecord:
        with self._lock:
            self._records.append(record)
            self._index[record.id] = record
            if len(self._records) > self.max_memory:
                dropped = self._records.pop(0)
                self._index.pop(dropped.id, None)
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        return record

    def create(
        self,
        *,
        prompt: str,
        response: str,
        prompt_lang: str,
        response_lang: str,
        toxicity: float,
        json_valid: bool,
        status: str,
        anomaly_flags: list[str],
        session_id: str | None = None,
        conversation_id: str | None = None,
        user_rating: int | None = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        model: str = "local",
        completion_id: str | None = None,
    ) -> InteractionRecord:
        record = InteractionRecord(
            id=completion_id or f"chatcmpl-{uuid.uuid4().hex}",
            created_at=time.time(),
            prompt=prompt,
            response=response,
            prompt_lang=prompt_lang,
            response_lang=response_lang,
            toxicity=toxicity,
            json_valid=json_valid,
            status=status,
            anomaly_flags=anomaly_flags,
            session_id=session_id,
            conversation_id=conversation_id,
            user_rating=user_rating,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model=model,
        )
        return self.append(record)

    def update_rating(
        self, completion_id: str, rating: int
    ) -> InteractionRecord | None:
        with self._lock:
            record = self._index.get(completion_id)
            if record is None:
                return None
            updated = InteractionRecord(
                **{
                    **record.to_dict(),
                    "user_rating": rating,
                }
            )
            if rating <= 2 and "low_rating" not in updated.anomaly_flags:
                updated.anomaly_flags = [*updated.anomaly_flags, "low_rating"]
            self._index[completion_id] = updated
            for idx, item in enumerate(self._records):
                if item.id == completion_id:
                    self._records[idx] = updated
                    break
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "event": "rating_update",
                            "id": completion_id,
                            "rating": rating,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            return updated

    def list_records(
        self,
        *,
        limit: int = 100,
        anomalies_only: bool = False,
        conversation_id: str | None = None,
    ) -> list[InteractionRecord]:
        with self._lock:
            rows = list(reversed(self._records))
        if conversation_id:
            rows = [
                row
                for row in rows
                if row.conversation_id == conversation_id
                or row.session_id == conversation_id
            ]
        if anomalies_only:
            rows = [row for row in rows if row.anomaly_flags]
        return rows[:limit]

    def get(self, record_id: str) -> InteractionRecord | None:
        with self._lock:
            return self._index.get(record_id)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            total = len(self._records)
            anomalous = sum(1 for row in self._records if row.anomaly_flags)
            flagged: dict[str, int] = {}
            for row in self._records:
                for flag in row.anomaly_flags:
                    flagged[flag] = flagged.get(flag, 0) + 1
        return {
            "total": total,
            "anomalous": anomalous,
            "flags": flagged,
        }
