import time
from collections import OrderedDict
from dataclasses import dataclass


@dataclass(frozen=True)
class CompletionRecord:
    in_baseline: bool
    created_at: float


class CompletionRegistry:
    def __init__(self, *, max_size: int = 2000, ttl_sec: int = 3600) -> None:
        self.max_size = max_size
        self.ttl_sec = ttl_sec
        self._items: OrderedDict[str, CompletionRecord] = OrderedDict()

    def put(self, completion_id: str, record: CompletionRecord) -> None:
        self._purge_expired()
        self._items[completion_id] = record
        self._items.move_to_end(completion_id)
        while len(self._items) > self.max_size:
            self._items.popitem(last=False)

    def get(self, completion_id: str) -> CompletionRecord | None:
        self._purge_expired()
        record = self._items.get(completion_id)
        if record is None:
            return None
        if time.time() - record.created_at > self.ttl_sec:
            self._items.pop(completion_id, None)
            return None
        return record

    def _purge_expired(self) -> None:
        now = time.time()
        expired = [
            key
            for key, record in self._items.items()
            if now - record.created_at > self.ttl_sec
        ]
        for key in expired:
            self._items.pop(key, None)
