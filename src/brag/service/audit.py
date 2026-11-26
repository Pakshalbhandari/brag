"""Append-only audit log: one JSON object per line.

Records who did what to which index: queries (with the question and the
documents the answer was built from), uploads, removals and denied requests.
"""

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class AuditLog:
    def __init__(self, path: str | Path | None, include_questions: bool = True):
        self.path = Path(path) if path else None
        self.include_questions = include_questions
        self._lock = threading.Lock()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: str, **fields: Any) -> None:
        if not self.path:
            return
        if not self.include_questions:
            fields.pop("question", None)
        record = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), "event": event, **fields}
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
