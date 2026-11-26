"""Token budgets and query rate limits.

Tokens are counted per index, exactly: each index has its own pipeline and
Claude client, so the client's running total is the index's spend, even with
concurrent requests. Counted: uncached input, cache writes and output; cache
reads are not counted (they cost a small fraction of input).

Queries are counted per user and index: per day in the usage file (survives
restarts), per minute in memory.
"""

import json
import os
import tempfile
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

from ..models import TokenUsage


def billed_tokens(usage: TokenUsage) -> int:
    return usage.input_tokens + usage.cache_write_tokens + usage.output_tokens


class LimitExceeded(Exception):
    def __init__(self, message: str, retry_after: int):
        super().__init__(message)
        self.retry_after = retry_after


class UsageStore:
    def __init__(self, path: Path | None):
        self.path = path
        self._lock = threading.Lock()
        self._seen: dict[str, int] = {}  # index -> client total already recorded
        self._recent: dict[tuple[str, str], deque] = defaultdict(deque)
        self.data = {"tokens": {}, "queries": {}}
        if path and path.exists():
            self.data = json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _month() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m")

    @staticmethod
    def _day() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".usage-", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)
        os.replace(tmp, self.path)

    def tokens_this_month(self, index: str) -> int:
        return self.data["tokens"].get(self._month(), {}).get(index, 0)

    def queries_today(self, index: str, user: str) -> int:
        return self.data["queries"].get(self._day(), {}).get(index, {}).get(user, 0)

    def record_tokens(self, index: str, usage: TokenUsage) -> int:
        """Add what the index's client spent since the last call; returns that amount."""
        with self._lock:
            total = billed_tokens(usage)
            delta = total - self._seen.get(index, 0)
            self._seen[index] = total
            if delta > 0:
                month = self.data["tokens"].setdefault(self._month(), {})
                month[index] = month.get(index, 0) + delta
                self._save()
            return max(delta, 0)

    def check_tokens(self, index: str, monthly_limit: int | None) -> None:
        if monthly_limit is not None and self.tokens_this_month(index) >= monthly_limit:
            now = datetime.now(timezone.utc)
            first_next = datetime(now.year + (now.month == 12), now.month % 12 + 1, 1, tzinfo=timezone.utc)
            raise LimitExceeded("This index has used its token budget for the month",
                                int((first_next - now).total_seconds()))

    def admit_query(self, index: str, user: str, per_minute: int | None, per_day: int | None) -> None:
        """Count a query, or raise LimitExceeded without counting it."""
        with self._lock:
            now = time.monotonic()
            recent = self._recent[(index, user)]
            while recent and now - recent[0] > 60:
                recent.popleft()
            if per_minute is not None and len(recent) >= per_minute:
                raise LimitExceeded("Too many queries; try again in a minute", int(61 - (now - recent[0])))
            if per_day is not None and self.queries_today(index, user) >= per_day:
                raise LimitExceeded("Daily query limit reached", _seconds_to_midnight())
            recent.append(now)
            day = self.data["queries"].setdefault(self._day(), {}).setdefault(index, {})
            day[user] = day.get(user, 0) + 1
            # Keep only the last 7 days of query counts.
            for old in sorted(self.data["queries"])[:-7]:
                del self.data["queries"][old]
            self._save()


def _seconds_to_midnight() -> int:
    now = datetime.now(timezone.utc)
    return 86400 - (now.hour * 3600 + now.minute * 60 + now.second)
