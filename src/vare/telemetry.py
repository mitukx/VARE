from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any


class EventLog:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self.counters: Counter[str] = Counter()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: str, **payload: Any) -> None:
        self.counters[event] += 1
        if not self.path:
            return
        row = {"ts": time.time(), "event": event, **payload}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
