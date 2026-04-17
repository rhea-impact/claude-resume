"""Shared session utility functions.

Small helpers used by both mcp_server.py and self_tools.py.
Extracted to keep mcp_server.py under 2000 lines.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def filter_automated(sessions: list[dict], cache_index: dict) -> list[dict]:
    """Remove sessions classified as 'automated' by the ML classifier.

    Shared helper — used by search_sessions, recent_sessions, what_changed,
    my_week, healthy_sessions, and suggest_next. Extracted to avoid 6
    copies of the same filter pattern.
    """
    return [
        s for s in sessions
        if cache_index.get(s.get("session_id", ""), {}).get("classification") != "automated"
    ]


def session_duration_hours(f: Path) -> float:
    """Estimate session duration. Prefers file birthtime (measures current
    file lifespan, conservative). Falls back to JSONL first→last timestamps
    when birthtime is unavailable. Capped at 24h — sessions left open for
    days shouldn't count full idle time as work.
    """
    try:
        stat = f.stat()
        try:
            birth = stat.st_birthtime
            delta = stat.st_mtime - birth
            if delta > 60:
                return min(delta / 3600, 24.0)
        except AttributeError:
            pass

        size = stat.st_size
        if size < 100:
            return 0.0
        first_ts = None
        with open(f, "rb") as fh:
            for _ in range(20):
                line = fh.readline()
                if not line:
                    break
                try:
                    entry = json.loads(line.decode("utf-8", errors="replace"))
                    ts = entry.get("timestamp")
                    if ts:
                        first_ts = ts
                        break
                except (json.JSONDecodeError, ValueError):
                    pass
        last_ts = None
        if first_ts:
            with open(f, "rb") as fh:
                fh.seek(max(0, size - 2048))
                tail = fh.read().decode("utf-8", errors="replace")
                for line in reversed(tail.splitlines()):
                    if not line.strip():
                        continue
                    try:
                        entry = json.loads(line.strip())
                        ts = entry.get("timestamp")
                        if ts:
                            last_ts = ts
                            break
                    except (json.JSONDecodeError, ValueError):
                        continue
        if first_ts and last_ts:
            t0 = datetime.fromisoformat(str(first_ts).replace("Z", "+00:00"))
            t1 = datetime.fromisoformat(str(last_ts).replace("Z", "+00:00"))
            delta = (t1 - t0).total_seconds()
            if delta > 60:
                return min(delta / 3600, 24.0)
    except (OSError, ValueError, TypeError):
        pass
    return 0.0
