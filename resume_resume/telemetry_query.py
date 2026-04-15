"""Read-side helpers for resume-resume telemetry.

The capture layer (telemetry.py) writes one JSONL line per MCP tool call.
This module reads those files back and produces the aggregations that power
the self_* MCP tools.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

from .bm25 import tokenize
from .telemetry import telemetry_root


def _day_files(root: Path, days: int) -> list[Path]:
    """Return telemetry file paths for the last `days` days (today first)."""
    today = datetime.now(timezone.utc).date()
    out = []
    for i in range(days):
        d = today - timedelta(days=i)
        p = root / f"{d.isoformat()}.jsonl"
        if p.exists():
            out.append(p)
    return out


def iter_events(days: int = 30, root: Path | None = None) -> Iterator[dict]:
    """Yield events from the last N days. Newest file first, but events
    within a file are chronological (write order)."""
    root = root or telemetry_root()
    if not root.exists():
        return
    for path in _day_files(root, days):
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
        except OSError:
            continue


def load_events(
    days: int = 30,
    *,
    tool: str | None = None,
    status: str | None = None,
    root: Path | None = None,
) -> list[dict]:
    out = []
    for e in iter_events(days=days, root=root):
        if tool and e.get("tool") != tool:
            continue
        if status and e.get("status") != status:
            continue
        out.append(e)
    return out


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] * (c - k) + s[c] * (k - f)


def usage_summary(events: list[dict]) -> list[dict]:
    """Per-tool counts, error rate, avg/p50/p95 duration. Sorted by count desc."""
    by_tool: dict[str, list[dict]] = defaultdict(list)
    for e in events:
        if tool := e.get("tool"):
            by_tool[tool].append(e)

    rows = []
    for tool, calls in by_tool.items():
        durs = [c.get("duration_ms", 0) for c in calls]
        errs = sum(1 for c in calls if c.get("status") == "error")
        rows.append({
            "tool": tool,
            "count": len(calls),
            "errors": errs,
            "error_rate": round(errs / len(calls), 4),
            "avg_ms": round(sum(durs) / len(durs), 2),
            "p50_ms": round(_percentile(durs, 0.50), 2),
            "p95_ms": round(_percentile(durs, 0.95), 2),
            "max_ms": round(max(durs), 2),
        })
    rows.sort(key=lambda r: r["count"], reverse=True)
    return rows


def dead_tools(summary: list[dict], threshold: int) -> list[dict]:
    return [r for r in summary if r["count"] <= threshold]


def slow_tools(summary: list[dict], p95_threshold_ms: float) -> list[dict]:
    return [r for r in summary if r["p95_ms"] >= p95_threshold_ms]


def error_prone_tools(summary: list[dict], min_rate: float, min_calls: int = 3) -> list[dict]:
    return [r for r in summary if r["error_rate"] >= min_rate and r["count"] >= min_calls]


def abandoned_queries(events: list[dict]) -> list[dict]:
    """Find searches/lookups that returned empty results.

    Heuristic: result is None, [] / {}, or result_size <= 2 (empty list/dict bytes).
    Returns the full events so the caller sees what was searched for.
    """
    out = []
    for e in events:
        if e.get("status") != "ok":
            continue
        tool = e.get("tool") or ""
        if "search" not in tool and "recent" not in tool and "find" not in tool:
            continue
        size = e.get("result_size") or 0
        result = e.get("result")
        empty = (
            size <= 2
            or result in (None, [], {}, "")
            or (isinstance(result, list) and len(result) == 0)
            or (isinstance(result, dict) and not result)
        )
        if empty:
            out.append(e)
    return out


def session_bundles(events: list[dict], gap_seconds: float = 30.0) -> list[dict]:
    """Group contiguous same-session calls into bundles.

    A bundle is a burst of calls on the same session_id with <gap_seconds
    between consecutive calls. This is the read-time derivation of the
    'session work unit' — computed from flat events, not stamped on write.
    """
    by_session: dict[str, list[dict]] = defaultdict(list)
    for e in events:
        sid = e.get("session_id") or f"pid-{e.get('pid')}"
        by_session[sid].append(e)

    bundles = []
    for sid, calls in by_session.items():
        calls.sort(key=lambda c: c.get("ts") or "")
        current: list[dict] = []
        last_end: datetime | None = None

        def flush():
            if not current:
                return
            starts: list[datetime] = []
            for c in current:
                t = _parse_ts(c.get("ts"))
                if t is not None:
                    starts.append(t)
            if not starts:
                return
            start = min(starts)
            durs = [c.get("duration_ms", 0) for c in current]
            bundles.append({
                "session_id": sid,
                "start": start.isoformat(),
                "call_count": len(current),
                "tools": sorted({c["tool"] for c in current if c.get("tool")}),
                "duration_ms": round(sum(durs), 2),
                "had_error": any(c.get("status") == "error" for c in current),
            })

        for c in calls:
            ts = _parse_ts(c.get("ts"))
            if ts is None:
                continue
            if last_end is not None and (ts - last_end).total_seconds() > gap_seconds:
                flush()
                current = []
            current.append(c)
            last_end = ts + timedelta(milliseconds=c.get("duration_ms", 0))
        flush()

    bundles.sort(key=lambda b: b["start"], reverse=True)
    return bundles


def _parse_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def _event_text(e: dict) -> str:
    """Flatten an event into a searchable text blob."""
    parts = [e.get("tool") or ""]
    for k in ("args", "result"):
        v = e.get(k)
        if v is not None:
            try:
                parts.append(json.dumps(v, default=str))
            except (TypeError, ValueError):
                parts.append(str(v))
    if em := e.get("error_msg"):
        parts.append(em)
    return " ".join(parts)


def bm25_search(events: list[dict], query: str, limit: int = 20) -> list[dict]:
    """Rank events against query using BM25 over a text rendering.

    Reuses `bm25.tokenize` for consistent tokenization with session search.
    Keeps a self-contained BM25 scorer here since the session-oriented
    `score_session` signature doesn't fit flat telemetry events.
    """
    q_tokens = tokenize(query)
    if not q_tokens or not events:
        return []

    k1, b = 1.5, 0.75

    docs = [(e, tokenize(_event_text(e))) for e in events]
    docs = [(e, toks) for e, toks in docs if toks]
    if not docs:
        return []

    n_docs = len(docs)
    avg_dl = sum(len(t) for _, t in docs) / n_docs

    df: Counter[str] = Counter()
    for _, toks in docs:
        for t in set(toks):
            df[t] += 1

    def idf(term: str) -> float:
        d = df.get(term, 0)
        if d == 0:
            return 0.0
        return math.log((n_docs - d + 0.5) / (d + 0.5) + 1.0)

    scored: list[tuple[float, dict]] = []
    for event, toks in docs:
        dl = len(toks)
        tf: Counter[str] = Counter(toks)
        score = 0.0
        for qt in q_tokens:
            f = tf.get(qt, 0)
            if f == 0:
                continue
            num = f * (k1 + 1)
            den = f + k1 * (1 - b + b * dl / avg_dl)
            score += idf(qt) * num / den
        if score > 0:
            scored.append((score, event))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [{"score": round(s, 3), **e} for s, e in scored[:limit]]


def insights_report(days: int = 30, root: Path | None = None) -> dict:
    """Opinionated product-learning report."""
    events = load_events(days=days, root=root)
    summary = usage_summary(events)
    total = len(events)
    errors = sum(1 for e in events if e.get("status") == "error")

    return {
        "days": days,
        "total_calls": total,
        "total_errors": errors,
        "overall_error_rate": round(errors / total, 4) if total else 0.0,
        "distinct_tools": len(summary),
        "usage": summary,
        "dead_tools": dead_tools(summary, threshold=max(1, total // 500)),
        "slow_tools": slow_tools(summary, p95_threshold_ms=1000),
        "error_prone_tools": error_prone_tools(summary, min_rate=0.05),
        "abandoned_queries": [
            {"ts": e.get("ts"), "tool": e.get("tool"), "args": e.get("args")}
            for e in abandoned_queries(events)[:20]
        ],
    }
