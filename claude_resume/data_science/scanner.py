"""Fast parallel session scanner — extracts metadata from all sessions.

Two tiers:
  - Light scan: file stats only (mtime, size, project). All 5000+ sessions in <1s.
  - Deep scan: parse JSONL for timestamps, tools, tokens, messages. Cached, incremental.
"""

import json
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from ..sessions import find_all_sessions, RESUME_CACHE_DIR

_DS_CACHE_DIR = RESUME_CACHE_DIR.parent / "ds-cache"
_DS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Cache file for aggregated scan results
_SCAN_CACHE = _DS_CACHE_DIR / "scan_cache.json"
_SCAN_TTL = 300  # 5 minutes


def scan_all_sessions() -> list[dict]:
    """Light scan: file stats from all sessions. Returns list of dicts with:
    session_id, project_dir, mtime, size, date, hour, weekday, project_short.
    """
    sessions = find_all_sessions()
    results = []
    for s in sessions:
        dt = datetime.fromtimestamp(s["mtime"])
        project = s["project_dir"]
        # Shorten project path
        home = str(Path.home())
        short = project.replace(home, "~") if project.startswith(home) else project
        # Extract just the repo name
        parts = short.split("/")
        repo = parts[-1] if len(parts) > 1 else short

        results.append({
            "session_id": s["session_id"],
            "project_dir": project,
            "project_short": short,
            "repo": repo,
            "mtime": s["mtime"],
            "size": s["size"],
            "date": dt.strftime("%Y-%m-%d"),
            "hour": dt.hour,
            "weekday": dt.strftime("%A"),
            "weekday_num": dt.weekday(),
            "month": dt.strftime("%Y-%m"),
            "file": str(s["file"]),
        })
    return results


def _cache_key_for_session(session_file: str, mtime: float) -> str:
    return hashlib.md5(f"{session_file}:{mtime}".encode()).hexdigest()[:12]


def _parse_single_session(s: dict) -> dict:
    """Deep parse a single session JSONL for analytics fields."""
    result = {
        "session_id": s["session_id"],
        "project_short": s.get("project_short", ""),
        "repo": s.get("repo", "?"),
        "size": s.get("size", 0),
        "mtime": s["mtime"],
        "date": s.get("date", ""),
        "hour": s.get("hour", 0),
        "weekday": s.get("weekday", ""),
        "weekday_num": s.get("weekday_num", 0),
        "month": s.get("month", s.get("date", "")[:7] if s.get("date") else ""),
    }

    timestamps = []
    user_msgs = 0
    assistant_msgs = 0
    tool_uses = {}
    models_used = {}
    input_tokens = 0
    output_tokens = 0
    cache_read_tokens = 0
    cache_write_tokens = 0
    git_branches = set()
    progress_count = 0

    try:
        with open(s["file"], "r", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if not isinstance(entry, dict):
                    continue

                entry_type = entry.get("type")

                # Extract timestamp
                ts = entry.get("timestamp")
                if ts and isinstance(ts, str):
                    try:
                        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        timestamps.append(dt.timestamp())
                    except (ValueError, TypeError):
                        pass

                if entry_type == "user":
                    user_msgs += 1
                    branch = entry.get("gitBranch")
                    if branch:
                        git_branches.add(branch)

                elif entry_type == "assistant":
                    assistant_msgs += 1
                    msg = entry.get("message", {})
                    if isinstance(msg, dict):
                        model = msg.get("model", "")
                        if model:
                            models_used[model] = models_used.get(model, 0) + 1
                        usage = msg.get("usage", {})
                        if isinstance(usage, dict):
                            input_tokens += usage.get("input_tokens", 0)
                            output_tokens += usage.get("output_tokens", 0)
                            cache_read_tokens += usage.get("cache_read_input_tokens", 0)
                            cache_write_tokens += usage.get("cache_creation_input_tokens", 0)
                        content = msg.get("content", [])
                        if isinstance(content, list):
                            for block in content:
                                if isinstance(block, dict) and block.get("type") == "tool_use":
                                    name = block.get("name", "unknown")
                                    tool_uses[name] = tool_uses.get(name, 0) + 1

                elif entry_type == "progress":
                    progress_count += 1

    except OSError:
        pass

    # Duration
    if len(timestamps) >= 2:
        duration_secs = max(timestamps) - min(timestamps)
        first_ts = min(timestamps)
        last_ts = max(timestamps)
    else:
        duration_secs = 0
        first_ts = s["mtime"]
        last_ts = s["mtime"]

    result.update({
        "duration_secs": duration_secs,
        "duration_mins": round(duration_secs / 60, 1),
        "first_ts": first_ts,
        "last_ts": last_ts,
        "user_msgs": user_msgs,
        "assistant_msgs": assistant_msgs,
        "total_msgs": user_msgs + assistant_msgs,
        "tool_uses": tool_uses,
        "tool_use_total": sum(tool_uses.values()),
        "models_used": models_used,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_tokens": cache_read_tokens,
        "cache_write_tokens": cache_write_tokens,
        "total_tokens": input_tokens + output_tokens,
        "git_branches": list(git_branches),
        "progress_count": progress_count,
        "subagent_heavy": progress_count > 100,
    })
    return result


def scan_history_jsonl() -> list[dict]:
    """Scan ~/.claude/history.jsonl for older sessions not in JSONL format.

    Groups prompts by (project, date) into synthetic session records.
    Returns sessions with the same schema as scan_deep results.
    """
    history_file = Path.home() / ".claude" / "history.jsonl"
    if not history_file.exists():
        return []

    from collections import defaultdict
    # Group by (project, date)
    groups = defaultdict(list)
    try:
        with open(history_file, "r", errors="replace") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ts = entry.get("timestamp", 0)
                if ts > 1_000_000_000_000:
                    ts = ts / 1000
                if ts == 0:
                    continue
                project = entry.get("project", "")
                dt = datetime.fromtimestamp(ts)
                key = (project, dt.strftime("%Y-%m-%d"))
                groups[key].append({
                    "ts": ts,
                    "hour": dt.hour,
                    "text": entry.get("display", ""),
                })
    except OSError:
        return []

    results = []
    home = str(Path.home())
    for (project, date), entries in groups.items():
        short = project.replace(home, "~") if project.startswith(home) else project
        parts = short.split("/")
        repo = parts[-1] if len(parts) > 1 else short

        timestamps = [e["ts"] for e in entries]
        first_ts = min(timestamps)
        last_ts = max(timestamps)
        dt = datetime.fromtimestamp(first_ts)

        # Estimate chars as rough token proxy (4 chars/token)
        total_chars = sum(len(e.get("text", "")) for e in entries)
        est_input_tokens = total_chars // 4
        est_output_tokens = est_input_tokens * 2  # rough 2x response

        results.append({
            "session_id": f"history-{date}-{repo}",
            "project_short": short,
            "project_dir": project,
            "repo": repo,
            "size": 0,
            "mtime": last_ts,
            "date": date,
            "hour": dt.hour,
            "weekday": dt.strftime("%A"),
            "weekday_num": dt.weekday(),
            "month": dt.strftime("%Y-%m"),
            "duration_secs": last_ts - first_ts,
            "duration_mins": round((last_ts - first_ts) / 60, 1),
            "first_ts": first_ts,
            "last_ts": last_ts,
            "user_msgs": len(entries),
            "assistant_msgs": len(entries),  # assume 1:1
            "total_msgs": len(entries) * 2,
            "tool_uses": {},
            "tool_use_total": 0,
            "models_used": {},
            "input_tokens": est_input_tokens,
            "output_tokens": est_output_tokens,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "total_tokens": est_input_tokens + est_output_tokens,
            "git_branches": [],
            "progress_count": 0,
            "subagent_heavy": False,
            "_source": "history",
        })

    return results


def scan_deep(
    max_sessions: int = 0,
    force: bool = False,
) -> list[dict]:
    """Deep scan sessions with incremental caching.

    Parses JSONL for timestamps, tools, tokens, messages.
    Results cached per-session by file+mtime key.
    """
    light = scan_all_sessions()

    # Load existing cache
    cache = {}
    if _SCAN_CACHE.exists() and not force:
        try:
            cache = json.loads(_SCAN_CACHE.read_text())
        except (json.JSONDecodeError, OSError):
            cache = {}

    to_parse = []
    cached_results = []
    for s in light:
        ck = _cache_key_for_session(s["file"], s["mtime"])
        if ck in cache:
            cached_results.append(cache[ck])
        else:
            s["_cache_key"] = ck
            to_parse.append(s)

    if max_sessions > 0:
        to_parse = to_parse[:max_sessions]

    # Parse uncached sessions in parallel
    if to_parse:
        with ThreadPoolExecutor(max_workers=8) as pool:
            new_results = list(pool.map(_parse_single_session, to_parse))

        for s, result in zip(to_parse, new_results):
            cache[s["_cache_key"]] = result
            cached_results.append(result)

        # Save updated cache
        try:
            _SCAN_CACHE.write_text(json.dumps(cache))
        except OSError:
            pass

    # Merge in history.jsonl data for dates before JSONL era
    jsonl_dates = set(r.get("date", "") for r in cached_results)
    earliest_jsonl = min(jsonl_dates) if jsonl_dates else "9999-99-99"

    history_sessions = scan_history_jsonl()
    # Only include history sessions from before the JSONL era
    legacy = [s for s in history_sessions if s["date"] < earliest_jsonl]
    cached_results.extend(legacy)

    return cached_results
