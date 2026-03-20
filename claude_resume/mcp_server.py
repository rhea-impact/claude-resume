"""MCP server exposing claude-resume session search and reading tools.

Design: minimize tokens returned. Claude can construct 'claude --resume {id}'
itself — don't waste tokens repeating it. Return the minimum needed to answer
the user's question in one tool call when possible.
"""

import json
import math
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from .sessions import (
    SessionCache,
    find_all_sessions,
    find_recent_sessions,
    parse_session,
    get_git_context,
    shorten_path,
    PROJECTS_DIR,
)
from claude_session_commons import decode_project_path
from .summarize import summarize_quick, summarize_deep, summarize_insight, auto_tier

mcp = FastMCP("claude-resume")

_cache = SessionCache()

_TRUNC = 300  # max chars per message/field
_UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def _summary_valid(summary: dict) -> bool:
    """Reject cached summaries that are garbage (XML blobs, fragments, etc.)."""
    if not isinstance(summary, dict):
        return False
    title = summary.get("title", "")
    state = summary.get("state", "")
    # Reject if title looks like XML/HTML (task notifications leak through)
    for field in (title, state):
        if isinstance(field, str) and ("<" in field and ">" in field):
            return False
    # Reject if title is too short to be useful (random fragments)
    if isinstance(title, str) and 0 < len(title) < 20 and not summary.get("goal") and not summary.get("what_was_done"):
        return False
    # Reject if no meaningful content at all
    has_content = any(
        summary.get(k) for k in ("title", "goal", "what_was_done", "state", "files")
    )
    return has_content


def _find_session(session_id: str) -> dict | None:
    """Find a session by targeted glob — O(1) dirs, not O(N) sessions."""
    # Validate UUID format to prevent glob injection (* ? [] etc)
    if not _UUID_RE.fullmatch(session_id):
        return None
    matches = list(PROJECTS_DIR.glob(f"*/{session_id}.jsonl"))
    if not matches:
        return None
    f = matches[0]
    stat = f.stat()
    return {
        "file": f,
        "session_id": session_id,
        "project_dir": decode_project_path(f.parent.name),
        "mtime": stat.st_mtime,
        "size": stat.st_size,
    }


def _trunc(text: str, limit: int = _TRUNC) -> str:
    return text if len(text) <= limit else text[:limit] + "..."


def _daemon_alive() -> bool:
    pid_file = Path.home() / ".claude" / "session-daemon.pid"
    try:
        if not pid_file.exists():
            return False
        pid = int(pid_file.read_text().strip())
        os.kill(pid, 0)
        return True
    except (ValueError, ProcessLookupError, PermissionError, OSError):
        return False


def _queue_to_daemon(session_id: str, session_file: str, project_dir: str) -> None:
    task_dir = Path.home() / ".claude" / "daemon-tasks"
    task_dir.mkdir(parents=True, exist_ok=True)
    priority = int(time.time() * 1000)
    task = {
        "kind": "summarize",
        "session_id": session_id,
        "file": session_file,
        "project_dir": project_dir,
        "quick_summary": None,
    }
    (task_dir / f"{priority}-summarize-{session_id[:8]}.json").write_text(json.dumps(task))


def _get_title(session_id: str, session_file: Path) -> str:
    """Get cached title, falling back to stale cache if current key doesn't match."""
    ck = _cache.cache_key(session_file)
    cached = _cache.get(session_id, ck, "summary")
    if cached:
        return cached.get("title", "")
    # Stale cache — read directly, title is still useful
    data = _cache._read(session_id)
    summary = data.get("summary")
    return summary.get("title", "") if isinstance(summary, dict) else ""


def _session_row(s: dict, extra: dict | None = None) -> dict:
    """Standard compact session row. Omits resume_cmd (caller knows the pattern)."""
    row = {
        "id": s["session_id"],
        "project": shorten_path(s["project_dir"]),
        "date": datetime.fromtimestamp(s["mtime"]).strftime("%Y-%m-%d %H:%M"),
        "title": _get_title(s["session_id"], s["file"]),
    }
    if extra:
        row.update(extra)
    return row


def _read_session_bytes(s: dict, chunk_size: int = 1024 * 1024) -> bytes | None:
    """Read and lowercase a session file. Returns None on error."""
    try:
        if s["size"] < chunk_size:
            return s["file"].read_bytes().lower()
        parts = []
        with open(s["file"], "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                parts.append(chunk)
        return b"".join(parts).lower()
    except OSError:
        return None


def _extract_snippet(raw: bytes, term: bytes, context_chars: int = 80) -> str:
    """Extract a short snippet around the first occurrence of term in raw bytes."""
    idx = raw.find(term)
    if idx < 0:
        return ""
    start = max(0, idx - context_chars)
    end = min(len(raw), idx + len(term) + context_chars)
    snippet = raw[start:end]
    # Try to decode, clean up JSON artifacts
    try:
        text = snippet.decode("utf-8", errors="replace")
    except Exception:
        return ""
    # Strip partial JSON escapes and control chars
    text = text.replace("\\n", " ").replace("\\t", " ").replace('\\"', '"')
    text = re.sub(r'["\{\}\[\]\\]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    if start > 0:
        text = "..." + text
    if end < len(raw):
        text = text + "..."
    return text


@mcp.tool()
def search_sessions(query: str, limit: int = 10, include_automated: bool = False) -> list[dict]:
    """Search all Claude Code sessions by keywords (~3s for 5000+ sessions).

    Query syntax:
      - Multiple words: AND logic (all must appear). "visa mastercard" finds
        sessions containing BOTH words.
      - Quoted phrases: exact match. '"mountain creek"' finds that exact phrase.
      - Single word: standard search.

    Returns matches ranked by multi-signal relevance:
      - Term frequency (TF): how often terms appear, with diminishing returns
      - Term density: matches per KB (favors focused sessions over huge dumps)
      - Recency: exponential decay with 30-day half-life
      - Title boost: 3x weight if terms appear in cached session title

    Each result includes a contextual snippet showing where the match occurs.
    Use read_session() to drill into a result. Resume with: claude --resume <id>

    Performance optimizations:
      - Fast path: bulk-loads all cache JSON files (~1KB each) once before
        searching. Cache files contain pre-extracted search_text and
        classification, avoiding reads of raw JSONL files (1-5MB each).
      - ML pre-filter: sessions classified as "automated" are skipped by
        default (set include_automated=True to include them).
      - Raw JSONL fallback: used only when no cached search_text is available.

    Parameters:
      include_automated: If False (default), skip sessions classified as
        "automated" by the ML classifier. Significantly reduces corpus size
        for interactive session searches.
    """
    query = query.strip()
    if not query:
        return []
    limit = max(1, min(limit, 50))

    # Parse query: support quoted phrases and individual terms
    phrases = re.findall(r'"([^"]+)"', query)
    remaining = re.sub(r'"[^"]*"', '', query).strip()
    words = [w for w in remaining.lower().split() if w]

    # Build search terms: phrases stay intact, words are individual
    terms_bytes: list[bytes] = []
    for phrase in phrases:
        terms_bytes.append(phrase.lower().encode("utf-8", errors="replace"))
    for word in words:
        terms_bytes.append(word.encode("utf-8", errors="replace"))

    if not terms_bytes:
        return []

    all_sessions = find_all_sessions()

    # Bulk-load all cache files ONCE before the thread pool (~1KB each vs 1-5MB JSONL).
    # This single pass replaces per-session JSONL reads for the vast majority of sessions.
    cache_index: dict[str, dict] = {}
    if _cache._dir.exists():
        for cache_file in _cache._dir.glob("*.json"):
            sid = cache_file.stem
            try:
                data = json.loads(cache_file.read_bytes())
                cache_index[sid] = data
            except Exception:
                pass

    def _check(s):
        sid = s["session_id"]
        cached = cache_index.get(sid)

        # ML pre-filter: skip automated sessions if not requested
        if cached is not None and not include_automated:
            if cached.get("classification") == "automated":
                return None

        # Fast path: use cached search_text (already lowercased plain text)
        if cached is not None and cached.get("search_text"):
            raw = cached["search_text"].encode("utf-8", errors="replace")
            # search_text is already lowercased; ensure terms match
            per_term_counts = []
            for term in terms_bytes:
                c = raw.count(term)
                if c == 0:
                    return None
                per_term_counts.append(c)
            total_count = sum(per_term_counts)
            min_count = min(per_term_counts)
            rarest_idx = per_term_counts.index(min_count)
            snippet = _extract_snippet(raw, terms_bytes[rarest_idx])
            return (s, total_count, min_count, snippet)

        # Slow path: read raw JSONL (fallback for uncached sessions)
        raw = _read_session_bytes(s)
        if raw is None:
            return None
        # ALL terms must be present (AND logic)
        per_term_counts = []
        for term in terms_bytes:
            c = raw.count(term)
            if c == 0:
                return None
            per_term_counts.append(c)
        total_count = sum(per_term_counts)
        # Min count across terms — a session strong in one term but weak in
        # another should rank lower than one balanced across all terms
        min_count = min(per_term_counts)
        # Find best snippet (use the rarest term for most relevant context)
        rarest_idx = per_term_counts.index(min_count)
        snippet = _extract_snippet(raw, terms_bytes[rarest_idx])
        return (s, total_count, min_count, snippet)

    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(_check, all_sessions))

    matches = [r for r in results if r is not None]
    if not matches:
        return []

    # Reciprocal Rank Fusion (RRF) scoring
    # Rank each signal independently, combine via 1/(k+rank).
    # Outlier-resistant: uses position, not magnitude.
    now = time.time()
    _LAMBDA = math.log(2) / (30 * 86400)  # 30-day half-life
    _K = 60  # RRF constant (standard value)

    # Compute raw signal values for each match
    signals: list[tuple] = []  # (idx, recency, freq, balance, density, title_hit)
    for i, item in enumerate(matches):
        s, total_count, min_count, _snippet = item
        age_s = max(now - s["mtime"], 0)
        recency = math.exp(-_LAMBDA * age_s)
        freq = math.sqrt(total_count)
        balance = math.sqrt(min_count)
        density = total_count / max(s["size"], 1)
        title = _get_title(s["session_id"], s["file"]).lower()
        title_hit = 1.0 if title and any(t.decode("utf-8", errors="replace") in title for t in terms_bytes) else 0.0
        signals.append((i, recency, freq, balance, density, title_hit))

    # Build per-signal rankings (higher value = better = rank 1)
    n = len(signals)
    rrf_scores = [0.0] * n
    # Weights per signal (sum to 1.0)
    weights = [0.25, 0.25, 0.20, 0.15, 0.15]
    for sig_idx in range(1, 6):  # signals 1-5
        # Sort by this signal descending
        ranked = sorted(range(n), key=lambda j: signals[j][sig_idx], reverse=True)
        for rank, j in enumerate(ranked):
            rrf_scores[j] += weights[sig_idx - 1] / (_K + rank + 1)

    scored = [(matches[i], rrf_scores[i]) for i in range(n)]
    scored.sort(key=lambda x: x[1], reverse=True)
    scored = scored[:limit]

    return [
        _session_row(item[0], {
            "matches": item[1],
            "snippet": item[3],
            "score": round(score, 3),
        })
        for item, score in scored
    ]


@mcp.tool()
def read_session(
    session_id: str,
    keyword: str = "",
    limit: int = 10,
) -> dict:
    """Read user/assistant messages from a Claude Code session.

    Returns head+tail messages for quick context. Optional keyword
    filters to only matching messages. Use session_summary() for
    AI-generated summaries instead.
    """
    limit = max(1, min(limit, 30))
    session = _find_session(session_id)
    if session is None:
        return {"error": f"Session {session_id[:36]} not found"}

    result = _read_messages(session["file"], keyword, limit)
    result["id"] = session_id
    result["project"] = shorten_path(session["project_dir"])
    result["date"] = datetime.fromtimestamp(session["mtime"]).strftime("%Y-%m-%d %H:%M")
    return result


def _read_messages(session_file: Path, keyword: str, limit: int) -> dict:
    """Extract user+assistant text messages from a session JSONL."""
    messages = []
    keyword_lower = keyword.lower() if keyword else ""

    try:
        with open(session_file, "r", errors="replace") as f:
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
                if entry_type not in ("user", "assistant"):
                    continue

                msg = entry.get("message", {})
                if not isinstance(msg, dict):
                    continue
                content = msg.get("content", "")
                if isinstance(content, list):
                    texts = []
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            texts.append(block.get("text", ""))
                        elif isinstance(block, str):
                            texts.append(block)
                    content = "\n".join(texts)
                elif not isinstance(content, str):
                    continue

                if not content.strip():
                    continue

                if keyword_lower and keyword_lower not in content.lower():
                    continue

                messages.append({"role": entry_type, "text": _trunc(content)})
    except OSError as e:
        return {"error": f"Could not read session file: {e}"}

    total = len(messages)
    limit = max(limit, 1)
    half = limit // 2 or 1
    if total <= limit:
        selected = messages
    else:
        selected = messages[:half] + messages[-half:]

    result = {"total": total, "messages": selected}
    if total > limit:
        result["note"] = f"First {half} + last {half} shown. {total - limit} omitted."
    if keyword:
        result["filter"] = keyword
        if total == 0:
            result["note"] = "Keyword not found in user/assistant messages. It may appear in tool calls or system entries. Try without keyword to see all messages."
    return result


@mcp.tool()
def recent_sessions(hours: int = 24, limit: int = 10) -> list[dict]:
    """List recently active Claude Code sessions.

    Resume any session with: claude --resume <id>
    """
    limit = max(1, min(limit, 25))
    sessions = find_recent_sessions(hours, max_sessions=limit)
    return [_session_row(s) for s in sessions]


@mcp.tool()
def session_summary(session_id: str, force_regenerate: bool = False,
                    depth: str = "auto") -> dict:
    """Get or generate an AI summary for a session.

    depth controls summarization tier:
      "auto"    — smart selection based on session size/messages/git state (default)
      "quick"   — Tier 1: fast Haiku summary, ~30s if uncached
      "deep"    — Tier 2: richer Haiku summary with decisions + next steps, ~60s
      "insight" — Tier 3: Sonnet-powered full analysis with architecture + blockers + confidence, ~90s

    Returns cached summary instantly. If uncached, generates synchronously.
    Higher tiers are cached separately so re-requesting "quick" doesn't discard "insight".
    """
    session = _find_session(session_id)
    if session is None:
        return {"error": f"Session {session_id[:36]} not found"}

    session_file = session["file"]
    ck = _cache.cache_key(session_file)

    # Normalize depth
    depth = depth.lower().strip() if depth else "auto"
    if depth not in ("auto", "quick", "deep", "insight"):
        depth = "auto"

    # For auto/quick, check cache first (fast path)
    if depth in ("auto", "quick") and not force_regenerate:
        cached = _cache.get(session_id, ck, "summary")
        if not cached:
            data = _cache._read(session_id)
            cached = data.get("summary") if isinstance(data.get("summary"), dict) else None
        if cached and _summary_valid(cached):
            # If auto and cached tier is already deep/insight, return it
            cached_tier = cached.get("_tier", 1)
            if depth == "auto" or cached_tier >= 1:
                return {"id": session_id, "source": "cache", "tier": cached_tier, **cached}

    # deep/insight: check their own cache keys
    if depth in ("deep", "insight") and not force_regenerate:
        cache_key = f"summary_{depth}"
        cached = _cache.get(session_id, ck, cache_key)
        if cached and _summary_valid(cached):
            return {"id": session_id, "source": "cache", "tier": 2 if depth == "deep" else 3, **cached}

    # Need to generate — always synchronous for tier 2/3 (too heavy for daemon queue)
    context, search_text = parse_session(session_file)
    git = get_git_context(session["project_dir"])
    file_size = session_file.stat().st_size if session_file.exists() else 0

    # Resolve auto tier
    if depth == "auto":
        tier = auto_tier(context, file_size, git)
    elif depth == "quick":
        tier = 1
    elif depth == "deep":
        tier = 2
    else:  # insight
        tier = 3

    if tier == 1:
        # Prefer daemon for quick summaries — non-blocking
        if _daemon_alive() and not force_regenerate:
            _queue_to_daemon(session_id, str(session_file), session["project_dir"])
            return {
                "id": session_id,
                "source": "queued",
                "tier": 1,
                "note": "Queued to daemon. Call again in ~15s.",
            }
        summary = summarize_quick(context, session["project_dir"], git)
        summary["_tier"] = 1
        _cache.set(session_id, ck, "summary", summary)
    elif tier == 2:
        quick = summarize_quick(context, session["project_dir"], git)
        summary = summarize_deep(context, session["project_dir"], quick, git)
        summary["_tier"] = 2
        _cache.set(session_id, ck, "summary_deep", summary)
        # Also cache as "summary" so quick requests benefit from the richer version
        _cache.set(session_id, ck, "summary", summary)
    else:  # tier == 3
        quick = summarize_quick(context, session["project_dir"], git)
        deep = summarize_deep(context, session["project_dir"], quick, git)
        summary = summarize_insight(context, session["project_dir"], deep, git, file_size)
        summary["_tier"] = 3
        _cache.set(session_id, ck, "summary_insight", summary)
        _cache.set(session_id, ck, "summary_deep", deep)
        _cache.set(session_id, ck, "summary", summary)

    full = (search_text + f" {session['project_dir']} {session_id}").lower()
    _cache.set(session_id, ck, "search_text", full)

    return {"id": session_id, "source": "generated", "tier": tier, **summary}


@mcp.tool()
def boot_up(hours: int = 24) -> dict:
    """Crash recovery: find interrupted Claude Code sessions that need attention.

    Detects sessions that were recently active but didn't exit cleanly —
    crashed terminals, killed processes, laptop sleep/restart, etc.
    Returns a prioritized list scored by urgency (recency + dirty files).

    Use after a reboot, crash, or "what was I working on?" moment.
    Resume any session with: claude --resume <id>
    """
    import subprocess

    hours = max(1, min(hours, 168))  # 1h to 7d
    now = time.time()
    cutoff = now - hours * 3600
    _LAMBDA = math.log(2) / (2 * 3600)  # 2-hour half-life (urgency, not search)

    # 1. Find sessions modified within the window
    recent = [s for s in find_all_sessions() if s["mtime"] >= cutoff]

    # 2. Find currently running claude processes and extract session IDs
    running_ids = set()
    try:
        ps = subprocess.run(
            ["ps", "aux"], capture_output=True, text=True, timeout=5
        )
        for line in ps.stdout.splitlines():
            if "--resume" in line:
                m = _UUID_RE.search(line)
                if m:
                    running_ids.add(m.group())
            elif "claude" in line and ".jsonl" in line:
                m = _UUID_RE.search(line)
                if m:
                    running_ids.add(m.group())
    except (subprocess.TimeoutExpired, OSError):
        pass

    # 3. Load all bookmarks
    bookmarks_dir = Path.home() / ".claude" / "bookmarks"
    bookmarks = {}
    if bookmarks_dir.exists():
        for bf in bookmarks_dir.glob("*-bookmark.json"):
            try:
                data = json.loads(bf.read_text())
                sid = data.get("session_id", "")
                if sid:
                    bookmarks[sid] = data
            except (json.JSONDecodeError, OSError):
                continue

    # 4. Classify each session
    sessions = []
    for s in recent:
        sid = s["session_id"]

        # Skip currently running sessions
        if sid in running_ids:
            continue

        bookmark = bookmarks.get(sid)
        lifecycle = bookmark.get("lifecycle_state", "") if bookmark else ""

        # Clean exits — skip
        if lifecycle in ("done", "paused", "blocked", "handing-off"):
            continue

        # Only recent sessions are plausible crash candidates.
        # Old sessions without bookmarks predate the bookmark system.
        # Old auto-closed sessions were already dealt with.
        age_h = (now - s["mtime"]) / 3600
        if not bookmark and age_h > 6:
            continue
        if lifecycle == "auto-closed" and age_h > 12:
            continue

        # What's left: recent auto-closed, or recent no-bookmark
        ws = bookmark.get("workspace_state", {}) if bookmark else {}
        dirty = ws.get("dirty", False)
        uncommitted = ws.get("uncommitted_files", [])
        last_commit = ws.get("last_commit", "")
        branch = bookmark.get("project", {}).get("git_branch", "") if bookmark else ""

        # Context: prefer cached title (richer), fall back to bookmark summary
        context_summary = _get_title(sid, s["file"])
        if not context_summary and bookmark:
            context_summary = bookmark.get("context", {}).get("summary", "")

        # Urgency score: exponential decay (2h half-life) + dirty file boost
        age_s = max(now - s["mtime"], 0)
        time_score = math.exp(-_LAMBDA * age_s)
        dirty_boost = 0.2 if dirty else 0
        file_boost = min(0.15, len(uncommitted) * 0.03) if uncommitted else 0
        score = time_score + dirty_boost + file_boost

        state = "crashed" if not bookmark else "interrupted"
        if lifecycle == "auto-closed":
            state = "auto-closed"

        row = {
            "id": sid,
            "project": shorten_path(s["project_dir"]),
            "date": datetime.fromtimestamp(s["mtime"]).strftime("%Y-%m-%d %H:%M"),
            "state": state,
            "summary": _trunc(context_summary, 100),
            "score": round(score, 3),
        }
        if dirty:
            row["dirty"] = True
            row["uncommitted_files"] = uncommitted[:10]
        if branch:
            row["branch"] = branch
        if last_commit:
            row["last_commit"] = last_commit

        sessions.append(row)

    # Sort by urgency score descending
    sessions.sort(key=lambda x: x["score"], reverse=True)

    return {
        "total": len(sessions),
        "running": len(running_ids),
        "checked": len(recent),
        "sessions": sessions[:15],
    }


def _launch_terminal(project_dir: str, command: str) -> dict | None:
    """Open a terminal window, cd to project, run command.

    Tries iTerm2 first (AppleScript), falls back to macOS Terminal.app.
    Returns error dict on failure, None on success.
    """
    import subprocess
    import platform

    if platform.system() != "Darwin":
        return {"error": "Terminal launch requires macOS. Run manually.", "command": command, "directory": project_dir}

    # Try iTerm2 first
    iterm_script = f'''
    tell application "iTerm2"
        activate
        set newWindow to (create window with default profile)
        tell current session of newWindow
            write text "cd {project_dir}"
            write text {json.dumps(command)}
        end tell
    end tell
    '''
    try:
        result = subprocess.run(
            ["osascript", "-e", iterm_script],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return None
    except (subprocess.TimeoutExpired, OSError):
        pass

    # Fall back to Terminal.app
    terminal_script = f'''
    tell application "Terminal"
        activate
        do script "cd {project_dir} && {command}"
    end tell
    '''
    try:
        subprocess.run(
            ["osascript", "-e", terminal_script],
            capture_output=True, text=True, timeout=10,
        )
        return None
    except (subprocess.TimeoutExpired, OSError) as e:
        return {"error": f"Failed to launch terminal: {e}", "command": command, "directory": project_dir}


@mcp.tool()
def resume_in_terminal(session_id: str, fork: bool = False) -> dict:
    """Resume or fork a Claude Code session in a new terminal window.

    Default: resumes the session (continues same session ID).
    With fork=True: creates a new session ID with the full conversation
    history — like git branch. Original session stays untouched.

    Tries iTerm2 first, falls back to Terminal.app. On non-macOS,
    returns the command to run manually.

    Note: --resume requires the correct project directory. The terminal
    window cd's there automatically.
    """
    session = _find_session(session_id)
    if session is None:
        return {"error": f"Session {session_id[:36]} not found"}

    project_dir = session["project_dir"]
    title = _get_title(session_id, session["file"]) or project_dir

    cmd = f"claude --resume {session_id}"
    if fork:
        cmd += " --fork-session"

    err = _launch_terminal(project_dir, cmd)
    if err:
        return err

    return {
        "launched": True,
        "forked": fork,
        "session_id": session_id,
        "project": shorten_path(project_dir),
        "title": _trunc(title, 80),
    }


@mcp.tool()
def merge_context(
    session_id: str,
    mode: str = "hybrid",
    keyword: str = "",
    message_limit: int = 6,
) -> dict:
    """Import context from another Claude Code session into this one.

    Use this to pull in research, decisions, or progress from a previous
    session without copy-pasting. The returned context is formatted for
    direct consumption — Claude understands it as imported session data.

    Modes:
      - "summary": AI-generated summary only (~1-2k tokens). Fast, compact.
      - "messages": Head+tail user/assistant messages (~1-5k tokens). Richer.
      - "hybrid": Summary + last few messages (~2-4k tokens). Best default.

    Optional keyword filter narrows messages to only matching content.
    """
    session = _find_session(session_id)
    if session is None:
        return {"error": f"Session {session_id[:36]} not found"}

    message_limit = max(2, min(message_limit, 20))
    project = shorten_path(session["project_dir"])
    date = datetime.fromtimestamp(session["mtime"]).strftime("%Y-%m-%d %H:%M")

    # --- Gather summary ---
    summary = None
    if mode in ("summary", "hybrid"):
        ck = _cache.cache_key(session["file"])
        summary = _cache.get(session_id, ck, "summary")
        if not summary:
            data = _cache._read(session_id)
            summary = data.get("summary") if isinstance(data.get("summary"), dict) else None

    # --- Gather messages ---
    msgs = None
    msgs_total = 0
    if mode in ("messages", "hybrid"):
        msg_limit = message_limit if mode == "messages" else min(message_limit, 6)
        raw = _read_messages(session["file"], keyword, msg_limit)
        if "messages" in raw:
            msgs = raw["messages"]
            msgs_total = raw.get("total", len(msgs))

    # --- Gather bookmark ---
    bookmark = None
    bookmark_file = Path.home() / ".claude" / "bookmarks" / f"{session_id}-bookmark.json"
    if bookmark_file.exists():
        try:
            bm = json.loads(bookmark_file.read_text())
            bookmark = {}
            ctx = bm.get("context", {})
            if ctx.get("summary"):
                bookmark["summary"] = ctx["summary"]
            if ctx.get("next_actions"):
                bookmark["next_actions"] = ctx["next_actions"][:5]
            ws = bm.get("workspace_state", {})
            if ws.get("uncommitted_files"):
                bookmark["uncommitted_files"] = ws["uncommitted_files"][:10]
            if bm.get("lifecycle_state"):
                bookmark["lifecycle_state"] = bm["lifecycle_state"]
            if not bookmark:
                bookmark = None
        except (json.JSONDecodeError, OSError):
            pass

    # --- Format context block ---
    lines = [f"## Imported Context from Session: {session_id[:8]}"]
    lines.append(f"**Project:** {project}  |  **Date:** {date}")

    if summary:
        if summary.get("title"):
            lines.append(f"**Title:** {summary['title']}")
        if summary.get("goal"):
            lines.append(f"**Goal:** {summary['goal']}")
        if summary.get("what_was_done"):
            lines.append(f"**Progress:** {summary['what_was_done']}")
        if summary.get("state"):
            lines.append(f"**State:** {summary['state']}")
        if summary.get("files"):
            lines.append(f"**Key Files:** {', '.join(summary['files'][:8])}")
        if summary.get("decisions_made"):
            lines.append("**Decisions:** " + "; ".join(summary["decisions_made"][:5]))
        if summary.get("next_steps"):
            lines.append(f"**Next Steps:** {summary['next_steps']}")

    if bookmark:
        lines.append("")
        if bookmark.get("lifecycle_state"):
            lines.append(f"**Session State:** {bookmark['lifecycle_state']}")
        if bookmark.get("next_actions"):
            lines.append("**Planned Actions:** " + "; ".join(bookmark["next_actions"]))
        if bookmark.get("uncommitted_files"):
            lines.append(f"**Uncommitted Files:** {', '.join(bookmark['uncommitted_files'])}")

    if msgs:
        lines.append("")
        lines.append("### Recent Conversation")
        for m in msgs:
            role = "User" if m["role"] == "user" else "Assistant"
            lines.append(f"**{role}:** {m['text']}")

    lines.append("")
    lines.append("---")
    lines.append("*Context imported via claude-resume merge. Continue from above.*")

    context_block = "\n".join(lines)

    result = {
        "session_id": session_id,
        "project": project,
        "date": date,
        "mode": mode,
        "context": context_block,
        "has_summary": summary is not None,
        "has_bookmark": bookmark is not None,
        "has_messages": msgs is not None,
    }
    if msgs is not None:
        result["messages_included"] = len(msgs)
        result["messages_total"] = msgs_total
    if keyword:
        result["keyword_filter"] = keyword
    if summary is None and mode in ("summary", "hybrid"):
        result["note"] = "No cached summary. Call session_summary() first for richer context, or use mode='messages'."

    return result


@mcp.tool()
def session_timeline(
    session_id: str,
    limit: int = 50,
    focus: str = "recent",
    after: str = "",
    before: str = "",
) -> dict:
    """Extract a structured timeline of milestones from a session.

    Solves the "black box" problem for long sessions — instead of
    head+tail messages, returns key events: file creates/edits,
    git commits, user instructions, and significant tool calls.

    Each event has a timestamp, type, and summary. Use this to
    understand what happened in a 2000+ message session without
    reading every message.

    Args:
        limit: Max events to return (10-200, default 50).
        focus: Sampling strategy when events exceed limit.
          - "recent": 70% from the tail, 30% from the rest (default).
            Best for "where did we leave off?"
          - "even": Evenly spaced across the full session.
            Best for understanding the whole arc.
          - "full": No sampling — return all events up to limit,
            most recent first. Best with after/before filters.
        after: ISO timestamp — only events after this time.
          e.g. "2026-03-11" or "2026-03-11T16:00"
        before: ISO timestamp — only events before this time.
    """
    session = _find_session(session_id)
    if session is None:
        return {"error": f"Session {session_id[:36]} not found"}

    limit = max(10, min(limit, 200))
    needs_full_scan = focus == "even" or after or before

    if needs_full_scan:
        events = _extract_events(session["file"])
        if isinstance(events, dict):  # error
            return events
        if after:
            events = [e for e in events if e["time"] >= after]
        if before:
            events = [e for e in events if e["time"] <= before]
    else:
        # Tail-read optimization: only parse enough lines from the end.
        # For "recent" we want ~limit events from the tail + a few from head.
        # For "full" we want limit events from the tail.
        # Over-read by 20x — sessions are ~5% events, 95% progress/thinking/system
        tail_lines = limit * 20
        events = _extract_events_tail(session["file"], tail_lines)
        if isinstance(events, dict):
            return events

    # Deduplicate consecutive file writes to same path
    deduped = _dedup_file_events(events)
    total = len(deduped)

    # Sampling
    if len(deduped) > limit:
        if focus == "recent":
            tail_count = int(limit * 0.7)
            head_count = limit - tail_count
            head_events = deduped[:len(deduped) - tail_count]
            tail_events = deduped[len(deduped) - tail_count:]
            if len(head_events) > head_count and head_count > 0:
                step = len(head_events) / head_count
                head_events = [head_events[int(i * step)] for i in range(head_count)]
            deduped = head_events + tail_events
        elif focus == "even":
            step = len(deduped) / limit
            deduped = [deduped[int(i * step)] for i in range(limit)]
        else:  # "full" — most recent first, truncate
            deduped = list(reversed(deduped[-limit:]))

    return {
        "id": session_id,
        "project": shorten_path(session["project_dir"]),
        "total_events": total,
        "shown": len(deduped),
        "focus": focus,
        "timeline": deduped,
    }


def _dedup_file_events(events: list[dict]) -> list[dict]:
    """Deduplicate consecutive file writes to the same path."""
    deduped = []
    seen_files: set[str] = set()
    for ev in events:
        if ev["type"] in ("file_write", "file_create", "file_update"):
            key = ev["detail"]
            if key in seen_files:
                continue
            seen_files.add(key)
        else:
            seen_files.clear()
        deduped.append(ev)
    return deduped


def _extract_events_tail(session_file: Path, max_lines: int) -> list[dict] | dict:
    """Read the last N lines of a JSONL file and extract events.

    Seeks to end of file, reads backward in chunks to find line
    boundaries, then parses forward. Much faster than full scan
    for "where did we leave off?" queries.
    """
    try:
        file_size = session_file.stat().st_size
        if file_size == 0:
            return []

        chunk_size = 256 * 1024  # 256KB chunks
        lines: list[str] = []

        with open(session_file, "rb") as f:
            # Read backward in chunks until we have enough lines
            pos = file_size
            remainder = b""
            while pos > 0 and len(lines) < max_lines:
                read_size = min(chunk_size, pos)
                pos -= read_size
                f.seek(pos)
                chunk = f.read(read_size) + remainder
                remainder = b""

                # Split into lines
                raw_lines = chunk.split(b"\n")

                # First element is partial if we're not at file start
                if pos > 0:
                    remainder = raw_lines[0]
                    raw_lines = raw_lines[1:]

                # Decode and collect (reverse to maintain order later)
                for raw in reversed(raw_lines):
                    raw = raw.strip()
                    if raw:
                        try:
                            lines.append(raw.decode("utf-8", errors="replace"))
                        except Exception:
                            continue
                        if len(lines) >= max_lines:
                            break

        # Lines were collected in reverse — flip to chronological
        lines.reverse()

        # Parse events from these lines
        return _parse_event_lines(lines)

    except OSError as e:
        return {"error": f"Could not read session: {e}"}


def _parse_event_lines(lines: list[str]) -> list[dict]:
    """Parse JSONL lines into milestone events. Shared by full and tail readers."""
    events = []
    for line in lines:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(entry, dict):
            continue

        ts = entry.get("timestamp", "")
        entry_type = entry.get("type")

        # 1. File creates/updates — from toolUseResult
        tur = entry.get("toolUseResult")
        if isinstance(tur, dict) and tur.get("type") in ("create", "update"):
            fp = tur.get("filePath", "")
            if fp:
                events.append({
                    "time": ts,
                    "type": "file_" + tur["type"],
                    "detail": shorten_path(fp),
                })
            continue

        # 2. Tool use entries — significant tool calls
        if entry_type == "assistant":
            msg = entry.get("message", {})
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        name = block.get("name", "")
                        inp = block.get("input", {})
                        if name == "Bash":
                            cmd = inp.get("command", "")
                            if "git commit" in cmd:
                                events.append({
                                    "time": ts,
                                    "type": "git_commit",
                                    "detail": _trunc(cmd, 120),
                                })
                            elif "git push" in cmd:
                                events.append({
                                    "time": ts,
                                    "type": "git_push",
                                    "detail": _trunc(cmd, 120),
                                })
                        elif name == "Write":
                            fp = inp.get("file_path", "")
                            if fp:
                                events.append({
                                    "time": ts,
                                    "type": "file_write",
                                    "detail": shorten_path(fp),
                                })
                        elif name.startswith("mcp__"):
                            events.append({
                                "time": ts,
                                "type": "mcp_call",
                                "detail": name,
                            })
            continue

        # 3. User text messages
        if entry_type == "user":
            msg = entry.get("message", {})
            content = msg.get("content", "")
            if isinstance(content, str) and len(content.strip()) > 5:
                events.append({
                    "time": ts,
                    "type": "user_message",
                    "detail": _trunc(content.strip(), 150),
                })

    return events


def _extract_events(session_file: Path) -> list[dict] | dict:
    """Parse full JSONL and extract milestone events. Uses threading for large files."""
    file_size = session_file.stat().st_size

    # Small files: single-threaded
    if file_size < 2 * 1024 * 1024:  # < 2MB
        try:
            with open(session_file, "r", errors="replace") as f:
                lines = [l.strip() for l in f if l.strip()]
            return _parse_event_lines(lines)
        except OSError as e:
            return {"error": f"Could not read session: {e}"}

    # Large files: split into chunks, parse in parallel
    try:
        with open(session_file, "r", errors="replace") as f:
            all_lines = [l.strip() for l in f if l.strip()]
    except OSError as e:
        return {"error": f"Could not read session: {e}"}

    # Split into ~4 chunks for ThreadPoolExecutor
    n_chunks = 4
    chunk_size = max(1, len(all_lines) // n_chunks)
    chunks = [
        all_lines[i:i + chunk_size]
        for i in range(0, len(all_lines), chunk_size)
    ]

    with ThreadPoolExecutor(max_workers=n_chunks) as pool:
        results = list(pool.map(_parse_event_lines, chunks))

    # Flatten — already in chronological order since chunks are sequential
    events = []
    for chunk_events in results:
        events.extend(chunk_events)
    return events




@mcp.tool()
def session_thread(session_id: str) -> dict:
    """Follow continuation links to reconstruct a multi-session thread.

    When sessions are continued (via merge_context, bookmarks, or
    explicit "continued from" messages), this tool traces the chain
    and returns all linked sessions in chronological order.

    Use this when you suspect a session is part of a longer arc —
    e.g., the user says "pick up where we left off" and the work
    spans multiple sessions.
    """
    session = _find_session(session_id)
    if session is None:
        return {"error": f"Session {session_id[:36]} not found"}

    bookmarks_dir = Path.home() / ".claude" / "bookmarks"
    all_bookmarks = {}
    if bookmarks_dir.exists():
        for bf in bookmarks_dir.glob("*-bookmark.json"):
            try:
                data = json.loads(bf.read_text())
                sid = data.get("session_id", "")
                if sid:
                    all_bookmarks[sid] = data
            except (json.JSONDecodeError, OSError):
                continue

    # Build thread: walk backward from this session, then forward
    chain = [session_id]
    visited = {session_id}

    # Walk backward: check if this session's JSONL mentions merging another
    _trace_merges(session["file"], chain, visited)

    # Walk forward: check if any bookmark references sessions in our chain
    changed = True
    while changed:
        changed = False
        for sid, bm in all_bookmarks.items():
            if sid in visited:
                continue
            # Check if this bookmark's session merged any session in our chain
            s = _find_session(sid)
            if s is None:
                continue
            merged_ids = _find_merged_ids(s["file"])
            if merged_ids & visited:
                chain.append(sid)
                visited.add(sid)
                changed = True

    # Sort chain chronologically
    chain_sessions = []
    for sid in chain:
        s = _find_session(sid)
        if s:
            bm = all_bookmarks.get(sid)
            row = _session_row(s)
            if bm:
                row["lifecycle"] = bm.get("lifecycle_state", "")
                ctx = bm.get("context", {})
                if ctx.get("summary"):
                    row["bookmark_summary"] = _trunc(ctx["summary"], 150)
                if ctx.get("next_actions"):
                    row["next_actions"] = ctx["next_actions"][:3]
            chain_sessions.append(row)

    chain_sessions.sort(key=lambda x: x["date"])

    return {
        "thread_length": len(chain_sessions),
        "sessions": chain_sessions,
    }


def _find_merged_ids(session_file: Path) -> set[str]:
    """Scan a session JSONL for merge_context calls and extract session IDs."""
    merged = set()
    try:
        with open(session_file, "r", errors="replace") as f:
            for line in f:
                if "merge_context" not in line and "resume" not in line.lower():
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(entry, dict):
                    continue
                # Check tool_use for merge_context calls
                msg = entry.get("message", {})
                content = msg.get("content", []) if isinstance(msg, dict) else []
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "tool_use":
                            if "merge" in block.get("name", "").lower():
                                inp = block.get("input", {})
                                sid = inp.get("session_id", "")
                                if _UUID_RE.fullmatch(sid):
                                    merged.add(sid)
                # Check user text for session IDs near "resume" or "continued"
                if isinstance(content, str):
                    lower = content.lower()
                    if "continued from" in lower or "resume" in lower or "bookmark" in lower:
                        for m in _UUID_RE.finditer(content):
                            merged.add(m.group())
    except OSError:
        pass
    return merged


def _trace_merges(session_file: Path, chain: list, visited: set) -> None:
    """Walk backward through merge links."""
    merged_ids = _find_merged_ids(session_file)
    for mid in merged_ids:
        if mid in visited:
            continue
        visited.add(mid)
        chain.append(mid)
        s = _find_session(mid)
        if s:
            _trace_merges(s["file"], chain, visited)


# Register data science tools on the same MCP instance
from .data_science.mcp_tools import register_tools as _register_ds_tools
_register_ds_tools(mcp)

# Register ADR-001 self-report primitive (report + known_issues)
from .self_report import register_tools as _register_report_tools
_register_report_tools(mcp)


def main():
    if "--install" in sys.argv:
        snippet = {
            "mcpServers": {
                "claude-resume": {
                    "command": "claude-resume-mcp",
                    "args": [],
                }
            }
        }
        print("Add this to ~/.claude/settings.json:\n")
        print(json.dumps(snippet, indent=2))
        print("\nThen restart Claude Code.")
        return
    mcp.run()
