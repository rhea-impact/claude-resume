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

from fastmcp import FastMCP

from .telemetry import TelemetryMiddleware
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
from .progress import progress

mcp = FastMCP("resume-resume")
mcp.add_middleware(TelemetryMiddleware())

_cache = SessionCache()

_TRUNC = 300  # max chars per message/field
_UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")

# Cached wrapper for find_all_sessions — the shared bottleneck across
# search_sessions, dirty_repos, and boot_up. Each call scans the filesystem
# (~1500ms). With a 10s TTL, back-to-back tool calls within a session
# share the scan result.
_ALL_SESSIONS_CACHE: dict = {"data": None, "ts": 0.0}
_ALL_SESSIONS_CACHE_TTL = 10.0


def _find_all_sessions_cached() -> list[dict]:
    now = time.time()
    if _ALL_SESSIONS_CACHE["data"] is not None and (now - _ALL_SESSIONS_CACHE["ts"]) < _ALL_SESSIONS_CACHE_TTL:
        return _ALL_SESSIONS_CACHE["data"]
    result = find_all_sessions()
    _ALL_SESSIONS_CACHE["data"] = result
    _ALL_SESSIONS_CACHE["ts"] = time.time()
    return result


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


def _search_l2_topics(query_tokens: list[str], limit: int = 5) -> list[dict]:
    """Search L2 project topic summaries for query matches.

    Called as a fallback when BM25 session search returns 0 results.
    Returns topic-level results the user can drill into via project_summary.
    """
    if not query_tokens:
        return []
    try:
        from claude_session_commons.insights import get_db
        conn = get_db()
        rows = conn.execute(
            """SELECT entity_id, title, summary_text, updated_at
               FROM summary_levels WHERE level = 2"""
        ).fetchall()
    except Exception:
        return []

    from .bm25 import tokenize as _tokenize

    query_set = set(query_tokens)
    hits = []
    for r in rows:
        entity_id, title, summary_text, updated_at = r
        parts = entity_id.split("::", 1)
        project_path = parts[0] if len(parts) > 1 else entity_id
        topic_name = parts[1] if len(parts) > 1 else title

        try:
            summary = json.loads(summary_text) if isinstance(summary_text, str) else (summary_text or {})
        except (json.JSONDecodeError, TypeError):
            summary = {}

        search_text = " ".join(filter(None, [
            topic_name,
            summary.get("status", ""),
            summary.get("narrative", ""),
            " ".join(summary.get("key_decisions", [])),
            " ".join(summary.get("open_threads", [])),
        ]))
        topic_tokens = set(_tokenize(search_text))

        overlap = query_set & topic_tokens
        if not overlap:
            continue
        score = round(len(overlap) / len(query_set) * 100, 1)

        hits.append({
            "type": "project_topic",
            "project": shorten_path(project_path),
            "project_path": project_path,
            "topic": topic_name,
            "score": score,
            "matched_terms": sorted(overlap),
            "status": summary.get("status", ""),
            "narrative": (summary.get("narrative", "") or "")[:200],
            "updated_at": updated_at,
        })

    hits.sort(key=lambda x: x["score"], reverse=True)
    return hits[:limit]


@mcp.tool()
def search_sessions(query: str, limit: int = 10, include_automated: bool = False,
                    hours: int = 0) -> dict:
    """Search all Claude Code sessions by keywords (~3s for 5000+ sessions).

    Query syntax:
      - Multiple words: AND logic (all must appear). "visa mastercard" finds
        sessions containing BOTH words.
      - Quoted phrases: exact match. '"mountain creek"' finds that exact phrase.
      - Single word: standard search.

    Returns matches ranked by BM25 relevance with summary-first scoring:
      - Summary BM25 (60%): matches in AI-generated session summaries
        (title + goal + what_was_done) — measures what the session IS ABOUT
      - Raw text BM25 (25%): matches in full conversation text with term
        saturation and length normalization
      - Recency (15%): exponential decay with 30-day half-life as tiebreaker

    Score is 0-100. Higher = more relevant. The score reflects real
    magnitude differences — a session scoring 75 is meaningfully more
    relevant than one scoring 30.

    Each result includes hits (raw term count) and a contextual snippet.
    Use read_session() to drill into a result. Resume with: claude --resume <id>

    Parameters:
      include_automated: If False (default), skip sessions classified as
        "automated" by the ML classifier.
      hours: If > 0, only search sessions modified within the last N hours.
        Useful for temporal queries ("what was I doing yesterday" → hours=48).
        Default 0 = search all sessions.
    """
    from .bm25 import tokenize, build_corpus_stats, score_session

    _empty = {"items": [], "count": 0}
    query = query.strip()
    if not query:
        return _empty
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
        return _empty

    # Tokenize query for BM25 (separate from byte terms used for matching)
    query_tokens = tokenize(query)

    all_sessions = _find_all_sessions_cached()

    # Temporal filter: restrict to sessions within the time window
    if hours > 0:
        cutoff = time.time() - hours * 3600
        all_sessions = [s for s in all_sessions if s["mtime"] >= cutoff]

    # Progress HUD — channel per search query
    p_ctx = progress(f"search: {query}")
    p = p_ctx.__enter__()
    p.update(f"Searching {len(all_sessions)} sessions...", icon="search")

    # Bulk-load all cache files ONCE before the thread pool (~1KB each vs 1-5MB JSONL).
    cache_index: dict[str, dict] = {}
    if _cache._dir.exists():
        for cache_file in _cache._dir.glob("*.json"):
            sid = cache_file.stem
            try:
                data = json.loads(cache_file.read_bytes())
                cache_index[sid] = data
            except Exception:
                pass

    # Build corpus-level BM25 statistics (IDF, avg doc lengths)
    corpus = build_corpus_stats(cache_index)
    p.update(f"BM25 index built, scanning...", icon="working")

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
            per_term_counts = []
            for term in terms_bytes:
                c = raw.count(term)
                if c == 0:
                    return None
                per_term_counts.append(c)
            total_count = sum(per_term_counts)
            rarest_idx = per_term_counts.index(min(per_term_counts))
            snippet = _extract_snippet(raw, terms_bytes[rarest_idx])
            raw_len = len(raw)
            return (s, total_count, snippet, raw_len)

        # Slow path: read raw JSONL (fallback for uncached sessions)
        raw = _read_session_bytes(s)
        if raw is None:
            return None
        per_term_counts = []
        for term in terms_bytes:
            c = raw.count(term)
            if c == 0:
                return None
            per_term_counts.append(c)
        total_count = sum(per_term_counts)
        rarest_idx = per_term_counts.index(min(per_term_counts))
        snippet = _extract_snippet(raw, terms_bytes[rarest_idx])
        raw_len = len(raw)
        return (s, total_count, snippet, raw_len)

    # Stream progress as results come in — Perplexity-style
    from concurrent.futures import as_completed

    matches = []
    total = len(all_sessions)
    checked = 0
    last_report = 0

    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(_check, s): s for s in all_sessions}
        for future in as_completed(futures):
            checked += 1
            result = future.result()
            if result is not None:
                matches.append(result)
            # Report every ~15% or every 200 sessions
            pct = int((checked / total) * 100)
            if pct >= last_report + 15 or checked == total:
                last_report = pct
                p.update(f"{checked}/{total} scanned, {len(matches)} matches so far", icon="working")

    p.update(f"{len(matches)} matches from {total} sessions", icon="done")
    if not matches:
        p_ctx.__exit__(None, None, None)
        return _empty

    # BM25 scoring: summary-first with magnitude-based combination
    scored = []
    for item in matches:
        s, total_count, snippet, raw_len = item
        sid = s["session_id"]
        cached = cache_index.get(sid)

        final, _, _, _ = score_session(
            query_tokens, cached, total_count, raw_len,
            s["mtime"], corpus,
        )

        scored.append((s, total_count, snippet, final))

    scored.sort(key=lambda x: x[3], reverse=True)
    scored = scored[:limit]

    p.update(f"Top {len(scored)} results scored", icon="done", highlight=True)
    for s, total_count, snippet, final in scored[:5]:
        title = s.get("title", s.get("session_id", "")[:30])
        proj = shorten_path(s.get("project", ""))
        p.result(title, f"{proj} | score {final:.0f} | {total_count} hits",
                 session_id=s.get("session_id", ""))

    results = [
        _session_row(s, {
            "score": final,
            "hits": total_count,
            "snippet": snippet,
        })
        for s, total_count, snippet, final in scored
    ]

    # L2 fallback: if BM25 found nothing, search project topic summaries.
    # This catches business-context queries ("Wrike renewal") whose terms
    # aren't in raw session text but ARE in AI-generated project summaries.
    if not results:
        l2_hits = _search_l2_topics(query_tokens, limit)
        if l2_hits:
            p.update(f"No session matches; found {len(l2_hits)} project topic(s)", icon="info")
            results = l2_hits

    time.sleep(0.1)  # let socket flush before closing
    p_ctx.__exit__(None, None, None)

    return {"items": results, "count": len(results)}


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


_RECENT_SESSIONS_CACHE: dict = {}
_RECENT_SESSIONS_CACHE_TTL = 10.0  # seconds — short because sessions change fast


@mcp.tool()
def recent_sessions(hours: int = 24, limit: int = 10) -> dict:
    """List recently active Claude Code sessions.

    Resume any session with: claude --resume <id>

    Result is cached for 10 seconds per (hours, limit) key so rapid
    back-to-back calls are free. Short TTL because sessions churn.
    """
    limit = max(1, min(limit, 25))
    cache_key = (hours, limit)
    now = time.time()
    cached = _RECENT_SESSIONS_CACHE.get(cache_key)
    if cached and (now - cached["ts"]) < _RECENT_SESSIONS_CACHE_TTL:
        return {**cached["data"], "cached": True, "cache_age_s": round(now - cached["ts"], 1)}

    sessions = find_recent_sessions(hours, max_sessions=limit)
    items = [_session_row(s) for s in sessions]
    data = {"items": items, "count": len(items), "cached": False}
    _RECENT_SESSIONS_CACHE[cache_key] = {"data": data, "ts": now}
    return data


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


def _scan_repo_git(repo_path: str) -> dict | None:
    """Run git status --porcelain -b (+ git log for dirty repos).

    Combines status and branch into one subprocess call. Clean repos skip
    the git log call and mtime scan entirely — 3 subprocess calls per repo
    drops to 1 for clean repos and 2 for dirty repos.

    Returns None if not a git repo.
    """
    import subprocess

    repo = Path(repo_path)
    if not repo.exists() or not (repo / ".git").exists():
        return None
    try:
        # Combined: status + branch in one call
        status = subprocess.run(
            ["git", "status", "--porcelain", "-b"],
            capture_output=True, text=True, timeout=5, cwd=repo,
        )
        lines = status.stdout.splitlines()

        # First line from -b is `## <branch>...<remote>` or `## HEAD (no branch)`
        branch = ""
        raw_dirty_lines: list[str] = []
        if lines:
            first = lines[0]
            if first.startswith("## "):
                rest = first[3:]
                branch = rest.split("...", 1)[0].split(" ", 1)[0]
                raw_dirty_lines = [ln for ln in lines[1:] if ln.strip()]
            else:
                # Older git or unexpected output — treat all non-empty lines as status
                raw_dirty_lines = [ln for ln in lines if ln.strip()]

        dirty_files = [ln.strip() for ln in raw_dirty_lines]

        # Clean repo fast path — skip git log and mtime scan
        if not dirty_files:
            return {
                "path": shorten_path(repo_path),
                "branch": branch,
                "dirty_files": [],
                "dirty_file_count": 0,
                "recent_commits": [],
                "dirty": False,
                "latest_dirty_mtime": 0.0,
            }

        # Dirty repo — fetch recent commits and mtimes
        log = subprocess.run(
            ["git", "log", "--oneline", "-3", "--format=%h %ar %s"],
            capture_output=True, text=True, timeout=5, cwd=repo,
        )
        recent_commits = [
            line.strip() for line in log.stdout.splitlines() if line.strip()
        ]

        latest_mtime = 0.0
        for line in raw_dirty_lines:
            fname = line[3:].strip().strip('"') if len(line) > 3 else line.strip()
            try:
                fpath = repo / fname
                if fpath.exists():
                    mt = fpath.stat().st_mtime
                    if mt > latest_mtime:
                        latest_mtime = mt
            except OSError:
                continue

        return {
            "path": shorten_path(repo_path),
            "branch": branch,
            "dirty_files": dirty_files[:15],
            "dirty_file_count": len(dirty_files),
            "recent_commits": recent_commits,
            "dirty": True,
            "latest_dirty_mtime": latest_mtime,
        }
    except (subprocess.TimeoutExpired, OSError):
        return None


def _extract_crash_context(session_file: Path) -> dict:
    """Extract rich context from a session JSONL tail for crash recovery.

    Returns:
        last_user_msg: last human message (truncated to 120 chars)
        last_assistant_msg: last AI response snippet (truncated to 120 chars)
        last_tool: last tool_use name + brief input (e.g. "Edit file.py")
        message_count: total entry count (approximate if file > 50KB)
        duration_estimate: "Xh Ym" from first to last entry timestamp
    """
    result: dict = {
        "last_user_msg": "",
        "last_assistant_msg": "",
        "last_tool": "",
        "message_count": 0,
        "duration_estimate": "",
    }
    try:
        size = session_file.stat().st_size
        read_size = min(size, 50 * 1024)
        with open(session_file, "rb") as f:
            if size > read_size:
                f.seek(size - read_size)
            tail = f.read().decode("utf-8", errors="replace")

        first_ts = None
        last_ts = None
        count = 0

        for line in tail.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            count += 1

            # Track timestamps for duration estimate
            ts = entry.get("timestamp")
            if ts:
                if first_ts is None:
                    first_ts = ts
                last_ts = ts

            entry_type = entry.get("type")

            if entry_type == "human":
                msg = entry.get("message", {})
                if isinstance(msg, dict):
                    content = msg.get("content", "")
                    if isinstance(content, str) and content.strip():
                        result["last_user_msg"] = content.strip()[:120]
                    elif isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "text":
                                text = part.get("text", "").strip()
                                if text:
                                    result["last_user_msg"] = text[:120]

            elif entry_type == "assistant":
                msg = entry.get("message", {})
                if isinstance(msg, dict):
                    content = msg.get("content", [])
                    if isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict):
                                if block.get("type") == "text":
                                    text = (block.get("text") or "").strip()
                                    if text:
                                        result["last_assistant_msg"] = text[:120]
                                elif block.get("type") == "tool_use":
                                    name = block.get("name", "")
                                    inp = block.get("input", {})
                                    # Brief summary of tool input
                                    if isinstance(inp, dict):
                                        # Try common keys for a meaningful label
                                        label = (
                                            inp.get("file_path")
                                            or inp.get("path")
                                            or inp.get("command", "")[:60]
                                            or inp.get("query", "")[:60]
                                            or inp.get("pattern", "")[:60]
                                            or ""
                                        )
                                        if label:
                                            result["last_tool"] = f"{name}: {label}"[:120]
                                        else:
                                            result["last_tool"] = name
                                    else:
                                        result["last_tool"] = name

        # Estimate message count (if we only read tail, scale up)
        if size > read_size:
            scale = size / read_size
            result["message_count"] = int(count * scale)
        else:
            result["message_count"] = count

        # Duration estimate
        if first_ts and last_ts and first_ts != last_ts:
            try:
                t0 = datetime.fromisoformat(str(first_ts).replace("Z", "+00:00"))
                t1 = datetime.fromisoformat(str(last_ts).replace("Z", "+00:00"))
                delta = t1 - t0
                hours = int(delta.total_seconds() // 3600)
                mins = int((delta.total_seconds() % 3600) // 60)
                if hours > 0:
                    result["duration_estimate"] = f"{hours}h {mins}m"
                else:
                    result["duration_estimate"] = f"{mins}m"
            except (ValueError, TypeError):
                pass

    except OSError:
        pass
    return result


@mcp.tool()
def boot_up(hours: int = 24) -> dict:
    """Crash recovery: find interrupted Claude Code sessions that need attention.

    Detects sessions that were recently active but didn't exit cleanly —
    crashed terminals, killed processes, laptop sleep/restart, etc.
    Returns a prioritized list scored by urgency (recency + dirty files).

    Also scans project directories for dirty git state — repos with
    uncommitted changes that may not have a matching session.

    Use after a reboot, crash, or "what was I working on?" moment.
    Resume any session with the resume_cmd provided in each result.
    """
    import subprocess

    hours = max(1, min(hours, 168))  # 1h to 7d
    now = time.time()
    cutoff = now - hours * 3600
    _LAMBDA = math.log(2) / (2 * 3600)  # 2-hour half-life (urgency, not search)

    p_ctx = progress("boot up")
    p = p_ctx.__enter__()

    # 1. Find all sessions; split into recent (time-windowed) and all (for repo discovery)
    all_sessions = _find_all_sessions_cached()
    recent = [s for s in all_sessions if s["mtime"] >= cutoff]
    p.update(f"{len(recent)} recent sessions (last {hours}h), {len(all_sessions)} total", icon="search")

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

    # 4. Collect unique project directories for git scanning (ALL sessions — dirty doesn't age out)
    project_dirs = set()
    for s in all_sessions:
        pd = s["project_dir"]
        if pd and pd != str(Path.home()) and Path(pd).exists():
            project_dirs.add(pd)

    p.update(f"Scanning {len(project_dirs)} repos for dirty git state...", icon="working")

    # 5. Scan repos for dirty git state (parallel, with timeout budget)
    from concurrent.futures import as_completed as _as_completed
    dirty_repos = {}
    repo_scan_results = {}
    repo_total = len(project_dirs)
    repo_checked = 0
    repo_last_pct = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_scan_repo_git, pd): pd for pd in project_dirs}
        for future in _as_completed(futures, timeout=30):
            repo_checked += 1
            pct = int((repo_checked / repo_total) * 100) if repo_total else 100
            if pct >= repo_last_pct + 20 or repo_checked == repo_total:
                repo_last_pct = pct
                p.update(f"Git: {repo_checked}/{repo_total} repos, {len(dirty_repos)} dirty", icon="working")
            try:
                result = future.result(timeout=10)
                pd = futures[future]
                repo_scan_results[pd] = result
                if result and result["dirty"]:
                    dirty_repos[pd] = result
            except Exception:
                continue

    p.update(f"{len(dirty_repos)} dirty repos found", icon="done")

    # 6. Classify each session
    #    Use recent sessions as the base, but also pull in the MOST RECENT
    #    session per dirty repo that isn't already in the window.
    #    Dirty state doesn't age out, but we only need one session per repo.
    candidates = {s["session_id"]: s for s in recent}
    dirty_repo_best: dict[str, dict] = {}  # project_dir -> most recent session
    for s in all_sessions:
        pd = s["project_dir"]
        if pd in dirty_repos and s["session_id"] not in candidates:
            if pd not in dirty_repo_best or s["mtime"] > dirty_repo_best[pd]["mtime"]:
                dirty_repo_best[pd] = s
    for s in dirty_repo_best.values():
        candidates[s["session_id"]] = s

    sessions = []
    session_project_dirs = set()  # track which repos have sessions
    for sid, s in candidates.items():

        # Skip currently running sessions
        if sid in running_ids:
            continue

        bookmark = bookmarks.get(sid)
        lifecycle = bookmark.get("lifecycle_state", "") if bookmark else ""

        # Clean exits — skip (UNLESS the repo is still dirty)
        repo_is_dirty = s["project_dir"] in dirty_repos
        if lifecycle in ("done", "paused", "blocked", "handing-off") and not repo_is_dirty:
            continue

        # Age filters: bypass if repo is dirty — uncommitted work doesn't age out
        age_h = (now - s["mtime"]) / 3600
        if not repo_is_dirty:
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

        # For sessions with no summary at all, extract rich crash context
        crash_context = None
        if not context_summary:
            crash_context = _extract_crash_context(s["file"])
            if crash_context.get("last_user_msg"):
                context_summary = f"Last message: {crash_context['last_user_msg']}"
            else:
                context_summary = "Session closed without explicit bookmark"

        # Enrich with live git state if available
        repo_state = repo_scan_results.get(s["project_dir"])
        if repo_state and repo_state["dirty"] and not dirty:
            dirty = True
            uncommitted = repo_state["dirty_files"][:10]
        if repo_state and not branch:
            branch = repo_state.get("branch", "")
        if repo_state and not last_commit and repo_state.get("recent_commits"):
            last_commit = repo_state["recent_commits"][0]

        # Urgency score: session recency + repo dirty urgency
        # Repo dirty urgency = file count + recency of dirty files (same formula as dirty_repos tool)
        age_s = max(now - s["mtime"], 0)
        time_score = math.exp(-_LAMBDA * age_s)

        repo_urgency = 0.0
        if repo_state and repo_state["dirty"]:
            file_score = min(repo_state.get("dirty_file_count", len(repo_state["dirty_files"])) / 15, 1.0)
            dirty_age = max(now - repo_state["latest_dirty_mtime"], 0) if repo_state.get("latest_dirty_mtime") else now
            dirty_recency = math.exp(-math.log(2) / (24 * 3600) * dirty_age)
            repo_urgency = 0.5 * file_score + 0.5 * dirty_recency

        score = time_score + repo_urgency

        state = "crashed" if not bookmark else "interrupted"
        if lifecycle == "auto-closed":
            state = "auto-closed"

        # Filter noise: ~ home sessions with no context and no dirty files
        project_short = shorten_path(s["project_dir"])
        if project_short == "~" and not dirty and "Last message:" not in context_summary:
            if context_summary == "Session closed without explicit bookmark":
                continue

        session_project_dirs.add(s["project_dir"])

        row = {
            "id": sid,
            "project": project_short,
            "date": datetime.fromtimestamp(s["mtime"]).strftime("%Y-%m-%d %H:%M"),
            "state": state,
            "summary": _trunc(context_summary, 100),
            "score": round(score, 3),
            "resume_cmd": f"cd {project_short} && claude --resume {sid}",
        }
        if dirty:
            row["dirty"] = True
            row["uncommitted_files"] = uncommitted[:10]
        if branch:
            row["branch"] = branch
        if last_commit:
            row["last_commit"] = last_commit

        # Rich crash context — what was Claude doing when the session ended?
        if crash_context:
            if crash_context.get("last_assistant_msg"):
                row["last_claude_said"] = crash_context["last_assistant_msg"]
            if crash_context.get("last_tool"):
                row["last_tool"] = crash_context["last_tool"]
            if crash_context.get("message_count"):
                row["message_count"] = crash_context["message_count"]
            if crash_context.get("duration_estimate"):
                row["duration"] = crash_context["duration_estimate"]

        sessions.append(row)

    # Sort by urgency score descending
    sessions.sort(key=lambda x: x["score"], reverse=True)
    p.update(f"{len(sessions)} sessions need attention", icon="done", highlight=True)
    for s in sessions[:5]:
        p.result(s["summary"][:60], f"{s['project']} | {s['state']} | {s['date']}", session_id=s["id"])

    # 7. Build the full dirty repos list — this is the "what needs attention" view.
    #    Dirty doesn't age out. This list only shrinks by committing.
    all_dirty = []
    for pd, repo_info in sorted(
        dirty_repos.items(),
        key=lambda x: len(x[1]["dirty_files"]),
        reverse=True,
    ):
        entry = dict(repo_info)
        entry["has_recent_session"] = pd in session_project_dirs
        all_dirty.append(entry)

    # 8. Negative space: what we checked
    scan_report = {
        "repos_scanned": len(repo_scan_results),
        "repos_dirty": len(dirty_repos),
        "repos_clean": len(repo_scan_results) - len(dirty_repos),
        "repos_with_sessions": len(session_project_dirs),
    }

    p_ctx.__exit__(None, None, None)

    return {
        "total": len(sessions),
        "running": len(running_ids),
        "checked": len(recent),
        "sessions": sessions[:15],
        "dirty_repos": all_dirty,
        "scan_report": scan_report,
    }


_DIRTY_REPOS_CACHE: dict = {"data": None, "ts": 0.0}
_DIRTY_REPOS_CACHE_TTL = 30.0  # seconds


@mcp.tool()
def dirty_repos() -> dict:
    """List all repos with uncommitted changes — your standing to-do list.

    Scans every project directory Claude Code has ever touched for dirty
    git state. Result is cached for 30 seconds, so back-to-back calls
    within the same session are nearly free.
    """
    now = time.time()
    cached = _DIRTY_REPOS_CACHE["data"]
    if cached is not None and (now - _DIRTY_REPOS_CACHE["ts"]) < _DIRTY_REPOS_CACHE_TTL:
        return {**cached, "cached": True, "cache_age_s": round(now - _DIRTY_REPOS_CACHE["ts"], 1)}

    all_sessions = _find_all_sessions_cached()

    # Collect unique project directories, skipping stale ones.
    # A repo not modified in 30+ days is unlikely to have new dirty state
    # worth scanning — skip it to cut subprocess calls on cold path.
    _STALE_CUTOFF = 30 * 86400  # 30 days in seconds
    now_ts = time.time()
    project_dirs = set()
    skipped_stale = 0
    for s in all_sessions:
        pd = s["project_dir"]
        if not pd or pd == str(Path.home()):
            continue
        p = Path(pd)
        if not p.exists():
            continue
        # Skip repos whose session was last touched > 30 days ago
        if (now_ts - s.get("mtime", 0)) > _STALE_CUTOFF:
            skipped_stale += 1
            continue
        project_dirs.add(pd)

    # Scan in parallel — 16 workers (git status is I/O-bound, not CPU-bound)
    dirty = []
    clean = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(_scan_repo_git, pd): pd for pd in project_dirs}
        for future in futures:
            try:
                result = future.result(timeout=10)
                if result and result["dirty"]:
                    dirty.append(result)
                elif result:
                    clean.append(result["path"])
            except Exception:
                continue

    # Score: file count (normalized) + recency (exponential decay, 24h half-life)
    # More dirty files + more recent changes = higher urgency
    now = time.time()
    _DIRTY_LAMBDA = math.log(2) / (24 * 3600)  # 24-hour half-life
    max_files = max((len(d["dirty_files"]) for d in dirty), default=1)
    for d in dirty:
        file_score = len(d["dirty_files"]) / max_files  # 0-1
        age_s = max(now - d["latest_dirty_mtime"], 0) if d["latest_dirty_mtime"] else now
        recency_score = math.exp(-_DIRTY_LAMBDA * age_s)  # 0-1
        d["urgency"] = round(0.5 * file_score + 0.5 * recency_score, 3)

    dirty.sort(key=lambda x: x["urgency"], reverse=True)

    result = {
        "dirty": dirty,
        "dirty_count": len(dirty),
        "clean_count": len(clean),
        "total_scanned": len(dirty) + len(clean),
        "skipped_stale": skipped_stale,
        "cached": False,
    }
    _DIRTY_REPOS_CACHE["data"] = result
    _DIRTY_REPOS_CACHE["ts"] = time.time()
    return result


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


# Register data science tools on the same MCP instance (optional — requires scipy/sklearn)
try:
    from .data_science.mcp_tools import register_tools as _register_ds_tools
    _register_ds_tools(mcp)
except ImportError:
    pass

# Register L2/L3 project summary tools (requires insights.db from commons daemon)
try:
    from .l2_tools import register_l2_tools
    register_l2_tools(mcp)
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Self-knowledge + Meta-AI tools (extracted to self_tools.py for file size)
# ---------------------------------------------------------------------------

from .self_tools import register_self_tools as _register_self_tools
_register_self_tools(mcp)


def main():
    if "--install" in sys.argv:
        snippet = {
            "mcpServers": {
                "resume-resume": {
                    "command": "resume-resume-mcp",
                    "args": [],
                }
            }
        }
        print("Add this to ~/.claude/settings.json:\n")
        print(json.dumps(snippet, indent=2))
        print("\nThen restart Claude Code.")
        return
    mcp.run()
