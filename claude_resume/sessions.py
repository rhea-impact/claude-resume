"""Session discovery, JSONL parsing, and caching."""

import json
import math
import os
import hashlib
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

CLAUDE_DIR = Path.home() / ".claude"
PROJECTS_DIR = CLAUDE_DIR / "projects"

MAX_SESSIONS_DEFAULT = 200
MAX_SESSIONS_ALL = 200
MIN_SESSION_BYTES = 500
COOLDOWN_SECONDS = 600


# ── Unified cache ──────────────────────────────────────────


class SessionCache:
    """Single cache class replacing 10 separate get/save functions.

    All session data lives in one JSON file per session:
        ~/.claude/resume-summaries/<session-id>.json

    Fields keyed on cache_key (invalidated when file changes):
        summary, deep_summary, patterns, search_text

    Fields independent of cache_key (persist across changes):
        last_seen
    """

    def __init__(self, cache_dir: Path | None = None):
        self._dir = cache_dir or (CLAUDE_DIR / "resume-summaries")
        self._dir.mkdir(exist_ok=True)

    def cache_key(self, session_file: Path) -> str:
        mtime = session_file.stat().st_mtime
        return hashlib.md5(f"{session_file}:{mtime}".encode()).hexdigest()

    def _read(self, session_id: str) -> dict:
        cache_file = self._dir / f"{session_id}.json"
        if not cache_file.exists():
            return {}
        try:
            return json.loads(cache_file.read_text())
        except json.JSONDecodeError:
            return {}

    def _write(self, session_id: str, data: dict):
        cache_file = self._dir / f"{session_id}.json"
        cache_file.write_text(json.dumps(data, indent=2))

    def get(self, session_id: str, cache_key: str, field: str):
        data = self._read(session_id)
        if data.get("cache_key") == cache_key:
            return data.get(field)
        return None

    def set(self, session_id: str, cache_key: str, field: str, value):
        data = self._read(session_id)
        data["cache_key"] = cache_key
        data[field] = value
        self._write(session_id, data)

    def is_recently_seen(self, session_id: str, cooldown: int = COOLDOWN_SECONDS) -> bool:
        data = self._read(session_id)
        return (time.time() - data.get("last_seen", 0)) < cooldown

    def touch_seen(self, session_id: str):
        data = self._read(session_id)
        data["last_seen"] = time.time()
        self._write(session_id, data)


# ── Operations bundle ──────────────────────────────────────


@dataclass
class SessionOps:
    """Everything the TUI needs — one object instead of 7 callbacks."""
    cache: SessionCache
    parse_session: Callable
    get_git_context: Callable
    summarize_quick: Callable
    summarize_deep: Callable
    analyze_patterns: Callable


# ── Utilities ──────────────────────────────────────────────


def decode_project_path(encoded: str) -> str:
    if encoded.startswith("-"):
        return "/" + encoded[1:].replace("-", "/")
    return encoded.replace("-", "/")


def shorten_path(path: str) -> str:
    home = str(Path.home())
    if path.startswith(home):
        return "~" + path[len(home):]
    return path


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'s' if n != 1 else ''}"


def relative_time(mtime: float) -> str:
    delta = int(time.time() - mtime)
    if delta < 60:
        return "just now"

    minutes = delta // 60
    hours = delta // 3600
    days = delta // 86400

    if delta < 3600:
        return f"{_plural(minutes, 'minute')} ago"
    elif delta < 86400:
        remaining_min = (delta % 3600) // 60
        if remaining_min:
            return f"{_plural(hours, 'hour')}, {_plural(remaining_min, 'minute')} ago"
        return f"{_plural(hours, 'hour')} ago"
    else:
        remaining_hrs = (delta % 86400) // 3600
        if remaining_hrs:
            return f"{_plural(days, 'day')}, {_plural(remaining_hrs, 'hour')} ago"
        return f"{_plural(days, 'day')} ago"


# ── Session discovery ──────────────────────────────────────


def _get_last_timestamp(jsonl_file: Path) -> float | None:
    """Read the last entry's timestamp from a JSONL file."""
    try:
        with open(jsonl_file, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            read_size = min(size, 4096)
            f.seek(size - read_size)
            chunk = f.read().decode("utf-8", errors="replace")

        for line in reversed(chunk.strip().split("\n")):
            try:
                obj = json.loads(line)
                ts = obj.get("timestamp")
                if ts:
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    return dt.timestamp()
            except (json.JSONDecodeError, ValueError):
                continue
    except Exception:
        pass
    return None


def find_all_sessions() -> list[dict]:
    """Find ALL .jsonl session files, sorted by last activity."""
    sessions = []
    if not PROJECTS_DIR.exists():
        return sessions

    for project_dir in PROJECTS_DIR.iterdir():
        if not project_dir.is_dir():
            continue
        original_path = decode_project_path(project_dir.name)
        for jsonl_file in project_dir.glob("*.jsonl"):
            stat = jsonl_file.stat()
            if stat.st_size < MIN_SESSION_BYTES:
                continue
            last_ts = _get_last_timestamp(jsonl_file)
            mtime = last_ts if last_ts else stat.st_mtime
            sessions.append({
                "file": jsonl_file,
                "session_id": jsonl_file.stem,
                "project_dir": original_path,
                "mtime": mtime,
                "size": stat.st_size,
            })

    sessions.sort(key=lambda s: s["mtime"], reverse=True)
    return sessions


def find_recent_sessions(hours: float, max_sessions: int = MAX_SESSIONS_DEFAULT) -> list[dict]:
    """Find sessions modified within the lookback window."""
    cutoff = time.time() - (hours * 3600)
    all_sessions = find_all_sessions()
    recent = [s for s in all_sessions if s["mtime"] >= cutoff]
    return recent[:max_sessions]


def get_git_context(project_dir: str) -> dict:
    """Pull recent commits and uncommitted changes from the project's git repo."""
    result = {"recent_commits": "", "uncommitted_changes": "", "is_git_repo": False}

    if not os.path.isdir(project_dir):
        return result

    try:
        check = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            capture_output=True, text=True, cwd=project_dir, timeout=5,
        )
        if check.returncode != 0:
            return result
        result["is_git_repo"] = True

        log = subprocess.run(
            ["git", "log", "--oneline", "--stat", "--no-color", "-5"],
            capture_output=True, text=True, cwd=project_dir, timeout=5,
        )
        if log.returncode == 0:
            result["recent_commits"] = log.stdout.strip()[:2000]

        diff = subprocess.run(
            ["git", "diff", "--stat", "--no-color", "HEAD"],
            capture_output=True, text=True, cwd=project_dir, timeout=5,
        )
        if diff.returncode == 0 and diff.stdout.strip():
            result["uncommitted_changes"] = diff.stdout.strip()[:1000]

        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            capture_output=True, text=True, cwd=project_dir, timeout=5,
        )
        if untracked.returncode == 0 and untracked.stdout.strip():
            files = untracked.stdout.strip().split("\n")[:10]
            if files:
                result["uncommitted_changes"] += "\n\nUntracked files:\n" + "\n".join(files)

    except (subprocess.TimeoutExpired, Exception):
        pass

    return result


# ── JSONL parsing (single pass) ────────────────────────────


# ── ML ensemble classifier ────────────────────────────────

_ML_MODEL = None
_ML_FEATURE_COLS = None


def _load_ml_model():
    """Load the serialized ML model (lazy, once)."""
    global _ML_MODEL, _ML_FEATURE_COLS
    if _ML_MODEL is not None:
        return True
    model_path = Path(__file__).parent / "classifier.pkl"
    if not model_path.exists():
        return False
    try:
        import pickle
        with open(model_path, "rb") as f:
            data = pickle.load(f)
        _ML_MODEL = data["model"]
        _ML_FEATURE_COLS = data["feature_cols"]
        return True
    except Exception:
        return False


def _opus_classify(session_file: Path) -> str | None:
    """Use Opus via `claude -p` to classify a gray-zone session.

    Only called for sessions where the ML model is uncertain (conf < 0.80).
    Returns 'interactive' or 'automated', or None on failure.
    """
    user_msgs = []
    try:
        with open(session_file) as fh:
            for line in fh:
                try:
                    obj = json.loads(line)
                    if obj.get("type") == "user":
                        text = _extract_user_text(obj)
                        if text:
                            user_msgs.append(text[:200])
                except (json.JSONDecodeError, Exception):
                    continue
    except Exception:
        return None

    if not user_msgs:
        return None

    sample = user_msgs[:5]
    if len(user_msgs) > 5:
        sample += user_msgs[-5:]

    prompt = (
        "Classify this Claude Code session as 'interactive' (human typing) or 'automated' (programmatic/scripted).\n"
        "Human sessions have: typos, casual language, questions, short messages, varied tone.\n"
        "Automated sessions have: template prompts, system instructions, consistent formatting, role assignments.\n\n"
        "User messages (sample):\n"
    )
    for i, msg in enumerate(sample):
        prompt += f"[{i+1}] {msg}\n"
    prompt += "\nRespond with ONLY the word 'interactive' or 'automated'."

    try:
        result = subprocess.run(
            ["claude", "-p", prompt, "--model", "opus"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            answer = result.stdout.strip().lower()
            if "interactive" in answer:
                return "interactive"
            elif "automated" in answer:
                return "automated"
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass
    return None


def _ensemble_classify(stats: dict) -> tuple[str, float]:
    """Internal: heuristic + calibrated ML. Returns (label, confidence).

    Never calls Opus — that's handled by the public interface.
    """
    heuristic = classify_session(stats)

    if not _load_ml_model():
        return heuristic, 1.0

    try:
        import numpy as np
        features = np.array([[stats.get(col, 0) for col in _ML_FEATURE_COLS]])
        ml_pred = _ML_MODEL.predict(features)[0]
        ml_proba = _ML_MODEL.predict_proba(features)[0]
        ml_label = "interactive" if ml_pred == 1 else "automated"
        confidence = ml_proba.max()

        if heuristic == ml_label:
            return heuristic, confidence

        if confidence < 0.80:
            return "interactive", confidence  # Uncertain → safe default

        if heuristic == "interactive" and ml_label == "automated":
            label = "automated" if confidence > 0.90 else "interactive"
            return label, confidence
        else:
            return "interactive", confidence
    except Exception:
        return heuristic, 1.0


# ── Public classification interface ───────────────────────────
# The UI calls get_label(). It doesn't know about ML, heuristics, or Opus.


def get_label(session_file: Path, cache: "SessionCache | None" = None) -> str:
    """Classify a session file → 'interactive' or 'automated'.

    This is the only classification function the UI should call.
    Reads from cache if available, otherwise computes via quick_scan + ML.
    Fast path only — no Opus calls. Use get_label_deep() for Opus fallback.
    """
    sid = session_file.stem
    if cache:
        ck = cache.cache_key(session_file)
        cached = cache.get(sid, ck, "classification")
        if cached:
            return cached

    scan = quick_scan(session_file)
    label, _ = _ensemble_classify(scan)
    scan["classification"] = label

    if cache:
        cache.set(sid, ck, "classification", label)
        cache.set(sid, ck, "stats", scan)

    return label


def get_label_deep(session_file: Path, cache: "SessionCache | None" = None) -> str:
    """Classify with Opus fallback for uncertain sessions.

    Use this in batch mode (--cache-all), not in the interactive TUI.
    Calls `claude -p` for gray-zone sessions where ML confidence < 0.80.
    """
    sid = session_file.stem
    if cache:
        ck = cache.cache_key(session_file)
        cached = cache.get(sid, ck, "classification")
        if cached:
            return cached

    scan = quick_scan(session_file)
    label, confidence = _ensemble_classify(scan)

    # Gray zone: ask Opus
    if confidence < 0.80:
        opus_answer = _opus_classify(session_file)
        if opus_answer:
            label = opus_answer

    scan["classification"] = label

    if cache:
        cache.set(sid, ck, "classification", label)
        cache.set(sid, ck, "stats", scan)

    return label


def classify_session(stats: dict) -> str:
    """Classify a session as 'interactive' or 'automated'.

    Primary signal: pace (seconds per user turn).
    A human can't sustain < 15 sec/turn. Machine callers can.
    Secondary signals fill in when pace is ambiguous or unavailable.
    Conservative: when in doubt, return 'interactive' (never hide real sessions).
    """
    user = stats.get("user_messages", 0)
    duration = stats.get("duration_secs", 0)
    has_progress = stats.get("has_progress", False)
    total_lines = stats.get("total_lines", 0)

    # Trivial sessions: <= 3 JSONL lines is a health check or failed start
    if total_lines <= 3:
        return "automated"

    # Zero or one user message with no tool use = single-shot programmatic call
    if user <= 1 and stats.get("tool_uses", 0) == 0:
        return "automated"

    # If we have progress entries, it's definitely interactive
    # (only long-running tool executions produce these)
    if has_progress:
        return "interactive"

    # Pace detection: use pre-computed secs_per_turn (based on effective users)
    secs_per_turn = stats.get("secs_per_turn", 0)
    if user >= 2 and duration > 0 and secs_per_turn > 0:
        if secs_per_turn < 10:
            return "automated"     # Machine speed
        if secs_per_turn > 30:
            return "interactive"   # Human speed

    # Duration alone: > 2 minutes with multiple messages = human in the loop
    if duration > 120 and user >= 3:
        return "interactive"

    # Short duration with few messages = ambiguous, but lean interactive
    # (could be a quick human question or a programmatic call)
    if user >= 3:
        return "interactive"

    # Default: show it (conservative — never hide a real session)
    return "interactive"


def _format_duration(secs: float) -> str:
    """Human-readable duration string."""
    if secs < 60:
        return f"{int(secs)}s"
    minutes = int(secs) // 60
    if secs < 3600:
        remaining = int(secs) % 60
        return f"{minutes}m {remaining}s" if remaining else f"{minutes}m"
    hours = int(secs) // 3600
    remaining_min = (int(secs) % 3600) // 60
    return f"{hours}h {remaining_min}m" if remaining_min else f"{hours}h"


_POLITENESS_WORDS = {"please", "thanks", "thank you", "could you", "would you"}
_CASUAL_MARKERS = {"lol", "lmao", "lfg", "lgtm", "tbh", "imo", "fwiw", "nvm",
                   "ok", "yep", "yea", "yeah", "nah", "hmm", "huh", "oh wait",
                   "got it", "nice", "awesome", "cool", "btw", "brb", "idk",
                   "omg", "wtf", "smh", "bet", "dope", "lit"}

# System dictionary for misspelling detection (macOS)
_DICT_PATH = Path("/usr/share/dict/words")
_WORD_SET: set[str] = set()
if _DICT_PATH.exists():
    try:
        _WORD_SET = set(_DICT_PATH.read_text().lower().splitlines())
        _WORD_SET.update({
            "api", "cli", "ui", "ux", "json", "yaml", "toml", "csv", "sql",
            "html", "css", "js", "ts", "jsx", "tsx", "py", "rb", "rs", "go",
            "npm", "pip", "git", "github", "gitlab", "docker", "kubernetes",
            "async", "await", "const", "var", "func", "def", "impl", "enum",
            "str", "int", "bool", "dict", "tuple", "stdin", "stdout", "stderr",
            "url", "urls", "http", "https", "tcp", "udp", "ssh", "ssl", "tls",
            "env", "config", "configs", "args", "kwargs", "params", "middleware",
            "webhook", "webhooks", "endpoint", "endpoints", "repo", "repos",
            "readme", "changelog", "dockerfile", "makefile", "workflow",
            "codebase", "refactor", "refactored", "frontend", "backend",
            "fullstack", "devops", "ci", "cd", "pr", "prs", "todo", "todos",
            "auth", "oauth", "jwt", "cors", "csrf", "xss", "sdk", "mcp",
            "claude", "anthropic", "openai", "llm", "llms", "gpt", "ai",
        })
    except Exception:
        pass


def _extract_user_text(obj: dict) -> str:
    """Extract user message text from a JSONL entry."""
    msg = obj.get("message", {}).get("content", "")
    if isinstance(msg, str):
        return msg.strip()
    if isinstance(msg, list):
        for c in msg:
            if isinstance(c, dict) and c.get("type") == "text":
                return c.get("text", "").strip()
    return ""


def _is_system_prompt(text: str) -> bool:
    """Detect if a user message is likely a programmatic system prompt.

    System prompts are long, structured, and assign roles.
    Human first messages are short and conversational.
    """
    if not text or len(text) < 200:
        return False
    lower = text[:300].lower()
    return (
        lower.startswith("you are ")
        or lower.startswith("research this ")
        or "\n# " in text[:500]
        or "instructions:" in lower
        or "your task is" in lower
        or "your role is" in lower
        or "your job is" in lower
    )


def _count_entry(obj: dict, stats: dict):
    """Count types, timestamps, and assistant content. No user text features."""
    t = obj.get("type", "")
    ts = obj.get("timestamp")
    if ts:
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            stats["_timestamps"].append(dt.timestamp())
        except (ValueError, TypeError):
            pass

    if t == "user":
        stats["user_messages"] += 1
    elif t == "assistant":
        stats["assistant_messages"] += 1
        amsg = obj.get("message", {})
        if isinstance(amsg, dict):
            for c in (amsg.get("content") or []):
                if isinstance(c, dict):
                    if c.get("type") == "tool_use":
                        stats["tool_uses"] += 1
                    elif c.get("type") == "text":
                        txt = c.get("text", "")
                        stats["_asst_char_counts"].append(len(txt))
                        if "```" in txt:
                            stats["_asst_code_blocks"] += 1
    elif t == "tool_result":
        stats["tool_results"] += 1
    elif t == "system":
        stats["system_entries"] += 1
    elif t == "summary":
        stats["summary_entries"] += 1
    elif t == "progress":
        stats["progress_entries"] += 1


def _is_human_typo(word: str) -> bool:
    """Detect human transposition typos: swapping adjacent chars yields a real word.

    "teh" → swap(1,2) → "the" ✓ (human finger slip)
    "shoudl" → swap(3,4) → "should" ✓
    "nginx" → no swap yields a word ✗ (tech term, not a typo)
    "taskr" → no swap yields a word ✗ (proper noun, not a typo)

    Only same-length edits, so tech terms can't accidentally match
    shorter/longer dictionary words. Near-zero false positives.
    """
    if not _WORD_SET or len(word) < 3:
        return False
    for i in range(len(word) - 1):
        swapped = word[:i] + word[i + 1] + word[i] + word[i + 2:]
        if swapped in _WORD_SET:
            return True
    return False


def _apply_user_text_features(text: str, stats: dict):
    """Extract text-based features from a single user message into stats."""
    words = text.split()
    stats["_user_word_counts"].append(len(words))
    stats["_user_char_counts"].append(len(text))
    lower = text.lower()
    if "?" in text:
        stats["_questions"] += 1
    if any(p in lower for p in _POLITENESS_WORDS):
        stats["_polite"] += 1
    if "```" in text:
        stats["_user_code_blocks"] += 1
    if any(c in lower for c in _CASUAL_MARKERS):
        stats["_casual"] += 1
    if text[0].islower():
        stats["_no_caps"] += 1
    if len(text) < 20:
        stats["_short_msgs"] += 1
    if "!" in text:
        stats["_exclamations"] += 1
    # Human typo detection: transposition of adjacent chars → real word
    for w in words:
        clean = w.strip(".,!?;:'\"()-/").lower()
        if len(clean) >= 3 and clean.isalpha() and clean not in _WORD_SET:
            stats["_dict_words"] += 1
            if _is_human_typo(clean):
                stats["_misspelled"] += 1


def _new_scan_stats() -> dict:
    """Create a fresh stats accumulator for quick_scan."""
    return {
        "user_messages": 0, "assistant_messages": 0,
        "tool_uses": 0, "tool_results": 0,
        "system_entries": 0, "progress_entries": 0, "summary_entries": 0,
        "total_lines": 0,
        # New: effective user count (non-empty, non-prompt)
        "_effective_users": 0, "_empty_messages": 0, "_first_is_prompt": False,
        # Private accumulators (converted to features at end)
        "_timestamps": [], "_user_word_counts": [], "_user_char_counts": [],
        "_asst_char_counts": [], "_questions": 0, "_polite": 0,
        "_user_code_blocks": 0, "_asst_code_blocks": 0,
        "_casual": 0, "_no_caps": 0, "_short_msgs": 0,
        "_exclamations": 0, "_dict_words": 0, "_misspelled": 0,
    }


def _finalize_scan_stats(stats: dict, file_size: int) -> dict:
    """Convert raw scan accumulators into the feature dict used by classify_session + ML."""

    user = stats["user_messages"]
    effective = stats["_effective_users"]  # non-empty, non-prompt
    ts = stats["_timestamps"]
    duration = (max(ts) - min(ts)) if len(ts) >= 2 else 0

    # Pace uses effective user count (empties shouldn't deflate pace)
    secs_per_turn = (duration / effective) if effective > 0 else 0
    msgs_per_minute = (effective / (duration / 60)) if duration > 60 else 0
    lines_per_minute = (stats["total_lines"] / (duration / 60)) if duration > 60 else 0
    tool_to_user = (stats["tool_uses"] / user) if user > 0 else 0

    # Text ratios use effective count (only messages that contributed)
    q_ratio = (stats["_questions"] / effective) if effective > 0 else 0
    p_ratio = (stats["_polite"] / effective) if effective > 0 else 0
    avg_user_chars = float(np.mean(stats["_user_char_counts"])) if stats["_user_char_counts"] else 0
    avg_asst_chars = float(np.mean(stats["_asst_char_counts"])) if stats["_asst_char_counts"] else 0
    empty_ratio = (stats["_empty_messages"] / user) if user > 0 else 0

    return {
        # Core counts
        "user_messages": user,
        "assistant_messages": stats["assistant_messages"],
        "tool_uses": stats["tool_uses"],
        "tool_results": stats["tool_results"],
        "system_entries": stats["system_entries"],
        "progress_entries": stats["progress_entries"],
        "summary_entries": stats["summary_entries"],
        "total_lines": stats["total_lines"],
        "file_size": file_size,
        # Timing (pace uses effective user count)
        "duration_secs": round(duration, 1),
        "log_duration": round(math.log1p(duration), 3),
        "secs_per_turn": round(secs_per_turn, 1),
        "msgs_per_minute": round(msgs_per_minute, 3),
        # Ratios (text ratios use effective count)
        "tool_to_user_ratio": round(tool_to_user, 2),
        "question_ratio": round(q_ratio, 3),
        "politeness_ratio": round(p_ratio, 3),
        # User message features
        "avg_user_chars": round(avg_user_chars, 1),
        "user_code_blocks": stats["_user_code_blocks"],
        # Assistant features
        "avg_assistant_chars": round(avg_asst_chars, 1),
        "assistant_code_blocks": stats["_asst_code_blocks"],
        # Human-messiness signals (ratios use effective count)
        "casual_ratio": round((stats["_casual"] / effective) if effective > 0 else 0, 3),
        "no_caps_ratio": round((stats["_no_caps"] / effective) if effective > 0 else 0, 3),
        "short_msg_ratio": round((stats["_short_msgs"] / effective) if effective > 0 else 0, 3),
        "exclamation_ratio": round((stats["_exclamations"] / effective) if effective > 0 else 0, 3),
        "typo_score": round((stats["_misspelled"] / stats["_dict_words"]) if stats["_dict_words"] > 0 else 0, 3),
        # New features
        "empty_msg_ratio": round(empty_ratio, 3),
        "first_is_prompt": 1 if stats["_first_is_prompt"] else 0,
        # Derived flags
        "has_progress": stats["progress_entries"] > 0,
        "duration_fmt": _format_duration(duration),
        "classification": "pending",
    }


def quick_scan(session_file: Path) -> dict:
    """Scan a session file for classification features.

    Two-phase approach:
    1. Count types/timestamps/assistant features for every line (_count_entry)
    2. Collect user message texts, then apply text features after filtering:
       - Skip system-prompt-like first messages (pollute casual/question ratios)
       - Skip empty messages (pollute pace calculation)

    This produces clean features where human-messiness signals actually
    reflect human behavior, not system prompt content.
    """
    stats = _new_scan_stats()
    size = 0
    user_texts = []  # all user message texts, in order

    try:
        size = session_file.stat().st_size
        with open(session_file) as fh:
            for line in fh:
                stats["total_lines"] += 1
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                _count_entry(obj, stats)
                if obj.get("type") == "user":
                    user_texts.append(_extract_user_text(obj))
    except Exception:
        pass

    # Phase 2: text features on clean messages only
    texts_for_features = user_texts
    if texts_for_features and _is_system_prompt(texts_for_features[0]):
        stats["_first_is_prompt"] = True
        texts_for_features = texts_for_features[1:]

    for text in texts_for_features:
        if not text:
            stats["_empty_messages"] += 1
            continue
        stats["_effective_users"] += 1
        _apply_user_text_features(text, stats)

    # If first was a prompt, count its emptiness separately
    if stats["_first_is_prompt"] and user_texts and not user_texts[0]:
        stats["_empty_messages"] += 1

    return _finalize_scan_stats(stats, size)


def parse_session(session_file: Path, deep: bool = False) -> tuple[dict, str]:
    """Parse JSONL once — returns (context_dict, searchable_text).

    context_dict includes a 'stats' sub-dict with message counts,
    tool use counts, duration, and classification.
    """
    user_messages = []
    assistant_texts = []
    tool_names = []
    search_parts = []
    total_lines = 0

    # Counters for stats
    assistant_count = 0
    tool_result_count = 0
    system_count = 0
    summary_count = 0
    progress_count = 0
    timestamps = []

    try:
        with open(session_file) as fh:
            for line_num, line in enumerate(fh):
                total_lines = line_num
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                entry_type = obj.get("type", "")

                # Collect timestamps for duration calc
                ts = obj.get("timestamp")
                if ts:
                    try:
                        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        timestamps.append(dt.timestamp())
                    except (ValueError, TypeError):
                        pass

                if entry_type == "user":
                    msg = obj.get("message", {}).get("content", "")
                    text = ""
                    if isinstance(msg, str):
                        text = msg
                        search_parts.append(msg)
                    elif isinstance(msg, list):
                        for c in msg:
                            if isinstance(c, dict) and c.get("type") == "text":
                                t = c.get("text", "")
                                search_parts.append(t)
                                text = t
                                break
                    text = text.strip()
                    if text and not text.startswith("[Request interrupted"):
                        limit = 1000 if deep else 500
                        user_messages.append(text[:limit])

                elif entry_type == "assistant":
                    assistant_count += 1
                    amsg = obj.get("message", {})
                    if isinstance(amsg, dict):
                        content = amsg.get("content", [])
                        if isinstance(content, list):
                            for c in content:
                                if isinstance(c, dict) and c.get("type") == "tool_use":
                                    name = c.get("name", "")
                                    tool_names.append(name)
                                    search_parts.append(name)
                                    inp = c.get("input", {})
                                    if isinstance(inp, dict):
                                        for v in inp.values():
                                            if isinstance(v, str):
                                                search_parts.append(v)
                                elif isinstance(c, dict) and c.get("type") == "text":
                                    t = c.get("text", "")
                                    search_parts.append(t)
                                    limit = 500 if deep else 300
                                    assistant_texts.append(t[:limit])

                elif entry_type == "tool_result":
                    tool_result_count += 1
                    content = obj.get("content", "")
                    if isinstance(content, str):
                        search_parts.append(content)
                    elif isinstance(content, list):
                        for c in content:
                            if isinstance(c, dict) and c.get("type") == "text":
                                search_parts.append(c.get("text", ""))

                elif entry_type == "system":
                    system_count += 1
                elif entry_type == "summary":
                    summary_count += 1
                elif entry_type == "progress":
                    progress_count += 1
    except Exception:
        pass

    if deep:
        first_msgs = user_messages[:4]
        last_msgs = user_messages[-8:] if len(user_messages) > 4 else []
        first_asst = assistant_texts[:3]
        last_asst = assistant_texts[-5:] if len(assistant_texts) > 3 else []
    else:
        first_msgs = user_messages[:2]
        last_msgs = user_messages[-6:] if len(user_messages) > 2 else []
        first_asst = []
        last_asst = assistant_texts[-3:] if len(assistant_texts) > 0 else []

    duration = (timestamps[-1] - timestamps[0]) if len(timestamps) >= 2 else 0

    stats = {
        "user_messages": len(user_messages),
        "assistant_messages": assistant_count,
        "tool_uses": len(tool_names),
        "tool_results": tool_result_count,
        "system_entries": system_count,
        "summary_entries": summary_count,
        "progress_entries": progress_count,
        "total_lines": total_lines,
        "duration_secs": round(duration, 1),
        "duration_fmt": _format_duration(duration),
        "has_progress": progress_count > 0,
    }
    stats["classification"] = classify_session(stats)

    context = {
        "first_messages": first_msgs,
        "last_messages": last_msgs,
        "first_assistant": first_asst,
        "last_assistant": last_asst,
        "recent_tools": tool_names[-15:],
        "all_tools": list(set(tool_names)),
        "total_user_messages": len(user_messages),
        "total_lines": total_lines,
        "stats": stats,
    }

    search_text = " ".join(search_parts)
    return context, search_text
