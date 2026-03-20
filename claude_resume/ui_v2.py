"""V2 TUI for claude-resume — three-pane drill-down with instant keypress navigation.

Layout:
  ┌─────────────┬─────────────────────────────────────────────┐
  │  Groups /   │  Preview (scrollable, PgUp/PgDn)            │
  │  Sessions   │                                             │
  │  (left)     │  Goal, status, last messages, files...      │
  │             │                                             │
  └─────────────┴─────────────────────────────────────────────┘

Navigation:
  - Number keys: instant select (no Enter needed)
  - Escape: back one level (preview → sessions → groups → quit)
  - Enter/r: resume session from preview
  - PgUp/PgDn/arrows: scroll preview
  - /: search
  - q: quit from any level
"""

import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Footer, Header, Input, ListItem, ListView, Static

from .sessions import (
    SessionCache,
    find_all_sessions,
    find_recent_sessions,
    relative_time,
    shorten_path,
)

import re


def _clean_title(title: str) -> str:
    if not title:
        return ""
    title = re.sub(r'<[^>]+>', '', title)
    title = re.sub(r'\s+', ' ', title).strip()
    return title


def esc(text: str) -> str:
    """Escape [ chars so Rich never interprets user content as markup."""
    return str(text).replace("[", "\\[")


# ── Resumability score ──────────────────────────────────────────


import math
import time as _time


def resumability_score(session: dict, cache: SessionCache) -> float:
    """Score how likely the user wants to resume this session (0–100).

    Signals (weighted):
      1. Total engagement  (35%) — file_size + user_messages + tool_uses
      2. Recency           (30%) — exponential decay from mtime
      3. Unfinished-ness   (25%) — last_entry_type, no bookmark-done
      4. Classification    (10%) — interactive > automated

    Bookmark override:
      - lifecycle_state == "done" → hard cap at 5 (finished work is not resume-worthy)
      - lifecycle_state == "blocked" → floor at 60 (needs attention)
      - lifecycle_state == "paused" → floor at 40 (intentional pause, still relevant)
    """
    sid = session["session_id"]
    ck = cache.cache_key(session["file"])

    # ── Bookmark overrides ──
    bookmark = cache.get(sid, ck, "bookmark")
    lifecycle = None
    if bookmark and isinstance(bookmark, dict):
        lifecycle = bookmark.get("lifecycle_state")

    if lifecycle == "done":
        return 5.0  # Explicitly finished — almost never want to resume

    # ── Signal 1: Total engagement (35 points max) ──
    stats = cache.get(sid, ck, "stats")
    engagement = 0.0

    if stats and isinstance(stats, dict):
        # File size: log scale, 100KB=5, 1MB=15, 10MB=25, 50MB+=35
        size = stats.get("file_size", session.get("size", 0))
        if size > 0:
            engagement += min(35, 5 * math.log10(max(size, 1000) / 1000) + 5)

        # Bonus for high interaction density
        user_msgs = stats.get("user_messages", 0)
        tool_uses = stats.get("tool_uses", 0)
        if user_msgs > 20:
            engagement += 5
        if tool_uses > 50:
            engagement += 5
    else:
        # Fallback: raw file size from discovery
        size = session.get("size", 0)
        if size > 0:
            engagement += min(35, 5 * math.log10(max(size, 1000) / 1000) + 5)

    engagement = min(engagement, 35)

    # ── Signal 2: Recency (30 points max) ──
    # Exponential decay: half-life of 4 hours
    age_hours = (_time.time() - session["mtime"]) / 3600
    recency = 30.0 * math.exp(-0.693 * age_hours / 4)  # ln(2)/half_life

    # ── Signal 3: Unfinished-ness (25 points max) ──
    unfinished = 0.0
    last_type = session.get("last_entry_type", "")

    # The more "mid-stream" the last entry, the more unfinished
    unfinished_map = {
        "user": 25,       # Typed something, never got response — maximum urgency
        "progress": 22,   # Mid-tool-execution crash
        "tool_result": 20, # Tool done but no response
        "assistant": 10,  # Normal conversation — could have been mid-thought
        "summary": 5,     # Context compaction — long session, probably fine
    }
    unfinished = unfinished_map.get(last_type, 8)

    # If there's no summary cached, the session hasn't been reviewed = more urgent
    summary = cache.get(sid, ck, "summary")
    if summary and isinstance(summary, dict):
        state = summary.get("state", "")
        # If state says "done", "completed", "finished" — lower unfinished score
        if state and any(w in state.lower() for w in ("done", "completed", "finished", "wrapped up")):
            unfinished = max(unfinished - 10, 0)

    # ── Signal 4: Classification (10 points max) ──
    classification = 0.0
    if stats and isinstance(stats, dict):
        cls = stats.get("classification", "pending")
        if cls == "interactive":
            classification = 10
        elif cls == "automated":
            classification = 2  # Bot sessions rarely need resuming
        else:
            classification = 5

    # ── Combine ──
    score = engagement + recency + unfinished + classification

    # ── Bookmark floors ──
    if lifecycle == "blocked":
        score = max(score, 60)
    elif lifecycle == "paused":
        score = max(score, 40)
    elif lifecycle == "handing-off":
        score = max(score, 50)

    return min(round(score, 1), 100)


def score_bar(score: float, width: int = 20) -> str:
    """Return a colored ASCII bar ████░░░░ with score number."""
    filled = round(score / 100 * width)
    empty = width - filled

    if score >= 70:
        color = "red"
    elif score >= 45:
        color = "yellow"
    elif score >= 20:
        color = "green"
    else:
        color = "dim"

    bar = "█" * filled + "░" * empty
    return f"[{color}]{bar}[/] [{color}]{score:.0f}[/]"


def score_label(score: float) -> str:
    """Return a colored Rich markup label for a resumability score."""
    if score >= 70:
        return f"[bold red]{score:.0f}[/]"
    elif score >= 45:
        return f"[bold yellow]{score:.0f}[/]"
    elif score >= 20:
        return f"[dim]{score:.0f}[/]"
    else:
        return f"[dim]{score:.0f}[/]"


# ── Data helpers ────────────────────────────────────────────────


def _get_cached_title(cache: SessionCache, s: dict) -> str:
    sid = s["session_id"]
    ck = cache.cache_key(s["file"])
    cached = cache.get(sid, ck, "summary")
    title = ""
    if cached and isinstance(cached, dict):
        title = cached.get("title", "")
    if not title:
        data = cache._read(sid)
        summary = data.get("summary")
        if isinstance(summary, dict):
            title = summary.get("title", "")
    return _clean_title(title)


def _get_cached_summary(cache: SessionCache, s: dict) -> dict | None:
    sid = s["session_id"]
    ck = cache.cache_key(s["file"])
    cached = cache.get(sid, ck, "summary")
    if cached and isinstance(cached, dict):
        return cached
    data = cache._read(sid)
    summary = data.get("summary")
    if isinstance(summary, dict):
        return summary
    return None


def _org_from_path(project: str) -> str:
    """Extract org group from a shortened project path."""
    parts = project.split("/")
    if len(parts) >= 2 and parts[1].startswith("repos-"):
        return parts[1]
    elif parts[0] == "~" and len(parts) == 1:
        return "~ (home)"
    else:
        return parts[1] if len(parts) > 1 else parts[0]


def _repo_from_path(project: str) -> str:
    """Extract repo name from a shortened project path."""
    parts = project.split("/")
    # ~/repos-org/repo-name → repo-name
    if len(parts) >= 3:
        return parts[2]
    elif len(parts) == 2:
        return parts[1]
    return parts[0]


def _group_sessions(sessions: list) -> list[tuple[str, list[dict]]]:
    """Group sessions by repo org, sorted by most recent."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for s in sessions:
        project = shorten_path(s["project_dir"])
        org = _org_from_path(project)
        groups[org].append(s)

    return sorted(
        groups.items(),
        key=lambda kv: max(s["mtime"] for s in kv[1]),
        reverse=True,
    )


def _subgroup_sessions(sessions: list) -> list[tuple[str, list[dict]]]:
    """Sub-group sessions by repo name within an org, sorted by most recent."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for s in sessions:
        project = shorten_path(s["project_dir"])
        repo = _repo_from_path(project)
        groups[repo].append(s)

    return sorted(
        groups.items(),
        key=lambda kv: max(s["mtime"] for s in kv[1]),
        reverse=True,
    )


LLM_MAX_INPUT_CHARS = 4000  # ~1K tokens — safe under Gemma 2B's 8K context window


def _cap_context(text: str, max_chars: int = LLM_MAX_INPUT_CHARS) -> str:
    """Cap context to max_chars. Keep first 1K + last 3K for long windows."""
    if not text or len(text) <= max_chars:
        return text
    head = max_chars // 4
    tail = max_chars - head
    return text[:head] + "\n\n[... middle truncated ...]\n\n" + text[-tail:]


def _extract_window_context(filepath) -> dict[str, str]:
    """Extract raw conversation context for 3 time windows relative to session end.

    Returns {"5m": "user: ...\nassistant: ...", "30m": "...", "2h": "..."}
    Each value is a condensed transcript suitable for LLM summarization.
    """
    from datetime import datetime

    WINDOWS = {"5m": 5, "30m": 30, "2h": 120}
    raw: dict[str, list[str]] = {k: [] for k in WINDOWS}

    try:
        with open(filepath, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            read_from = max(0, size - 524288)  # last 512KB
            f.seek(read_from)
            chunk = f.read().decode("utf-8", errors="replace")
            lines = chunk.strip().split("\n")
    except OSError:
        return {}

    # Find last timestamp
    last_ts = None
    for line in reversed(lines):
        try:
            entry = json.loads(line)
            ts = entry.get("timestamp")
            if ts:
                last_ts = datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
                break
        except (json.JSONDecodeError, ValueError):
            continue

    if last_ts is None:
        return {}

    # Collect entries per window (iterate forward for chronological order)
    for line in lines:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue

        ts = entry.get("timestamp")
        if not ts:
            continue
        try:
            entry_ts = datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
        except ValueError:
            continue

        age_min = (last_ts - entry_ts) / 60
        entry_type = entry.get("type", "")

        for key, minutes in WINDOWS.items():
            if age_min > minutes:
                continue

            if entry_type == "user":
                msg = entry.get("message", {})
                content = msg.get("content", "") if isinstance(msg, dict) else ""
                text = ""
                if isinstance(content, str) and len(content) > 5:
                    text = content[:300]
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text = part.get("text", "")[:300]
                            break
                if text:
                    raw[key].append(f"USER: {text}")

            elif entry_type == "assistant":
                msg = entry.get("message", {})
                content = msg.get("content", "") if isinstance(msg, dict) else ""
                text = ""
                if isinstance(content, str):
                    text = content[:300]
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text = part.get("text", "")[:300]
                            break
                if text:
                    raw[key].append(f"ASSISTANT: {text}")

            elif entry_type == "tool_use":
                name = entry.get("tool_name", entry.get("name", "?"))
                raw[key].append(f"TOOL: {name}")

    # Trim each window to a reasonable size for the LLM prompt
    result = {}
    for key in WINDOWS:
        entries = raw[key]
        if not entries:
            result[key] = ""
            continue
        # Keep last N entries per window, cap for LLM input safety
        trimmed = entries[-30:]  # last 30 events max
        text = "\n".join(trimmed)
        if len(text) > LLM_MAX_INPUT_CHARS:
            head = LLM_MAX_INPUT_CHARS // 4
            tail = LLM_MAX_INPUT_CHARS - head
            text = text[:head] + "\n\n[... middle truncated ...]\n\n" + text[-tail:]
        result[key] = text

    return result



def _window_summary_adapter(context: str) -> str:
    """Generate a window summary from context text.

    Adapter — swap this implementation when a local model is ready.
    Currently: extract the last meaningful user message from the context (~500 chars).
    Future: call local ONNX/GGUF model here, no claude -p, no network.
    """
    # Pull last user: line from context
    last_user = ""
    for line in reversed(context.splitlines()):
        line = line.strip()
        if line.lower().startswith("user:"):
            last_user = line[5:].strip()
            break
    text = last_user or context.strip()
    # Trim to ~500 chars at a word boundary
    if len(text) > 500:
        text = text[:500].rsplit(" ", 1)[0] + "…"
    return text or "no activity"


def _summarize_single_window(
    key: str,
    context: str,
    session_id: str,
    cache: SessionCache,
    filepath,
) -> str:
    """Generate summary for ONE time window. Returns the summary string.
    Writes to cache incrementally (merges into existing window_summaries dict).
    """
    if not context:
        return "no activity"

    summary = _window_summary_adapter(_cap_context(context))

    # Merge into cached window_summaries dict
    ck = cache.cache_key(filepath)
    existing = cache.get(session_id, ck, "window_summaries")
    if not existing or not isinstance(existing, dict):
        existing = {}
    existing[key] = summary
    cache.set(session_id, ck, "window_summaries", existing)

    return summary


def _extract_last_messages(filepath, max_msgs: int = 8) -> list[str]:
    """Extract last N user messages from a session JSONL file."""
    try:
        with open(filepath, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            read_from = max(0, size - 131072)  # last 128KB
            f.seek(read_from)
            chunk = f.read().decode("utf-8", errors="replace")
            lines = chunk.strip().split("\n")
    except OSError:
        return []

    user_msgs = []
    for line in reversed(lines):
        try:
            entry = json.loads(line)
            if entry.get("type") == "user" and isinstance(entry.get("message"), dict):
                content = entry["message"].get("content", "")
                if isinstance(content, str) and len(content) > 5:
                    user_msgs.append(content[:200])
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text = part.get("text", "")
                            if len(text) > 5:
                                user_msgs.append(text[:200])
                                break
        except (json.JSONDecodeError, KeyError):
            continue
        if len(user_msgs) >= max_msgs:
            break

    return list(reversed(user_msgs))


# ── Widgets ─────────────────────────────────────────────────────


class NavItem(ListItem):
    """A numbered navigation item in the left pane."""

    def __init__(self, number: int, label: str, sublabel: str = "",
                 bar: str = "", **kwargs) -> None:
        super().__init__(**kwargs)
        self.number = number
        self.label_text = label
        self.sublabel_text = sublabel
        self.bar_markup = bar  # Pre-built Rich markup, not escaped

    def compose(self) -> ComposeResult:
        line1 = f"[bold yellow]{self.number}[/]  [bold]{esc(self.label_text)}[/]"
        if self.bar_markup:
            line1 += f"\n   {self.bar_markup}"
        if self.sublabel_text:
            line1 += f"\n   [dim]{esc(self.sublabel_text)}[/]"
        yield Static(line1)


class SearchResultItem(ListItem):
    """A search result in the left pane."""

    def __init__(self, number: int, title: str, project: str, age: str, hits: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.number = number
        self.title_text = title
        self.project_text = project
        self.age_text = age
        self.hits = hits

    def compose(self) -> ComposeResult:
        title = esc(self.title_text) if self.title_text else "[dim]Untitled[/]"
        yield Static(
            f"[bold yellow]{self.number}[/]  {title}\n"
            f"   [cyan]{esc(self.project_text)}[/]  [dim]{self.age_text}  ({self.hits} hit{'s' if self.hits != 1 else ''})[/]"
        )


# ── Messages ────────────────────────────────────────────────────


class SessionChosen(Message):
    """User wants to resume a session."""
    def __init__(self, session: dict) -> None:
        super().__init__()
        self.session = session


class SearchDone(Message):
    """Background search completed."""
    def __init__(self, results: list[tuple[dict, int]]) -> None:
        super().__init__()
        self.results = results


class ScoresReady(Message):
    """Background score computation finished."""
    def __init__(self, scores: dict[str, float]) -> None:
        super().__init__()
        self.scores = scores


# ── Main TUI App ────────────────────────────────────────────────


class ResumeV2App(App):
    """Three-pane session browser with instant keypress navigation."""

    CSS = """
    Screen { layout: vertical; }

    #main-layout { height: 1fr; }
    #nav-pane { width: 38%; border-right: heavy $primary; }
    #nav-list { height: 1fr; }
    #preview-pane { width: 62%; padding: 1 2; overflow-y: auto; }
    #preview-content { width: 100%; }
    #breadcrumb { dock: top; height: 1; padding: 0 1; background: $primary-background; }
    #search-input { dock: top; margin: 0 1; display: none; }

    .nav-hint { dock: bottom; height: 1; padding: 0 1; }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("slash", "search", "Search"),
        ("escape", "back", "Back"),
        ("v", "noop", "VS Code"),
        ("r", "noop", "iTerm2"),
    ]

    def action_noop(self) -> None:
        pass  # handled in on_key

    # State
    level = reactive("groups")  # "groups", "repos", "sessions", "search"
    _current_group_idx: int = 0
    _current_org: str = ""
    _current_repo_groups: list[tuple[str, list[dict]]] = []

    def __init__(
        self,
        hours: int = 48,
        search_term: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.hours = hours
        self.search_term = search_term
        self.cache = SessionCache()
        self.sessions: list[dict] = []
        self.grouped: list[tuple[str, list[dict]]] = []
        self.result_session: dict | None = None
        self._search_results: list[tuple[dict, int]] = []
        self._current_items: list[dict] = []  # sessions shown in nav pane
        self._scores: dict[str, float] = {}  # session_id -> score (pre-computed)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield Static("", id="breadcrumb")
        yield Input(placeholder="Search all sessions...", id="search-input")
        with Horizontal(id="main-layout"):
            with Vertical(id="nav-pane"):
                yield ListView(id="nav-list")
            with VerticalScroll(id="preview-pane"):
                yield Static("", id="preview-content", markup=True)
        yield Footer()

    def on_mount(self) -> None:
        self.title = "cr v2"
        self.sub_title = "Claude Code Session Browser"
        self._load_sessions()

        if self.search_term:
            self._start_search(self.search_term)
        else:
            self._show_groups()

        # Pre-compute all scores in background — UI uses _scores dict for instant lookup
        self._precompute_scores_bg()

    def _load_sessions(self) -> None:
        self.sessions = find_recent_sessions(self.hours, max_sessions=200)
        self.grouped = _group_sessions(self.sessions)

    @work(thread=True)
    def _precompute_scores_bg(self) -> None:
        """Pre-compute resumability scores for all sessions off the UI thread."""
        scores = {}
        for s in self.sessions:
            scores[s["session_id"]] = resumability_score(s, self.cache)
        self.post_message(ScoresReady(scores))

    def _get_score(self, session: dict) -> float:
        """Get pre-computed score (instant) or fall back to 0 if not yet computed."""
        return self._scores.get(session["session_id"], 0.0)

    def on_scores_ready(self, message: ScoresReady) -> None:
        """Scores finished computing — update the dict and refresh current view."""
        self._scores = message.scores
        # Re-render the current level to show real scores
        if self.level == "groups":
            self._show_groups()
        elif self.level == "repos":
            self._show_repos(self._current_group_idx)
        elif self.level == "sessions":
            # find repo idx from current state and refresh
            pass  # scores will show on next navigation

    # ── Level: Groups ───────────────────────────────────────

    def _show_groups(self) -> None:
        self.level = "groups"
        self._update_breadcrumb("Groups")

        lv = self.query_one("#nav-list", ListView)
        for child in list(lv.children):
            child.remove()

        items = []
        for i, (org, group_sessions) in enumerate(self.grouped, 1):
            most_recent = max(s["mtime"] for s in group_sessions)
            age = relative_time(most_recent)
            count = len(group_sessions)
            # Show the highest-scoring session's title
            top = max(group_sessions, key=lambda s: self._get_score(s))
            top_score = self._get_score(top)
            title = _get_cached_title(self.cache, top)
            sublabel = f"{count} session{'s' if count != 1 else ''}  •  {age}"
            if title:
                sublabel += f"  •  {title[:40]}"
            items.append(NavItem(i, org, sublabel, bar=score_bar(top_score)))

        if items:
            lv.mount(*items)
            lv.index = 0

        self._update_preview_for_group(0)
        lv.focus()

    # ── Level: Repos (sub-clusters within an org) ─────────

    def _show_repos(self, group_idx: int) -> None:
        self._current_group_idx = group_idx
        org, group_sessions = self.grouped[group_idx]
        self._current_org = org
        self._current_repo_groups = _subgroup_sessions(group_sessions)

        # If only 1 repo in the org, skip straight to sessions
        if len(self._current_repo_groups) == 1:
            self._show_sessions_for_repo(0)
            return

        self.level = "repos"
        self._update_breadcrumb(f"Groups > [bold cyan]{esc(org)}[/]")

        lv = self.query_one("#nav-list", ListView)
        for child in list(lv.children):
            child.remove()

        # Sort repo groups by top resumability score
        self._current_repo_groups.sort(
            key=lambda kv: max(self._get_score(s) for s in kv[1]),
            reverse=True,
        )

        items = []
        for i, (repo, repo_sessions) in enumerate(self._current_repo_groups, 1):
            most_recent = max(s["mtime"] for s in repo_sessions)
            age = relative_time(most_recent)
            count = len(repo_sessions)
            top = max(repo_sessions, key=lambda s: self._get_score(s))
            top_score = self._get_score(top)
            title = _get_cached_title(self.cache, top)
            sublabel = f"{count} session{'s' if count != 1 else ''}  •  {age}"
            if title:
                sublabel += f"  •  {title[:35]}"
            items.append(NavItem(i, repo, sublabel, bar=score_bar(top_score)))

        if items:
            lv.mount(*items)
            lv.index = 0

        self._update_preview_for_repo_group(0)
        lv.focus()

    def _update_preview_for_repo_group(self, repo_idx: int) -> None:
        if repo_idx < 0 or repo_idx >= len(self._current_repo_groups):
            return
        repo, repo_sessions = self._current_repo_groups[repo_idx]
        repo_sessions_sorted = sorted(repo_sessions, key=lambda s: s["mtime"], reverse=True)

        text = f"[bold underline]{esc(self._current_org)} / {esc(repo)}[/]  [dim]{len(repo_sessions)} sessions[/]\n\n"

        for i, s in enumerate(repo_sessions_sorted[:10], 1):
            title = _get_cached_title(self.cache, s)
            age = relative_time(s["mtime"])
            display = title[:60] if title else shorten_path(s["project_dir"])
            text += f"[bold yellow]{i:2d}[/]  {esc(display)}  [dim]{age}[/]\n"
            text += f"    [dim]{s['session_id'][:8]}[/]\n\n"

        if len(repo_sessions) > 10:
            text += f"[dim]... and {len(repo_sessions) - 10} more[/]\n"

        self._update_preview_text(text)

    # ── Level: Sessions (within a repo) ─────────────────────

    def _show_sessions_for_repo(self, repo_idx: int) -> None:
        repo, repo_sessions = self._current_repo_groups[repo_idx]
        # Sort by resumability score, highest first
        repo_sessions.sort(key=lambda s: self._get_score(s), reverse=True)
        self._current_items = repo_sessions
        self.level = "sessions"
        self._update_breadcrumb(
            f"Groups > [bold cyan]{esc(self._current_org)}[/] > [bold]{esc(repo)}[/]"
        )

        lv = self.query_one("#nav-list", ListView)
        for child in list(lv.children):
            child.remove()

        items = []
        for i, s in enumerate(repo_sessions, 1):
            title = _get_cached_title(self.cache, s)
            age = relative_time(s["mtime"])
            score = self._get_score(s)
            label = title[:50] if title else shorten_path(s["project_dir"])
            sublabel = f"{age}  •  {s['session_id'][:8]}"
            items.append(NavItem(i, label, sublabel, bar=score_bar(score)))

        if items:
            lv.mount(*items)
            lv.index = 0

        self._update_preview_for_session(repo_sessions[0] if repo_sessions else None)
        lv.focus()

    # ── Level: Search ───────────────────────────────────────

    def _start_search(self, term: str) -> None:
        self.level = "search"
        self._update_breadcrumb(f"Search: [bold yellow]{esc(term)}[/]")
        self._update_preview_text("[dim]Searching...[/]")
        self._run_search_bg(term)

    @work(thread=True)
    def _run_search_bg(self, term: str) -> None:
        """Search all sessions in background with ThreadPoolExecutor."""
        all_sessions = find_all_sessions()
        term_bytes = term.lower().encode("utf-8", errors="replace")

        def check(s):
            try:
                raw = s["file"].read_bytes()
            except OSError:
                return None
            raw_lower = raw.lower()
            if term_bytes not in raw_lower:
                return None
            return (s, raw_lower.count(term_bytes))

        with ThreadPoolExecutor(max_workers=16) as pool:
            results = list(pool.map(check, all_sessions))

        matches = [r for r in results if r is not None]
        matches.sort(key=lambda r: r[0]["mtime"], reverse=True)
        self.post_message(SearchDone(matches))

    def on_search_done(self, message: SearchDone) -> None:
        self._search_results = message.results
        self._current_items = [r[0] for r in message.results]
        count = len(message.results)

        lv = self.query_one("#nav-list", ListView)
        for child in list(lv.children):
            child.remove()

        items = []
        for i, (s, hits) in enumerate(message.results[:50], 1):  # cap at 50
            title = _get_cached_title(self.cache, s)
            age = relative_time(s["mtime"])
            project = shorten_path(s["project_dir"])
            items.append(SearchResultItem(i, title, project, age, hits))

        if items:
            lv.mount(*items)
            lv.index = 0
            self._update_preview_for_session(message.results[0][0])
        else:
            self._update_preview_text(f"[dim]No sessions found matching the search term.[/]")

        self._update_breadcrumb(
            f"Search: [bold yellow]{esc(self.search_term or '')}[/]  "
            f"[dim]({count} result{'s' if count != 1 else ''})[/]"
        )
        lv.focus()

    # ── Preview rendering ───────────────────────────────────

    def _update_preview_for_group(self, group_idx: int) -> None:
        if group_idx < 0 or group_idx >= len(self.grouped):
            return
        org, group_sessions = self.grouped[group_idx]
        group_sessions_sorted = sorted(group_sessions, key=lambda s: s["mtime"], reverse=True)

        text = f"[bold underline]{esc(org)}[/]  [dim]{len(group_sessions)} sessions[/]\n\n"

        for i, s in enumerate(group_sessions_sorted[:10], 1):
            title = _get_cached_title(self.cache, s)
            age = relative_time(s["mtime"])
            project = shorten_path(s["project_dir"])
            display = title[:60] if title else project

            text += f"[bold yellow]{i:2d}[/]  {esc(display)}  [dim]{age}[/]\n"
            text += f"    [cyan]{esc(project)}[/]  [dim]{s['session_id'][:8]}[/]\n\n"

        if len(group_sessions) > 10:
            text += f"[dim]... and {len(group_sessions) - 10} more[/]\n"

        self._update_preview_text(text)

    def _update_preview_for_session(self, s: dict | None) -> None:
        if s is None:
            self._update_preview_text("[dim]No session selected[/]")
            return

        self._build_preview_bg(s)

    def _build_preview_text(self, s: dict, window_sums: dict[str, str] | None = None) -> str:
        """Build preview markup from cached data only — no I/O, no ML. Instant."""
        sid = s["session_id"]
        project = shorten_path(s["project_dir"])
        age = relative_time(s["mtime"])
        title = _get_cached_title(self.cache, s)
        summary = _get_cached_summary(self.cache, s)
        score = self._scores.get(sid, 0.0)

        text = f"[bold underline]{esc(title or 'Untitled session')}[/]  {score_label(score)}\n\n"
        text += f"[bold]Directory:[/]   [cyan]{esc(project)}[/]\n"
        text += f"[bold]Last active:[/] {age}\n"
        text += f"[bold]Resumability:[/] {score_label(score)} / 100\n"
        text += f"[bold]Session:[/]     [dim]{sid}[/]\n"

        size_mb = s.get("size", 0) / (1024 * 1024)
        if size_mb > 0:
            text += f"[bold]Size:[/]        {size_mb:.1f} MB\n"

        # Active time / focus (daemon-cached)
        ck_at = self.cache.cache_key(s["file"])
        active_time = self.cache.get(sid, ck_at, "active_time")
        if active_time and isinstance(active_time, dict) and active_time.get("active_seconds"):
            active_s = active_time["active_seconds"]
            total_s = active_time["total_seconds"]
            focus = active_time.get("focus_pct", 0)
            active_h, active_m = divmod(active_s // 60, 60)
            total_h, total_m = divmod(total_s // 60, 60)
            active_str = f"{active_h}h{active_m:02d}m" if active_h else f"{active_m}m"
            total_str = f"{total_h}h{total_m:02d}m" if total_h else f"{total_m}m"
            focus_color = "green" if focus >= 70 else ("yellow" if focus >= 40 else "red")
            text += f"[bold]Focus:[/]       [{focus_color}]{focus:.0f}%[/] active  ({active_str} of {total_str})\n"

        if summary:
            goal = summary.get("goal", "")
            what = summary.get("what_was_done", "")
            state = summary.get("state", "")
            files = summary.get("files", [])

            if goal:
                text += f"\n[bold]Goal:[/]\n{esc(_clean_title(goal))}\n"
            if what:
                text += f"\n[bold]What was done:[/]\n{esc(_clean_title(what))}\n"
            if state:
                text += f"\n[bold]Where you left off:[/]\n{esc(_clean_title(state))}\n"
            if files:
                text += "\n[bold]Key files:[/]\n"
                for f in files[:8]:
                    text += f"  [dim]•[/] {esc(str(f))}\n"

        ck = self.cache.cache_key(s["file"])
        bookmark = self.cache.get(sid, ck, "bookmark")
        if bookmark and isinstance(bookmark, dict):
            blockers = bookmark.get("blockers", [])
            if blockers:
                text += "\n[bold red]Blockers:[/]\n"
                for b in blockers:
                    desc = b.get("description", str(b)) if isinstance(b, dict) else str(b)
                    text += f"  [red]•[/] {esc(desc)}\n"
            next_actions = bookmark.get("next_actions", [])
            if next_actions:
                text += "\n[bold]Next actions:[/]\n"
                for i, a in enumerate(next_actions, 1):
                    text += f"  {i}. {esc(a)}\n"

        deep = self.cache.get(sid, ck, "deep_summary")
        if deep and isinstance(deep, dict):
            for field, label in [("objective", "Objective"), ("progress", "Progress"), ("next_steps", "Next steps")]:
                val = deep.get(field, "")
                if val:
                    text += f"\n[bold]{label}:[/]\n{esc(val)}\n"
            decisions = deep.get("decisions_made", [])
            if decisions:
                text += "\n[bold]Decisions:[/]\n"
                for d in decisions:
                    text += f"  [dim]•[/] {esc(d)}\n"

        # Activity timeline — show each window: cached value, passed-in value, or placeholder
        text += "\n[bold magenta]━━━ Activity Timeline ━━━[/]\n\n"
        # Merge: passed-in window_sums take priority, then cache, then placeholder
        ck_ws = self.cache.cache_key(s["file"])
        cached_wins = self.cache.get(sid, ck_ws, "window_summaries")
        if not isinstance(cached_wins, dict):
            cached_wins = {}
        merged = {**cached_wins, **(window_sums or {})}

        for label, key in [("Last 5 min", "5m"), ("Last 30 min", "30m"), ("Last 2 hours", "2h")]:
            val = merged.get(key)
            if val:
                text += f"  [bold]{label}:[/]  {esc(val)}\n"
            else:
                text += f"  [bold]{label}:[/]  [dim italic]generating...[/]\n"

        # Last messages — read from JSONL tail (fast I/O, no ML)
        msgs = _extract_last_messages(s["file"])
        if msgs:
            text += "\n[bold yellow]━━━ Last messages ━━━[/]\n\n"
            for msg in msgs:
                cleaned = esc(_clean_title(msg))
                text += f"  [yellow]>[/] {cleaned}\n\n"

        text += "\n[dim]v = VS Code  •  r = iTerm2  •  Esc = back  •  PgUp/PgDn = scroll[/]\n"
        return text

    @work(thread=True, exclusive=True, group="preview")
    def _build_preview_bg(self, s: dict) -> None:
        """Phase 1: instant cache-only preview. Phase 2: generate each window one at a time."""
        sid = s["session_id"]

        # Phase 1 — instant: render from cache only
        text = self._build_preview_text(s)
        self.call_from_thread(self._update_preview_text, text)

        # Phase 2 — generate missing window summaries one at a time, re-render after each
        ck = self.cache.cache_key(s["file"])
        cached_wins = self.cache.get(sid, ck, "window_summaries")
        if not isinstance(cached_wins, dict):
            cached_wins = {}

        # Extract contexts once (fast — just JSONL tail read)
        contexts = _extract_window_context(s["file"])
        if not any(contexts.values()):
            return

        # Generate each window sequentially: 5m → 30m → 2h
        for key in ("5m", "30m", "2h"):
            if cached_wins.get(key):
                continue  # already cached, skip

            _summarize_single_window(key, contexts.get(key, ""), sid, self.cache, s["file"])

            # Re-read merged cache and re-render so user sees the update
            updated_wins = self.cache.get(sid, ck, "window_summaries") or {}
            text = self._build_preview_text(s, window_sums=updated_wins)
            self.call_from_thread(self._update_preview_text, text)

    def _update_preview_text(self, text: str) -> None:
        preview = self.query_one("#preview-content", Static)
        try:
            preview.update(text)
        except Exception:
            preview.update(esc(text))
        self.query_one("#preview-pane", VerticalScroll).scroll_home(animate=False)

    def _update_breadcrumb(self, text: str) -> None:
        self.query_one("#breadcrumb", Static).update(f" {text}")

    # ── Navigation ──────────────────────────────────────────

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        lv = self.query_one("#nav-list", ListView)
        idx = lv.index
        if idx is None:
            return

        if self.level == "groups":
            self._update_preview_for_group(idx)
        elif self.level == "repos":
            self._update_preview_for_repo_group(idx)
        elif self.level in ("sessions", "search"):
            if idx < len(self._current_items):
                self._update_preview_for_session(self._current_items[idx])

    def on_key(self, event) -> None:
        # Escape — go back one level (intercept before ListView eats it)
        if event.key == "escape":
            if self.level == "sessions":
                if len(self._current_repo_groups) > 1:
                    self._show_repos(self._current_group_idx)
                else:
                    self._show_groups()
            elif self.level == "repos":
                self._show_groups()
            elif self.level == "search":
                si = self.query_one("#search-input", Input)
                if si.styles.display == "block":
                    si.styles.display = "none"
                self._show_groups()
            else:
                self.exit()
            event.prevent_default()
            event.stop()
            return

        # Number keys for instant selection
        if event.character and event.character.isdigit() and event.character != '0':
            num = int(event.character) - 1
            self._select_item(num)
            event.prevent_default()
            event.stop()
            return

        # Enter to drill in or resume
        if event.key == "enter":
            lv = self.query_one("#nav-list", ListView)
            idx = lv.index
            if idx is None:
                return
            if self.level == "groups":
                if idx < len(self.grouped):
                    self._show_repos(idx)
            elif self.level == "repos":
                if idx < len(self._current_repo_groups):
                    self._show_sessions_for_repo(idx)
            elif self.level in ("sessions", "search"):
                if idx < len(self._current_items):
                    self._launch_vscode_bg(self._current_items[idx])
            event.prevent_default()
            event.stop()
            return

        # r = resume in iTerm2, v = open in VS Code
        if event.character in ("r", "v") and self.level in ("sessions", "search"):
            lv = self.query_one("#nav-list", ListView)
            idx = lv.index
            if idx is not None and idx < len(self._current_items):
                s = self._current_items[idx]
                if event.character == "v":
                    self._launch_vscode_bg(s)
                else:
                    self._launch_iterm_bg(s)
            event.prevent_default()
            event.stop()
            return

        # Page up/down for preview scrolling
        if event.key == "pagedown":
            self.query_one("#preview-pane", VerticalScroll).scroll_page_down(animate=False)
            event.prevent_default()
            event.stop()
            return
        if event.key == "pageup":
            self.query_one("#preview-pane", VerticalScroll).scroll_page_up(animate=False)
            event.prevent_default()
            event.stop()
            return

    def _select_item(self, idx: int) -> None:
        """Handle number key selection — drill into the current level."""
        if self.level == "groups":
            if idx < len(self.grouped):
                self._show_repos(idx)
        elif self.level == "repos":
            if idx < len(self._current_repo_groups):
                self._show_sessions_for_repo(idx)
        elif self.level in ("sessions", "search"):
            if idx < len(self._current_items):
                # Highlight and preview the session
                lv = self.query_one("#nav-list", ListView)
                lv.index = idx
                self._update_preview_for_session(self._current_items[idx])

    @work(thread=True)
    def _launch_vscode_bg(self, s: dict) -> None:
        """v key — Open VS Code in project folder, resume in its integrated terminal."""
        import subprocess
        import shlex
        import time as _t

        project_dir = s["project_dir"]
        session_id = s["session_id"]
        resume_cmd = f"claude --resume {shlex.quote(session_id)} --dangerously-skip-permissions"

        try:
            subprocess.run(["code", project_dir], capture_output=True, timeout=5)
            _t.sleep(1.5)

            applescript = f'''
                tell application "Visual Studio Code"
                    activate
                end tell
                delay 0.5
                tell application "System Events"
                    tell process "Code"
                        keystroke "`" using control down
                        delay 0.3
                        keystroke {shlex.quote(resume_cmd)}
                        delay 0.1
                        key code 36
                    end tell
                end tell
            '''
            subprocess.run(["osascript", "-e", applescript], capture_output=True, timeout=10)

            title = _get_cached_title(self.cache, s) or session_id[:8]
            self.call_from_thread(self.notify, f"VS Code: {title}", title="Opened", severity="information")
        except Exception as e:
            self.call_from_thread(self.notify, f"VS Code failed: {e}", severity="error")

    @work(thread=True)
    def _launch_iterm_bg(self, s: dict) -> None:
        """r key — Open a new iTerm2 tab, cd to project, resume session."""
        import subprocess
        import shlex

        project_dir = s["project_dir"]
        session_id = s["session_id"]
        cmd = f"cd {shlex.quote(project_dir)} && claude --resume {shlex.quote(session_id)} --dangerously-skip-permissions"

        applescript = f'''
            tell application "iTerm2"
                activate
                tell current window
                    create tab with default profile
                    tell current session
                        write text {shlex.quote(cmd)}
                    end tell
                end tell
            end tell
        '''
        try:
            subprocess.run(["osascript", "-e", applescript], capture_output=True, timeout=5)
            title = _get_cached_title(self.cache, s) or session_id[:8]
            self.call_from_thread(self.notify, f"iTerm2: {title}", title="Resumed", severity="information")
        except Exception as e:
            self.call_from_thread(self.notify, f"iTerm2 failed: {e}", severity="error")

    async def action_back(self) -> None:
        if self.level == "sessions":
            # Go back to repos (or groups if only 1 repo)
            if len(self._current_repo_groups) > 1:
                self._show_repos(self._current_group_idx)
            else:
                self._show_groups()
        elif self.level == "repos":
            self._show_groups()
        elif self.level == "search":
            si = self.query_one("#search-input", Input)
            if si.styles.display == "block":
                si.styles.display = "none"
            self._show_groups()
        else:
            self.exit()

    def action_search(self) -> None:
        si = self.query_one("#search-input", Input)
        si.styles.display = "block"
        si.focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        term = event.value.strip()
        if term:
            self.search_term = term
            si = self.query_one("#search-input", Input)
            si.styles.display = "none"
            self._start_search(term)

    def on_input_changed(self, event: Input.Changed) -> None:
        pass  # Don't filter live — wait for submit


def run_v2(hours: int = 48, search_term: str | None = None) -> None:
    """Entry point for cr v2."""
    app = ResumeV2App(hours=hours, search_term=search_term)
    app.run()
