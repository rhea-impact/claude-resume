"""Textual TUI for the session picker."""

from datetime import datetime, timedelta
from enum import Enum, auto
from textual.app import App, ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Header, Footer, Static, Input, ListView, ListItem
from textual.message import Message
from textual import on, events, work

from .sessions import SessionOps, shorten_path, relative_time, get_label


def _get_date_group(mtime: float) -> str:
    """Bucket a timestamp into macOS Finder-style date groups."""
    now = datetime.now()
    dt = datetime.fromtimestamp(mtime)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday_start = today_start - timedelta(days=1)
    week_start = today_start - timedelta(days=7)
    month_start = today_start - timedelta(days=30)

    if dt >= today_start:
        return "Today"
    elif dt >= yesterday_start:
        return "Yesterday"
    elif dt >= week_start:
        return "Last 7 Days"
    elif dt >= month_start:
        return "Last 30 Days"
    else:
        return "Older"


class PreviewMode(Enum):
    SUMMARY = auto()
    PATTERNS = auto()


class SearchInput(Input):
    """Custom Input that emits Escaped message instead of letting Textual handle it."""

    class Escaped(Message):
        pass

    def key_escape(self) -> None:
        self.value = ""
        self.post_message(self.Escaped())


class DateHeader(ListItem):
    """Non-interactive date group separator."""

    def __init__(self, label: str) -> None:
        super().__init__()
        self.label_text = label

    def compose(self) -> ComposeResult:
        yield Static(f"[bold dim]── {self.label_text} ──[/]")


class SessionItem(ListItem):
    """A single session row in the list."""
    def __init__(self, idx: int, session: dict, summary: dict | None, has_deep: bool = False) -> None:
        super().__init__()
        self.idx = idx
        self.session = session
        self.summary = summary
        self.has_deep = has_deep

    def compose(self) -> ComposeResult:
        title = self.summary.get("title", "Unknown") if self.summary else "Summarizing..."
        short_path = shorten_path(self.session["project_dir"])
        age = relative_time(self.session["mtime"])
        badge = " [bold magenta]◆[/]" if self.has_deep else ""
        if self.summary:
            yield Static(f"[bold]{title}[/]{badge}\n[cyan]{short_path}[/]  [dim]{age}[/]")
        else:
            yield Static(f"[bold yellow]{title}[/] [dim]⟳[/]\n[cyan]{short_path}[/]  [dim]{age}[/]")


class TaskDone(Message):
    """Single message type for all background task completions."""
    def __init__(self, kind: str, idx: int, result=None, error: str | None = None) -> None:
        super().__init__()
        self.kind = kind
        self.idx = idx
        self.result = result
        self.error = error


class SessionPickerApp(App):
    """Full-screen session picker with search and preview."""

    CSS = """
    Screen { layout: vertical; }
    #search { dock: top; margin: 0 1; height: 3; }
    #main { height: 1fr; }
    #session-list { width: 45%; border-right: heavy $primary; }
    #preview-scroll { width: 55%; padding: 1 2; overflow-y: auto; }
    #preview-scroll.focused { border-left: heavy $accent; padding: 1 1; }
    #preview { width: 100%; }
    DateHeader { height: auto; padding: 1 0 0 1; }
    """

    BINDINGS = []

    def __init__(self, sessions: list, summaries: list, ops: SessionOps, **kwargs):
        super().__init__(**kwargs)
        self.sessions = sessions
        self.summaries = summaries
        self._ops = ops
        self.filtered_items: list[tuple[int, dict, dict | None]] = []
        self.result_data = None
        self._saved_session_idx = 0
        self._skip_permissions = True
        self._show_bots = False
        self._in_preview = False
        self._preview_mode = PreviewMode.SUMMARY
        self._lv_map: dict[int, int] = {}
        self._last_lv_index = 0
        self._search_index: list[str] = []
        self._pending: dict[str, set[int]] = {
            "summarize": set(), "deep": set(), "patterns": set(), "index": set(), "scan": set(),
        }

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield SearchInput(
            placeholder="/ search  ↑↓←→ navigate  Enter resume  d perms  D dive  p patterns  b bots  Esc quit",
            id="search",
        )
        with Horizontal(id="main"):
            yield ListView(id="session-list")
            with VerticalScroll(id="preview-scroll"):
                yield Static("", id="preview", markup=True)
        yield Footer()

    # ── Lifecycle ──────────────────────────────────────────

    def on_mount(self) -> None:
        self.title = "cc-restore"
        self.sub_title = "Claude Code Session Recovery"
        self._init_search_index()
        self._classify_uncached()
        self._populate_list()
        self.query_one("#session-list", ListView).focus()

    def _classify_uncached(self) -> None:
        """Quick-scan sessions that don't have stats cached yet.

        Uses get_label() which handles scan + classify + cache internally.
        Cheap enough to run synchronously for cached sessions, background for uncached.
        """
        unscanned = []
        cache = self._ops.cache
        for i, s in enumerate(self.sessions):
            ck = cache.cache_key(s["file"])
            if cache.get(s["session_id"], ck, "stats") is None:
                unscanned.append(i)
        if unscanned:
            self._classify_batch_bg(unscanned)

    @work(thread=True)
    def _classify_batch_bg(self, indices: list[int]) -> None:
        """Quick-scan a batch of sessions for classification. Single refresh at end."""
        for i in indices:
            s = self.sessions[i]
            try:
                # get_label computes and caches both label + stats in one pass
                get_label(s["file"], self._ops.cache)
            except Exception:
                pass
        self.post_message(TaskDone("scan", 0, None))

    def _init_search_index(self) -> None:
        """Cached values sync (fast), uncached in background."""
        self._search_index = []
        uncached = []
        for i, s in enumerate(self.sessions):
            ck = self._ops.cache.cache_key(s["file"])
            cached = self._ops.cache.get(s["session_id"], ck, "search_text")
            self._search_index.append(cached or "")
            if not cached:
                uncached.append(i)
        if uncached:
            self._index_batch_bg(uncached)

    # ── Background task engine ─────────────────────────────

    def _start_task(self, kind: str, idx: int) -> None:
        """Start a background task if not already running for this session."""
        if idx not in self._pending[kind]:
            self._pending[kind].add(idx)
            self._run_task(kind, idx)

    @work(thread=True)
    def _run_task(self, kind: str, idx: int) -> None:
        """Single generic worker for all task types."""
        s = self.sessions[idx]
        ops = self._ops
        ck = ops.cache.cache_key(s["file"])
        try:
            if kind == "summarize":
                context, search_text = ops.parse_session(s["file"])
                git = ops.get_git_context(s["project_dir"])
                summary = ops.summarize_quick(context, s["project_dir"], git)
                ops.cache.set(s["session_id"], ck, "summary", summary)
                full = (search_text + f" {s['project_dir']} {s['session_id']}").lower()
                ops.cache.set(s["session_id"], ck, "search_text", full)
                if "stats" in context:
                    ops.cache.set(s["session_id"], ck, "stats", context["stats"])
                self.post_message(TaskDone(kind, idx, {"summary": summary, "search_text": full}))

            elif kind == "scan":
                # get_label handles everything: scan, classify, cache
                get_label(s["file"], ops.cache)
                # Read back cached stats for UI
                scan = ops.cache.get(s["session_id"], ck, "stats") or {}
                self.post_message(TaskDone(kind, idx, scan))

            elif kind == "deep":
                context, _ = ops.parse_session(s["file"], deep=True)
                git = ops.get_git_context(s["project_dir"])
                quick = self.summaries[idx] or {"title": "Unknown", "state": "", "files": []}
                deep = ops.summarize_deep(context, s["project_dir"], quick, git)
                ops.cache.set(s["session_id"], ck, "deep_summary", deep)
                self.post_message(TaskDone(kind, idx, deep))

            elif kind == "patterns":
                context, _ = ops.parse_session(s["file"], deep=True)
                quick = self.summaries[idx] or {"title": "Unknown", "state": "", "files": []}
                patterns = ops.analyze_patterns(context, s["project_dir"], quick)
                ops.cache.set(s["session_id"], ck, "patterns", patterns)
                self.post_message(TaskDone(kind, idx, patterns))

            elif kind == "index":
                _, search_text = ops.parse_session(s["file"])
                full = (search_text + f" {s['project_dir']} {s['session_id']}").lower()
                ops.cache.set(s["session_id"], ck, "search_text", full)
                self.post_message(TaskDone(kind, idx, full))

        except Exception as e:
            self.post_message(TaskDone(kind, idx, error=str(e)))

    @work(thread=True)
    def _index_batch_bg(self, indices: list[int]) -> None:
        """Batch-index multiple sessions' search text in one thread."""
        for i in indices:
            s = self.sessions[i]
            try:
                _, search_text = self._ops.parse_session(s["file"])
                full = (search_text + f" {s['project_dir']} {s['session_id']}").lower()
                ck = self._ops.cache.cache_key(s["file"])
                self._ops.cache.set(s["session_id"], ck, "search_text", full)
                self.post_message(TaskDone("index", i, full))
            except Exception as e:
                self.post_message(TaskDone("index", i, error=str(e)))

    def on_task_done(self, message: TaskDone) -> None:
        """Single dispatcher for all background task completions."""
        self._pending[message.kind].discard(message.idx)

        if message.kind == "summarize":
            if message.error:
                self.summaries[message.idx] = {
                    "title": "Summary failed", "goal": "", "what_was_done": "",
                    "state": f"Error: {message.error}", "files": [],
                }
            else:
                self.summaries[message.idx] = message.result["summary"]
                if message.idx < len(self._search_index):
                    self._search_index[message.idx] = message.result["search_text"]
            self._refresh_list()

        elif message.kind == "deep":
            fi = self._current_filtered_index()
            if fi is not None and self.filtered_items[fi][0] == message.idx:
                if message.error:
                    self._show_preview_error(f"Deep dive failed: {message.error}")
                elif self._preview_mode == PreviewMode.SUMMARY:
                    self._update_preview(fi)
            self._refresh_list()

        elif message.kind == "patterns":
            fi = self._current_filtered_index()
            if (self._preview_mode == PreviewMode.PATTERNS
                    and fi is not None
                    and self.filtered_items[fi][0] == message.idx):
                if message.error:
                    self._show_preview_error(f"Patterns analysis failed: {message.error}")
                else:
                    self._display_patterns(message.idx, message.result)

        elif message.kind == "index":
            if not message.error and message.idx < len(self._search_index):
                self._search_index[message.idx] = message.result

        elif message.kind == "scan":
            self._refresh_list()

    # ── List management ────────────────────────────────────

    def _refresh_list(self) -> None:
        search = self.query_one("#search", SearchInput)
        self._populate_list(search.value)

    def _populate_list(self, filter_text: str = "") -> None:
        query = filter_text.lower()
        cache = self._ops.cache
        self.filtered_items = []
        hidden_count = 0
        for i, (s, sm) in enumerate(zip(self.sessions, self.summaries)):
            ck = cache.cache_key(s["file"])
            deep = cache.get(s["session_id"], ck, "deep_summary")
            best = deep or sm

            # Filter automated sessions unless toggled on
            if not self._show_bots:
                stats = cache.get(s["session_id"], ck, "stats")
                if stats and stats.get("classification") == "automated":
                    hidden_count += 1
                    continue

            if not query or (i < len(self._search_index) and query in self._search_index[i]):
                self.filtered_items.append((i, s, best))

        lv = self.query_one("#session-list", ListView)
        lv.clear()
        self._lv_map = {}
        current_group = None
        lv_idx = 0

        for fi, (idx, session, summary) in enumerate(self.filtered_items):
            group = _get_date_group(session["mtime"])
            if group != current_group:
                current_group = group
                lv.append(DateHeader(group))
                lv_idx += 1
            ck = cache.cache_key(session["file"])
            has_deep = cache.get(session["session_id"], ck, "deep_summary") is not None
            self._lv_map[lv_idx] = fi
            lv.append(SessionItem(idx, session, summary, has_deep=has_deep))
            lv_idx += 1

        # Show hidden automated session count
        if hidden_count and not self._show_bots:
            lv.append(DateHeader(f"{hidden_count} automated sessions hidden (b to show)"))
            lv_idx += 1

        if self.filtered_items:
            target_lv = None
            if not query:
                for lv_i, fi in self._lv_map.items():
                    if self.filtered_items[fi][0] == self._saved_session_idx:
                        target_lv = lv_i
                        break
            if target_lv is None:
                target_lv = min(self._lv_map.keys()) if self._lv_map else 0
            lv.index = target_lv
            self._last_lv_index = target_lv
        else:
            self.query_one("#preview", Static).update("[dim]No matching sessions[/]")

    def _current_filtered_index(self) -> int | None:
        lv = self.query_one("#session-list", ListView)
        if lv.index is not None and lv.index in self._lv_map:
            return self._lv_map[lv.index]
        return None

    # ── Preview rendering ──────────────────────────────────

    def _update_preview(self, filtered_idx: int) -> None:
        if filtered_idx < 0 or filtered_idx >= len(self.filtered_items):
            self.query_one("#preview", Static).update("")
            return

        orig_idx, session, summary = self.filtered_items[filtered_idx]

        if summary is None:
            short_path = shorten_path(session["project_dir"])
            age = relative_time(session["mtime"])
            self.query_one("#preview", Static).update(
                f"[bold yellow]Summarizing...[/]\n\n"
                f"[bold]Directory:[/]   [cyan]{short_path}[/]\n"
                f"[bold]Last active:[/] {age}\n"
            )
            return

        cache = self._ops.cache
        ck = cache.cache_key(session["file"])
        deep = cache.get(session["session_id"], ck, "deep_summary")
        display = deep or summary

        short_path = shorten_path(session["project_dir"])
        age = relative_time(session["mtime"])
        size_mb = session["size"] / (1024 * 1024)

        title = display.get("title", summary.get("title", "Unknown"))
        goal = display.get("goal", summary.get("goal", ""))
        what_was_done = display.get("what_was_done", summary.get("what_was_done", ""))
        state = display.get("state", summary.get("state", "No context"))
        files = display.get("files", summary.get("files", []))

        diving = orig_idx in self._pending["deep"]
        analyzing = orig_idx in self._pending["patterns"]
        perms_status = "[green]ON[/]" if self._skip_permissions else "[red]OFF[/]"

        status_parts = []
        if diving:
            status_parts.append("[bold yellow]⟳ Deep analyzing...[/]")
        if analyzing:
            status_parts.append("[bold yellow]⟳ Analyzing patterns...[/]")
        status = "  " + " ".join(status_parts) if status_parts else ""

        # Session stats from cache
        stats = cache.get(session["session_id"], ck, "stats")

        text = f"""[bold underline]{title}[/]{status}

[bold]Directory:[/]   [cyan]{short_path}[/]
[bold]Last active:[/] {age}
[bold]Size:[/]        {size_mb:.1f} MB
[bold]Skip perms:[/]  {perms_status}  [dim](d to toggle)[/]
"""
        if stats:
            dur = stats.get("duration_fmt", "?")
            cls = stats.get("classification", "?")
            cls_color = "green" if cls == "interactive" else "dim"
            text += f"\n[bold]Session stats:[/]\n"
            text += f"  Duration:     {dur}\n"
            text += f"  User msgs:    {stats.get('user_messages', '?')}\n"
            text += f"  Asst msgs:    {stats.get('assistant_messages', '?')}\n"
            text += f"  Tool uses:    {stats.get('tool_uses', '?')}\n"
            text += f"  Tool results: {stats.get('tool_results', '?')}\n"
            if stats.get("system_entries"):
                text += f"  System:       {stats['system_entries']}\n"
            if stats.get("progress_entries"):
                text += f"  Progress:     {stats['progress_entries']}\n"
            text += f"  Type:         [{cls_color}]{cls}[/]\n"
        if goal:
            text += f"\n[bold]Goal:[/]\n{goal}\n"
        if what_was_done:
            text += f"\n[bold]What was done:[/]\n{what_was_done}\n"
        text += f"\n[bold]Where you left off:[/]\n{state}\n"
        if files:
            text += "\n[bold]Key files:[/]\n"
            for f in files[:5]:
                text += f"  [dim]•[/] {f}\n"

        if deep:
            objective = deep.get("objective", "")
            if objective:
                text += f"\n[bold]Objective:[/]\n{objective}\n"
            progress = deep.get("progress", "")
            if progress:
                text += f"\n[bold]Progress:[/]\n{progress}\n"
            next_steps = deep.get("next_steps", "")
            if next_steps:
                text += f"\n[bold]Next steps:[/]\n{next_steps}\n"
            decisions = deep.get("decisions_made", [])
            if decisions:
                text += "\n[bold]Decisions:[/]\n"
                for d in decisions:
                    text += f"  [dim]•[/] {d}\n"

        self.query_one("#preview", Static).update(text)
        self.query_one("#preview-scroll", VerticalScroll).scroll_home(animate=False)

    def _display_patterns(self, orig_idx: int, patterns: dict) -> None:
        """Render patterns analysis in the preview pane."""
        session = self.sessions[orig_idx]
        short_path = shorten_path(session["project_dir"])

        text = f"[bold yellow]━━━ Patterns: {short_path} ━━━[/]\n\n"

        pp = patterns.get("prompt_patterns", {})
        effective = pp.get("effective", [])
        ineffective = pp.get("ineffective", [])
        tips = pp.get("tips", [])

        if effective:
            text += "[bold green]Effective prompts:[/]\n"
            for e in effective:
                text += f'  [green]✓[/] "{e.get("example", "")}"\n'
                text += f'    {e.get("why", "")}\n'
            text += "\n"

        if ineffective:
            text += "[bold red]Ineffective prompts:[/]\n"
            for e in ineffective:
                text += f'  [red]✗[/] "{e.get("example", "")}"\n'
                text += f'    {e.get("issue", "")}\n'
            text += "\n"

        if tips:
            text += "[bold cyan]Tips:[/]\n"
            for t in tips:
                text += f"  • {t}\n"
            text += "\n"

        wp = patterns.get("workflow_patterns", {})
        sequences = wp.get("common_sequences", [])
        style = wp.get("iteration_style", "")
        if style:
            text += f"[bold]Iteration style:[/] {style}\n"
        if sequences:
            text += "[bold]Common tool sequences:[/]\n"
            for seq in sequences:
                tools = " → ".join(seq.get("tools", []))
                eff = seq.get("efficiency", "")
                ctx = seq.get("context", "")
                text += f"  \\[{eff}] {tools}\n"
                if ctx:
                    text += f"         {ctx}\n"
            text += "\n"

        anti = patterns.get("anti_patterns", [])
        if anti:
            text += "[bold red]Anti-patterns:[/]\n"
            for a in anti:
                text += f"  ⚠ {a.get('pattern', '')}\n"
                text += f"    Cost: {a.get('cost', '')}\n"
                text += f"    Fix:  {a.get('fix', '')}\n"
            text += "\n"

        lesson = patterns.get("key_lesson", "")
        if lesson:
            text += f"[bold magenta]Key lesson:[/] {lesson}\n"

        text += "\n[dim]Press p to return to summary[/]"

        self.query_one("#preview", Static).update(text)
        self.query_one("#preview-scroll", VerticalScroll).scroll_home(animate=False)

    def _show_preview_error(self, msg: str) -> None:
        """Surface errors visibly in the preview pane."""
        self.query_one("#preview", Static).update(
            f"[bold red]{msg}[/]\n\n[dim]Will retry on next run[/]"
        )
        self.query_one("#preview-scroll", VerticalScroll).scroll_home(animate=False)

    # ── Search events ──────────────────────────────────────

    @on(Input.Changed, "#search")
    def on_search_changed(self, event: Input.Changed) -> None:
        self._populate_list(event.value)

    @on(Input.Submitted, "#search")
    def on_search_submit(self, event: Input.Submitted) -> None:
        self.query_one("#session-list", ListView).focus()

    @on(SearchInput.Escaped)
    def on_search_escaped(self, event: SearchInput.Escaped) -> None:
        self._populate_list("")
        self.query_one("#session-list", ListView).focus()

    # ── List events ────────────────────────────────────────

    @on(ListView.Highlighted)
    def on_highlight(self, event: ListView.Highlighted) -> None:
        lv = self.query_one("#session-list", ListView)
        if lv.index is None:
            return
        # Skip DateHeaders by jumping in the direction of travel
        if lv.index not in self._lv_map:
            if lv.index > self._last_lv_index:
                new = lv.index + 1
            else:
                new = lv.index - 1
            if 0 <= new < len(lv.children) and new in self._lv_map:
                lv.index = new
            return
        self._last_lv_index = lv.index
        fi = self._lv_map[lv.index]
        new_session_idx = self.filtered_items[fi][0]
        # Reset preview mode when switching to a different session
        if new_session_idx != self._saved_session_idx:
            self._preview_mode = PreviewMode.SUMMARY
        self._saved_session_idx = new_session_idx
        # Lazy classify: quick-scan on scroll if no stats cached yet
        s = self.sessions[new_session_idx]
        ck = self._ops.cache.cache_key(s["file"])
        if self._ops.cache.get(s["session_id"], ck, "stats") is None:
            self._start_task("scan", new_session_idx)
        # Lazy summarize: kick off when user scrolls to an unsummarized session
        if self.summaries[new_session_idx] is None:
            self._start_task("summarize", new_session_idx)
        if self._preview_mode == PreviewMode.SUMMARY:
            self._update_preview(fi)

    @on(ListView.Selected)
    def on_selected(self, event: ListView.Selected) -> None:
        fi = self._current_filtered_index()
        if fi is not None:
            idx, session, _ = self.filtered_items[fi]
            cmd = f"cd {session['project_dir']} && claude --resume {session['session_id']}"
            if self._skip_permissions:
                cmd += " --dangerously-skip-permissions"
            self.result_data = ("select", idx, cmd)
            self.exit()

    # ── Key handling ───────────────────────────────────────

    def _build_resume_cmd(self) -> tuple[int, str] | None:
        fi = self._current_filtered_index()
        if fi is None:
            return None
        idx, session, _ = self.filtered_items[fi]
        cmd = f"cd {session['project_dir']} && claude --resume {session['session_id']}"
        if self._skip_permissions:
            cmd += " --dangerously-skip-permissions"
        return idx, cmd

    def on_key(self, event: events.Key) -> None:
        search = self.query_one("#search", SearchInput)
        lv = self.query_one("#session-list", ListView)
        scroll_pane = self.query_one("#preview-scroll", VerticalScroll)
        in_search = search == self.focused

        # Arrow keys in search → move to list
        if event.key in ("down", "up") and in_search:
            lv.focus()
            if event.key == "down" and lv.index is not None and lv.index < len(lv.children) - 1:
                lv.index += 1
            elif event.key == "up" and lv.index is not None and lv.index > 0:
                lv.index -= 1
            event.prevent_default()
            event.stop()
            return

        if in_search:
            return

        # ── Preview pane mode ──
        if self._in_preview:
            if event.key in ("left", "escape"):
                self._in_preview = False
                scroll_pane.remove_class("focused")
                lv.focus()
                event.prevent_default()
                event.stop()
            elif event.key == "down":
                scroll_pane.scroll_relative(y=10, animate=False)
                event.prevent_default()
                event.stop()
            elif event.key == "up":
                scroll_pane.scroll_relative(y=-10, animate=False)
                event.prevent_default()
                event.stop()
            elif event.key == "enter":
                result = self._build_resume_cmd()
                if result:
                    idx, cmd = result
                    self.result_data = ("select", idx, cmd)
                    self.exit()
                event.prevent_default()
                event.stop()
            elif event.character == "q":
                self.exit()
                event.prevent_default()
                event.stop()
            return

        # ── List mode ──
        if event.key == "right":
            self._in_preview = True
            scroll_pane.add_class("focused")
            event.prevent_default()
            event.stop()

        elif event.key == "slash":
            fi = self._current_filtered_index()
            if fi is not None:
                self._saved_session_idx = self.filtered_items[fi][0]
            search.focus()
            event.prevent_default()
            event.stop()

        elif event.key == "escape":
            self.exit()
            event.prevent_default()
            event.stop()

        elif event.character == "d":
            self._skip_permissions = not self._skip_permissions
            fi = self._current_filtered_index()
            if fi is not None:
                self._update_preview(fi)
            event.prevent_default()
            event.stop()

        elif event.character == "D":
            fi = self._current_filtered_index()
            if fi is not None:
                idx, session, _ = self.filtered_items[fi]
                ck = self._ops.cache.cache_key(session["file"])
                if self._ops.cache.get(session["session_id"], ck, "deep_summary"):
                    self._update_preview(fi)
                else:
                    self._start_task("deep", idx)
                    self._update_preview(fi)  # Shows spinner via _pending check
            event.prevent_default()
            event.stop()

        elif event.character == "p":
            fi = self._current_filtered_index()
            if fi is not None:
                idx, session, _ = self.filtered_items[fi]
                if self._preview_mode == PreviewMode.PATTERNS:
                    # Toggle off
                    self._preview_mode = PreviewMode.SUMMARY
                    self._update_preview(fi)
                else:
                    ck = self._ops.cache.cache_key(session["file"])
                    cached = self._ops.cache.get(session["session_id"], ck, "patterns")
                    self._preview_mode = PreviewMode.PATTERNS
                    if cached:
                        self._display_patterns(idx, cached)
                    else:
                        self.query_one("#preview", Static).update(
                            "[bold yellow]⟳ Analyzing patterns...[/]"
                        )
                        self._start_task("patterns", idx)
            event.prevent_default()
            event.stop()

        elif event.character == "b":
            self._show_bots = not self._show_bots
            self._refresh_list()
            event.prevent_default()
            event.stop()

        elif event.character == "q":
            self.exit()
            event.prevent_default()
            event.stop()
