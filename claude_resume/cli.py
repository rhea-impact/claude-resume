"""CLI entry point for claude-resume."""

import os
import subprocess
import sys
import termios
import tty

from .sessions import (
    SessionCache,
    SessionOps,
    find_all_sessions,
    find_recent_sessions,
    get_git_context,
    get_label_deep,
    interruption_score,
    parse_session,
    relative_time,
    shorten_path,
    MAX_SESSIONS_ALL,
)
from claude_session_commons.summarize import analyze_patterns, summarize_deep, summarize_quick
from .ui import SessionPickerApp

DEFAULT_HOURS = 4


def _read_key() -> str | None:
    """Read a single keypress without requiring Enter. Returns the character,
    'esc' for escape, or None for Ctrl-C/EOF."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == '\x1b':  # escape sequence
            # Peek for arrow key sequences (don't block)
            import select
            if select.select([sys.stdin], [], [], 0.05)[0]:
                sys.stdin.read(1)  # consume [
                sys.stdin.read(1)  # consume A/B/C/D
            return 'esc'
        if ch == '\x03':  # Ctrl-C
            return None
        if ch == '\x04':  # Ctrl-D
            return None
        return ch
    except (KeyboardInterrupt, EOFError):
        return None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

USAGE = """\
claude-resume — Post-crash Claude Code session picker.

Finds your most recently active Claude Code sessions, uses AI to summarize
what each one was doing, and copies the resume command to your clipboard.

Usage:
    claude-resume              # Show sessions from last 4 hours
    claude-resume 24           # Show sessions from last 24 hours
    claude-resume --all        # Show all sessions (up to 200)
    claude-resume --cache-all  # Background-index every session you've ever had
    claude-resume --search <term>  # Search all sessions for a keyword
"""


def _copy_to_clipboard(text: str):
    subprocess.run(["pbcopy"], input=text.encode(), check=True)


def _open_iterm_tabs(commands: list[str]):
    """Open each command in a new iTerm tab."""
    for cmd in commands:
        # Escape double quotes and backslashes for AppleScript
        escaped = cmd.replace("\\", "\\\\").replace('"', '\\"')
        script = f'''
        tell application "iTerm"
            activate
            tell current window
                create tab with default profile
                tell current session
                    write text "{escaped}"
                end tell
            end tell
        end tell
        '''
        subprocess.run(["osascript", "-e", script], capture_output=True)


def _daemon_alive() -> bool:
    """Check if the session daemon is running."""
    pid_file = os.path.join(os.path.expanduser("~"), ".claude", "session-daemon.pid")
    try:
        if not os.path.exists(pid_file):
            return False
        with open(pid_file) as f:
            pid = int(f.read().strip())
        os.kill(pid, 0)
        return True
    except (ValueError, ProcessLookupError, PermissionError, OSError):
        return False


def _clean_title(title: str) -> str:
    """Strip XML/HTML tags and command artifacts from cached titles."""
    import re
    if not title:
        return ""
    # Remove XML/HTML tags
    title = re.sub(r'<[^>]+>', '', title)
    # Collapse whitespace
    title = re.sub(r'\s+', ' ', title).strip()
    return title


def _search_sessions(term: str):
    """Search sessions with cached titles, sorted most-recent-first."""
    from concurrent.futures import ThreadPoolExecutor
    from datetime import datetime

    cache = SessionCache()
    term_bytes = term.lower().encode("utf-8", errors="replace")
    all_sessions = find_all_sessions()

    print(f"\n  Searching {len(all_sessions)} sessions for \033[1m{term}\033[0m...", end="", flush=True)

    def _check(s):
        try:
            raw = s["file"].read_bytes()
        except OSError:
            return None
        raw_lower = raw.lower()
        if term_bytes not in raw_lower:
            return None
        return (s, raw_lower.count(term_bytes))

    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(_check, all_sessions))

    matches = [r for r in results if r is not None]
    # Sort by recency — most recent first (visible at bottom in scrolling terminal)
    matches.sort(key=lambda r: r[0]["mtime"], reverse=True)

    print(f" done.\n")

    for i, (s, count) in enumerate(matches, 1):
        dt = datetime.fromtimestamp(s["mtime"])
        age = relative_time(s["mtime"])
        project = shorten_path(s["project_dir"])
        sid = s["session_id"]
        count_str = f"{count} hit{'s' if count != 1 else ''}"

        # Get cached title
        ck = cache.cache_key(s["file"])
        cached = cache.get(sid, ck, "summary")
        title = ""
        if cached and isinstance(cached, dict):
            title = _clean_title(cached.get("title", ""))
        if not title:
            data = cache._read(sid)
            summary = data.get("summary")
            if isinstance(summary, dict):
                title = _clean_title(summary.get("title", ""))

        title_display = f"  \033[2m{title[:65]}\033[0m" if title else ""

        print(f"  \033[1;33m{i:2d}\033[0m  {project}  \033[2m{age}  ({count_str})\033[0m{title_display}")
        print(f"      \033[36mclaude --resume {sid}\033[0m")
        print()

    if not matches:
        print(f"  No sessions found containing \"{term}\".\n")
    else:
        print(f"  \033[1;32m{len(matches)} session{'s' if len(matches) != 1 else ''} found.\033[0m\n")


def _cache_all_sessions():
    """Background-index every session that doesn't have a cached summary."""
    import json
    from pathlib import Path

    cache = SessionCache()
    all_sessions = find_all_sessions()

    # Count uncached
    uncached = []
    for s in all_sessions:
        ck = cache.cache_key(s["file"])
        if not cache.get(s["session_id"], ck, "summary"):
            uncached.append(s)

    cached = len(all_sessions) - len(uncached)

    if not uncached:
        print(f"\n  All {len(all_sessions)} sessions already cached.\n")
        return

    # If daemon is alive, write task files and let it handle everything
    if _daemon_alive():
        task_dir = Path.home() / ".claude" / "daemon-tasks"
        task_dir.mkdir(parents=True, exist_ok=True)
        for s in uncached:
            import time
            priority = int(time.time() * 1000)
            filename = f"{priority}-summarize-{s['session_id'][:8]}.json"
            task = {
                "kind": "summarize",
                "session_id": s["session_id"],
                "file": str(s["file"]),
                "project_dir": s["project_dir"],
                "quick_summary": None,
            }
            (task_dir / filename).write_text(json.dumps(task))
            time.sleep(0.001)  # ensure unique timestamps
        print(f"\n  Daemon is running — queued {len(uncached)} sessions for processing.")
        print(f"  ({cached} already cached)")
        print(f"  Monitor: tail -f ~/.claude/daemon.log\n")
        return

    # Fallback: local processing (daemon not running)
    total = len(all_sessions)
    generated = 0
    failed = 0

    print(f"\n  Daemon not running — indexing {len(uncached)} sessions locally...\n", flush=True)

    for i, s in enumerate(uncached, 1):
        ck = cache.cache_key(s["file"])
        short = shorten_path(s["project_dir"])
        age = relative_time(s["mtime"])
        print(f"  [{i}/{len(uncached)}] {short} ({age})...", end="", flush=True)

        try:
            context, search_text = parse_session(s["file"])
            git = get_git_context(s["project_dir"])
            summary = summarize_quick(context, s["project_dir"], git)
            cache.set(s["session_id"], ck, "summary", summary)
            full = (search_text + f" {s['project_dir']} {s['session_id']}").lower()
            cache.set(s["session_id"], ck, "search_text", full)
            get_label_deep(s["file"], cache)
            title = summary.get("title", "?")
            print(f" \033[32m{title}\033[0m", flush=True)
            generated += 1
        except Exception as e:
            print(f" \033[31mfailed: {e}\033[0m", flush=True)
            failed += 1

    print(f"\n  Done. {cached} already cached, {generated} newly indexed, {failed} failed.\n")


def _get_cached_title(cache, s):
    """Get cleaned cached title for a session."""
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


def _preview_session(s, cache):
    """Show a quick preview of a session before resuming."""
    import json
    from datetime import datetime

    sid = s["session_id"]
    project = shorten_path(s["project_dir"])
    age = relative_time(s["mtime"])
    title = _get_cached_title(cache, s)

    print(f"\n  \033[1;36m{'─' * 60}\033[0m")
    print(f"  \033[1m{title or 'Untitled session'}\033[0m")
    print(f"  \033[2m{project}  •  {age}  •  {sid[:12]}\033[0m")

    # Get cached summary details
    ck = cache.cache_key(s["file"])
    cached = cache.get(sid, ck, "summary")
    if cached and isinstance(cached, dict):
        goal = cached.get("goal", "")
        what = cached.get("what_was_done", "")
        status = cached.get("status", "")
        if goal:
            print(f"\n  \033[1mGoal:\033[0m {_clean_title(goal)[:80]}")
        if what:
            print(f"  \033[1mDone:\033[0m {_clean_title(what)[:80]}")
        if status:
            print(f"  \033[1mStatus:\033[0m {_clean_title(status)[:80]}")

    # Show last few user messages from the JSONL
    try:
        lines = []
        with open(s["file"], "rb") as f:
            # Read last 64KB
            f.seek(0, 2)
            size = f.tell()
            read_from = max(0, size - 65536)
            f.seek(read_from)
            chunk = f.read().decode("utf-8", errors="replace")
            lines = chunk.strip().split("\n")

        user_msgs = []
        for line in reversed(lines):
            try:
                entry = json.loads(line)
                if entry.get("type") == "user" and isinstance(entry.get("message"), dict):
                    content = entry["message"].get("content", "")
                    if isinstance(content, str) and len(content) > 5:
                        user_msgs.append(content[:100])
                    elif isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "text":
                                text = part.get("text", "")
                                if len(text) > 5:
                                    user_msgs.append(text[:100])
                                    break
            except (json.JSONDecodeError, KeyError):
                continue
            if len(user_msgs) >= 3:
                break

        if user_msgs:
            print(f"\n  \033[1mLast messages:\033[0m")
            for msg in reversed(user_msgs):
                cleaned = _clean_title(msg)
                print(f"    \033[33m>\033[0m {cleaned}")
    except OSError:
        pass

    print(f"  \033[1;36m{'─' * 60}\033[0m")
    print(f"\n  \033[2m[enter] resume  [esc] back\033[0m", flush=True)

    while True:
        key = _read_key()
        if key is None or key == 'esc':
            return False  # go back
        if key in ('\r', '\n'):
            return True  # resume


def _show_group_menu(sorted_groups, cache, hours, total_sessions):
    """Display the group selection menu. Returns selected index or None."""
    print(f"\n  \033[1m{total_sessions} sessions \033[1;36m→\033[0m\033[1m {len(sorted_groups)} groups\033[0m  \033[2m(last {hours}h)\033[0m\n")

    for i, (org, group_sessions) in enumerate(sorted_groups, 1):
        most_recent = max(s["mtime"] for s in group_sessions)
        age = relative_time(most_recent)
        count = len(group_sessions)

        newest = max(group_sessions, key=lambda s: s["mtime"])
        title = _get_cached_title(cache, newest)
        title_str = f"  \033[2m{title[:50]}\033[0m" if title else ""

        print(f"  \033[1;33m{i:2d}\033[0m  \033[1;36m{org}\033[0m  \033[2m({count} session{'s' if count != 1 else ''}, {age})\033[0m{title_str}")

    print(f"\n  \033[2m[1-{len(sorted_groups)}] select  [esc/q] quit\033[0m", flush=True)


def _show_session_menu(org, group_sessions, cache):
    """Display the session selection menu."""
    print(f"\n  \033[1;36m{org}\033[0m  \033[2m({len(group_sessions)} sessions)\033[0m\n")

    for i, s in enumerate(group_sessions, 1):
        sid = s["session_id"]
        project = shorten_path(s["project_dir"])
        age = relative_time(s["mtime"])
        title = _get_cached_title(cache, s)
        title_display = title[:60] if title else project

        print(f"  \033[1;33m{i:2d}\033[0m  {title_display}  \033[2m{age}\033[0m")
        print(f"      \033[2m{project}\033[0m  \033[36m{sid[:8]}\033[0m")
        print()

    print(f"  \033[2m[1-{len(group_sessions)}] preview  [esc] back\033[0m", flush=True)


def _cluster_sessions(hours: int = 48):
    """Two-level drill-down with instant keypress navigation.

    Level 1: Pick a repo group — single keypress, no Enter needed
    Level 2: Pick a session to preview — single keypress
    Preview: See summary + last messages, Enter to resume, Esc to go back
    """
    from collections import defaultdict

    sessions = find_recent_sessions(hours, max_sessions=100)
    if not sessions:
        print(f"  No sessions in last {hours}h.")
        return

    cache = SessionCache()

    # Group by repo org
    groups = defaultdict(list)
    for s in sessions:
        project = shorten_path(s["project_dir"])
        parts = project.split("/")
        if len(parts) >= 2 and parts[1].startswith("repos-"):
            org = parts[1]
        elif parts[0] == "~" and len(parts) == 1:
            org = "~ (home)"
        else:
            org = parts[1] if len(parts) > 1 else parts[0]
        groups[org].append(s)

    sorted_groups = sorted(
        groups.items(),
        key=lambda kv: max(s["mtime"] for s in kv[1]),
        reverse=True,
    )

    # --- Level 1 loop: group selection ---
    while True:
        _show_group_menu(sorted_groups, cache, hours, len(sessions))

        key = _read_key()
        if key is None or key in ('esc', 'q'):
            print()
            return

        if not key.isdigit() or key == '0':
            continue

        group_idx = int(key) - 1
        if group_idx >= len(sorted_groups):
            continue

        org, group_sessions = sorted_groups[group_idx]
        group_sessions.sort(key=lambda s: s["mtime"], reverse=True)

        # --- Level 2 loop: session selection ---
        while True:
            _show_session_menu(org, group_sessions, cache)

            key = _read_key()
            if key is None or key == 'esc':
                break  # back to groups

            if not key.isdigit() or key == '0':
                continue

            sess_idx = int(key) - 1
            if sess_idx >= len(group_sessions):
                continue

            s = group_sessions[sess_idx]

            # --- Preview loop ---
            if _preview_session(s, cache):
                # User confirmed resume
                print(f"\n  \033[1;32m⟶ Resuming {s['session_id'][:8]}...\033[0m\n")
                os.chdir(s["project_dir"])
                os.execlp(
                    "claude", "claude",
                    "--resume", s["session_id"],
                    "--dangerously-skip-permissions",
                )
            # else: user pressed Esc, back to session list


def main():
    # cr v2 — new TUI
    if len(sys.argv) > 1 and sys.argv[1] == "v2":
        from .ui_v2 import run_v2
        hours = 48
        search_term = None
        args = sys.argv[2:]
        if args and args[0] in ("s", "search"):
            if len(args) > 1:
                search_term = " ".join(args[1:])
                hours = 8760  # search all
            else:
                print("Usage: cr v2 s <search term>")
                sys.exit(1)
        elif args:
            try:
                hours = int(args[0])
            except ValueError:
                pass
        run_v2(hours=hours, search_term=search_term)
        sys.exit(0)

    if len(sys.argv) > 1 and sys.argv[1] in ("k", "--cluster"):
        hours = 48
        if len(sys.argv) > 2:
            try:
                hours = int(sys.argv[2])
            except ValueError:
                pass
        _cluster_sessions(hours)
        sys.exit(0)

    if len(sys.argv) > 1 and sys.argv[1] == "--cache-all":
        _cache_all_sessions()
        sys.exit(0)

    if len(sys.argv) > 1 and sys.argv[1] == "--search":
        if len(sys.argv) < 3:
            print("Usage: claude-resume --search <term>")
            sys.exit(1)
        _search_sessions(" ".join(sys.argv[2:]))
        sys.exit(0)

    hours = DEFAULT_HOURS
    show_all = False
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "--all":
            hours = 8760
            show_all = True
        elif arg in ("--help", "-h"):
            print(USAGE)
            sys.exit(0)
        else:
            try:
                hours = float(arg)
            except ValueError:
                print("Usage: claude-resume [hours|--all|--cache-all|--search <term>]")
                sys.exit(1)

    max_sessions = MAX_SESSIONS_ALL if show_all else None
    sessions = find_recent_sessions(hours, max_sessions=max_sessions) if max_sessions else find_recent_sessions(hours)

    if not sessions:
        print(f"  No sessions found in the last {int(hours)} hours.")
        print("  Try: claude-resume --all")
        sys.exit(0)

    # Sort by date group first (preserves grouping), then by interruption score within each group
    from .sessions import get_date_group as _get_date_group
    group_order = {"Today": 0, "Yesterday": 1, "Last 7 Days": 2, "Last 30 Days": 3, "Older": 4}
    sessions.sort(key=lambda s: (group_order.get(_get_date_group(s["mtime"]), 9), -interruption_score(s)))

    cache = SessionCache()
    ops = SessionOps(
        cache=cache,
        parse_session=parse_session,
        get_git_context=get_git_context,
        summarize_quick=summarize_quick,
        summarize_deep=summarize_deep,
        analyze_patterns=analyze_patterns,
    )

    summaries = []
    for s in sessions:
        ck = cache.cache_key(s["file"])
        cached = cache.get(s["session_id"], ck, "summary")
        summaries.append(cached)

    termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)

    app = SessionPickerApp(sessions, summaries, ops)
    app.run()

    if not app.result_data:
        sys.exit(0)

    action, idx, cmd = app.result_data

    if action == "resume":
        # Exec directly into the session — replaces this process
        print(f"\n  \033[1;32m⟶ Resuming session...\033[0m\n")
        os.execlp("bash", "bash", "-c", cmd)

    elif action == "multi_resume":
        # cmd is a list of commands — open each in an iTerm tab
        cmds = cmd  # it's a list
        _open_iterm_tabs(cmds)
        print(f"\n  \033[1;32m✓ Opened {len(cmds)} sessions in iTerm tabs\033[0m\n")

    elif action == "select":
        _copy_to_clipboard(cmd)
        print(f"\n  \033[1;32m✓ Copied to clipboard:\033[0m")
        print(f"    {cmd}\n")
