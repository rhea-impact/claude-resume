"""ADR-001 Self-Report for claude-resume — thin wrapper around mcp-self-report.

Provides claude-resume-specific auto_context and component list.
The shared package handles schema, dedup, priority, and tool registration.
"""

import os
from pathlib import Path

from mcp_self_report import register_self_report

from .sessions import find_all_sessions


def _auto_context() -> dict:
    """Capture claude-resume server state at report time."""
    ctx: dict = {}
    try:
        ctx["total_sessions"] = len(find_all_sessions())
    except Exception:
        ctx["total_sessions"] = "error"

    # Cache stats
    cache_dir = Path.home() / ".claude" / "resume-summaries"
    if cache_dir.exists():
        try:
            ctx["cached_summaries"] = len(list(cache_dir.glob("*.json")))
        except Exception:
            pass

    # Daemon status
    pid_file = Path.home() / ".claude" / "session-daemon.pid"
    daemon_alive = False
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            os.kill(pid, 0)
            daemon_alive = True
        except (ValueError, ProcessLookupError, PermissionError, OSError):
            pass
    ctx["daemon_alive"] = daemon_alive

    return ctx


_COMPONENTS = [
    "search",        # search_sessions ranking, snippets, query parsing
    "read",          # read_session message extraction, head/tail sampling
    "summary",       # session_summary generation, caching, daemon
    "merge",         # merge_context import quality
    "boot",          # boot_up crash detection
    "resume",        # resume_in_terminal launch
    "data_science",  # analytics, insights, reports
]


def register_tools(mcp_instance):
    """Register claude-resume self-report tools via the shared package."""
    register_self_report(
        mcp_instance,
        server_name="claude-resume",
        server_version="0.1.0",
        auto_context_fn=_auto_context,
        components=_COMPONENTS,
    )
