"""Self-knowledge + meta-AI MCP tools for resume-resume.

Extracted from mcp_server.py to keep that file under the 2000-line
threshold per CLAUDE.md file-size policy. Registered via
register_self_tools(mcp) at import time.

Two tool groups:
1. Self-knowledge (telemetry introspection): self_insights, self_recent_calls,
   self_slow_calls, self_errors, self_search, self_bundles
2. Meta-AI (A1/A2 pyramid): self_a1_file, self_a2_file, self_load_thresholds,
   self_a1_prompt, self_process_proposals, self_process_decide, self_a1_output,
   self_a1_auto_applied, self_proposal_history, self_a2_scorecard
"""

from __future__ import annotations

import subprocess
import time
from datetime import datetime
from pathlib import Path

from . import telemetry_query as _tq
from . import meta_ai as _meta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wrap(items: list) -> dict:
    """Wrap a list in {items, count} for uniform MCP response shape."""
    return {"items": items, "count": len(items)}


# ---------------------------------------------------------------------------
# Caches (module-level, per-process)
# ---------------------------------------------------------------------------

_SELF_INSIGHTS_CACHE: dict = {}
_SELF_INSIGHTS_CACHE_TTL = 15.0


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_self_tools(mcp_instance):
    """Register all self_* tools on the MCP server."""

    # --- Telemetry introspection ---

    @mcp_instance.tool()
    def self_insights(days: int = 30) -> dict:
        """Opinionated report on resume-resume's own MCP usage.

        Returns total calls, per-tool summary (counts, error rate, p50/p95
        latency), plus callouts for dead tools, slow tools, error-prone tools,
        and abandoned queries. This is the weekly roadmap-steering tool.

        Cached for 15 seconds per `days` key so A1/A2 skill loops that re-hit
        insights multiple times per turn don't re-scan the JSONL corpus.
        """
        now = time.time()
        cached = _SELF_INSIGHTS_CACHE.get(days)
        if cached and (now - cached["ts"]) < _SELF_INSIGHTS_CACHE_TTL:
            return {**cached["data"], "cached": True, "cache_age_s": round(now - cached["ts"], 1)}

        data = _tq.insights_report(days=days)
        data["cached"] = False
        _SELF_INSIGHTS_CACHE[days] = {"data": data, "ts": now}
        return data

    @mcp_instance.tool()
    def self_recent_calls(limit: int = 50, tool: str = "") -> dict:
        """Tail of recent MCP tool calls. Most recent first."""
        events = _tq.load_events(days=7, tool=tool or None)
        events.sort(key=lambda e: e.get("ts") or "", reverse=True)
        return _wrap(events[:limit])

    @mcp_instance.tool()
    def self_slow_calls(threshold_ms: int = 1000, days: int = 7) -> dict:
        """Calls that took longer than `threshold_ms`. Slowest first."""
        events = _tq.load_events(days=days)
        slow = [e for e in events if (e.get("duration_ms") or 0) >= threshold_ms]
        slow.sort(key=lambda e: e.get("duration_ms") or 0, reverse=True)
        return _wrap(slow)

    @mcp_instance.tool()
    def self_errors(days: int = 7) -> dict:
        """Recent failing MCP calls with full context to debug."""
        events = _tq.load_events(days=days, status="error")
        events.sort(key=lambda e: e.get("ts") or "", reverse=True)
        return _wrap(events)

    @mcp_instance.tool()
    def self_search(query: str, days: int = 30, limit: int = 20) -> dict:
        """BM25 search over past MCP calls (tool name + args + result + error)."""
        events = _tq.load_events(days=days)
        return _wrap(_tq.bm25_search(events, query, limit=limit))

    @mcp_instance.tool()
    def self_bundles(days: int = 7, gap_seconds: int = 30) -> dict:
        """Group contiguous same-session calls into work bundles.

        A bundle is a burst of calls within `gap_seconds` on the same session.
        Useful for finding slow workflows even when individual calls are fast.
        """
        events = _tq.load_events(days=days)
        return _wrap(_tq.session_bundles(events, gap_seconds=gap_seconds))

    # --- Meta-AI: A1 + A2 + human inbox ---

    @mcp_instance.tool()
    def self_a1_file(
        type: str,
        title: str,
        evidence: str,
        confidence: float,
        action_class: str = "queued",
        target: str = "",
        new_value: float | int | None = None,
        suggested_action: str = "",
    ) -> dict:
        """File an A1 product recommendation. Called by the A1 skill.

        Validates confidence against threshold, dedupes against recent entries,
        auto-applies the 'auto' class when guardrails pass (tune + target in
        tunable keys + numeric new_value), downgrades unsafe auto-requests to
        queued. Returns the recorded record or a {skipped: <reason>} dict.

        type: remove | optimize | tune | investigate | ship | other
        action_class: "auto" | "queued"
        """
        return _meta.file_a1_recommendation(
            type=type, title=title, evidence=evidence, confidence=confidence,
            action_class=action_class, target=target, new_value=new_value,
            suggested_action=suggested_action,
        )

    @mcp_instance.tool()
    def self_a2_file(
        target: str,
        change_type: str,
        title: str,
        evidence: str,
        confidence: float,
        diff: dict | str | None = None,
        expected_effect: str = "",
    ) -> dict:
        """File an A2 process-management proposal. Called by the A2 skill.

        target: "a1_prompt" | "thresholds.json" | "cadence"
        change_type: "prompt_edit" | "threshold_change" | "criterion_add"
                     | "criterion_remove" | "authority_change" | "other"
        diff: for prompt_edit -> {"full_new_text": "..."}; for threshold_change
              -> {"key": "...", "from": X, "to": Y}; else descriptive string.
        """
        return _meta.file_a2_proposal(
            target=target, change_type=change_type, title=title,
            evidence=evidence, confidence=confidence, diff=diff,
            expected_effect=expected_effect,
        )

    @mcp_instance.tool()
    def self_load_thresholds() -> dict:
        """Current config/thresholds.json plus the list of keys A1 may auto-tune."""
        return {
            "thresholds": _meta.load_thresholds(),
            "tunable_keys": sorted(_meta.TUNABLE_KEYS),
        }

    @mcp_instance.tool()
    def self_a1_prompt() -> str:
        """Read A1's skill prompt (SKILL.md). A2 uses this to reason about prompt edits."""
        return _meta.read_a1_prompt()

    @mcp_instance.tool()
    def self_process_proposals(state: str = "pending", limit: int = 50) -> dict:
        """Human inbox. Returns A2 process-management proposals by state."""
        return _wrap(_meta.list_proposals(state=state, limit=limit))

    @mcp_instance.tool()
    def self_process_decide(proposal_id: str, verdict: str, reason: str = "") -> dict:
        """Approve, reject, or defer an A2 proposal.

        verdict: approved | rejected | deferred. On 'approved', the proposal's
        change is applied to A1's config/prompt (leaves the working tree dirty;
        commit manually).
        """
        return _meta.decide_proposal(proposal_id, verdict, reason=reason)

    @mcp_instance.tool()
    def self_a1_output(limit: int = 20, action_class: str = "") -> dict:
        """Recent A1 recommendations. Optional filter by action_class: 'auto' or 'queued'."""
        return _wrap(_meta.a1_recent_recommendations(limit=limit, action_class=action_class or None))

    @mcp_instance.tool()
    def self_a1_auto_applied(limit: int = 50) -> dict:
        """History of changes A1 has auto-applied to config/thresholds.json."""
        return _wrap(_meta.a1_auto_applied_history(limit=limit))

    @mcp_instance.tool()
    def self_proposal_history(limit: int = 100) -> dict:
        """Decided A2 proposals (approved/rejected/deferred) — your audit trail."""
        return _wrap(_meta.proposal_history(limit=limit))

    @mcp_instance.tool()
    def self_a2_scorecard(days: int = 90) -> dict:
        """Score A2's effectiveness: for each approved proposal, compare A1's
        output before vs after the change. Shows whether A2's methodology
        changes actually improved A1.

        Each row: proposal title, expected_effect, A1 stats before/after
        (count, auto_applied, queued, avg_confidence). The human judges
        the trend.
        """
        return _meta.a2_scorecard(days=days)

    # --- Cross-session project changelog ---

    @mcp_instance.tool()
    def what_changed(project: str, hours: int = 168, limit: int = 20) -> dict:
        """What happened on a project across sessions in a time window.

        Synthesizes across all sessions for a project — not reading one
        session, but producing a changelog: which sessions ran, what was
        each about, what commits landed.

        Use this for standup summaries, handoff context, or "what did I
        do on X this week."

        Parameters:
          project: Substring match on project path (case-insensitive).
          hours: Lookback window (default 168 = 1 week).
          limit: Max sessions to include (default 20).
        """
        from .mcp_server import _find_all_sessions_cached, _get_title, shorten_path

        all_sessions = _find_all_sessions_cached()
        cutoff = time.time() - hours * 3600
        project_lower = project.lower()

        matched = [
            s for s in all_sessions
            if project_lower in s.get("project_dir", "").lower()
            and s["mtime"] >= cutoff
        ]
        matched.sort(key=lambda s: s["mtime"], reverse=True)
        matched = matched[:limit]

        if not matched:
            return {
                "project": project, "hours": hours, "sessions": 0,
                "message": f"No sessions matching '{project}' in the last {hours} hours.",
            }

        entries = []
        project_path = matched[0].get("project_dir", "")

        for s in matched:
            sid = s["session_id"]
            title = _get_title(sid, s["file"])
            entries.append({
                "session_id": sid,
                "date": datetime.fromtimestamp(s["mtime"]).strftime("%Y-%m-%d %H:%M"),
                "title": title or "(no summary)",
                "size_kb": round(s.get("size", 0) / 1024, 1),
            })

        git_commits = []
        if project_path and Path(project_path).is_dir():
            try:
                since = datetime.fromtimestamp(cutoff).strftime("%Y-%m-%d")
                log = subprocess.run(
                    ["git", "log", f"--since={since}", "--oneline", "--format=%h %ar %s", "-20"],
                    cwd=project_path, capture_output=True, text=True, timeout=5,
                )
                git_commits = [l.strip() for l in log.stdout.splitlines() if l.strip()]
            except (subprocess.TimeoutExpired, OSError):
                pass

        dirty_files = []
        if project_path and Path(project_path).is_dir():
            try:
                status = subprocess.run(
                    ["git", "status", "--porcelain"],
                    cwd=project_path, capture_output=True, text=True, timeout=5,
                )
                dirty_files = [l.strip() for l in status.stdout.splitlines() if l.strip()][:15]
            except (subprocess.TimeoutExpired, OSError):
                pass

        # L2 topic summaries if available — project narrative context
        topics = None
        try:
            from claude_session_commons.insights import get_db
            conn = get_db()
            rows = conn.execute(
                """SELECT title, summary_text FROM summary_levels
                   WHERE level = 2 AND entity_id LIKE ? || '::%'
                   ORDER BY updated_at DESC""",
                (project_path,),
            ).fetchall()
            if rows:
                import json as _json
                topics = []
                for r in rows:
                    try:
                        s = _json.loads(r[1]) if isinstance(r[1], str) else (r[1] or {})
                    except Exception:
                        s = {}
                    topics.append({
                        "topic": s.get("topic_name", r[0]),
                        "status": s.get("status", ""),
                    })
        except Exception:
            pass

        result = {
            "project": project,
            "project_path": project_path,
            "hours": hours,
            "sessions": len(entries),
            "changelog": entries,
            "git_commits": git_commits,
            "git_dirty_files": dirty_files,
            "git_dirty_count": len(dirty_files),
        }
        if topics:
            result["l2_topics"] = topics
        return result

    # --- Cross-project activity summary ---

    @mcp_instance.tool()
    def my_week(hours: int = 168, min_sessions: int = 1,
                include_automated: bool = False) -> dict:
        """What you shipped across ALL projects in a time window.

        Cross-project activity summary — not one project but all of them.
        Returns per-project session counts, git commit counts, estimated
        hours, and dirty state. Use for standups, time tracking, or
        "what did I do?"

        Parameters:
          hours: Lookback window (default 168 = 1 week).
          min_sessions: Only include projects with at least this many
            sessions in the window (default 1). Set to 2+ to filter noise.
          include_automated: If False (default), skip sessions classified
            as "automated" by the ML classifier.
        """
        from .mcp_server import _find_all_sessions_cached, _get_cache_index, shorten_path

        all_sessions = _find_all_sessions_cached()
        cutoff = time.time() - hours * 3600

        # Filter automated sessions
        if not include_automated:
            cache_index = _get_cache_index()
            all_sessions = [
                s for s in all_sessions
                if cache_index.get(s["session_id"], {}).get("classification") != "automated"
            ]

        # Group recent sessions by project
        by_project: dict[str, list] = {}
        for s in all_sessions:
            if s["mtime"] < cutoff:
                continue
            pd = s.get("project_dir", "")
            if not pd or pd == str(Path.home()):
                continue
            by_project.setdefault(pd, []).append(s)

        # Build per-project summary
        projects = []
        total_sessions = 0
        total_commits = 0

        for pd, sessions in sorted(by_project.items(),
                                     key=lambda x: max(s["mtime"] for s in x[1]),
                                     reverse=True):
            if len(sessions) < min_sessions:
                continue

            # Git commit count in the window
            commit_count = 0
            if Path(pd).is_dir():
                try:
                    since = datetime.fromtimestamp(cutoff).strftime("%Y-%m-%d")
                    log = subprocess.run(
                        ["git", "log", f"--since={since}", "--oneline", "-100"],
                        cwd=pd, capture_output=True, text=True, timeout=5,
                    )
                    commit_count = len([l for l in log.stdout.splitlines() if l.strip()])
                except (subprocess.TimeoutExpired, OSError):
                    pass

            # Estimate hours from session file timestamps
            est_hours = 0.0
            for s in sessions:
                try:
                    birth = s["file"].stat().st_birthtime
                    dur = s["mtime"] - birth
                    if 60 < dur < 86400:  # between 1 min and 24 hours
                        est_hours += dur / 3600
                except (OSError, AttributeError):
                    pass

            latest = max(s["mtime"] for s in sessions)
            projects.append({
                "project": shorten_path(pd),
                "sessions": len(sessions),
                "commits": commit_count,
                "est_hours": round(est_hours, 1),
                "last_active": datetime.fromtimestamp(latest).strftime("%Y-%m-%d %H:%M"),
            })
            total_sessions += len(sessions)
            total_commits += commit_count

        total_hours = round(sum(p["est_hours"] for p in projects), 1)
        return {
            "hours": hours,
            "active_projects": len(projects),
            "total_sessions": total_sessions,
            "total_commits": total_commits,
            "total_est_hours": total_hours,
            "projects": projects,
        }
