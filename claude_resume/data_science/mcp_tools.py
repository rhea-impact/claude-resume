"""MCP tool endpoints for claude-resume data science.

Registers tools on the shared FastMCP instance from the parent mcp_server.
"""

import json
import time
from pathlib import Path

from .scanner import scan_deep
from . import analytics
from . import models


# Cache for deep scan results (avoid re-scanning within same MCP process)
_deep_cache: list[dict] | None = None
_deep_cache_ts: float = 0
_DEEP_CACHE_TTL = 300  # 5 minutes in-memory


def _get_deep_sessions(max_sessions: int = 0) -> list[dict]:
    """Get deep-scanned sessions with in-memory caching."""
    global _deep_cache, _deep_cache_ts
    now = time.time()
    if _deep_cache is not None and (now - _deep_cache_ts) < _DEEP_CACHE_TTL:
        return _deep_cache
    _deep_cache = scan_deep(max_sessions=max_sessions)
    _deep_cache_ts = now
    return _deep_cache


def register_tools(mcp_instance):
    """Register data science tools on the MCP server."""

    @mcp_instance.tool()
    def session_insights(
        section: str = "all",
        max_sessions: int = 0,
    ) -> dict:
        """Deep analytics on ALL your Claude Code sessions. Mind-blowing stats.

        Analyzes 5000+ sessions for temporal patterns, project insights,
        tool usage, personality traits, predictions, records, and fun facts.

        First call takes 30-60s (parses JSONL files). Subsequent calls are cached.

        Sections: "all", "overview", "temporal", "projects", "tools",
                  "models", "records", "predictions", "personality", "fun_facts"
        """
        sessions = _get_deep_sessions(max_sessions)

        if not sessions:
            return {"error": "No sessions found. Run some Claude Code sessions first!"}

        if section == "all":
            return analytics.analyze(sessions)
        elif section == "overview":
            return analytics.overview(sessions)
        elif section == "temporal":
            return analytics.temporal_patterns(sessions)
        elif section == "projects":
            return analytics.project_insights(sessions)
        elif section == "tools":
            return analytics.tool_usage(sessions)
        elif section == "models":
            return analytics.model_usage(sessions)
        elif section == "records":
            return analytics.streaks_and_records(sessions)
        elif section == "predictions":
            return analytics.predictions(sessions)
        elif section == "personality":
            return analytics.personality_profile(sessions)
        elif section == "fun_facts":
            return {"fun_facts": analytics.fun_facts(sessions)}
        else:
            return {"error": f"Unknown section: {section}. Use: all, overview, temporal, projects, tools, models, records, predictions, personality, fun_facts"}

    @mcp_instance.tool()
    def session_xray(session_id: str) -> dict:
        """Deep x-ray of a single session — duration, tools, tokens, branches, everything."""
        from .scanner import _parse_single_session
        from ..sessions import find_all_sessions, shorten_path

        all_sessions = find_all_sessions()
        target = None
        for s in all_sessions:
            if s["session_id"] == session_id:
                home = str(Path.home())
                project = s["project_dir"]
                short = project.replace(home, "~") if project.startswith(home) else project
                parts = short.split("/")
                target = {
                    "session_id": s["session_id"],
                    "project_dir": project,
                    "project_short": short,
                    "repo": parts[-1] if len(parts) > 1 else short,
                    "mtime": s["mtime"],
                    "size": s["size"],
                    "date": "",
                    "hour": 0,
                    "weekday": "",
                    "weekday_num": 0,
                    "file": str(s["file"]),
                }
                from datetime import datetime
                dt = datetime.fromtimestamp(s["mtime"])
                target["date"] = dt.strftime("%Y-%m-%d")
                target["hour"] = dt.hour
                target["weekday"] = dt.strftime("%A")
                target["weekday_num"] = dt.weekday()
                break

        if target is None:
            return {"error": f"Session {session_id} not found"}

        return _parse_single_session(target)

    @mcp_instance.tool()
    def session_report(output_path: str = "", org: str = "") -> dict:
        """Generate a stunning HTML report of all your Claude session analytics.

        Creates a self-contained HTML file with charts, heatmaps, personality
        traits, predictions, and fun facts. Opens in browser.

        Args:
            output_path: Where to save. Default: ~/claude-sessions-report.html
            org: Filter to an org (e.g. "eidos-agi", "personal", "aic",
                 "greenmark-waste-solutions", "rheaimpact"). Empty = all.

        Returns the file path.
        """
        from .report import generate_report
        path = generate_report(output_path=output_path or None, org=org)
        return {"report": path, "note": f"Open {path} in your browser"}

    @mcp_instance.tool()
    def session_data_science(
        analysis: str = "all",
    ) -> dict:
        """REAL data science on your Claude sessions.

        Runs ML models, statistical tests, and predictive analytics:

        Analyses:
          - "all": Everything below
          - "clustering": K-means session type discovery with silhouette optimization
          - "markov": Project transition probabilities (what you work on after X)
          - "circadian": Sinusoidal fit to your biological activity clock
          - "power_law": Pareto exponent — do marathon sessions drive your output?
          - "anomalies": DBSCAN outlier detection on session feature space
          - "flow": Flow state detection (sustained high-throughput sessions)
          - "burnout": Burnout risk signals and trend analysis
          - "cooccurrence": Project co-occurrence network
          - "duration": Distribution analysis with bimodality testing
          - "entropy": Shannon entropy — how predictable are you?

        First call: ~15s (deep scan + model fitting). Cached after.
        """
        sessions = _get_deep_sessions()
        if not sessions:
            return {"error": "No sessions found"}

        if analysis == "all":
            return models.full_analysis(sessions)
        elif analysis == "clustering":
            return models.cluster_sessions(sessions)
        elif analysis == "markov":
            return models.project_markov_chain(sessions)
        elif analysis == "circadian":
            return models.circadian_model(sessions)
        elif analysis == "power_law":
            return models.power_law_analysis(sessions)
        elif analysis == "anomalies":
            return models.detect_anomalies(sessions)
        elif analysis == "flow":
            return models.detect_flow_states(sessions)
        elif analysis == "burnout":
            return models.burnout_indicators(sessions)
        elif analysis == "cooccurrence":
            return models.project_cooccurrence(sessions)
        elif analysis == "duration":
            return models.duration_distribution(sessions)
        elif analysis == "entropy":
            return models.work_entropy(sessions)
        else:
            return {"error": f"Unknown analysis: {analysis}"}
