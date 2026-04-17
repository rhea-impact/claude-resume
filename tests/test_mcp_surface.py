"""MCP surface integration tests.

Round-trips every self_* MCP tool through fastmcp.Client with canonical
args. Catches schema regressions, parameter-type changes, and transport
serialization bugs that unit tests can't see (correctness-001 and -002
both slipped through unit tests for exactly this reason).

Each test calls the tool and asserts the response shape matches the
documented contract: dict tools return specific keys, list tools return
{items, count}. We don't assert exact values — just structural health.

See docs/known-issues.md dx-001.
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from fastmcp import Client

from resume_resume import mcp_server as ms


@pytest_asyncio.fixture
async def client():
    async with Client(ms.mcp) as c:
        yield c


# ---------------------------------------------------------------------------
# Dict-returning tools — assert key presence
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_self_insights_shape(client):
    r = await client.call_tool("self_insights", {"days": 1})
    d = r.data
    assert isinstance(d, dict)
    for key in ("total_calls", "total_errors", "distinct_tools", "usage",
                "dead_tools", "slow_tools", "error_prone_tools", "thresholds"):
        assert key in d, f"self_insights missing key {key!r}"


@pytest.mark.asyncio
async def test_self_load_thresholds_shape(client):
    r = await client.call_tool("self_load_thresholds", {})
    d = r.data
    assert isinstance(d, dict)
    assert "thresholds" in d
    assert "tunable_keys" in d
    assert isinstance(d["tunable_keys"], list)
    assert len(d["tunable_keys"]) > 0


@pytest.mark.asyncio
async def test_self_a1_prompt_returns_string(client):
    r = await client.call_tool("self_a1_prompt", {})
    # fastmcp wraps non-dict returns; the underlying return is str
    text = r.data if isinstance(r.data, str) else r.data.get("result", "")
    assert "# A1" in text


@pytest.mark.asyncio
async def test_self_process_decide_rejects_bad_id(client):
    from fastmcp.exceptions import ToolError
    with pytest.raises(ToolError, match="no proposal"):
        await client.call_tool("self_process_decide", {
            "proposal_id": "nonexistent",
            "verdict": "approved",
        })


@pytest.mark.asyncio
async def test_self_a1_file_rejects_low_confidence(client):
    r = await client.call_tool("self_a1_file", {
        "type": "tune",
        "title": "low conf test",
        "evidence": "x",
        "confidence": 0.1,
    })
    d = r.data
    assert d.get("skipped") == "low_confidence"


@pytest.mark.asyncio
async def test_self_a2_file_rejects_invalid_target(client):
    r = await client.call_tool("self_a2_file", {
        "target": "invalid_target",
        "change_type": "prompt_edit",
        "title": "bad target test",
        "evidence": "x",
        "confidence": 0.9,
    })
    d = r.data
    assert d.get("skipped") == "invalid_target"


# ---------------------------------------------------------------------------
# List-returning tools — assert {items, count} wrapper shape
# ---------------------------------------------------------------------------

_LIST_TOOLS = [
    ("self_recent_calls", {"limit": 3}),
    ("self_slow_calls", {"threshold_ms": 999999}),
    ("self_errors", {}),
    ("self_search", {"query": "test"}),
    ("self_bundles", {}),
    ("self_a1_output", {}),
    ("self_a1_auto_applied", {}),
    ("self_process_proposals", {}),
    ("self_proposal_history", {}),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("tool,args", _LIST_TOOLS, ids=[t[0] for t in _LIST_TOOLS])
async def test_list_tool_returns_items_count(client, tool, args):
    r = await client.call_tool(tool, args)
    d = r.data
    assert isinstance(d, dict), f"{tool} returned {type(d).__name__}"
    assert "items" in d, f"{tool} missing 'items'"
    assert "count" in d, f"{tool} missing 'count'"
    assert isinstance(d["items"], list)
    assert d["count"] == len(d["items"])


# ---------------------------------------------------------------------------
# Core tools — basic smoke (not self_*, but critical path)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_self_a2_scorecard_shape(client):
    r = await client.call_tool("self_a2_scorecard", {"days": 90})
    d = r.data
    assert isinstance(d, dict)
    assert "proposals_approved" in d
    assert "proposals_with_after_data" in d
    assert "rows" in d
    assert "summary" in d
    assert isinstance(d["rows"], list)


# ---------------------------------------------------------------------------
# _extract_crash_context unit tests (TASK-0002)
# ---------------------------------------------------------------------------

def test_extract_crash_context_basic(tmp_path):
    """Extracts last user msg, assistant msg, tool use, and count."""
    import json as _json
    from resume_resume.mcp_server import _extract_crash_context

    session = tmp_path / "session.jsonl"
    lines = [
        _json.dumps({"type": "human", "timestamp": "2026-04-16T10:00:00Z",
                     "message": {"content": "fix the login bug"}}),
        _json.dumps({"type": "assistant", "timestamp": "2026-04-16T10:01:00Z",
                     "message": {"content": [
                         {"type": "text", "text": "I'll look at the auth module."},
                         {"type": "tool_use", "name": "Read",
                          "input": {"file_path": "/src/auth.py"}},
                     ]}}),
        _json.dumps({"type": "human", "timestamp": "2026-04-16T10:05:00Z",
                     "message": {"content": "now check the tests"}}),
        _json.dumps({"type": "assistant", "timestamp": "2026-04-16T10:06:00Z",
                     "message": {"content": [
                         {"type": "tool_use", "name": "Bash",
                          "input": {"command": "pytest tests/test_auth.py"}},
                         {"type": "text", "text": "Running the auth tests now."},
                     ]}}),
    ]
    session.write_text("\n".join(lines))

    ctx = _extract_crash_context(session)
    assert ctx["last_user_msg"] == "now check the tests"
    assert "auth tests" in ctx["last_assistant_msg"]
    assert ctx["last_tool"].startswith("Bash:")
    assert "pytest" in ctx["last_tool"]
    assert ctx["message_count"] == 4
    assert ctx["duration_estimate"]  # should be "6m" or similar


def test_extract_crash_context_empty_file(tmp_path):
    session = tmp_path / "empty.jsonl"
    session.write_text("")
    from resume_resume.mcp_server import _extract_crash_context
    ctx = _extract_crash_context(session)
    assert ctx["last_user_msg"] == ""
    assert ctx["message_count"] == 0


def test_extract_crash_context_missing_file(tmp_path):
    from resume_resume.mcp_server import _extract_crash_context
    ctx = _extract_crash_context(tmp_path / "nonexistent.jsonl")
    assert ctx["last_user_msg"] == ""
    assert ctx["message_count"] == 0


# ---------------------------------------------------------------------------
# Core session tools — smoke tests (no side effects)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_search_sessions_returns_dict(client):
    r = await client.call_tool("search_sessions", {"query": "test", "limit": 2})
    d = r.data
    assert isinstance(d, dict)
    assert "items" in d
    assert "count" in d
    assert isinstance(d["items"], list)


@pytest.mark.asyncio
async def test_search_sessions_project_filter(client):
    """Project filter restricts results to matching projects."""
    # Search with a project that likely has no sessions matching "zzzznotaword"
    r = await client.call_tool("search_sessions", {
        "query": "session", "limit": 5, "project": "resume-resume"
    })
    d = r.data
    assert isinstance(d, dict)
    assert "items" in d
    # All results (if any) should be from resume-resume project
    for item in d["items"]:
        proj = item.get("project", "")
        assert "resume-resume" in proj or "resume" in proj.lower(), (
            f"Project filter leaked: {proj}"
        )


@pytest.mark.asyncio
async def test_search_sessions_hours_filter(client):
    """Hours filter restricts to recent sessions."""
    r = await client.call_tool("search_sessions", {
        "query": "session", "limit": 5, "hours": 1
    })
    d = r.data
    assert isinstance(d, dict)
    assert "items" in d
    # Can't assert exact count but shape must be correct
    assert d["count"] == len(d["items"])


@pytest.mark.asyncio
async def test_recent_sessions_returns_dict(client):
    r = await client.call_tool("recent_sessions", {"hours": 1, "limit": 2})
    d = r.data
    assert isinstance(d, dict)
    assert "items" in d
    assert "count" in d


@pytest.mark.asyncio
async def test_read_session_with_bad_id(client):
    r = await client.call_tool("read_session", {"session_id": "nonexistent-id"})
    d = r.data
    assert isinstance(d, dict)
    assert "error" in d or "messages" in d


@pytest.mark.asyncio
async def test_boot_up_returns_dict(client):
    r = await client.call_tool("boot_up", {"hours": 1})
    d = r.data
    assert isinstance(d, dict)
    assert "sessions" in d or "total" in d


@pytest.mark.asyncio
async def test_session_summary_with_bad_id(client):
    r = await client.call_tool("session_summary", {"session_id": "nonexistent-id"})
    d = r.data
    assert isinstance(d, dict)
    assert "error" in d


@pytest.mark.asyncio
async def test_list_projects_returns_list_or_wrapped(client):
    r = await client.call_tool("list_projects", {"limit": 3})
    d = r.data
    # list_projects may return a list (from l2_tools) or fail gracefully
    assert isinstance(d, (list, dict))


@pytest.mark.asyncio
async def test_merge_context_with_bad_id(client):
    r = await client.call_tool("merge_context", {"session_id": "nonexistent-id"})
    d = r.data
    assert isinstance(d, dict)
    assert "error" in d


@pytest.mark.asyncio
async def test_session_timeline_with_bad_id(client):
    r = await client.call_tool("session_timeline", {"session_id": "nonexistent-id"})
    d = r.data
    assert isinstance(d, dict)
    assert "error" in d


@pytest.mark.asyncio
async def test_session_thread_with_bad_id(client):
    r = await client.call_tool("session_thread", {"session_id": "nonexistent-id"})
    d = r.data
    assert isinstance(d, dict)
    assert "error" in d or "sessions" in d


@pytest.mark.asyncio
async def test_project_orient_shape(client):
    import os
    r = await client.call_tool("project_orient", {
        "project_path": os.getcwd()
    })
    d = r.data
    assert isinstance(d, dict)
    assert "project" in d


@pytest.mark.asyncio
async def test_dirty_repos_shape(client):
    r = await client.call_tool("dirty_repos", {})
    d = r.data
    assert isinstance(d, dict)
    assert "dirty_count" in d
    assert "clean_count" in d
    assert "total_scanned" in d
    assert isinstance(d.get("dirty"), list)
