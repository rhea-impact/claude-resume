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
    r = await client.call_tool("self_process_decide", {
        "proposal_id": "nonexistent",
        "verdict": "approved",
    })
    # fastmcp wraps ValueError as is_error=True or includes error text
    err_text = str(r.data).lower() if r.data else ""
    assert r.is_error or "error" in err_text or "no proposal" in err_text or "valueerror" in err_text, (
        f"Expected error for nonexistent proposal_id, got is_error={r.is_error} data={r.data}"
    )


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
async def test_dirty_repos_shape(client):
    r = await client.call_tool("dirty_repos", {})
    d = r.data
    assert isinstance(d, dict)
    assert "dirty_count" in d
    assert "clean_count" in d
    assert "total_scanned" in d
    assert isinstance(d.get("dirty"), list)
