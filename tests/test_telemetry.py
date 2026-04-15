"""Tests for the telemetry writer + middleware."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from resume_resume import telemetry


def test_telemetry_enabled_default(monkeypatch):
    monkeypatch.delenv("RESUME_RESUME_TELEMETRY", raising=False)
    assert telemetry.telemetry_enabled() is True


def test_telemetry_disabled_by_env(monkeypatch):
    monkeypatch.setenv("RESUME_RESUME_TELEMETRY", "0")
    assert telemetry.telemetry_enabled() is False


def test_telemetry_root_uses_username():
    root = telemetry.telemetry_root()
    assert root.parts[-3:] == (".resume-resume", "telemetry", root.parts[-1])
    # username should be the last segment
    import getpass
    assert root.parts[-1] == getpass.getuser()


def test_write_event_appends_jsonl(tmp_path: Path):
    target = tmp_path / "test.jsonl"
    telemetry.write_event({"tool": "a", "status": "ok"}, path=target)
    telemetry.write_event({"tool": "b", "status": "error"}, path=target)

    lines = target.read_text().strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    second = json.loads(lines[1])
    assert first["tool"] == "a"
    assert second["tool"] == "b"


def test_write_event_creates_parent_dir(tmp_path: Path):
    target = tmp_path / "nested" / "deeper" / "day.jsonl"
    telemetry.write_event({"tool": "x"}, path=target)
    assert target.exists()
    assert json.loads(target.read_text().strip())["tool"] == "x"


def test_write_event_never_raises(tmp_path: Path, monkeypatch):
    # Non-writable path should be swallowed, not raise.
    bad = tmp_path / "readonly"
    bad.mkdir()
    bad.chmod(0o400)
    try:
        telemetry.write_event({"tool": "x"}, path=bad / "nope" / "f.jsonl")
    finally:
        bad.chmod(0o700)


def test_jsonable_passthrough_for_primitives():
    assert telemetry._jsonable({"a": 1, "b": [2, 3]}) == {"a": 1, "b": [2, 3]}
    assert telemetry._jsonable("hello") == "hello"
    assert telemetry._jsonable(42) == 42


def test_jsonable_handles_objects_with_dict():
    class Foo:
        def __init__(self):
            self.x = 1
            self.y = "two"

    out = telemetry._jsonable(Foo())
    assert out == {"x": 1, "y": "two"}


def test_jsonable_handles_nested_unserializable():
    class Bar:
        def __init__(self, v):
            self.v = v

    out = telemetry._jsonable([Bar(1), Bar(2)])
    assert out == [{"v": 1}, {"v": 2}]


def test_jsonable_falls_back_to_repr():
    class Weird:
        __slots__ = ()

        def __repr__(self):
            return "<weird>"

    assert telemetry._jsonable(Weird()) == "<weird>"


def test_safe_size_returns_length():
    assert telemetry._safe_size({"a": 1}) == len('{"a": 1}')
    assert telemetry._safe_size("hi") == len('"hi"')


def test_today_path_uses_date(tmp_path: Path):
    p = telemetry._today_path(tmp_path)
    expected_day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    assert p.name == f"{expected_day}.jsonl"
    assert p.parent == tmp_path


@pytest.mark.asyncio
async def test_middleware_captures_successful_call(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(telemetry, "_today_path", lambda: tmp_path / "today.jsonl")

    class FakeMsg:
        name = "test_tool"
        arguments = {"q": "hello"}

    class FakeCtx:
        message = FakeMsg()
        fastmcp_context = None

    async def fake_next(ctx):
        return {"result": "ok", "items": [1, 2, 3]}

    mw = telemetry.TelemetryMiddleware()
    out = await mw.on_call_tool(FakeCtx(), fake_next)
    assert out == {"result": "ok", "items": [1, 2, 3]}

    lines = (tmp_path / "today.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["tool"] == "test_tool"
    assert event["args"] == {"q": "hello"}
    assert event["status"] == "ok"
    assert event["error_type"] is None
    assert event["result"] == {"result": "ok", "items": [1, 2, 3]}
    assert event["duration_ms"] >= 0
    assert event["pid"] == os.getpid()


@pytest.mark.asyncio
async def test_middleware_captures_error(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(telemetry, "_today_path", lambda: tmp_path / "today.jsonl")

    class FakeMsg:
        name = "broken_tool"
        arguments = {}

    class FakeCtx:
        message = FakeMsg()
        fastmcp_context = None

    async def fake_next(ctx):
        raise ValueError("boom")

    mw = telemetry.TelemetryMiddleware()
    with pytest.raises(ValueError):
        await mw.on_call_tool(FakeCtx(), fake_next)

    lines = (tmp_path / "today.jsonl").read_text().strip().splitlines()
    event = json.loads(lines[0])
    assert event["tool"] == "broken_tool"
    assert event["status"] == "error"
    assert event["error_type"] == "ValueError"
    assert event["error_msg"] == "boom"
    assert event["result"] is None
    assert "Traceback" in (event["error_tb"] or "")


@pytest.mark.asyncio
async def test_middleware_respects_disable_env(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("RESUME_RESUME_TELEMETRY", "0")
    monkeypatch.setattr(telemetry, "_today_path", lambda: tmp_path / "today.jsonl")

    class FakeMsg:
        name = "test_tool"
        arguments = {}

    class FakeCtx:
        message = FakeMsg()
        fastmcp_context = None

    async def fake_next(ctx):
        return "ok"

    mw = telemetry.TelemetryMiddleware()
    await mw.on_call_tool(FakeCtx(), fake_next)
    assert not (tmp_path / "today.jsonl").exists()


# ---------------------------------------------------------------------------
# Query-side tests (telemetry_query.py)
# ---------------------------------------------------------------------------

from datetime import timedelta
from resume_resume import telemetry_query as tq


def _write_events(root: Path, day_offset: int, events: list[dict]) -> None:
    date = (datetime.now(timezone.utc) - timedelta(days=day_offset)).strftime("%Y-%m-%d")
    target = root / f"{date}.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")


def _mk_event(tool: str, duration_ms: float = 10.0, status: str = "ok",
              args: dict | None = None, result=None, ts_offset_s: int = 0) -> dict:
    ts = (datetime.now(timezone.utc) - timedelta(seconds=ts_offset_s)).isoformat()
    return {
        "ts": ts,
        "session_id": "s1",
        "tool": tool,
        "args": args or {},
        "duration_ms": duration_ms,
        "status": status,
        "error_type": None if status == "ok" else "ValueError",
        "error_msg": None if status == "ok" else "boom",
        "result_size": len(json.dumps(result)) if result is not None else 0,
        "result": result,
        "pid": 123,
    }


def test_iter_events_reads_across_days(tmp_path: Path):
    _write_events(tmp_path, 0, [_mk_event("today_tool")])
    _write_events(tmp_path, 1, [_mk_event("yesterday_tool")])
    _write_events(tmp_path, 5, [_mk_event("old_tool")])

    got = list(tq.iter_events(days=2, root=tmp_path))
    tools = {e["tool"] for e in got}
    assert tools == {"today_tool", "yesterday_tool"}


def test_iter_events_skips_bad_lines(tmp_path: Path):
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    (tmp_path / f"{date}.jsonl").write_text(
        '{"tool": "good"}\ngarbage not json\n{"tool": "also good"}\n'
    )
    got = list(tq.iter_events(days=1, root=tmp_path))
    assert [e["tool"] for e in got] == ["good", "also good"]


def test_load_events_filters(tmp_path: Path):
    _write_events(tmp_path, 0, [
        _mk_event("a", status="ok"),
        _mk_event("a", status="error"),
        _mk_event("b", status="ok"),
    ])
    assert len(tq.load_events(days=1, tool="a", root=tmp_path)) == 2
    assert len(tq.load_events(days=1, tool="a", status="error", root=tmp_path)) == 1
    assert len(tq.load_events(days=1, status="error", root=tmp_path)) == 1


def test_percentile():
    assert tq._percentile([], 0.5) == 0
    assert tq._percentile([10], 0.5) == 10
    assert tq._percentile([1, 2, 3, 4, 5], 0.5) == 3
    assert tq._percentile([1, 2, 3, 4, 5], 0.95) == pytest.approx(4.8)


def test_usage_summary_aggregates():
    events = [
        _mk_event("fast", duration_ms=10),
        _mk_event("fast", duration_ms=20),
        _mk_event("slow", duration_ms=2000),
        _mk_event("slow", duration_ms=3000, status="error"),
        _mk_event("slow", duration_ms=1500),
    ]
    rows = tq.usage_summary(events)
    by_tool = {r["tool"]: r for r in rows}
    assert by_tool["slow"]["count"] == 3
    assert by_tool["slow"]["errors"] == 1
    assert by_tool["slow"]["error_rate"] == pytest.approx(0.3333, abs=1e-3)
    assert by_tool["fast"]["avg_ms"] == 15
    # sorted desc by count
    assert rows[0]["tool"] == "slow"


def test_dead_and_slow_and_error_prone():
    summary = [
        {"tool": "dead", "count": 1, "errors": 0, "error_rate": 0, "avg_ms": 5, "p50_ms": 5, "p95_ms": 5, "max_ms": 5},
        {"tool": "slow", "count": 100, "errors": 0, "error_rate": 0, "avg_ms": 50, "p50_ms": 50, "p95_ms": 1500, "max_ms": 2000},
        {"tool": "buggy", "count": 20, "errors": 5, "error_rate": 0.25, "avg_ms": 10, "p50_ms": 10, "p95_ms": 15, "max_ms": 20},
    ]
    assert [r["tool"] for r in tq.dead_tools(summary, threshold=1)] == ["dead"]
    assert [r["tool"] for r in tq.slow_tools(summary, 1000)] == ["slow"]
    assert [r["tool"] for r in tq.error_prone_tools(summary, 0.05)] == ["buggy"]


def test_abandoned_queries_finds_empty_results():
    events = [
        _mk_event("search_sessions", args={"query": "nothing"}, result=[]),
        _mk_event("search_sessions", args={"query": "hits"}, result=[{"id": 1}]),
        _mk_event("recent_sessions", result=[]),
        _mk_event("unrelated_tool", result=[]),
    ]
    out = tq.abandoned_queries(events)
    tools = [e["tool"] for e in out]
    assert "search_sessions" in tools
    assert "recent_sessions" in tools
    assert "unrelated_tool" not in tools
    assert len(out) == 2


def test_bm25_search_ranks_matching_calls():
    events = [
        _mk_event("search_sessions", args={"query": "apple"}),
        _mk_event("search_sessions", args={"query": "banana orange"}),
        _mk_event("recent_sessions", args={"hours": 24}),
    ]
    hits = tq.bm25_search(events, "apple", limit=5)
    assert hits
    assert "apple" in json.dumps(hits[0])
    # every hit has a score attached
    assert all("score" in h for h in hits)


def test_bm25_search_empty_query():
    assert tq.bm25_search([_mk_event("a")], "") == []
    assert tq.bm25_search([], "anything") == []


def test_session_bundles_groups_bursts():
    events = [
        _mk_event("a", ts_offset_s=0, duration_ms=100),
        _mk_event("b", ts_offset_s=-5, duration_ms=100),
        _mk_event("c", ts_offset_s=-600, duration_ms=100),
    ]
    bundles = tq.session_bundles(events, gap_seconds=30)
    assert len(bundles) == 2
    call_counts = sorted(b["call_count"] for b in bundles)
    assert call_counts == [1, 2]


def test_insights_report_shape(tmp_path: Path):
    _write_events(tmp_path, 0, [
        _mk_event("search_sessions", duration_ms=5, result=[{"id": 1}]),
        _mk_event("search_sessions", duration_ms=5, args={"query": "x"}, result=[]),
        _mk_event("slow_tool", duration_ms=2000),
        _mk_event("broken_tool", status="error"),
    ])
    report = tq.insights_report(days=1, root=tmp_path)
    assert report["total_calls"] == 4
    assert report["total_errors"] == 1
    assert report["distinct_tools"] == 3
    assert len(report["abandoned_queries"]) >= 1
    assert any(r["tool"] == "search_sessions" for r in report["usage"])
