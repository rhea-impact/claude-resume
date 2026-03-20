#!/usr/bin/env python3
"""
resume-resume token budget analysis.

Thesis: resume-resume spends cheap Haiku tokens to save expensive Sonnet/Opus
tokens. This script measures actual usage and estimates the net token impact
for a Claude Max subscriber.

Two hard problems:
  1. Claude Max doesn't publish a monthly token limit — we back-calculate it
     from known rate limits and pricing.
  2. "Tokens saved" requires estimating what context re-establishment costs
     without resume-resume — inherently a model, not a measurement.

We show our math so you can tune the assumptions.
"""

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


# ── Token pricing (per million, 2025) ──────────────────────────────────────────

# claude-haiku-3 (used for summaries via claude -p)
HAIKU_IN  = 0.25   # $/M input
HAIKU_OUT = 1.25   # $/M output

# claude-sonnet-4-5 (the model Claude Max users burn budget on)
SONNET_IN  = 3.00  # $/M input
SONNET_OUT = 15.00 # $/M output

# Ratio: how many Haiku tokens = 1 Sonnet token (by cost)
HAIKU_TO_SONNET_INPUT  = SONNET_IN  / HAIKU_IN   # 12x
HAIKU_TO_SONNET_OUTPUT = SONNET_OUT / HAIKU_OUT  # 12x

# ── Claude Max budget estimate ──────────────────────────────────────────────────
#
# Anthropic publishes rate limits, not token budgets. We back-calculate:
#   - Claude Pro ($20/mo): ~45 Sonnet messages per 5-hour window
#   - Claude Max ($100/mo): 5x Pro = ~225 messages per 5-hour window
#   - Typical exchange: ~3,000 input tokens + ~1,500 output tokens = 4,500 tokens
#   - Per 5h window: 225 × 4,500 = ~1,012,500 tokens
#   - Working sessions per month: 22 days × ~1.5 windows/day = ~33 windows
#   - Monthly budget estimate: 33 × 1,012,500 = ~33M Sonnet tokens/month
#
# This is an upper bound — most users don't max out every window.
# Conservative estimate: 40% utilization = ~13M Sonnet tokens/month
#
CLAUDE_MAX_MONTHLY_SONNET_TOKENS = 13_000_000  # conservative estimate

# ── resume-resume token cost estimates per operation ───────────────────────────

# Haiku summary: session context in, summary out
SUMMARY_INPUT_TOKENS  = 1_200
SUMMARY_OUTPUT_TOKENS = 350

# merge_context output added to Sonnet context window (the merged block)
MERGE_CONTEXT_OUTPUT_TOKENS = 3_000  # avg hybrid mode

# Tokens saved by merge_context vs manual re-establishment:
# Without it: user explains context over ~3 back-and-forth exchanges
#   = 3 × (2,000 input + 800 output) = 8,400 Sonnet tokens
# With it: one merge call adds 3k tokens but replaces those exchanges
MANUAL_REESTABLISH_TOKENS = 8_400
MERGE_NET_SAVING_TOKENS = MANUAL_REESTABLISH_TOKENS - MERGE_CONTEXT_OUTPUT_TOKENS  # 5,400

# search_sessions output in context: ~800 tokens (results list)
# Without it: user and Claude explore wrong sessions, ~2,000 tokens of dead-end conversation
SEARCH_OUTPUT_TOKENS = 800
MANUAL_SEARCH_TOKENS = 2_000
SEARCH_NET_SAVING_TOKENS = MANUAL_SEARCH_TOKENS - SEARCH_OUTPUT_TOKENS  # 1,200

# ── Paths ───────────────────────────────────────────────────────────────────────

CACHE_DIR    = Path.home() / ".claude/resume-summaries"
PROJECTS_DIR = Path.home() / ".claude/projects"

_TOOL_BARE = {
    "search_sessions", "read_session", "recent_sessions",
    "session_summary", "boot_up", "resume_in_terminal",
    "merge_context", "session_timeline", "session_thread",
    "session_insights", "session_xray", "session_report",
    "session_data_science",
}
# Claude Code prefixes MCP tool names with mcp__<server>__<tool>
RESUME_TOOLS = _TOOL_BARE | {f"mcp__resume-resume__{t}" for t in _TOOL_BARE}


# ── Data collection ─────────────────────────────────────────────────────────────

def load_cache_files():
    if not CACHE_DIR.exists():
        return []
    files = []
    for f in CACHE_DIR.glob("*.json"):
        try:
            data = json.loads(f.read_text())
            data["_mtime"] = f.stat().st_mtime
            files.append(data)
        except Exception:
            continue
    return files


def scan_all_sessions_for_mcp_usage() -> dict:
    """
    Scan ALL session JSONL files for resume-resume MCP tool calls.
    Uses a fast byte-level pre-filter before parsing JSON.
    """
    if not PROJECTS_DIR.exists():
        return {}

    tool_counts = defaultdict(int)
    all_jsonl = list(PROJECTS_DIR.glob("*/*.jsonl"))
    total = len(all_jsonl)

    print(f"  Scanning {total:,} session files…", flush=True)

    hits = 0
    for i, jsonl in enumerate(all_jsonl):
        if i % 1000 == 0 and i > 0:
            print(f"  {i:,}/{total:,} scanned, {hits} sessions with resume-resume usage…", flush=True)
        try:
            raw = jsonl.read_bytes()
            # Pre-filter: skip anything that can't possibly have our tools
            if not any(t.encode() in raw for t in ["search_sessions", "merge_context", "boot_up", "session_summary", "resume-resume"]):
                continue
            hits += 1
            with open(jsonl, "r", errors="replace") as fh:
                for line in fh:
                    if "tool_use" not in line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    msg = entry.get("message", {})
                    content = msg.get("content", []) if isinstance(msg, dict) else []
                    if not isinstance(content, list):
                        continue
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "tool_use":
                            name = block.get("name", "")
                            if name in RESUME_TOOLS:
                                tool_counts[name] += 1
        except Exception:
            continue

    return dict(tool_counts)


def build_timeline(cache_files: list) -> dict:
    by_month = defaultdict(lambda: {"interactive": 0, "automated": 0, "total": 0})
    for c in cache_files:
        dt = datetime.fromtimestamp(c.get("_mtime", 0), tz=timezone.utc)
        key = dt.strftime("%Y-%m")
        by_month[key]["total"] += 1
        if c.get("classification") == "interactive":
            by_month[key]["interactive"] += 1
        else:
            by_month[key]["automated"] += 1
    return dict(sorted(by_month.items()))


def fmt_tok(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}k"
    return str(n)


def pct_of_max(tokens: int) -> str:
    p = (tokens / CLAUDE_MAX_MONTHLY_SONNET_TOKENS) * 100
    return f"{p:.2f}%"


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    print("\n━━━ resume-resume: token budget analysis ━━━\n")

    # 1. Cache
    print("Loading summary cache…")
    cache_files = load_cache_files()
    interactive = sum(1 for c in cache_files if c.get("classification") == "interactive")
    summarized  = sum(1 for c in cache_files if c.get("summary"))
    timeline    = build_timeline(cache_files)
    months      = max(1, len(timeline))

    print(f"  {len(cache_files):,} sessions indexed  ({interactive:,} interactive, {summarized:,} summarized)")
    print(f"  Across {months} months\n")

    recent = list(timeline.items())[-6:]
    for month, counts in recent:
        bar = "█" * min(36, counts["total"] // 50)
        print(f"  {month}  {bar:<36}  {counts['interactive']} human / {counts['total']} total")

    # 2. Actual MCP usage
    print("\nScanning ALL sessions for resume-resume MCP calls…")
    tool_counts = scan_all_sessions_for_mcp_usage()

    searches    = tool_counts.get("search_sessions", 0) + tool_counts.get("mcp__resume-resume__search_sessions", 0)
    merges      = tool_counts.get("merge_context", 0)   + tool_counts.get("mcp__resume-resume__merge_context", 0)
    total_calls = sum(tool_counts.values())

    print(f"\n  Total resume-resume MCP calls found: {total_calls:,}")
    if tool_counts:
        for tool, count in sorted(tool_counts.items(), key=lambda x: -x[1]):
            bar = "█" * min(30, count)
            print(f"    {tool:<30} {count:>5}  {bar}")

    # Monthly averages
    monthly_searches  = searches / months
    monthly_merges    = merges   / months
    monthly_summaries = summarized / months

    # ── Token math ────────────────────────────────────────────────────────────

    print("\n━━━ Token budget: what resume-resume COSTS ━━━")
    print(f"\n  (Claude Max estimated monthly budget: ~{fmt_tok(CLAUDE_MAX_MONTHLY_SONNET_TOKENS)} Sonnet tokens)")
    print(f"   Methodology: 5x Pro rate limits × avg exchange size × 33 sessions/month\n")

    # Haiku tokens for summaries (one-time, cached forever after)
    haiku_in  = summarized * SUMMARY_INPUT_TOKENS
    haiku_out = summarized * SUMMARY_OUTPUT_TOKENS

    # Convert Haiku tokens → Sonnet-equivalent (by cost ratio)
    sonnet_equiv_in  = haiku_in  / HAIKU_TO_SONNET_INPUT
    sonnet_equiv_out = haiku_out / HAIKU_TO_SONNET_OUTPUT
    sonnet_equiv_total = sonnet_equiv_in + sonnet_equiv_out

    print(f"  Summaries ({summarized:,} sessions, one-time, cached forever):")
    print(f"    Haiku input:       {fmt_tok(haiku_in)} tokens")
    print(f"    Haiku output:      {fmt_tok(haiku_out)} tokens")
    print(f"    Sonnet-equivalent: {fmt_tok(int(sonnet_equiv_total))} tokens  ({pct_of_max(int(sonnet_equiv_total))} of Max budget)")

    # MCP tool results add to Sonnet context window (these are real Sonnet tokens)
    search_context_tokens = searches * SEARCH_OUTPUT_TOKENS
    merge_context_tokens  = merges   * MERGE_CONTEXT_OUTPUT_TOKENS
    total_context_added   = search_context_tokens + merge_context_tokens

    print(f"\n  MCP results added to Sonnet context (real Sonnet tokens):")
    print(f"    search_sessions × {searches}:   {fmt_tok(search_context_tokens)} tokens")
    print(f"    merge_context   × {merges}:   {fmt_tok(merge_context_tokens)} tokens")
    print(f"    Total:               {fmt_tok(total_context_added)} tokens  ({pct_of_max(total_context_added)} of Max budget)")

    total_cost_tokens = int(sonnet_equiv_total) + total_context_added
    print(f"\n  Total token cost (Haiku equiv + Sonnet context):")
    print(f"    {fmt_tok(total_cost_tokens)} Sonnet-equivalent tokens  ({pct_of_max(total_cost_tokens)} of Max budget)")

    # ── Token savings ─────────────────────────────────────────────────────────

    print("\n━━━ Token budget: what resume-resume SAVES ━━━\n")
    print(f"  Each merge_context replaces ~{fmt_tok(MANUAL_REESTABLISH_TOKENS)} tokens of manual re-explanation")
    print(f"  Each search_sessions replaces ~{fmt_tok(MANUAL_SEARCH_TOKENS)} tokens of dead-end exploration\n")

    merge_savings  = merges  * MERGE_NET_SAVING_TOKENS
    search_savings = searches * SEARCH_NET_SAVING_TOKENS
    total_saved    = merge_savings + search_savings

    print(f"  merge_context  × {merges}:  {fmt_tok(merge_savings)} Sonnet tokens saved  ({pct_of_max(merge_savings)} of Max budget)")
    print(f"  search_sessions × {searches}: {fmt_tok(search_savings)} Sonnet tokens saved  ({pct_of_max(search_savings)} of Max budget)")
    print(f"\n  Total saved: {fmt_tok(total_saved)} Sonnet tokens  ({pct_of_max(total_saved)} of Max budget)")

    # ── Net ───────────────────────────────────────────────────────────────────

    net = total_saved - total_cost_tokens
    print(f"\n━━━ Net ━━━\n")
    print(f"  Tokens spent:  {fmt_tok(total_cost_tokens)}")
    print(f"  Tokens saved:  {fmt_tok(total_saved)}")
    if net > 0:
        ratio = total_saved / max(total_cost_tokens, 1)
        print(f"  Net saving:    {fmt_tok(net)} Sonnet tokens  ({pct_of_max(net)} of Max budget)")
        print(f"  Multiplier:    {ratio:.1f}x — every token spent returns {ratio:.1f} tokens")
    else:
        print(f"  Net cost:      {fmt_tok(abs(net))} tokens  (more usage = more savings)")

    # Monthly
    print(f"\n  Per month average (over {months} months):")
    print(f"    Summaries generated:   {monthly_summaries:.0f}/month")
    print(f"    Searches run:          {monthly_searches:.1f}/month")
    print(f"    Merges run:            {monthly_merges:.1f}/month")

    monthly_net = net / months
    print(f"    Net token impact:      {fmt_tok(int(monthly_net))} Sonnet tokens/month")
    print(f"                           ({pct_of_max(int(monthly_net))} of estimated Max budget)")

    # ── Steady-state (ongoing monthly cost after initial indexing) ────────────

    print(f"\n━━━ Steady-state monthly cost (after initial indexing) ━━━\n")
    print(f"  The 28k+ summaries above are a one-time cost — your backlog, indexed once.")
    print(f"  Ongoing, only NEW sessions get summarized (~500 interactive/month estimate).\n")

    new_sessions_per_month = interactive / max(1, months)
    ongoing_haiku_in  = new_sessions_per_month * SUMMARY_INPUT_TOKENS
    ongoing_haiku_out = new_sessions_per_month * SUMMARY_OUTPUT_TOKENS
    ongoing_sonnet_equiv = (ongoing_haiku_in / HAIKU_TO_SONNET_INPUT) + (ongoing_haiku_out / HAIKU_TO_SONNET_OUTPUT)

    monthly_mcp_sonnet = (monthly_searches * SEARCH_OUTPUT_TOKENS) + (monthly_merges * MERGE_CONTEXT_OUTPUT_TOKENS)
    monthly_total_cost = ongoing_sonnet_equiv + monthly_mcp_sonnet
    monthly_total_saved = (monthly_searches * SEARCH_NET_SAVING_TOKENS) + (monthly_merges * MERGE_NET_SAVING_TOKENS)
    monthly_net = monthly_total_saved - monthly_total_cost

    print(f"  New summaries/month:   {new_sessions_per_month:.0f} sessions → {fmt_tok(int(ongoing_sonnet_equiv))} Sonnet-equiv ({pct_of_max(int(ongoing_sonnet_equiv))})")
    print(f"  MCP calls/month:       {monthly_searches:.0f} searches + {monthly_merges:.1f} merges → {fmt_tok(int(monthly_mcp_sonnet))} Sonnet tokens ({pct_of_max(int(monthly_mcp_sonnet))})")
    print(f"  Total ongoing cost:    {fmt_tok(int(monthly_total_cost))} Sonnet-equiv/month  ({pct_of_max(int(monthly_total_cost))})")
    print(f"  Tokens saved/month:    {fmt_tok(int(monthly_total_saved))} Sonnet tokens  ({pct_of_max(int(monthly_total_saved))})")
    if monthly_net >= 0:
        print(f"  Net saving/month:      {fmt_tok(int(monthly_net))} Sonnet tokens  ({pct_of_max(int(monthly_net))})")
        if monthly_total_cost > 0:
            print(f"  Multiplier:            {monthly_total_saved/monthly_total_cost:.1f}x token return")
    else:
        print(f"  Net cost/month:        {fmt_tok(int(abs(monthly_net)))} tokens — savings grow with usage")
    print(f"\n  Note: Summaries cached permanently. Initial indexing is a one-time cost.\n")


if __name__ == "__main__":
    main()
