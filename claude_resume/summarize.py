"""AI-powered session summarization via claude -p.

Three tiers:
  Tier 1 — Quick (Haiku):    Always auto-generated. ~1,500 tokens. Title, goal, state, files.
  Tier 2 — Deep  (Haiku):    On demand for important sessions. ~3,000 tokens. Adds decisions, next steps.
  Tier 3 — Insight (Sonnet): Explicit only. ~6,000 tokens. Architecture map, blockers, confidence, patterns.

Smart auto-selection triggers Tier 2 when:
  - Session has >40 user messages
  - Uncommitted git changes exist
  - Session JSONL is >500KB (long session)

Tier 3 is always explicit.
"""

import json
import subprocess

# ── Tier thresholds for auto-promotion ────────────────────────────────────────
TIER2_MSG_THRESHOLD  = 40    # promote if user sent ≥ this many messages
TIER2_SIZE_THRESHOLD = 500_000  # promote if JSONL ≥ 500KB

QUICK_SCHEMA = json.dumps({
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "goal": {"type": "string"},
        "what_was_done": {"type": "string"},
        "state": {"type": "string"},
        "files": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["title", "goal", "what_was_done", "state", "files"],
})

DEEP_SCHEMA = json.dumps({
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "objective": {"type": "string"},
        "progress": {"type": "string"},
        "state": {"type": "string"},
        "next_steps": {"type": "string"},
        "files": {"type": "array", "items": {"type": "string"}},
        "decisions_made": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["title", "objective", "progress", "state", "next_steps", "files", "decisions_made"],
})


PATTERNS_SCHEMA = json.dumps({
    "type": "object",
    "properties": {
        "prompt_patterns": {
            "type": "object",
            "properties": {
                "effective": {"type": "array", "items": {"type": "object", "properties": {"example": {"type": "string"}, "why": {"type": "string"}}}},
                "ineffective": {"type": "array", "items": {"type": "object", "properties": {"example": {"type": "string"}, "issue": {"type": "string"}}}},
                "tips": {"type": "array", "items": {"type": "string"}},
            },
        },
        "workflow_patterns": {
            "type": "object",
            "properties": {
                "common_sequences": {"type": "array", "items": {"type": "object", "properties": {"tools": {"type": "array", "items": {"type": "string"}}, "context": {"type": "string"}, "efficiency": {"type": "string"}}}},
                "iteration_style": {"type": "string"},
            },
        },
        "anti_patterns": {"type": "array", "items": {"type": "object", "properties": {"pattern": {"type": "string"}, "cost": {"type": "string"}, "fix": {"type": "string"}}}},
        "key_lesson": {"type": "string"},
    },
    "required": ["prompt_patterns", "workflow_patterns", "anti_patterns", "key_lesson"],
})


INSIGHT_SCHEMA = json.dumps({
    "type": "object",
    "properties": {
        "title":           {"type": "string"},
        "objective":       {"type": "string"},
        "progress":        {"type": "string"},
        "state":           {"type": "string"},
        "next_steps":      {"type": "string"},
        "files":           {"type": "array", "items": {"type": "string"}},
        "decisions_made":  {"type": "array", "items": {"type": "string"}},
        "architecture":    {"type": "string"},
        "blockers":        {"type": "array", "items": {"type": "string"}},
        "confidence":      {"type": "string", "enum": ["high", "medium", "low"]},
        "session_quality": {"type": "string"},
        "key_insight":     {"type": "string"},
    },
    "required": [
        "title", "objective", "progress", "state", "next_steps",
        "files", "decisions_made", "architecture", "blockers",
        "confidence", "session_quality", "key_insight",
    ],
})


def _call_claude(prompt: str, context: dict, timeout: int = 30,
                 schema: str | None = None, model: str = "claude-haiku-4-5-20251001") -> dict:
    try:
        cmd = ["claude", "-p", prompt, "--no-session-persistence", "--output-format", "json",
               "--model", model]
        if schema:
            cmd.extend(["--json-schema", schema])
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                                stdin=subprocess.DEVNULL)
        output = result.stdout.strip()
        parsed = json.loads(output)
        if "structured_output" in parsed and isinstance(parsed["structured_output"], dict):
            return parsed["structured_output"]
        if "result" in parsed and isinstance(parsed["result"], dict):
            return parsed["result"]
        if "result" in parsed and isinstance(parsed["result"], str):
            return json.loads(parsed["result"])
        return parsed
    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception):
        last = context.get("last_messages", [""])[-1] if context.get("last_messages") else ""
        first = context.get("first_messages", [""])[0] if context.get("first_messages") else ""
        return {
            "title": first[:60] if first else "Unknown session",
            "state": last[:120] if last else "No context available",
            "files": [],
        }


def summarize_quick(context: dict, project_dir: str, git: dict | None = None) -> dict:
    """Quick summary via claude -p."""
    git_section = ""
    if git and git.get("is_git_repo"):
        if git.get("recent_commits"):
            git_section += f"\nRECENT GIT COMMITS:\n{git['recent_commits']}\n"
        if git.get("uncommitted_changes"):
            git_section += f"\nUNCOMMITTED CHANGES:\n{git['uncommitted_changes']}\n"

    prompt = f"""You are summarizing a Claude Code session to help a developer resume after a crash.
The session was in: {project_dir}

FIRST USER MESSAGES (original goal):
{json.dumps(context['first_messages'], indent=2)}

LAST USER MESSAGES (where they left off):
{json.dumps(context['last_messages'], indent=2)}

LAST ASSISTANT RESPONSES (what Claude was doing):
{json.dumps(context['last_assistant'], indent=2)}

RECENT TOOLS USED: {', '.join(context['recent_tools'][-10:])}
TOTAL USER MESSAGES IN SESSION: {context['total_user_messages']}
{git_section}
Return ONLY a JSON object with these fields:
- "title": A descriptive title for what this session was about (max 12 words)
- "goal": What was the developer trying to accomplish? Be specific about the feature, bug, or task. (2-3 sentences)
- "what_was_done": What was actually accomplished so far? Mention specific changes, files modified, features built, bugs fixed. Reference what the assistant was doing. (2-4 sentences)
- "state": Where exactly did the work stop? What was the last thing being worked on? Were there errors, blockers, or was it mid-task? Be very specific — mention file names, function names, error messages if visible. (2-3 sentences)
- "files": Array of the 3-6 most important file paths being worked on (infer from tool usage and conversation)

Be detailed and specific. This summary is the developer's lifeline for getting back to work after a crash. Vague summaries are useless.

Return raw JSON only. No markdown, no code fences, no explanation."""

    return _call_claude(prompt, context, schema=QUICK_SCHEMA)


def summarize_deep(context: dict, project_dir: str, quick_summary: dict, git: dict | None = None) -> dict:
    """Deep second-pass summary with much more context."""
    git_section = ""
    if git and git.get("is_git_repo"):
        if git.get("recent_commits"):
            git_section += f"\nRECENT GIT COMMITS (actual code changes made):\n{git['recent_commits']}\n"
        if git.get("uncommitted_changes"):
            git_section += f"\nUNCOMMITTED CHANGES (work in progress at time of crash):\n{git['uncommitted_changes']}\n"

    prompt = f"""You are doing a DEEP ANALYSIS of a Claude Code session to help a developer fully regain context after a crash.
The session was in: {project_dir}

QUICK SUMMARY (from first pass): {json.dumps(quick_summary)}

FIRST USER MESSAGES (original goal):
{json.dumps(context['first_messages'], indent=2)}

FIRST ASSISTANT RESPONSES:
{json.dumps(context['first_assistant'], indent=2)}

LAST USER MESSAGES (most recent work):
{json.dumps(context['last_messages'], indent=2)}

LAST ASSISTANT RESPONSES:
{json.dumps(context['last_assistant'], indent=2)}

ALL UNIQUE TOOLS USED: {', '.join(context['all_tools'][:30])}
RECENT TOOL SEQUENCE: {', '.join(context['recent_tools'])}
TOTAL USER MESSAGES: {context['total_user_messages']}
TOTAL JSONL LINES: {context['total_lines']}
{git_section}

Return ONLY a JSON object with these fields:
- "title": Descriptive title (max 10 words)
- "objective": What was the overall goal of this session? (2-3 sentences)
- "progress": What was accomplished before the crash? Key milestones reached. (2-3 sentences)
- "state": Exactly where the work left off — the immediate task, any errors or blockers. (1-2 sentences)
- "next_steps": What should the developer do first when resuming? (1-2 sentences)
- "files": Array of the key file paths or entities involved (up to 6)
- "decisions_made": Array of key architectural or implementation decisions made during the session (up to 4 short strings)

Return raw JSON only. No markdown, no code fences, no explanation."""

    return _call_claude(prompt, context, timeout=90, schema=DEEP_SCHEMA)


def summarize_insight(context: dict, project_dir: str, deep_summary: dict,
                      git: dict | None = None, session_file_size: int = 0) -> dict:
    """Tier 3 insight summary using Sonnet — explicit only.

    Builds on the Tier 2 deep summary to produce a richer analysis:
    architecture map, blockers, confidence score, session quality assessment,
    and the single most important insight from the session.
    """
    git_section = ""
    if git and git.get("is_git_repo"):
        if git.get("recent_commits"):
            git_section += f"\nGIT COMMITS THIS SESSION:\n{git['recent_commits']}\n"
        if git.get("uncommitted_changes"):
            git_section += f"\nUNCOMMITTED CHANGES:\n{git['uncommitted_changes']}\n"

    size_note = f"\nSession size: {session_file_size // 1024}KB JSONL ({context['total_lines']} lines, {context['total_user_messages']} user messages)"

    prompt = f"""You are a senior engineering lead doing a thorough debrief of a Claude Code session.
The session was in: {project_dir}
{size_note}

DEEP SUMMARY (Tier 2 analysis): {json.dumps(deep_summary, indent=2)}

FULL MESSAGE SAMPLES:
First messages: {json.dumps(context['first_messages'], indent=2)}
Last messages: {json.dumps(context['last_messages'], indent=2)}
First assistant: {json.dumps(context['first_assistant'], indent=2)}
Last assistant: {json.dumps(context['last_assistant'], indent=2)}

ALL TOOLS USED ({len(context['all_tools'])} unique): {', '.join(context['all_tools'][:40])}
RECENT TOOL SEQUENCE: {', '.join(context['recent_tools'])}
{git_section}

Return ONLY a JSON object with:
- "title": Concise, specific title (max 10 words)
- "objective": The full goal of this session, including the WHY if discernible (2-3 sentences)
- "progress": Specific milestones reached — name files changed, tests passed, features shipped (3-4 sentences)
- "state": Exact stopping point — file, function, line if visible; error message if present (1-2 sentences)
- "next_steps": The single best first action to take on resume, with specific file/function (1-2 sentences)
- "files": Up to 8 most important file paths (most-modified first)
- "decisions_made": Up to 5 specific architectural or implementation decisions made (quote exact choices)
- "architecture": How do the changed components fit together? What pattern or structure was established? (2-3 sentences)
- "blockers": Array of unresolved issues, errors, or open questions blocking progress (empty array if none)
- "confidence": "high" if session had clear goal + clear progress, "medium" if goal shifted or partial, "low" if chaotic
- "session_quality": One sentence describing the efficiency and focus of this session
- "key_insight": The single most important thing to know about this session — what makes it notable or what you'd tell someone resuming cold

Be surgical and specific. This is for a developer who needs to pick up a thread with zero warm-up.
Return raw JSON only. No markdown, no code fences."""

    return _call_claude(prompt, context, timeout=120, schema=INSIGHT_SCHEMA,
                        model="claude-sonnet-4-6")


def auto_tier(context: dict, session_file_size: int, git: dict | None = None) -> int:
    """Pick the appropriate summarization tier based on session characteristics.

    Returns: 1, 2, or 3
    - Tier 1: Default — fast, cheap, good enough for most sessions
    - Tier 2: Promoted when session is substantive (many messages or large file or dirty git)
    - Tier 3: Never auto-selected — always explicit
    """
    msgs = context.get("total_user_messages", 0)
    has_dirty = bool(git and git.get("uncommitted_changes"))

    if msgs >= TIER2_MSG_THRESHOLD or session_file_size >= TIER2_SIZE_THRESHOLD or has_dirty:
        return 2
    return 1


def analyze_patterns(context: dict, project_dir: str, summary: dict) -> dict:
    """Analyze a session for prompting and workflow patterns."""
    prompt = f"""Analyze this Claude Code session for patterns that help the developer improve.
Session: {project_dir}
Summary: {json.dumps(summary)}

USER MESSAGES (samples):
{json.dumps(context['first_messages'] + context['last_messages'], indent=2)}

ASSISTANT RESPONSES (samples):
{json.dumps(context['first_assistant'] + context['last_assistant'], indent=2)}

TOOL SEQUENCE: {', '.join(context['recent_tools'])}
ALL TOOLS: {', '.join(context['all_tools'][:20])}
TOTAL MESSAGES: {context['total_user_messages']}

Return JSON with:
- "prompt_patterns": {{"effective": [{{"example": "quoted instruction", "why": "reason"}}], "ineffective": [{{"example": "quoted instruction", "issue": "problem"}}], "tips": ["actionable advice"]}}
- "workflow_patterns": {{"common_sequences": [{{"tools": ["Read","Edit"], "context": "when used", "efficiency": "high|medium|low"}}], "iteration_style": "single-pass|iterative|exploratory"}}
- "anti_patterns": [{{"pattern": "description", "cost": "time wasted", "fix": "how to avoid"}}]
- "key_lesson": "most important takeaway from this session"

Focus on ACTIONABLE insights. Quote specific examples from the messages. Be concise.
Return raw JSON only."""

    return _call_claude(prompt, context, timeout=90, schema=PATTERNS_SCHEMA)
