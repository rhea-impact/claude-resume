# claude-resume

## The Problem Nobody Talks About

Claude Code is stateless between sessions. Every time you close a terminal, switch projects, reboot your machine, or your laptop kernel panics mid-debug — the conversation is gone. Not the data — Claude Code dutifully writes every exchange to JSONL files in `~/.claude/projects/`. But the *context* is gone. The understanding of what you were doing, why you were doing it, and where you left off.

This matters more than it sounds like it should. A developer running Claude Code across three projects doesn't just lose a conversation when their Mac crashes. They lose the thread. They lose the decisions that were made forty messages ago. They lose the fact that they'd already tried approach A and it failed for a specific reason, so they pivoted to approach B. They lose the half-finished refactor that was three files into a six-file change.

The JSONL files are there. Thousands of them. But browsing them manually is like reading a database dump to remember what you had for lunch. The format is machine-readable, not human-readable. The files are huge — some sessions run to hundreds of megabytes. There's no index, no search, no way to ask "which of my 3,000 sessions was the one where I figured out the auth token refresh bug?"

claude-resume exists because **session continuity is the bottleneck** in AI-assisted development, and nobody is solving it.

## What It Actually Is

claude-resume is two things in one package:

1. **A terminal UI (TUI)** for humans — a full-screen session picker that shows your recent Claude Code sessions, tells you what each one was doing, and lets you jump back in with one keypress.

2. **An MCP server** for Claude — a set of tools that let any Claude Code session search, read, summarize, fork, merge, and resume other sessions programmatically.

The TUI is what you use when you sit down at your machine after a crash. The MCP server is what Claude uses to move context between sessions while you're working. Together, they turn Claude Code's flat pile of JSONL files into a navigable, searchable, composable session graph.

## Why It Was Built

### The Crash Recovery Problem

The original motivation was literal crash recovery. macOS kernel panics. Terminals get killed. Laptops go to sleep mid-session and never wake up properly. You reboot, and now you're staring at a blank terminal trying to remember what you were doing across three different projects.

Before claude-resume, the recovery process was: open `~/.claude/projects/`, eyeball timestamps on hundreds of JSONL files, guess which ones were recent, `cat` one to see if it looks familiar, repeat until you find the right sessions, then manually construct `claude --resume <uuid>` commands.

This took 5-15 minutes per crash. For someone running Claude Code heavily, that's multiple times a week. The friction compounds — you start avoiding complex multi-session workflows because the recovery cost is too high.

### The Context Fragmentation Problem

But crash recovery was just the presenting symptom. The deeper problem is that Claude Code sessions are islands. Each session starts fresh. If you spent two hours in session A researching an authentication architecture, and then start session B to implement it, session B knows nothing about session A. You end up re-explaining context, re-making decisions, re-discovering constraints.

Developers work around this by keeping sessions alive as long as possible, which creates its own problems — context windows fill up, conversations get confused, Claude starts hallucinating about things discussed 200 messages ago. Or they manually copy-paste summaries between sessions, which is tedious and lossy.

The real need isn't just "find my crashed sessions." It's "let my sessions talk to each other."

### The Visibility Problem

When you're running Claude Code across multiple projects — as anyone doing serious AI-assisted development does — you lose visibility into your own work. Which sessions have uncommitted changes? Which ones were mid-task when you switched away? Where did you leave that investigation into the database performance issue?

Your sessions are your work log, but they're invisible. You can't search them. You can't summarize them. You can't ask "what have I been working on this week?" and get an answer. All that context is locked in binary-looking JSONL files that no human wants to read.

## How It Works

### The TUI: Session Discovery and Recovery

When you run `claude-resume`, it scans `~/.claude/projects/` for JSONL session files modified within the time window (default: last 4 hours). For each session, it:

1. **Parses the JSONL** to extract first messages (what the goal was), last messages (where you left off), recent tools used, and conversation statistics.

2. **Generates an AI summary** using `claude -p` with Haiku (fast and cheap). The summary includes a title, the goal, what was accomplished, the current state, and key files. Summaries are cached in `~/.claude/resume-summaries/` — second launch is instant.

3. **Scores each session by interruption severity.** Sessions that crashed mid-tool-use score higher than sessions that ended cleanly. Sessions with uncommitted git changes score higher than clean repos. Sessions from 10 minutes ago score higher than sessions from 10 hours ago. The scoring uses exponential decay so the most urgent sessions float to the top.

4. **Classifies sessions as human or automated.** A gradient boosting classifier trained on ~3,800 labeled sessions separates interactive human sessions from CI pipelines, scripts, and subagent runs. It uses signals like typing pace, message casualness, typo frequency, and conversation patterns. Automated sessions are hidden by default — they clutter the recovery list and aren't what you're looking for after a crash.

5. **Presents everything in a Textual TUI** with two panes: a navigable session list on the left, and a detailed preview on the right. Sessions are grouped by time (Today, Yesterday, Last 7 Days) and sorted by interruption score within each group.

The keyboard shortcuts are designed for speed:
- **`r`** resumes directly — `exec`s into the session, replacing the current process. No clipboard, no extra steps.
- **`Space`** selects multiple sessions, then **`r`** opens all of them in iTerm tabs. This is the "I had three sessions open and they all died" recovery path.
- **`Enter`** copies the resume command to clipboard for more controlled re-entry.
- **`x`** exports a markdown context briefing — useful for handing off to another person or another AI.
- **`D`** generates a deep second-pass summary with more detail (objective, progress, decisions made, next steps).
- **`p`** analyzes your prompting patterns in the session — what worked, what didn't, anti-patterns, workflow efficiency.

### Session Bookmarks: Ground Truth vs. Guessing

By default, claude-resume infers session state from AI summaries — it's guessing whether you finished, crashed, or got stuck. Bookmarks replace guessing with ground truth.

The `/bookmark` skill runs inside any Claude Code session before you close it. In under 30 seconds, it captures:

- **Lifecycle state** — done, paused, blocked, or handing off
- **Next actions** — what you (or the next person) should do first
- **Blockers** — what's preventing progress
- **Confidence** — how stable the current state is
- **Workspace snapshot** — git branch, uncommitted files, last commit

A `Stop` hook automatically captures minimal workspace state when you close a session without explicitly bookmarking. This auto-bookmark is less rich than a manual one, but better than nothing.

Bookmarked sessions display colored lifecycle badges in the TUI (green for DONE, yellow for PAUSED, red for BLOCKED, cyan for HANDOFF). The sorting algorithm uses lifecycle-aware scoring: done sessions sort to the bottom (no urgency), blocked sessions sort high (needs attention), paused sessions get minimum scores (low urgency).

Bookmarks are stored in `~/.claude/bookmarks/` as JSON files and are also written to devlog for cross-machine durability.

### The MCP Server: Sessions as an API

The MCP server is where claude-resume becomes more than a recovery tool. It exposes session operations as tools that any Claude Code session can call:

**`search_sessions(query, limit)`** — Full-text search across all session JSONL files. Uses 16-thread parallel scanning with 1MB chunked streaming for large files (some sessions are 100MB+). Results are ranked by Reciprocal Rank Fusion (RRF) across five signals: term frequency, term density (matches per KB), recency (30-day half-life exponential decay), term balance across query words, and title match (3x boost if terms appear in the cached session title). AND logic across all query terms — every word must appear. Searches 5,000+ sessions in ~3 seconds.

**`read_session(session_id, keyword, limit)`** — Reads the actual user/assistant messages from a session. Returns head + tail messages (first few and last few) for quick context, with optional keyword filtering. This is the raw conversation — not a summary, but the actual exchanges.

**`recent_sessions(hours, limit)`** — Lists recently active sessions with cached titles. The lightweight discovery tool — what's been happening lately?

**`session_summary(session_id, force_regenerate)`** — Gets or generates an AI summary. Returns cached summaries instantly. If uncached, prefers queuing to a background daemon (non-blocking, returns in ~15s) or falls back to synchronous generation (~30s). The summary includes title, goal, what was done, current state, and key files.

**`boot_up(hours)`** — Crash recovery as an API. Finds sessions that were recently active but didn't exit cleanly. Cross-references against currently running Claude processes (via `ps aux`) to exclude active sessions. Loads bookmarks to exclude clean exits. Scores remaining sessions by urgency (exponential decay with 2-hour half-life + dirty file boost + uncommitted file count). Returns a prioritized list of sessions that probably need attention.

**`resume_in_terminal(session_id, fork)`** — Opens a terminal window with `claude --resume <id>`. Tries iTerm2 first via AppleScript, falls back to Terminal.app. On non-macOS, returns the command string for manual execution. With `fork=True`, uses Claude Code's native `--fork-session` flag to create a new session ID with the full conversation history — the original session stays untouched, like `git branch`.

**`merge_context(session_id, mode, keyword, message_limit)`** — The core cross-session operation. Pulls context from another session into the current one. Three modes:
- **`summary`**: AI-generated summary only (~1-2k tokens). Fast, compact, good for "what was that session about?"
- **`messages`**: Head + tail user/assistant messages (~1-5k tokens). Richer, preserves the actual conversation.
- **`hybrid`** (default): Summary + last few messages (~2-4k tokens). Best of both — you get the structured overview plus the recent conversation thread.

The returned context includes bookmark data (lifecycle state, next actions, uncommitted files) when available, formatted as a markdown block that Claude understands as imported session data.

**`session_timeline(session_id, limit, focus, after, before)`** — Structured timeline of milestones from a session: file creates/edits, git commits, user instructions, significant tool calls. Solves the "black box" problem for long 2,000+ message sessions. Three focus modes: `recent` (70% tail, best for "where did we leave off?"), `even` (full arc sampling), `full` (most recent first). Supports ISO timestamp filters for `after`/`before`.

**`session_thread(session_id)`** — Follows continuation links to reconstruct a multi-session thread. When sessions are continued via `merge_context` or bookmarks, traces the chain and returns all linked sessions in chronological order. Use when work spans multiple sessions.

**Data science tools** (`session_insights`, `session_xray`, `session_report`, `session_data_science`) — A second tier of tools for analyzing session history at scale. `session_insights` produces deep analytics across all sessions: temporal patterns, project breakdowns, tool usage, prompting personality, streaks and records, predictions. `session_xray` gives a single-session deep breakdown — duration, tool counts, token usage, conversation branches, edit/revert patterns. First call takes 30-60s (parses all JSONL); subsequent calls are cached.

## The Core Operations: Fork and Merge

The conceptual heart of claude-resume is two operations borrowed from version control:

### Fork: Branch a Session

Fork creates a new independent session from an existing one. The original session stays untouched. The new session has the full conversation history but a fresh session ID — it's a branch point.

This uses Claude Code's native `--fork-session` flag, which was built for exactly this purpose. claude-resume discovered and integrated it rather than building a custom implementation, because the native approach gives you the full conversation history (not just a summary) and is maintained by the Claude Code team.

Why fork matters: You're deep into a session investigating a bug. You think you've found the root cause, but you're not sure. You want to try a risky fix without polluting your investigation session. Fork it. If the fix works, great — you have a clean session with the solution. If it doesn't, the original session is untouched with all your investigation intact.

Or: you've done extensive research in one session and now need to apply those findings across three different projects. Fork the research session three times, each into the relevant project directory.

### Merge: Import Context

Merge pulls context from one session into another. Unlike fork (which creates a new session), merge enriches an existing session with knowledge from elsewhere.

This is the operation that breaks the session isolation problem. Session A researched the authentication architecture. Session B is implementing the API. Session B calls `merge_context` on session A, and now it has the architectural decisions, the constraints discovered, the approaches that were tried and rejected — all without the human re-explaining anything.

The keyword filter makes merge surgical. You don't have to import an entire 200-message session. You can say "merge context from session X, but only messages about the database schema" and get exactly the relevant context.

Why merge matters more than fork: Fork is a convenience — you can achieve the same thing by starting a new session and manually re-establishing context. Merge is fundamentally new. Before merge, there was no way for one Claude Code session to access another session's knowledge. Each session was a clean room. Merge turns sessions from isolated rooms into a connected graph where context flows between them.

## Composability

claude-resume is designed to compose with other tools rather than trying to do everything itself.

### With Claude Code's Native Features

claude-resume doesn't replicate Claude Code functionality — it extends it:

- **`--resume`** is Claude Code's native session continuation. claude-resume wraps it with discovery (finding which session to resume) and presentation (showing you what each session was doing).
- **`--fork-session`** is Claude Code's native session branching. claude-resume discovered this flag and exposes it through the MCP server, adding the ability to fork from any session in any project (not just the current directory).
- **Session JSONL files** are Claude Code's native persistence format. claude-resume reads them but never writes to them — it's a read-only consumer that adds a search/summary/navigation layer on top.

### With the Bookmark System

The `/bookmark` skill and the auto-bookmark `Stop` hook are independent components that write JSON files to `~/.claude/bookmarks/`. claude-resume reads these files to enrich its session data, but the bookmark system works standalone — other tools can read the same bookmark files.

The session resume protocol in `CLAUDE.md` reads bookmarks on startup to present a briefing. claude-resume's `boot_up` tool reads them to classify sessions. The bookmark data is a shared contract, not a private internal.

### With claude-session-commons

The session parsing, caching, scoring, and classification logic lives in a separate `claude-session-commons` package. claude-resume's `sessions.py` is an 85-line wrapper that adds resume-specific defaults. This means other tools can build on the same session infrastructure:

- A different TUI could use the same session discovery and scoring
- A CI tool could use the same classifier to identify automated sessions
- A dashboard could use the same caching layer for summaries

The commons package provides: session discovery (`find_all_sessions`, `find_recent_sessions`), JSONL parsing (`parse_session`, `quick_scan`), caching (`SessionCache`), classification (`classify_session`, `get_label`), scoring (`interruption_score`), git integration (`get_git_context`, `has_uncommitted_changes`), and TUI components (`SessionPickerPanel`, `SessionOps`).

### With the MCP Ecosystem

As an MCP server, claude-resume composes with any Claude Code session automatically. You don't import a library or configure an integration — you add the server to `~/.claude/settings.json` and every Claude Code session on the machine gains the ability to search, read, fork, and merge sessions.

This means higher-level workflows can build on claude-resume without knowing its internals:

- A `/takeoff` skill can call `boot_up()` to check for interrupted sessions before starting new work
- A `/handoff` skill can call `merge_context()` to prepare a briefing for the next person
- A planning agent can call `search_sessions()` to find prior work on a topic before proposing an implementation
- A session resume protocol can call `merge_context()` to auto-load context from a pending fork

The MCP interface is the composability surface. Each tool does one thing, returns structured data, and can be chained with other tools by Claude itself. Claude decides when to search, when to merge, when to fork — the tools just provide the capabilities.

### With the Filesystem as Integration Layer

claude-resume uses the filesystem as its integration layer, not a database or API:

- Session data: `~/.claude/projects/` (read-only, owned by Claude Code)
- Summary cache: `~/.claude/resume-summaries/` (owned by claude-resume)
- Bookmarks: `~/.claude/bookmarks/` (shared, written by bookmark skill, read by claude-resume)
- Daemon tasks: `~/.claude/daemon-tasks/` (shared queue between CLI and background daemon)
- Heartbeats: `~/.claude/bookmarks/.heartbeat-*.json` (crash detection)

Every piece of state is a JSON file in a predictable location. No database to manage, no service to keep running, no API keys to configure. Any tool that can read JSON files can integrate with claude-resume. Any tool that can write to the bookmarks directory can enrich claude-resume's data.

This is intentional. The filesystem is the lowest-common-denominator integration layer. It works across languages, across processes, across machines (via synced home directories). It's inspectable — you can `cat` any file to see what's going on. It's debuggable — you can delete a cache file and it regenerates. It's composable — you can write a shell script that reads the same files.

## The Design Philosophy

### Minimize Tokens, Maximize Signal

The MCP server is designed around a specific constraint: MCP tool responses consume context window tokens. Every byte returned is a byte that can't be used for actual work. So the server is ruthlessly compact:

- Session rows omit the resume command (Claude can construct `claude --resume <id>` itself — don't waste tokens telling it)
- Text fields are truncated to 300 characters by default
- `merge_context` has three modes so the caller can choose the right token budget
- Search results include a relevance score so Claude can decide whether to drill deeper without reading the full session

### Read-Only on Session Data

claude-resume never modifies Claude Code's JSONL files. It's a read-only layer that adds search, summaries, and navigation. This means it can never corrupt a session, never lose data, never conflict with Claude Code's own session management. The worst thing that can happen is a stale cache, which regenerates automatically.

### Inference Over Configuration

Session state (crashed, interrupted, completed) is inferred from signals rather than requiring explicit configuration. The interruption score combines recency, tool-use state at session end, git dirty status, and file counts. The classifier separates human from automated sessions using behavioral signals. Bookmarks add ground truth when available but aren't required — the system works without them, just less precisely.

This matters because the recovery scenario is inherently unconfigured. If your machine crashed, you didn't get a chance to bookmark your sessions. The tool needs to work with whatever it can find.

### Progressive Disclosure

The TUI shows a title and one-line summary by default. Arrow to a session to see the full preview. Press `D` for a deep summary. Press `p` for pattern analysis. Each level costs more (more AI calls, more time) but gives more detail. You only pay for what you need.

The MCP server follows the same pattern. `recent_sessions` returns minimal metadata. `session_summary` adds an AI summary. `read_session` shows actual messages. `merge_context` assembles a full context block. Each step gives more detail at higher cost.

## What Makes It Different

Most session management tools treat sessions as a list to pick from. claude-resume treats sessions as a graph to navigate. The combination of search, summary, fork, and merge means sessions aren't isolated events — they're connected nodes in an ongoing body of work.

The fork/merge model is borrowed from version control because the problem is structurally identical: you have a stream of changes (conversation turns instead of code commits), you need to branch (explore alternatives without losing the original), and you need to merge (bring discoveries from one branch into another). Claude Code sessions are branches of thought, and claude-resume gives you the operations to manage them as such.

The MCP server is what makes this more than a recovery tool. Recovery is the entry point — you start using it because your Mac crashed. But once it's installed, every Claude Code session on your machine can search, read, and merge other sessions. The capability is ambient. You don't have to think about it until you need it, and then it's already there.

That's the real value: not recovering from crashes (though it does that), but making the accumulated knowledge across all your Claude Code sessions accessible to every future session. Your past work becomes a resource that compounds rather than a pile of files that rots.
