# claude-resume — Feature Roadmap

## Implemented
- [x] AI-powered session summaries (quick + deep dive)
- [x] Full-text search across entire session JSONL files
- [x] Date-grouped list (Today, Yesterday, Last 7 Days, etc.)
- [x] Preview pane with arrow-key scrolling
- [x] Instant TUI launch with background summarization
- [x] `--dangerously-skip-permissions` toggle (d key)
- [x] Patterns analysis (p key) — prompting, workflow, anti-patterns
- [x] Git context (recent commits + uncommitted changes) in summaries
- [x] Caching layer (summaries, deep dives, patterns, search text)
- [x] `--cache-all` batch indexing mode
- [x] **Architectural rebuild v2:**
  - [x] `SessionCache` class — unified cache replacing ~10 separate get/save functions (2 core methods: get/set)
  - [x] `SessionOps` dataclass — replaces 7 separate callback parameters with one object
  - [x] Generic background task engine — single `_run_task` worker + `TaskDone` message replaces 5 copy-paste workers
  - [x] `PreviewMode` enum — explicit state machine for preview pane (SUMMARY vs PATTERNS)
  - [x] Single-pass JSONL parsing — `parse_session()` combines old `extract_context` + `extract_searchable_text`
  - [x] Error visibility — `_show_preview_error()` surfaces background task failures in the UI
  - [x] Slim entry point — `claude-resume` is just a launcher, all logic lives in the package
  - [x] All operations non-blocking — patterns, search indexing, deep dives all run as background tasks
  - [x] `--all` mode shows up to 50 sessions (was capped at 10)

## High Priority

### r — Resume Directly
Skip the clipboard. `exec` straight into `cd <dir> && claude --resume <id>` from the TUI. One keypress and you're back in your session. Optional: if `--dangerously-skip-permissions` is toggled on, include it.

### x — Export Context Briefing
Dump a session's deep summary + patterns + key files into a markdown file that can be dropped into a project as `CLAUDE.md` context or pasted into a new session. "Here's where I left off, here's what matters."

### Multi-Select Resume
Space to select multiple sessions. Enter opens them all — each in a new iTerm tab/pane via `osascript`. Get your whole workspace back in one shot after a crash.

### Auto-Resume on Crash Recovery
Detect macOS crash recovery (or iTerm relaunch) and auto-launch claude-resume. Could be a LaunchAgent plist or an iTerm trigger script that checks if sessions were recently active but no Claude processes are running.

### s — Session Stats Dashboard
Aggregate view across all cached sessions:
- Total hours spent in Claude Code
- Most active projects (ranked by session count + total time)
- Average session length
- Most-used tools across all sessions
- Prompting score trend (from patterns analysis)
- Sessions per day heatmap (like GitHub contributions)

## Medium Priority

### c — Compare Sessions
Select two sessions on the same project and see a side-by-side diff:
- How did the approach evolve?
- What was tried first vs what worked?
- Decision changes between sessions

### t — Tag / Bookmark Sessions
Mark sessions as "important", "reference", "failed experiment", etc. Tags persist in cache and are searchable. Useful for finding that one session where you figured out the auth flow.

### Session Groups by Project
Auto-group sessions by project directory. Show a tree: project → sessions. Quick way to see "what have I done in this repo?"

### Prompt Library
Extract the most effective prompts (from patterns analysis) across all sessions into a personal prompt library. Searchable collection of "prompts that worked" organized by task type.

### Session Timeline
Visual timeline of a single session: when did the user send messages, when did tools run, where were the long gaps (thinking/testing), where were the rapid-fire exchanges (flow state).

### Smart Resume Suggestions
On launch, highlight the session most likely to need resuming: crashed mid-tool-use, had uncommitted changes, was actively being worked on. Don't just sort by time — sort by "how interrupted does this look?"

## Lower Priority / Exploratory

### Session Replay
Step through a session message-by-message in the TUI. See the conversation unfold. Useful for learning from past sessions or showing others your workflow.

### Cross-Session Pattern Aggregation
Aggregate patterns analysis across all sessions to build a personal "Claude Code skill profile":
- Your most common anti-patterns (things to work on)
- Your strongest prompting patterns (things to keep doing)
- Tool usage evolution over time (are you getting more efficient?)
- Project-specific insights (you prompt differently in Python vs TypeScript)

### Session Sharing
Export a session summary (sanitized — no secrets) as a shareable link or markdown. "Here's what I built today" for standups, PRs, or team knowledge sharing.

### Cost Estimation
Estimate token usage per session based on message lengths and tool calls. Track spend over time. Flag sessions that burned tokens (lots of retries, huge context).

### CLAUDE.md Generator
Analyze a project's sessions and auto-generate a CLAUDE.md with:
- Common patterns for this codebase
- Key architectural decisions made across sessions
- Files that get touched most often
- Gotchas learned the hard way

### Session Health Score
Rate each session 1-10 based on:
- Ratio of productive tool calls vs retries
- How specific the prompts were
- Whether it reached its objective
- How much thrashing happened
Show the score in the list view. Track your average over time.

### iTerm Integration
- Custom iTerm status bar component showing active Claude session info
- iTerm trigger that auto-saves session metadata on tab close
- Keyboard shortcut to invoke claude-resume from any iTerm tab

### Watch Mode
`claude-resume --watch` — live-updating dashboard that shows all active Claude sessions across all terminal tabs. See what each one is doing in real time.

### Session Merge
When you crash and resume, you now have two sessions for the same task. Merge their summaries and patterns into one logical unit so the history makes sense.

### AI Coach Mode
After analyzing patterns, offer a brief "coaching session":
- "You tend to give vague instructions when frustrated. Try: [specific rewrite]"
- "Your most productive sessions start with reading files first. Keep doing that."
- "You've retried this same approach 3 times across sessions. Consider a different strategy."
