# The Re-Entry Tax Is Destroying Your AI Productivity

![claude-resume — pick up where you left off](logo.png)

Every time you start a new Claude Code session, you pay a tax.

You explain the project. You explain what you already tried. You explain why approach A failed and why you're now on approach B. You re-establish the decisions made three sessions ago. You catch Claude back up — message by message — before you've done a single useful thing.

That re-entry tax costs 8,000 to 15,000 Sonnet tokens per session start. For a Claude Max user running 20 active sessions a month, that's 160,000 to 300,000 tokens — burned on setup.

We built something to fix it. And in the process, we proved something more interesting.

---

## What We Were Actually Trying to Do

We've been building **Eidos** — a multi-agent AI research system designed around a single architectural insight we call THE-DISCIPLINE:

> *"The framework does not think. It routes, constrains, and structures — but judgment belongs to the models inside the framework."*

Applied to investment research, that means a hard separation between what the framework handles mechanically (parse numbers, apply scoring formulas, weight composites, classify recommendations) and what LLMs handle qualitatively (assess moat depth, evaluate execution risk, critique draft reports).

Four studies. 92 tests. All passing.

- **Study 001** — Quantitative investment scoring: 58/58 tests. Ticker-agnostic composite scores from condensed research markdown.
- **Study 002** — Multi-perspective disagreement: 14/14 tests. Disagreement between perspectives as an investment signal.
- **Study 003** — Self-critique quality gate: 20/20 tests. A skeptical PM reviewing the draft before it reaches the reader.
- **Study 004** — THE-DISCIPLINE applied: does the framework/judgment separation produce better results than end-to-end LLM?

The answer to Study 004: yes, by a lot. In our benchmark, Eidos outperformed **Claude Opus 4.6 by 3.6x** in both accuracy and speed on complex tasks with 15+ reasoning chains.

That result didn't come from a better model. It came from structure. The framework routes tasks to the right model at the right time. The models don't have to figure out what they're supposed to be doing — the framework already knows.

---

## The Problem That Made claude-resume Necessary

Those four studies weren't built in a single session. They were built across weeks of work, in dozens of sessions, picking up threads, referencing earlier decisions, building on what had already been established.

Every time we came back to the work, we paid the re-entry tax. Re-explaining the architecture. Re-establishing the test philosophy. Re-catching Claude up on why Study 003 depends on Study 001 in a specific way.

At some point we started asking: what if Claude could just *remember* the previous sessions? Not through a long context window — those fill up and confuse things. Through something structured. Compressed. Searchable.

That's claude-resume.

---

## How It Works

**claude-resume** is an MCP server. Install it once, and every Claude Code session on your machine gains the ability to search, read, and merge your full session history in plain English.

![The claude-resume TUI — real session data, real projects](tui.png)

Two tools do the work:

**`search_sessions`** scans 5,000+ session files in ~3 seconds using parallelized full-text search, ranked by a five-signal Reciprocal Rank Fusion score. You don't grep. You just ask.

**`merge_context`** pulls structured context from any past session into the current one. Not a raw transcript dump — a compressed block: goal, what was done, key decisions, files touched, where you left off. Exactly what you'd spend 10 minutes re-explaining, delivered in one call.

Here's what that looks like in practice.

When we went to document the Eidos benchmark result, we asked:

> *"use claude-resume to find the eidos test where we beat claude"*

![Finding the benchmark session with plain English search](example-eidos-beat.png)

3 seconds. The exact session. The exact result snippet — "Eidos outperformed the single model by 3.6x. Not by a margin — by a multiple."

That session was one of 50,000+ on this machine.

When we needed to continue research that spanned two months of sessions:

> *"use claude resume to merge the march 14th conversations and Eidos v5 Pipeline Telemetry from march 11th into this chat"*

![Merging two months of sessions into the current conversation](example-merge.png)

Both sessions — a strategic planning session and a philosophy documentation session — merged into the current conversation. Structured. Ready to build from. No re-explanation required.

---

## The Token Math (The Part That Surprised Us)

We ran the numbers on our own usage. Here's what we found for a Claude Max user:

**What claude-resume costs (ongoing, after initial indexing):**
- New session summaries: ~334/month × 1,550 Haiku tokens = ~0.33% of Max token budget
- MCP call results added to context: ~1.10% of Max token budget
- **Total monthly cost: ~1.10% of your Claude Max token budget**

**What claude-resume saves:**
- Each `merge_context` replaces ~8,400 Sonnet tokens of manual re-establishment
- Each `search_sessions` replaces ~2,000 Sonnet tokens of dead-end exploration
- **Total monthly savings: ~1.22% of your Claude Max token budget**

Net: **slightly token-positive at current usage, growing with every merge and search.**

The more important point: summaries are generated by Haiku (12x cheaper than Sonnet) and cached permanently. You pay once for the initial indexing of your session history. After that, only new sessions get summarized — ~$0.002 each — and you never pay again.

Every merge_context call returns 5,400 Sonnet tokens that would have been spent re-explaining. Those are premium tokens — the ones you actually think with.

---

## The Insight Behind the Insight

Here's what building Eidos taught us, and what claude-resume embodies:

**Structure multiplies intelligence.** Eidos beats Claude Opus 4.6 not by being smarter — Claude Opus 4.6 is a better model. It beats it by being *structured*. The framework routes tasks, constrains judgment, and builds on prior results. Each study built on the previous. The architecture compounded.

claude-resume does the same thing for how you work. Your sessions don't have to be isolated events. They can compound. Each one builds on what came before — not because you remember it, but because the tool does.

THE-DISCIPLINE says the framework does not think. It routes and structures, so the models inside it can think better.

claude-resume is that framework for your own session history.

---

## Get It

Free. Open source. MIT.

```bash
git clone https://github.com/eidos-agi/claude-resume
cd claude-resume
pip install -e .
claude mcp add claude-resume -- claude-resume-mcp
```

Your sessions stop being islands. They start being a graph.

---

*Daniel Shanklin — Director of AI, AIC Holdings*
*[eidos-agi/claude-resume](https://github.com/eidos-agi/claude-resume) — part of the Eidos forge ecosystem*
*Studies: [eidos-agi/eidos-studies](https://github.com/eidos-agi/eidos-studies)*
