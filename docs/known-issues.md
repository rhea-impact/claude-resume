# Known Issues — resume-resume

Structured catalogue of product issues, kept in source control so the
telemetry pyramid (A1, A2) and humans can read from a single source of
truth instead of rediscovering the same problems every run.

**Updating this file:**
- Add a new entry any time you observe a real issue you can't immediately fix.
- When you fix one, move it to the commit message and delete the entry
  (or strike it through if you want a visible record).
- Each entry should have a perf-regression test where possible; link it
  in the `Test` field.
- A1 and A2 are instructed to consult this file before drafting
  recommendations — don't re-propose things that are already here.

**Severity:**
- ★★★ = blocks real work, user-visible
- ★★  = noticeable friction, slows down workflow
- ★   = paper cut / polish

---

## Performance

### perf-001: `dirty_repos` cold-path scan is slow

- **Severity:** ★★
- **Evidence:** 3071ms p95 in A1's first telemetry observation. After
  29d2f73 optimizations (combined status+branch, skip git-log for clean
  repos), measured ~2000ms on the same machine state (30 repos: 21 clean,
  9 dirty).
- **Why it sucks:** `boot_up` calls `dirty_repos`; user waits ~2 seconds
  on every session resume. Not fatal, but visible friction every session.
- **Workaround:** 30-second result cache (29d2f73) makes subsequent calls
  in the same session ~2ms.
- **Further improvements shipped (this commit):**
  1. ~~Skip project dirs not touched in 30+ days~~ DONE — 63 of 90
     repos skipped on measured run. Remaining 27 scanned.
  2. ~~Increase `ThreadPoolExecutor` workers above 8~~ DONE — bumped to 16.
  3. Stale-while-revalidate: return cached result instantly and refresh
     in background. NOT YET — future work.
- **Test:** `tests/test_perf_regression.py::test_dirty_repos_cold_under_generous_ceiling`
  and `test_dirty_repos_cached_is_dramatically_faster_than_cold`

### perf-002: `recent_sessions` is slow

- **Severity:** ★★
- **Evidence:** 1132ms in A1's first telemetry observation; ~1200ms when
  remeasured in the same session. Scans every session JSONL to find ones
  within the time window.
- **Why it sucks:** A common tool — users check "what was I doing" often.
  Over 1 second of latency for a listing is sluggish.
- **Root cause (unverified):** `find_recent_sessions` in
  `resume_resume.sessions` likely walks every session file. No index.
- **Workaround:** none shipped.
- **Proposed fix:** similar shape to `dirty_repos` — cache recent listing
  for N seconds and/or maintain a mtime-sorted index of session files.
- **Test:** `tests/test_perf_regression.py::test_recent_sessions_under_generous_ceiling`
  (ceiling is generous — 3000ms — because no fix has shipped yet. Tighten
  when a fix lands.)

### perf-003: `self_insights` has no cache

- **Severity:** ★
- **Evidence:** Currently ~5ms at low telemetry volume. No measured
  regression yet, but the function re-reads all JSONL files on every call.
- **Why it sucks:** As telemetry grows (weeks or months of data), every
  call to `self_insights` scans N daily JSONL files. A1, A2, and the human
  may each call it multiple times per run. Will degrade silently.
- **Proposed fix:** in-memory cache keyed by `(days, mtime of latest jsonl)`,
  invalidated when a new file appears. Or a daily index.
- **Test:** `tests/test_perf_regression.py::test_self_insights_fast` (ceiling
  500ms, currently passing with large margin.)

---

## Correctness

### correctness-001: `_apply_proposal` couldn't handle JSON-string diffs (FIXED 29d2f73)

- Fixed. Retained here as a documented failure-mode-to-test.
- **Test:** `tests/test_meta_ai.py::test_decide_approve_prompt_edit_json_string_form`
- **Why it mattered:** MCP transport serialized a dict `diff` parameter
  into a JSON-encoded string. The apply branches expected either a dict
  or a bare markdown string — neither branch matched a string starting
  with `{"full_new_text": ...`. Approval would have silently failed with
  `apply_error` set. Fixed by `_coerce_diff` which best-effort
  json-decodes strings that look like JSON before dispatching.

### correctness-002: `self_*` list tools used to return bare lists (FIXED 29d2f73)

- Fixed. `fastmcp` serializes bare list returns as `{"result": [...]}`,
  which is inconsistent with tools returning dicts directly. Nine
  self_* tools now return `{"items": [...], "count": N}`.
- **Test:** `tests/test_perf_regression.py::test_self_list_tools_return_wrapped_dict`
- **Why it mattered:** Non-uniform response shape across the MCP surface
  made clients (including me, as an A1/A2 skill) handle each tool
  specially. Wrapping is uniform now.

---

## Observability

### obs-001: Telemetry thresholds are miscalibrated below ~100 total calls

- **Severity:** ★★
- **Evidence:** `dead_tools = max(1, total_calls // 500)` → at any volume
  below 500, the threshold is 1, so every tool with only 1 call gets
  flagged as "dead." A1 observed this in its first run; A2 proposed a
  prompt guardrail (pending inbox).
- **Why it sucks:** `self_insights` produces noise that looks like signal
  at low volume. A1 had to learn (and A2 had to codify) that the
  `dead_tools` list is useless at this scale.
- **Proposed fix:** either (a) raise the divisor so threshold floors at 0
  until volume is meaningful, or (b) rewrite as a percentile-based metric.
  Leaving to A2 for now; see pending proposal `f41baad32ae3`.
- **Test:** none yet — behavior is threshold-driven and would need a
  fixture with controlled telemetry to assert sensibly.

### obs-002: Telemetry JSONL files grow unbounded

- **Severity:** ★
- **Evidence:** `telemetry.py` appends one line per MCP call to
  `~/.resume-resume/telemetry/<user>/YYYY-MM-DD.jsonl`. Daily rotation
  exists (new file per day) but no retention cap, no gzip for old files.
- **Why it sucks:** At heavy use (say 1000 calls/day × 365 days) that's
  ~100MB raw. Gzipped, 1-5MB/year — fine. Raw is annoying over time.
  Deferred in the original plan.
- **Proposed fix:** rotate .jsonl files older than 7 days to .jsonl.gz;
  env var `RESUME_RESUME_TELEMETRY_RETENTION_DAYS` to trim beyond N days.
  See TASK-0017 / TASK-0018 in ike.
- **Test:** none. Infrastructure concern, not behavior.

---

## Infrastructure / Dev ergonomics

### dx-001: No integration tests for most `self_*` MCP tools

- **Severity:** ★
- **Evidence:** Existing tests hit the underlying Python functions in
  `meta_ai.py` and `telemetry_query.py` directly. `test_perf_regression.py`
  adds some end-to-end coverage via `fastmcp.Client`, but most tools are
  only tested at the unit layer.
- **Why it sucks:** Tool schema regressions, parameter-type changes, and
  MCP serialization bugs can slip through unit tests. The
  `correctness-001` and `correctness-002` bugs both landed because of
  MCP-layer behavior (transport serialization, response wrapping) that
  unit tests couldn't see.
- **Proposed fix:** a small `tests/test_mcp_surface.py` that round-trips
  every MCP tool through `fastmcp.Client` with canonical args, asserting
  the response shape matches the tool's type annotation.
- **Test:** none yet.

### dx-002: `pyright` can't resolve some internal imports

- **Severity:** ★
- **Evidence:** Diagnostics on `claude_session_commons` and various
  `mcp_server.py` internal imports show as unresolved. The tests pass, the
  MCP server loads, but the IDE complains.
- **Why it sucks:** Noise in the IDE. Makes real diagnostics harder to
  spot. Masks real issues.
- **Proposed fix:** add a `pyrightconfig.json` with correct `venvPath` /
  `venv` / `extraPaths` so the dev env is recognized. Not a product bug.

---

## Meta / process

### process-001: A1 surface area is small; meta-layer value won't show until volume grows

- **Severity:** N/A (a known tradeoff, not a bug)
- **Evidence:** First A1 run filed 1 recommendation for a problem that
  was already visible in raw telemetry. First A2 run filed 1 proposal
  codifying a rule A1 had already followed correctly. At current volume
  the pyramid produces meta-activity more than product progress.
- **Why it's documented:** research ADR-002 anticipated this, but the
  observation is worth keeping visible so future work isn't mistaken for
  pyramid-failure when it's actually pyramid-at-low-volume. Re-evaluate
  when total_calls > 500.
- **Reference:** `.research/DECISION.md`, `.visionlog/adr/ADR-002-*.md`
