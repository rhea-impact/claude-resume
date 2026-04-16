---
id: TASK-0021
title: 'dx-001: MCP-layer integration test suite'
status: In Progress
created: '2026-04-15'
priority: medium
tags:
  - testing
  - known-issues
updated: '2026-04-16'
---
Most self_* tools only have unit tests on underlying functions. Need tests/test_mcp_surface.py that round-trips every tool through fastmcp.Client with canonical args. Correctness-001 and -002 bugs both slipped through unit tests because they were MCP-layer serialization issues. See docs/known-issues.md dx-001.
