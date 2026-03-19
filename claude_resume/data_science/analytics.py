"""Analytics engine — transforms raw scan data into insights, stats, and predictions.

Design: all functions take a list of deep-scanned session dicts and return
structured results. No I/O, no side effects, pure computation.
"""

import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timedelta


def _median(values: list) -> float:
    if not values:
        return 0
    return statistics.median(values)


def _mean(values: list) -> float:
    if not values:
        return 0
    return statistics.mean(values)


def _format_duration(secs: float) -> str:
    if secs < 60:
        return f"{secs:.0f}s"
    if secs < 3600:
        return f"{secs / 60:.0f}m"
    h = int(secs // 3600)
    m = int((secs % 3600) // 60)
    return f"{h}h {m}m"


def _format_tokens(n: int) -> str:
    if n < 1000:
        return str(n)
    if n < 1_000_000:
        return f"{n / 1000:.1f}K"
    if n < 1_000_000_000:
        return f"{n / 1_000_000:.1f}M"
    return f"{n / 1_000_000_000:.2f}B"


def _format_bytes(n: int) -> str:
    if n < 1024:
        return f"{n}B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f}KB"
    if n < 1024 * 1024 * 1024:
        return f"{n / (1024 * 1024):.1f}MB"
    return f"{n / (1024 * 1024 * 1024):.2f}GB"


def overview(sessions: list[dict]) -> dict:
    """High-level stats that make you go 'whoa'."""
    total = len(sessions)
    if total == 0:
        return {"error": "No sessions to analyze"}

    total_tokens = sum(s.get("total_tokens", 0) for s in sessions)
    output_tokens = sum(s.get("output_tokens", 0) for s in sessions)
    input_tokens = sum(s.get("input_tokens", 0) for s in sessions)
    total_size = sum(s.get("size", 0) for s in sessions)
    total_tool_uses = sum(s.get("tool_use_total", 0) for s in sessions)
    durations = [s["duration_mins"] for s in sessions if s.get("duration_mins", 0) > 0]
    total_msgs = sum(s.get("total_msgs", 0) for s in sessions)
    user_msgs = sum(s.get("user_msgs", 0) for s in sessions)
    total_hours = sum(d for d in durations) / 60

    # Unique days with sessions
    active_days = len(set(s["date"] for s in sessions))

    # Date range
    dates = sorted(s["date"] for s in sessions)
    first_date = dates[0] if dates else "?"
    last_date = dates[-1] if dates else "?"

    # Projects
    projects = set(s.get("project_short", "") for s in sessions)

    # Words approximation: ~0.75 tokens per word for English
    approx_words_output = int(output_tokens * 0.75)
    # Average novel is ~80,000 words
    novels_equivalent = approx_words_output / 80_000

    # Token cost estimate (rough: $15/M input, $75/M output for Opus)
    cost_estimate = (input_tokens / 1_000_000 * 15) + (output_tokens / 1_000_000 * 75)

    return {
        "total_sessions": total,
        "active_days": active_days,
        "sessions_per_day": round(total / max(active_days, 1), 1),
        "date_range": f"{first_date} to {last_date}",
        "total_projects": len(projects),
        "total_hours": round(total_hours, 1),
        "avg_session_mins": round(_mean(durations), 1),
        "median_session_mins": round(_median(durations), 1),
        "longest_session_mins": round(max(durations) if durations else 0, 1),
        "total_messages": total_msgs,
        "user_messages": user_msgs,
        "total_tokens": _format_tokens(total_tokens),
        "total_tokens_raw": total_tokens,
        "output_tokens": _format_tokens(output_tokens),
        "input_tokens": _format_tokens(input_tokens),
        "total_tool_uses": total_tool_uses,
        "data_size": _format_bytes(total_size),
        "words_generated": f"{approx_words_output:,}",
        "novels_equivalent": round(novels_equivalent, 1),
        "estimated_cost_usd": round(cost_estimate, 2),
    }


def temporal_patterns(sessions: list[dict]) -> dict:
    """When do you work? Hour-by-hour, day-by-day patterns."""
    if not sessions:
        return {}

    hour_counts = Counter()
    weekday_counts = Counter()
    hour_tokens = defaultdict(int)
    hour_durations = defaultdict(list)
    weekday_tokens = defaultdict(int)

    for s in sessions:
        h = s.get("hour", 0)
        wd = s.get("weekday", "Monday")
        hour_counts[h] += 1
        weekday_counts[wd] += 1
        hour_tokens[h] += s.get("total_tokens", 0)
        if s.get("duration_mins", 0) > 0:
            hour_durations[h].append(s["duration_mins"])
        weekday_tokens[wd] += s.get("total_tokens", 0)

    # Find peak hour
    peak_hour = hour_counts.most_common(1)[0] if hour_counts else (0, 0)
    dead_hour = min(hour_counts, key=hour_counts.get) if hour_counts else 0

    # Night owl score: % of sessions between 10 PM and 4 AM
    night_sessions = sum(hour_counts.get(h, 0) for h in [22, 23, 0, 1, 2, 3])
    morning_sessions = sum(hour_counts.get(h, 0) for h in [6, 7, 8, 9, 10, 11])
    total_sessions = sum(hour_counts.values()) or 1
    night_pct = round(night_sessions / total_sessions * 100, 1)
    morning_pct = round(morning_sessions / total_sessions * 100, 1)

    if night_pct > 30:
        chronotype = "Night Owl"
    elif morning_pct > 40:
        chronotype = "Early Bird"
    else:
        chronotype = "Flexible"

    # Most productive hour (by tokens)
    productive_hour = max(hour_tokens, key=hour_tokens.get) if hour_tokens else 0

    # Weekend warrior score
    weekend_sessions = weekday_counts.get("Saturday", 0) + weekday_counts.get("Sunday", 0)
    weekend_pct = round(weekend_sessions / total_sessions * 100, 1)

    # Build hourly heatmap (0-23)
    max_count = max(hour_counts.values()) if hour_counts else 1
    heatmap = {}
    for h in range(24):
        c = hour_counts.get(h, 0)
        intensity = round(c / max_count, 2) if max_count else 0
        label = f"{h:02d}:00"
        heatmap[label] = {
            "sessions": c,
            "intensity": intensity,
            "avg_duration_mins": round(_mean(hour_durations.get(h, [])), 1),
        }

    # Weekly pattern
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekly = {d: weekday_counts.get(d, 0) for d in day_order}
    busiest_day = max(weekly, key=weekly.get) if weekly else "Monday"

    return {
        "chronotype": chronotype,
        "peak_hour": f"{peak_hour[0]:02d}:00 ({peak_hour[1]} sessions)",
        "dead_hour": f"{dead_hour:02d}:00",
        "most_productive_hour": f"{productive_hour:02d}:00 (by token volume)",
        "night_pct": night_pct,
        "morning_pct": morning_pct,
        "weekend_pct": weekend_pct,
        "busiest_day": busiest_day,
        "hourly_heatmap": heatmap,
        "weekly_pattern": weekly,
    }


def project_insights(sessions: list[dict]) -> dict:
    """Project-level analytics: attention, lifecycle, context switching."""
    if not sessions:
        return {}

    project_stats = defaultdict(lambda: {
        "sessions": 0, "tokens": 0, "hours": 0,
        "first_seen": None, "last_seen": None, "dates": set(),
    })

    for s in sessions:
        repo = s.get("repo", "unknown")
        ps = project_stats[repo]
        ps["sessions"] += 1
        ps["tokens"] += s.get("total_tokens", 0)
        ps["hours"] += s.get("duration_mins", 0) / 60
        date = s.get("date", "")
        if date:
            ps["dates"].add(date)
            if ps["first_seen"] is None or date < ps["first_seen"]:
                ps["first_seen"] = date
            if ps["last_seen"] is None or date > ps["last_seen"]:
                ps["last_seen"] = date

    # Sort by sessions desc
    sorted_projects = sorted(project_stats.items(), key=lambda x: x[1]["sessions"], reverse=True)

    # Pareto analysis: how many projects account for 80% of sessions
    total_sessions = sum(p[1]["sessions"] for p in sorted_projects)
    cumulative = 0
    pareto_count = 0
    for _, stats in sorted_projects:
        cumulative += stats["sessions"]
        pareto_count += 1
        if cumulative >= total_sessions * 0.8:
            break

    # Context switching: average unique projects per active day
    day_projects = defaultdict(set)
    for s in sessions:
        day_projects[s.get("date", "")].add(s.get("repo", ""))
    switches_per_day = [len(projs) for projs in day_projects.values()]
    avg_context_switches = round(_mean(switches_per_day), 1)

    # Abandoned projects (no activity in 30+ days)
    today = datetime.now().strftime("%Y-%m-%d")
    today_dt = datetime.strptime(today, "%Y-%m-%d")
    abandoned = []
    active = []
    for name, stats in sorted_projects:
        if stats["last_seen"]:
            last_dt = datetime.strptime(stats["last_seen"], "%Y-%m-%d")
            days_inactive = (today_dt - last_dt).days
            if days_inactive > 30 and stats["sessions"] >= 3:
                abandoned.append({
                    "project": name,
                    "sessions": stats["sessions"],
                    "last_seen": stats["last_seen"],
                    "days_inactive": days_inactive,
                })
            elif days_inactive <= 7:
                active.append(name)

    # Top projects with clean stats
    top_projects = []
    for name, stats in sorted_projects[:15]:
        top_projects.append({
            "project": name,
            "sessions": stats["sessions"],
            "tokens": _format_tokens(stats["tokens"]),
            "hours": round(stats["hours"], 1),
            "active_days": len(stats["dates"]),
            "first_seen": stats["first_seen"],
            "last_seen": stats["last_seen"],
        })

    return {
        "total_projects": len(project_stats),
        "pareto": f"{pareto_count} projects account for 80% of sessions ({pareto_count}/{len(project_stats)})",
        "avg_context_switches_per_day": avg_context_switches,
        "currently_active": active[:10],
        "abandoned_count": len(abandoned),
        "abandoned": abandoned[:10],
        "top_projects": top_projects,
    }


def tool_usage(sessions: list[dict]) -> dict:
    """What tools do you use and how?"""
    if not sessions:
        return {}

    all_tools = Counter()
    for s in sessions:
        tools = s.get("tool_uses", {})
        if isinstance(tools, dict):
            for name, count in tools.items():
                all_tools[name] += count

    total_uses = sum(all_tools.values())

    # Categorize tools
    categories = {
        "reading": ["Read", "Glob", "Grep", "LS"],
        "writing": ["Edit", "Write", "NotebookEdit"],
        "execution": ["Bash", "BashOutput"],
        "delegation": ["Agent"],
        "search": ["WebSearch", "WebFetch"],
        "communication": ["AskUserQuestion"],
    }

    category_counts = {}
    for cat, tool_names in categories.items():
        count = sum(all_tools.get(t, 0) for t in tool_names)
        category_counts[cat] = count

    # Work style: reader vs writer vs executor
    read_ratio = category_counts.get("reading", 0) / max(total_uses, 1)
    write_ratio = category_counts.get("writing", 0) / max(total_uses, 1)
    exec_ratio = category_counts.get("execution", 0) / max(total_uses, 1)

    if read_ratio > 0.5:
        work_style = "Researcher — you read more than you write"
    elif write_ratio > 0.3:
        work_style = "Builder — heavy on code creation"
    elif exec_ratio > 0.3:
        work_style = "Operator — command-line warrior"
    else:
        work_style = "Balanced — mix of reading, writing, and executing"

    # MCP tool usage (external integrations)
    mcp_tools = {k: v for k, v in all_tools.items() if k.startswith("mcp__")}
    mcp_total = sum(mcp_tools.values())

    top_tools = [{"tool": name, "uses": count, "pct": round(count / max(total_uses, 1) * 100, 1)}
                 for name, count in all_tools.most_common(20)]

    top_mcp = [{"tool": name.replace("mcp__", ""), "uses": count}
               for name, count in sorted(mcp_tools.items(), key=lambda x: x[1], reverse=True)[:10]]

    return {
        "total_tool_uses": total_uses,
        "unique_tools": len(all_tools),
        "work_style": work_style,
        "category_breakdown": category_counts,
        "read_write_ratio": f"{round(read_ratio * 100)}% read / {round(write_ratio * 100)}% write / {round(exec_ratio * 100)}% execute",
        "top_tools": top_tools,
        "mcp_integrations_used": len(mcp_tools),
        "mcp_total_calls": mcp_total,
        "top_mcp_tools": top_mcp,
    }


def model_usage(sessions: list[dict]) -> dict:
    """Which models do you use?"""
    if not sessions:
        return {}

    all_models = Counter()
    for s in sessions:
        models = s.get("models_used", {})
        if isinstance(models, dict):
            for name, count in models.items():
                all_models[name] += count

    return {
        "models": [{"model": name, "responses": count}
                   for name, count in all_models.most_common(10)],
    }


def streaks_and_records(sessions: list[dict]) -> dict:
    """Find impressive streaks, records, and outliers."""
    if not sessions:
        return {}

    # Sessions per day
    day_counts = Counter(s["date"] for s in sessions)
    busiest_day = day_counts.most_common(1)[0] if day_counts else ("?", 0)

    # Consecutive day streaks
    all_dates = sorted(set(s["date"] for s in sessions))
    if all_dates:
        date_objects = [datetime.strptime(d, "%Y-%m-%d") for d in all_dates]
        max_streak = 1
        current_streak = 1
        streak_end = date_objects[0]
        best_streak_end = date_objects[0]

        for i in range(1, len(date_objects)):
            if (date_objects[i] - date_objects[i - 1]).days == 1:
                current_streak += 1
                if current_streak > max_streak:
                    max_streak = current_streak
                    best_streak_end = date_objects[i]
            else:
                current_streak = 1

        streak_start = best_streak_end - timedelta(days=max_streak - 1)
    else:
        max_streak = 0
        streak_start = None
        best_streak_end = None

    # Longest session
    longest = max(sessions, key=lambda s: s.get("duration_mins", 0))
    # Most messages in a session
    most_msgs = max(sessions, key=lambda s: s.get("total_msgs", 0))
    # Most tokens in a session
    most_tokens = max(sessions, key=lambda s: s.get("total_tokens", 0))
    # Most tool uses in a session
    most_tools = max(sessions, key=lambda s: s.get("tool_use_total", 0))

    # Late night records (sessions starting after midnight)
    late_sessions = [s for s in sessions if s.get("hour", 12) in [0, 1, 2, 3, 4]]
    latest_session = max(late_sessions, key=lambda s: s.get("duration_mins", 0)) if late_sessions else None

    return {
        "busiest_day": {
            "date": busiest_day[0],
            "sessions": busiest_day[1],
        },
        "longest_streak_days": max_streak,
        "streak_period": f"{streak_start.strftime('%Y-%m-%d') if streak_start else '?'} to {best_streak_end.strftime('%Y-%m-%d') if best_streak_end else '?'}",
        "longest_session": {
            "duration": _format_duration(longest.get("duration_mins", 0) * 60),
            "project": longest.get("repo", "?"),
            "date": longest.get("date", "?"),
        },
        "most_messages_session": {
            "messages": most_msgs.get("total_msgs", 0),
            "project": most_msgs.get("repo", "?"),
            "date": most_msgs.get("date", "?"),
        },
        "most_tokens_session": {
            "tokens": _format_tokens(most_tokens.get("total_tokens", 0)),
            "project": most_tokens.get("repo", "?"),
            "date": most_tokens.get("date", "?"),
        },
        "most_tool_uses_session": {
            "tool_uses": most_tools.get("tool_use_total", 0),
            "project": most_tools.get("repo", "?"),
            "date": most_tools.get("date", "?"),
        },
        "late_night_sessions": len(late_sessions),
        "latest_marathon": {
            "duration": _format_duration((latest_session.get("duration_mins", 0) * 60)) if latest_session else "N/A",
            "project": latest_session.get("repo", "?") if latest_session else "N/A",
            "date": latest_session.get("date", "?") if latest_session else "N/A",
            "start_hour": f"{latest_session.get('hour', 0):02d}:00" if latest_session else "N/A",
        },
    }


def predictions(sessions: list[dict]) -> dict:
    """Predict what you'll do next based on patterns."""
    if not sessions:
        return {}

    today = datetime.now()
    today_str = today.strftime("%Y-%m-%d")
    today_weekday = today.strftime("%A")
    current_hour = today.hour

    # What project will you work on next?
    # Signal: recent frequency + day-of-week affinity + hour affinity
    project_signals = defaultdict(float)

    for s in sessions:
        repo = s.get("repo", "unknown")
        date = s.get("date", "")
        if not date:
            continue

        days_ago = (today - datetime.strptime(date, "%Y-%m-%d")).days

        # Recency (exponential decay, 7-day half-life)
        recency = math.exp(-0.099 * days_ago)

        # Day-of-week match
        dow_match = 1.5 if s.get("weekday") == today_weekday else 1.0

        # Hour proximity (gaussian around current hour)
        hour_diff = min(abs(s.get("hour", 12) - current_hour),
                       24 - abs(s.get("hour", 12) - current_hour))
        hour_match = math.exp(-0.5 * (hour_diff / 3) ** 2)

        project_signals[repo] += recency * dow_match * hour_match

    # Normalize and sort
    max_signal = max(project_signals.values()) if project_signals else 1
    predicted_projects = sorted(
        [(repo, round(score / max_signal * 100)) for repo, score in project_signals.items()],
        key=lambda x: x[1],
        reverse=True,
    )[:5]

    # When will you likely start your next session?
    # Look at typical start hours for this day of the week
    dow_hours = [s["hour"] for s in sessions if s.get("weekday") == today_weekday]
    if dow_hours:
        typical_start = round(_mean(dow_hours))
    else:
        typical_start = round(_mean([s["hour"] for s in sessions]))

    # Session count prediction for today
    dow_counts = [v for d, v in Counter(s["date"] for s in sessions
                                         if s.get("weekday") == today_weekday).items()]
    predicted_today = round(_mean(dow_counts)) if dow_counts else 0

    # Streak prediction: are you on a streak?
    recent_dates = sorted(set(s["date"] for s in sessions), reverse=True)
    current_streak = 0
    check_date = today
    for d in recent_dates:
        if d == check_date.strftime("%Y-%m-%d"):
            current_streak += 1
            check_date -= timedelta(days=1)
        elif d == (check_date - timedelta(days=1)).strftime("%Y-%m-%d"):
            check_date -= timedelta(days=1)
            current_streak += 1
            check_date -= timedelta(days=1)
        else:
            break

    return {
        "next_project": [{"project": p, "confidence": f"{c}%"} for p, c in predicted_projects],
        "typical_start_hour": f"{typical_start:02d}:00 on {today_weekday}s",
        "predicted_sessions_today": predicted_today,
        "current_streak_days": current_streak,
        "streak_message": f"You're on a {current_streak}-day streak!" if current_streak >= 3 else
                          f"Current streak: {current_streak} day(s)",
    }


def personality_profile(sessions: list[dict]) -> dict:
    """Derive personality traits from usage patterns."""
    if not sessions:
        return {}

    traits = []

    # Chronotype
    night_sessions = len([s for s in sessions if s.get("hour", 12) in [22, 23, 0, 1, 2, 3, 4]])
    total = len(sessions)
    if night_sessions / max(total, 1) > 0.25:
        traits.append("Night Owl — you do your best thinking when the world sleeps")
    elif len([s for s in sessions if s.get("hour", 12) in [5, 6, 7, 8]]) / max(total, 1) > 0.25:
        traits.append("Early Bird — you hit the ground running before most people wake up")

    # Session length distribution
    durations = [s.get("duration_mins", 0) for s in sessions if s.get("duration_mins", 0) > 0]
    if durations:
        median_dur = _median(durations)
        long_sessions = len([d for d in durations if d > 60])
        short_sessions = len([d for d in durations if d < 10])

        if long_sessions / max(len(durations), 1) > 0.3:
            traits.append("Deep Diver — you favor long, focused work sessions")
        if short_sessions / max(len(durations), 1) > 0.4:
            traits.append("Quick Draw — lots of short, targeted sessions")
        if median_dur > 30 and long_sessions > 10 and short_sessions > 50:
            traits.append("Bimodal Worker — you either go deep or get in and out fast, rarely in between")

    # Project breadth
    projects = set(s.get("repo", "") for s in sessions)
    day_projects = defaultdict(set)
    for s in sessions:
        day_projects[s.get("date", "")].add(s.get("repo", ""))
    avg_daily_projects = _mean([len(p) for p in day_projects.values()])

    if avg_daily_projects > 4:
        traits.append("Context Switcher — you juggle 4+ projects daily like a maestro")
    elif avg_daily_projects <= 2:
        traits.append("Laser Focused — you stick to 1-2 projects per day")

    if len(projects) > 50:
        traits.append("Polymath — you've touched 50+ different projects")

    # Tool preferences
    all_tools = Counter()
    for s in sessions:
        tools = s.get("tool_uses", {})
        if isinstance(tools, dict):
            for name, count in tools.items():
                all_tools[name] += count

    if all_tools.get("Agent", 0) > 100:
        traits.append("Delegator — you spawn subagents like a general commanding troops")
    if all_tools.get("Bash", 0) > all_tools.get("Edit", 0) * 2:
        traits.append("Terminal Native — Bash is your primary weapon")
    if all_tools.get("Edit", 0) > all_tools.get("Bash", 0):
        traits.append("Code Surgeon — you edit more than you execute")

    # Weekend warrior
    weekend = len([s for s in sessions if s.get("weekday_num", 0) >= 5])
    if weekend / max(total, 1) > 0.3:
        traits.append("Weekend Warrior — 30%+ of your sessions are on weekends")

    # Subagent usage
    subagent_sessions = len([s for s in sessions if s.get("subagent_heavy", False)])
    if subagent_sessions / max(total, 1) > 0.2:
        traits.append("Orchestrator — you frequently run complex multi-agent workflows")

    return {
        "traits": traits,
        "summary": f"Based on {total} sessions across {len(projects)} projects",
    }


def fun_facts(sessions: list[dict]) -> list[str]:
    """Generate surprising, funny, or thought-provoking facts."""
    if not sessions:
        return []

    facts = []
    total = len(sessions)
    total_tokens = sum(s.get("total_tokens", 0) for s in sessions)
    output_tokens = sum(s.get("output_tokens", 0) for s in sessions)
    total_msgs = sum(s.get("total_msgs", 0) for s in sessions)
    total_tool_uses = sum(s.get("tool_use_total", 0) for s in sessions)
    durations = [s.get("duration_mins", 0) for s in sessions if s.get("duration_mins", 0) > 0]
    total_hours = sum(durations) / 60

    # Scale comparisons
    words = int(output_tokens * 0.75)
    novels = words / 80_000
    tweets = words / 40
    phd_theses = words / 80_000  # ~80K words for a thesis

    facts.append(f"Claude has written you {words:,} words — that's {novels:.1f} novels or {int(tweets):,} tweets")

    if total_hours > 100:
        netflix_seasons = total_hours / 10  # ~10 hours per season
        facts.append(f"You've spent {total_hours:.0f} hours in Claude sessions — equivalent to binging {netflix_seasons:.0f} seasons of TV")

    if total_tokens > 1_000_000_000:
        facts.append(f"You've processed {total_tokens / 1_000_000_000:.1f} BILLION tokens. That's more text than the entire English Wikipedia.")
    elif total_tokens > 100_000_000:
        encyclopedias = total_tokens / 40_000_000  # Britannica is ~40M words
        facts.append(f"You've processed {total_tokens / 1_000_000:.0f}M tokens — about {encyclopedias:.1f}x the Encyclopaedia Britannica")

    if total_tool_uses > 10000:
        facts.append(f"Claude has used {total_tool_uses:,} tools for you. That's {total_tool_uses / max(total, 1):.0f} tool calls per session on average.")

    # Day patterns
    day_counts = Counter(s["date"] for s in sessions)
    if day_counts:
        busiest = day_counts.most_common(1)[0]
        facts.append(f"Your most intense day: {busiest[0]} with {busiest[1]} sessions. What were you building?")

    # Late night
    late = len([s for s in sessions if s.get("hour", 12) in [0, 1, 2, 3, 4]])
    if late > 20:
        facts.append(f"You've had {late} sessions between midnight and 5 AM. Sleep is a suggestion, not a requirement.")
    elif late > 5:
        facts.append(f"{late} sessions between midnight and 5 AM. The code doesn't write itself... or does it?")

    # Project hopping
    day_projects = defaultdict(set)
    for s in sessions:
        day_projects[s.get("date", "")].add(s.get("repo", ""))
    max_projects_day = max((len(p), d) for d, p in day_projects.items()) if day_projects else (0, "?")
    if max_projects_day[0] > 5:
        facts.append(f"On {max_projects_day[1]}, you touched {max_projects_day[0]} different projects. ADHD or ambition? Yes.")

    # Cost
    input_tokens = sum(s.get("input_tokens", 0) for s in sessions)
    cost = (input_tokens / 1_000_000 * 15) + (output_tokens / 1_000_000 * 75)
    if cost > 100:
        coffees = cost / 5
        facts.append(f"Estimated API spend: ${cost:,.0f}. That's {coffees:.0f} cups of coffee. Worth it.")

    return facts


def analyze(sessions: list[dict]) -> dict:
    """Run all analytics and return combined results."""
    return {
        "overview": overview(sessions),
        "temporal": temporal_patterns(sessions),
        "projects": project_insights(sessions),
        "tools": tool_usage(sessions),
        "models": model_usage(sessions),
        "records": streaks_and_records(sessions),
        "predictions": predictions(sessions),
        "personality": personality_profile(sessions),
        "fun_facts": fun_facts(sessions),
    }
