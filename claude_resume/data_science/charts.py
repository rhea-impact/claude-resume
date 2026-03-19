"""SVG chart library for claude-resume data science reports.

All functions return self-contained SVG strings. No external deps.
"""

import math
from collections import Counter, defaultdict
from datetime import datetime, timedelta

import numpy as np


# Palette
C = {
    "bg": "#12121a", "surface": "#1a1a28", "border": "#2a2a3a",
    "text": "#e4e4ef", "text2": "#8888a0",
    "indigo": "#6366f1", "indigo2": "#818cf8",
    "green": "#22c55e", "amber": "#f59e0b", "rose": "#f43f5e",
    "cyan": "#06b6d4", "purple": "#a855f7", "pink": "#ec4899",
    "teal": "#14b8a6", "blue": "#3b82f6", "orange": "#f97316",
}
PALETTE = [C["indigo"], C["cyan"], C["green"], C["amber"], C["rose"],
           C["purple"], C["pink"], C["teal"], C["blue"], C["orange"]]


def _svg(w, h, content, title=""):
    hdr = f'<text x="{w/2}" y="22" text-anchor="middle" fill="{C["text2"]}" font-size="11" font-weight="600" text-transform="uppercase" letter-spacing="0.05em">{title}</text>' if title else ""
    return f'<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg"><rect width="{w}" height="{h}" rx="12" fill="{C["bg"]}"/>{hdr}{content}</svg>'


# -----------------------------------------------------------------------
# 1. Cumulative sessions over time
# -----------------------------------------------------------------------
def cumulative_sessions(sessions, width=600, height=250):
    dates = sorted(s.get("date", "") for s in sessions if s.get("date"))
    if not dates:
        return ""
    by_date = Counter(dates)
    all_dates = sorted(by_date.keys())
    cumulative = []
    total = 0
    for d in all_dates:
        total += by_date[d]
        cumulative.append(total)

    m = 45
    pw, ph = width - 2*m, height - 2*m - 10
    mx = max(cumulative)
    n = len(cumulative)

    points = []
    for i, v in enumerate(cumulative):
        x = m + i / max(n-1, 1) * pw
        y = m + 10 + ph - v / mx * ph
        points.append(f"{x:.1f},{y:.1f}")

    fill_pts = f"{m},{m+10+ph} " + " ".join(points) + f" {m + pw},{m+10+ph}"
    line = f'<polyline points="{" ".join(points)}" fill="none" stroke="{C["indigo"]}" stroke-width="2.5" stroke-linejoin="round"/>'
    fill = f'<polygon points="{fill_pts}" fill="url(#cg)" opacity="0.3"/>'
    grad = f'<defs><linearGradient id="cg" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="{C["indigo"]}"/><stop offset="100%" stop-color="transparent"/></linearGradient></defs>'

    # X-axis labels
    labels = ""
    step = max(n // 6, 1)
    for i in range(0, n, step):
        x = m + i / max(n-1,1) * pw
        labels += f'<text x="{x:.0f}" y="{m+10+ph+16}" text-anchor="middle" fill="{C["text2"]}" font-size="9">{all_dates[i][5:]}</text>'

    return _svg(width, height, grad + fill + line + labels +
        f'<text x="{m+pw}" y="{m+5}" text-anchor="end" fill="{C["text"]}" font-size="12" font-weight="700">{mx:,}</text>',
        "Cumulative Sessions")


# -----------------------------------------------------------------------
# 2. Tokens per day (area chart)
# -----------------------------------------------------------------------
def tokens_per_day(sessions, width=600, height=250):
    by_date = defaultdict(int)
    for s in sessions:
        by_date[s.get("date", "")] += s.get("total_tokens", 0)
    dates = sorted(by_date.keys())
    if not dates:
        return ""
    values = [by_date[d] for d in dates]

    m = 45
    pw, ph = width - 2*m, height - 2*m - 10
    mx = max(values) or 1
    n = len(values)

    points = []
    for i, v in enumerate(values):
        x = m + i / max(n-1,1) * pw
        y = m + 10 + ph - v / mx * ph
        points.append(f"{x:.1f},{y:.1f}")

    fill_pts = f"{m},{m+10+ph} " + " ".join(points) + f" {m+pw},{m+10+ph}"
    return _svg(width, height,
        f'<defs><linearGradient id="tg" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="{C["cyan"]}"/><stop offset="100%" stop-color="transparent"/></linearGradient></defs>'
        f'<polygon points="{fill_pts}" fill="url(#tg)" opacity="0.3"/>'
        f'<polyline points="{" ".join(points)}" fill="none" stroke="{C["cyan"]}" stroke-width="2" stroke-linejoin="round"/>',
        "Tokens per Day")


# -----------------------------------------------------------------------
# 3. Hour × Day heatmap (7×24 grid)
# -----------------------------------------------------------------------
def hour_day_heatmap(sessions, width=600, height=250):
    grid = defaultdict(int)
    for s in sessions:
        grid[(s.get("weekday_num", 0), s.get("hour", 0))] += 1
    mx = max(grid.values()) if grid else 1
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    m_left, m_top = 50, 35
    cw = (width - m_left - 20) / 24
    ch = (height - m_top - 20) / 7

    cells = ""
    for d in range(7):
        for h in range(24):
            v = grid.get((d, h), 0)
            alpha = 0.05 + 0.95 * (v / mx)
            x = m_left + h * cw
            y = m_top + d * ch
            cells += f'<rect x="{x:.1f}" y="{y:.1f}" width="{cw-1:.1f}" height="{ch-1:.1f}" rx="3" fill="rgba(99,102,241,{alpha:.2f})"/>'

    # Labels
    labels = ""
    for d, name in enumerate(days):
        labels += f'<text x="{m_left-6}" y="{m_top + d*ch + ch/2 + 4}" text-anchor="end" fill="{C["text2"]}" font-size="10">{name}</text>'
    for h in range(0, 24, 3):
        labels += f'<text x="{m_left + h*cw + cw/2}" y="{m_top - 6}" text-anchor="middle" fill="{C["text2"]}" font-size="9">{h:02d}</text>'

    return _svg(width, height, cells + labels, "Hour × Day Heatmap")


# -----------------------------------------------------------------------
# 4. Project timeline (gantt-style)
# -----------------------------------------------------------------------
def project_timeline(sessions, width=700, height=350):
    proj_dates = defaultdict(lambda: {"first": None, "last": None, "count": 0})
    for s in sessions:
        repo = s.get("repo", "?")
        date = s.get("date", "")
        if not date:
            continue
        p = proj_dates[repo]
        p["count"] += 1
        if p["first"] is None or date < p["first"]:
            p["first"] = date
        if p["last"] is None or date > p["last"]:
            p["last"] = date

    # Top 15 by session count
    top = sorted(proj_dates.items(), key=lambda x: x[1]["count"], reverse=True)[:15]
    if not top:
        return ""

    all_dates = sorted(set(s.get("date", "") for s in sessions if s.get("date")))
    if not all_dates:
        return ""
    date_min = datetime.strptime(all_dates[0], "%Y-%m-%d")
    date_max = datetime.strptime(all_dates[-1], "%Y-%m-%d")
    date_range = (date_max - date_min).days or 1

    m_left, m_top, m_right = 130, 35, 20
    pw = width - m_left - m_right
    row_h = (height - m_top - 20) / len(top)

    bars = ""
    for i, (name, info) in enumerate(top):
        y = m_top + i * row_h
        x1 = m_left + (datetime.strptime(info["first"], "%Y-%m-%d") - date_min).days / date_range * pw
        x2 = m_left + (datetime.strptime(info["last"], "%Y-%m-%d") - date_min).days / date_range * pw
        w = max(x2 - x1, 4)
        color = PALETTE[i % len(PALETTE)]
        bars += f'<rect x="{x1:.0f}" y="{y+2:.0f}" width="{w:.0f}" height="{row_h-4:.0f}" rx="4" fill="{color}" opacity="0.7"/>'
        bars += f'<text x="{m_left-6}" y="{y+row_h/2+4:.0f}" text-anchor="end" fill="{C["text2"]}" font-size="9">{name[:16]}</text>'

    return _svg(width, height, bars, "Project Timeline")


# -----------------------------------------------------------------------
# 5. Rolling 7-day session count
# -----------------------------------------------------------------------
def rolling_sessions(sessions, window=7, width=600, height=200):
    by_date = Counter(s.get("date", "") for s in sessions)
    dates = sorted(by_date.keys())
    if len(dates) < window:
        return ""

    values = [by_date[d] for d in dates]
    rolling = [sum(values[max(0,i-window+1):i+1])/min(i+1,window) for i in range(len(values))]

    m = 45
    pw, ph = width - 2*m, height - 2*m
    mx = max(rolling) or 1

    points = " ".join(f"{m + i/max(len(rolling)-1,1)*pw:.1f},{m + ph - v/mx*ph:.1f}" for i, v in enumerate(rolling))
    return _svg(width, height,
        f'<polyline points="{points}" fill="none" stroke="{C["green"]}" stroke-width="2.5" stroke-linejoin="round"/>'
        f'<text x="{m+pw}" y="{m-4}" text-anchor="end" fill="{C["text"]}" font-size="11" font-weight="700">{rolling[-1]:.0f}/day avg</text>',
        f"Rolling {window}-Day Average Sessions")


# -----------------------------------------------------------------------
# 6. Session duration by project (horizontal box-like bars)
# -----------------------------------------------------------------------
def duration_by_project(sessions, width=600, height=350):
    proj_durs = defaultdict(list)
    for s in sessions:
        d = s.get("duration_mins", 0)
        if d > 0.5:
            proj_durs[s.get("repo", "?")].append(d)

    # Top 12 by session count
    top = sorted(proj_durs.items(), key=lambda x: len(x[1]), reverse=True)[:12]
    if not top:
        return ""

    m_left, m_top = 130, 35
    pw = width - m_left - 30
    row_h = (height - m_top - 20) / len(top)

    # Use log scale
    all_vals = [v for _, durs in top for v in durs]
    mx = np.log1p(np.percentile(all_vals, 95))

    bars = ""
    for i, (name, durs) in enumerate(top):
        y = m_top + i * row_h
        arr = np.array(durs)
        p25, median, p75 = np.log1p(np.percentile(arr, [25, 50, 75]))
        color = PALETTE[i % len(PALETTE)]

        x1 = m_left + p25 / mx * pw
        x2 = m_left + p75 / mx * pw
        xm = m_left + median / mx * pw

        # IQR bar
        bars += f'<rect x="{x1:.0f}" y="{y+row_h*0.2:.0f}" width="{max(x2-x1,2):.0f}" height="{row_h*0.6:.0f}" rx="3" fill="{color}" opacity="0.6"/>'
        # Median line
        bars += f'<line x1="{xm:.0f}" y1="{y+row_h*0.1:.0f}" x2="{xm:.0f}" y2="{y+row_h*0.9:.0f}" stroke="{C["text"]}" stroke-width="2"/>'
        bars += f'<text x="{m_left-6}" y="{y+row_h/2+4:.0f}" text-anchor="end" fill="{C["text2"]}" font-size="9">{name[:16]}</text>'
        bars += f'<text x="{x2+6:.0f}" y="{y+row_h/2+4:.0f}" fill="{C["text2"]}" font-size="8">{np.median(arr):.0f}m</text>'

    return _svg(width, height, bars, "Session Duration by Project (IQR)")


# -----------------------------------------------------------------------
# 7. Donut chart — message ratio
# -----------------------------------------------------------------------
def message_donut(sessions, width=250, height=250):
    user = sum(s.get("user_msgs", 0) for s in sessions)
    asst = sum(s.get("assistant_msgs", 0) for s in sessions)
    total = user + asst or 1

    cx, cy, r = width/2, height/2 + 5, 80
    circumference = 2 * math.pi * r
    user_arc = user / total * circumference
    asst_arc = asst / total * circumference

    return _svg(width, height,
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{C["cyan"]}" stroke-width="20" '
        f'stroke-dasharray="{user_arc:.1f} {circumference:.1f}" transform="rotate(-90 {cx} {cy})"/>'
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{C["indigo"]}" stroke-width="20" '
        f'stroke-dasharray="{asst_arc:.1f} {circumference:.1f}" stroke-dashoffset="-{user_arc:.1f}" transform="rotate(-90 {cx} {cy})"/>'
        f'<text x="{cx}" y="{cy-6}" text-anchor="middle" fill="{C["text"]}" font-size="18" font-weight="800">{total:,}</text>'
        f'<text x="{cx}" y="{cy+12}" text-anchor="middle" fill="{C["text2"]}" font-size="10">messages</text>'
        f'<circle cx="{cx-50}" cy="{cy+r+20}" r="5" fill="{C["cyan"]}"/><text x="{cx-40}" y="{cy+r+24}" fill="{C["text2"]}" font-size="10">You {user/total*100:.0f}%</text>'
        f'<circle cx="{cx+20}" cy="{cy+r+20}" r="5" fill="{C["indigo"]}"/><text x="{cx+30}" y="{cy+r+24}" fill="{C["text2"]}" font-size="10">Claude {asst/total*100:.0f}%</text>',
        "Message Ratio")


# -----------------------------------------------------------------------
# 8. Monthly comparison
# -----------------------------------------------------------------------
def monthly_bars(sessions, width=500, height=220):
    by_month = Counter(s.get("month", "") for s in sessions)
    months = sorted(by_month.keys())
    if not months:
        return ""

    values = [by_month[m] for m in months]
    mx = max(values) or 1
    m_left, m_top = 40, 35
    pw = width - m_left - 20
    ph = height - m_top - 30
    bar_w = pw / len(months) - 4

    bars = ""
    for i, (month, v) in enumerate(zip(months, values)):
        h = v / mx * ph
        x = m_left + i * (bar_w + 4)
        y = m_top + ph - h
        bars += f'<rect x="{x:.0f}" y="{y:.0f}" width="{bar_w:.0f}" height="{h:.0f}" rx="4" fill="{C["purple"]}" opacity="0.7"/>'
        bars += f'<text x="{x+bar_w/2:.0f}" y="{m_top+ph+14}" text-anchor="middle" fill="{C["text2"]}" font-size="9">{month[5:]}</text>'
        bars += f'<text x="{x+bar_w/2:.0f}" y="{y-4}" text-anchor="middle" fill="{C["text"]}" font-size="10" font-weight="700">{v}</text>'

    return _svg(width, height, bars, "Sessions by Month")


# -----------------------------------------------------------------------
# 9. Top MCP integrations
# -----------------------------------------------------------------------
def mcp_tools_chart(sessions, width=500, height=280):
    mcp = Counter()
    for s in sessions:
        tools = s.get("tool_uses", {})
        if isinstance(tools, dict):
            for name, count in tools.items():
                if name.startswith("mcp__"):
                    # Extract server name
                    parts = name.split("__")
                    server = parts[1] if len(parts) > 1 else name
                    mcp[server] += count

    top = mcp.most_common(12)
    if not top:
        return ""
    mx = top[0][1]

    m_left, m_top = 120, 35
    pw = width - m_left - 20
    row_h = (height - m_top - 10) / len(top)

    bars = ""
    for i, (name, count) in enumerate(top):
        y = m_top + i * row_h
        w = count / mx * pw
        color = PALETTE[i % len(PALETTE)]
        bars += f'<rect x="{m_left}" y="{y+2:.0f}" width="{w:.0f}" height="{row_h-4:.0f}" rx="3" fill="{color}" opacity="0.7"/>'
        bars += f'<text x="{m_left-6}" y="{y+row_h/2+4:.0f}" text-anchor="end" fill="{C["text2"]}" font-size="9">{name[:15]}</text>'
        bars += f'<text x="{m_left+w+6:.0f}" y="{y+row_h/2+4:.0f}" fill="{C["text2"]}" font-size="9">{count:,}</text>'

    return _svg(width, height, bars, "MCP Server Usage")


# -----------------------------------------------------------------------
# 10. Session size distribution
# -----------------------------------------------------------------------
def size_distribution(sessions, width=500, height=200):
    sizes = [s.get("size", 0) / 1024 for s in sessions if s.get("size", 0) > 0]  # KB
    if not sizes:
        return ""
    log_sizes = np.log1p(sizes)
    hist, edges = np.histogram(log_sizes, bins=30)
    mx = max(hist) or 1
    m = 40
    pw, ph = width - 2*m, height - 2*m
    bar_w = pw / 30 - 1

    bars = ""
    for i, c in enumerate(hist):
        h = c / mx * ph
        x = m + i * (bar_w + 1)
        y = m + ph - h
        bars += f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" rx="1" fill="{C["teal"]}" opacity="0.7"/>'

    return _svg(width, height, bars, "Session File Size Distribution (log KB)")


# -----------------------------------------------------------------------
# 11. Productivity by day of week (tokens per session)
# -----------------------------------------------------------------------
def productivity_by_day(sessions, width=500, height=220):
    day_tokens = defaultdict(list)
    for s in sessions:
        if s.get("total_tokens", 0) > 0:
            day_tokens[s.get("weekday", "?")].append(s["total_tokens"])

    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    avgs = [np.mean(day_tokens.get(d, [0])) for d in days]
    mx = max(avgs) or 1

    m_left, m_top = 40, 35
    pw = width - m_left - 20
    ph = height - m_top - 30
    bar_w = pw / 7 - 6

    bars = ""
    for i, (day, avg) in enumerate(zip(days, avgs)):
        h = avg / mx * ph
        x = m_left + i * (bar_w + 6)
        y = m_top + ph - h
        bars += f'<rect x="{x:.0f}" y="{y:.0f}" width="{bar_w:.0f}" height="{h:.0f}" rx="4" fill="{C["amber"]}" opacity="0.7"/>'
        bars += f'<text x="{x+bar_w/2:.0f}" y="{m_top+ph+14}" text-anchor="middle" fill="{C["text2"]}" font-size="9">{day[:3]}</text>'

    return _svg(width, height, bars, "Avg Tokens per Session by Day")


# -----------------------------------------------------------------------
# 12. Session start time distribution (density)
# -----------------------------------------------------------------------
def start_time_density(sessions, width=500, height=180):
    hours = [s.get("hour", 0) + np.random.uniform(-0.3, 0.3) for s in sessions]
    hist, edges = np.histogram(hours, bins=48, range=(0, 24))
    mx = max(hist) or 1

    m = 40
    pw, ph = width - 2*m, height - 2*m
    points = []
    for i, c in enumerate(hist):
        x = m + (edges[i] + edges[i+1]) / 2 / 24 * pw
        y = m + ph - c / mx * ph
        points.append(f"{x:.1f},{y:.1f}")

    fill_pts = f"{m},{m+ph} " + " ".join(points) + f" {m+pw},{m+ph}"
    labels = "".join(f'<text x="{m + h/24*pw:.0f}" y="{m+ph+14}" text-anchor="middle" fill="{C["text2"]}" font-size="9">{h:02d}</text>' for h in range(0, 24, 4))

    return _svg(width, height,
        f'<defs><linearGradient id="sd" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="{C["pink"]}"/><stop offset="100%" stop-color="transparent"/></linearGradient></defs>'
        f'<polygon points="{fill_pts}" fill="url(#sd)" opacity="0.4"/>'
        f'<polyline points="{" ".join(points)}" fill="none" stroke="{C["pink"]}" stroke-width="2" stroke-linejoin="round"/>'
        + labels, "Session Start Time Density")


# -----------------------------------------------------------------------
# 13. Tool category pie
# -----------------------------------------------------------------------
def tool_category_donut(sessions, width=280, height=280):
    cats = {"Read": 0, "Write": 0, "Execute": 0, "Search": 0, "Agent": 0, "MCP": 0, "Other": 0}
    read_tools = {"Read", "Glob", "Grep", "LS"}
    write_tools = {"Edit", "Write", "NotebookEdit"}
    exec_tools = {"Bash", "BashOutput"}
    search_tools = {"WebSearch", "WebFetch"}

    for s in sessions:
        tools = s.get("tool_uses", {})
        if not isinstance(tools, dict):
            continue
        for name, count in tools.items():
            if name in read_tools: cats["Read"] += count
            elif name in write_tools: cats["Write"] += count
            elif name in exec_tools: cats["Execute"] += count
            elif name in search_tools: cats["Search"] += count
            elif name == "Agent": cats["Agent"] += count
            elif name.startswith("mcp__"): cats["MCP"] += count
            else: cats["Other"] += count

    total = sum(cats.values()) or 1
    cx, cy, r = width/2, height/2 + 5, 85
    colors = [C["cyan"], C["indigo"], C["green"], C["amber"], C["purple"], C["pink"], C["teal"]]

    arcs = ""
    offset = 0
    circumference = 2 * math.pi * r
    legend = ""
    ly = cy + r + 15
    for i, (name, count) in enumerate(cats.items()):
        if count == 0:
            continue
        arc = count / total * circumference
        arcs += f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{colors[i]}" stroke-width="22" stroke-dasharray="{arc:.1f} {circumference:.1f}" stroke-dashoffset="-{offset:.1f}" transform="rotate(-90 {cx} {cy})"/>'
        offset += arc

    return _svg(width, height, arcs +
        f'<text x="{cx}" y="{cy}" text-anchor="middle" fill="{C["text"]}" font-size="13" font-weight="700">Tool Mix</text>',
        "")


# -----------------------------------------------------------------------
# 14. Sessions per hour radar
# -----------------------------------------------------------------------
def hourly_radar(sessions, width=300, height=300):
    hour_counts = Counter(s.get("hour", 0) for s in sessions)
    mx = max(hour_counts.values()) if hour_counts else 1
    cx, cy, r = width/2, height/2 + 10, 110

    # Grid circles
    grid = ""
    for frac in [0.25, 0.5, 0.75, 1.0]:
        grid += f'<circle cx="{cx}" cy="{cy}" r="{r*frac}" fill="none" stroke="{C["border"]}" stroke-width="1"/>'

    # Data polygon
    points = []
    for h in range(24):
        angle = 2 * math.pi * h / 24 - math.pi / 2
        v = hour_counts.get(h, 0) / mx
        px = cx + r * v * math.cos(angle)
        py = cy + r * v * math.sin(angle)
        points.append(f"{px:.1f},{py:.1f}")

    # Hour labels
    labels = ""
    for h in range(0, 24, 3):
        angle = 2 * math.pi * h / 24 - math.pi / 2
        lx = cx + (r + 16) * math.cos(angle)
        ly_pos = cy + (r + 16) * math.sin(angle)
        labels += f'<text x="{lx:.0f}" y="{ly_pos + 4:.0f}" text-anchor="middle" fill="{C["text2"]}" font-size="9">{h:02d}</text>'

    return _svg(width, height,
        grid +
        f'<polygon points="{" ".join(points)}" fill="{C["indigo"]}" opacity="0.2" stroke="{C["indigo"]}" stroke-width="2"/>'
        + labels, "")


# -----------------------------------------------------------------------
# 15. Token efficiency (output/input ratio over time)
# -----------------------------------------------------------------------
def token_efficiency(sessions, width=600, height=200):
    by_date = defaultdict(lambda: [0, 0])
    for s in sessions:
        d = s.get("date", "")
        by_date[d][0] += s.get("output_tokens", 0)
        by_date[d][1] += s.get("input_tokens", 0)

    dates = sorted(by_date.keys())
    if not dates:
        return ""
    ratios = [by_date[d][0] / max(by_date[d][1], 1) for d in dates]
    # 3-day rolling
    smoothed = [np.mean(ratios[max(0,i-2):i+1]) for i in range(len(ratios))]

    m = 45
    pw, ph = width - 2*m, height - 2*m
    mx = max(smoothed) or 1

    points = " ".join(f"{m+i/max(len(smoothed)-1,1)*pw:.1f},{m+ph-v/mx*ph:.1f}" for i, v in enumerate(smoothed))

    return _svg(width, height,
        f'<polyline points="{points}" fill="none" stroke="{C["orange"]}" stroke-width="2" stroke-linejoin="round"/>'
        f'<line x1="{m}" y1="{m+ph/2}" x2="{m+pw}" y2="{m+ph/2}" stroke="{C["border"]}" stroke-dasharray="4 4"/>',
        "Output/Input Token Ratio (3-day rolling)")


# -----------------------------------------------------------------------
# 16. Subagent usage over time
# -----------------------------------------------------------------------
def subagent_trend(sessions, width=600, height=200):
    by_date = defaultdict(lambda: [0, 0])
    for s in sessions:
        d = s.get("date", "")
        by_date[d][0] += 1
        if s.get("subagent_heavy"):
            by_date[d][1] += 1

    dates = sorted(by_date.keys())
    if not dates:
        return ""
    rates = [by_date[d][1] / max(by_date[d][0], 1) * 100 for d in dates]
    smoothed = [np.mean(rates[max(0,i-2):i+1]) for i in range(len(rates))]

    m = 45
    pw, ph = width - 2*m, height - 2*m
    mx = max(smoothed) if smoothed else 1
    if mx == 0:
        mx = 1

    points = " ".join(f"{m+i/max(len(smoothed)-1,1)*pw:.1f},{m+ph-v/mx*ph:.1f}" for i, v in enumerate(smoothed))

    return _svg(width, height,
        f'<polyline points="{points}" fill="none" stroke="{C["purple"]}" stroke-width="2.5" stroke-linejoin="round"/>',
        "Multi-Agent Session Rate (% of daily sessions)")


# -----------------------------------------------------------------------
# 17. Sessions per day scatter
# -----------------------------------------------------------------------
def daily_scatter(sessions, width=600, height=200):
    by_date = Counter(s.get("date", "") for s in sessions)
    dates = sorted(by_date.keys())
    if not dates:
        return ""
    values = [by_date[d] for d in dates]

    m = 45
    pw, ph = width - 2*m, height - 2*m
    mx = max(values) or 1

    dots = ""
    for i, v in enumerate(values):
        x = m + i / max(len(values)-1, 1) * pw
        y = m + ph - v / mx * ph
        r = min(2 + v / mx * 4, 8)
        dots += f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{C["rose"]}" opacity="0.6"/>'

    return _svg(width, height, dots, "Sessions per Day")


# -----------------------------------------------------------------------
# 18. Flow rate by hour
# -----------------------------------------------------------------------
def flow_by_hour_chart(flow_data, width=500, height=200):
    rates = flow_data.get("flow_rate_by_hour", {})
    if not rates:
        return ""

    m = 45
    pw, ph = width - 2*m, height - 2*m
    mx = max(rates.values()) or 1
    bar_w = pw / 24 - 2

    bars = ""
    for h in range(24):
        rate = rates.get(f"{h:02d}:00", 0)
        bh = rate / mx * ph
        x = m + h * (bar_w + 2)
        y = m + ph - bh
        color = C["green"] if rate > 10 else C["indigo"]
        bars += f'<rect x="{x:.0f}" y="{y:.0f}" width="{bar_w:.0f}" height="{bh:.0f}" rx="2" fill="{color}" opacity="0.7"/>'

    labels = "".join(f'<text x="{m + h*(bar_w+2) + bar_w/2:.0f}" y="{m+ph+13}" text-anchor="middle" fill="{C["text2"]}" font-size="8">{h:02d}</text>' for h in range(0, 24, 3))
    return _svg(width, height, bars + labels, "Flow State Rate by Hour (%)")


# -----------------------------------------------------------------------
# 19. Burnout trend
# -----------------------------------------------------------------------
def burnout_trend(burnout_data, width=600, height=200):
    wd = burnout_data.get("weekly_data", {})
    weeks = wd.get("weeks", [])
    hours = wd.get("hours", [])
    late = wd.get("late_nights", [])
    if not weeks:
        return ""

    m = 45
    pw, ph = width - 2*m, height - 2*m

    def line(values, color, label_text):
        mx = max(values) if values else 1
        if mx == 0:
            mx = 1
        pts = " ".join(f"{m+i/max(len(values)-1,1)*pw:.1f},{m+ph-v/mx*ph:.1f}" for i, v in enumerate(values))
        return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round"/>'

    return _svg(width, height,
        line(hours, C["amber"], "Hours") +
        line(late, C["rose"], "Late nights") +
        f'<circle cx="{m+pw-60}" cy="18" r="4" fill="{C["amber"]}"/><text x="{m+pw-52}" y="22" fill="{C["text2"]}" font-size="9">Hours/wk</text>'
        f'<circle cx="{m+pw}" cy="18" r="4" fill="{C["rose"]}"/><text x="{m+pw+8}" y="22" fill="{C["text2"]}" font-size="9">Late nights</text>',
        "Weekly Burnout Signals")


# -----------------------------------------------------------------------
# 20. Co-occurrence network
# -----------------------------------------------------------------------
def cooccurrence_graph(cooccurrence_data, width=500, height=400):
    edges = cooccurrence_data.get("edges", [])[:20]
    if not edges:
        return ""

    nodes = set()
    for e in edges:
        nodes.add(e["from"])
        nodes.add(e["to"])
    node_list = list(nodes)
    n = len(node_list)
    if n == 0:
        return ""

    # Layout: circular
    cx, cy = width / 2, height / 2 + 10
    radius = min(width, height) / 2 - 60
    pos = {}
    for i, node in enumerate(node_list):
        angle = 2 * math.pi * i / n - math.pi / 2
        pos[node] = (cx + radius * math.cos(angle), cy + radius * math.sin(angle))

    max_w = max(e["weight"] for e in edges)
    lines = ""
    for e in edges:
        x1, y1 = pos[e["from"]]
        x2, y2 = pos[e["to"]]
        w = 0.5 + 3.5 * e["weight"] / max_w
        opacity = 0.15 + 0.6 * e["weight"] / max_w
        lines += f'<line x1="{x1:.0f}" y1="{y1:.0f}" x2="{x2:.0f}" y2="{y2:.0f}" stroke="{C["indigo"]}" stroke-width="{w:.1f}" opacity="{opacity:.2f}"/>'

    node_elems = ""
    for node in node_list:
        x, y = pos[node]
        node_elems += f'<circle cx="{x:.0f}" cy="{y:.0f}" r="18" fill="{C["surface"]}" stroke="{C["cyan"]}" stroke-width="1.5"/>'
        node_elems += f'<text x="{x:.0f}" y="{y+3:.0f}" text-anchor="middle" fill="{C["text"]}" font-size="7" font-weight="600">{node[:10]}</text>'

    return _svg(width, height, lines + node_elems, "Project Co-occurrence Network")


# -----------------------------------------------------------------------
# 21. GitHub-style contribution calendar
# -----------------------------------------------------------------------
def contribution_calendar(sessions, width=700, height=160):
    by_date = Counter(s.get("date", "") for s in sessions)
    if not by_date:
        return ""
    dates = sorted(by_date.keys())
    start = datetime.strptime(dates[0], "%Y-%m-%d")
    end = datetime.strptime(dates[-1], "%Y-%m-%d")
    mx = max(by_date.values())

    cell = 11
    gap = 2
    m_left, m_top = 30, 35
    day = start
    cells = ""
    week = 0
    while day <= end:
        dow = day.weekday()
        key = day.strftime("%Y-%m-%d")
        v = by_date.get(key, 0)
        alpha = 0.06 + 0.94 * (v / mx) if v > 0 else 0.03
        x = m_left + week * (cell + gap)
        y = m_top + dow * (cell + gap)
        cells += f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" rx="2" fill="rgba(99,102,241,{alpha:.2f})"/>'
        if dow == 6:
            week += 1
        day += timedelta(days=1)

    # Day labels
    day_labels = ""
    for i, name in enumerate(["M", "", "W", "", "F", "", "S"]):
        if name:
            day_labels += f'<text x="{m_left-8}" y="{m_top + i*(cell+gap) + cell - 1}" text-anchor="end" fill="{C["text2"]}" font-size="8">{name}</text>'

    return _svg(width, height, cells + day_labels, "Activity Calendar")


# -----------------------------------------------------------------------
# 22. Session duration trend (are sessions getting longer?)
# -----------------------------------------------------------------------
def duration_trend(sessions, width=600, height=200):
    by_date = defaultdict(list)
    for s in sessions:
        d = s.get("duration_mins", 0)
        if d > 0.5:
            by_date[s.get("date", "")].append(d)
    dates = sorted(by_date.keys())
    if len(dates) < 3:
        return ""
    avgs = [np.mean(by_date[d]) for d in dates]
    # 5-day rolling
    smoothed = [np.mean(avgs[max(0,i-4):i+1]) for i in range(len(avgs))]

    m = 45
    pw, ph = width - 2*m, height - 2*m
    mx = max(smoothed) or 1
    points = " ".join(f"{m+i/max(len(smoothed)-1,1)*pw:.1f},{m+ph-v/mx*ph:.1f}" for i, v in enumerate(smoothed))

    return _svg(width, height,
        f'<polyline points="{points}" fill="none" stroke="{C["amber"]}" stroke-width="2.5" stroke-linejoin="round"/>'
        f'<text x="{m+pw}" y="{m-4}" text-anchor="end" fill="{C["text"]}" font-size="11" font-weight="700">{smoothed[-1]:.0f}min avg</text>',
        "Avg Session Duration Trend (5-day rolling)")


# -----------------------------------------------------------------------
# 23. Session velocity (messages per minute)
# -----------------------------------------------------------------------
def velocity_scatter(sessions, width=600, height=220):
    dots = ""
    m = 45
    pw, ph = width - 2*m, height - 2*m

    pts = []
    for s in sessions:
        dur = s.get("duration_mins", 0)
        msgs = s.get("total_msgs", 0)
        if dur > 1 and msgs > 2:
            pts.append((dur, msgs / dur))

    if not pts:
        return ""
    arr = np.array(pts)
    mx_x = np.percentile(arr[:, 0], 95)
    mx_y = np.percentile(arr[:, 1], 95)

    sample = np.random.choice(len(arr), min(600, len(arr)), replace=False)
    for i in sample:
        dur, vel = arr[i]
        x = m + min(dur / mx_x, 1) * pw
        y = m + ph - min(vel / mx_y, 1) * ph
        dots += f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.5" fill="{C["cyan"]}" opacity="0.4"/>'

    return _svg(width, height, dots +
        f'<text x="{m+pw/2}" y="{m+ph+16}" text-anchor="middle" fill="{C["text2"]}" font-size="9">Duration (min)</text>'
        f'<text x="{m-8}" y="{m+ph/2}" text-anchor="middle" fill="{C["text2"]}" font-size="9" transform="rotate(-90 {m-8} {m+ph/2})">msgs/min</text>',
        "Session Velocity (Messages per Minute)")


# -----------------------------------------------------------------------
# 24. Cost estimate over time
# -----------------------------------------------------------------------
def cost_trend(sessions, width=600, height=200):
    by_date = defaultdict(int)
    for s in sessions:
        tokens = s.get("total_tokens", 0)
        by_date[s.get("date", "")] += tokens
    dates = sorted(by_date.keys())
    if not dates:
        return ""

    # rough cost: $3/M input, $15/M output  → approximate as $8/M blended
    costs = [by_date[d] * 8 / 1_000_000 for d in dates]
    cumulative = []
    total = 0.0
    for c in costs:
        total += c
        cumulative.append(total)

    m = 45
    pw, ph = width - 2*m, height - 2*m
    mx = max(cumulative) or 1
    points = []
    for i, v in enumerate(cumulative):
        x = m + i / max(len(cumulative)-1, 1) * pw
        y = m + ph - v / mx * ph
        points.append(f"{x:.1f},{y:.1f}")

    fill_pts = f"{m},{m+ph} " + " ".join(points) + f" {m+pw},{m+ph}"
    return _svg(width, height,
        f'<defs><linearGradient id="cst" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="{C["rose"]}"/><stop offset="100%" stop-color="transparent"/></linearGradient></defs>'
        f'<polygon points="{fill_pts}" fill="url(#cst)" opacity="0.3"/>'
        f'<polyline points="{" ".join(points)}" fill="none" stroke="{C["rose"]}" stroke-width="2.5" stroke-linejoin="round"/>'
        f'<text x="{m+pw}" y="{m}" text-anchor="end" fill="{C["text"]}" font-size="12" font-weight="800">${total:,.0f}</text>',
        "Estimated Cumulative Cost")


# -----------------------------------------------------------------------
# 25. Weekend vs weekday comparison
# -----------------------------------------------------------------------
def weekend_vs_weekday(sessions, width=400, height=220):
    wd_tokens, wd_sessions, wd_duration = 0, 0, 0.0
    we_tokens, we_sessions, we_duration = 0, 0, 0.0
    for s in sessions:
        if s.get("weekday_num", 0) < 5:
            wd_tokens += s.get("total_tokens", 0)
            wd_sessions += 1
            wd_duration += s.get("duration_mins", 0)
        else:
            we_tokens += s.get("total_tokens", 0)
            we_sessions += 1
            we_duration += s.get("duration_mins", 0)

    if wd_sessions == 0 or we_sessions == 0:
        return ""

    # Normalize per day (5 weekdays vs 2 weekend days)
    wd_daily = wd_sessions / 5
    we_daily = we_sessions / 2
    mx = max(wd_daily, we_daily)

    m = 50
    pw = width - 2*m
    ph = 100
    bw = 80

    bars = ""
    for i, (label, val, color) in enumerate([("Weekday", wd_daily, C["indigo"]), ("Weekend", we_daily, C["amber"])]):
        x = m + i * (bw + 40)
        h = val / mx * ph
        y = m + 30 + ph - h
        bars += f'<rect x="{x}" y="{y}" width="{bw}" height="{h}" rx="6" fill="{color}" opacity="0.7"/>'
        bars += f'<text x="{x+bw/2}" y="{y-8}" text-anchor="middle" fill="{C["text"]}" font-size="14" font-weight="800">{val:.0f}</text>'
        bars += f'<text x="{x+bw/2}" y="{m+30+ph+18}" text-anchor="middle" fill="{C["text2"]}" font-size="11">{label}</text>'

    return _svg(width, height, bars +
        f'<text x="{width/2}" y="{m+30+ph+36}" text-anchor="middle" fill="{C["text2"]}" font-size="9">sessions/day average</text>',
        "Weekend vs Weekday Intensity")


# -----------------------------------------------------------------------
# 26. Project diversity over time (unique projects per day)
# -----------------------------------------------------------------------
def project_diversity(sessions, width=600, height=200):
    by_date = defaultdict(set)
    for s in sessions:
        by_date[s.get("date", "")].add(s.get("repo", "?"))
    dates = sorted(by_date.keys())
    if len(dates) < 3:
        return ""
    values = [len(by_date[d]) for d in dates]
    smoothed = [np.mean(values[max(0,i-2):i+1]) for i in range(len(values))]

    m = 45
    pw, ph = width - 2*m, height - 2*m
    mx = max(smoothed) or 1

    points = []
    for i, v in enumerate(smoothed):
        x = m + i / max(len(smoothed)-1,1) * pw
        y = m + ph - v / mx * ph
        points.append(f"{x:.1f},{y:.1f}")

    fill_pts = f"{m},{m+ph} " + " ".join(points) + f" {m+pw},{m+ph}"
    return _svg(width, height,
        f'<defs><linearGradient id="pdiv" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="{C["teal"]}"/><stop offset="100%" stop-color="transparent"/></linearGradient></defs>'
        f'<polygon points="{fill_pts}" fill="url(#pdiv)" opacity="0.3"/>'
        f'<polyline points="{" ".join(points)}" fill="none" stroke="{C["teal"]}" stroke-width="2.5" stroke-linejoin="round"/>'
        f'<text x="{m+pw}" y="{m-4}" text-anchor="end" fill="{C["text"]}" font-size="11" font-weight="700">{values[-1]} projects today</text>',
        "Project Diversity (unique repos/day)")


# -----------------------------------------------------------------------
# 27. Cache efficiency (cache_read / input ratio)
# -----------------------------------------------------------------------
def cache_efficiency(sessions, width=600, height=200):
    by_date = defaultdict(lambda: [0, 0])
    for s in sessions:
        inp = s.get("input_tokens", 0)
        cache = s.get("cache_read_tokens", 0)
        if inp > 0:
            by_date[s.get("date", "")][0] += cache
            by_date[s.get("date", "")][1] += inp
    dates = sorted(by_date.keys())
    if not dates:
        return ""
    ratios = [by_date[d][0] / max(by_date[d][1], 1) * 100 for d in dates]
    smoothed = [np.mean(ratios[max(0,i-2):i+1]) for i in range(len(ratios))]

    m = 45
    pw, ph = width - 2*m, height - 2*m
    mx = max(smoothed) if smoothed else 1
    if mx == 0:
        mx = 1

    points = " ".join(f"{m+i/max(len(smoothed)-1,1)*pw:.1f},{m+ph-v/mx*ph:.1f}" for i, v in enumerate(smoothed))
    return _svg(width, height,
        f'<polyline points="{points}" fill="none" stroke="{C["green"]}" stroke-width="2.5" stroke-linejoin="round"/>'
        f'<text x="{m+pw}" y="{m-4}" text-anchor="end" fill="{C["text"]}" font-size="11" font-weight="700">{smoothed[-1]:.0f}% cache hit</text>',
        "Cache Efficiency (% input from cache)")


# -----------------------------------------------------------------------
# 28. Conversation depth histogram
# -----------------------------------------------------------------------
def conversation_depth(sessions, width=500, height=200):
    depths = [s.get("total_msgs", 0) for s in sessions if s.get("total_msgs", 0) > 0]
    if not depths:
        return ""
    log_depths = np.log1p(depths)
    hist, edges = np.histogram(log_depths, bins=30)
    mx = max(hist) or 1

    m = 40
    pw, ph = width - 2*m, height - 2*m
    bar_w = pw / 30 - 1

    bars = ""
    for i, c in enumerate(hist):
        h = c / mx * ph
        x = m + i * (bar_w + 1)
        y = m + ph - h
        bars += f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" rx="1" fill="{C["purple"]}" opacity="0.7"/>'

    return _svg(width, height, bars, "Conversation Depth (messages per session, log)")


# -----------------------------------------------------------------------
# 29. Model usage pie
# -----------------------------------------------------------------------
def model_usage_donut(sessions, width=280, height=280):
    model_counts = Counter()
    for s in sessions:
        models = s.get("models_used", {})
        if isinstance(models, dict):
            for m, c in models.items():
                # Shorten model names
                short = m.split("-")[0] if m else "unknown"
                if "opus" in m.lower(): short = "Opus"
                elif "sonnet" in m.lower(): short = "Sonnet"
                elif "haiku" in m.lower(): short = "Haiku"
                model_counts[short] += c

    top = model_counts.most_common(6)
    if not top:
        return ""
    total = sum(c for _, c in top)

    cx, cy, r = width/2, height/2 + 5, 85
    circumference = 2 * math.pi * r
    arcs = ""
    offset = 0
    for i, (name, count) in enumerate(top):
        arc = count / total * circumference
        arcs += f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{PALETTE[i % len(PALETTE)]}" stroke-width="22" stroke-dasharray="{arc:.1f} {circumference:.1f}" stroke-dashoffset="-{offset:.1f}" transform="rotate(-90 {cx} {cy})"/>'
        offset += arc

    legend = ""
    for i, (name, count) in enumerate(top):
        ly = cy + r + 15 + i * 14
        pct = count / total * 100
        legend += f'<circle cx="{cx-60}" cy="{ly}" r="4" fill="{PALETTE[i % len(PALETTE)]}"/>'
        legend += f'<text x="{cx-50}" y="{ly+4}" fill="{C["text2"]}" font-size="9">{name} ({pct:.0f}%)</text>'

    return _svg(width, max(height, height + len(top) * 14), arcs +
        f'<text x="{cx}" y="{cy}" text-anchor="middle" fill="{C["text"]}" font-size="13" font-weight="700">Models</text>' + legend,
        "")


# -----------------------------------------------------------------------
# 30. Session gap analysis (time between sessions)
# -----------------------------------------------------------------------
def session_gaps(sessions, width=500, height=200):
    mtimes = sorted(s.get("mtime", 0) for s in sessions)
    if len(mtimes) < 10:
        return ""
    gaps = [(mtimes[i+1] - mtimes[i]) / 60 for i in range(len(mtimes)-1)]  # minutes
    gaps = [g for g in gaps if 0 < g < 1440]  # filter > 0 and < 24h
    if not gaps:
        return ""

    log_gaps = np.log1p(gaps)
    hist, edges = np.histogram(log_gaps, bins=30)
    mx = max(hist) or 1

    m = 40
    pw, ph = width - 2*m, height - 2*m
    bar_w = pw / 30 - 1

    bars = ""
    for i, c in enumerate(hist):
        h = c / mx * ph
        x = m + i * (bar_w + 1)
        y = m + ph - h
        bars += f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" rx="1" fill="{C["orange"]}" opacity="0.7"/>'

    median_gap = np.median(gaps)
    return _svg(width, height, bars +
        f'<text x="{m+pw}" y="{m-4}" text-anchor="end" fill="{C["text"]}" font-size="10" font-weight="700">median: {median_gap:.0f}min</text>',
        "Time Between Sessions (log minutes)")


# -----------------------------------------------------------------------
# 31. Night owl index over time
# -----------------------------------------------------------------------
def night_owl_trend(sessions, width=600, height=200):
    by_date = defaultdict(lambda: [0, 0])
    for s in sessions:
        by_date[s.get("date", "")][0] += 1
        if s.get("hour", 12) >= 22 or s.get("hour", 12) < 6:
            by_date[s.get("date", "")][1] += 1
    dates = sorted(by_date.keys())
    if len(dates) < 3:
        return ""
    rates = [by_date[d][1] / max(by_date[d][0], 1) * 100 for d in dates]
    smoothed = [np.mean(rates[max(0,i-2):i+1]) for i in range(len(rates))]

    m = 45
    pw, ph = width - 2*m, height - 2*m
    mx = max(smoothed) if smoothed else 1
    if mx == 0: mx = 1

    points = " ".join(f"{m+i/max(len(smoothed)-1,1)*pw:.1f},{m+ph-v/mx*ph:.1f}" for i, v in enumerate(smoothed))
    return _svg(width, height,
        f'<polyline points="{points}" fill="none" stroke="{C["purple"]}" stroke-width="2.5" stroke-linejoin="round"/>',
        "Night Owl Index (% sessions 10PM-6AM)")


# -----------------------------------------------------------------------
# 32. Marathon sessions (longest each day)
# -----------------------------------------------------------------------
def marathon_sessions(sessions, width=600, height=200):
    by_date = defaultdict(float)
    for s in sessions:
        d = s.get("duration_mins", 0)
        by_date[s.get("date", "")] = max(by_date[s.get("date", "")], d)
    dates = sorted(by_date.keys())
    if not dates:
        return ""
    values = [by_date[d] for d in dates]

    m = 45
    pw, ph = width - 2*m, height - 2*m
    mx = max(values) or 1

    bars = ""
    bar_w = max(pw / len(values) - 1, 1.5)
    for i, v in enumerate(values):
        h = v / mx * ph
        x = m + i * (bar_w + 1)
        y = m + ph - h
        color = C["rose"] if v > np.percentile(values, 90) else C["indigo"]
        bars += f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" rx="1" fill="{color}" opacity="0.7"/>'

    return _svg(width, height, bars +
        f'<text x="{m+pw}" y="{m-4}" text-anchor="end" fill="{C["text"]}" font-size="10" font-weight="700">peak: {max(values):.0f}min</text>',
        "Longest Session Each Day")


# -----------------------------------------------------------------------
# 33. Git branch activity (top branches)
# -----------------------------------------------------------------------
def branch_activity(sessions, width=500, height=280):
    branch_counts = Counter()
    for s in sessions:
        branches = s.get("git_branches", [])
        if isinstance(branches, list):
            for b in branches:
                if b and b != "main" and b != "master":
                    branch_counts[b] += 1

    top = branch_counts.most_common(12)
    if not top:
        return ""
    mx = top[0][1]

    m_left, m_top = 140, 35
    pw = width - m_left - 20
    row_h = (height - m_top - 10) / len(top)

    bars = ""
    for i, (name, count) in enumerate(top):
        y = m_top + i * row_h
        w = count / mx * pw
        color = PALETTE[i % len(PALETTE)]
        bars += f'<rect x="{m_left}" y="{y+2:.0f}" width="{w:.0f}" height="{row_h-4:.0f}" rx="3" fill="{color}" opacity="0.7"/>'
        bars += f'<text x="{m_left-6}" y="{y+row_h/2+4:.0f}" text-anchor="end" fill="{C["text2"]}" font-size="8" font-family="monospace">{name[:18]}</text>'
        bars += f'<text x="{m_left+w+6:.0f}" y="{y+row_h/2+4:.0f}" fill="{C["text2"]}" font-size="9">{count}</text>'

    return _svg(width, height, bars, "Git Branch Activity (excl main/master)")


# -----------------------------------------------------------------------
# 34. Project concentration (Gini-like over time)
# -----------------------------------------------------------------------
def project_concentration(sessions, width=600, height=200):
    by_week = defaultdict(lambda: Counter())
    for s in sessions:
        d = s.get("date", "")
        if not d:
            continue
        dt = datetime.strptime(d, "%Y-%m-%d")
        week = dt.strftime("%Y-W%W")
        by_week[week][s.get("repo", "?")] += 1

    weeks = sorted(by_week.keys())
    if len(weeks) < 2:
        return ""

    def gini(counts):
        arr = np.array(sorted(counts))
        n = len(arr)
        if n == 0 or arr.sum() == 0:
            return 0
        index = np.arange(1, n + 1)
        return (2 * (index * arr).sum() / (n * arr.sum())) - (n + 1) / n

    ginis = [gini(list(by_week[w].values())) for w in weeks]

    m = 45
    pw, ph = width - 2*m, height - 2*m
    mx = max(ginis) if ginis else 1
    if mx == 0: mx = 1

    points = " ".join(f"{m+i/max(len(ginis)-1,1)*pw:.1f},{m+ph-v/mx*ph:.1f}" for i, v in enumerate(ginis))
    return _svg(width, height,
        f'<polyline points="{points}" fill="none" stroke="{C["rose"]}" stroke-width="2.5" stroke-linejoin="round"/>'
        f'<line x1="{m}" y1="{m+ph/2}" x2="{m+pw}" y2="{m+ph/2}" stroke="{C["border"]}" stroke-dasharray="4 4"/>',
        "Project Concentration (Gini index per week)")


# -----------------------------------------------------------------------
# 35. First session time of day
# -----------------------------------------------------------------------
def first_session_time(sessions, width=500, height=200):
    by_date = defaultdict(lambda: 24)
    for s in sessions:
        h = s.get("hour", 12)
        d = s.get("date", "")
        by_date[d] = min(by_date[d], h)

    dates = sorted(by_date.keys())
    if len(dates) < 3:
        return ""
    values = [by_date[d] for d in dates]

    m = 45
    pw, ph = width - 2*m, height - 2*m

    dots = ""
    for i, v in enumerate(values):
        x = m + i / max(len(values)-1, 1) * pw
        y = m + v / 24 * ph  # 0=top (midnight), 24=bottom
        dots += f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{C["amber"]}" opacity="0.6"/>'

    # Y-axis labels
    labels = ""
    for h in [0, 6, 12, 18, 24]:
        y = m + h / 24 * ph
        labels += f'<text x="{m-6}" y="{y+4}" text-anchor="end" fill="{C["text2"]}" font-size="9">{h:02d}:00</text>'
        labels += f'<line x1="{m}" y1="{y}" x2="{m+pw}" y2="{y}" stroke="{C["border"]}" stroke-width="0.5"/>'

    return _svg(width, height, labels + dots, "First Session Start Time Each Day")


# -----------------------------------------------------------------------
# 36. Tool adoption timeline (when tools first appeared)
# -----------------------------------------------------------------------
def tool_adoption(sessions, width=600, height=300):
    tool_first = {}
    sorted_sessions = sorted(sessions, key=lambda s: s.get("date", ""))
    for s in sorted_sessions:
        tools = s.get("tool_uses", {})
        if not isinstance(tools, dict):
            continue
        d = s.get("date", "")
        for name in tools:
            if name not in tool_first:
                tool_first[name] = d

    if not tool_first:
        return ""

    # Get top 20 most-used tools, sorted by first appearance
    tool_counts = Counter()
    for s in sessions:
        tools = s.get("tool_uses", {})
        if isinstance(tools, dict):
            for name, count in tools.items():
                tool_counts[name] += count

    top_tools = [t for t, _ in tool_counts.most_common(20)]
    entries = [(t, tool_first.get(t, "")) for t in top_tools if t in tool_first]
    entries.sort(key=lambda x: x[1])

    if not entries:
        return ""

    all_dates = sorted(set(s.get("date", "") for s in sessions if s.get("date")))
    if not all_dates:
        return ""
    date_min = datetime.strptime(all_dates[0], "%Y-%m-%d")
    date_max = datetime.strptime(all_dates[-1], "%Y-%m-%d")
    date_range = (date_max - date_min).days or 1

    m_left, m_top = 110, 35
    pw = width - m_left - 20
    row_h = (height - m_top - 10) / len(entries)

    elems = ""
    for i, (name, first) in enumerate(entries):
        y = m_top + i * row_h
        x = m_left + (datetime.strptime(first, "%Y-%m-%d") - date_min).days / date_range * pw
        short = name.replace("mcp__", "").replace("__", ":")[:15]
        color = PALETTE[i % len(PALETTE)]
        elems += f'<circle cx="{x:.0f}" cy="{y+row_h/2:.0f}" r="5" fill="{color}"/>'
        elems += f'<line x1="{x:.0f}" y1="{y+row_h/2:.0f}" x2="{m_left+pw}" y2="{y+row_h/2:.0f}" stroke="{color}" stroke-width="1" opacity="0.3"/>'
        elems += f'<text x="{m_left-6}" y="{y+row_h/2+4:.0f}" text-anchor="end" fill="{C["text2"]}" font-size="8" font-family="monospace">{short}</text>'

    return _svg(width, height, elems, "Tool Adoption Timeline (first use)")


# -----------------------------------------------------------------------
# 37. Tokens by project treemap approximation (grid)
# -----------------------------------------------------------------------
def token_treemap(sessions, width=600, height=300):
    by_proj = Counter()
    for s in sessions:
        by_proj[s.get("repo", "?")] += s.get("total_tokens", 0)

    top = by_proj.most_common(20)
    if not top:
        return ""
    total = sum(c for _, c in top)

    m = 10
    cells = ""
    x, y = m, m + 25
    row_h = 50
    row_w = width - 2 * m

    for i, (name, count) in enumerate(top):
        pct = count / total
        w = max(pct * row_w, 30)
        if x + w > width - m:
            x = m
            y += row_h + 4
        color = PALETTE[i % len(PALETTE)]
        cells += f'<rect x="{x:.0f}" y="{y:.0f}" width="{w:.0f}" height="{row_h}" rx="6" fill="{color}" opacity="0.6"/>'
        if w > 40:
            cells += f'<text x="{x+w/2:.0f}" y="{y+row_h/2-4:.0f}" text-anchor="middle" fill="{C["text"]}" font-size="9" font-weight="700">{name[:12]}</text>'
            cells += f'<text x="{x+w/2:.0f}" y="{y+row_h/2+10:.0f}" text-anchor="middle" fill="{C["text2"]}" font-size="8">{pct*100:.0f}%</text>'
        x += w + 4

    return _svg(width, min(y + row_h + 15, height + 100), cells, "Token Distribution by Project")


# -----------------------------------------------------------------------
# 38. Session intensity heatmap (sessions × tokens)
# -----------------------------------------------------------------------
def intensity_scatter(sessions, width=600, height=250):
    m = 45
    pw, ph = width - 2*m, height - 2*m

    pts = []
    for s in sessions:
        dur = s.get("duration_mins", 0)
        tokens = s.get("total_tokens", 0)
        if dur > 0.5 and tokens > 0:
            pts.append((dur, tokens))

    if not pts:
        return ""
    arr = np.array(pts)
    mx_x = np.percentile(arr[:, 0], 95)
    mx_y = np.percentile(arr[:, 1], 95)

    sample = np.random.choice(len(arr), min(600, len(arr)), replace=False)
    dots = ""
    for i in sample:
        dur, tok = arr[i]
        x = m + min(dur / mx_x, 1) * pw
        y = m + ph - min(tok / mx_y, 1) * ph
        r = 2 + min(tok / mx_y, 1) * 3
        dots += f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{C["indigo"]}" opacity="0.35"/>'

    return _svg(width, height, dots +
        f'<text x="{m+pw/2}" y="{m+ph+16}" text-anchor="middle" fill="{C["text2"]}" font-size="9">Duration (min)</text>'
        f'<text x="{m-8}" y="{m+ph/2}" text-anchor="middle" fill="{C["text2"]}" font-size="9" transform="rotate(-90 {m-8} {m+ph/2})">Tokens</text>',
        "Session Intensity (Duration × Tokens)")


# -----------------------------------------------------------------------
# 39. Weekly rhythm stacked area
# -----------------------------------------------------------------------
def weekly_stacked(sessions, width=600, height=220):
    by_week = defaultdict(lambda: defaultdict(int))
    for s in sessions:
        d = s.get("date", "")
        if not d:
            continue
        dt = datetime.strptime(d, "%Y-%m-%d")
        week = dt.strftime("%Y-W%W")
        by_week[week][s.get("repo", "?")] += 1

    weeks = sorted(by_week.keys())
    if len(weeks) < 2:
        return ""

    # Top 5 projects
    overall = Counter()
    for w in weeks:
        for p, c in by_week[w].items():
            overall[p] += c
    top5 = [p for p, _ in overall.most_common(5)]

    m = 45
    pw, ph = width - 2*m, height - 2*m

    # Build stacked areas from bottom up
    areas = ""
    prev = [0.0] * len(weeks)
    max_stack = max(sum(by_week[w].values()) for w in weeks)
    if max_stack == 0:
        max_stack = 1

    for pi, proj in enumerate(reversed(top5)):
        values = [by_week[w].get(proj, 0) for w in weeks]
        new_prev = [prev[i] + values[i] for i in range(len(weeks))]

        # Build polygon
        top_pts = []
        bot_pts = []
        for i in range(len(weeks)):
            x = m + i / max(len(weeks)-1, 1) * pw
            yt = m + ph - new_prev[i] / max_stack * ph
            yb = m + ph - prev[i] / max_stack * ph
            top_pts.append(f"{x:.1f},{yt:.1f}")
            bot_pts.append(f"{x:.1f},{yb:.1f}")

        poly = " ".join(top_pts) + " " + " ".join(reversed(bot_pts))
        color = PALETTE[(len(top5) - 1 - pi) % len(PALETTE)]
        areas += f'<polygon points="{poly}" fill="{color}" opacity="0.5"/>'
        prev = new_prev

    # Legend
    legend = ""
    for i, proj in enumerate(top5):
        lx = m + i * 110
        legend += f'<rect x="{lx}" y="{m+ph+8}" width="8" height="8" rx="2" fill="{PALETTE[i % len(PALETTE)]}"/>'
        legend += f'<text x="{lx+12}" y="{m+ph+16}" fill="{C["text2"]}" font-size="8">{proj[:12]}</text>'

    return _svg(width, height, areas + legend, "Weekly Stacked (Top 5 Projects)")


# -----------------------------------------------------------------------
# 40. Response ratio (how many user msgs per assistant msg)
# -----------------------------------------------------------------------
def response_ratio(sessions, width=600, height=200):
    by_date = defaultdict(lambda: [0, 0])
    for s in sessions:
        by_date[s.get("date", "")][0] += s.get("user_msgs", 0)
        by_date[s.get("date", "")][1] += s.get("assistant_msgs", 0)
    dates = sorted(by_date.keys())
    if not dates:
        return ""
    ratios = [by_date[d][1] / max(by_date[d][0], 1) for d in dates]
    smoothed = [np.mean(ratios[max(0,i-2):i+1]) for i in range(len(ratios))]

    m = 45
    pw, ph = width - 2*m, height - 2*m
    mx = max(smoothed) if smoothed else 1
    if mx == 0: mx = 1

    points = " ".join(f"{m+i/max(len(smoothed)-1,1)*pw:.1f},{m+ph-v/mx*ph:.1f}" for i, v in enumerate(smoothed))
    return _svg(width, height,
        f'<polyline points="{points}" fill="none" stroke="{C["blue"]}" stroke-width="2.5" stroke-linejoin="round"/>'
        f'<line x1="{m}" y1="{m+ph/2}" x2="{m+pw}" y2="{m+ph/2}" stroke="{C["border"]}" stroke-dasharray="4 4"/>'
        f'<text x="{m+pw}" y="{m-4}" text-anchor="end" fill="{C["text"]}" font-size="10" font-weight="700">{smoothed[-1]:.1f}x</text>',
        "Claude/You Response Ratio (3-day rolling)")


# -----------------------------------------------------------------------
# 41. Prompt sophistication — are your prompts getting longer?
# -----------------------------------------------------------------------
def prompt_evolution(sessions, width=600, height=200):
    """Track average user message length over time — proxy for prompt maturity."""
    by_date = defaultdict(list)
    for s in sessions:
        um = s.get("user_msgs", 0)
        tokens = s.get("input_tokens", 0)
        if um > 0 and tokens > 0:
            by_date[s.get("date", "")].append(tokens / um)
    dates = sorted(by_date.keys())
    if len(dates) < 5:
        return ""
    values = [np.mean(by_date[d]) for d in dates]
    smoothed = [np.mean(values[max(0,i-4):i+1]) for i in range(len(values))]

    m = 45
    pw, ph = width - 2*m, height - 2*m
    mx = max(smoothed) or 1

    points = " ".join(f"{m+i/max(len(smoothed)-1,1)*pw:.1f},{m+ph-v/mx*ph:.1f}" for i, v in enumerate(smoothed))
    # Trend arrow
    delta = smoothed[-1] - smoothed[0]
    trend = "longer" if delta > 0 else "shorter"

    return _svg(width, height,
        f'<polyline points="{points}" fill="none" stroke="{C["amber"]}" stroke-width="2.5" stroke-linejoin="round"/>'
        f'<text x="{m+pw}" y="{m-4}" text-anchor="end" fill="{C["text"]}" font-size="10" font-weight="700">{smoothed[-1]:.0f} tok/msg ({trend})</text>',
        "Prompt Sophistication (tokens per user message)")


# -----------------------------------------------------------------------
# 42. Focus score — inverse entropy of daily project mix
# -----------------------------------------------------------------------
def focus_score(sessions, width=600, height=200):
    """1.0 = laser focused on one project. 0.0 = scattered across many."""
    by_date = defaultdict(Counter)
    for s in sessions:
        by_date[s.get("date", "")][s.get("repo", "?")] += 1
    dates = sorted(by_date.keys())
    if len(dates) < 5:
        return ""

    scores = []
    for d in dates:
        counts = list(by_date[d].values())
        total = sum(counts)
        if total == 0:
            scores.append(1.0)
            continue
        probs = [c / total for c in counts]
        ent = -sum(p * math.log2(p) for p in probs if p > 0)
        max_ent = math.log2(len(counts)) if len(counts) > 1 else 1
        scores.append(1 - ent / max_ent if max_ent > 0 else 1.0)

    smoothed = [np.mean(scores[max(0,i-4):i+1]) for i in range(len(scores))]

    m = 45
    pw, ph = width - 2*m, height - 2*m

    # Color gradient from red (scattered) to green (focused)
    points = []
    for i, v in enumerate(smoothed):
        x = m + i / max(len(smoothed)-1,1) * pw
        y = m + ph - v * ph
        points.append(f"{x:.1f},{y:.1f}")

    fill_pts = f"{m},{m+ph} " + " ".join(points) + f" {m+pw},{m+ph}"
    return _svg(width, height,
        f'<defs><linearGradient id="fsg" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="{C["green"]}"/><stop offset="100%" stop-color="transparent"/></linearGradient></defs>'
        f'<polygon points="{fill_pts}" fill="url(#fsg)" opacity="0.3"/>'
        f'<polyline points="{" ".join(points)}" fill="none" stroke="{C["green"]}" stroke-width="2.5" stroke-linejoin="round"/>'
        f'<line x1="{m}" y1="{m+ph*0.5}" x2="{m+pw}" y2="{m+ph*0.5}" stroke="{C["border"]}" stroke-dasharray="4 4"/>'
        f'<text x="{m+pw}" y="{m-4}" text-anchor="end" fill="{C["text"]}" font-size="10" font-weight="700">Today: {smoothed[-1]:.0%}</text>',
        "Daily Focus Score (1.0 = one project, 0 = scattered)")


# -----------------------------------------------------------------------
# 43. Project gravity — which projects pull you back most?
# -----------------------------------------------------------------------
def project_gravity(sessions, width=600, height=300):
    """Measures return frequency: how many distinct days you revisit a project."""
    proj_days = defaultdict(set)
    for s in sessions:
        proj_days[s.get("repo", "?")].add(s.get("date", ""))

    data = [(repo, len(days)) for repo, days in proj_days.items() if len(days) > 1]
    data.sort(key=lambda x: x[1], reverse=True)
    top = data[:15]
    if not top:
        return ""

    mx = top[0][1]
    m_left, m_top = 130, 35
    pw = width - m_left - 40
    row_h = (height - m_top - 10) / len(top)

    bars = ""
    for i, (name, days) in enumerate(top):
        y = m_top + i * row_h
        w = days / mx * pw
        # Size of circle proportional to total sessions
        total = sum(1 for s in sessions if s.get("repo") == name)
        r = min(3 + total / 50, 12)
        color = PALETTE[i % len(PALETTE)]
        bars += f'<rect x="{m_left}" y="{y+row_h*0.25:.0f}" width="{w:.0f}" height="{row_h*0.5:.0f}" rx="4" fill="{color}" opacity="0.6"/>'
        bars += f'<circle cx="{m_left+w+r+4:.0f}" cy="{y+row_h/2:.0f}" r="{r:.0f}" fill="{color}" opacity="0.8"/>'
        bars += f'<text x="{m_left-6}" y="{y+row_h/2+4:.0f}" text-anchor="end" fill="{C["text2"]}" font-size="9">{name[:16]}</text>'
        bars += f'<text x="{m_left+w+r*2+10:.0f}" y="{y+row_h/2+4:.0f}" fill="{C["text2"]}" font-size="9">{days}d · {total} sessions</text>'

    return _svg(width, height, bars, "Project Gravity (distinct days you returned)")


# -----------------------------------------------------------------------
# 44. Momentum streaks — consecutive days on same project
# -----------------------------------------------------------------------
def momentum_streaks(sessions, width=600, height=250):
    """Find the longest consecutive-day streaks per project."""
    proj_dates = defaultdict(set)
    for s in sessions:
        proj_dates[s.get("repo", "?")].add(s.get("date", ""))

    streaks = []
    for repo, date_set in proj_dates.items():
        if len(date_set) < 2:
            continue
        sorted_dates = sorted(date_set)
        current_streak = 1
        best_streak = 1
        for i in range(1, len(sorted_dates)):
            prev = datetime.strptime(sorted_dates[i-1], "%Y-%m-%d")
            curr = datetime.strptime(sorted_dates[i], "%Y-%m-%d")
            if (curr - prev).days == 1:
                current_streak += 1
                best_streak = max(best_streak, current_streak)
            else:
                current_streak = 1
        if best_streak >= 2:
            streaks.append((repo, best_streak, len(date_set)))

    streaks.sort(key=lambda x: x[1], reverse=True)
    top = streaks[:12]
    if not top:
        return ""

    mx = top[0][1]
    m_left, m_top = 130, 35
    pw = width - m_left - 40
    row_h = (height - m_top - 10) / len(top)

    bars = ""
    for i, (name, streak, total_days) in enumerate(top):
        y = m_top + i * row_h
        w = streak / mx * pw
        color = PALETTE[i % len(PALETTE)]
        # Draw streak as connected dots
        dot_spacing = min(w / max(streak, 1), 14)
        for d in range(streak):
            dx = m_left + d * dot_spacing
            bars += f'<circle cx="{dx+6:.0f}" cy="{y+row_h/2:.0f}" r="4" fill="{color}" opacity="0.8"/>'
            if d > 0:
                bars += f'<line x1="{dx+6-dot_spacing:.0f}" y1="{y+row_h/2:.0f}" x2="{dx+6:.0f}" y2="{y+row_h/2:.0f}" stroke="{color}" stroke-width="2" opacity="0.4"/>'
        bars += f'<text x="{m_left-6}" y="{y+row_h/2+4:.0f}" text-anchor="end" fill="{C["text2"]}" font-size="9">{name[:16]}</text>'
        bars += f'<text x="{m_left+streak*dot_spacing+14:.0f}" y="{y+row_h/2+4:.0f}" fill="{C["text"]}" font-size="10" font-weight="700">{streak}d</text>'

    return _svg(width, height, bars, "Momentum Streaks (consecutive days on project)")


# -----------------------------------------------------------------------
# 45. Throughput by hour — tokens per minute, not just count
# -----------------------------------------------------------------------
def throughput_by_hour(sessions, width=500, height=220):
    """Which hours produce the most output per minute of work?"""
    by_hour = defaultdict(lambda: [0, 0])
    for s in sessions:
        dur = s.get("duration_mins", 0)
        tok = s.get("output_tokens", 0)
        if dur > 1 and tok > 0:
            h = s.get("hour", 0)
            by_hour[h][0] += tok
            by_hour[h][1] += dur

    if not by_hour:
        return ""
    rates = {h: by_hour[h][0] / max(by_hour[h][1], 1) for h in range(24)}
    mx = max(rates.values()) or 1

    m = 45
    pw, ph = width - 2*m, height - 2*m - 20
    bar_w = pw / 24 - 2

    bars = ""
    peak_h = max(rates, key=rates.get)
    for h in range(24):
        rate = rates.get(h, 0)
        bh = rate / mx * ph
        x = m + h * (bar_w + 2)
        y = m + 10 + ph - bh
        color = C["green"] if h == peak_h else C["indigo"]
        bars += f'<rect x="{x:.0f}" y="{y:.0f}" width="{bar_w:.0f}" height="{bh:.0f}" rx="2" fill="{color}" opacity="0.7"/>'

    labels = "".join(f'<text x="{m + h*(bar_w+2) + bar_w/2:.0f}" y="{m+10+ph+13}" text-anchor="middle" fill="{C["text2"]}" font-size="8">{h:02d}</text>' for h in range(0, 24, 3))
    return _svg(width, height, bars + labels +
        f'<text x="{m+pw}" y="{m}" text-anchor="end" fill="{C["green"]}" font-size="10" font-weight="700">Peak: {peak_h:02d}:00</text>',
        "Throughput (output tokens/min by hour)")


# -----------------------------------------------------------------------
# 46. Multi-tasking penalty — more projects/day = worse session quality?
# -----------------------------------------------------------------------
def multitask_penalty(sessions, width=500, height=250):
    """Scatter: projects touched that day vs avg tokens per session."""
    day_projects: dict[str, set] = defaultdict(set)
    day_tokens: dict[str, list] = defaultdict(list)
    day_sessions: dict[str, int] = defaultdict(int)
    for s in sessions:
        d = s.get("date", "")
        day_projects[d].add(s.get("repo", "?"))
        day_tokens[d].append(s.get("total_tokens", 0))
        day_sessions[d] += 1

    pts = []
    for d in day_projects:
        n_proj = len(day_projects[d])
        avg_tok = np.mean(day_tokens[d]) if day_tokens[d] else 0
        if avg_tok > 0:
            pts.append((n_proj, avg_tok, day_sessions[d]))

    if len(pts) < 5:
        return ""
    arr = np.array(pts)

    m = 50
    pw, ph = width - 2*m, height - 2*m
    mx_x = arr[:, 0].max()
    mx_y = np.percentile(arr[:, 1], 95)

    dots = ""
    for n_proj, avg_tok, n_sess in pts:
        x = m + (n_proj - 1) / max(mx_x - 1, 1) * pw
        y = m + ph - min(avg_tok / mx_y, 1) * ph
        r = min(2 + n_sess / 10, 8)
        dots += f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{C["rose"]}" opacity="0.4"/>'

    # Add correlation line
    corr = np.corrcoef(arr[:, 0], arr[:, 1])[0, 1]
    corr_label = f'r={corr:.2f} {"(penalty!)" if corr < -0.1 else "(no penalty)" if abs(corr) < 0.1 else "(bonus!)"}'

    labels = ""
    for i in range(1, int(mx_x) + 1, max(1, int(mx_x) // 6)):
        x = m + (i - 1) / max(mx_x - 1, 1) * pw
        labels += f'<text x="{x:.0f}" y="{m+ph+16}" text-anchor="middle" fill="{C["text2"]}" font-size="9">{i}</text>'

    return _svg(width, height, dots + labels +
        f'<text x="{m+pw}" y="{m-4}" text-anchor="end" fill="{C["text"]}" font-size="10" font-weight="700">{corr_label}</text>'
        f'<text x="{m+pw/2}" y="{m+ph+30}" text-anchor="middle" fill="{C["text2"]}" font-size="9">projects touched that day</text>',
        "Multi-tasking Effect (projects/day vs avg tokens)")


# -----------------------------------------------------------------------
# 47. Tool evolution — how tool mix changed over months
# -----------------------------------------------------------------------
def tool_evolution(sessions, width=600, height=250):
    """Stacked area of tool category usage by month."""
    cats_by_month = defaultdict(lambda: Counter())
    read_tools = {"Read", "Glob", "Grep", "LS"}
    write_tools = {"Edit", "Write", "NotebookEdit"}
    exec_tools = {"Bash", "BashOutput"}

    for s in sessions:
        month = s.get("month", "")
        tools = s.get("tool_uses", {})
        if not isinstance(tools, dict):
            continue
        for name, count in tools.items():
            if name in read_tools: cats_by_month[month]["Read"] += count
            elif name in write_tools: cats_by_month[month]["Write"] += count
            elif name in exec_tools: cats_by_month[month]["Exec"] += count
            elif name == "Agent": cats_by_month[month]["Agent"] += count
            elif name.startswith("mcp__"): cats_by_month[month]["MCP"] += count

    months = sorted(cats_by_month.keys())
    if len(months) < 2:
        return ""

    categories = ["Read", "Write", "Exec", "Agent", "MCP"]
    cat_colors = [C["cyan"], C["indigo"], C["green"], C["purple"], C["pink"]]

    m = 45
    pw, ph = width - 2*m, height - 2*m - 20
    max_total = max(sum(cats_by_month[mo].values()) for mo in months) or 1

    areas = ""
    prev = [0.0] * len(months)
    for ci, cat in enumerate(reversed(categories)):
        values = [cats_by_month[mo].get(cat, 0) for mo in months]
        new_prev = [prev[i] + values[i] for i in range(len(months))]

        top_pts = []
        bot_pts = []
        for i in range(len(months)):
            x = m + i / max(len(months)-1, 1) * pw
            yt = m + 10 + ph - new_prev[i] / max_total * ph
            yb = m + 10 + ph - prev[i] / max_total * ph
            top_pts.append(f"{x:.1f},{yt:.1f}")
            bot_pts.append(f"{x:.1f},{yb:.1f}")

        poly = " ".join(top_pts) + " " + " ".join(reversed(bot_pts))
        color = cat_colors[len(categories) - 1 - ci]
        areas += f'<polygon points="{poly}" fill="{color}" opacity="0.5"/>'
        prev = new_prev

    # X labels
    labels = "".join(f'<text x="{m+i/max(len(months)-1,1)*pw:.0f}" y="{m+10+ph+14}" text-anchor="middle" fill="{C["text2"]}" font-size="9">{mo[5:]}</text>' for i, mo in enumerate(months))
    # Legend
    legend = ""
    for i, cat in enumerate(categories):
        lx = m + i * 75
        legend += f'<rect x="{lx}" y="{m-2}" width="8" height="8" rx="2" fill="{cat_colors[i]}"/>'
        legend += f'<text x="{lx+12}" y="{m+6}" fill="{C["text2"]}" font-size="8">{cat}</text>'

    return _svg(width, height, areas + labels + legend, "")


# -----------------------------------------------------------------------
# 48. Recovery time — gap after marathon sessions
# -----------------------------------------------------------------------
def recovery_time(sessions, width=500, height=220):
    """After long sessions, how long before you start another?"""
    sorted_s = sorted(sessions, key=lambda s: s.get("first_ts", 0))
    pts = []
    for i in range(len(sorted_s) - 1):
        dur = sorted_s[i].get("duration_mins", 0)
        gap = (sorted_s[i+1].get("first_ts", 0) - sorted_s[i].get("last_ts", 0)) / 60
        if dur > 5 and 0 < gap < 1440:
            pts.append((dur, gap))

    if len(pts) < 20:
        return ""
    arr = np.array(pts)

    m = 50
    pw, ph = width - 2*m, height - 2*m
    mx_x = np.percentile(arr[:, 0], 95)
    mx_y = np.percentile(arr[:, 1], 95)

    sample = np.random.choice(len(arr), min(500, len(arr)), replace=False)
    dots = ""
    for i in sample:
        dur, gap = arr[i]
        x = m + min(dur / mx_x, 1) * pw
        y = m + ph - min(gap / mx_y, 1) * ph
        dots += f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.5" fill="{C["teal"]}" opacity="0.35"/>'

    corr = np.corrcoef(arr[:, 0], arr[:, 1])[0, 1]
    return _svg(width, height, dots +
        f'<text x="{m+pw/2}" y="{m+ph+16}" text-anchor="middle" fill="{C["text2"]}" font-size="9">Session duration (min)</text>'
        f'<text x="{m+pw}" y="{m-4}" text-anchor="end" fill="{C["text"]}" font-size="10" font-weight="700">r={corr:.2f}</text>',
        "Recovery Time (gap after session vs duration)")


# -----------------------------------------------------------------------
# 49. Project lifecycle — birth, peak, dormancy
# -----------------------------------------------------------------------
def project_lifecycle(sessions, width=700, height=300):
    """Sparklines showing each project's activity over time."""
    by_proj_date = defaultdict(Counter)
    for s in sessions:
        by_proj_date[s.get("repo", "?")][s.get("date", "")] += 1

    # Top 10 by total
    totals = {repo: sum(dates.values()) for repo, dates in by_proj_date.items()}
    top = sorted(totals, key=lambda r: totals[r], reverse=True)[:10]
    if not top:
        return ""

    all_dates = sorted(set(s.get("date", "") for s in sessions if s.get("date")))
    if not all_dates:
        return ""

    m_left, m_top = 130, 35
    pw = width - m_left - 20
    row_h = (height - m_top - 10) / len(top)

    lines = ""
    for pi, repo in enumerate(top):
        y_base = m_top + pi * row_h + row_h / 2
        counts = by_proj_date[repo]
        mx = max(counts.values()) if counts else 1

        # 7-day bucketed sparkline
        values = [counts.get(d, 0) for d in all_dates]
        # Bucket by week
        bucket_size = max(len(values) // 50, 1)
        bucketed = [sum(values[i:i+bucket_size]) for i in range(0, len(values), bucket_size)]
        mx_b = max(bucketed) if bucketed else 1
        if mx_b == 0: mx_b = 1

        points = []
        for i, v in enumerate(bucketed):
            x = m_left + i / max(len(bucketed)-1, 1) * pw
            y = y_base - v / mx_b * (row_h * 0.4)
            points.append(f"{x:.1f},{y:.1f}")

        fill_pts = f"{m_left},{y_base} " + " ".join(points) + f" {m_left+pw},{y_base}"
        color = PALETTE[pi % len(PALETTE)]
        lines += f'<polygon points="{fill_pts}" fill="{color}" opacity="0.2"/>'
        lines += f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="1.5" stroke-linejoin="round"/>'
        lines += f'<text x="{m_left-6}" y="{y_base+4:.0f}" text-anchor="end" fill="{C["text2"]}" font-size="9">{repo[:16]}</text>'

    return _svg(width, height, lines, "Project Lifecycles (activity sparklines)")


# -----------------------------------------------------------------------
# 50. Delegation index — Agent tool usage trend
# -----------------------------------------------------------------------
def delegation_index(sessions, width=600, height=200):
    """How much are you delegating to subagents over time?"""
    by_date = defaultdict(lambda: [0, 0])
    for s in sessions:
        tools = s.get("tool_uses", {})
        if not isinstance(tools, dict):
            continue
        total_tools = sum(tools.values())
        agent_tools = tools.get("Agent", 0)
        if total_tools > 0:
            by_date[s.get("date", "")][0] += agent_tools
            by_date[s.get("date", "")][1] += total_tools

    dates = sorted(by_date.keys())
    if len(dates) < 5:
        return ""
    ratios = [by_date[d][0] / max(by_date[d][1], 1) * 100 for d in dates]
    smoothed = [np.mean(ratios[max(0,i-4):i+1]) for i in range(len(ratios))]

    m = 45
    pw, ph = width - 2*m, height - 2*m
    mx = max(smoothed) if smoothed else 1
    if mx == 0: mx = 1

    points = []
    for i, v in enumerate(smoothed):
        x = m + i / max(len(smoothed)-1,1) * pw
        y = m + ph - v / mx * ph
        points.append(f"{x:.1f},{y:.1f}")

    fill_pts = f"{m},{m+ph} " + " ".join(points) + f" {m+pw},{m+ph}"
    return _svg(width, height,
        f'<defs><linearGradient id="delg" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="{C["purple"]}"/><stop offset="100%" stop-color="transparent"/></linearGradient></defs>'
        f'<polygon points="{fill_pts}" fill="url(#delg)" opacity="0.3"/>'
        f'<polyline points="{" ".join(points)}" fill="none" stroke="{C["purple"]}" stroke-width="2.5" stroke-linejoin="round"/>'
        f'<text x="{m+pw}" y="{m-4}" text-anchor="end" fill="{C["text"]}" font-size="10" font-weight="700">{smoothed[-1]:.0f}% delegation</text>',
        "Delegation Index (% of tool calls that are Agent)")


def _extract_org(path: str) -> str:
    """Extract org name from project path like ~/repos-{org}/{repo}."""
    parts = path.replace("~/", "").split("/")
    for p in parts:
        if p.startswith("repos"):
            return p.replace("repos-", "").replace("repos", "misc")
    return "other"


# -----------------------------------------------------------------------
# 51. Org breakdown — sessions by organization
# -----------------------------------------------------------------------
def org_breakdown(sessions, width=500, height=280):
    org_counts = Counter()
    org_tokens = Counter()
    for s in sessions:
        org = _extract_org(s.get("project_short", ""))
        org_counts[org] += 1
        org_tokens[org] += s.get("total_tokens", 0)

    top = org_counts.most_common(10)
    if not top:
        return ""
    total = sum(c for _, c in top)
    total_tok = sum(org_tokens.values()) or 1

    cx, cy, r = width/2, 130, 90
    circumference = 2 * math.pi * r
    arcs = ""
    offset = 0
    legend = ""
    for i, (org, count) in enumerate(top):
        arc = count / total * circumference
        color = PALETTE[i % len(PALETTE)]
        arcs += f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{color}" stroke-width="24" stroke-dasharray="{arc:.1f} {circumference:.1f}" stroke-dashoffset="-{offset:.1f}" transform="rotate(-90 {cx} {cy})"/>'
        offset += arc
        pct = count / total * 100
        tok_pct = org_tokens[org] / total_tok * 100
        ly = cy + r + 20 + i * 14
        legend += f'<circle cx="20" cy="{ly}" r="4" fill="{color}"/>'
        legend += f'<text x="30" y="{ly+4}" fill="{C["text2"]}" font-size="9">{org} — {count} sessions ({pct:.0f}%) · {tok_pct:.0f}% tokens</text>'

    arcs += f'<text x="{cx}" y="{cy-4}" text-anchor="middle" fill="{C["text"]}" font-size="18" font-weight="800">{len(set(_extract_org(s.get("project_short","")) for s in sessions))}</text>'
    arcs += f'<text x="{cx}" y="{cy+14}" text-anchor="middle" fill="{C["text2"]}" font-size="10">orgs</text>'

    return _svg(width, max(height, cy + r + 30 + len(top) * 14), arcs + legend, "")


# -----------------------------------------------------------------------
# 52. Org activity over time — stacked area
# -----------------------------------------------------------------------
def org_timeline(sessions, width=700, height=250):
    by_week_org = defaultdict(Counter)
    for s in sessions:
        d = s.get("date", "")
        if not d:
            continue
        dt = datetime.strptime(d, "%Y-%m-%d")
        week = dt.strftime("%Y-W%W")
        org = _extract_org(s.get("project_short", ""))
        by_week_org[week][org] += 1

    weeks = sorted(by_week_org.keys())
    if len(weeks) < 2:
        return ""

    # Top 5 orgs
    overall = Counter()
    for w in weeks:
        for o, c in by_week_org[w].items():
            overall[o] += c
    top5 = [o for o, _ in overall.most_common(5)]

    m = 45
    pw, ph = width - 2*m, height - 2*m - 20
    max_stack = max(sum(by_week_org[w].values()) for w in weeks) or 1

    areas = ""
    prev = [0.0] * len(weeks)
    for oi, org in enumerate(reversed(top5)):
        values = [by_week_org[w].get(org, 0) for w in weeks]
        new_prev = [prev[i] + values[i] for i in range(len(weeks))]
        top_pts = []
        bot_pts = []
        for i in range(len(weeks)):
            x = m + i / max(len(weeks)-1, 1) * pw
            yt = m + 10 + ph - new_prev[i] / max_stack * ph
            yb = m + 10 + ph - prev[i] / max_stack * ph
            top_pts.append(f"{x:.1f},{yt:.1f}")
            bot_pts.append(f"{x:.1f},{yb:.1f}")
        poly = " ".join(top_pts) + " " + " ".join(reversed(bot_pts))
        color = PALETTE[(len(top5) - 1 - oi) % len(PALETTE)]
        areas += f'<polygon points="{poly}" fill="{color}" opacity="0.5"/>'
        prev = new_prev

    legend = ""
    for i, org in enumerate(top5):
        lx = m + i * 110
        legend += f'<rect x="{lx}" y="{m-2}" width="8" height="8" rx="2" fill="{PALETTE[i % len(PALETTE)]}"/>'
        legend += f'<text x="{lx+12}" y="{m+6}" fill="{C["text2"]}" font-size="8">{org[:12]}</text>'

    # Month labels
    labels = ""
    for i, w in enumerate(weeks):
        if i % max(len(weeks) // 6, 1) == 0:
            x = m + i / max(len(weeks)-1, 1) * pw
            labels += f'<text x="{x:.0f}" y="{m+10+ph+14}" text-anchor="middle" fill="{C["text2"]}" font-size="9">{w[5:]}</text>'

    return _svg(width, height, areas + legend + labels, "Org Activity Over Time")


# -----------------------------------------------------------------------
# 53. Repo leaderboard — top repos with multi-metric bars
# -----------------------------------------------------------------------
def repo_leaderboard(sessions, width=700, height=400):
    repo_sessions: dict[str, int] = Counter()
    repo_tokens: dict[str, int] = Counter()
    repo_tools: dict[str, int] = Counter()
    repo_days: dict[str, set] = defaultdict(set)
    repo_hours: dict[str, float] = defaultdict(float)
    for s in sessions:
        repo = s.get("repo", "?")
        repo_sessions[repo] += 1
        repo_tokens[repo] += s.get("total_tokens", 0)
        repo_tools[repo] += s.get("tool_use_total", 0)
        repo_days[repo].add(s.get("date", ""))
        repo_hours[repo] += s.get("duration_mins", 0) / 60

    top_repos = sorted(repo_sessions, key=lambda r: repo_sessions[r], reverse=True)[:15]
    top = [(r, {"sessions": repo_sessions[r], "tokens": repo_tokens[r], "tools": repo_tools[r], "days": repo_days[r], "hours": repo_hours[r]}) for r in top_repos]
    if not top:
        return ""

    m_left, m_top = 130, 35
    pw = width - m_left - 20
    row_h = (height - m_top - 10) / len(top)
    mx_sessions = max(d["sessions"] for _, d in top)
    mx_tokens = max(d["tokens"] for _, d in top) or 1

    bars = ""
    for i, (name, d) in enumerate(top):
        y = m_top + i * row_h
        # Session bar (full width reference)
        w_sess = d["sessions"] / mx_sessions * pw * 0.6
        w_tok = d["tokens"] / mx_tokens * pw * 0.6
        color = PALETTE[i % len(PALETTE)]

        bars += f'<rect x="{m_left}" y="{y+2:.0f}" width="{w_sess:.0f}" height="{row_h*0.4:.0f}" rx="3" fill="{color}" opacity="0.7"/>'
        bars += f'<rect x="{m_left}" y="{y+row_h*0.5:.0f}" width="{w_tok:.0f}" height="{row_h*0.3:.0f}" rx="2" fill="{color}" opacity="0.35"/>'
        bars += f'<text x="{m_left-6}" y="{y+row_h/2+4:.0f}" text-anchor="end" fill="{C["text2"]}" font-size="9">{name[:16]}</text>'

        # Stats at end
        days = len(d["days"])
        bars += f'<text x="{m_left+pw}" y="{y+row_h*0.35:.0f}" text-anchor="end" fill="{C["text2"]}" font-size="8">{d["sessions"]} sess · {days}d · {d["hours"]:.0f}h · {d["tokens"]/1e6:.1f}M tok</text>'

    return _svg(width, height, bars +
        f'<rect x="{m_left}" y="{m_top-16}" width="8" height="8" rx="2" fill="{C["indigo"]}"/>'
        f'<text x="{m_left+12}" y="{m_top-8}" fill="{C["text2"]}" font-size="8">sessions</text>'
        f'<rect x="{m_left+80}" y="{m_top-16}" width="8" height="8" rx="2" fill="{C["indigo"]}" opacity="0.35"/>'
        f'<text x="{m_left+92}" y="{m_top-8}" fill="{C["text2"]}" font-size="8">tokens</text>',
        "Repo Leaderboard")


# -----------------------------------------------------------------------
# 54. Org × hour heatmap
# -----------------------------------------------------------------------
def org_hour_heatmap(sessions, width=600, height=250):
    grid = defaultdict(int)
    all_orgs = Counter()
    for s in sessions:
        org = _extract_org(s.get("project_short", ""))
        all_orgs[org] += 1
        grid[(org, s.get("hour", 0))] += 1

    top_orgs = [o for o, _ in all_orgs.most_common(7)]
    if not top_orgs:
        return ""

    mx = max(grid.values()) if grid else 1
    m_left, m_top = 100, 35
    cw = (width - m_left - 20) / 24
    ch = (height - m_top - 20) / len(top_orgs)

    cells = ""
    for oi, org in enumerate(top_orgs):
        for h in range(24):
            v = grid.get((org, h), 0)
            alpha = 0.05 + 0.95 * (v / mx)
            x = m_left + h * cw
            y = m_top + oi * ch
            cells += f'<rect x="{x:.1f}" y="{y:.1f}" width="{cw-1:.1f}" height="{ch-1:.1f}" rx="3" fill="rgba(99,102,241,{alpha:.2f})"/>'
        cells += f'<text x="{m_left-6}" y="{m_top + oi*ch + ch/2 + 4}" text-anchor="end" fill="{C["text2"]}" font-size="9">{org[:12]}</text>'

    labels = ""
    for h in range(0, 24, 3):
        labels += f'<text x="{m_left + h*cw + cw/2}" y="{m_top - 6}" text-anchor="middle" fill="{C["text2"]}" font-size="9">{h:02d}</text>'

    return _svg(width, height, cells + labels, "Org × Hour Heatmap")


# -----------------------------------------------------------------------
# 55. Repo churn — new repos appearing per week
# -----------------------------------------------------------------------
def repo_churn(sessions, width=600, height=200):
    first_seen = {}
    for s in sorted(sessions, key=lambda x: x.get("date", "")):
        repo = s.get("repo", "?")
        d = s.get("date", "")
        if repo not in first_seen and d:
            first_seen[repo] = d

    if not first_seen:
        return ""

    by_week = Counter()
    for repo, date in first_seen.items():
        try:
            dt = datetime.strptime(date, "%Y-%m-%d")
            by_week[dt.strftime("%Y-W%W")] += 1
        except ValueError:
            pass

    weeks = sorted(by_week.keys())
    if len(weeks) < 2:
        return ""
    values = [by_week[w] for w in weeks]
    cumulative = []
    total = 0
    for v in values:
        total += v
        cumulative.append(total)

    m = 45
    pw, ph = width - 2*m, height - 2*m

    # Bar for new repos per week
    mx = max(values) or 1
    bar_w = max(pw / len(weeks) - 2, 2)
    bars = ""
    for i, v in enumerate(values):
        h = v / mx * ph * 0.7
        x = m + i * (bar_w + 2)
        y = m + ph - h
        bars += f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" rx="2" fill="{C["cyan"]}" opacity="0.6"/>'

    # Cumulative line overlay
    mx_c = max(cumulative) or 1
    points = " ".join(f"{m+i*(bar_w+2)+bar_w/2:.1f},{m+ph-v/mx_c*ph:.1f}" for i, v in enumerate(cumulative))
    line = f'<polyline points="{points}" fill="none" stroke="{C["amber"]}" stroke-width="2" stroke-linejoin="round"/>'

    return _svg(width, height, bars + line +
        f'<text x="{m+pw}" y="{m-4}" text-anchor="end" fill="{C["text"]}" font-size="10" font-weight="700">{total} repos discovered</text>',
        "Repo Discovery Rate (new repos per week + cumulative)")


# -----------------------------------------------------------------------
# 56. Repo diversity by org (treemap-style)
# -----------------------------------------------------------------------
def org_repo_treemap(sessions, width=600, height=300):
    org_repos = defaultdict(Counter)
    for s in sessions:
        org = _extract_org(s.get("project_short", ""))
        org_repos[org][s.get("repo", "?")] += 1

    top_orgs = sorted(org_repos, key=lambda o: sum(org_repos[o].values()), reverse=True)[:6]
    if not top_orgs:
        return ""

    m = 10
    y = m + 25
    row_h = 42
    row_w = width - 2 * m
    cells = ""

    for oi, org in enumerate(top_orgs):
        repos = org_repos[org].most_common(8)
        total = sum(c for _, c in repos)
        x = m
        color = PALETTE[oi % len(PALETTE)]

        # Org label
        cells += f'<text x="{m}" y="{y-4}" fill="{C["text"]}" font-size="10" font-weight="700">{org}</text>'

        for ri, (repo, count) in enumerate(repos):
            pct = count / total
            w = max(pct * row_w, 25)
            if x + w > width - m:
                break
            cells += f'<rect x="{x:.0f}" y="{y:.0f}" width="{w:.0f}" height="{row_h}" rx="4" fill="{color}" opacity="{0.3 + 0.5 * pct:.2f}"/>'
            if w > 40:
                cells += f'<text x="{x+w/2:.0f}" y="{y+row_h/2-2:.0f}" text-anchor="middle" fill="{C["text"]}" font-size="8" font-weight="600">{repo[:10]}</text>'
                cells += f'<text x="{x+w/2:.0f}" y="{y+row_h/2+10:.0f}" text-anchor="middle" fill="{C["text2"]}" font-size="7">{count}</text>'
            x += w + 3

        y += row_h + 20

    return _svg(width, min(y + 10, 500), cells, "Org → Repo Map")


# -----------------------------------------------------------------------
# 57. Cross-org context switches
# -----------------------------------------------------------------------
def org_switches(sessions, width=500, height=200):
    """How often do you switch between orgs in a single day?"""
    by_date = defaultdict(list)
    for s in sorted(sessions, key=lambda s: s.get("first_ts", 0)):
        org = _extract_org(s.get("project_short", ""))
        by_date[s.get("date", "")].append(org)

    dates = sorted(by_date.keys())
    if len(dates) < 3:
        return ""

    switches = []
    for d in dates:
        orgs_today = by_date[d]
        sw = sum(1 for i in range(1, len(orgs_today)) if orgs_today[i] != orgs_today[i-1])
        switches.append(sw)

    smoothed = [np.mean(switches[max(0,i-4):i+1]) for i in range(len(switches))]

    m = 45
    pw, ph = width - 2*m, height - 2*m
    mx = max(smoothed) if smoothed else 1
    if mx == 0: mx = 1

    points = " ".join(f"{m+i/max(len(smoothed)-1,1)*pw:.1f},{m+ph-v/mx*ph:.1f}" for i, v in enumerate(smoothed))
    avg = np.mean(switches)
    return _svg(width, height,
        f'<polyline points="{points}" fill="none" stroke="{C["rose"]}" stroke-width="2.5" stroke-linejoin="round"/>'
        f'<text x="{m+pw}" y="{m-4}" text-anchor="end" fill="{C["text"]}" font-size="10" font-weight="700">avg {avg:.1f} switches/day</text>',
        "Cross-Org Context Switches per Day")


# -----------------------------------------------------------------------
# 58. Human-equivalent output per day
# -----------------------------------------------------------------------
def human_output(sessions, width=700, height=280):
    """Estimate daily human-equivalent output.

    Methodology:
    - Each Edit/Write tool call ≈ a code change (avg dev: ~50 meaningful changes/day)
    - Output tokens / 750 ≈ pages of prose
    - Bash calls ≈ commands run
    - Weighted composite: 1 edit = 1 unit, 1 page = 0.5 units, 1 bash = 0.3 units
    - Industry benchmark: senior dev ≈ 40-80 "units"/day
    """
    by_date = defaultdict(lambda: {"edits": 0, "pages": 0, "commands": 0, "hours": 0.0})
    edit_tools = {"Edit", "Write", "NotebookEdit"}
    exec_tools = {"Bash", "BashOutput"}

    for s in sessions:
        d = s.get("date", "")
        tools = s.get("tool_uses", {})
        if not isinstance(tools, dict):
            continue
        info = by_date[d]
        for name, count in tools.items():
            if name in edit_tools:
                info["edits"] += count
            elif name in exec_tools:
                info["commands"] += count
        info["pages"] += s.get("output_tokens", 0) / 750
        info["hours"] += s.get("duration_mins", 0) / 60

    dates = sorted(by_date.keys())
    if len(dates) < 3:
        return ""

    # Composite score
    scores = []
    for d in dates:
        info = by_date[d]
        score = info["edits"] * 1.0 + info["pages"] * 0.5 + info["commands"] * 0.3
        scores.append(score)

    # Human equivalent: senior dev ≈ 60 units/day
    human_equiv = [s / 60 for s in scores]
    smoothed = [np.mean(human_equiv[max(0,i-2):i+1]) for i in range(len(human_equiv))]

    m = 50
    pw, ph = width - 2*m, height - 2*m - 40

    # Bars for raw daily output
    mx = max(human_equiv) or 1
    bar_w = max(pw / len(dates) - 1, 1.5)
    bars = ""
    for i, v in enumerate(human_equiv):
        h = min(v / mx, 1) * ph
        x = m + i * (bar_w + 1)
        y = m + 20 + ph - h
        color = C["green"] if v >= 1.0 else C["amber"] if v >= 0.5 else C["indigo"]
        bars += f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" rx="1" fill="{color}" opacity="0.6"/>'

    # Smoothed line
    points = " ".join(f"{m+i*(bar_w+1)+bar_w/2:.1f},{m+20+ph-min(v/mx,1)*ph:.1f}" for i, v in enumerate(smoothed))
    line = f'<polyline points="{points}" fill="none" stroke="{C["text"]}" stroke-width="2" stroke-linejoin="round"/>'

    # 1x human line
    if mx > 0:
        y_1x = m + 20 + ph - 1.0 / mx * ph
        ref_line = f'<line x1="{m}" y1="{y_1x:.0f}" x2="{m+pw}" y2="{y_1x:.0f}" stroke="{C["green"]}" stroke-dasharray="6 3" opacity="0.5"/>'
        ref_label = f'<text x="{m+pw+4}" y="{y_1x+4:.0f}" fill="{C["green"]}" font-size="9" font-weight="700">1x dev</text>'
    else:
        ref_line = ref_label = ""

    avg_he = np.mean(human_equiv)
    total_days = sum(1 for v in human_equiv if v > 0)
    peak = max(human_equiv)

    stats = (
        f'<text x="{m}" y="{m+20+ph+20}" fill="{C["text2"]}" font-size="10">'
        f'Avg: <tspan fill="{C["text"]}" font-weight="700">{avg_he:.1f}x</tspan> dev/day · '
        f'Peak: <tspan fill="{C["amber"]}" font-weight="700">{peak:.1f}x</tspan> · '
        f'{total_days} active days · '
        f'Total: <tspan fill="{C["green"]}" font-weight="700">{sum(human_equiv):.0f}</tspan> dev-days equivalent'
        f'</text>'
    )

    return _svg(width, height, bars + line + ref_line + ref_label + stats,
        "Human-Equivalent Output per Day (1x = senior dev)")


# -----------------------------------------------------------------------
# 59. Output composition — what kind of work each day
# -----------------------------------------------------------------------
def output_composition(sessions, width=600, height=220):
    """Stacked area: edits vs prose vs commands per day."""
    by_date = defaultdict(lambda: [0, 0, 0])  # edits, pages, commands
    edit_tools = {"Edit", "Write", "NotebookEdit"}
    exec_tools = {"Bash", "BashOutput"}

    for s in sessions:
        d = s.get("date", "")
        tools = s.get("tool_uses", {})
        if not isinstance(tools, dict):
            continue
        for name, count in tools.items():
            if name in edit_tools:
                by_date[d][0] += count
            elif name in exec_tools:
                by_date[d][2] += count
        by_date[d][1] += int(s.get("output_tokens", 0) / 750)

    dates = sorted(by_date.keys())
    if len(dates) < 3:
        return ""

    categories = ["Code Changes", "Pages Written", "Commands Run"]
    cat_colors = [C["indigo"], C["cyan"], C["green"]]

    m = 45
    pw, ph = width - 2*m, height - 2*m - 20
    max_total = max(sum(by_date[d]) for d in dates) or 1

    areas = ""
    prev = [0] * len(dates)
    for ci in reversed(range(3)):
        values = [by_date[d][ci] for d in dates]
        new_prev = [prev[i] + values[i] for i in range(len(dates))]
        top_pts = []
        bot_pts = []
        for i in range(len(dates)):
            x = m + i / max(len(dates)-1, 1) * pw
            yt = m + 10 + ph - new_prev[i] / max_total * ph
            yb = m + 10 + ph - prev[i] / max_total * ph
            top_pts.append(f"{x:.1f},{yt:.1f}")
            bot_pts.append(f"{x:.1f},{yb:.1f}")
        poly = " ".join(top_pts) + " " + " ".join(reversed(bot_pts))
        areas += f'<polygon points="{poly}" fill="{cat_colors[ci]}" opacity="0.5"/>'
        prev = new_prev

    legend = ""
    for i, cat in enumerate(categories):
        lx = m + i * 130
        legend += f'<rect x="{lx}" y="{m-2}" width="8" height="8" rx="2" fill="{cat_colors[i]}"/>'
        legend += f'<text x="{lx+12}" y="{m+6}" fill="{C["text2"]}" font-size="8">{cat}</text>'

    return _svg(width, height, areas + legend, "")


# -----------------------------------------------------------------------
# 60. Leverage points — where tech debt / practice investment pays off
# -----------------------------------------------------------------------
def leverage_points(sessions, width=700, height=400):
    """Identify projects where investment would yield highest returns.

    Signals:
    - High revisit + short sessions = debugging loops (tech debt)
    - Read/write ratio > 5 = too much exploration (unclear code)
    - Duration increasing over time = growing complexity
    - Many sessions, few edits = overhead-heavy workflow
    """
    proj_data = defaultdict(lambda: {
        "sessions": 0, "total_dur": 0.0, "reads": 0, "writes": 0,
        "revisit_days": set(), "short_sessions": 0,
        "durations": [], "dates": [], "edits": 0, "tokens": 0
    })
    read_tools = {"Read", "Glob", "Grep", "LS"}
    write_tools = {"Edit", "Write", "NotebookEdit"}

    for s in sessions:
        repo = s.get("repo", "?")
        d = proj_data[repo]
        d["sessions"] += 1
        d["total_dur"] += s.get("duration_mins", 0)
        d["revisit_days"].add(s.get("date", ""))
        d["tokens"] += s.get("total_tokens", 0)
        dur = s.get("duration_mins", 0)
        if 0 < dur < 5:
            d["short_sessions"] += 1
        d["durations"].append(dur)
        d["dates"].append(s.get("date", ""))

        tools = s.get("tool_uses", {})
        if isinstance(tools, dict):
            for name, count in tools.items():
                if name in read_tools:
                    d["reads"] += count
                elif name in write_tools:
                    d["writes"] += count
                    d["edits"] += count

    # Score each project
    scored = []
    for repo, d in proj_data.items():
        if d["sessions"] < 5:
            continue

        signals = []
        score = 0

        # Debug loop signal: high revisit + many short sessions
        short_ratio = d["short_sessions"] / d["sessions"]
        if short_ratio > 0.3 and len(d["revisit_days"]) > 3:
            signals.append(f"Debug loops ({d['short_sessions']} quick sessions)")
            score += short_ratio * 30

        # Exploration overhead: read/write ratio
        rw_ratio = d["reads"] / max(d["writes"], 1)
        if rw_ratio > 8 and d["reads"] > 50:
            signals.append(f"High read/write ratio ({rw_ratio:.0f}:1)")
            score += min(rw_ratio / 2, 20)

        # Complexity growth: duration trend
        if len(d["durations"]) > 10:
            first_half = np.mean(d["durations"][:len(d["durations"])//2])
            second_half = np.mean(d["durations"][len(d["durations"])//2:])
            if second_half > first_half * 1.5 and second_half > 10:
                signals.append(f"Sessions getting longer ({first_half:.0f}→{second_half:.0f}min)")
                score += 15

        # Low edit efficiency: many sessions, few edits
        edit_per_session = d["edits"] / d["sessions"]
        if edit_per_session < 2 and d["sessions"] > 10:
            signals.append(f"Low edit rate ({edit_per_session:.1f}/session)")
            score += 10

        # Weight by total investment
        hours = d["total_dur"] / 60
        score *= (1 + min(hours / 20, 2))  # projects with more time invested matter more

        if signals:
            scored.append((repo, score, signals, d["sessions"], hours))

    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:10]
    if not top:
        return _svg(width, 60, f'<text x="{width/2}" y="35" text-anchor="middle" fill="{C["green"]}" font-size="12">No significant leverage points detected</text>', "")

    mx = top[0][1]
    m_left, m_top = 130, 40
    pw = width - m_left - 20
    row_h = (height - m_top - 10) / len(top)

    bars = ""
    for i, (name, score, signals, n_sess, hours) in enumerate(top):
        y = m_top + i * row_h
        w = score / mx * pw * 0.5
        urgency_color = C["rose"] if score > mx * 0.7 else C["amber"] if score > mx * 0.3 else C["cyan"]

        bars += f'<rect x="{m_left}" y="{y+2:.0f}" width="{w:.0f}" height="{row_h*0.4:.0f}" rx="4" fill="{urgency_color}" opacity="0.7"/>'
        bars += f'<text x="{m_left-6}" y="{y+row_h*0.3:.0f}" text-anchor="end" fill="{C["text"]}" font-size="9" font-weight="700">{name[:16]}</text>'
        bars += f'<text x="{m_left}" y="{y+row_h*0.7:.0f}" fill="{C["text2"]}" font-size="8">{" · ".join(signals[:2])} — {n_sess} sessions, {hours:.0f}h invested</text>'

    return _svg(width, height, bars, "Leverage Points (where practice/debt investment pays off)")


# -----------------------------------------------------------------------
# 61. Efficiency trend — edits per hour over time
# -----------------------------------------------------------------------
def efficiency_trend(sessions, width=600, height=200):
    """Are you getting more efficient? Edits per hour of session time."""
    by_date = defaultdict(lambda: [0, 0.0])  # edits, hours
    edit_tools = {"Edit", "Write", "NotebookEdit"}

    for s in sessions:
        d = s.get("date", "")
        tools = s.get("tool_uses", {})
        if isinstance(tools, dict):
            for name, count in tools.items():
                if name in edit_tools:
                    by_date[d][0] += count
        by_date[d][1] += s.get("duration_mins", 0) / 60

    dates = sorted(by_date.keys())
    if len(dates) < 5:
        return ""

    rates = [by_date[d][0] / max(by_date[d][1], 0.1) for d in dates]
    smoothed = [np.mean(rates[max(0,i-4):i+1]) for i in range(len(rates))]

    m = 45
    pw, ph = width - 2*m, height - 2*m
    mx = max(smoothed) or 1

    points = " ".join(f"{m+i/max(len(smoothed)-1,1)*pw:.1f},{m+ph-v/mx*ph:.1f}" for i, v in enumerate(smoothed))

    delta = smoothed[-1] - smoothed[0]
    trend_word = "improving" if delta > 0 else "declining"
    trend_color = C["green"] if delta > 0 else C["rose"]

    return _svg(width, height,
        f'<polyline points="{points}" fill="none" stroke="{trend_color}" stroke-width="2.5" stroke-linejoin="round"/>'
        f'<text x="{m+pw}" y="{m-4}" text-anchor="end" fill="{trend_color}" font-size="10" font-weight="700">{smoothed[-1]:.1f} edits/hr ({trend_word})</text>',
        "Edit Efficiency (code changes per hour, 5-day rolling)")


# -----------------------------------------------------------------------
# 62. Cumulative human-days
# -----------------------------------------------------------------------
def cumulative_human_days(sessions, width=600, height=220):
    """Running total of human-equivalent dev-days produced."""
    by_date = defaultdict(lambda: {"edits": 0, "pages": 0.0, "commands": 0})
    edit_tools = {"Edit", "Write", "NotebookEdit"}
    exec_tools = {"Bash", "BashOutput"}

    for s in sessions:
        d = s.get("date", "")
        tools = s.get("tool_uses", {})
        if isinstance(tools, dict):
            for name, count in tools.items():
                if name in edit_tools:
                    by_date[d]["edits"] += count
                elif name in exec_tools:
                    by_date[d]["commands"] += count
        by_date[d]["pages"] += s.get("output_tokens", 0) / 750

    dates = sorted(by_date.keys())
    if not dates:
        return ""

    cumulative = []
    total = 0.0
    for d in dates:
        info = by_date[d]
        score = info["edits"] * 1.0 + info["pages"] * 0.5 + info["commands"] * 0.3
        total += score / 60  # 60 units = 1 dev-day
        cumulative.append(total)

    m = 45
    pw, ph = width - 2*m, height - 2*m
    mx = max(cumulative) or 1

    points = []
    for i, v in enumerate(cumulative):
        x = m + i / max(len(cumulative)-1, 1) * pw
        y = m + ph - v / mx * ph
        points.append(f"{x:.1f},{y:.1f}")

    fill_pts = f"{m},{m+ph} " + " ".join(points) + f" {m+pw},{m+ph}"
    return _svg(width, height,
        f'<defs><linearGradient id="chd" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="{C["green"]}"/><stop offset="100%" stop-color="transparent"/></linearGradient></defs>'
        f'<polygon points="{fill_pts}" fill="url(#chd)" opacity="0.3"/>'
        f'<polyline points="{" ".join(points)}" fill="none" stroke="{C["green"]}" stroke-width="2.5" stroke-linejoin="round"/>'
        f'<text x="{m+pw}" y="{m}" text-anchor="end" fill="{C["text"]}" font-size="14" font-weight="800">{total:.0f} dev-days</text>',
        "Cumulative Human-Equivalent Output")


# -----------------------------------------------------------------------
# 63. Engineer Score — daily composite with moving average
# -----------------------------------------------------------------------
def engineer_score(sessions, width=700, height=300):
    """Daily Engineer Score (0-100) combining multiple signals.

    Components (weighted):
    - Output volume: edits + code changes (25%)
    - Throughput: output tokens per hour of work (20%)
    - Focus: inverse project entropy for the day (15%)
    - Depth: avg session duration > 15min (15%)
    - Craft: write/read ratio — are you producing, not just exploring? (15%)
    - Momentum: consecutive active days streak bonus (10%)
    """
    edit_tools = {"Edit", "Write", "NotebookEdit"}
    read_tools = {"Read", "Glob", "Grep", "LS"}
    exec_tools = {"Bash", "BashOutput"}

    es_edits: dict[str, int] = defaultdict(int)
    es_reads: dict[str, int] = defaultdict(int)
    es_out_tokens: dict[str, int] = defaultdict(int)
    es_hours: dict[str, float] = defaultdict(float)
    es_projects: dict[str, set] = defaultdict(set)
    es_durations: dict[str, list] = defaultdict(list)
    es_sessions: dict[str, int] = defaultdict(int)

    for s in sessions:
        d = s.get("date", "")
        es_sessions[d] += 1
        es_out_tokens[d] += s.get("output_tokens", 0)
        es_hours[d] += s.get("duration_mins", 0) / 60
        es_projects[d].add(s.get("repo", "?"))
        es_durations[d].append(s.get("duration_mins", 0))
        tools = s.get("tool_uses", {})
        if isinstance(tools, dict):
            for name, count in tools.items():
                if name in edit_tools:
                    es_edits[d] += count
                elif name in read_tools:
                    es_reads[d] += count

    dates = sorted(es_sessions.keys())
    if len(dates) < 5:
        return ""

    # Compute percentiles for normalization
    all_edits = [es_edits[d] for d in dates]
    all_throughput = [es_out_tokens[d] / max(es_hours[d], 0.1) for d in dates]
    all_durations = [np.mean(es_durations[d]) if es_durations[d] else 0 for d in dates]

    p95_edits = np.percentile(all_edits, 95) or 1
    p95_throughput = np.percentile(all_throughput, 95) or 1
    p95_duration = np.percentile(all_durations, 95) or 1

    scores = []
    streak = 0
    for i, d in enumerate(dates):
        # Output volume (0-25)
        vol = min(es_edits[d] / p95_edits, 1) * 25

        # Throughput (0-20)
        tp = es_out_tokens[d] / max(es_hours[d], 0.1)
        throughput = min(tp / p95_throughput, 1) * 20

        # Focus (0-15)
        n_proj = len(es_projects[d])
        if n_proj <= 1:
            focus = 15
        else:
            counts = list(Counter(s.get("repo", "?") for s in sessions if s.get("date") == d).values())
            total_c = sum(counts)
            probs = [c / total_c for c in counts]
            ent = -sum(p * math.log2(p) for p in probs if p > 0)
            max_ent = math.log2(len(counts)) if len(counts) > 1 else 1
            focus = (1 - ent / max_ent) * 15 if max_ent > 0 else 15

        # Depth (0-15)
        avg_dur = np.mean(es_durations[d]) if es_durations[d] else 0
        depth = min(avg_dur / p95_duration, 1) * 15

        # Craft — write/read ratio (0-15)
        rw = es_edits[d] / max(es_reads[d], 1)
        craft = min(rw / 0.5, 1) * 15  # 0.5 ratio = perfect score

        # Momentum (0-10)
        if i > 0 and dates[i-1] == (datetime.strptime(d, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d"):
            streak += 1
        else:
            streak = 0
        momentum = min(streak / 7, 1) * 10  # 7-day streak = max

        total_score = vol + throughput + focus + depth + craft + momentum
        scores.append(min(total_score, 100))

    # 7-day moving average
    ma7 = [np.mean(scores[max(0,i-6):i+1]) for i in range(len(scores))]
    # 30-day moving average
    ma30 = [np.mean(scores[max(0,i-29):i+1]) for i in range(len(scores))]

    m = 50
    pw, ph = width - 2*m, height - 2*m - 40

    # Daily score bars (faded)
    mx = 100
    bar_w = max(pw / len(dates) - 1, 1.5)
    bars = ""
    for i, v in enumerate(scores):
        h = v / mx * ph
        x = m + i * (bar_w + 1)
        y = m + 20 + ph - h
        if v >= 70: color = C["green"]
        elif v >= 40: color = C["amber"]
        else: color = C["indigo"]
        bars += f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" rx="1" fill="{color}" opacity="0.3"/>'

    # 7-day MA line
    pts7 = " ".join(f"{m+i*(bar_w+1)+bar_w/2:.1f},{m+20+ph-v/mx*ph:.1f}" for i, v in enumerate(ma7))
    line7 = f'<polyline points="{pts7}" fill="none" stroke="{C["amber"]}" stroke-width="2.5" stroke-linejoin="round"/>'

    # 30-day MA line
    pts30 = " ".join(f"{m+i*(bar_w+1)+bar_w/2:.1f},{m+20+ph-v/mx*ph:.1f}" for i, v in enumerate(ma30))
    line30 = f'<polyline points="{pts30}" fill="none" stroke="{C["text"]}" stroke-width="2" stroke-linejoin="round" opacity="0.6"/>'

    # Grade thresholds
    ref_lines = ""
    for threshold, label, color in [(70, "A", C["green"]), (40, "B", C["amber"])]:
        y = m + 20 + ph - threshold / mx * ph
        ref_lines += f'<line x1="{m}" y1="{y:.0f}" x2="{m+pw}" y2="{y:.0f}" stroke="{color}" stroke-dasharray="4 3" opacity="0.3"/>'
        ref_lines += f'<text x="{m+pw+4}" y="{y+4:.0f}" fill="{color}" font-size="9" font-weight="700">{label}</text>'

    current = ma7[-1]
    grade = "A+" if current >= 85 else "A" if current >= 70 else "B+" if current >= 55 else "B" if current >= 40 else "C"
    grade_color = C["green"] if current >= 70 else C["amber"] if current >= 40 else C["rose"]

    stats = (
        f'<text x="{m}" y="{m+20+ph+20}" fill="{C["text2"]}" font-size="10">'
        f'Current: <tspan fill="{grade_color}" font-size="14" font-weight="800">{current:.0f} ({grade})</tspan> · '
        f'7d avg: <tspan fill="{C["amber"]}" font-weight="700">{ma7[-1]:.0f}</tspan> · '
        f'30d avg: <tspan fill="{C["text"]}" font-weight="700">{ma30[-1]:.0f}</tspan> · '
        f'Best: <tspan fill="{C["green"]}" font-weight="700">{max(scores):.0f}</tspan>'
        f'</text>'
    )

    legend = (
        f'<line x1="{m+pw-170}" y1="{m+8}" x2="{m+pw-150}" y2="{m+8}" stroke="{C["amber"]}" stroke-width="2.5"/>'
        f'<text x="{m+pw-146}" y="{m+12}" fill="{C["text2"]}" font-size="8">7d MA</text>'
        f'<line x1="{m+pw-100}" y1="{m+8}" x2="{m+pw-80}" y2="{m+8}" stroke="{C["text"]}" stroke-width="2" opacity="0.6"/>'
        f'<text x="{m+pw-76}" y="{m+12}" fill="{C["text2"]}" font-size="8">30d MA</text>'
    )

    return _svg(width, height, bars + ref_lines + line30 + line7 + stats + legend,
        "Engineer Score (daily composite, 0-100)")


# ═══════════════════════════════════════════════════════════════════════
# PSYCHOLOGICAL / PROWESS / TEACHING CHARTS
# ═══════════════════════════════════════════════════════════════════════


# -----------------------------------------------------------------------
# 64. Developer Archetype Radar — "What kind of engineer are you?"
# -----------------------------------------------------------------------
def developer_archetype(sessions, width=400, height=420):
    """6-axis radar: Builder, Explorer, Debugger, Architect, Automator, Scholar.

    Builder: Edit/Write heavy
    Explorer: Read/Glob/Grep heavy
    Debugger: Bash heavy + short sessions + same-repo revisits
    Architect: Agent/MCP heavy + long sessions + many repos
    Automator: High subagent usage + high tool/message ratio
    Scholar: High token density + long sessions + few tool calls
    """
    totals = {"edits": 0, "reads": 0, "bash": 0, "agent": 0, "mcp": 0,
              "total_tools": 0, "sessions": 0, "tokens": 0, "duration": 0.0,
              "msgs": 0, "repos": set(), "subagent_heavy": 0}

    for s in sessions:
        totals["sessions"] += 1
        totals["tokens"] += s.get("total_tokens", 0)
        totals["duration"] += s.get("duration_mins", 0)
        totals["msgs"] += s.get("user_msgs", 0)
        totals["repos"].add(s.get("repo", "?"))
        if s.get("subagent_heavy"):
            totals["subagent_heavy"] += 1
        tools = s.get("tool_uses", {})
        if isinstance(tools, dict):
            for name, count in tools.items():
                totals["total_tools"] += count
                if name in {"Edit", "Write", "NotebookEdit"}:
                    totals["edits"] += count
                elif name in {"Read", "Glob", "Grep", "LS"}:
                    totals["reads"] += count
                elif name in {"Bash", "BashOutput"}:
                    totals["bash"] += count
                elif name == "Agent":
                    totals["agent"] += count
                elif name.startswith("mcp__"):
                    totals["mcp"] += count

    n = totals["sessions"] or 1
    tt = totals["total_tools"] or 1

    # Raw scores (0-1 range)
    builder = min(totals["edits"] / tt * 3, 1)  # edit-heavy
    explorer = min(totals["reads"] / tt * 2, 1)  # read-heavy
    debugger = min(totals["bash"] / tt * 4, 1)  # bash-heavy
    architect = min((totals["agent"] + totals["mcp"]) / tt * 5, 1) * min(len(totals["repos"]) / 30, 1)
    automator = min(totals["subagent_heavy"] / n * 5, 1) * min(totals["total_tools"] / max(totals["msgs"], 1) / 5, 1)
    scholar = min(totals["tokens"] / max(totals["duration"] * 60, 1) * 0.01, 1) * (1 - min(totals["total_tools"] / max(totals["msgs"], 1) / 10, 0.5))

    axes = [
        ("Builder", builder, C["indigo"]),
        ("Explorer", explorer, C["cyan"]),
        ("Debugger", debugger, C["green"]),
        ("Architect", architect, C["amber"]),
        ("Automator", automator, C["purple"]),
        ("Scholar", scholar, C["pink"]),
    ]

    # Find dominant archetype
    dominant = max(axes, key=lambda x: x[1])

    cx, cy, r = width / 2, 190, 120
    n_axes = len(axes)

    # Grid
    grid = ""
    for frac in [0.25, 0.5, 0.75, 1.0]:
        pts = []
        for i in range(n_axes):
            angle = 2 * math.pi * i / n_axes - math.pi / 2
            pts.append(f"{cx + r * frac * math.cos(angle):.0f},{cy + r * frac * math.sin(angle):.0f}")
        grid += f'<polygon points="{" ".join(pts)}" fill="none" stroke="{C["border"]}" stroke-width="1"/>'

    # Data polygon
    data_pts = []
    for i, (name, val, color) in enumerate(axes):
        angle = 2 * math.pi * i / n_axes - math.pi / 2
        px = cx + r * val * math.cos(angle)
        py = cy + r * val * math.sin(angle)
        data_pts.append(f"{px:.0f},{py:.0f}")

    data_poly = f'<polygon points="{" ".join(data_pts)}" fill="{C["indigo"]}" opacity="0.15" stroke="{C["indigo"]}" stroke-width="2"/>'

    # Axis labels + dots
    labels = ""
    for i, (name, val, color) in enumerate(axes):
        angle = 2 * math.pi * i / n_axes - math.pi / 2
        lx = cx + (r + 28) * math.cos(angle)
        ly = cy + (r + 28) * math.sin(angle)
        dx = cx + r * val * math.cos(angle)
        dy = cy + r * val * math.sin(angle)
        labels += f'<text x="{lx:.0f}" y="{ly + 4:.0f}" text-anchor="middle" fill="{color}" font-size="11" font-weight="700">{name}</text>'
        labels += f'<text x="{lx:.0f}" y="{ly + 16:.0f}" text-anchor="middle" fill="{C["text2"]}" font-size="9">{val:.0%}</text>'
        labels += f'<circle cx="{dx:.0f}" cy="{dy:.0f}" r="4" fill="{color}"/>'

    # Dominant label
    dom_label = f'<text x="{cx}" y="{cy + r + 55}" text-anchor="middle" fill="{dominant[2]}" font-size="16" font-weight="800">The {dominant[0]}</text>'

    return _svg(width, height, grid + data_poly + labels + dom_label, "Developer Archetype")


# -----------------------------------------------------------------------
# 65. Cognitive Load Index — daily mental burden estimate
# -----------------------------------------------------------------------
def cognitive_load(sessions, width=600, height=220):
    """Composite cognitive load: context switches + token density + session count.

    High load = many projects + many sessions + high token throughput.
    Sustained high load → burnout risk.
    """
    cl_projects: dict[str, set] = defaultdict(set)
    cl_sessions: dict[str, int] = defaultdict(int)
    cl_tokens: dict[str, int] = defaultdict(int)
    cl_hours: dict[str, float] = defaultdict(float)
    for s in sessions:
        d = s.get("date", "")
        cl_projects[d].add(s.get("repo", "?"))
        cl_sessions[d] += 1
        cl_tokens[d] += s.get("total_tokens", 0)
        cl_hours[d] += s.get("duration_mins", 0) / 60

    dates = sorted(cl_projects.keys())
    if len(dates) < 5:
        return ""

    loads = []
    for d in dates:
        ctx = len(cl_projects[d]) / 10  # 10 projects = max
        vol = cl_sessions[d] / 50  # 50 sessions = max
        density = cl_tokens[d] / max(cl_hours[d], 0.1) / 50000  # 50k tok/hr = max
        load = min((ctx * 0.4 + vol * 0.3 + density * 0.3) * 100, 100)
        loads.append(load)

    smoothed = [np.mean(loads[max(0,i-2):i+1]) for i in range(len(loads))]

    m = 50
    pw, ph = width - 2*m, height - 2*m

    # Color zones
    zones = ""
    for threshold, color, label in [(80, C["rose"], "Overload"), (50, C["amber"], "High"), (25, C["green"], "Optimal")]:
        y = m + ph - threshold / 100 * ph
        zones += f'<line x1="{m}" y1="{y:.0f}" x2="{m+pw}" y2="{y:.0f}" stroke="{color}" stroke-dasharray="4 3" opacity="0.3"/>'
        zones += f'<text x="{m+pw+4}" y="{y+4:.0f}" fill="{color}" font-size="8">{label}</text>'

    # Area fill
    points = []
    for i, v in enumerate(smoothed):
        x = m + i / max(len(smoothed)-1, 1) * pw
        y = m + ph - v / 100 * ph
        points.append(f"{x:.1f},{y:.1f}")

    fill_pts = f"{m},{m+ph} " + " ".join(points) + f" {m+pw},{m+ph}"
    current = smoothed[-1]
    load_color = C["rose"] if current > 70 else C["amber"] if current > 40 else C["green"]

    return _svg(width, height,
        zones +
        f'<defs><linearGradient id="clg" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="{load_color}"/><stop offset="100%" stop-color="transparent"/></linearGradient></defs>'
        f'<polygon points="{fill_pts}" fill="url(#clg)" opacity="0.25"/>'
        f'<polyline points="{" ".join(points)}" fill="none" stroke="{load_color}" stroke-width="2.5" stroke-linejoin="round"/>'
        f'<text x="{m}" y="{m-4}" fill="{load_color}" font-size="12" font-weight="800">Cognitive Load: {current:.0f}/100</text>',
        "")


# -----------------------------------------------------------------------
# 66. Decision Fatigue — does edit quality decline through the day?
# -----------------------------------------------------------------------
def decision_fatigue(sessions, width=500, height=220):
    """Edit-to-read ratio by hour — proxy for decisiveness.
    High ratio = confident editing. Low ratio = lots of reading, hesitation.
    """
    by_hour = defaultdict(lambda: [0, 0])  # [edits, reads]
    for s in sessions:
        h = s.get("hour", 0)
        tools = s.get("tool_uses", {})
        if isinstance(tools, dict):
            for name, count in tools.items():
                if name in {"Edit", "Write", "NotebookEdit"}:
                    by_hour[h][0] += count
                elif name in {"Read", "Glob", "Grep", "LS"}:
                    by_hour[h][1] += count

    if not by_hour:
        return ""

    ratios = {}
    for h in range(24):
        e, r = by_hour.get(h, [0, 0])
        ratios[h] = e / max(r, 1)

    mx = max(ratios.values()) or 1
    m = 50
    pw, ph = width - 2*m, height - 2*m
    bar_w = pw / 24 - 2

    bars = ""
    for h in range(24):
        v = ratios.get(h, 0)
        bh = v / mx * ph
        x = m + h * (bar_w + 2)
        y = m + ph - bh
        # Green = decisive, red = fatigued
        color = C["green"] if v > mx * 0.6 else C["amber"] if v > mx * 0.3 else C["rose"]
        bars += f'<rect x="{x:.0f}" y="{y:.0f}" width="{bar_w:.0f}" height="{bh:.0f}" rx="2" fill="{color}" opacity="0.7"/>'

    labels = "".join(f'<text x="{m + h*(bar_w+2) + bar_w/2:.0f}" y="{m+ph+13}" text-anchor="middle" fill="{C["text2"]}" font-size="8">{h:02d}</text>' for h in range(0, 24, 3))

    # Find fatigue pattern
    morning = np.mean([ratios.get(h, 0) for h in range(8, 12)])
    evening = np.mean([ratios.get(h, 0) for h in range(20, 24)])
    if evening < morning * 0.6:
        insight = "Significant decision fatigue detected in evening hours"
        ins_color = C["rose"]
    elif evening > morning:
        insight = "You get MORE decisive as the day goes on — night owl pattern"
        ins_color = C["green"]
    else:
        insight = "Consistent decision-making throughout the day"
        ins_color = C["amber"]

    return _svg(width, height, bars + labels +
        f'<text x="{m}" y="{m-6}" fill="{ins_color}" font-size="10" font-weight="700">{insight}</text>',
        "Decision Fatigue (edit/read ratio by hour)")


# -----------------------------------------------------------------------
# 67. Creative vs Analytical Mode
# -----------------------------------------------------------------------
def creative_analytical(sessions, width=600, height=220):
    """Track daily mode: Write-heavy = creative, Read/Grep-heavy = analytical.

    Score: (edits + writes) / (reads + greps) — >1 = creative, <1 = analytical
    """
    by_date = defaultdict(lambda: [0, 0])
    for s in sessions:
        d = s.get("date", "")
        tools = s.get("tool_uses", {})
        if isinstance(tools, dict):
            for name, count in tools.items():
                if name in {"Edit", "Write", "NotebookEdit"}:
                    by_date[d][0] += count
                elif name in {"Read", "Glob", "Grep", "LS"}:
                    by_date[d][1] += count

    dates = sorted(by_date.keys())
    if len(dates) < 5:
        return ""

    ratios = [by_date[d][0] / max(by_date[d][1], 1) for d in dates]
    smoothed = [np.mean(ratios[max(0,i-2):i+1]) for i in range(len(ratios))]

    m = 50
    pw, ph = width - 2*m, height - 2*m
    # Center line at ratio=1.0 (balance)
    mx = max(max(smoothed), 2)

    # Two-tone area
    points_top = []
    points_bot = []
    center_y = m + ph / 2
    for i, v in enumerate(smoothed):
        x = m + i / max(len(smoothed)-1, 1) * pw
        # Map ratio to position: 1.0 = center, >1 = above (creative), <1 = below (analytical)
        y = center_y - (v - 1) / (mx - 1) * (ph / 2) if v >= 1 else center_y + (1 - v) / 1 * (ph / 2)
        y = max(m, min(m + ph, y))
        if v >= 1:
            points_top.append(f"{x:.1f},{y:.1f}")
        else:
            points_bot.append(f"{x:.1f},{y:.1f}")

    # Full line
    all_pts = []
    for i, v in enumerate(smoothed):
        x = m + i / max(len(smoothed)-1, 1) * pw
        y = center_y - (v - 1) / max(mx - 1, 0.1) * (ph / 2)
        y = max(m, min(m + ph, y))
        all_pts.append(f"{x:.1f},{y:.1f}")

    creative_pct = sum(1 for r in ratios if r > 1) / len(ratios) * 100

    return _svg(width, height,
        f'<line x1="{m}" y1="{center_y}" x2="{m+pw}" y2="{center_y}" stroke="{C["border"]}" stroke-width="1"/>'
        f'<text x="{m+pw+4}" y="{m+10}" fill="{C["indigo"]}" font-size="9">Creative</text>'
        f'<text x="{m+pw+4}" y="{m+ph-4}" fill="{C["cyan"]}" font-size="9">Analytical</text>'
        f'<polyline points="{" ".join(all_pts)}" fill="none" stroke="{C["indigo"]}" stroke-width="2.5" stroke-linejoin="round"/>'
        f'<text x="{m}" y="{m-6}" fill="{C["text"]}" font-size="10" font-weight="700">{creative_pct:.0f}% creative days · {100-creative_pct:.0f}% analytical days</text>',
        "Creative vs Analytical Mode (edit/read ratio, 1.0 = balanced)")


# -----------------------------------------------------------------------
# 68. Prowess Pentagon — 5-axis qualitative assessment
# -----------------------------------------------------------------------
def prowess_pentagon(sessions, width=400, height=400):
    """5-axis prowess assessment with letter grades.

    Craftsmanship: edit quality (edit/read ratio)
    Velocity: tokens per hour
    Depth: avg meaningful session duration
    Breadth: unique repos touched
    Delegation: effective use of Agent tool
    """
    total_edits, total_reads, total_tokens = 0, 0, 0
    total_hours, total_agent, total_tools = 0.0, 0, 0
    meaningful_durations = []
    repos = set()

    for s in sessions:
        repos.add(s.get("repo", "?"))
        total_tokens += s.get("output_tokens", 0)
        dur = s.get("duration_mins", 0)
        total_hours += dur / 60
        if dur > 5:
            meaningful_durations.append(dur)
        tools = s.get("tool_uses", {})
        if isinstance(tools, dict):
            for name, count in tools.items():
                total_tools += count
                if name in {"Edit", "Write", "NotebookEdit"}:
                    total_edits += count
                elif name in {"Read", "Glob", "Grep", "LS"}:
                    total_reads += count
                elif name == "Agent":
                    total_agent += count

    # Score each axis 0-1
    craft = min(total_edits / max(total_reads, 1) / 0.5, 1)  # 0.5 ratio = perfect
    velocity = min(total_tokens / max(total_hours, 1) / 15000, 1)  # 15k tok/hr = perfect
    depth = min(np.mean(meaningful_durations) / 60, 1) if meaningful_durations else 0  # 60min avg = perfect
    breadth = min(len(repos) / 50, 1)  # 50 repos = perfect
    delegation = min(total_agent / max(total_tools, 1) / 0.1, 1)  # 10% agent = perfect

    axes = [
        ("Craftsmanship", craft),
        ("Velocity", velocity),
        ("Depth", depth),
        ("Breadth", breadth),
        ("Delegation", delegation),
    ]

    def grade(v):
        if v >= 0.9: return "A+"
        if v >= 0.75: return "A"
        if v >= 0.6: return "B+"
        if v >= 0.45: return "B"
        if v >= 0.3: return "C+"
        return "C"

    def grade_color(v):
        if v >= 0.75: return C["green"]
        if v >= 0.45: return C["amber"]
        return C["rose"]

    cx, cy, r = width / 2, 180, 110
    n_axes = len(axes)

    # Grid
    grid = ""
    for frac in [0.25, 0.5, 0.75, 1.0]:
        pts = []
        for i in range(n_axes):
            angle = 2 * math.pi * i / n_axes - math.pi / 2
            pts.append(f"{cx + r * frac * math.cos(angle):.0f},{cy + r * frac * math.sin(angle):.0f}")
        grid += f'<polygon points="{" ".join(pts)}" fill="none" stroke="{C["border"]}" stroke-width="1"/>'

    # Data polygon
    data_pts = []
    for i, (name, val) in enumerate(axes):
        angle = 2 * math.pi * i / n_axes - math.pi / 2
        px = cx + r * val * math.cos(angle)
        py = cy + r * val * math.sin(angle)
        data_pts.append(f"{px:.0f},{py:.0f}")

    data_poly = f'<polygon points="{" ".join(data_pts)}" fill="{C["green"]}" opacity="0.12" stroke="{C["green"]}" stroke-width="2.5"/>'

    # Labels + grades
    labels = ""
    for i, (name, val) in enumerate(axes):
        angle = 2 * math.pi * i / n_axes - math.pi / 2
        lx = cx + (r + 35) * math.cos(angle)
        ly = cy + (r + 35) * math.sin(angle)
        dx = cx + r * val * math.cos(angle)
        dy = cy + r * val * math.sin(angle)
        gc = grade_color(val)
        labels += f'<text x="{lx:.0f}" y="{ly:.0f}" text-anchor="middle" fill="{C["text"]}" font-size="10" font-weight="700">{name}</text>'
        labels += f'<text x="{lx:.0f}" y="{ly + 14:.0f}" text-anchor="middle" fill="{gc}" font-size="13" font-weight="800">{grade(val)}</text>'
        labels += f'<circle cx="{dx:.0f}" cy="{dy:.0f}" r="4" fill="{gc}"/>'

    overall = np.mean([v for _, v in axes])
    overall_grade = grade(overall)
    overall_color = grade_color(overall)

    return _svg(width, height, grid + data_poly + labels +
        f'<text x="{cx}" y="{cy + r + 60}" text-anchor="middle" fill="{overall_color}" font-size="20" font-weight="900">Overall: {overall_grade}</text>'
        f'<text x="{cx}" y="{cy + r + 76}" text-anchor="middle" fill="{C["text2"]}" font-size="10">{overall:.0%} composite prowess</text>',
        "Prowess Pentagon")


# -----------------------------------------------------------------------
# 69. Self-Teaching Insights — auto-generated "aha" moments
# -----------------------------------------------------------------------
def teaching_insights(sessions, width=650, height=500):
    """Generate surprising, data-backed personal insights as SVG text cards."""
    insights = []

    # Best hour
    hour_tokens = defaultdict(int)
    hour_sessions = defaultdict(int)
    for s in sessions:
        h = s.get("hour", 0)
        hour_tokens[h] += s.get("output_tokens", 0)
        hour_sessions[h] += 1
    if hour_tokens:
        best_h = max(hour_tokens, key=lambda h: hour_tokens[h] / max(hour_sessions[h], 1))
        worst_h = min(hour_tokens, key=lambda h: hour_tokens[h] / max(hour_sessions[h], 1))
        ratio = (hour_tokens[best_h] / max(hour_sessions[best_h], 1)) / max(hour_tokens[worst_h] / max(hour_sessions[worst_h], 1), 1)
        insights.append(f"Your peak hour is {best_h:02d}:00 — you produce {ratio:.1f}x more output than at {worst_h:02d}:00")

    # Best day
    day_tokens = defaultdict(list)
    for s in sessions:
        day_tokens[s.get("weekday", "?")].append(s.get("output_tokens", 0))
    if day_tokens:
        best_day = max(day_tokens, key=lambda d: np.mean(day_tokens[d]))
        worst_day = min(day_tokens, key=lambda d: np.mean(day_tokens[d]))
        if best_day != worst_day:
            ratio = np.mean(day_tokens[best_day]) / max(np.mean(day_tokens[worst_day]), 1)
            insights.append(f"{best_day}s are your power day — {ratio:.1f}x more productive than {worst_day}s")

    # Session length sweet spot
    dur_tokens = defaultdict(list)
    for s in sessions:
        d = s.get("duration_mins", 0)
        if d > 1:
            bucket = "< 10min" if d < 10 else "10-30min" if d < 30 else "30-60min" if d < 60 else "1-2hrs" if d < 120 else "2hrs+"
            t = s.get("output_tokens", 0) / max(d, 1)
            dur_tokens[bucket].append(t)
    if dur_tokens:
        best_bucket = max(dur_tokens, key=lambda b: np.mean(dur_tokens[b]))
        insights.append(f"Your sweet spot is {best_bucket} sessions — highest tokens per minute")

    # Delegation insight
    agent_sessions = sum(1 for s in sessions if s.get("tool_uses", {}).get("Agent", 0) > 0)
    total = len(sessions) or 1
    if agent_sessions > 0:
        insights.append(f"You delegate to subagents in {agent_sessions/total*100:.0f}% of sessions — that's your force multiplier")

    # Rework pattern
    by_date_repo = defaultdict(list)
    for s in sorted(sessions, key=lambda x: x.get("first_ts", 0)):
        by_date_repo[s.get("date", "")].append(s.get("repo", "?"))
    revisits = 0
    total_days = 0
    for d, repos in by_date_repo.items():
        total_days += 1
        seen = set()
        for r in repos:
            if r in seen:
                revisits += 1
                break
            seen.add(r)
    if total_days > 0:
        insights.append(f"You revisit the same repo multiple times in {revisits/total_days*100:.0f}% of days — potential rework signal")

    # Weekend warrior
    we_sessions = sum(1 for s in sessions if s.get("weekday_num", 0) >= 5)
    if we_sessions > 0:
        insights.append(f"You've coded {we_sessions} sessions on weekends — {we_sessions/total*100:.0f}% of all work")

    # Growing or shrinking sessions
    dates = sorted(set(s.get("date", "") for s in sessions))
    if len(dates) > 14:
        first_week = [s.get("duration_mins", 0) for s in sessions if s.get("date", "") in dates[:7]]
        last_week = [s.get("duration_mins", 0) for s in sessions if s.get("date", "") in dates[-7:]]
        if first_week and last_week:
            growth = np.mean(last_week) / max(np.mean(first_week), 1)
            if growth > 1.3:
                insights.append(f"Sessions are {growth:.1f}x longer now than when you started — you're going deeper")
            elif growth < 0.7:
                insights.append(f"Sessions are {1/growth:.1f}x shorter now — you're getting faster or more fragmented")

    m = 15
    y = 35
    cards = ""
    for i, insight in enumerate(insights[:8]):
        card_h = 45
        accent = PALETTE[i % len(PALETTE)]
        cards += f'<rect x="{m}" y="{y}" width="{width-2*m}" height="{card_h}" rx="8" fill="{C["surface"]}" stroke="{C["border"]}" stroke-width="1"/>'
        cards += f'<rect x="{m}" y="{y}" width="4" height="{card_h}" rx="2" fill="{accent}"/>'
        cards += f'<text x="{m+16}" y="{y+28}" fill="{C["text"]}" font-size="11">{insight}</text>'
        y += card_h + 8

    return _svg(width, min(y + 10, 600), cards, "What Your Data Reveals About You")


# -----------------------------------------------------------------------
# 70. Growth Trajectory — first month vs last month
# -----------------------------------------------------------------------
def growth_trajectory(sessions, width=600, height=300):
    """Compare metrics from first 2 weeks vs last 2 weeks."""
    dates = sorted(set(s.get("date", "") for s in sessions if s.get("date")))
    if len(dates) < 14:
        return ""

    early_dates = set(dates[:14])
    late_dates = set(dates[-14:])

    early = [s for s in sessions if s.get("date", "") in early_dates]
    late = [s for s in sessions if s.get("date", "") in late_dates]

    def metrics(sess):
        n = len(sess) or 1
        total_edits = sum(s.get("tool_uses", {}).get("Edit", 0) + s.get("tool_uses", {}).get("Write", 0) for s in sess if isinstance(s.get("tool_uses"), dict))
        total_tokens = sum(s.get("output_tokens", 0) for s in sess)
        total_hours = sum(s.get("duration_mins", 0) for s in sess) / 60
        avg_dur = np.mean([s.get("duration_mins", 0) for s in sess if s.get("duration_mins", 0) > 1])
        repos = len(set(s.get("repo", "?") for s in sess))
        return {
            "Sessions/day": n / 14,
            "Edits/session": total_edits / n,
            "Tokens/hour": total_tokens / max(total_hours, 1),
            "Avg duration": avg_dur if not np.isnan(avg_dur) else 0,
            "Repos touched": repos,
        }

    m_early = metrics(early)
    m_late = metrics(late)

    m_left, m_top = 120, 40
    pw = width - m_left - 80
    row_h = (height - m_top - 20) / len(m_early)

    bars = ""
    for i, key in enumerate(m_early):
        y = m_top + i * row_h
        ev = m_early[key]
        lv = m_late[key]
        mx = max(ev, lv) or 1
        ew = ev / mx * pw * 0.4
        lw = lv / mx * pw * 0.4

        bars += f'<text x="{m_left-6}" y="{y+row_h/2+4:.0f}" text-anchor="end" fill="{C["text2"]}" font-size="10">{key}</text>'
        # Early bar
        bars += f'<rect x="{m_left}" y="{y+4:.0f}" width="{ew:.0f}" height="{row_h*0.35:.0f}" rx="3" fill="{C["cyan"]}" opacity="0.6"/>'
        # Late bar
        bars += f'<rect x="{m_left}" y="{y+row_h*0.45:.0f}" width="{lw:.0f}" height="{row_h*0.35:.0f}" rx="3" fill="{C["green"]}" opacity="0.6"/>'

        # Change indicator
        if ev > 0:
            change = (lv - ev) / ev * 100
            ch_color = C["green"] if change > 0 else C["rose"]
            arrow = "+" if change > 0 else ""
            bars += f'<text x="{m_left+pw*0.4+10}" y="{y+row_h/2+4:.0f}" fill="{ch_color}" font-size="11" font-weight="800">{arrow}{change:.0f}%</text>'

    legend = (
        f'<rect x="{m_left}" y="{m_top-18}" width="8" height="8" rx="2" fill="{C["cyan"]}"/>'
        f'<text x="{m_left+12}" y="{m_top-10}" fill="{C["text2"]}" font-size="9">First 2 weeks</text>'
        f'<rect x="{m_left+110}" y="{m_top-18}" width="8" height="8" rx="2" fill="{C["green"]}"/>'
        f'<text x="{m_left+122}" y="{m_top-10}" fill="{C["text2"]}" font-size="9">Last 2 weeks</text>'
    )

    return _svg(width, height, bars + legend, "Growth Trajectory")


# -----------------------------------------------------------------------
# 71. FAANG Level Estimator — L3-L8 modeled over time like a stock chart
# -----------------------------------------------------------------------
def faang_level(sessions, width=700, height=340):
    """Estimate FAANG-equivalent engineering level (L3-L8) per day.

    Signals:
      - Scope breadth: unique repos touched (L3=1, L8=8+)
      - Delegation: Agent tool usage rate (L3=0%, L8=60%+)
      - Output velocity: edits + writes per hour
      - Session depth: average session duration
      - Tool sophistication: unique tools per session
      - Cross-project integration: repos/day without multitask penalty
    Composite mapped to 3.0–8.0 scale, with 7d and 30d moving averages.
    """
    day_repos: dict[str, set] = defaultdict(set)
    day_sessions: dict[str, int] = defaultdict(int)
    day_edits: dict[str, int] = defaultdict(int)
    day_hours: dict[str, float] = defaultdict(float)
    day_agents: dict[str, int] = defaultdict(int)
    day_tools: dict[str, set] = defaultdict(set)
    day_tokens: dict[str, int] = defaultdict(int)
    day_durations: dict[str, list] = defaultdict(list)

    for s in sessions:
        d = s.get("date", "")
        if not d:
            continue
        day_repos[d].add(s.get("repo", "?"))
        day_sessions[d] += 1
        tu = s.get("tool_uses", {}) if isinstance(s.get("tool_uses"), dict) else {}
        day_edits[d] += tu.get("Edit", 0) + tu.get("Write", 0)
        day_hours[d] += s.get("duration_mins", 0) / 60
        day_agents[d] += tu.get("Agent", 0)
        for t in tu:
            day_tools[d].add(t)
        day_tokens[d] += s.get("output_tokens", 0)
        dur = s.get("duration_mins", 0)
        if dur > 1:
            day_durations[d].append(dur)

    dates = sorted(day_repos.keys())
    if len(dates) < 7:
        return ""

    daily_levels: list[float] = []
    for d in dates:
        repos = len(day_repos[d])
        n_sess = day_sessions[d]
        hrs = max(day_hours[d], 0.1)
        edits_hr = day_edits[d] / hrs
        agent_rate = day_agents[d] / max(n_sess, 1)
        tool_variety = len(day_tools[d])
        avg_dur = float(np.mean(day_durations[d])) if day_durations[d] else 5.0
        tok_hr = day_tokens[d] / hrs

        # Score each dimension 0-1
        scope = min(repos / 6, 1.0)                    # 6+ repos = max
        delegation = min(agent_rate / 0.5, 1.0)         # 50%+ = max
        velocity = min(edits_hr / 40, 1.0)              # 40 edits/hr = max
        depth = min(avg_dur / 90, 1.0)                   # 90min avg = max
        sophistication = min(tool_variety / 15, 1.0)     # 15+ tools = max
        throughput = min(tok_hr / 30000, 1.0)            # 30k tok/hr = max

        # Weighted composite → 3.0 to 8.0
        composite = (
            scope * 0.20 +
            delegation * 0.18 +
            velocity * 0.18 +
            depth * 0.15 +
            sophistication * 0.14 +
            throughput * 0.15
        )
        level = 3.0 + composite * 5.0  # maps 0→L3, 1→L8
        daily_levels.append(level)

    # Moving averages
    ma7 = [float(np.mean(daily_levels[max(0, i-6):i+1])) for i in range(len(daily_levels))]
    ma30 = [float(np.mean(daily_levels[max(0, i-29):i+1])) for i in range(len(daily_levels))]

    # Current level (30d avg)
    current = ma30[-1]
    level_name = (
        "L3 (Junior)" if current < 3.8 else
        "L4 (Mid-Level)" if current < 4.6 else
        "L5 (Senior)" if current < 5.4 else
        "L6 (Staff)" if current < 6.2 else
        "L7 (Principal)" if current < 7.0 else
        "L8 (Distinguished)"
    )

    m = 55
    pw, ph = width - m - 20, height - m - 50
    y_min, y_max = 3.0, 8.0
    y_range = y_max - y_min

    # Level zone backgrounds
    zones = ""
    level_labels = [
        (3.0, "L3"), (4.0, "L4"), (5.0, "L5"),
        (6.0, "L6"), (7.0, "L7"), (8.0, "L8")
    ]
    zone_colors = ["#22c55e20", "#06b6d420", "#6366f120", "#a855f720", "#f59e0b20"]
    for i in range(5):
        low = level_labels[i][0]
        high = level_labels[i + 1][0]
        y_top = m + ph - (high - y_min) / y_range * ph
        y_bot = m + ph - (low - y_min) / y_range * ph
        zones += f'<rect x="{m}" y="{y_top:.0f}" width="{pw}" height="{y_bot - y_top:.0f}" fill="{zone_colors[i]}"/>'
        zones += f'<text x="{m-4}" y="{(y_top + y_bot) / 2 + 4:.0f}" text-anchor="end" fill="{C["text2"]}" font-size="9" font-weight="700">{level_labels[i][1]}</text>'

    # Daily bars (thin candlestick-style)
    bars = ""
    n = len(daily_levels)
    bar_w = max(pw / n - 1, 1)
    for i, lv in enumerate(daily_levels):
        x = m + i / max(n - 1, 1) * pw - bar_w / 2
        y = m + ph - (lv - y_min) / y_range * ph
        h = (lv - y_min) / y_range * ph
        color = C["indigo"] if lv >= ma30[i] else C["rose"]
        bars += f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" rx="1" fill="{color}" opacity="0.25"/>'

    # MA lines
    def _ma_path(vals, color, dash=""):
        pts = []
        for i, v in enumerate(vals):
            x = m + i / max(n - 1, 1) * pw
            y = m + ph - (v - y_min) / y_range * ph
            pts.append(f"{x:.1f},{y:.1f}")
        style = f' stroke-dasharray="{dash}"' if dash else ""
        return f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="2"{style}/>'

    ma7_line = _ma_path(ma7, C["cyan"])
    ma30_line = _ma_path(ma30, C["amber"], "6 3")

    # Current level callout
    cy = m + ph - (current - y_min) / y_range * ph
    callout = (
        f'<circle cx="{m + pw}" cy="{cy:.0f}" r="4" fill="{C["amber"]}"/>'
        f'<text x="{m + pw + 8}" y="{cy + 4:.0f}" fill="{C["amber"]}" font-size="12" font-weight="800">L{current:.1f}</text>'
    )

    # Legend
    legend = (
        f'<rect x="{m}" y="{m + ph + 18}" width="12" height="3" rx="1" fill="{C["cyan"]}"/>'
        f'<text x="{m + 16}" y="{m + ph + 22}" fill="{C["text2"]}" font-size="9">7-day MA</text>'
        f'<rect x="{m + 90}" y="{m + ph + 18}" width="12" height="3" rx="1" fill="{C["amber"]}"/>'
        f'<text x="{m + 106}" y="{m + ph + 22}" fill="{C["text2"]}" font-size="9">30-day MA</text>'
        f'<text x="{m + 200}" y="{m + ph + 22}" fill="{C["text"]}" font-size="10" font-weight="700">Current: {level_name}</text>'
    )

    # Date labels
    date_labels = ""
    step = max(n // 6, 1)
    for i in range(0, n, step):
        x = m + i / max(n - 1, 1) * pw
        date_labels += f'<text x="{x:.0f}" y="{m + ph + 12}" text-anchor="middle" fill="{C["text2"]}" font-size="8">{dates[i][5:]}</text>'

    return _svg(width, height, zones + bars + ma7_line + ma30_line + callout + legend + date_labels,
        "FAANG Level Estimation (L3-L8)")


# -----------------------------------------------------------------------
# 72. Architecture Score — quality of architectural choices over time
# -----------------------------------------------------------------------
def architecture_score(sessions, width=700, height=340):
    """Score architectural decision quality per day, stock-chart style.

    Signals:
      - Research ratio: reads / (reads + writes) — architects read more before writing
      - Session focus: single-repo deep dives vs scattered multi-repo
      - Planning signals: Glob/Grep before Edit patterns
      - Code review signals: high read, low write sessions
      - Refactor detection: Edit-heavy sessions in mature repos
      - Design sessions: long duration + high read ratio + few edits
    Each day scored 0-100, with 7d and 30d moving averages + Bollinger-style bands.
    """
    day_reads: dict[str, int] = defaultdict(int)
    day_writes: dict[str, int] = defaultdict(int)
    day_globs: dict[str, int] = defaultdict(int)
    day_edits: dict[str, int] = defaultdict(int)
    day_repos: dict[str, set] = defaultdict(set)
    day_sessions: dict[str, int] = defaultdict(int)
    day_durations: dict[str, list] = defaultdict(list)
    day_agents: dict[str, int] = defaultdict(int)

    for s in sessions:
        d = s.get("date", "")
        if not d:
            continue
        tu = s.get("tool_uses", {}) if isinstance(s.get("tool_uses"), dict) else {}
        day_reads[d] += tu.get("Read", 0)
        day_writes[d] += tu.get("Write", 0) + tu.get("Edit", 0)
        day_globs[d] += tu.get("Glob", 0) + tu.get("Grep", 0)
        day_edits[d] += tu.get("Edit", 0)
        day_repos[d].add(s.get("repo", "?"))
        day_sessions[d] += 1
        dur = s.get("duration_mins", 0)
        if dur > 1:
            day_durations[d].append(dur)
        day_agents[d] += tu.get("Agent", 0)

    dates = sorted(day_reads.keys())
    if len(dates) < 7:
        return ""

    daily_scores: list[float] = []
    for d in dates:
        reads = day_reads[d]
        writes = day_writes[d]
        globs = day_globs[d]
        edits = day_edits[d]
        repos = len(day_repos[d])
        n_sess = day_sessions[d]
        durs = day_durations[d]
        agents = day_agents[d]

        # 1. Research ratio (architects read before writing) — 0 to 1
        research = reads / max(reads + writes, 1)

        # 2. Planning ratio (search before edit) — 0 to 1
        planning = globs / max(globs + edits, 1)

        # 3. Focus score (fewer repos = deeper architecture work)
        focus = 1.0 / max(repos, 1)  # 1 repo = 1.0, 5 repos = 0.2

        # 4. Session depth (longer = more architectural)
        avg_dur = float(np.mean(durs)) if durs else 5.0
        depth = min(avg_dur / 60, 1.0)  # 60min+ = max

        # 5. Delegation maturity (using agents = systems thinking)
        deleg = min(agents / max(n_sess * 2, 1), 1.0)

        # 6. Design session detection (high read, long, few writes)
        design = 0.0
        if reads > writes * 2 and avg_dur > 30:
            design = 1.0
        elif reads > writes:
            design = 0.5

        score = (
            research * 25 +
            planning * 20 +
            focus * 10 +
            depth * 15 +
            deleg * 15 +
            design * 15
        )
        daily_scores.append(min(score, 100))

    n = len(daily_scores)
    ma7 = [float(np.mean(daily_scores[max(0, i-6):i+1])) for i in range(n)]
    ma30 = [float(np.mean(daily_scores[max(0, i-29):i+1])) for i in range(n)]

    # Bollinger bands (20d, 2 std)
    upper_band: list[float] = []
    lower_band: list[float] = []
    for i in range(n):
        window = daily_scores[max(0, i-19):i+1]
        avg = float(np.mean(window))
        std = float(np.std(window)) if len(window) > 1 else 0
        upper_band.append(min(avg + 2 * std, 100))
        lower_band.append(max(avg - 2 * std, 0))

    m = 50
    pw, ph = width - m - 20, height - m - 50
    # Grade zones
    zones = ""
    grade_defs = [
        (0, 25, "Tactical", "#f43f5e15"),
        (25, 50, "Emerging", "#f59e0b15"),
        (50, 75, "Mature", "#06b6d415"),
        (75, 100, "Visionary", "#22c55e15"),
    ]
    for low, high, label, color in grade_defs:
        y_top = m + ph - high / 100 * ph
        y_bot = m + ph - low / 100 * ph
        zones += f'<rect x="{m}" y="{y_top:.0f}" width="{pw}" height="{y_bot - y_top:.0f}" fill="{color}"/>'
        zones += f'<text x="{m - 4}" y="{(y_top + y_bot) / 2 + 4:.0f}" text-anchor="end" fill="{C["text2"]}" font-size="8">{label}</text>'

    # Bollinger band fill
    band_pts_up = []
    band_pts_down = []
    for i in range(n):
        x = m + i / max(n - 1, 1) * pw
        y_up = m + ph - upper_band[i] / 100 * ph
        y_down = m + ph - lower_band[i] / 100 * ph
        band_pts_up.append(f"{x:.1f},{y_up:.1f}")
        band_pts_down.append(f"{x:.1f},{y_down:.1f}")
    band_polygon = " ".join(band_pts_up) + " " + " ".join(reversed(band_pts_down))
    band_fill = f'<polygon points="{band_polygon}" fill="{C["indigo"]}" opacity="0.08"/>'

    # Daily candlestick bars
    bars = ""
    bar_w = max(pw / n - 1, 1)
    for i, sc in enumerate(daily_scores):
        x = m + i / max(n - 1, 1) * pw - bar_w / 2
        y = m + ph - sc / 100 * ph
        h = sc / 100 * ph
        # Green if above 30d MA, red if below
        color = C["green"] if sc >= ma30[i] else C["rose"]
        bars += f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" rx="1" fill="{color}" opacity="0.2"/>'

    # MA lines
    def _line(vals, color, dash=""):
        pts = []
        for i, v in enumerate(vals):
            x = m + i / max(n - 1, 1) * pw
            y = m + ph - v / 100 * ph
            pts.append(f"{x:.1f},{y:.1f}")
        style = f' stroke-dasharray="{dash}"' if dash else ""
        return f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="2"{style}/>'

    ma7_line = _line(ma7, C["cyan"])
    ma30_line = _line(ma30, C["amber"], "6 3")

    # Current score callout
    current = ma30[-1]
    grade = "Visionary" if current >= 75 else "Mature" if current >= 50 else "Emerging" if current >= 25 else "Tactical"
    cy = m + ph - current / 100 * ph
    callout = (
        f'<circle cx="{m + pw}" cy="{cy:.0f}" r="4" fill="{C["amber"]}"/>'
        f'<text x="{m + pw + 8}" y="{cy + 4:.0f}" fill="{C["amber"]}" font-size="12" font-weight="800">{current:.0f}</text>'
    )

    # Trend arrow
    if len(ma30) > 7:
        trend = ma30[-1] - ma30[-8]
        arrow = "↑" if trend > 2 else "↓" if trend < -2 else "→"
    else:
        arrow = "→"

    legend = (
        f'<rect x="{m}" y="{m + ph + 18}" width="12" height="3" rx="1" fill="{C["cyan"]}"/>'
        f'<text x="{m + 16}" y="{m + ph + 22}" fill="{C["text2"]}" font-size="9">7-day MA</text>'
        f'<rect x="{m + 80}" y="{m + ph + 18}" width="12" height="3" rx="1" fill="{C["amber"]}"/>'
        f'<text x="{m + 96}" y="{m + ph + 22}" fill="{C["text2"]}" font-size="9">30-day MA</text>'
        f'<text x="{m + 175}" y="{m + ph + 22}" fill="{C["text"]}" font-size="10" font-weight="700">{grade} ({current:.0f}/100) {arrow}</text>'
    )

    # Date labels
    date_labels = ""
    step = max(n // 6, 1)
    for i in range(0, n, step):
        x = m + i / max(n - 1, 1) * pw
        date_labels += f'<text x="{x:.0f}" y="{m + ph + 12}" text-anchor="middle" fill="{C["text2"]}" font-size="8">{dates[i][5:]}</text>'

    return _svg(width, height, zones + band_fill + bars + ma7_line + ma30_line + callout + legend + date_labels,
        "Architecture Score (Bollinger Bands)")
