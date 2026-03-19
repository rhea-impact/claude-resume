"""Generate a stunning HTML report with real data science visualizations.

Self-contained HTML with embedded SVG charts, cluster plots, heatmaps,
Markov diagrams, distribution charts, and animated transitions.
"""

import json
from datetime import datetime
from pathlib import Path

from .scanner import scan_deep
from . import analytics
from . import models
from . import charts


def _match_org(session: dict, org_lower: str) -> bool:
    """Check if a session belongs to the given org."""
    path = session.get("project_short", "") or session.get("project_dir", "")
    parts = path.replace("~/", "").lower().split("/")
    for p in parts:
        if p.startswith("repos"):
            org_name = p.replace("repos-", "").replace("repos", "misc")
            if org_lower in org_name or org_name in org_lower:
                return True
    return False


def _svg_scatter(points_2d, labels, clusters, width=500, height=350):
    """SVG scatter plot of session clusters in PCA space."""
    if not points_2d:
        return "<p>No cluster data</p>"

    import numpy as np
    pts = np.array(points_2d)
    if len(pts) == 0:
        return ""

    colors = ["#6366f1", "#22c55e", "#f59e0b", "#f43f5e", "#06b6d4", "#a855f7", "#ec4899", "#14b8a6"]
    margin = 40
    pw = width - 2 * margin
    ph = height - 2 * margin

    x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
    y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
    x_range = x_max - x_min or 1
    y_range = y_max - y_min or 1

    # Sample points for performance (max 800)
    sample_size = min(800, len(pts))
    indices = np.random.choice(len(pts), sample_size, replace=False) if len(pts) > sample_size else np.arange(len(pts))

    dots = []
    for i in indices:
        x = margin + (pts[i, 0] - x_min) / x_range * pw
        y = margin + (1 - (pts[i, 1] - y_min) / y_range) * ph
        c = colors[labels[i] % len(colors)] if labels[i] >= 0 else "#555"
        dots.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{c}" opacity="0.6"/>')

    # Cluster centroids
    centroids = []
    for cl in clusters:
        cx, cy = cl["centroid_2d"]
        x = margin + (cx - x_min) / x_range * pw
        y = margin + (1 - (cy - y_min) / y_range) * ph
        idx = cl["id"]
        color = colors[idx % len(colors)]
        centroids.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="{color}" stroke="white" stroke-width="2"/>'
            f'<text x="{x:.1f}" y="{y - 14:.1f}" text-anchor="middle" fill="{color}" '
            f'font-size="11" font-weight="700">{cl["label"]}</text>'
        )

    return f'''<svg width="{width}" height="{height}" class="cluster-plot">
        <rect width="{width}" height="{height}" rx="12" fill="#12121a"/>
        {"".join(dots)}
        {"".join(centroids)}
    </svg>'''


def _svg_circadian(actual, fitted, width=600, height=250):
    """SVG overlay of actual vs fitted circadian curve."""
    if not actual or not fitted:
        return ""

    margin = 50
    pw = width - 2 * margin
    ph = height - 2 * margin
    mx = max(max(actual), max(fitted)) or 1

    # Actual bars
    bar_w = pw / 24 - 2
    bars = []
    for i, v in enumerate(actual):
        h = v / mx * ph
        x = margin + i * (bar_w + 2)
        y = margin + ph - h
        bars.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" '
                    f'rx="2" fill="#6366f1" opacity="0.4"/>')

    # Fitted curve
    points = []
    for i, v in enumerate(fitted):
        x = margin + i * (bar_w + 2) + bar_w / 2
        y = margin + ph - (v / mx * ph)
        points.append(f"{x:.1f},{y:.1f}")
    polyline = f'<polyline points="{" ".join(points)}" fill="none" stroke="#f59e0b" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>'

    # Hour labels
    labels = []
    for i in range(0, 24, 3):
        x = margin + i * (bar_w + 2) + bar_w / 2
        labels.append(f'<text x="{x:.1f}" y="{margin + ph + 20}" text-anchor="middle" fill="#8888a0" font-size="11">{i:02d}</text>')

    return f'''<svg width="{width}" height="{height}">
        <rect width="{width}" height="{height}" rx="12" fill="#12121a"/>
        {"".join(bars)}
        {polyline}
        {"".join(labels)}
        <circle cx="{width - 130}" cy="20" r="5" fill="#6366f1" opacity="0.5"/>
        <text x="{width - 120}" y="24" fill="#8888a0" font-size="11">Actual</text>
        <line x1="{width - 70}" y1="20" x2="{width - 50}" y2="20" stroke="#f59e0b" stroke-width="3"/>
        <text x="{width - 45}" y="24" fill="#8888a0" font-size="11">Fitted</text>
    </svg>'''


def _svg_markov(transitions, width=600, height=400):
    """SVG Markov chain visualization — top transitions as a force diagram."""
    if not transitions:
        return ""

    import math
    nodes = set()
    for t in transitions[:12]:
        nodes.add(t["from"])
        nodes.add(t["to"])
    node_list = list(nodes)
    n = len(node_list)
    if n == 0:
        return ""

    cx, cy = width / 2, height / 2
    radius = min(width, height) / 2 - 60
    positions = {}
    for i, node in enumerate(node_list):
        angle = 2 * math.pi * i / n - math.pi / 2
        positions[node] = (cx + radius * math.cos(angle), cy + radius * math.sin(angle))

    # Edges
    edges = []
    for t in transitions[:12]:
        x1, y1 = positions.get(t["from"], (0, 0))
        x2, y2 = positions.get(t["to"], (0, 0))
        prob = t["probability"]
        opacity = 0.3 + 0.7 * prob
        stroke_w = 1 + 4 * prob
        # Arrow
        edges.append(
            f'<line x1="{x1:.0f}" y1="{y1:.0f}" x2="{x2:.0f}" y2="{y2:.0f}" '
            f'stroke="#6366f1" stroke-width="{stroke_w:.1f}" opacity="{opacity:.2f}" '
            f'marker-end="url(#arrow)"/>'
        )

    # Nodes
    node_elems = []
    for node in node_list:
        x, y = positions[node]
        short = node[:12]
        node_elems.append(
            f'<circle cx="{x:.0f}" cy="{y:.0f}" r="24" fill="#1a1a28" stroke="#6366f1" stroke-width="2"/>'
            f'<text x="{x:.0f}" y="{y + 4:.0f}" text-anchor="middle" fill="#e4e4ef" '
            f'font-size="9" font-weight="600">{short}</text>'
        )

    return f'''<svg width="{width}" height="{height}">
        <defs><marker id="arrow" markerWidth="8" markerHeight="6" refX="30" refY="3" orient="auto">
            <polygon points="0 0, 8 3, 0 6" fill="#6366f1" opacity="0.7"/>
        </marker></defs>
        <rect width="{width}" height="{height}" rx="12" fill="#12121a"/>
        {"".join(edges)}
        {"".join(node_elems)}
    </svg>'''


def _svg_histogram(bins, counts, width=500, height=200):
    """SVG histogram from pre-computed bins/counts."""
    if not bins or not counts:
        return ""

    margin = 40
    pw = width - 2 * margin
    ph = height - 2 * margin
    mx = max(counts) or 1
    n = len(counts)
    bar_w = pw / n - 1

    bars = []
    for i, c in enumerate(counts):
        h = c / mx * ph
        x = margin + i * (bar_w + 1)
        y = margin + ph - h
        bars.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" '
                    f'rx="1" fill="#06b6d4" opacity="0.7"/>')

    return f'''<svg width="{width}" height="{height}">
        <rect width="{width}" height="{height}" rx="12" fill="#12121a"/>
        {"".join(bars)}
    </svg>'''


def _svg_entropy_gauge(value, label, width=160, height=160):
    """SVG radial gauge for entropy normalized value (0-1)."""
    import math
    cx, cy, r = width / 2, height / 2 + 10, 55
    circumference = 2 * math.pi * r
    filled = circumference * value
    color = "#22c55e" if value < 0.5 else "#f59e0b" if value < 0.8 else "#f43f5e"

    return f'''<svg width="{width}" height="{height}">
        <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="#2a2a3a" stroke-width="10"/>
        <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{color}" stroke-width="10"
            stroke-dasharray="{filled:.1f} {circumference:.1f}" stroke-linecap="round"
            transform="rotate(-90 {cx} {cy})"/>
        <text x="{cx}" y="{cy - 5}" text-anchor="middle" fill="#e4e4ef" font-size="22" font-weight="800">{int(value * 100)}%</text>
        <text x="{cx}" y="{cy + 15}" text-anchor="middle" fill="#8888a0" font-size="10">{label}</text>
    </svg>'''


def generate_report(output_path: str | None = None, max_sessions: int = 0, org: str = "") -> str:
    """Generate comprehensive HTML report with real data science.

    Args:
        org: Filter to a specific org (e.g. "eidos-agi", "personal", "aic", "greenmark-waste-solutions").
             Empty string = all orgs.
    """
    sessions = scan_deep(max_sessions=max_sessions)
    if not sessions:
        return "No sessions to analyze."

    # Filter by org if specified
    if org:
        org_lower = org.lower()
        sessions = [s for s in sessions if _match_org(s, org_lower)]
        if not sessions:
            return f"No sessions found for org '{org}'."

    # Run all analyses
    desc = analytics.analyze(sessions)
    ds = models.full_analysis(sessions)

    ov = desc["overview"]
    temp = desc["temporal"]
    personality = desc["personality"]
    fun = desc["fun_facts"]
    proj = desc["projects"]
    tools_data = desc["tools"]

    now = datetime.now().strftime("%B %d, %Y at %I:%M %p")

    # --- Build visualizations ---

    # Cluster scatter
    cl = ds.get("clustering", {})
    cluster_svg = _svg_scatter(
        cl.get("points_2d", []), cl.get("labels", []),
        cl.get("clusters", [])
    )
    cluster_cards = ""
    for c in cl.get("clusters", [])[:7]:
        cluster_cards += f'''
        <div class="cluster-card">
            <div class="cluster-label">{c["label"]}</div>
            <div class="cluster-pct">{c["pct"]}%</div>
            <div class="cluster-meta">{c["count"]} sessions · {c["avg_duration_mins"]}min avg · {c["avg_tool_calls"]} tools/session</div>
        </div>'''

    # Circadian
    circ = ds.get("circadian", {})
    circadian_svg = _svg_circadian(circ.get("actual_hourly", []), circ.get("fitted_hourly", []))

    # Markov
    mk = ds.get("markov_chain", {})
    markov_svg = _svg_markov(mk.get("strong_transitions", []))
    markov_list = ""
    for t in mk.get("strong_transitions", [])[:8]:
        markov_list += f'<div class="markov-row"><span class="markov-from">{t["from"]}</span> → <span class="markov-to">{t["to"]}</span> <span class="markov-prob">{int(t["probability"] * 100)}%</span></div>'

    # Power law
    pl = ds.get("power_law", {})
    concentration = pl.get("concentration", {})

    # Duration distribution
    dd = ds.get("duration_dist", {})
    duration_svg = _svg_histogram(
        dd.get("histogram", {}).get("bins", []),
        dd.get("histogram", {}).get("counts", []),
    )
    categories = dd.get("categories", {})

    # Flow states
    fl = ds.get("flow_states", {})
    flow_by_hour = fl.get("flow_rate_by_hour", {})
    flow_projects = fl.get("flow_rate_by_project", {})

    # Burnout
    bo = ds.get("burnout", {})
    burnout_signals = bo.get("signals", [])

    # Entropy
    ent = ds.get("entropy", {})
    proj_ent = ent.get("project_entropy", {})
    hour_ent = ent.get("hour_entropy", {})
    day_ent = ent.get("day_entropy", {})

    entropy_gauges = (
        _svg_entropy_gauge(proj_ent.get("normalized", 0), "Project") +
        _svg_entropy_gauge(hour_ent.get("normalized", 0), "Hour") +
        _svg_entropy_gauge(day_ent.get("normalized", 0), "Day")
    )

    # --- Generate 40 charts ---
    chart_cumulative = charts.cumulative_sessions(sessions)
    chart_tokens_day = charts.tokens_per_day(sessions)
    chart_heatmap = charts.hour_day_heatmap(sessions)
    chart_timeline = charts.project_timeline(sessions)
    chart_rolling = charts.rolling_sessions(sessions)
    chart_duration_proj = charts.duration_by_project(sessions)
    chart_message_donut = charts.message_donut(sessions)
    chart_monthly = charts.monthly_bars(sessions)
    chart_mcp = charts.mcp_tools_chart(sessions)
    chart_size_dist = charts.size_distribution(sessions)
    chart_productivity_day = charts.productivity_by_day(sessions)
    chart_start_density = charts.start_time_density(sessions)
    chart_tool_donut = charts.tool_category_donut(sessions)
    chart_radar = charts.hourly_radar(sessions)
    chart_token_eff = charts.token_efficiency(sessions)
    chart_subagent = charts.subagent_trend(sessions)
    chart_daily_scatter = charts.daily_scatter(sessions)
    chart_flow_hour = charts.flow_by_hour_chart(fl)
    chart_burnout_trend = charts.burnout_trend(bo)
    chart_cooccurrence = charts.cooccurrence_graph(ds.get("cooccurrence", {}))
    # New 20
    chart_calendar = charts.contribution_calendar(sessions)
    chart_dur_trend = charts.duration_trend(sessions)
    chart_velocity = charts.velocity_scatter(sessions)
    chart_cost = charts.cost_trend(sessions)
    chart_weekend = charts.weekend_vs_weekday(sessions)
    chart_diversity = charts.project_diversity(sessions)
    chart_cache = charts.cache_efficiency(sessions)
    chart_depth = charts.conversation_depth(sessions)
    chart_model_donut = charts.model_usage_donut(sessions)
    chart_gaps = charts.session_gaps(sessions)
    chart_nightowl = charts.night_owl_trend(sessions)
    chart_marathon = charts.marathon_sessions(sessions)
    chart_branch = charts.branch_activity(sessions)
    chart_concentration = charts.project_concentration(sessions)
    chart_first_time = charts.first_session_time(sessions)
    chart_tool_adopt = charts.tool_adoption(sessions)
    chart_treemap = charts.token_treemap(sessions)
    chart_intensity = charts.intensity_scatter(sessions)
    chart_weekly_stack = charts.weekly_stacked(sessions)
    chart_response = charts.response_ratio(sessions)
    # 10 new dimensions
    chart_prompt_evo = charts.prompt_evolution(sessions)
    chart_focus = charts.focus_score(sessions)
    chart_gravity = charts.project_gravity(sessions)
    chart_momentum = charts.momentum_streaks(sessions)
    chart_throughput = charts.throughput_by_hour(sessions)
    chart_multitask = charts.multitask_penalty(sessions)
    chart_tool_evo = charts.tool_evolution(sessions)
    chart_recovery = charts.recovery_time(sessions)
    chart_lifecycle = charts.project_lifecycle(sessions)
    chart_delegation = charts.delegation_index(sessions)
    # Org/repo charts
    chart_org_pie = charts.org_breakdown(sessions)
    chart_org_timeline = charts.org_timeline(sessions)
    chart_repo_board = charts.repo_leaderboard(sessions)
    chart_org_hour = charts.org_hour_heatmap(sessions)
    chart_repo_churn = charts.repo_churn(sessions)
    chart_org_treemap = charts.org_repo_treemap(sessions)
    chart_org_switches = charts.org_switches(sessions)
    # Human output & leverage
    chart_human_out = charts.human_output(sessions)
    chart_output_comp = charts.output_composition(sessions)
    chart_leverage = charts.leverage_points(sessions)
    chart_efficiency = charts.efficiency_trend(sessions)
    chart_cum_days = charts.cumulative_human_days(sessions)
    chart_eng_score = charts.engineer_score(sessions)
    # Psychological & prowess charts
    chart_archetype = charts.developer_archetype(sessions)
    chart_cognitive = charts.cognitive_load(sessions)
    chart_fatigue = charts.decision_fatigue(sessions)
    chart_creative = charts.creative_analytical(sessions)
    chart_prowess = charts.prowess_pentagon(sessions)
    chart_teaching = charts.teaching_insights(sessions)
    chart_growth = charts.growth_trajectory(sessions)
    # FAANG level & architecture score
    chart_faang = charts.faang_level(sessions)
    chart_arch = charts.architecture_score(sessions)

    # Anomalies
    an = ds.get("anomalies", {})
    anomaly_rows = ""
    for a in an.get("anomalies", [])[:8]:
        reasons_str = ", ".join(a.get("reasons", []))
        anomaly_rows += f'''
        <div class="anomaly-row">
            <span class="anomaly-project">{a["project"]}</span>
            <span class="anomaly-date">{a["date"]}</span>
            <span class="anomaly-dur">{a["duration_mins"]}min</span>
            <span class="anomaly-reasons">{reasons_str}</span>
        </div>'''

    # Project bars (from descriptive)
    top_proj = proj.get("top_projects", [])
    max_proj_sessions = max((p["sessions"] for p in top_proj), default=1)
    project_rows = ""
    for p in top_proj[:12]:
        pct = p["sessions"] / max_proj_sessions * 100
        project_rows += f'''
        <div class="proj-row">
            <div class="proj-name">{p["project"]}</div>
            <div class="proj-bar-track"><div class="proj-bar" style="width:{pct}%"></div></div>
            <div class="proj-stats"><span class="proj-num">{p["sessions"]}</span> · {p["hours"]}h · {p["tokens"]}</div>
        </div>'''

    # Tool bars
    top_tools = tools_data.get("top_tools", [])[:12]
    max_tool = max((t["uses"] for t in top_tools), default=1)
    tool_rows = ""
    for t in top_tools:
        pct = t["uses"] / max_tool * 100
        tool_rows += f'''
        <div class="tool-row">
            <div class="tool-name">{t["tool"]}</div>
            <div class="tool-bar-track"><div class="tool-bar" style="width:{pct}%"></div></div>
            <div class="tool-count">{t["uses"]:,}</div>
        </div>'''

    # Personality
    trait_icons = ["🦉", "⚡", "🔀", "🧠", "🎯", "💻", "🌙", "🏗️", "🎭", "🔬"]
    trait_cards = ""
    for i, trait in enumerate(personality.get("traits", [])):
        parts = trait.split(" — ", 1)
        title = parts[0]
        desc_text = parts[1] if len(parts) > 1 else ""
        trait_cards += f'''
        <div class="trait-card">
            <div class="trait-icon">{trait_icons[i % len(trait_icons)]}</div>
            <div class="trait-title">{title}</div>
            <div class="trait-desc">{desc_text}</div>
        </div>'''

    # Fun facts
    fact_cards = "".join(f'<div class="fact-card">{f}</div>' for f in fun)

    # Predictions
    preds = desc.get("predictions", {})
    pred_rows = ""
    for p in preds.get("next_project", []):
        conf = int(p["confidence"].replace("%", ""))
        pred_rows += f'''
        <div class="pred-row">
            <div class="pred-name">{p["project"]}</div>
            <div class="pred-bar-track"><div class="pred-bar" style="width:{conf}%"></div></div>
            <div class="pred-conf">{p["confidence"]}</div>
        </div>'''

    # Weekly heatmap
    weekly = temp.get("weekly_pattern", {})
    max_weekly = max(weekly.values()) if weekly else 1
    weekly_bars = ""
    for day, count in weekly.items():
        pct = count / max_weekly * 100
        weekly_bars += f'''
        <div class="week-bar-group">
            <div class="week-bar-container"><div class="week-bar" style="height:{pct}%"></div></div>
            <div class="week-label">{day[:3]}</div>
            <div class="week-count">{count}</div>
        </div>'''

    # Hourly heatmap
    heatmap = temp.get("hourly_heatmap", {})
    heat_cells = ""
    for h in range(24):
        data = heatmap.get(f"{h:02d}:00", {})
        intensity = data.get("intensity", 0)
        sessions_count = data.get("sessions", 0)
        alpha = 0.08 + 0.92 * intensity
        heat_cells += (
            f'<div class="heat-cell" style="background:rgba(99,102,241,{alpha})" '
            f'title="{h:02d}:00 — {sessions_count} sessions">'
            f'<span class="heat-label">{sessions_count if sessions_count else ""}</span></div>'
        )
    heat_hours = "".join(f'<div class="heat-hour">{h:02d}</div>' for h in range(24))

    html = f'''<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Claude Sessions — Data Science Report</title>
<style>
:root {{ --bg:#0a0a0f; --s:#12121a; --s2:#1a1a28; --b:#2a2a3a; --t:#e4e4ef; --t2:#8888a0;
  --a:#6366f1; --a2:#818cf8; --ag:rgba(99,102,241,0.15); --g:#22c55e; --am:#f59e0b;
  --r:#f43f5e; --c:#06b6d4; --p:#a855f7; }}
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:var(--bg);color:var(--t);font-family:'Inter',-apple-system,system-ui,sans-serif;line-height:1.6;overflow-x:hidden}}
.container{{max-width:1200px;margin:0 auto;padding:40px 24px}}
.header{{text-align:center;padding:80px 0 40px;position:relative}}
.header::before{{content:'';position:absolute;top:0;left:50%;transform:translateX(-50%);width:800px;height:800px;
  background:radial-gradient(circle,var(--ag) 0%,transparent 70%);pointer-events:none}}
.header h1{{font-size:3rem;font-weight:800;letter-spacing:-0.03em;
  background:linear-gradient(135deg,var(--a2),var(--c),var(--p));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;position:relative}}
.header .sub{{color:var(--t2);margin-top:8px;position:relative}}
.header .gen{{color:var(--t2);font-size:.75rem;margin-top:4px;opacity:.5;position:relative}}

.stat-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:32px 0}}
.stat-card{{background:var(--s);border:1px solid var(--b);border-radius:14px;padding:20px;transition:transform .2s,border-color .2s}}
.stat-card:hover{{transform:translateY(-2px);border-color:var(--a)}}
.stat-value{{font-size:1.8rem;font-weight:800;background:linear-gradient(135deg,var(--t),var(--a2));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}}
.stat-label{{color:var(--t2);font-size:.75rem;text-transform:uppercase;letter-spacing:.05em;margin-top:2px}}

.section{{margin:56px 0}} .section-title{{font-size:1.5rem;font-weight:700;margin-bottom:20px;display:flex;align-items:center;gap:10px}}
.section-title .icon{{font-size:1.3rem}} .panel{{background:var(--s);border:1px solid var(--b);border-radius:14px;padding:28px;margin-bottom:20px}}
.two-col{{display:grid;grid-template-columns:1fr 1fr;gap:20px}} @media(max-width:768px){{.two-col{{grid-template-columns:1fr}}}}
.three-col{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:20px}} @media(max-width:768px){{.three-col{{grid-template-columns:1fr}}}}
.insight{{background:linear-gradient(135deg,var(--s),var(--s2));border:1px solid var(--b);border-left:4px solid var(--am);
  border-radius:0 12px 12px 0;padding:16px 20px;margin:12px 0;font-size:.95rem;line-height:1.7}}

.streak-banner{{background:linear-gradient(135deg,var(--a),#7c3aed,var(--r));border-radius:18px;padding:28px;
  text-align:center;margin:32px 0;position:relative;overflow:hidden}}
.streak-number{{font-size:3.5rem;font-weight:900;line-height:1}}
.streak-label{{font-size:1.1rem;font-weight:600;opacity:.9}}

.heat-row{{display:flex;gap:3px;justify-content:center;flex-wrap:wrap}}
.heat-cell{{width:38px;height:38px;border-radius:5px;display:flex;align-items:center;justify-content:center;
  font-size:.65rem;transition:transform .15s;cursor:default}} .heat-cell:hover{{transform:scale(1.3);z-index:10}}
.heat-label{{color:rgba(255,255,255,.8);font-weight:600}}
.heat-hours{{display:flex;gap:3px;justify-content:center;margin-top:4px;flex-wrap:wrap}}
.heat-hour{{width:38px;text-align:center;font-size:.6rem;color:var(--t2)}}

.week-chart{{display:flex;align-items:flex-end;justify-content:center;gap:10px;height:160px;padding:16px 0}}
.week-bar-group{{text-align:center;flex:1;max-width:70px}}
.week-bar-container{{height:110px;display:flex;align-items:flex-end;justify-content:center}}
.week-bar{{width:32px;background:linear-gradient(180deg,var(--a2),var(--a));border-radius:5px 5px 0 0;min-height:3px;
  transition:height .8s cubic-bezier(.4,0,.2,1)}}
.week-label{{color:var(--t2);font-size:.75rem;margin-top:6px;font-weight:600}} .week-count{{color:var(--t);font-size:.7rem;font-weight:700}}

.proj-row{{display:grid;grid-template-columns:160px 1fr auto;align-items:center;gap:14px;padding:8px 0;border-bottom:1px solid var(--b)}}
.proj-row:last-child{{border-bottom:none}} .proj-name{{font-weight:600;font-size:.85rem;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.proj-bar-track{{height:7px;background:var(--s2);border-radius:4px;overflow:hidden}}
.proj-bar{{height:100%;background:linear-gradient(90deg,var(--a),var(--c));border-radius:4px;transition:width 1s cubic-bezier(.4,0,.2,1)}}
.proj-stats{{font-size:.75rem;color:var(--t2);white-space:nowrap}} .proj-num{{color:var(--t);font-weight:700}}

.tool-row{{display:grid;grid-template-columns:130px 1fr 65px;align-items:center;gap:10px;padding:6px 0}}
.tool-name{{font-family:'JetBrains Mono',monospace;font-size:.8rem;font-weight:600;color:var(--a2)}}
.tool-bar-track{{height:5px;background:var(--s2);border-radius:3px;overflow:hidden}}
.tool-bar{{height:100%;background:linear-gradient(90deg,var(--g),var(--c));border-radius:3px}} .tool-count{{text-align:right;font-size:.8rem;font-weight:700;color:var(--t2)}}

.cluster-card{{background:var(--s2);border:1px solid var(--b);border-radius:10px;padding:14px;text-align:center}}
.cluster-label{{font-weight:700;color:var(--a2);font-size:.9rem}} .cluster-pct{{font-size:1.6rem;font-weight:800;color:var(--t)}}
.cluster-meta{{font-size:.7rem;color:var(--t2);margin-top:4px}}

.trait-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px}}
.trait-card{{background:var(--s);border:1px solid var(--b);border-radius:14px;padding:20px;transition:border-color .2s,box-shadow .2s}}
.trait-card:hover{{border-color:var(--a);box-shadow:0 0 30px var(--ag)}}
.trait-icon{{font-size:1.8rem;margin-bottom:6px}} .trait-title{{font-size:1rem;font-weight:700;color:var(--a2)}}
.trait-desc{{color:var(--t2);font-size:.85rem;margin-top:3px}}

.fact-card{{background:linear-gradient(135deg,var(--s),var(--s2));border:1px solid var(--b);border-left:4px solid var(--a);
  border-radius:0 12px 12px 0;padding:16px 20px;margin-bottom:10px;font-size:1rem;line-height:1.7;transition:border-left-color .2s}}
.fact-card:hover{{border-left-color:var(--c)}}

.pred-row{{display:grid;grid-template-columns:180px 1fr 55px;align-items:center;gap:10px;padding:8px 0}}
.pred-name{{font-weight:600}} .pred-bar-track{{height:7px;background:var(--s2);border-radius:4px;overflow:hidden}}
.pred-bar{{height:100%;background:linear-gradient(90deg,var(--am),var(--r));border-radius:4px}} .pred-conf{{text-align:right;font-weight:800;color:var(--am)}}

.record-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:14px}}
.record-card{{background:var(--s);border:1px solid var(--b);border-radius:14px;padding:20px;text-align:center}}
.record-value{{font-size:1.6rem;font-weight:800;color:var(--am)}} .record-label{{color:var(--t2);font-size:.8rem;margin-top:2px}}

.burnout-meter{{height:12px;background:var(--s2);border-radius:6px;overflow:hidden;margin:12px 0}}
.burnout-fill{{height:100%;border-radius:6px;transition:width 1s}}

.markov-row{{padding:6px 0;border-bottom:1px solid var(--b);font-size:.9rem}}
.markov-from{{font-weight:700;color:var(--a2)}} .markov-to{{font-weight:700;color:var(--c)}}
.markov-prob{{float:right;font-weight:800;color:var(--am)}}

.anomaly-row{{display:grid;grid-template-columns:120px 80px 70px 1fr;gap:8px;padding:6px 0;border-bottom:1px solid var(--b);font-size:.8rem}}
.anomaly-project{{font-weight:700;color:var(--a2)}} .anomaly-date{{color:var(--t2)}}
.anomaly-dur{{color:var(--am);font-weight:700}} .anomaly-reasons{{color:var(--t2)}}

.cluster-plot{{display:block;margin:0 auto}}
.footer{{text-align:center;padding:50px 0 30px;color:var(--t2);font-size:.75rem;opacity:.4}}
.label{{font-size:.8rem;color:var(--t2);text-transform:uppercase;letter-spacing:.05em;margin-bottom:10px}}
.gauges{{display:flex;justify-content:center;gap:20px;flex-wrap:wrap}}
</style></head><body><div class="container">

<div class="header">
    <h1>Claude Sessions: Data Science</h1>
    <div class="sub">{f'<strong style="color:var(--a2)">{org.upper()}</strong> · ' if org else ''}{ov["total_sessions"]:,} sessions · {ov["total_projects"]} projects · {ov["date_range"]}</div>
    <div class="gen">Generated {now} · K-Means · Markov Chains · Circadian Modeling · Power Law · DBSCAN · Shannon Entropy</div>
</div>

<div class="streak-banner">
    <div class="streak-number">{preds.get("current_streak_days", 0)}</div>
    <div class="streak-label">{preds.get("streak_message", "")}</div>
</div>

<div class="stat-grid">
    <div class="stat-card"><div class="stat-value">{ov["total_sessions"]:,}</div><div class="stat-label">Sessions</div></div>
    <div class="stat-card"><div class="stat-value">{ov["active_days"]}</div><div class="stat-label">Active Days</div></div>
    <div class="stat-card"><div class="stat-value">{ov["total_tokens"]}</div><div class="stat-label">Tokens</div></div>
    <div class="stat-card"><div class="stat-value">{ov["novels_equivalent"]}</div><div class="stat-label">Novels Written</div></div>
    <div class="stat-card"><div class="stat-value">${ov["estimated_cost_usd"]:,.0f}</div><div class="stat-label">Est. Cost</div></div>
    <div class="stat-card"><div class="stat-value">{ov["total_tool_uses"]:,}</div><div class="stat-label">Tool Calls</div></div>
    <div class="stat-card"><div class="stat-value">{ov["total_messages"]:,}</div><div class="stat-label">Messages</div></div>
    <div class="stat-card"><div class="stat-value">{round(ent.get("overall_predictability", 0))}%</div><div class="stat-label">Predictability</div></div>
</div>

<!-- DEVELOPER ARCHETYPE & PROWESS -->
<div class="section">
    <div class="section-title"><span class="icon">🧬</span> Developer DNA</div>
    <div class="two-col">
        <div class="panel" style="text-align:center">{chart_archetype}</div>
        <div class="panel" style="text-align:center">{chart_prowess}</div>
    </div>
</div>

<!-- FAANG LEVEL -->
<div class="section">
    <div class="section-title"><span class="icon">📊</span> FAANG Level Estimation</div>
    <div class="panel" style="text-align:center">{chart_faang}</div>
</div>

<!-- ARCHITECTURE SCORE -->
<div class="section">
    <div class="section-title"><span class="icon">🏛️</span> Architecture Score</div>
    <div class="panel" style="text-align:center">{chart_arch}</div>
</div>

<!-- ENGINEER SCORE -->
<div class="section">
    <div class="section-title"><span class="icon">🏆</span> Engineer Score</div>
    <div class="panel" style="text-align:center">{chart_eng_score}</div>
</div>

<!-- TEACHING INSIGHTS -->
<div class="section">
    <div class="section-title"><span class="icon">💡</span> What Your Data Reveals</div>
    <div class="panel" style="text-align:center">{chart_teaching}</div>
</div>

<!-- GROWTH TRAJECTORY -->
<div class="section">
    <div class="section-title"><span class="icon">📈</span> Growth Trajectory</div>
    <div class="panel" style="text-align:center">{chart_growth}</div>
</div>

<!-- HUMAN OUTPUT -->
<div class="section">
    <div class="section-title"><span class="icon">⚡</span> Human-Equivalent Output</div>
    <div class="panel" style="text-align:center">{chart_human_out}</div>
    <div class="two-col">
        <div class="panel" style="text-align:center">{chart_cum_days}</div>
        <div class="panel" style="text-align:center">{chart_output_comp}</div>
    </div>
    <div class="panel" style="text-align:center">{chart_efficiency}</div>
</div>

<!-- LEVERAGE POINTS -->
<div class="section">
    <div class="section-title"><span class="icon">🎯</span> Leverage Points — Where to Invest</div>
    <div class="panel" style="text-align:center">{chart_leverage}</div>
</div>

<!-- PERSONALITY -->
<div class="section">
    <div class="section-title"><span class="icon">🧬</span> AI Personality Profile</div>
    <div class="trait-grid">{trait_cards}</div>
</div>

<!-- FUN FACTS -->
<div class="section">
    <div class="section-title"><span class="icon">🎯</span> Fun Facts</div>
    {fact_cards}
</div>

<!-- CLUSTERING -->
<div class="section">
    <div class="section-title"><span class="icon">🔬</span> Session Clustering (K-Means, k={cl.get("n_clusters", "?")})</div>
    <div class="insight">Silhouette score: {cl.get("silhouette_score", "?")} · {cl.get("variance_explained_2d", "?")}% variance explained in 2D PCA projection</div>
    <div class="two-col">
        <div class="panel" style="text-align:center">{cluster_svg}</div>
        <div class="panel">
            <div class="label">Discovered Session Types</div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">{cluster_cards}</div>
        </div>
    </div>
</div>

<!-- CIRCADIAN -->
<div class="section">
    <div class="section-title"><span class="icon">🌙</span> Circadian Rhythm Model</div>
    <div class="insight">{circ.get("insight", "")}<br>Model fit: R²={circ.get("r_squared", "?")} ({circ.get("model_fit", "?")})</div>
    <div class="two-col">
        <div class="panel" style="text-align:center">
            <div class="label">Actual vs Sinusoidal Fit</div>
            {circadian_svg}
        </div>
        <div class="panel">
            <div class="label">24-Hour Heatmap</div>
            <div class="heat-row">{heat_cells}</div>
            <div class="heat-hours">{heat_hours}</div>
        </div>
    </div>
</div>

<!-- MARKOV -->
<div class="section">
    <div class="section-title"><span class="icon">🔗</span> Project Markov Chain</div>
    <div class="insight">Transition probabilities: after finishing one project, where do you go next?</div>
    <div class="two-col">
        <div class="panel" style="text-align:center">{markov_svg}</div>
        <div class="panel">
            <div class="label">Strongest Transitions</div>
            {markov_list}
        </div>
    </div>
</div>

<!-- POWER LAW & DURATION -->
<div class="section">
    <div class="section-title"><span class="icon">📊</span> Duration Distribution & Power Law</div>
    <div class="insight">{pl.get("insight", "")}</div>
    <div class="two-col">
        <div class="panel" style="text-align:center">
            <div class="label">Duration Distribution (log scale)</div>
            {duration_svg}
            <div style="margin-top:12px;font-size:.8rem;color:var(--t2)">
                Bimodal: <strong>{"Yes" if dd.get("is_bimodal") else "No"}</strong> ·
                Skewness: {dd.get("stats", {}).get("skewness", "?")} ·
                Kurtosis: {dd.get("stats", {}).get("kurtosis", "?")}
            </div>
        </div>
        <div class="panel">
            <div class="label">Session Categories</div>
            <div class="record-grid" style="margin-top:12px">
                {"".join(f'<div class="record-card"><div class="record-value">{v}</div><div class="record-label">{k}</div></div>' for k, v in categories.items())}
            </div>
            <div style="margin-top:16px;font-size:.85rem;color:var(--t2)">
                <strong>Pareto exponent α={pl.get("alpha", "?")}</strong><br>
                Top 1% → {concentration.get("top_1pct_share", "?")}% of time<br>
                Top 10% → {concentration.get("top_10pct_share", "?")}% of time
            </div>
        </div>
    </div>
</div>

<!-- FLOW STATES -->
<div class="section">
    <div class="section-title"><span class="icon">🧘</span> Flow State Detection</div>
    <div class="insight">{fl.get("insight", "")}</div>
    <div class="panel">
        <div class="label">Flow Rate by Project</div>
        {"".join(f'<div class="proj-row"><div class="proj-name">{p}</div><div class="proj-bar-track"><div class="proj-bar" style="width:{d["flow_rate"]}%"></div></div><div class="proj-stats">{d["flow_rate"]}% flow · {d["flow_sessions"]}/{d["total_sessions"]}</div></div>' for p, d in list(flow_projects.items())[:8])}
    </div>
</div>

<!-- ENTROPY -->
<div class="section">
    <div class="section-title"><span class="icon">🎲</span> Shannon Entropy — How Predictable Are You?</div>
    <div class="insight">{ent.get("insight", "")}</div>
    <div class="panel" style="text-align:center">
        <div class="gauges">{entropy_gauges}</div>
        <div style="margin-top:12px;font-size:.85rem;color:var(--t2)">
            Higher entropy = more chaotic/unpredictable · Lower = more routine
        </div>
    </div>
</div>

<!-- BURNOUT -->
<div class="section">
    <div class="section-title"><span class="icon">🔥</span> Burnout Risk Assessment</div>
    <div class="panel">
        <div style="display:flex;align-items:center;gap:16px;margin-bottom:16px">
            <div class="stat-value" style="font-size:2.5rem;{
                "color:#22c55e" if bo.get("burnout_score",0) < 30 else
                "color:#f59e0b" if bo.get("burnout_score",0) < 60 else "color:#f43f5e"
            }">{bo.get("burnout_score", 0)}/100</div>
            <div><div style="font-size:1.2rem;font-weight:700">{bo.get("risk_level", "?")}</div>
                <div style="color:var(--t2);font-size:.85rem">Burnout Risk Score</div></div>
        </div>
        <div class="burnout-meter"><div class="burnout-fill" style="width:{bo.get("burnout_score",0)}%;background:{
            "#22c55e" if bo.get("burnout_score",0) < 30 else "#f59e0b" if bo.get("burnout_score",0) < 60 else "#f43f5e"
        }"></div></div>
        {"".join(f'<div class="insight" style="border-left-color:var(--r)">{s}</div>' for s in burnout_signals)}
    </div>
</div>

<!-- ANOMALIES -->
<div class="section">
    <div class="section-title"><span class="icon">👾</span> Anomaly Detection (DBSCAN)</div>
    <div class="insight">{an.get("total_anomalies", 0)} anomalous sessions detected ({an.get("anomaly_rate", 0)}% of total) — statistical outliers in duration, token usage, or tool intensity</div>
    <div class="panel">{anomaly_rows}</div>
</div>

<!-- PROJECTS -->
<div class="section">
    <div class="section-title"><span class="icon">📁</span> Projects</div>
    <div class="insight">{proj.get("pareto", "")} · {proj.get("avg_context_switches_per_day", 0)} context switches/day</div>
    <div class="panel">{project_rows}</div>
</div>

<!-- TOOLS -->
<div class="section">
    <div class="section-title"><span class="icon">🔧</span> Tool Usage</div>
    <div class="insight">{tools_data.get("work_style", "")} · {tools_data.get("unique_tools", 0)} unique tools · {tools_data.get("read_write_ratio", "")}</div>
    <div class="panel">{tool_rows}</div>
</div>

<!-- TEMPORAL -->
<div class="section">
    <div class="section-title"><span class="icon">⏰</span> Weekly Pattern</div>
    <div class="panel"><div class="week-chart">{weekly_bars}</div></div>
</div>

<!-- PREDICTIONS -->
<div class="section">
    <div class="section-title"><span class="icon">🔮</span> Predictions</div>
    <div class="panel">
        <div class="label">What you'll work on next</div>
        {pred_rows}
    </div>
</div>

<!-- ═══════════════════════════════════════════════════════════════════ -->
<!-- CHARTS GALLERY — 40 data science visualizations                    -->
<!-- ═══════════════════════════════════════════════════════════════════ -->

<!-- CONTRIBUTION CALENDAR -->
<div class="section">
    <div class="section-title"><span class="icon">📅</span> Activity Calendar</div>
    <div class="panel" style="text-align:center;overflow-x:auto">{chart_calendar}</div>
</div>

<div class="section">
    <div class="section-title"><span class="icon">📈</span> Growth & Activity</div>
    <div class="two-col">
        <div class="panel" style="text-align:center">{chart_cumulative}</div>
        <div class="panel" style="text-align:center">{chart_tokens_day}</div>
    </div>
    <div class="two-col">
        <div class="panel" style="text-align:center">{chart_rolling}</div>
        <div class="panel" style="text-align:center">{chart_daily_scatter}</div>
    </div>
</div>

<div class="section">
    <div class="section-title"><span class="icon">🗓️</span> Time Patterns</div>
    <div class="panel" style="text-align:center">{chart_heatmap}</div>
    <div class="two-col">
        <div class="panel" style="text-align:center">{chart_start_density}</div>
        <div class="panel" style="text-align:center">{chart_productivity_day}</div>
    </div>
    <div class="three-col">
        <div class="panel" style="text-align:center">{chart_radar}</div>
        <div class="panel" style="text-align:center">{chart_monthly}</div>
        <div class="panel" style="text-align:center">{chart_message_donut}</div>
    </div>
</div>

<div class="section">
    <div class="section-title"><span class="icon">🏗️</span> Project Deep Dive</div>
    <div class="panel" style="text-align:center">{chart_timeline}</div>
    <div class="two-col">
        <div class="panel" style="text-align:center">{chart_duration_proj}</div>
        <div class="panel" style="text-align:center">{chart_size_dist}</div>
    </div>
</div>

<div class="section">
    <div class="section-title"><span class="icon">🔧</span> Tooling & Efficiency</div>
    <div class="two-col">
        <div class="panel" style="text-align:center">{chart_mcp}</div>
        <div class="panel" style="text-align:center">{chart_tool_donut}</div>
    </div>
    <div class="two-col">
        <div class="panel" style="text-align:center">{chart_token_eff}</div>
        <div class="panel" style="text-align:center">{chart_subagent}</div>
    </div>
</div>

<div class="section">
    <div class="section-title"><span class="icon">🧘</span> Flow & Sustainability</div>
    <div class="two-col">
        <div class="panel" style="text-align:center">{chart_flow_hour}</div>
        <div class="panel" style="text-align:center">{chart_burnout_trend}</div>
    </div>
    <div class="panel" style="text-align:center">{chart_cooccurrence}</div>
</div>

<!-- NEW GALLERY: 20 more -->
<div class="section">
    <div class="section-title"><span class="icon">💰</span> Cost & Efficiency</div>
    <div class="two-col">
        <div class="panel" style="text-align:center">{chart_cost}</div>
        <div class="panel" style="text-align:center">{chart_cache}</div>
    </div>
    <div class="two-col">
        <div class="panel" style="text-align:center">{chart_response}</div>
        <div class="panel" style="text-align:center">{chart_token_eff}</div>
    </div>
</div>

<div class="section">
    <div class="section-title"><span class="icon">⏱️</span> Session Dynamics</div>
    <div class="two-col">
        <div class="panel" style="text-align:center">{chart_dur_trend}</div>
        <div class="panel" style="text-align:center">{chart_marathon}</div>
    </div>
    <div class="two-col">
        <div class="panel" style="text-align:center">{chart_velocity}</div>
        <div class="panel" style="text-align:center">{chart_intensity}</div>
    </div>
    <div class="two-col">
        <div class="panel" style="text-align:center">{chart_gaps}</div>
        <div class="panel" style="text-align:center">{chart_depth}</div>
    </div>
</div>

<div class="section">
    <div class="section-title"><span class="icon">🌗</span> Behavioral Patterns</div>
    <div class="two-col">
        <div class="panel" style="text-align:center">{chart_nightowl}</div>
        <div class="panel" style="text-align:center">{chart_first_time}</div>
    </div>
    <div class="two-col">
        <div class="panel" style="text-align:center">{chart_weekend}</div>
        <div class="panel" style="text-align:center">{chart_concentration}</div>
    </div>
</div>

<div class="section">
    <div class="section-title"><span class="icon">🧬</span> Project Ecosystem</div>
    <div class="panel" style="text-align:center">{chart_treemap}</div>
    <div class="two-col">
        <div class="panel" style="text-align:center">{chart_diversity}</div>
        <div class="panel" style="text-align:center">{chart_weekly_stack}</div>
    </div>
</div>

<div class="section">
    <div class="section-title"><span class="icon">🛠️</span> Tool & Model Evolution</div>
    <div class="two-col">
        <div class="panel" style="text-align:center">{chart_tool_adopt}</div>
        <div class="panel" style="text-align:center">{chart_branch}</div>
    </div>
    <div class="panel" style="text-align:center;display:flex;justify-content:center">{chart_model_donut}</div>
</div>

<!-- ═══════════════════════════════════════════════════════════════════ -->
<!-- ORGS & REPOS                                                       -->
<!-- ═══════════════════════════════════════════════════════════════════ -->

<div class="section">
    <div class="section-title"><span class="icon">🏢</span> Organizations</div>
    <div class="two-col">
        <div class="panel" style="text-align:center">{chart_org_pie}</div>
        <div class="panel" style="text-align:center">{chart_org_hour}</div>
    </div>
    <div class="panel" style="text-align:center">{chart_org_timeline}</div>
    <div class="panel" style="text-align:center">{chart_org_switches}</div>
</div>

<div class="section">
    <div class="section-title"><span class="icon">📦</span> Repositories</div>
    <div class="panel" style="text-align:center">{chart_repo_board}</div>
    <div class="panel" style="text-align:center">{chart_org_treemap}</div>
    <div class="panel" style="text-align:center">{chart_repo_churn}</div>
</div>

<!-- ═══════════════════════════════════════════════════════════════════ -->
<!-- 10 NEW DIMENSIONS                                                  -->
<!-- ═══════════════════════════════════════════════════════════════════ -->

<div class="section">
    <div class="section-title"><span class="icon">🧠</span> Cognitive Patterns</div>
    <div class="two-col">
        <div class="panel" style="text-align:center">{chart_cognitive}</div>
        <div class="panel" style="text-align:center">{chart_fatigue}</div>
    </div>
    <div class="two-col">
        <div class="panel" style="text-align:center">{chart_creative}</div>
        <div class="panel" style="text-align:center">{chart_prompt_evo}</div>
    </div>
    <div class="two-col">
        <div class="panel" style="text-align:center">{chart_focus}</div>
        <div class="panel" style="text-align:center">{chart_multitask}</div>
    </div>
    <div class="panel" style="text-align:center">{chart_throughput}</div>
</div>

<div class="section">
    <div class="section-title"><span class="icon">🪐</span> Project Gravity & Momentum</div>
    <div class="two-col">
        <div class="panel" style="text-align:center">{chart_gravity}</div>
        <div class="panel" style="text-align:center">{chart_momentum}</div>
    </div>
    <div class="panel" style="text-align:center">{chart_lifecycle}</div>
</div>

<div class="section">
    <div class="section-title"><span class="icon">🔄</span> Work Style Evolution</div>
    <div class="panel" style="text-align:center">{chart_tool_evo}</div>
    <div class="two-col">
        <div class="panel" style="text-align:center">{chart_delegation}</div>
        <div class="panel" style="text-align:center">{chart_recovery}</div>
    </div>
</div>

<div class="footer">Claude Sessions Data Science · {ov["total_sessions"]:,} sessions · {ov["data_size"]} ·
K-Means · DBSCAN · Markov · Sinusoidal Fit · Power Law · Shannon Entropy · Flow Detection · Prowess Assessment · FAANG Level · Bollinger Bands · 72 Visualizations</div>
</div></body></html>'''

    if output_path is None:
        filename = f"claude-sessions-report-{org}.html" if org else "claude-sessions-report.html"
        output_path = str(Path.home() / filename)
    Path(output_path).write_text(html)
    return output_path
