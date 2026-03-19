"""Real data science on Claude sessions.

Clustering, Markov chains, time series decomposition, anomaly detection,
circadian rhythm modeling, power law analysis, flow state detection,
project co-occurrence networks, burnout indicators, and forecasting.
"""

import math
from collections import Counter, defaultdict
from datetime import datetime, timedelta

import numpy as np
from scipy import stats as sp_stats
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# Session type clustering
# ---------------------------------------------------------------------------

def cluster_sessions(sessions: list[dict], n_clusters: int = 5) -> dict:
    """K-means clustering on session feature vectors.

    Features: duration, message count, tool diversity, token volume,
    hour-of-day, subagent usage ratio, read/write/exec ratio.
    Returns cluster labels, centroids, and human-readable descriptions.
    """
    if len(sessions) < n_clusters * 2:
        return {"error": "Not enough sessions to cluster"}

    features = []
    valid_idx = []
    for i, s in enumerate(sessions):
        dur = s.get("duration_mins", 0)
        msgs = s.get("total_msgs", 0)
        tools = s.get("tool_uses", {})
        n_unique_tools = len(tools) if isinstance(tools, dict) else 0
        tokens = s.get("total_tokens", 0)
        hour = s.get("hour", 12)
        progress = s.get("progress_count", 0)
        total_tool = s.get("tool_use_total", 0)

        # Tool category ratios
        if isinstance(tools, dict) and total_tool > 0:
            read_r = sum(tools.get(t, 0) for t in ["Read", "Glob", "Grep", "LS"]) / total_tool
            write_r = sum(tools.get(t, 0) for t in ["Edit", "Write"]) / total_tool
            exec_r = sum(tools.get(t, 0) for t in ["Bash", "BashOutput"]) / total_tool
        else:
            read_r = write_r = exec_r = 0

        features.append([
            np.log1p(dur),
            np.log1p(msgs),
            n_unique_tools,
            np.log1p(tokens),
            math.sin(2 * math.pi * hour / 24),  # Circular encoding
            math.cos(2 * math.pi * hour / 24),
            progress / max(msgs, 1),  # Delegation ratio
            read_r,
            write_r,
            exec_r,
        ])
        valid_idx.append(i)

    X = np.array(features)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Determine optimal k via silhouette score
    from sklearn.metrics import silhouette_score
    best_k = n_clusters
    best_score = -1
    for k in range(3, min(8, len(X_scaled) // 10)):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X_scaled)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(X_scaled, labels, sample_size=min(5000, len(X_scaled)))
        if score > best_score:
            best_score = score
            best_k = k

    km = KMeans(n_clusters=best_k, n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)

    # PCA for 2D visualization
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_scaled)

    # Describe clusters
    cluster_info = []
    for c in range(best_k):
        mask = labels == c
        cluster_sessions_list = [sessions[valid_idx[i]] for i, m in enumerate(mask) if m]
        n = len(cluster_sessions_list)
        if n == 0:
            continue

        avg_dur = np.mean([s.get("duration_mins", 0) for s in cluster_sessions_list])
        avg_msgs = np.mean([s.get("total_msgs", 0) for s in cluster_sessions_list])
        avg_tokens = np.mean([s.get("total_tokens", 0) for s in cluster_sessions_list])
        avg_tools = np.mean([s.get("tool_use_total", 0) for s in cluster_sessions_list])
        avg_hour = np.mean([s.get("hour", 12) for s in cluster_sessions_list])
        top_projects = Counter(s.get("repo", "?") for s in cluster_sessions_list).most_common(3)

        # Auto-label
        if avg_dur < 5 and avg_msgs < 10:
            label = "Quick Check"
        elif avg_dur > 120 and avg_tools > 50:
            label = "Deep Work Marathon"
        elif avg_tokens > 100000:
            label = "Heavy Computation"
        elif avg_hour >= 22 or avg_hour < 5:
            label = "Late Night Session"
        elif np.mean([s.get("progress_count", 0) for s in cluster_sessions_list]) > 100:
            label = "Multi-Agent Orchestration"
        else:
            label = f"Cluster {c}"

        centroid_2d = X_2d[mask].mean(axis=0).tolist()

        cluster_info.append({
            "id": c,
            "label": label,
            "count": n,
            "pct": round(n / len(sessions) * 100, 1),
            "avg_duration_mins": round(avg_dur, 1),
            "avg_messages": round(avg_msgs, 1),
            "avg_tokens": int(avg_tokens),
            "avg_tool_calls": round(avg_tools, 1),
            "avg_hour": round(avg_hour, 1),
            "top_projects": [{"project": p, "count": c_} for p, c_ in top_projects],
            "centroid_2d": centroid_2d,
        })

    return {
        "n_clusters": best_k,
        "silhouette_score": round(best_score, 3),
        "variance_explained_2d": round(sum(pca.explained_variance_ratio_) * 100, 1),
        "clusters": sorted(cluster_info, key=lambda x: x["count"], reverse=True),
        "points_2d": X_2d.tolist(),
        "labels": labels.tolist(),
    }


# ---------------------------------------------------------------------------
# Project Markov chain
# ---------------------------------------------------------------------------

def project_markov_chain(sessions: list[dict], min_transitions: int = 3) -> dict:
    """Build transition probability matrix between projects.

    Given a sequence of sessions ordered by time, what project do you
    typically work on AFTER project X?
    """
    # Sort by timestamp
    sorted_sessions = sorted(sessions, key=lambda s: s.get("mtime", 0))

    transitions = defaultdict(Counter)
    prev_repo = None
    prev_date = None

    for s in sorted_sessions:
        repo = s.get("repo", "?")
        date = s.get("date", "")

        # Only count transitions within the same day
        if prev_repo and prev_date == date and prev_repo != repo:
            transitions[prev_repo][repo] += 1

        prev_repo = repo
        prev_date = date

    # Build probability matrix for top projects
    project_counts = Counter(s.get("repo", "?") for s in sessions)
    top_projects = [p for p, _ in project_counts.most_common(15)]

    matrix = {}
    for src in top_projects:
        total = sum(transitions[src].values())
        if total < min_transitions:
            continue
        row = {}
        for dst in top_projects:
            count = transitions[src].get(dst, 0)
            if count > 0:
                row[dst] = round(count / total, 3)
        if row:
            matrix[src] = row

    # Find strongest transitions
    strong = []
    for src, dsts in matrix.items():
        for dst, prob in dsts.items():
            if prob >= 0.15:
                strong.append({
                    "from": src,
                    "to": dst,
                    "probability": prob,
                    "description": f"After {src}, {int(prob * 100)}% chance you work on {dst}",
                })
    strong.sort(key=lambda x: x["probability"], reverse=True)

    return {
        "transition_matrix": matrix,
        "strong_transitions": strong[:15],
        "projects_analyzed": len(matrix),
    }


# ---------------------------------------------------------------------------
# Circadian rhythm model
# ---------------------------------------------------------------------------

def circadian_model(sessions: list[dict]) -> dict:
    """Fit a sinusoidal model to hourly activity — your biological clock in data."""
    hour_counts = np.zeros(24)
    for s in sessions:
        hour_counts[s.get("hour", 0)] += 1

    hours = np.arange(24)

    # Fit: A * sin(2pi * (h - phase) / 24) + offset
    def sinusoidal(h, amplitude, phase, offset):
        return amplitude * np.sin(2 * np.pi * (h - phase) / 24) + offset

    try:
        popt, pcov = curve_fit(
            sinusoidal, hours, hour_counts,
            p0=[max(hour_counts) / 2, 14, np.mean(hour_counts)],
            maxfev=10000,
        )
        amplitude, phase, offset = popt
        # R-squared
        residuals = hour_counts - sinusoidal(hours, *popt)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((hour_counts - np.mean(hour_counts)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # Peak and trough
        peak_hour = phase % 24
        trough_hour = (phase + 12) % 24

        # Energy windows
        fitted = sinusoidal(hours, *popt)
        high_energy = [int(h) for h in hours if fitted[int(h)] > offset + amplitude * 0.5]
        low_energy = [int(h) for h in hours if fitted[int(h)] < offset - amplitude * 0.5]

    except (RuntimeError, ValueError):
        return {"error": "Could not fit circadian model", "raw_hourly": hour_counts.tolist()}

    return {
        "peak_hour": round(peak_hour, 1),
        "trough_hour": round(trough_hour, 1),
        "amplitude": round(amplitude, 1),
        "r_squared": round(r_squared, 3),
        "model_fit": "strong" if r_squared > 0.7 else "moderate" if r_squared > 0.4 else "weak",
        "high_energy_hours": high_energy,
        "low_energy_hours": low_energy,
        "actual_hourly": hour_counts.tolist(),
        "fitted_hourly": sinusoidal(hours, *popt).tolist(),
        "insight": (
            f"Your biological clock peaks at {int(peak_hour)}:00 and dips at {int(trough_hour)}:00. "
            f"Schedule hard problems for {int(peak_hour - 1)}:00-{int(peak_hour + 2) % 24}:00."
        ),
    }


# ---------------------------------------------------------------------------
# Power law analysis
# ---------------------------------------------------------------------------

def power_law_analysis(sessions: list[dict]) -> dict:
    """Test if session durations follow a power law (Pareto principle in action).

    Heavy-tailed distributions mean a few sessions dominate your output.
    """
    durations = [s.get("duration_mins", 0) for s in sessions if s.get("duration_mins", 0) > 1]
    if len(durations) < 50:
        return {"error": "Not enough sessions with duration data"}

    durations = np.array(sorted(durations, reverse=True))
    n = len(durations)

    # Fit power law: P(X > x) ~ x^(-alpha)
    # MLE for Pareto exponent
    x_min = np.percentile(durations, 10)  # Threshold
    tail = durations[durations >= x_min]
    if len(tail) < 10:
        return {"error": "Not enough tail data"}

    alpha = 1 + len(tail) / np.sum(np.log(tail / x_min))

    # KS test against exponential (is it truly heavy-tailed?)
    ks_stat, ks_p = sp_stats.kstest(tail, 'expon', args=(x_min, np.mean(tail) - x_min))

    # Concentration metrics
    top_1pct = durations[:max(1, n // 100)]
    top_10pct = durations[:max(1, n // 10)]
    total = durations.sum()

    return {
        "alpha": round(alpha, 2),
        "is_heavy_tailed": alpha < 3,
        "x_min": round(x_min, 1),
        "ks_vs_exponential": {"statistic": round(ks_stat, 3), "p_value": round(ks_p, 4)},
        "concentration": {
            "top_1pct_share": round(top_1pct.sum() / total * 100, 1),
            "top_10pct_share": round(top_10pct.sum() / total * 100, 1),
        },
        "insight": (
            f"Alpha={alpha:.2f}. "
            f"Top 1% of sessions account for {top_1pct.sum() / total * 100:.0f}% of total time. "
            f"Top 10% account for {top_10pct.sum() / total * 100:.0f}%. "
            f"{'Your work follows a power law — a few marathon sessions drive most output.' if alpha < 3 else 'Your sessions are more evenly distributed than a power law.'}"
        ),
    }


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

def detect_anomalies(sessions: list[dict], contamination: float = 0.05) -> dict:
    """DBSCAN anomaly detection on session feature space.

    Finds sessions that are statistical outliers — unusually long,
    token-heavy, late-night, or otherwise abnormal.
    """
    features = []
    valid_sessions = []
    for s in sessions:
        dur = s.get("duration_mins", 0)
        msgs = s.get("total_msgs", 0)
        tokens = s.get("total_tokens", 0)
        tools = s.get("tool_use_total", 0)
        hour = s.get("hour", 12)

        features.append([
            np.log1p(dur),
            np.log1p(msgs),
            np.log1p(tokens),
            np.log1p(tools),
            math.sin(2 * math.pi * hour / 24),
            math.cos(2 * math.pi * hour / 24),
        ])
        valid_sessions.append(s)

    if len(features) < 50:
        return {"error": "Not enough sessions"}

    X = np.array(features)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    db = DBSCAN(eps=1.5, min_samples=10)
    labels = db.fit_predict(X_scaled)

    anomaly_indices = [i for i, l in enumerate(labels) if l == -1]
    anomalies = []
    for i in anomaly_indices:
        s = valid_sessions[i]
        # Compute z-scores for interpretability
        z_scores = X_scaled[i]
        reasons = []
        feat_names = ["duration", "messages", "tokens", "tool_calls", "hour_sin", "hour_cos"]
        for j, (name, z) in enumerate(zip(feat_names, z_scores)):
            if abs(z) > 2:
                direction = "extremely high" if z > 0 else "extremely low"
                if name not in ("hour_sin", "hour_cos"):
                    reasons.append(f"{name} {direction} (z={z:.1f})")

        anomalies.append({
            "session_id": s.get("session_id", "?"),
            "project": s.get("repo", "?"),
            "date": s.get("date", "?"),
            "duration_mins": round(s.get("duration_mins", 0), 1),
            "messages": s.get("total_msgs", 0),
            "tokens": s.get("total_tokens", 0),
            "reasons": reasons,
        })

    # Sort by most anomalous (most extreme z-scores)
    anomalies.sort(key=lambda a: len(a["reasons"]), reverse=True)

    return {
        "total_anomalies": len(anomalies),
        "anomaly_rate": round(len(anomalies) / len(sessions) * 100, 1),
        "anomalies": anomalies[:20],
    }


# ---------------------------------------------------------------------------
# Flow state detection
# ---------------------------------------------------------------------------

def detect_flow_states(sessions: list[dict]) -> dict:
    """Identify sessions where you were likely in a flow state.

    Flow indicators: sustained duration (>45min), high tool throughput,
    low message-to-tool ratio (more doing, less asking), consistent pacing.
    """
    flow_sessions = []
    non_flow = []

    for s in sessions:
        dur = s.get("duration_mins", 0)
        msgs = s.get("total_msgs", 0)
        user_msgs = s.get("user_msgs", 0)
        tools = s.get("tool_use_total", 0)
        tokens = s.get("total_tokens", 0)

        if dur < 30 or msgs < 5:
            continue

        # Flow score: 0-100
        score = 0

        # Duration: 45-180 min is optimal flow zone
        if 45 <= dur <= 180:
            score += 25
        elif dur > 180:
            score += 15  # Marathon, but possibly fatigued

        # Tool throughput: tools per minute
        throughput = tools / max(dur, 1)
        if throughput > 0.5:
            score += 25
        elif throughput > 0.2:
            score += 15

        # Low ask ratio: (user_msgs / total_msgs) — lower = more doing
        ask_ratio = user_msgs / max(msgs, 1)
        if ask_ratio < 0.3:
            score += 25
        elif ask_ratio < 0.5:
            score += 15

        # Token density: output per minute
        token_rate = tokens / max(dur, 1)
        if token_rate > 100:
            score += 25
        elif token_rate > 50:
            score += 15

        entry = {
            "session_id": s.get("session_id", "?"),
            "project": s.get("repo", "?"),
            "date": s.get("date", "?"),
            "hour": s.get("hour", 0),
            "duration_mins": round(dur, 1),
            "flow_score": score,
            "tool_throughput": round(throughput, 2),
            "ask_ratio": round(ask_ratio, 2),
        }

        if score >= 60:
            flow_sessions.append(entry)
        else:
            non_flow.append(entry)

    flow_sessions.sort(key=lambda x: x["flow_score"], reverse=True)

    # Flow by hour
    flow_hours = Counter(s["hour"] for s in flow_sessions)
    total_by_hour = Counter(s.get("hour", 0) for s in sessions)
    flow_rate_by_hour = {}
    for h in range(24):
        total = total_by_hour.get(h, 0)
        if total >= 5:
            flow_rate_by_hour[f"{h:02d}:00"] = round(flow_hours.get(h, 0) / total * 100, 1)

    # Best flow hour
    best_flow_hour = max(flow_rate_by_hour, key=flow_rate_by_hour.get) if flow_rate_by_hour else "?"

    # Flow by project
    flow_projects = Counter(s["project"] for s in flow_sessions)
    total_projects = Counter(s.get("repo", "?") for s in sessions)
    flow_rate_by_project = {}
    for proj, count in flow_projects.most_common(10):
        total = total_projects.get(proj, 0)
        if total >= 3:
            flow_rate_by_project[proj] = {
                "flow_sessions": count,
                "total_sessions": total,
                "flow_rate": round(count / total * 100, 1),
            }

    return {
        "total_flow_sessions": len(flow_sessions),
        "flow_rate": round(len(flow_sessions) / max(len(sessions), 1) * 100, 1),
        "best_flow_hour": best_flow_hour,
        "flow_rate_by_hour": flow_rate_by_hour,
        "flow_rate_by_project": flow_rate_by_project,
        "top_flow_sessions": flow_sessions[:10],
        "insight": (
            f"{len(flow_sessions)} flow sessions ({len(flow_sessions) / max(len(sessions), 1) * 100:.0f}% of total). "
            f"Your best flow hour is {best_flow_hour}. "
            f"Protect this time."
        ),
    }


# ---------------------------------------------------------------------------
# Burnout risk indicators
# ---------------------------------------------------------------------------

def burnout_indicators(sessions: list[dict]) -> dict:
    """Track burnout risk signals over time.

    Signals: increasing late-night frequency, session duration trends,
    shrinking time between sessions, weekend encroachment.
    """
    # Sort by date
    by_date = defaultdict(list)
    for s in sessions:
        by_date[s.get("date", "")].append(s)

    dates = sorted(by_date.keys())
    if len(dates) < 7:
        return {"error": "Need at least 7 days of data"}

    # Compute weekly metrics
    weekly = defaultdict(lambda: {
        "sessions": 0, "late_night": 0, "total_hours": 0,
        "weekend_sessions": 0, "dates": set(),
    })

    for date_str, day_sessions in by_date.items():
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            continue
        week = dt.strftime("%Y-W%W")
        w = weekly[week]
        w["sessions"] += len(day_sessions)
        w["late_night"] += sum(1 for s in day_sessions if s.get("hour", 12) in [0, 1, 2, 3, 4, 23])
        w["total_hours"] += sum(s.get("duration_mins", 0) for s in day_sessions) / 60
        w["weekend_sessions"] += sum(1 for s in day_sessions if s.get("weekday_num", 0) >= 5)
        w["dates"].add(date_str)

    weeks = sorted(weekly.keys())
    if len(weeks) < 2:
        return {"error": "Need at least 2 weeks of data"}

    # Trends
    session_counts = [weekly[w]["sessions"] for w in weeks]
    late_night_counts = [weekly[w]["late_night"] for w in weeks]
    hours_per_week = [weekly[w]["total_hours"] for w in weeks]
    weekend_counts = [weekly[w]["weekend_sessions"] for w in weeks]

    def trend_slope(values):
        if len(values) < 2:
            return 0
        x = np.arange(len(values))
        slope, _, r, p, _ = sp_stats.linregress(x, values)
        return {"slope": round(slope, 2), "r": round(r, 3), "p": round(p, 4)}

    session_trend = trend_slope(session_counts)
    late_night_trend = trend_slope(late_night_counts)
    hours_trend = trend_slope(hours_per_week)
    weekend_trend = trend_slope(weekend_counts)

    # Burnout score: 0-100
    score = 0
    signals = []

    # Rising late nights
    if isinstance(late_night_trend, dict) and late_night_trend["slope"] > 1 and late_night_trend["p"] < 0.1:
        score += 25
        signals.append(f"Late-night sessions increasing ({late_night_trend['slope']:+.1f}/week)")

    # Rising hours
    if isinstance(hours_trend, dict) and hours_trend["slope"] > 5:
        score += 20
        signals.append(f"Weekly hours increasing ({hours_trend['slope']:+.1f}h/week)")

    # Weekend encroachment
    if isinstance(weekend_trend, dict) and weekend_trend["slope"] > 0.5 and weekend_trend["p"] < 0.1:
        score += 20
        signals.append(f"Weekend work increasing ({weekend_trend['slope']:+.1f} sessions/week)")

    # Recent intensity: last week vs average
    if len(weeks) >= 3:
        recent = hours_per_week[-1]
        avg_prior = np.mean(hours_per_week[:-1])
        if recent > avg_prior * 1.5:
            score += 15
            signals.append(f"Last week was {recent / max(avg_prior, 1):.1f}x your average intensity")

    # No rest days
    recent_dates = set()
    for w in weeks[-2:]:
        recent_dates.update(weekly[w]["dates"])
    consecutive = 0
    check = datetime.now()
    for _ in range(14):
        if check.strftime("%Y-%m-%d") in recent_dates:
            consecutive += 1
        else:
            break
        check -= timedelta(days=1)
    if consecutive >= 10:
        score += 20
        signals.append(f"{consecutive} consecutive days without a break")

    if score < 20:
        risk_level = "Low"
    elif score < 50:
        risk_level = "Moderate"
    elif score < 75:
        risk_level = "Elevated"
    else:
        risk_level = "High"

    return {
        "burnout_score": score,
        "risk_level": risk_level,
        "signals": signals,
        "trends": {
            "sessions_per_week": session_trend,
            "late_nights_per_week": late_night_trend,
            "hours_per_week": hours_trend,
            "weekend_sessions_per_week": weekend_trend,
        },
        "weekly_data": {
            "weeks": weeks,
            "sessions": session_counts,
            "late_nights": late_night_counts,
            "hours": [round(h, 1) for h in hours_per_week],
            "weekend": weekend_counts,
        },
    }


# ---------------------------------------------------------------------------
# Project co-occurrence network
# ---------------------------------------------------------------------------

def project_cooccurrence(sessions: list[dict], min_cooccurrence: int = 3) -> dict:
    """Which projects do you tend to work on together in the same day?

    Builds a co-occurrence graph — edges weighted by frequency.
    """
    day_projects = defaultdict(set)
    for s in sessions:
        day_projects[s.get("date", "")].add(s.get("repo", "?"))

    edges = Counter()
    for date, projects in day_projects.items():
        projects = list(projects)
        for i in range(len(projects)):
            for j in range(i + 1, len(projects)):
                edge = tuple(sorted([projects[i], projects[j]]))
                edges[edge] += 1

    # Filter weak edges
    strong_edges = [(a, b, c) for (a, b), c in edges.items() if c >= min_cooccurrence]
    strong_edges.sort(key=lambda x: x[2], reverse=True)

    # Node degrees (centrality)
    degree = Counter()
    for a, b, c in strong_edges:
        degree[a] += c
        degree[b] += c

    return {
        "edges": [{"from": a, "to": b, "weight": c} for a, b, c in strong_edges[:30]],
        "central_projects": [{"project": p, "degree": d} for p, d in degree.most_common(10)],
        "insight": f"{len(strong_edges)} project pairs frequently co-occur. "
                   f"Most central: {degree.most_common(1)[0][0] if degree else '?'}",
    }


# ---------------------------------------------------------------------------
# Session duration distribution analysis
# ---------------------------------------------------------------------------

def duration_distribution(sessions: list[dict]) -> dict:
    """Statistical analysis of session duration distribution.

    Tests for bimodality, fits distributions, identifies natural breakpoints.
    """
    durations = [s.get("duration_mins", 0) for s in sessions if s.get("duration_mins", 0) > 0.5]
    if len(durations) < 30:
        return {"error": "Not enough sessions with duration data"}

    d = np.array(durations)

    # Basic stats
    basic = {
        "mean": round(np.mean(d), 1),
        "median": round(np.median(d), 1),
        "std": round(np.std(d), 1),
        "skewness": round(float(sp_stats.skew(d)), 2),
        "kurtosis": round(float(sp_stats.kurtosis(d)), 2),
        "p10": round(np.percentile(d, 10), 1),
        "p25": round(np.percentile(d, 25), 1),
        "p75": round(np.percentile(d, 75), 1),
        "p90": round(np.percentile(d, 90), 1),
        "p99": round(np.percentile(d, 99), 1),
    }

    # Bimodality test: Hartigan's dip test approximation
    # Use histogram peaks as proxy
    log_d = np.log1p(d)
    hist, bin_edges = np.histogram(log_d, bins=30)
    peaks, properties = find_peaks(hist, height=len(d) * 0.02, distance=3)
    is_bimodal = len(peaks) >= 2

    # If bimodal, find the natural breakpoint
    breakpoint = None
    if is_bimodal and len(peaks) >= 2:
        # Valley between two highest peaks
        sorted_peaks = sorted(peaks, key=lambda p: hist[p], reverse=True)[:2]
        valley_start = min(sorted_peaks)
        valley_end = max(sorted_peaks)
        valley_idx = valley_start + np.argmin(hist[valley_start:valley_end + 1])
        breakpoint = round(np.expm1(bin_edges[valley_idx]), 1)

    # Bin into categories
    categories = {
        "micro (<5min)": len(d[d < 5]),
        "short (5-30min)": len(d[(d >= 5) & (d < 30)]),
        "medium (30-90min)": len(d[(d >= 30) & (d < 90)]),
        "long (90-240min)": len(d[(d >= 90) & (d < 240)]),
        "marathon (>240min)": len(d[d >= 240]),
    }

    return {
        "stats": basic,
        "is_bimodal": is_bimodal,
        "n_modes": len(peaks),
        "breakpoint_mins": breakpoint,
        "categories": categories,
        "histogram": {
            "bins": [round(np.expm1(b), 1) for b in bin_edges.tolist()],
            "counts": hist.tolist(),
        },
        "insight": (
            f"{'Bimodal distribution detected' if is_bimodal else 'Unimodal distribution'}. "
            f"Median session: {basic['median']}min. "
            f"{'You have two distinct modes: quick checks and deep dives, split around ' + str(breakpoint) + ' minutes.' if breakpoint else ''} "
            f"Skewness: {basic['skewness']} ({'right-skewed — a few long sessions pull the average up' if basic['skewness'] > 1 else 'relatively symmetric'})."
        ),
    }


# ---------------------------------------------------------------------------
# Entropy / predictability
# ---------------------------------------------------------------------------

def work_entropy(sessions: list[dict]) -> dict:
    """Shannon entropy of your work patterns — how predictable are you?

    High entropy = chaotic/unpredictable. Low entropy = routine.
    Computed over: project choice, hour, day-of-week.
    """
    def entropy(counts):
        total = sum(counts.values())
        if total == 0:
            return 0
        probs = [c / total for c in counts.values() if c > 0]
        return -sum(p * math.log2(p) for p in probs)

    def max_entropy(n):
        return math.log2(n) if n > 0 else 0

    project_counts = Counter(s.get("repo", "?") for s in sessions)
    hour_counts = Counter(s.get("hour", 0) for s in sessions)
    day_counts = Counter(s.get("weekday", "?") for s in sessions)

    project_h = entropy(project_counts)
    hour_h = entropy(hour_counts)
    day_h = entropy(day_counts)

    project_max = max_entropy(len(project_counts))
    hour_max = max_entropy(24)
    day_max = max_entropy(7)

    return {
        "project_entropy": {
            "bits": round(project_h, 2),
            "max_bits": round(project_max, 2),
            "normalized": round(project_h / max(project_max, 1), 3),
            "interpretation": "chaotic" if project_h / max(project_max, 1) > 0.8 else
                            "varied" if project_h / max(project_max, 1) > 0.5 else "focused",
        },
        "hour_entropy": {
            "bits": round(hour_h, 2),
            "max_bits": round(hour_max, 2),
            "normalized": round(hour_h / max(hour_max, 1), 3),
            "interpretation": "works all hours" if hour_h / max(hour_max, 1) > 0.85 else
                            "some routine" if hour_h / max(hour_max, 1) > 0.7 else "clockwork",
        },
        "day_entropy": {
            "bits": round(day_h, 2),
            "max_bits": round(day_max, 2),
            "normalized": round(day_h / max(day_max, 1), 3),
            "interpretation": "every day is work day" if day_h / max(day_max, 1) > 0.9 else
                            "some weekly rhythm" if day_h / max(day_max, 1) > 0.7 else "strong weekly routine",
        },
        "overall_predictability": round(
            (1 - (project_h / max(project_max, 1) + hour_h / max(hour_max, 1) + day_h / max(day_max, 1)) / 3) * 100, 1
        ),
        "insight": (
            f"Predictability score: {round((1 - (project_h / max(project_max, 1) + hour_h / max(hour_max, 1) + day_h / max(day_max, 1)) / 3) * 100)}%. "
            f"Project choice: {round(project_h / max(project_max, 1) * 100)}% entropy "
            f"({'highly unpredictable' if project_h / max(project_max, 1) > 0.8 else 'somewhat predictable'}). "
            f"Schedule: {round(hour_h / max(hour_max, 1) * 100)}% entropy."
        ),
    }


# ---------------------------------------------------------------------------
# Master analysis
# ---------------------------------------------------------------------------

def full_analysis(sessions: list[dict]) -> dict:
    """Run all data science models. Returns structured results."""
    results = {}

    results["clustering"] = cluster_sessions(sessions)
    results["markov_chain"] = project_markov_chain(sessions)
    results["circadian"] = circadian_model(sessions)
    results["power_law"] = power_law_analysis(sessions)
    results["anomalies"] = detect_anomalies(sessions)
    results["flow_states"] = detect_flow_states(sessions)
    results["burnout"] = burnout_indicators(sessions)
    results["cooccurrence"] = project_cooccurrence(sessions)
    results["duration_dist"] = duration_distribution(sessions)
    results["entropy"] = work_entropy(sessions)

    return results
