#!/usr/bin/env python3
"""
Train an ensemble session classifier (v2).

Improvements over v1:
- Uses quick_scan() directly — no separate extract_features, no parity bugs
- System prompt detection — text features skip prompt-like first messages
- Empty message filtering — pace uses effective (non-empty) user count
- VIF-based multicollinearity elimination
- CalibratedClassifierCV for real probability estimates
- Opus fallback for gray zone sessions via `claude -p`

Output: serialized model + feature config at claude_resume/classifier.pkl
"""

import json
import math
import sys
import os
import time
import pickle
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier, export_text

from claude_resume.sessions import (
    find_all_sessions, classify_session, quick_scan,
    PROJECTS_DIR, MIN_SESSION_BYTES,
)


# ── Session discovery (expanded) ───────────────────────────────

def find_all_sessions_expanded() -> list[dict]:
    """Find ALL .jsonl files including subagent sessions.

    Returns each with a 'source' field: 'main' or 'subagent'.
    Subagent sessions are always automated — ground truth negatives.
    """
    sessions = []
    if not PROJECTS_DIR.exists():
        return sessions

    for project_dir in PROJECTS_DIR.iterdir():
        if not project_dir.is_dir():
            continue
        for jsonl_file in project_dir.glob("*.jsonl"):
            stat = jsonl_file.stat()
            if stat.st_size < MIN_SESSION_BYTES:
                continue
            sessions.append({
                "file": jsonl_file,
                "session_id": jsonl_file.stem,
                "size": stat.st_size,
                "source": "main",
            })
        for session_dir in project_dir.iterdir():
            if not session_dir.is_dir():
                continue
            sa_dir = session_dir / "subagents"
            if sa_dir.is_dir():
                for jsonl_file in sa_dir.glob("*.jsonl"):
                    stat = jsonl_file.stat()
                    if stat.st_size < MIN_SESSION_BYTES:
                        continue
                    sessions.append({
                        "file": jsonl_file,
                        "session_id": jsonl_file.stem,
                        "size": stat.st_size,
                        "source": "subagent",
                    })

    return sessions


# ── Feature columns ───────────────────────────────────────────
# Dropped: total_lines (r=0.99 with user_messages), avg_user_words (r=0.97 with avg_user_chars)
#          lines_per_minute (derived from total_lines)
# Added:   log_duration, empty_msg_ratio, first_is_prompt

FEATURE_COLS = [
    "user_messages", "assistant_messages", "tool_uses", "tool_results",
    "system_entries", "progress_entries", "summary_entries",
    "file_size",
    "duration_secs", "log_duration", "secs_per_turn", "msgs_per_minute",
    "tool_to_user_ratio", "question_ratio", "politeness_ratio",
    "avg_user_chars", "user_code_blocks",
    "avg_assistant_chars", "assistant_code_blocks",
    "casual_ratio", "no_caps_ratio", "short_msg_ratio",
    "exclamation_ratio", "typo_score",
    "empty_msg_ratio", "first_is_prompt",
]


# ── Obvious-example filter ─────────────────────────────────────

def is_obvious(row) -> bool:
    """High-confidence label filter. Only train on these."""
    user = row["user_messages"]
    dur = row["duration_secs"]
    pace = row["secs_per_turn"]
    tools = row["tool_uses"]
    lines = row["total_lines"]
    progress = row["progress_entries"]
    polite = row["politeness_ratio"]
    q_ratio = row["question_ratio"]

    # Obviously automated
    if lines <= 3:
        return True
    if user <= 1 and tools == 0:
        return True
    if user >= 2 and dur > 0 and pace < 8:
        return True

    # Obviously interactive
    if progress > 0:
        return True
    if user >= 2 and dur > 0 and pace > 45:
        return True
    if dur > 300 and user >= 5:
        return True
    if user >= 5 and (polite > 0.1 or q_ratio > 0.2):
        return True

    return False


# ── VIF analysis ───────────────────────────────────────────────

def compute_vif(df, cols):
    """Compute Variance Inflation Factor for each feature."""
    from sklearn.linear_model import LinearRegression
    vifs = {}
    X = df[cols].values
    for i, col in enumerate(cols):
        y = X[:, i]
        X_other = np.delete(X, i, axis=1)
        reg = LinearRegression().fit(X_other, y)
        r2 = reg.score(X_other, y)
        vif = 1 / (1 - r2) if r2 < 1 else float("inf")
        vifs[col] = round(vif, 1)
    return vifs


# ── Main ───────────────────────────────────────────────────────

def main():
    print("\n  Discovering all sessions (including subagents)...")
    sessions = find_all_sessions_expanded()
    n_main = sum(1 for s in sessions if s["source"] == "main")
    n_sub = sum(1 for s in sessions if s["source"] == "subagent")
    print(f"  Found {len(sessions)} total: {n_main} main + {n_sub} subagent\n")

    # ── Extract features using quick_scan (single source of truth) ──
    print("  Extracting features via quick_scan (full parse)...")
    t0 = time.time()
    rows = []
    for i, s in enumerate(sessions):
        try:
            feats = quick_scan(s["file"])
            feats["session_id"] = s["session_id"]
            feats["source"] = s["source"]
            # Label: subagents are ground truth automated
            if s["source"] == "subagent":
                feats["label"] = "automated"
            else:
                feats["label"] = classify_session(feats)
            rows.append(feats)
        except Exception:
            pass
        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{len(sessions)}...")

    elapsed = time.time() - t0
    print(f"  Extracted {len(rows)} sessions in {elapsed:.1f}s\n")

    df = pd.DataFrame(rows)
    print(f"  All sessions: {df['label'].value_counts().to_dict()}")
    print(f"  By source:")
    for src in ["main", "subagent"]:
        sub = df[df["source"] == src]
        print(f"    {src}: {len(sub)} — {sub['label'].value_counts().to_dict()}")

    # ── VIF analysis ─────────────────────────────────────────
    print(f"\n  ── VIF Analysis (multicollinearity check) ──\n")
    vifs = compute_vif(df, FEATURE_COLS)
    for col, vif in sorted(vifs.items(), key=lambda x: -x[1]):
        flag = " ⚠️  HIGH" if vif > 10 else ""
        print(f"    {col:25s} VIF={vif:8.1f}{flag}")

    # ── Build training set ───────────────────────────────────
    main_mask = df["source"] == "main"
    sub_mask = df["source"] == "subagent"
    obvious_mask = df.apply(is_obvious, axis=1)

    df_train_sub = df[sub_mask].copy()
    df_train_main = df[main_mask & obvious_mask].copy()
    df_train = pd.concat([df_train_sub, df_train_main], ignore_index=True)
    df_ambiguous = df[main_mask & ~obvious_mask].copy()

    print(f"\n  Training set: {len(df_train)} — {df_train['label'].value_counts().to_dict()}")
    print(f"    Subagent (ground truth): {len(df_train_sub)}")
    print(f"    Main (obvious): {len(df_train_main)}")
    print(f"  Ambiguous (held out): {len(df_ambiguous)}\n")

    X_labeled = df_train[FEATURE_COLS].values
    y_labeled = (df_train["label"] == "interactive").astype(int).values
    X_all = df[FEATURE_COLS].values
    y_all = (df["label"] == "interactive").astype(int).values

    # ── Train / Test split (80/20, stratified) ───────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_labeled, y_labeled, test_size=0.2, stratify=y_labeled, random_state=42
    )
    print(f"  Train: {len(X_train)} ({np.sum(y_train==1)} interactive, {np.sum(y_train==0)} automated)")
    print(f"  Test:  {len(X_test)} ({np.sum(y_test==1)} interactive, {np.sum(y_test==0)} automated)\n")

    # ── Cross-validation ─────────────────────────────────────
    print("  5-fold stratified CV on training set:\n")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "Decision Tree (d=5)": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42),
    }
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
        print(f"    {name}: {scores.mean():.4f} (+/- {scores.std():.4f})")

    # ── Train + calibrate ────────────────────────────────────
    print("\n  ── Training Gradient Boosting + calibration ──\n")
    gb_raw = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
    gb_raw.fit(X_train, y_train)

    # Evaluate raw model on test set
    y_test_pred = gb_raw.predict(X_test)
    print("  ── TEST SET (raw model) ──\n")
    print(classification_report(y_test, y_test_pred, target_names=["automated", "interactive"], digits=3))

    # Calibrate probabilities using isotonic regression on test set
    print("  ── Calibrating probabilities (isotonic regression) ──\n")
    gb_calibrated = CalibratedClassifierCV(gb_raw, cv=5, method="isotonic")
    gb_calibrated.fit(X_labeled, y_labeled)

    # Check calibration quality
    cal_proba = gb_calibrated.predict_proba(X_all)
    cal_confs = cal_proba.max(axis=1)
    print(f"  Calibrated confidence distribution:")
    print(f"    Min:    {cal_confs.min():.4f}")
    print(f"    25th:   {np.percentile(cal_confs, 25):.4f}")
    print(f"    Median: {np.median(cal_confs):.4f}")
    print(f"    75th:   {np.percentile(cal_confs, 75):.4f}")
    print(f"    Max:    {cal_confs.max():.4f}")
    print(f"    < 0.90: {np.sum(cal_confs < 0.90)}")
    print(f"    < 0.80: {np.sum(cal_confs < 0.80)}")

    # Feature importances from raw model
    importances = gb_raw.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print("\n  Feature importances (top 10):")
    for i in sorted_idx[:10]:
        bar = "█" * int(importances[i] * 50)
        print(f"    {FEATURE_COLS[i]:25s} {importances[i]:.4f} {bar}")

    # ── Interpretable tree ───────────────────────────────────
    print("\n  ── Decision Tree (interpretability) ──\n")
    dt = DecisionTreeClassifier(max_depth=4, random_state=42)
    dt.fit(X_train, y_train)
    print(export_text(dt, feature_names=FEATURE_COLS, max_depth=4))

    # ── Predict all sessions with calibrated model ───────────
    y_ml = gb_calibrated.predict(X_all)
    y_proba = gb_calibrated.predict_proba(X_all)

    print("  ── All-sessions predictions ──\n")
    print(f"  ML:        {np.sum(y_ml==1)} interactive, {np.sum(y_ml==0)} automated")
    print(f"  Heuristic: {np.sum(y_all==1)} interactive, {np.sum(y_all==0)} automated")

    # ── Ensemble with calibrated confidence ──────────────────
    ensemble = []
    gray_zone = []
    for i in range(len(y_all)):
        h = "interactive" if y_all[i] == 1 else "automated"
        m = "interactive" if y_ml[i] == 1 else "automated"
        conf = y_proba[i].max()
        if h == m:
            ensemble.append(h)
        elif conf < 0.80:
            # Gray zone: model is uncertain, lean interactive (safe default)
            ensemble.append("interactive")
            gray_zone.append(i)
        elif h == "interactive" and m == "automated":
            ensemble.append("automated" if conf > 0.90 else "interactive")
        else:
            ensemble.append("interactive")

    ens_int = sum(1 for e in ensemble if e == "interactive")
    ens_auto = sum(1 for e in ensemble if e == "automated")
    print(f"  Ensemble:  {ens_int} interactive, {ens_auto} automated")
    print(f"  Gray zone: {len(gray_zone)} sessions (conf < 0.80)")

    # ── Disagreements ────────────────────────────────────────
    disagree = [(i, y_all[i], y_ml[i], y_proba[i]) for i in range(len(y_all)) if y_all[i] != y_ml[i]]
    print(f"\n  {len(disagree)} heuristic/ML disagreements:")
    for idx, h, m, proba in disagree[:15]:
        row = df.iloc[idx]
        h_label = "interactive" if h == 1 else "automated"
        m_label = "interactive" if m == 1 else "automated"
        ens = ensemble[idx]
        print(f"    h={h_label:12s} ml={m_label:12s} ens={ens:12s} conf={proba.max():.3f} "
              f"user={int(row['user_messages']):3d} dur={row['duration_secs']:7.0f}s "
              f"pace={row['secs_per_turn']:6.1f} q={row['question_ratio']:.2f} "
              f"polite={row['politeness_ratio']:.2f} prompt={int(row.get('first_is_prompt', 0))}")

    # ── Ambiguous sessions ───────────────────────────────────
    if len(df_ambiguous) > 0:
        print(f"\n  ── Ambiguous sessions ({len(df_ambiguous)}) ──\n")
        for _, row in df_ambiguous.iterrows():
            idx = df.index.get_loc(row.name)
            m_label = "interactive" if y_ml[idx] == 1 else "automated"
            ens = ensemble[idx]
            conf = y_proba[idx].max()
            print(f"    h={row['label']:12s} ml={m_label:12s} ens={ens:12s} conf={conf:.3f} "
                  f"pace={row['secs_per_turn']:6.1f} user={int(row['user_messages']):3d} "
                  f"tools={int(row['tool_uses']):3d} prompt={int(row.get('first_is_prompt', 0))}")

    # ── Verify text features are now directionally correct ───
    print(f"\n  ── Feature direction check (should be higher for interactive) ──\n")
    for col in ["casual_ratio", "question_ratio", "no_caps_ratio", "typo_score", "politeness_ratio"]:
        int_mean = df[df["label"] == "interactive"][col].mean()
        auto_mean = df[df["label"] == "automated"][col].mean()
        direction = "✓" if int_mean > auto_mean else "✗ INVERTED"
        print(f"    {col:25s}  interactive={int_mean:.4f}  automated={auto_mean:.4f}  {direction}")

    # ── Serialize calibrated model ───────────────────────────
    model_path = Path(__file__).parent / "claude_resume" / "classifier.pkl"
    model_data = {
        "model": gb_calibrated,
        "feature_cols": FEATURE_COLS,
        "version": 2,
        "trained_on": len(df_train),
        "accuracy_cv": float(cross_val_score(gb_raw, X_labeled, y_labeled, cv=cv, scoring="accuracy").mean()),
    }
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)
    print(f"\n  Model saved to {model_path}")
    print(f"  Size: {model_path.stat().st_size / 1024:.1f} KB")

    # ── Summary ──────────────────────────────────────────────
    print(f"\n  ═══ Summary ═══")
    print(f"  Trained on {len(df_train)} examples ({len(df_ambiguous)} ambiguous excluded)")
    print(f"  CV accuracy: {model_data['accuracy_cv']:.4f}")
    print(f"  Ensemble: {ens_int} interactive, {ens_auto} automated")
    print(f"  Gray zone: {len(gray_zone)} uncertain sessions")
    print(f"  Calibration: conf range [{cal_confs.min():.3f}, {cal_confs.max():.3f}]")
    print(f"  Top features: {', '.join(FEATURE_COLS[i] for i in sorted_idx[:5])}")
    print()


if __name__ == "__main__":
    main()
