#!/usr/bin/env python3
"""
Date: Feb 16, 2026
Description: Group trials by stimulus side (left/right) and compute windowed AUBC statistics for each group.
Inputs:
  - Trial folders containing metadata.txt and AUBC arrays.
  - Command-line arguments selecting sides, time margins, and target Betti dimensions.
Outputs:
  - Per-group CSV tables and summary TXT files written to the specified output directory.
"""

import os
import re
import numpy as np
import pandas as pd
from glob import glob
from typing import Dict, Optional, Tuple, List

from sklearn.feature_selection import mutual_info_classif
from scipy.stats import ttest_rel  # retained for future paired tests

import argparse

def parse_args() -> argparse.Namespace:
    """Parse command‑line options for side analysis."""
    parser = argparse.ArgumentParser(description="Group trials by stimulus side and compute AUBC statistics")
    parser.add_argument("--root_dir", type=str, default="../ibl_data/ZM_2240/0/",
                        help="Path to directory containing trial_* folders")
    parser.add_argument("--margin_ms", type=int, default=100, help="Milliseconds for window boundaries around events")
    parser.add_argument("--target_dims", type=int, nargs="+", default=[0, 1, 2, 3],
                        help="Betti dimensions to include in analysis")
    parser.add_argument("--aubc_subpath", type=str, default="tda_w100_s25_k10_n256/AUBC",
                        help="Relative subpath from each trial directory to the AUBC folder")
    parser.add_argument("--output_dir", type=str, default="./lr_analysis_results",
                        help="Directory to write per‑side CSV and TXT files")
    parser.add_argument("--sides", type=str, nargs="+", default=["left", "right"],
                        help="List of sides to analyze (e.g. left right)")
    return parser.parse_args()



def parse_metadata(metadata_path: str) -> Dict[str, float]:
    """Parse key: value pairs as floats; 'nan' becomes np.nan."""
    data = {}
    with open(metadata_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        m = re.match(r"(\w+):\s*([-\w\.]+)", line.strip())
        if m:
            key, raw = m.groups()
            try:
                data[key] = float(raw) if raw != 'nan' else np.nan
            except ValueError:
                # ignore non-numeric values
                pass
    return data

def determine_side(md: Dict[str, float],
                   eps: float = 1e-9) -> Optional[str]:
    """
    Decide which side the stimulus was presented on.
    Rules:
      - If exactly one of {contrast_left, contrast_right} is > 0 → that side
      - If both <= 0 or both NaN → None (no stimulus / skip)
      - If both > 0 and unequal → side with larger contrast
      - If both > 0 and equal (within eps) → None (ambiguous; skip to keep groups clean)
    """
    cl = md.get("contrast_left", np.nan)
    cr = md.get("contrast_right", np.nan)

    def is_pos(x):
        return (not np.isnan(x)) and (x > 0)

    left_pos = is_pos(cl)
    right_pos = is_pos(cr)

    if left_pos and not right_pos:
        return "left"
    if right_pos and not left_pos:
        return "right"
    if not left_pos and not right_pos:
        return None  # no clear stimulus side

    # both positive
    if abs(cl - cr) <= eps:
        return None  # ambiguous, equal contrasts; skip
    return "left" if cl > cr else "right"

def load_aubc_files(aubc_dir: str) -> Dict[int, np.ndarray]:
    """Load all w*_AUBC.npy into a dict keyed by center time (ms)."""
    aubc_data = {}
    for file in glob(os.path.join(aubc_dir, "w*_AUBC.npy")):
        m = re.search(r"w(\d+)_AUBC\.npy", os.path.basename(file))
        if not m:
            continue
        t_ms = int(m.group(1))
        aubc_data[t_ms] = np.load(file)
    return aubc_data

def extract_window_stats(aubc_data: Dict[int, np.ndarray],
                         t_start: int, t_end: int,
                         dims: List[int]) -> Dict[int, float]:
    """Mean AUBC per requested dimension within [t_start, t_end] (inclusive)."""
    times = sorted(aubc_data.keys())
    sel = [aubc_data[t] for t in times
           if (t_start <= t <= t_end) and all(d < len(aubc_data[t]) for d in dims)]
    if not sel:
        return {dim: np.nan for dim in dims}
    arr = np.array(sel)
    return {dim: float(np.mean(arr[:, dim])) for dim in dims}

def cohens_d_independent(x: np.ndarray, y: np.ndarray) -> float:
    """
    Cohen's d using pooled SD of two (independent) groups.
    Mirrors your baseline script's approach (across-trial groups).
    """
    x, y = np.asarray(x), np.asarray(y)
    mask = ~np.isnan(x) & ~np.isnan(y)
    if np.sum(mask) < 2:
        return np.nan
    x, y = x[mask], y[mask]
    if len(x) < 2 or len(y) < 2:
        return np.nan
    pooled_std = np.sqrt((np.var(x, ddof=1) + np.var(y, ddof=1)) / 2)
    return (np.mean(y) - np.mean(x)) / pooled_std if pooled_std != 0 else np.nan



def analyze_group_by_side(
    side: str,
    *,
    root_dir: str,
    margin_ms: int,
    target_dims: List[int],
    aubc_subpath: str,
    output_dir: str,
) -> None:
    """Filter trials by stimulus side and compute per-window AUBC statistics."""
    assert side in {"left", "right"}
    results: List[Dict[str, float]] = []
    n_seen, n_used = 0, 0
    for trial_dir in sorted(glob(os.path.join(root_dir, "trial_*"))):
        n_seen += 1
        trial_id = os.path.basename(trial_dir).split("_")[-1]
        metadata_path = os.path.join(trial_dir, "metadata.txt")
        aubc_dir = os.path.join(trial_dir, aubc_subpath)
        if not (os.path.exists(metadata_path) and os.path.exists(aubc_dir)):
            continue
        md = parse_metadata(metadata_path)
        trial_side = determine_side(md)
        if trial_side != side:
            continue
        stimon = md.get("stimon_time")
        feedback = md.get("feedback_time")
        endtime = md.get("intervals_1")
        if stimon is None or feedback is None or endtime is None:
            continue
        aubc_data = load_aubc_files(aubc_dir)
        if not aubc_data:
            continue
        max_time_ms = max(aubc_data.keys())
        stimon_ms = max(0, int(stimon * 1000))
        feedback_ms = int(feedback * 1000)
        endtime_ms = int(endtime * 1000)
        if feedback_ms + margin_ms > max_time_ms:
            continue
        windows = {
            "baseline_pre": (0, max(0, stimon_ms - margin_ms)),
            "stimon": (stimon_ms, stimon_ms + margin_ms),
            "trial": (stimon_ms + margin_ms, feedback_ms),
            "feedback": (feedback_ms, feedback_ms + margin_ms),
            "baseline_post": (feedback_ms + margin_ms, endtime_ms),
        }
        window_means: Dict[str, float] = {}
        for wname, (start, end) in windows.items():
            stats = extract_window_stats(aubc_data, start, end, target_dims)
            for dim, val in stats.items():
                window_means[f"{wname}_dim{dim}"] = val
        for dim in target_dims:
            baseline = window_means.get(f"baseline_pre_dim{dim}", np.nan)
            for w in ["stimon", "trial", "feedback", "baseline_post"]:
                key = f"{w}_dim{dim}"
                val = window_means.get(key, np.nan)
                if not np.isnan(baseline) and not np.isnan(val):
                    window_means[f"{key}_delta"] = val - baseline
                    window_means[f"{key}_ratio"] = val / baseline if baseline != 0 else np.nan
                else:
                    window_means[f"{key}_delta"] = np.nan
                    window_means[f"{key}_ratio"] = np.nan
        window_means["trial_id"] = trial_id
        window_means["side"] = side
        results.append(window_means)
        n_used += 1
    df = pd.DataFrame(results)
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"aubc_window_stats_side_{side}.csv")
    df.to_csv(csv_path, index=False)
    print(f"[{side}] Saved per-trial results to {csv_path} (trials used: {n_used} / seen: {n_seen})")
    summary_lines: List[str] = [f"=== SIDE: {side} ===", f"Trials used: {n_used} / seen: {n_seen}"]
    for dim in target_dims:
        summary_lines.append(f"\n--- Dimension {dim} ---")
        base_col = f"baseline_pre_dim{dim}"
        if base_col not in df.columns or df[base_col].dropna().empty:
            summary_lines.append("  No data for this dimension.")
            continue
        baseline = df[base_col].values
        for w in ["stimon", "trial", "feedback", "baseline_post"]:
            key = f"{w}_dim{dim}"
            dkey = f"{key}_delta"
            rkey = f"{key}_ratio"
            values = df[key].values if key in df.columns else np.array([])
            deltas = df[dkey].values if dkey in df.columns else np.array([])
            ratios = df[rkey].values if rkey in df.columns else np.array([])
            mean_delta = float(np.nanmean(deltas)) if deltas.size else np.nan
            mean_ratio = float(np.nanmean(ratios)) if ratios.size else np.nan
            cd = cohens_d_independent(baseline, values) if values.size else np.nan
            labels: List[int] = []
            feats: List[List[float]] = []
            for _, row in df.iterrows():
                for widx, ww in enumerate(["baseline_pre", "stimon", "trial", "feedback", "baseline_post"]):
                    col = f"{ww}_dim{dim}"
                    val = row.get(col)
                    if val is not None and not np.isnan(val):
                        labels.append(widx)
                        feats.append([val])
            mi_val = mutual_info_classif(feats, labels, discrete_features=False)[0] if feats else np.nan
            def fmt(x: float) -> str:
                return f"{x:.4f}" if not np.isnan(x) else "NaN"
            summary_lines.append(f"  {w}:")
            summary_lines.append(f"    Mean delta vs baseline: {fmt(mean_delta)}")
            summary_lines.append(f"    Mean ratio vs baseline: {fmt(mean_ratio)}")
            summary_lines.append(f"    Cohen's d: {fmt(cd)}")
            summary_lines.append(f"    Mutual Information: {fmt(mi_val)}")
    txt_path = os.path.join(output_dir, f"aubc_stats_summary_side_{side}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    print(f"[{side}] Saved summary to {txt_path}")

def main() -> None:
    args = parse_args()
    for side in args.sides:
        analyze_group_by_side(
            side,
            root_dir=args.root_dir,
            margin_ms=args.margin_ms,
            target_dims=args.target_dims,
            aubc_subpath=args.aubc_subpath,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
