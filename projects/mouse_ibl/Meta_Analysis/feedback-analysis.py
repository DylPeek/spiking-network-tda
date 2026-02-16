#!/usr/bin/env python3
"""
Date: Feb 16, 2026
Description: Group trials by feedback outcome (correct/incorrect, optionally no-feedback) and compute windowed AUBC statistics per group.
Inputs:
  - Trial folders containing metadata.txt and AUBC arrays.
  - Command-line arguments selecting feedback grouping rules, window margins, and target Betti dimensions.
Outputs:
  - Per-group CSV tables and summary TXT files written to the specified output directory.
"""

import os
import re
from glob import glob
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
import argparse


def parse_args() -> argparse.Namespace:
    """Parse command‑line options for feedback analysis."""
    parser = argparse.ArgumentParser(description="Analyze AUBC windows grouped by feedback type")
    parser.add_argument("--root_dir", type=str, default="../ibl_data/ZM_2240/0/",
                        help="Directory containing trial_* folders")
    parser.add_argument("--aubc_subpath", type=str, default="tda_w100_s25_k10_n256/AUBC",
                        help="Relative path from each trial directory to the AUBC folder")
    parser.add_argument("--output_dir", type=str, default="./fb_analysis_results",
                        help="Directory to write output files")
    parser.add_argument("--margin_ms", type=int, default=100,
                        help="Milliseconds used for window margins around events")
    parser.add_argument("--target_dims", type=int, nargs="+", default=[0, 1, 2, 3],
                        help="Betti dimensions to include")
    parser.add_argument("--min_times_per_window", type=int, default=1,
                        help="Minimum number of AUBC timepoints per window (strict coverage)")
    parser.add_argument("--strict_window_coverage", action="store_true",
                        help="Enable strict coverage checking (requires min_times_per_window per window)")
    parser.add_argument("--no_strict_window_coverage", action="store_true",
                        help="Disable strict coverage; overrides --strict_window_coverage")
    parser.add_argument("--include_no_feedback", action="store_true",
                        help="Include a 'no_feedback' group in analysis")
    parser.add_argument("--eps", type=float, default=1e-6,
                        help="Tolerance for comparing feedback values to ±1")
    return parser.parse_args()



def parse_metadata(path: str) -> Dict[str, float]:
    data: Dict[str, float] = {}
    if not os.path.exists(path):
        return data
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = re.match(r"(\w+):\s*([-\w\.]+)", line.strip())
            if not m:
                continue
            k, raw = m.groups()
            try:
                data[k] = float(raw) if raw != "nan" else np.nan
            except ValueError:
                pass
    return data

def load_aubc_files(aubc_dir: str) -> Dict[int, np.ndarray]:
    d: Dict[int, np.ndarray] = {}
    for fn in glob(os.path.join(aubc_dir, "w*_AUBC.npy")):
        m = re.search(r"w(\d+)_AUBC\.npy", os.path.basename(fn))
        if not m:
            continue
        t_ms = int(m.group(1))
        d[t_ms] = np.load(fn)
    return d

def collect_window_times(aubc_data: Dict[int, np.ndarray],
                         t_start: int, t_end: int,
                         dims: List[int]) -> List[int]:
    """Return timepoints within [t_start, t_end] whose vectors cover all dims."""
    times = sorted(aubc_data.keys())
    return [t for t in times if (t_start <= t <= t_end) and all(d < len(aubc_data[t]) for d in dims)]

def extract_window_mean(aubc_data: Dict[int, np.ndarray],
                        times: List[int], dim: int) -> float:
    if not times:
        return np.nan
    return float(np.mean([aubc_data[t][dim] for t in times]))

def cohens_d_independent(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x), np.asarray(y)
    mask = ~np.isnan(x) & ~np.isnan(y)
    if np.sum(mask) < 2:
        return np.nan
    x, y = x[mask], y[mask]
    if len(x) < 2 or len(y) < 2:
        return np.nan
    pooled = np.sqrt((np.var(x, ddof=1) + np.var(y, ddof=1)) / 2)
    return (float(np.mean(y)) - float(np.mean(x))) / pooled if pooled != 0 else np.nan

def feedback_group(md: Dict[str, float], *, eps: float, include_no_feedback: bool) -> Optional[str]:
    ft = md.get("feedback_type", np.nan)
    if not np.isnan(ft):
        if np.isclose(ft, 1.0, rtol=0.0, atol=eps):  # correct
            return "correct"
        if np.isclose(ft, -1.0, rtol=0.0, atol=eps):  # incorrect
            return "incorrect"
    return "no_feedback" if include_no_feedback else None



def analyze_group(
    group: str,
    trial_dirs: List[str],
    *,
    aubc_subpath: str,
    margin_ms: int,
    target_dims: List[int],
    min_times_per_window: int,
    strict_window_coverage: bool,
    include_no_feedback: bool,
    output_dir: str,
    eps: float,
    print_skip_counts: bool,
) -> None:
    used_rows: List[Dict[str, float]] = []
    n_scanned = 0
    n_used = 0
    skip_reasons = {
        "no_metadata_or_aubc": 0,
        "no_aubc_files": 0,
        "bad_times": 0,
        "insufficient_max_time": 0,
        "coverage_fail": 0,
    }
    for tdir in trial_dirs:
        n_scanned += 1
        md_path = os.path.join(tdir, "metadata.txt")
        aubc_dir = os.path.join(tdir, aubc_subpath)
        if not (os.path.exists(md_path) and os.path.exists(aubc_dir)):
            skip_reasons["no_metadata_or_aubc"] += 1
            continue
        md = parse_metadata(md_path)
        g = feedback_group(md, eps=eps, include_no_feedback=include_no_feedback)
        if g != group:
            continue
        stimon = md.get("stimon_time")
        feedback = md.get("feedback_time")
        endtime = md.get("intervals_1")
        if stimon is None or feedback is None or endtime is None:
            skip_reasons["bad_times"] += 1
            continue
        aubc_data = load_aubc_files(aubc_dir)
        if not aubc_data:
            skip_reasons["no_aubc_files"] += 1
            continue
        max_time_ms = max(aubc_data.keys())
        stimon_ms = max(0, int(stimon * 1000))
        feedback_ms = int(feedback * 1000)
        endtime_ms = int(endtime * 1000)
        if feedback_ms + margin_ms > max_time_ms:
            skip_reasons["insufficient_max_time"] += 1
            continue
        windows = {
            "baseline_pre": (0, max(0, stimon_ms - margin_ms)),
            "stimon": (stimon_ms, stimon_ms + margin_ms),
            "trial": (stimon_ms + margin_ms, feedback_ms),
            "feedback": (feedback_ms, feedback_ms + margin_ms),
            "baseline_post": (feedback_ms + margin_ms, endtime_ms),
        }
        # strict coverage: ensure each window has enough timepoints with all dims
        win_times: Dict[str, List[int]] = {}
        coverage_ok = True
        for wname, (start, end) in windows.items():
            ts = collect_window_times(aubc_data, start, end, target_dims)
            win_times[wname] = ts
            if strict_window_coverage and len(ts) < min_times_per_window:
                coverage_ok = False
        if not coverage_ok:
            skip_reasons["coverage_fail"] += 1
            continue
        row: Dict[str, float] = {}
        for wname, ts in win_times.items():
            for d in target_dims:
                row[f"{wname}_dim{d}"] = extract_window_mean(aubc_data, ts, d)
        # compute deltas and ratios relative to baseline_pre
        for d in target_dims:
            base = row.get(f"baseline_pre_dim{d}", np.nan)
            for w in ["stimon", "trial", "feedback", "baseline_post"]:
                key = f"{w}_dim{d}"
                val = row.get(key, np.nan)
                if not np.isnan(base) and not np.isnan(val):
                    row[f"{key}_delta"] = val - base
                    row[f"{key}_ratio"] = val / base if base != 0 else np.nan
                else:
                    row[f"{key}_delta"] = np.nan
                    row[f"{key}_ratio"] = np.nan
        row["trial_id"] = os.path.basename(tdir).split("_")[-1]
        row["feedback_type"] = md.get("feedback_type", np.nan)
        row["group"] = group
        used_rows.append(row)
        n_used += 1
    df = pd.DataFrame(used_rows)
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"aubc_window_stats_feedback_{group}.csv")
    df.to_csv(csv_path, index=False)
    print(f"[{group}] saved per‑trial results to {csv_path} (kept: {n_used}, scanned: {n_scanned})")
    if print_skip_counts:
        print(f"[{group}] skipped trials by reason: {skip_reasons}")
    summary: List[str] = []
    summary.append(f"=== FEEDBACK GROUP: {group} ===")
    summary.append(f"Trials kept: {n_used}")
    if print_skip_counts:
        summary.append(f"Skipped (by reason): {skip_reasons}")
    for d in target_dims:
        base_col = f"baseline_pre_dim{d}"
        summary.append(f"\n--- Dimension {d} ---")
        if base_col not in df.columns or df[base_col].dropna().empty:
            summary.append("  No data for this dimension.")
            continue
        baseline = df[base_col].values
        for w in ["stimon", "trial", "feedback", "baseline_post"]:
            key = f"{w}_dim{d}"
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
            for _, r in df.iterrows():
                for widx, ww in enumerate(["baseline_pre", "stimon", "trial", "feedback", "baseline_post"]):
                    col = f"{ww}_dim{d}"
                    val = r.get(col)
                    if val is not None and not np.isnan(val):
                        labels.append(widx)
                        feats.append([val])
            mi = mutual_info_classif(feats, labels, discrete_features=False)[0] if feats else np.nan
            def fmt(x: float) -> str:
                return f"{x:.4f}" if not np.isnan(x) else "NaN"
            summary.append(f"  {w}:")
            summary.append(f"    Mean delta vs baseline: {fmt(mean_delta)}")
            summary.append(f"    Mean ratio vs baseline: {fmt(mean_ratio)}")
            summary.append(f"    Cohen's d: {fmt(cd)}")
            summary.append(f"    Mutual Information: {fmt(mi)}")
    txt_path = os.path.join(output_dir, f"aubc_stats_summary_feedback_{group}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary))
    print(f"[{group}] wrote summary to {txt_path}")



def main() -> None:
    args = parse_args()
    # Determine strict coverage based on flags
    strict_cov = args.strict_window_coverage and not args.no_strict_window_coverage
    os.makedirs(args.output_dir, exist_ok=True)
    trial_dirs = sorted(glob(os.path.join(args.root_dir, "trial_*")))
    counts = {"correct": 0, "incorrect": 0, "no_feedback": 0}
    for tdir in trial_dirs:
        md = parse_metadata(os.path.join(tdir, "metadata.txt"))
        grp = feedback_group(md, eps=args.eps, include_no_feedback=args.include_no_feedback)
        if grp is not None:
            counts[grp] = counts.get(grp, 0) + 1
    print("Trials by feedback (pre-filter):")
    print(f"  correct    (~+1): {counts.get('correct', 0)}")
    print(f"  incorrect  (~-1): {counts.get('incorrect', 0)}")
    if args.include_no_feedback:
        print(f"  no_feedback(else): {counts.get('no_feedback', 0)}")
    for g in ["correct", "incorrect"]:
        analyze_group(
            g,
            trial_dirs,
            aubc_subpath=args.aubc_subpath,
            margin_ms=args.margin_ms,
            target_dims=args.target_dims,
            min_times_per_window=args.min_times_per_window,
            strict_window_coverage=strict_cov,
            include_no_feedback=args.include_no_feedback,
            output_dir=args.output_dir,
            eps=args.eps,
            print_skip_counts=True,
        )
    if args.include_no_feedback and counts.get("no_feedback", 0) > 0:
        analyze_group(
            "no_feedback",
            trial_dirs,
            aubc_subpath=args.aubc_subpath,
            margin_ms=args.margin_ms,
            target_dims=args.target_dims,
            min_times_per_window=args.min_times_per_window,
            strict_window_coverage=strict_cov,
            include_no_feedback=args.include_no_feedback,
            output_dir=args.output_dir,
            eps=args.eps,
            print_skip_counts=True,
        )


if __name__ == "__main__":
    main()
