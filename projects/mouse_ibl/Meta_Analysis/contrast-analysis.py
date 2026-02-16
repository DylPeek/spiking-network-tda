#!/usr/bin/env python3
"""
Date: Feb 16, 2026
Description: Group trials by stimulus contrast intensity and compute windowed AUBC statistics per contrast level.
Inputs:
  - Trial folders containing metadata.txt and AUBC arrays.
  - Command-line arguments selecting contrast levels, matching tolerance, time margins, and target Betti dimensions.
Outputs:
  - Per-level CSV tables and summary TXT files written to the specified output directory.
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
    """Parse command‑line options for contrast analysis."""
    parser = argparse.ArgumentParser(description="Group trials by contrast intensity and compute AUBC statistics")
    parser.add_argument("--root_dir", type=str, default="../ibl_data/ZM_2240/0/",
                        help="Directory containing trial_* folders")
    parser.add_argument("--aubc_subpath", type=str, default="tda_w100_s25_k10_n256/AUBC",
                        help="Relative path from each trial directory to the AUBC folder")
    parser.add_argument("--output_dir", type=str, default="./contrast_analysis_results",
                        help="Directory to write output CSV and TXT files")
    parser.add_argument("--margin_ms", type=int, default=100,
                        help="Milliseconds defining window lengths around events")
    parser.add_argument("--target_dims", type=int, nargs="+", default=[0, 1, 2, 3],
                        help="Betti dimensions to analyse")
    parser.add_argument("--levels", type=float, nargs="+", default=[0.0, 0.0625, 0.125, 0.25, 1.0],
                        help="Contrast levels to group trials by")
    parser.add_argument("--eps", type=float, default=1e-9, help="Tolerance for matching intensities to levels")
    return parser.parse_args()

# (Configuration constants removed in favour of command‑line arguments)

# ---------- Helpers ----------

def parse_metadata(path: str) -> Dict[str, float]:
    """Parse numeric 'key: value' pairs from metadata.txt; 'nan'→np.nan."""
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

def trial_contrast_intensity(md: Dict[str, float]) -> Optional[float]:
    """
    Contrast *intensity* ignoring side:
      intensity = max(contrast_left, contrast_right) if any > 0, else None.
    """
    cl = md.get("contrast_left", np.nan)
    cr = md.get("contrast_right", np.nan)
    vals = [v for v in (cl, cr) if (not np.isnan(v)) and (v > -0.1)]
    return float(max(vals)) if vals else None

def load_aubc_files(aubc_dir: str) -> Dict[int, np.ndarray]:
    """Load all w*_AUBC.npy into dict keyed by center time (ms)."""
    d: Dict[int, np.ndarray] = {}
    for fn in glob(os.path.join(aubc_dir, "w*_AUBC.npy")):
        m = re.search(r"w(\d+)_AUBC\.npy", os.path.basename(fn))
        if not m:
            continue
        t_ms = int(m.group(1))
        d[t_ms] = np.load(fn)
    return d

def extract_window_stats(aubc_data: Dict[int, np.ndarray],
                         t_start: int, t_end: int,
                         dims: List[int]) -> Dict[int, float]:
    """Mean AUBC per dim within [t_start, t_end] inclusive."""
    times = sorted(aubc_data.keys())
    sel = [aubc_data[t] for t in times
           if (t_start <= t <= t_end) and all(d < len(aubc_data[t]) for d in dims)]
    if not sel:
        return {d: np.nan for d in dims}
    arr = np.array(sel)
    return {d: float(np.mean(arr[:, d])) for d in dims}

def cohens_d_independent(x: np.ndarray, y: np.ndarray) -> float:
    """Cohen's d with pooled SD (independent groups)."""
    x, y = np.asarray(x), np.asarray(y)
    mask = ~np.isnan(x) & ~np.isnan(y)
    if np.sum(mask) < 2:
        return np.nan
    x, y = x[mask], y[mask]
    if len(x) < 2 or len(y) < 2:
        return np.nan
    pooled = np.sqrt((np.var(x, ddof=1) + np.var(y, ddof=1)) / 2)
    return (float(np.mean(y)) - float(np.mean(x))) / pooled if pooled != 0 else np.nan

# ---------- Discovery ----------

def discover_trials_with_intensity(root_dir: str) -> List[Tuple[str, str, float]]:
    """
    Return list of (trial_dir, trial_id, intensity) for all trials with positive intensity.
    """
    rows: List[Tuple[str, str, float]] = []
    for tdir in sorted(glob(os.path.join(root_dir, "trial_*"))):
        tid = os.path.basename(tdir).split("_")[-1]
        md = parse_metadata(os.path.join(tdir, "metadata.txt"))
        c = trial_contrast_intensity(md)
        if c is not None:
            rows.append((tdir, tid, c))
    return rows

# ---------- Core analysis per level ----------

def analyze_contrast_level(
    level: float,
    rows: List[Tuple[str, str, float]],
    *,
    aubc_subpath: str,
    margin_ms: int,
    target_dims: List[int],
    eps: float,
    output_dir: str,
) -> None:
    """Compute per-window metrics and aggregated stats for a given contrast level."""
    used: List[Dict[str, float]] = []
    n_seen, n_used = 0, 0
    for tdir, tid, intensity in rows:
        n_seen += 1
        if not np.isclose(intensity, level, rtol=0.0, atol=eps):
            continue
        md = parse_metadata(os.path.join(tdir, "metadata.txt"))
        stimon = md.get("stimon_time")
        feedback = md.get("feedback_time")
        endtime = md.get("intervals_1")
        if stimon is None or feedback is None or endtime is None:
            continue
        aubc_dir = os.path.join(tdir, aubc_subpath)
        if not os.path.exists(aubc_dir):
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
        row: Dict[str, float] = {}
        for wname, (start, end) in windows.items():
            stats = extract_window_stats(aubc_data, start, end, target_dims)
            for d, v in stats.items():
                row[f"{wname}_dim{d}"] = v
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
        row["trial_id"] = tid
        row["contrast_intensity"] = float(intensity)
        row["contrast_level"] = float(level)
        used.append(row)
        n_used += 1
    df = pd.DataFrame(used)
    level_tag = f"{level:.6f}"
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"aubc_window_stats_contrast_{level_tag}.csv")
    df.to_csv(csv_path, index=False)
    print(f"[contrast {level_tag}] saved per-trial results to {csv_path} (trials used: {n_used})")
    summary: List[str] = []
    summary.append(f"=== CONTRAST LEVEL: {level_tag} ===")
    summary.append(f"Trials in group: {n_used}")
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
    txt_path = os.path.join(output_dir, f"aubc_stats_summary_contrast_{level_tag}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary))
    print(f"[contrast {level_tag}] wrote summary to {txt_path}")

# ---------- Main ----------

def main() -> None:
    args = parse_args()
    rows = discover_trials_with_intensity(args.root_dir)
    if not rows:
        print("No trials with positive contrast intensity found. Exiting.")
        return
    print("Trial counts per requested contrast level (±eps):")
    for L in args.levels:
        cnt = sum(1 for _, _, c in rows if np.isclose(c, L, rtol=0.0, atol=args.eps))
        print(f"  {L:.6f}: {cnt}")
    for L in args.levels:
        analyze_contrast_level(
            L,
            rows,
            aubc_subpath=args.aubc_subpath,
            margin_ms=args.margin_ms,
            target_dims=args.target_dims,
            eps=args.eps,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
