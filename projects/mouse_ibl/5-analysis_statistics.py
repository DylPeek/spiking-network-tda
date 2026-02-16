#!/usr/bin/env python3
"""
Date: Feb 16, 2026
Description: Compute windowed statistics over AUBC time series (means, deltas, ratios, effect sizes, and mutual information) for IBL trials.
Inputs:
  - Trial folders containing metadata.txt and AUBC arrays produced by 3-PH_pipeline.py.
  - Command-line arguments selecting the time margin and Betti dimensions to analyze.
Outputs:
  - Per-trial metrics written to CSV and global summary statistics written to a TXT file.
"""

import argparse
import os
import re
import numpy as np
import pandas as pd
from glob import glob
from sklearn.feature_selection import mutual_info_classif


def parse_args() -> argparse.Namespace:
    """Parse command‑line options for the AUBC statistics analyzer."""
    parser = argparse.ArgumentParser(description="Compute windowed AUBC statistics for IBL trials")
    parser.add_argument("--root_dir", type=str, default="./ibl_data/ZM_2240/0/", help="Directory containing trial_* subfolders")
    parser.add_argument("--margin_ms", type=int, default=100, help="Number of milliseconds for pre/post windows around events")
    parser.add_argument("--target_dims", type=int, nargs="+", default=[0, 1, 2, 3], help="Indices of Betti dimensions to analyze")
    parser.add_argument(
        "--aubc_subpath", type=str,
        default="tda_w100_s25_k10_n256/AUBC",
        help="Relative subpath from each trial directory to the folder containing AUBC files"
    )
    parser.add_argument("--output_csv", type=str, default="./analysis_results/aubc_window_stats.csv", help="Path to write per‑trial statistics CSV")
    parser.add_argument("--output_txt", type=str, default="./analysis_results/aubc_stats_summary.txt", help="Path to write summary statistics TXT")
    return parser.parse_args()


def parse_metadata(metadata_path: str) -> dict[str, float]:
    """Parse numeric key:value pairs from a trial's metadata file."""
    data: dict[str, float] = {}
    with open(metadata_path, 'r') as f:
        for line in f:
            m = re.match(r"(\w+):\s+([\d\.\-nan]+)", line.strip())
            if m:
                key, val = m.groups()
                try:
                    data[key] = float(val) if val != 'nan' else np.nan
                except ValueError:
                    pass
    return data


def load_aubc_files(aubc_dir: str) -> dict[int, np.ndarray]:
    """Load all AUBC files in a directory, keyed by integer window center times."""
    aubc_data: dict[int, np.ndarray] = {}
    for file in glob(os.path.join(aubc_dir, "w*_AUBC.npy")):
        m = re.search(r"w(\d+)_AUBC\.npy", file)
        if m:
            time_ms = int(m.group(1))
            aubc_data[time_ms] = np.load(file)
    return aubc_data


def extract_window_stats(aubc_data: dict[int, np.ndarray], t_start: int, t_end: int, dims: list[int]) -> dict[int, float]:
    """Compute mean AUBC values within [t_start, t_end] for specified dimensions."""
    vals = [aubc_data[t] for t in sorted(aubc_data) if t_start <= t <= t_end and all(d < len(aubc_data[t]) for d in dims)]
    if not vals:
        return {d: np.nan for d in dims}
    arr = np.array(vals)
    return {d: float(np.mean(arr[:, d])) for d in dims}


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cohen's d for paired samples x (baseline) and y (other window). Returns NaN if insufficient data."""
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    mask = ~np.isnan(x) & ~np.isnan(y)
    if np.sum(mask) < 2:
        return np.nan
    x_f = x[mask]
    y_f = y[mask]
    pooled_std = np.sqrt((np.var(x_f, ddof=1) + np.var(y_f, ddof=1)) / 2)
    if pooled_std == 0:
        return np.nan
    return float((np.mean(y_f) - np.mean(x_f)) / pooled_std)


def main() -> None:
    args = parse_args()
    root_dir = args.root_dir
    margin_ms = args.margin_ms
    target_dims = args.target_dims
    aubc_subpath = args.aubc_subpath
    results: list[dict[str, float]] = []
    for trial_dir in sorted(glob(os.path.join(root_dir, "trial_*"))):
        trial_id = os.path.basename(trial_dir).split("_")[-1]
        metadata_path = os.path.join(trial_dir, "metadata.txt")
        aubc_dir = os.path.join(trial_dir, aubc_subpath)
        if not (os.path.exists(metadata_path) and os.path.exists(aubc_dir)):
            continue
        metadata = parse_metadata(metadata_path)
        stimon = metadata.get("stimon_time")
        feedback = metadata.get("feedback_time")
        endtime = metadata.get("intervals_1")
        if stimon is None or feedback is None or endtime is None:
            continue
        aubc_data = load_aubc_files(aubc_dir)
        if not aubc_data:
            continue
        max_time_ms = max(aubc_data)
        if feedback * 1000 + margin_ms > max_time_ms:
            continue
        stimon_ms = int(stimon * 1000)
        feedback_ms = int(feedback * 1000)
        endtime_ms = int(endtime * 1000)
        windows = {
            "baseline_pre": (0, stimon_ms - margin_ms),
            "stimon": (stimon_ms, stimon_ms + margin_ms),
            "trial": (stimon_ms + margin_ms, feedback_ms),
            "feedback": (feedback_ms, feedback_ms + margin_ms),
            "baseline_post": (feedback_ms + margin_ms, endtime_ms),
        }
        window_means: dict[str, float] = {}
        for win_name, (start, end) in windows.items():
            stats = extract_window_stats(aubc_data, start, end, target_dims)
            for d, val in stats.items():
                window_means[f"{win_name}_dim{d}"] = val
        # compute deltas and ratios relative to baseline_pre
        for d in target_dims:
            baseline = window_means.get(f"baseline_pre_dim{d}", np.nan)
            for win in ["stimon", "trial", "feedback", "baseline_post"]:
                key = f"{win}_dim{d}"
                val = window_means.get(key, np.nan)
                if not np.isnan(baseline) and not np.isnan(val):
                    window_means[f"{key}_delta"] = val - baseline
                    window_means[f"{key}_ratio"] = val / baseline if baseline != 0 else np.nan
                else:
                    window_means[f"{key}_delta"] = np.nan
                    window_means[f"{key}_ratio"] = np.nan
        window_means["trial_id"] = trial_id
        results.append(window_means)
    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"Saved per‑trial results to {args.output_csv}")
    # compute global statistics
    summary_lines: list[str] = []
    for d in target_dims:
        summary_lines.append(f"\n=== Dimension {d} ===")
        baseline = df[f"baseline_pre_dim{d}"]
        for win in ["stimon", "trial", "feedback", "baseline_post"]:
            key = f"{win}_dim{d}"
            delta_key = f"{key}_delta"
            ratio_key = f"{key}_ratio"
            values = df[key]
            deltas = df[delta_key]
            ratios = df[ratio_key]
            mean_delta = float(np.nanmean(deltas))
            mean_ratio = float(np.nanmean(ratios))
            cd = cohens_d(baseline.values, values.values)
            # mutual information across all windows
            labels: list[int] = []
            features: list[list[float]] = []
            for idx, row in df.iterrows():
                for widx, win_name in enumerate(["baseline_pre", "stimon", "trial", "feedback", "baseline_post"]):
                    col = f"{win_name}_dim{d}"
                    val = row.get(col)
                    if not np.isnan(val):
                        labels.append(widx)
                        features.append([val])
            mi = mutual_info_classif(features, labels, discrete_features=False) if features else [np.nan]
            summary_lines.append(f"  {win}:")
            summary_lines.append(f"    Mean delta vs baseline: {mean_delta:.4f}")
            summary_lines.append(f"    Mean ratio vs baseline: {mean_ratio:.4f}")
            summary_lines.append(f"    Cohen's d: {cd:.4f}")
            summary_lines.append(f"    Mutual Information: {mi[0]:.4f}")
    os.makedirs(os.path.dirname(args.output_txt), exist_ok=True)
    with open(args.output_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    print(f"Saved summary to {args.output_txt}")


if __name__ == "__main__":
    main()
