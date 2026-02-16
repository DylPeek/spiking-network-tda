#!/usr/bin/env python3
"""
Date: Feb 16, 2026
Description: Convert AUBC numpy arrays into a CSV time series and per-dimension plots with optional event markers.
Inputs:
  - A TDA results directory containing AUBC/ and a trial metadata.txt file.
Outputs:
  - aubc.csv and per-dimension plot images written under the selected TDA directory.
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
from glob import glob


def parse_args() -> argparse.Namespace:
    """Parse arguments for the AUBC analysis plot script."""
    parser = argparse.ArgumentParser(description="Generate CSV and plots from AUBC numpy files with optional event markers")
    parser.add_argument("--tda_path", type=str, default="./ibl_data/ZM_2240/0/trial_004/tda_w100_s25_k10_n256",
                        help="Path to the TDA results directory (containing AUBC, PH, TE subfolders)")
    parser.add_argument("--metadata_filename", type=str, default="metadata.txt",
                        help="Name of the metadata file in the trial folder")
    parser.add_argument("--analysis_dir", type=str, default="analysis/timeseries_plots",
                        help="Relative directory under tda_path where plots will be saved")
    return parser.parse_args()


def extract_event_times(metadata_path: str) -> tuple[float | None, float | None]:
    """Parse stimon_time and feedback_time from the given metadata text file (in seconds). Returns values in milliseconds."""
    stimon_time, feedback_time = None, None
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            for line in f:
                if "stimon_time" in line:
                    stimon_time = float(line.split(":")[1].strip()) * 1000.0
                elif "feedback_time" in line:
                    feedback_time = float(line.split(":")[1].strip()) * 1000.0
    return stimon_time, feedback_time


def main() -> None:
    args = parse_args()
    tda_path = args.tda_path
    aubc_path = os.path.join(tda_path, "AUBC")
    metadata_path = os.path.join(os.path.dirname(tda_path), args.metadata_filename)

    stimon_ms, feedback_ms = extract_event_times(metadata_path)

    # collect AUBC arrays and window indices
    aubc_files = sorted(glob(os.path.join(aubc_path, "w*_AUBC.npy")))
    if not aubc_files:
        raise RuntimeError(f"No AUBC files found in: {aubc_path}")
    records: list[tuple[int, *float]] = []
    for f in aubc_files:
        match = re.search(r"w(\d+)_AUBC\.npy", os.path.basename(f))
        if not match:
            continue
        time = int(match.group(1))
        values = np.load(f)
        records.append((time, *values))
    records.sort(key=lambda x: x[0])
    df = pd.DataFrame(records)
    df.columns = ["window_time"] + [f"dim_{i-1}" for i in range(1, df.shape[1])]
    # save CSV
    csv_path = os.path.join(tda_path, "aubc.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved AUBC CSV to {csv_path}")
    # create plot directory
    plot_dir = os.path.join(tda_path, args.analysis_dir)
    os.makedirs(plot_dir, exist_ok=True)
    # generate per-dimension plots
    for dim in df.columns[1:]:
        plt.figure(figsize=(8, 4))
        plt.plot(df["window_time"], df[dim], marker='o', linewidth=1)
        plt.xlabel("Window Center Time (ms)")
        plt.ylabel("Betti AUBC")
        plt.title(f"AUBC Time Series — {dim}")
        ymin, ymax = plt.ylim()
        if stimon_ms is not None:
            plt.axvline(x=stimon_ms, color='r', linestyle='--', label='stimon')
            plt.text(stimon_ms - 10.0, ymax * 0.95, 'stimulus on', color='r', rotation=90, va='top', ha='right')
        if feedback_ms is not None:
            plt.axvline(x=feedback_ms, color='g', linestyle='--', label='feedback')
            plt.text(feedback_ms - 10.0, ymax * 0.95, 'feedback', color='g', rotation=90, va='top', ha='right')
        if stimon_ms or feedback_ms:
            plt.legend()
        plt.tight_layout()
        plot_path = os.path.join(plot_dir, f"{dim}.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Plot saved: {plot_path}")
    print("Analysis complete.")


if __name__ == "__main__":
    main()
