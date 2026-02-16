#!/usr/bin/env python3
"""
Date: Feb 16, 2026
Description: Extract per-trial spike trains from an IBL session and write boolean spike arrays plus metadata to disk.
Inputs:
  - Command-line arguments selecting subject, session index, cache directory, output directory, and binning frequency.
Outputs:
  - Folder structure under output_root/subject/session_index/trial_XXX/ containing metadata.txt and per-neuron .npy spike arrays.
"""

import argparse
import os
import numpy as np
import pandas as pd
from one.api import ONE


def parse_args() -> argparse.Namespace:
    """Parse command‑line arguments for data extraction."""
    parser = argparse.ArgumentParser(description="Extract IBL trial and spike data into numpy arrays")
    parser.add_argument("--subject", type=str, default="ZM_2240",
                        help="Subject identifier to search for sessions")
    parser.add_argument("--session_index", type=int, default=0,
                        help="Which session index to extract from the search results")
    parser.add_argument("--data_root", type=str, default="./data",
                        help="Folder where IBL downloads cached data")
    parser.add_argument("--output_root", type=str, default="./ibl_data",
                        help="Output folder to write extracted trials")
    parser.add_argument("--target_freq", type=float, default=1000.0,
                        help="Target sampling frequency in Hz (e.g. 1000 for 1 ms bins)")
    parser.add_argument("--probe_collection", type=str, default="alf/probe00/pykilosort",
                        help="Collection name for spike datasets")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # derive target resolution from frequency
    target_resolution = 1.0 / args.target_freq

    # initialise ONE
    ONE.setup(silent=True)
    one = ONE(
        base_url='https://openalyx.internationalbrainlab.org',
        password='',
        cache_dir=args.data_root,
    )

    # search sessions and select by index
    sessions = one.search(subject=args.subject)
    if not sessions:
        raise SystemExit(f"No sessions found for subject {args.subject}")
    try:
        eid = sessions[args.session_index]
    except IndexError:
        raise SystemExit(f"session_index {args.session_index} out of range (found {len(sessions)} sessions)")

    # fetch and load data
    session_info = one.get_details(eid)
    trials = one.load_dataset(eid, '_ibl_trials.table.pqt')
    spike_times = one.load_dataset(eid, dataset='spikes.times.npy', collection=args.probe_collection)
    spike_clusters = one.load_dataset(eid, dataset='spikes.clusters.npy', collection=args.probe_collection)

    # compute native resolution and neuron list
    native_resolution = float(np.round(np.median(np.diff(np.sort(spike_times))), 6))
    unique_neurons = np.unique(spike_clusters)
    num_neurons = len(unique_neurons)

    # build output directory
    subject_dir = os.path.join(args.output_root, args.subject, str(args.session_index))
    os.makedirs(subject_dir, exist_ok=True)

    # write session info text file
    with open(os.path.join(subject_dir, 'session_info.txt'), 'w') as f:
        f.write(f"ALF path: {session_info['local_path']}\n")

    # iterate through trials
    for i, trial in trials.iterrows():
        trial_dir = os.path.join(subject_dir, f"trial_{i:03d}")
        os.makedirs(trial_dir, exist_ok=True)

        # determine trial window and sampling steps
        t0 = trial['intervals_0']
        t1 = trial['intervals_1']
        duration = t1 - t0
        native_steps = int(np.round(duration / native_resolution))
        target_steps = int(np.round(duration / target_resolution))

        def rel_time(x: float) -> str:
            return 'NaN' if pd.isna(x) else f"{x - t0:.4f}"

        # assemble metadata lines
        lines = []
        lines.append("=== Trial Interval ===")
        lines.append(f"intervals_0: {t0:.4f}")
        lines.append(f"intervals_1: {t1:.4f}")
        lines.append("")
        lines.append("=== Relative Timing ===")
        lines.append(f"stimon_time: {rel_time(trial['stimOn_times'])}")
        lines.append(f"goCue_time: {rel_time(trial['goCue_times'])}")
        lines.append(f"response_time: {rel_time(trial['response_times'])}")
        lines.append(f"first_movement_time: {rel_time(trial['firstMovement_times'])}")
        lines.append(f"feedback_time: {rel_time(trial['feedback_times'])}")
        lines.append("")
        lines.append("=== Trial Parameters ===")
        lines.append(f"contrast_left: {trial['contrastLeft']}")
        lines.append(f"contrast_right: {trial['contrastRight']}")
        lines.append(f"choice: {trial['choice']}")
        lines.append(f"feedback_type: {trial['feedbackType']}")
        lines.append(f"reward_volume: {trial['rewardVolume']}")
        lines.append(f"probability_left: {trial['probabilityLeft']}")
        lines.append("")
        lines.append("=== Sampling Info ===")
        lines.append(f"trial_duration: {duration:.4f}")
        lines.append(f"native_resolution: {native_resolution}")
        lines.append(f"native_spike_time_steps: {native_steps}")
        lines.append(f"target_frequency: {args.target_freq} Hz")
        lines.append(f"target_resolution: {target_resolution}")
        lines.append(f"target_spike_time_steps: {target_steps}")
        lines.append("")
        lines.append("=== Neuron Info ===")
        lines.append(f"num_neurons: {num_neurons}")

        with open(os.path.join(trial_dir, 'metadata.txt'), 'w') as f:
            f.write("\n".join(lines))

        # write per-neuron spike arrays
        neuron_dir = os.path.join(trial_dir, 'neurons')
        os.makedirs(neuron_dir, exist_ok=True)

        for idx, neuron_id in enumerate(unique_neurons):
            neuron_spikes = spike_times[spike_clusters == neuron_id]
            mask = (neuron_spikes >= t0) & (neuron_spikes < t1)
            trial_spikes = neuron_spikes[mask] - t0
            spike_array = np.zeros(target_steps, dtype=bool)
            spike_bins = np.floor(trial_spikes / target_resolution).astype(int)
            spike_bins = spike_bins[(spike_bins >= 0) & (spike_bins < target_steps)]
            spike_array[spike_bins] = True
            out_path = os.path.join(neuron_dir, f"neuron_{idx:03d}.npy")
            np.save(out_path, spike_array)


if __name__ == "__main__":
    main()
