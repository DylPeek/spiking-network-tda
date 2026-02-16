#!/usr/bin/env python3
"""
Date: Feb 16, 2026
Description: Fetch and preview IBL session datasets via the ONE API and export the trials table to CSV.
Inputs:
  - Command-line arguments selecting subject, session index, cache directory, and dataset identifiers.
Outputs:
  - Trial metadata CSV written to disk and session/dataset information printed to stdout.
"""

import argparse
from one.api import ONE
import pandas as pd
import pyarrow as pa  # required for loading .pqt trial tables
import numpy as np


def parse_args() -> argparse.Namespace:
    """Parse command‑line arguments for data source configuration."""
    parser = argparse.ArgumentParser(description="Fetch and preview IBL dataset metadata")
    parser.add_argument("--subject", type=str, default="ZM_2240",
                        help="Subject identifier (e.g. ZM_2240)")
    parser.add_argument("--session_index", type=int, default=0,
                        help="Index of the session to load from the search results")
    parser.add_argument("--cache_dir", type=str, default="./data/",
                        help="Directory to cache downloaded IBL data")
    parser.add_argument("--save_csv", type=str, default="data.csv",
                        help="Filename to save the trials table CSV")
    parser.add_argument("--probe_collection", type=str, default="alf/probe00/pykilosort",
                        help="Collection name for spike datasets")
    parser.add_argument("--spike_times_dataset", type=str, default="spikes.times.npy",
                        help="Dataset name for spike times")
    parser.add_argument("--spike_clusters_dataset", type=str, default="spikes.clusters.npy",
                        help="Dataset name for spike clusters")
    parser.add_argument("--trials_dataset", type=str, default="_ibl_trials.table.pqt",
                        help="Dataset name for trials table")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # initialise ONE client
    ONE.setup(silent=True)
    one = ONE(
        base_url='https://openalyx.internationalbrainlab.org',
        password='', #TODO: Add the dataset access password.
        cache_dir=args.cache_dir,
    )

    # find sessions for the given subject
    sessions = one.search(subject=args.subject)
    if not sessions:
        raise SystemExit(f"No sessions found for subject {args.subject}")

    # select a session by index
    try:
        eid = sessions[args.session_index]
    except IndexError:
        raise SystemExit(f"session_index {args.session_index} out of range (found {len(sessions)} sessions)")

    # print session list and details
    print("Sessions:", sessions)
    exp_info = one.get_details(eid)
    print("Experiment details:", exp_info)

    # list available datasets in the session
    datasets = one.list_datasets(eid)
    print("Datasets available:", datasets)

    # load spike and trial datasets
    spike_times = one.load_dataset(eid, dataset=args.spike_times_dataset, collection=args.probe_collection)
    spike_clusters = one.load_dataset(eid, dataset=args.spike_clusters_dataset, collection=args.probe_collection)
    trials = one.load_dataset(eid, dataset=args.trials_dataset, collection='alf')

    # export trials table to CSV
    trials.to_csv(args.save_csv, index=False)
    print(f"Saved trials table to {args.save_csv}")

    # summary statistics
    print("Number of trials:", len(trials))
    print("Trial columns:", list(trials.columns))

    # trial timing summary
    start = trials['goCue_times'].min()
    end = trials['goCue_times'].max()
    print(f"Experiment duration (goCue): {end - start:.2f} seconds")

    n_neurons = len(np.unique(spike_clusters))
    print("Total neurons recorded:", n_neurons)

    duration = float(spike_times.max() - spike_times.min())
    print(f"Spike train duration: {duration:.2f} seconds")


if __name__ == "__main__":
    main()