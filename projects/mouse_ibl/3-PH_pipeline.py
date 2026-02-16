#!/usr/bin/env python3
"""
Date: Feb 16, 2026
Description: Run the TE+PH pipeline on extracted IBL trials using sliding windows, producing TE matrices, persistence diagrams, and AUBC time series.
Inputs:
  - Extracted trial folders containing neuron spike arrays (from 1-data_extractor.py).
  - Command-line arguments controlling window size, step size, TE history length, homology dimensions, and neuron count.
Outputs:
  - For each trial, a tda_w<win>_s<step>_k<k>_n<num>/ folder containing TE/, PH/, and AUBC/ artifacts.
"""

import os
import numpy as np
import sys
import argparse

# locate the TDA helper modules relative to this script
this_dir = os.path.dirname(os.path.abspath(__file__))
tda_path = os.path.abspath(os.path.join(this_dir, "..", "..", "common", "time-series-tda"))
sys.path.append(tda_path)

from transfer_entropy import compute_transfer_entropy_matrix
from adjacency import preprocess_adjacency_matrix
from persistence import compute_persistence_diagram
from io_utils import save_persistence_diagram  # optional if you want .txt

def parse_args() -> argparse.Namespace:
    """Parse command‑line arguments controlling the TDA pipeline."""
    parser = argparse.ArgumentParser(
        description="Compute transfer entropy, persistent homology and Betti AUBC in sliding windows for IBL trial spike trains"
    )
    parser.add_argument(
        "--trial_template", type=str,
        default="./ibl_data/ZM_2240/0/trial_{:03d}",
        help="Python format string path pointing to trial folders; must include one integer placeholder"
    )
    parser.add_argument(
        "--tests", type=int, default=999,
        help="Number of consecutive trial indices to process starting from 0"
    )
    parser.add_argument(
        "--window_size", type=int, default=100,
        help="Size of the sliding window in timesteps"
    )
    parser.add_argument(
        "--step_size", type=int, default=25,
        help="Step size between successive windows"
    )
    parser.add_argument(
        "--k_history", type=int, default=10,
        help="History length parameter k for transfer entropy"
    )
    parser.add_argument(
        "--ph_dims", type=int, nargs="+", default=[0, 1, 2, 3, 4],
        help="Persistent homology dimensions to compute"
    )
    parser.add_argument(
        "--num_neurons", type=int, default=256,
        help="Number of neurons to include in the analysis (-1 for all available)"
    )
    return parser.parse_args()


def betti_auc(dgm: np.ndarray, dims: list[int]) -> np.ndarray:
    """Compute AUBC (sum of bar lengths) for each persistent homology dimension."""
    accum = {d: 0.0 for d in dims}
    for b, d, dim in dgm:
        dim_int = int(dim)
        if dim_int in dims and np.isfinite(b) and np.isfinite(d):
            accum[dim_int] += d - b
    return np.array([accum[d] for d in dims], dtype=float)


def main() -> None:
    args = parse_args()
    # iterate over trial indices
    for test in range(args.tests):
        trial_path = args.trial_template.format(test)
        window_size = args.window_size
        step_size = args.step_size
        k_history = args.k_history
        ph_dims = args.ph_dims
        num_neurons = args.num_neurons

        neurons_path = os.path.join(trial_path, 'neurons')
        neuron_files = sorted(f for f in os.listdir(neurons_path) if f.endswith('.npy'))
        if num_neurons > 0:
            neuron_files = neuron_files[:num_neurons]
        spikes = np.stack([np.load(os.path.join(neurons_path, f)).astype(float) for f in neuron_files], axis=0)

        n_neurons, T = spikes.shape
        half_w = window_size // 2
        centers = range(half_w, T - half_w, step_size)

        tda_root = os.path.join(trial_path, f'tda_w{window_size}_s{step_size}_k{k_history}_n{num_neurons}')
        te_dir = os.path.join(tda_root, 'TE')
        ph_dir = os.path.join(tda_root, 'PH')
        auc_dir = os.path.join(tda_root, 'AUBC')
        os.makedirs(te_dir, exist_ok=True)
        os.makedirs(ph_dir, exist_ok=True)
        os.makedirs(auc_dir, exist_ok=True)

        print(f"Starting TDA on {trial_path}")
        print(f"Neurons: {n_neurons}, Time Steps: {T}")
        print(f"Window: {window_size}, Step: {step_size}, History: {k_history}")
        print(f"Output: {tda_root}")
        print(f"Total Windows: {len(list(centers))}")

        # sliding window computation
        for t in centers:
            tag = f"w{t:05d}"
            te_path = os.path.join(te_dir, f"{tag}_te.npy")
            pd_path = os.path.join(ph_dir, f"{tag}_pd.npy")
            auc_path = os.path.join(auc_dir, f"{tag}_AUBC.npy")
            # skip if outputs exist
            if all(os.path.exists(p) for p in (te_path, pd_path, auc_path)):
                continue
            X_win = spikes[:, t - half_w: t + half_w]
            te_raw = compute_transfer_entropy_matrix(X_win, k=k_history, normalize=False)
            te_mat = preprocess_adjacency_matrix(
                te_raw,
                invert=True,
                allow_reflexive=False,
                allow_bijective=False,
                normalize=False,
            )
            np.save(te_path, te_mat)
            pd = compute_persistence_diagram(te_mat, homology_dimensions=ph_dims)
            np.save(pd_path, pd)
            auc = betti_auc(pd, ph_dims)
            np.save(auc_path, auc)
        print(f"Done. Results saved under → {tda_root}\n")


if __name__ == "__main__":
    main()
