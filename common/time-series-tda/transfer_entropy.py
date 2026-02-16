"""
Date: Feb 16, 2026
Description: Compute pairwise transfer entropy (TE) matrices for discrete time-series signals using pyinform.
Inputs:
  - Discrete signals as a numpy array of shape (N, T), where N is the number of channels and T is time.
  - TE history length parameter k and optional normalization flag.
Outputs:
  - Dense TE matrix (numpy.ndarray, shape N x N) where entry (i, j) is TE(i -> j).
"""

import itertools
import numpy as np
import pyinform

def compute_transfer_entropy_matrix(signals, k=1, normalize=True):
    """
    Computes a pairwise transfer entropy matrix for a given set of signals.
    signals: numpy array of shape (N, W)
    k: embedding parameter for TE

    returns: adjacency matrix (N, N) where element (i,j) = TE(i->j)
    """
    N = signals.shape[0]
    matrix = np.zeros((N, N))
    max_te_value = 0

    for i, j in itertools.combinations(range(N), 2):
        te_ij = pyinform.transferentropy.transfer_entropy(signals[i], signals[j], k=k)
        te_ji = pyinform.transferentropy.transfer_entropy(signals[j], signals[i], k=k)
        matrix[i, j] = te_ij
        matrix[j, i] = te_ji
        max_te_value = max(max_te_value, te_ij, te_ji)

    # Normalize if needed
    if normalize and max_te_value > 0:
        matrix /= max_te_value
    return matrix
