"""
Date: Feb 16, 2026
Description: Preprocess transfer-entropy adjacency matrices into a form suitable for persistent-homology computations.
Inputs:
  - TE adjacency matrix (numpy.ndarray, shape N x N).
  - Options controlling inversion, normalization, self-loop removal, and asymmetry enforcement.
Outputs:
  - Preprocessed adjacency matrix (numpy.ndarray, shape N x N).
"""

import numpy as np

from scipy.stats import norm

def normalize_to_gaussian(matrix):
    flat = matrix.flatten()
    cdf = np.argsort(np.argsort(flat)) / len(flat)  # Compute ranks as a fraction of total
    transformed = norm.ppf(cdf)  # Map to Gaussian percent-point function
    return transformed.reshape(matrix.shape)

def preprocess_adjacency_matrix(matrix, invert=False, allow_reflexive=False, allow_bijective=False, normalize=True):
    processed_matrix = matrix.copy()

    # Handle bidirectional edges
    if not allow_bijective:
        for i in range(processed_matrix.shape[0]):
            for j in range(i+1, processed_matrix.shape[1]):
                if processed_matrix[i, j] >= processed_matrix[j, i]:
                    processed_matrix[j, i] = 0
                else:
                    processed_matrix[i, j] = 0

    # Invert if needed
    if invert:
        processed_matrix = -1 * processed_matrix
        W_max = processed_matrix.max()

    processed_matrix[np.isinf(processed_matrix)] = 0

    # Normalize to [0,1] if needed
    if normalize:
        min_val = processed_matrix.min()
        max_val = processed_matrix.max()
        if max_val > min_val:
            processed_matrix = (processed_matrix - min_val) / (max_val - min_val)
        else:
            processed_matrix = np.zeros_like(processed_matrix)

    # Remove reflexive connections
    if not allow_reflexive:
        np.fill_diagonal(processed_matrix, processed_matrix.min())

    return processed_matrix
