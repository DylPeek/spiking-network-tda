"""
Date: Feb 16, 2026
Description: Small I/O helpers for saving and loading intermediate artifacts (adjacency matrices, persistence diagrams, and Betti summaries).
Inputs:
  - Numpy arrays and tabular Betti-number records.
Outputs:
  - .npy and .csv files written to disk, plus corresponding load functions.
"""

import numpy as np
import pandas as pd
import os

def save_adjacency_matrix(matrix, filename):
    np.save(filename, matrix)

def load_adjacency_matrix(filename):
    return np.load(filename)

def save_betti_numbers_to_csv(betti_numbers, csv_path):
    # betti_numbers: list of tuples (start_time, b0, b1, ...)
    columns = ["Time Window"] + [f"Betti_{i}" for i in range(len(betti_numbers[0]) - 1)]
    df = pd.DataFrame(betti_numbers, columns=columns)
    df.to_csv(csv_path, index=False)

def load_betti_numbers_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    return df.values.tolist()

def save_persistence_diagram(diagram, filename):
    np.save(filename, diagram)

def load_persistence_diagram(filename):
    return np.load(filename)
