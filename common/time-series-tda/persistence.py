"""
Date: Feb 16, 2026
Description: Persistent-homology wrappers and helpers (FlagserPersistence via giotto-tda) for directed or undirected adjacency matrices.
Inputs:
  - Weighted adjacency matrix (numpy.ndarray or sparse), typically derived from transfer entropy.
  - Homology dimensions and directed/undirected setting.
Outputs:
  - Persistence diagram arrays and optional derived summaries (Betti counts, persistence images).
"""

import numpy as np
from gtda.homology import FlagserPersistence
from scipy.sparse import csr_matrix
from gtda.diagrams import Scaler, PersistenceImage


def compute_persistence_diagram(adjacency_matrix, homology_dimensions=[0,1], directed=True, filtration="max"):
    """
    Compute persistence diagram using FlagserPersistence from gtda.
    """
    sparse_matrix = csr_matrix(adjacency_matrix)
    sparse_matrix.eliminate_zeros()

    fp = FlagserPersistence(directed=directed, homology_dimensions=homology_dimensions, n_jobs=-1)
    diagrams = fp.fit_transform([sparse_matrix])
    return diagrams[0]

def extract_betti_numbers(diagram, thresholds):
    # Initialize betti numbers
    betti_numbers = [0]*len(thresholds)
    if diagram is None or len(diagram) == 0:
        return betti_numbers

    for feature in diagram:
        birth, death, dim = feature
        dim = int(dim)
        if dim < len(thresholds):
            if (death - birth) >= thresholds[dim]:
                betti_numbers[dim] += 1
    return betti_numbers

def clean_diagram(diagram, upper_bound=1.1):
    diagram = diagram.copy()
    diagram[np.isposinf(diagram)] = upper_bound
    diagram[np.isneginf(diagram)] = 0
    diagram_cleaned = diagram[~np.isnan(diagram).any(axis=1)]
    return diagram_cleaned

def persistence_diagram_to_images(diagram, sigma=0.1, n_bins=100, weight_function=None):
    # Clean the diagram
    diagram_clean = clean_diagram(diagram)
    if diagram_clean.size == 0:
        print("Warning: Diagram is empty after cleaning, skipping.")
        return None

    # Scale the diagram
    scaler = Scaler()
    diagram_scaled = scaler.fit_transform([diagram_clean])[0]

    # Define and compute the persistence image
    pi_transformer = PersistenceImage(
        sigma=sigma,
        n_bins=n_bins,
        weight_function=weight_function
    )

    persistence_images = pi_transformer.fit_transform([diagram_scaled])[0]

    return persistence_images

def compute_betti_numbers(adjacency_matrix, thresholds=[0.1,0.1], homology_dimensions=[0,1,2,3]):
    diag = compute_persistence_diagram(adjacency_matrix, homology_dimensions=tuple(range(len(thresholds))))
    betti = extract_betti_numbers(diag, thresholds)
    return betti, diag
