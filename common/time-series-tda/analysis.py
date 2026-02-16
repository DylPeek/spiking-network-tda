"""
Date: Feb 16, 2026
Description: Analysis utilities used by the project scripts (diagram vectorization, clustering helpers, and simple anomaly-score routines).
Inputs:
  - Persistence diagrams (lists/arrays of birth-death(-dimension) tuples) or their vectorized representations.
Outputs:
  - Numeric summaries such as centroids, vector embeddings, thresholds, and anomaly scores.
"""

import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

from sklearn.metrics import precision_recall_curve
import numpy as np

def compute_anomaly_scores(vectors, mean, covariance):
    """
    Compute anomaly scores for a set of vectors using the Mahalanobis-like distance.

    Parameters:
        vectors: np.ndarray
            Array of vectorized persistence diagrams with shape (n_samples, n_features).
        mean: np.ndarray
            Mean vector of shape (n_features,).
        covariance: np.ndarray
            Covariance matrix of shape (n_features, n_features).

    Returns:
        scores: np.ndarray
            Anomaly scores for each vector.
    """
    # Ensure inputs are numpy arrays
    vectors = np.array(vectors)
    mean = np.array(mean)
    covariance = np.array(covariance)

    # Compute inverse of the covariance matrix
    covariance_inv = np.linalg.inv(covariance)

    # Compute anomaly scores
    scores = np.array([
        (v - mean).T @ covariance_inv @ (v - mean)
        for v in vectors
    ])

    return scores


def find_optimal_threshold(y, scores):
    """
    Find the threshold t that maximizes precision and recall on the training data.

    Parameters:
        y: np.ndarray
            Ground truth binary labels for the data (1 for anomaly, 0 for normal).
        scores: np.ndarray
            Anomaly scores computed for the data.

    Returns:
        optimal_t: float
            The threshold t that maximizes precision and recall.
        precision: float
            Precision at the optimal threshold.
        recall: float
            Recall at the optimal threshold.
    """
    # Compute precision-recall curve
    precisions, recalls, thresholds = precision_recall_curve(y, scores)

    # Compute F1 score to identify the best threshold
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)

    optimal_t = thresholds[best_idx]
    return optimal_t, precisions[best_idx], recalls[best_idx]


def compute_mean_and_covariance(vectors):
    """
    Compute the mean vector and covariance matrix from a set of vectorized persistence diagrams.

    Parameters:
        vectors: np.ndarray
            Array of vectorized persistence diagrams with shape (n_samples, n_features),
            where n_samples is the number of vectors and n_features is the dimensionality of each vector.

    Returns:
        mean: np.ndarray
            Mean vector of shape (n_features,).
        covariance: np.ndarray
            Covariance matrix of shape (n_features, n_features).
    """
    if vectors.ndim != 2:
        raise ValueError("Input vectors must be a 2D array with shape (n_samples, n_features).")

    # Compute mean vector
    mean = np.mean(vectors, axis=0)

    # Compute covariance matrix
    covariance = np.cov(vectors, rowvar=False)  # rowvar=False ensures features are along columns

    return mean, covariance


def compute_centroids_by_dimension(diagrams, num_centroids=10, max_iter=300, random_state=None):
    """
    Compute centroids for each homology dimension in persistence diagrams using K-Means.

    Parameters:
        diagrams: list of np.ndarray
            List of persistence diagrams, each with shape (N, 3) (birth, death, dimension).
        num_centroids: int
            Number of centroids to compute for each homology dimension.
        max_iter: int
            Maximum iterations for K-Means clustering.
        random_state: int or None
            Random seed for reproducibility.

    Returns:
        centroids_by_dimension: dict
            Dictionary where keys are homology dimensions (0, 1, 2) and values are arrays of centroids.
    """
    centroids_by_dimension = {}

    for dim in [0, 1, 2]:  # Process H0, H1, H2
        print(f"Computing centroids for H{dim}...")
        # Extract points for the current dimension
        points = np.vstack([d[d[:, 2] == dim, :2] for d in diagrams if len(d) > 0 and dim in d[:, 2]])

        if points.size == 0:
            print(f"No points found for H{dim}. Skipping...")
            centroids_by_dimension[dim] = np.array([])  # Empty array for this dimension
            continue

        # Fit K-Means
        kmeans = KMeans(n_clusters=num_centroids, max_iter=max_iter, random_state=random_state)
        kmeans.fit(points)
        centroids_by_dimension[dim] = kmeans.cluster_centers_

    return centroids_by_dimension


def compute_centroids(diagrams, num_centroids=10, max_iter=300, random_state=None):
    """
    Compute centroids for persistence diagrams using K-Means.

    Parameters:
        diagrams: list of np.ndarray
            List of persistence diagrams, each with shape (N, 2) or (N, 3) (birth, death, optional dimension).
        num_centroids: int
            Number of centroids to compute.
        max_iter: int
            Maximum iterations for K-Means clustering.
        random_state: int or None
            Random seed for reproducibility.

    Returns:
        centroids: np.ndarray
            Array of centroids, shape (num_centroids, 2).
    """
    # Stack all points from all diagrams
    all_points = np.vstack([d[:, :2] for d in diagrams if len(d) > 0])  # Exclude dimension if present

    print("Inspecting all_points data...")
    print("Shape:", all_points.shape)
    print("Max value:", np.max(all_points))
    print("Min value:", np.min(all_points))
    print("Contains Inf:", np.any(np.isinf(all_points)))
    print("Contains NaN:", np.any(np.isnan(all_points)))

    # Run K-Means clustering
    kmeans = KMeans(n_clusters=num_centroids, max_iter=max_iter, random_state=random_state)
    kmeans.fit(all_points)
    centroids = kmeans.cluster_centers_

    return centroids

def vectorize_diagram_by_dimension(diagram, centroids_by_dimension, bandwidth=0.1):
    """
    Vectorize a single persistence diagram for each homology dimension using separate centroids.

    Parameters:
        diagram: np.ndarray
            A single persistence diagram, shape (N, 3) (birth, death, dimension).
        centroids_by_dimension: dict
            Dictionary where keys are homology dimensions (0, 1, 2) and values are arrays of centroids.
        bandwidth: float
            Bandwidth for the Gaussian kernel.

    Returns:
        vector: np.ndarray
            Concatenated vectorized representation for all dimensions, shape (num_centroids * num_dimensions,).
    """
    vectors = []

    for dim, centroids in centroids_by_dimension.items():
        if centroids.size == 0:
            # No centroids for this dimension
            vectors.append(np.zeros(len(centroids)))
            continue

        # Extract points for this dimension
        dim_points = diagram[diagram[:, 2] == dim, :2]

        if dim_points.size == 0:
            # No points for this dimension in the diagram
            vectors.append(np.zeros(len(centroids)))
            continue

        # Compute distances and apply Gaussian kernel
        distances = cdist(dim_points, centroids, metric="euclidean")
        weights = np.exp(-distances**2 / (2 * bandwidth**2))
        vector = weights.sum(axis=0)
        vectors.append(vector)

    # Concatenate vectors for all dimensions
    return np.concatenate(vectors)


def vectorize_diagram(diagram, centroids, bandwidth=0.1):
    """
    Vectorize a single persistence diagram using centroids.

    Parameters:
        diagram: np.ndarray
            A single persistence diagram, shape (N, 2) or (N, 3).
        centroids: np.ndarray
            Array of centroids, shape (num_centroids, 2).
        bandwidth: float
            Bandwidth for the Gaussian kernel.

    Returns:
        vector: np.ndarray
            Vectorized representation of the diagram, shape (num_centroids,).
    """
    if diagram.size == 0:
        return np.zeros(len(centroids))

    # Compute distances between points and centroids
    distances = cdist(diagram[:, :2], centroids, metric="euclidean")

    # Apply Gaussian kernel
    weights = np.exp(-distances**2 / (2 * bandwidth**2))

    # Aggregate weights to form the vector
    vector = weights.sum(axis=0)

    return vector


def vectorize_diagrams(diagrams, centroids, bandwidth=0.1):
    """
    Vectorize a list of persistence diagrams using centroids.

    Parameters:
        diagrams: list of np.ndarray
            List of persistence diagrams.
        centroids: np.ndarray
            Array of centroids, shape (num_centroids, 2).
        bandwidth: float
            Bandwidth for the Gaussian kernel.

    Returns:
        vectors: np.ndarray
            Matrix of vectorized diagrams, shape (len(diagrams), num_centroids).
    """
    vectors = np.array([vectorize_diagram(d, centroids, bandwidth) for d in diagrams])
    return vectors
