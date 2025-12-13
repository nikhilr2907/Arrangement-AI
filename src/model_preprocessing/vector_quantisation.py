
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np


def vector_quantisation(feature_matrices: list[np.ndarray], num_categories: int) -> np.ndarray:
    """
    Cluster feature matrices into discrete categories using Frobenius norm.

    Each matrix is flattened into a vector and clustered using KMeans.
    The Euclidean distance between flattened matrices equals the Frobenius norm
    between the original matrices, making this approach mathematically equivalent
    to matrix clustering with Frobenius norm.

    Mathematical equivalence:
        ||A - B||_F = ||flatten(A) - flatten(B)||_2

    Args:
        feature_matrices: List of feature matrices, all with shape (n_features, n_frames).
                         All matrices must have identical dimensions.
        num_categories: Number of clusters to create.

    Returns:
        Array of shape (n_matrices,) containing cluster labels for each matrix.

    Example:
        >>> matrices = [matrix1, matrix2, matrix3]  # Each shape (25, 255)
        >>> labels = vector_quantisation(matrices, num_categories=4)
        >>> labels
        array([0, 2, 1])  # matrix1 is cluster 0, matrix2 is cluster 2, etc.
    """
    if not feature_matrices:
        raise ValueError("feature_matrices cannot be empty")

    # Verify all matrices have the same shape
    first_shape = feature_matrices[0].shape
    for i, fm in enumerate(feature_matrices):
        if fm.shape != first_shape:
            raise ValueError(
                f"All feature matrices must have the same shape. "
                f"Matrix 0 has shape {first_shape}, but matrix {i} has shape {fm.shape}"
            )

    # Flatten each matrix into a vector
    # For (25, 255) matrices, this creates (6375,) vectors
    flattened = [fm.flatten() for fm in feature_matrices]
    X = np.array(flattened)  # Shape: (n_matrices, n_features * n_frames)

    # Standardize features to give equal importance to all dimensions
    # Important: prevents features with larger scales from dominating
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Cluster using KMeans
    # Euclidean distance on flattened matrices = Frobenius norm on original matrices
    kmeans = KMeans(n_clusters=num_categories, random_state=0)
    labels = kmeans.fit_predict(X_scaled)

    return labels  # Shape: (n_matrices,)   







