from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np


class HierarchicalCodebook:
    """
    Two-level hierarchical codebook for arrangement patterns.

    Level 1: Coarse-grained arrangement types (intro, verse, chorus, bridge, outro)
    Level 2: Fine-grained variations within each type
    """

    def __init__(self, num_coarse_clusters: int = 8, num_fine_clusters: int = 4):
        """
        Args:
            num_coarse_clusters: Number of high-level arrangement types
            num_fine_clusters: Number of variations per type
        """
        self.num_coarse = num_coarse_clusters
        self.num_fine = num_fine_clusters

        self.scaler = StandardScaler()
        self.coarse_kmeans = KMeans(n_clusters=num_coarse_clusters, random_state=0)
        self.fine_kmeans = {}  # One KMeans per coarse cluster
        self.is_fitted = False

    def fit(self, feature_matrices: list[np.ndarray]):
        """Train hierarchical codebook."""
        if len(feature_matrices) == 0:
            raise ValueError("feature_matrices cannot be empty")

        # Flatten and scale
        flattened = [fm.flatten() for fm in feature_matrices]
        X = np.array(flattened)

        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)

        # Level 1: Coarse clustering
        coarse_labels = self.coarse_kmeans.fit_predict(X_scaled)

        # Level 2: Fine clustering within each coarse cluster
        for coarse_id in range(self.num_coarse):
            # Get samples in this coarse cluster
            mask = coarse_labels == coarse_id
            X_cluster = X_scaled[mask]

            if len(X_cluster) > 0:
                # Train fine-grained KMeans for this cluster
                fine_kmeans = KMeans(
                    n_clusters=min(self.num_fine, len(X_cluster)),
                    random_state=0
                )
                fine_kmeans.fit(X_cluster)
                self.fine_kmeans[coarse_id] = fine_kmeans

        self.is_fitted = True
        return self

    def transform(self, feature_matrices: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """
        Transform matrices to hierarchical labels.

        Returns:
            (coarse_labels, fine_labels): Two arrays of cluster IDs
        """
        if not self.is_fitted:
            raise RuntimeError("Must fit before transform")

        flattened = [fm.flatten() for fm in feature_matrices]
        X = np.array(flattened)
        X_scaled = self.scaler.transform(X)

        # Coarse labels
        coarse_labels = self.coarse_kmeans.predict(X_scaled)

        # Fine labels
        fine_labels = np.zeros(len(X_scaled), dtype=int)

        for i, (x, coarse_id) in enumerate(zip(X_scaled, coarse_labels)):
            if coarse_id in self.fine_kmeans:
                fine_id = self.fine_kmeans[coarse_id].predict([x])[0]
                fine_labels[i] = fine_id
            else:
                fine_labels[i] = 0

        return coarse_labels, fine_labels

    def get_composite_labels(self, feature_matrices: list[np.ndarray]) -> np.ndarray:
        """
        Get composite labels combining coarse and fine.

        Returns:
            Array where each label = coarse_id * num_fine + fine_id
        """
        coarse, fine = self.transform(feature_matrices)
        return coarse * self.num_fine + fine
