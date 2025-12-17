from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle


class GlobalCodebook:
    """
    Global codebook for vector quantization of audio feature matrices.

    Train once on a corpus of data, then use to quantize new samples consistently.
    """

    def __init__(self, num_categories: int):
        """
        Args:
            num_categories: Number of clusters in the codebook
        """
        self.num_categories = num_categories
        self.kmeans = KMeans(n_clusters=num_categories, random_state=0)
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, feature_matrices: list[np.ndarray]):
        """
        Train the codebook on a corpus of feature matrices.

        Args:
            feature_matrices: List of matrices, all with shape (n_features, n_frames)

        Returns:
            self (for chaining)
        """
        if len(feature_matrices) == 0:
            raise ValueError("feature_matrices cannot be empty")

        # Verify all matrices have the same shape
        first_shape = feature_matrices[0].shape
        for i, fm in enumerate(feature_matrices):
            if fm.shape != first_shape:
                raise ValueError(
                    f"All feature matrices must have the same shape. "
                    f"Matrix 0 has shape {first_shape}, but matrix {i} has shape {fm.shape}"
                )

        # Flatten and stack all matrices
        flattened = [fm.flatten() for fm in feature_matrices]
        X = np.array(flattened)

        # Fit scaler
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)

        # Fit KMeans
        self.kmeans.fit(X_scaled)

        self.is_fitted = True
        return self

    def transform(self, feature_matrices: list[np.ndarray]) -> np.ndarray:
        """
        Quantize feature matrices using the learned codebook.

        Args:
            feature_matrices: List of matrices to quantize

        Returns:
            Array of shape (n_matrices,) with cluster labels
        """
        if not self.is_fitted:
            raise RuntimeError("Codebook must be fitted before transform. Call fit() first.")

        if len(feature_matrices) == 0:
            raise ValueError("feature_matrices cannot be empty")

        # Flatten matrices
        flattened = [fm.flatten() for fm in feature_matrices]
        X = np.array(flattened)

        # Scale using fitted scaler
        X_scaled = self.scaler.transform(X)

        # Predict using fitted KMeans
        labels = self.kmeans.predict(X_scaled)

        return labels

    def fit_transform(self, feature_matrices: list[np.ndarray]) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(feature_matrices).transform(feature_matrices)

    def get_centroids(self) -> np.ndarray:
        """
        Get the learned cluster centroids.

        Returns:
            Array of shape (num_categories, n_features * n_frames)
        """
        if not self.is_fitted:
            raise RuntimeError("Codebook not fitted yet")
        return self.kmeans.cluster_centers_

    def save(self, filepath: str):
        """Save the codebook to disk."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted codebook")

        state = {
            'num_categories': self.num_categories,
            'kmeans': self.kmeans,
            'scaler': self.scaler,
            'is_fitted': self.is_fitted
        }

        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, filepath: str):
        """Load a codebook from disk."""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        codebook = cls(num_categories=state['num_categories'])
        codebook.kmeans = state['kmeans']
        codebook.scaler = state['scaler']
        codebook.is_fitted = state['is_fitted']

        return codebook
