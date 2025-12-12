
from sklearn.cluster import KMeans
import numpy as np



""" Train a VQ-VAE model for vector quantisation of audio features. Categorise multiple feature_matricies into discrete sequences. """
def vector_quantisation(feature_matrices: list[np.ndarray], num_categories: int) -> list[np.ndarray]:
    """ """
    ## Extract feature vectors from matrices
    feature_vectors = [fm.T for fm in feature_matrices]  # Transpose to
    ## Combine all feature vectors for clustering
    all_vectors = np.vstack(feature_vectors)
    ## Perform VQ - VAE clustering (placeholder for actual implementation) use KMeans for simplicity at the moment.

    # Reconstruct quantized feature matrices
    kmeans = KMeans(n_clusters=num_categories, random_state=0)
    kmeans.fit(all_vectors)
    quantized_matrices = []
    start = 0

    for fv in feature_vectors:
        end = start + fv.shape[0]
        labels = kmeans.labels_[start:end]
        quantized_matrices.append(labels)
        start = end
    return quantized_matrices   







