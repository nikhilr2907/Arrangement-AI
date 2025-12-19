import numpy as np


class ComputeSimilarityMapping:
    def __init__(self, predicted_clusters, candidate_points: np.ndarray):

        self.predicted_clusters = predicted_clusters
        self.candidate_points = candidate_points

    def cosine_similarity(self,vector_1, vector_2) -> float:
        """ Find Most Similar Vector to Hidden Dim Vector using Cosine Similarity """
        dot_product = np.dot(vector_1, vector_2)
        norm_vector_1 = np.linalg.norm(vector_1)
        norm_vector_2 = np.linalg.norm(vector_2)
        if norm_vector_1 == 0 or norm_vector_2 == 0:
            return 0.0
        return dot_product / (norm_vector_1 * norm_vector_2)
    
    def find_most_similar_vector(self,hidden_vector,candidates) -> np.ndarray:
        """ Find the most similar vector from candidates to the hidden vector """
        max_similarity = -1
        most_similar_vector = None
        for candidate in candidates:
            similarity = self.cosine_similarity(hidden_vector, candidate)
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_vector = candidate
        return most_similar_vector
    
    def inverse_map_tokens_to_features(self) -> np.ndarray:
        """ Map predicted cluster tokens back to feature vectors using similarity mapping """
        mapped_features = []
        for cluster_token in self.predicted_clusters:
            candidate_vectors = self.candidate_points[cluster_token]
            hidden_vector = np.mean(candidate_vectors, axis=0)
            most_similar_vector = self.find_most_similar_vector(hidden_vector, candidate_vectors)
            mapped_features.append(most_similar_vector)
        return np.array(mapped_features)




        
        
