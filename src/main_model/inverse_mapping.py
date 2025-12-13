import numpy as np


class ComputeSimilarityMapping:
    def __init__(self, predicted_clusters, candidate_points: np.ndarray):

        self.predicted_clusters = predicted_clusters
        self.candidate_points = candidate_points


    def scoring_function(self,predicted_clusters,predictions) -> float:
        """ Compute similarity score between two points """
        
        return np.sqrt((self.candidate_points[predicted_clusters] - predictions)**2)
    
    def autoregressive_prediction(self,starting_melody) -> np.ndarray:

        self.predictions = np.empty_like(self.predicted_clusters)
        self.predictions[0] = self.scoring_function(starting_melody,self.candidate_points[0]).argmin()
        """ Sequentially predicts """
        for i,value in enumerate(self.predicted_clusters,start=1):
            self.predictions[i] = self.candidate_points[i][np.argmin(self.scoring_function(self.predictions[i-1],self.candidate_points[value]))]

        return self.predictions
