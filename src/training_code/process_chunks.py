from audio_training_breakdown_automation.audio_breakdown import run as audio_breakdown_run
import numpy as np 

class DataTransformation:
    """ This is meant to conduct the full data pipeline operation to create training data sequences"""
    def forward(self,stem_paths:list[str],tempo:float=None,use_manual_tempo:bool=True) -> None :
        """ Run the complete audio breakdown and processing pipeline.

        Args:
            stem_paths: List of file paths to audio stems
            tempo: Optional tempo value to use for beat tracking
            use_manual_tempo: Whether to use the provided tempo or estimate it
            saves the processed vectors to storage file.
        Returns:
            None
        """
        overall_vectors ,chunked_training_segments = audio_breakdown_run(
            stem_paths,tempo,use_manual_tempo
        )
        np.savez_compressed(
            "processed_training_data.npz",
            my_array=overall_vectors,
            
        )
        