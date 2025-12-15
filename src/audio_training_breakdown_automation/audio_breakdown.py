from collections import defaultdict
from pathlib import Path
from typing import Dict, List,Tuple
import os
import librosa
import numpy as np


from src.latent_preprocessing.melodic_candidates import find_melodic_candidates
from src.latent_preprocessing.extract_leading_melody import is_leading_melody
from src.model_preprocessing.feature_vector_segmentation import extract_feature_matrix
from src.model_preprocessing.vector_quantisation import vector_quantisation
from src.model_preprocessing.chunking_transformation import chunking_transformation

def process_audio_stems(stems: List[str]) -> Dict[str, np.ndarray]:
    """
    Process audio stems into bars with filename tracking.
    """
    # Load audio files with sample rate
    loaded_audio = {Path(stem).name: librosa.load(stem) for stem in stems}

    # Convert each into bars and sort into groups where bars are not empty
    BARS = {}
    for stem_filename, (audio_array, sr) in loaded_audio.items():
        # Extract filename without extension for key
        filename = Path(stem_filename).stem

        # Detect beats
        tempo, beat_frames = librosa.beat.beat_track(y=audio_array, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)

        # Estimate bars (assuming 4/4 time signature)
        beats_per_bar = 4
        bar_indices = beat_frames[::beats_per_bar]

        # Split audio into bars
        bars_for_stem = []
        for i in range(len(bar_indices) - 1):
            start_sample = librosa.frames_to_samples(bar_indices[i])
            end_sample = librosa.frames_to_samples(bar_indices[i + 1])
            bar_audio = audio_array[start_sample:end_sample]

            # Filter out empty bars (check if RMS energy is above threshold)
            if np.sqrt(np.mean(bar_audio**2)) > 0.01:
                bars_for_stem.append(bar_audio)

        # Add to BARS dictionary if bars exist
        if bars_for_stem:
            BARS[filename] = np.array(bars_for_stem, dtype=object)

    return BARS


def extract_melodic_content(BARS: Dict[str, np.ndarray], sr: int = 22050) -> Dict[str, tuple]:
    """
    Extract the leading melodic content from bars.

    Args:
        BARS: Dictionary mapping filename to array of bars
        sr: Sample rate for audio

    Returns:
        Dictionary mapping filename to (audio_array, sample_rate) for leading melody group
    """
    # Find the candidates which can be main melodies
    melodic_candidates = find_melodic_candidates(BARS, activity_threshold=0.5)

    # Convert bars to AudioClip format (audio_array, sample_rate)
    audio_clips = {}
    for filename, bars in melodic_candidates.items():
        # Concatenate all bars for this stem into a single array
        concatenated = np.concatenate(bars)
        audio_clips[filename] = (concatenated, sr)

    # Find the leading melody group
    leading_melody = is_leading_melody(audio_clips, max_group_size=3, compatibility_threshold=0.3)

    return leading_melody

def find_pattern_arrangement(BARS: Dict[str, np.ndarray]):
    """ Organise BARS into a proper sequence of arrays."""
    structure = defaultdict(int)
    for i in range(len(BARS[0])):
        # For each time step extract the vectors for the present instruments so any vector with non zero values in it, group into a dictionary
        # where each key are the names of instruments and the values are arrays of soundwave arrays for that particular bar.
        time_step_dict = defaultdict(int)
        for stem_name in BARS.keys():
            if i < len(BARS[stem_name]):
                time_step_dict[stem_name] = BARS[stem_name][i] if np.sum(BARS[stem_name][i]) > 0 else None
        structure[i] = time_step_dict
    return structure        


    
def sort_structure(BARS: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Sort stems into melody and harmony groups based on melodic content.

    Args:
        BARS: Dictionary mapping filename to array of bars
    """
    # Find a sequence where all melody clips are playing together, must be a sequence for this
    melodic_candidates = find_melodic_candidates(BARS, activity_threshold=0.5)
    harmony_candidates = {k: v for k, v in BARS.items() if k not in melodic_candidates}
    return melodic_candidates, harmony_candidates
    
def process_melodic_harmony_groups(melodic_stems, harmony_stems, BARS):
    """ Sort into matrices for each time step representing feature vectors"""
    structure = find_pattern_arrangement(BARS)
    overall_vectors = []
    for time_step, stem_dict in structure.items():
        melodic_clips = []
        harmony_clips = []
        for stem_name, bar_audio in stem_dict.items():
            if bar_audio is not None:
                if stem_name in melodic_stems:
                    melodic_clips.append(bar_audio)
                    # Assuming sample rate 22050
                elif stem_name in harmony_stems:
                    harmony_clips.append(bar_audio)  
        overall_vectors.append((np.array(melodic_clips), np.array(harmony_clips)))
    return overall_vectors

def convert_to_feature_matrices(overall_vectors):
    """ Convert the overall vectors into feature matrices for each time step."""
    feature_matrices = []
    for melodic_clips, harmony_clips in overall_vectors:
        # Assuming leading melody is the first melodic clip
        if len(melodic_clips) == 0:
            continue
        leading_melody_clip = melodic_clips
        harmonic_clips = harmony_clips

        feature_matrix = extract_feature_matrix(harmonic_clips, leading_melody_clip)
        feature_matrices.append(feature_matrix)
    return np.array(feature_matrices)


def process_harmony_groups_clustering(harmony_stems):

    # Process harmony groups into clustered categories.
    clustered_harmony = vector_quantisation(harmony_stems, num_categories=3)
    return clustered_harmony
    
def chunk_into_training_segments(feature_matrices):

    # Chunk the feature matrices into training segments.
    return chunking_transformation(feature_matrices, chunk_size=32, overlap=0.5)



def run(stem_paths: List[str]) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Main function to run the audio breakdown and processing pipeline.

    Args:
        stem_paths: List of file paths to audio stems
    Returns:
    """
    # Step 1: Process audio stems into bars
    BARS = process_audio_stems(stem_paths)
    print(f"Processed {len(BARS)} stems into bars.")
    # Step 2: Extract melodic and harmony groups
    melodic_stems, harmony_stems = sort_structure(BARS)
    print(f"Identified {len(melodic_stems)} melodic stems and {len(harmony_stems)} harmony stems.")
    # Step 3: Process melodic and harmony groups into feature vectors
    overall_vectors = process_melodic_harmony_groups(melodic_stems, harmony_stems, BARS)
    print(f"Processed overall vectors for {len(overall_vectors)} time steps.")
    # Step 4: Convert to feature matrices
    feature_matrices = convert_to_feature_matrices(overall_vectors)
    print(f"Converted to feature matrices with shape {feature_matrices.shape}.")
    # Step 5: Process harmony groups with clustering
    clustered_harmony = process_harmony_groups_clustering(feature_matrices)
    print(f"Processed harmony groups into {len(clustered_harmony)} clustered categories.")
    # Step 6: Chunk into training segments
    training_segments = chunk_into_training_segments(clustered_harmony)
    print(f"Chunked into {len(training_segments)} training segments.")
    return BARS, training_segments
