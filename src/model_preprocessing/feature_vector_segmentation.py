from typing import TypeVar
import librosa 
import numpy as np 

class AudioClip:
    def __init__(self, y: np.ndarray, sr: int):
        self.y = y
        self.sr = sr

def extract_feature_matrix(harmonic_clips: list[AudioClip], main_melody: AudioClip):
    """
    Extracts the combined, time-series feature matrix for arrangement analysis.
    
    The output matrix F has the shape (N_features x N_frames), where each column
    F[:, t] is the synchronized feature vector for time step t (25 features: 
    12 Melody Chroma + 12 Harmony Chroma + 1 Rhythmic Density).

    Args:
        harmonic_clips: A list of AudioClip objects for harmony parts.
        main_melody: The single AudioClip object for the main melody.

    Returns:
        A 2D numpy array (Feature Matrix) where columns are time steps (frames).
    """
    if not main_melody or not harmonic_clips:
        raise ValueError("Both main_melody and harmonic_clips must be provided.")

    sr = main_melody.sr
    n_fft = 2048
    hop_length = 512

    # --- 1. Calculate Melody Chroma (12 dim) ---
    y_harm_mel = librosa.effects.hpss(main_melody.y, margin=3.0, kernel_size=n_fft)[0]
    chroma_mel_matrix = librosa.feature.chroma_cens(
        y=y_harm_mel, 
        sr=sr, 
        hop_length=hop_length
    )
    
    # --- 2. Calculate and Average Harmony Chroma (12 dim) ---
    chroma_matrices = []
    for clip in harmonic_clips:
        y_harm = librosa.effects.hpss(clip.y, margin=3.0, kernel_size=n_fft)[0]
        chroma_matrix = librosa.feature.chroma_cens(
            y=y_harm, 
            sr=sr, 
            hop_length=hop_length
        )
        chroma_matrices.append(chroma_matrix)

    if not chroma_matrices:
        raise ValueError("No valid chroma features extracted from harmony clips.")
    
    # Determine minimum number of frames for consistent feature lengths
    min_frames = min(m.shape[1] for m in chroma_matrices + [chroma_mel_matrix])
    
    # Average and truncate harmony chroma matrices
    chroma_harm_matrix = np.mean(
        [m[:, :min_frames] for m in chroma_matrices], 
        axis=0
    )
    chroma_mel_matrix = chroma_mel_matrix[:, :min_frames]

    # --- 3. Calculate Rhythmic Density (1 dim) ---
    
    # Create unified mixed signal (truncated)
    mixed_signals = [c.y[:min(c.y.shape[0], main_melody.y.shape[0])] for c in harmonic_clips] + [main_melody.y]
    y_mix = np.sum(mixed_signals, axis=0)

    # Calculate Onset Strength Function
    o_env = librosa.onset.onset_strength(
        y=y_mix, 
        sr=sr, 
        hop_length=hop_length
    )
    
    # Truncate and reshape density vector for concatenation
    rhythm_density_vector = o_env[:min_frames]
    rhythm_density_matrix = rhythm_density_vector[np.newaxis, :]
    
    # --- 4. Concatenate Features ---
    feature_matrix = np.vstack([
        chroma_mel_matrix, 
        chroma_harm_matrix, 
        rhythm_density_matrix
    ])
    
    return feature_matrix

