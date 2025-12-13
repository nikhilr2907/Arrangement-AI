
import librosa
import numpy as np

def extract_feature_matrix(harmonic_clips: list[tuple[np.ndarray, int]], main_melody: tuple[np.ndarray, int]):
    """
    Extracts the combined, time-series feature matrix for arrangement analysis.

    The output matrix F has the shape (N_features x N_frames), where each column
    F[:, t] is the synchronized feature vector for time step t (25 features:
    12 Melody Chroma + 12 Harmony Chroma + 1 Rhythmic Density).

    Args:
        harmonic_clips: A list of tuples (audio_array, sample_rate) for harmony parts.
        main_melody: A tuple (audio_array, sample_rate) for the main melody.

    Returns:
        A 2D numpy array (Feature Matrix) where columns are time steps (frames).
    """
    if main_melody is None or harmonic_clips is None:
        raise ValueError("Both main_melody and harmonic_clips must be provided.")

    y_melody, sr = main_melody
    n_fft = 2048
    hop_length = 512

    # --- 1. Calculate Melody Chroma (12 dim) ---
    y_harm_mel = librosa.effects.hpss(y_melody, margin=3.0, kernel_size=n_fft)[0]
    chroma_mel_matrix = librosa.feature.chroma_cens(
        y=y_harm_mel,
        sr=sr,
        hop_length=hop_length
    )

    # --- 2. Calculate and Average Harmony Chroma (12 dim) ---
    chroma_matrices = []
    for y_clip, clip_sr in harmonic_clips:
        y_harm = librosa.effects.hpss(y_clip, margin=3.0, kernel_size=n_fft)[0]
        chroma_matrix = librosa.feature.chroma_cens(
            y=y_harm,
            sr=clip_sr,
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
    mixed_signals = [y_clip[:min(y_clip.shape[0], y_melody.shape[0])] for y_clip, _ in harmonic_clips] + [y_melody]
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

