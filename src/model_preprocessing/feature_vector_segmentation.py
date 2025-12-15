
from typing import List, Tuple, Union

import librosa
import numpy as np


def extract_feature_matrix(
    harmonic_clips: List[Tuple[np.ndarray, int]],
    main_melody: Union[Tuple[np.ndarray, int], List[Tuple[np.ndarray, int]]],
) -> np.ndarray:
    """
    Extracts the combined, time-series feature matrix for arrangement analysis.

    The output matrix F has the shape (N_features x N_frames), where each column
    F[:, t] is the synchronized feature vector for time step t (25 features:
    12 Melody Chroma + 12 Harmony Chroma + 1 Rhythmic Density).

    Args:
        harmonic_clips: A list of tuples (audio_array, sample_rate) for harmony parts.
        main_melody: Either a single tuple (audio_array, sample_rate) or a list of
                     tuples for multiple leading melody clips that will be averaged.

    Returns:
        A 2D numpy array (Feature Matrix) where columns are time steps (frames).
    """
    if main_melody is None or harmonic_clips is None:
        raise ValueError("Both main_melody and harmonic_clips must be provided.")

    # Handle both single melody and multiple melody clips
    if isinstance(main_melody, tuple):
        # Single melody clip
        melody_clips = [main_melody]
    else:
        # Multiple melody clips
        melody_clips = main_melody

    if not melody_clips:
        raise ValueError("main_melody must contain at least one clip.")

    # Use sample rate from first melody clip
    sr = melody_clips[0][1]
    n_fft = 2048
    hop_length = 512

    # --- 1. Calculate Melody Chroma (12 dim) ---
    # If multiple melody clips, compute chroma for each and average
    melody_chroma_matrices = []
    melody_signals = []

    for y_melody, melody_sr in melody_clips:
        y_harm_mel = librosa.effects.hpss(y_melody, margin=3.0, kernel_size=n_fft)[0]
        chroma_mel = librosa.feature.chroma_cens(
            y=y_harm_mel, sr=melody_sr, hop_length=hop_length
        )
        melody_chroma_matrices.append(chroma_mel)
        melody_signals.append(y_melody)

    # --- 2. Calculate and Average Harmony Chroma (12 dim) ---
    harmony_chroma_matrices = []
    harmony_signals = []

    for y_clip, clip_sr in harmonic_clips:
        y_harm = librosa.effects.hpss(y_clip, margin=3.0, kernel_size=n_fft)[0]
        chroma_matrix = librosa.feature.chroma_cens(
            y=y_harm, sr=clip_sr, hop_length=hop_length
        )
        harmony_chroma_matrices.append(chroma_matrix)
        harmony_signals.append(y_clip)

    if not harmony_chroma_matrices:
        raise ValueError("No valid chroma features extracted from harmony clips.")

    # Determine minimum number of frames for consistent feature lengths
    all_chroma_matrices = melody_chroma_matrices + harmony_chroma_matrices
    min_frames = min(m.shape[1] for m in all_chroma_matrices)

    # Average melody chroma matrices (if multiple leading melodies)
    chroma_mel_matrix = np.mean(
        [m[:, :min_frames] for m in melody_chroma_matrices], axis=0
    )

    # Average harmony chroma matrices
    chroma_harm_matrix = np.mean(
        [m[:, :min_frames] for m in harmony_chroma_matrices], axis=0
    )

    # --- 3. Calculate Rhythmic Density (1 dim) ---
    # Create unified mixed signal from all clips (melody + harmony)
    # Truncate all signals to minimum length
    min_signal_length = min(
        min(sig.shape[0] for sig in melody_signals),
        min(sig.shape[0] for sig in harmony_signals),
    )

    mixed_signals = (
        [sig[:min_signal_length] for sig in melody_signals]
        + [sig[:min_signal_length] for sig in harmony_signals]
    )
    y_mix = np.sum(mixed_signals, axis=0)

    # Calculate Onset Strength Function
    o_env = librosa.onset.onset_strength(y=y_mix, sr=sr, hop_length=hop_length)

    # Truncate and reshape density vector for concatenation
    rhythm_density_vector = o_env[:min_frames]
    rhythm_density_matrix = rhythm_density_vector[np.newaxis, :]

    # --- 4. Concatenate Features ---
    feature_matrix = np.vstack(
        [chroma_mel_matrix, chroma_harm_matrix, rhythm_density_matrix]
    )

    return feature_matrix

