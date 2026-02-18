"""
Feature extraction from bar-level audio clips.

Two entry points:
    extract_feature_matrix()    → (25, T_frames)  full time-series, used internally
    extract_bar_feature_vector() → (25,)           mean-pooled, used for VQ-VAE training
"""

from typing import List, Optional

import librosa
import numpy as np


def extract_feature_matrix(
    main_melody: Optional[np.ndarray] = None,
    harmonic_clips: Optional[np.ndarray] = None,
    sr: int = 22050,
) -> np.ndarray:
    """
    Extracts the combined time-series feature matrix for a bar.

    Args:
        main_melody:    Array of audio arrays for leading melody clips
        harmonic_clips: Array of audio arrays for harmony parts
        sr:             Sample rate

    Returns:
        (25, T_frames) numpy array
        Rows: 12 melody chroma + 12 harmony chroma + 1 rhythmic density
    """
    has_melody  = main_melody is not None and len(main_melody) > 0
    has_harmony = harmonic_clips is not None and len(harmonic_clips) > 0

    if not has_melody and not has_harmony:
        raise ValueError("At least one of main_melody or harmonic_clips must be provided.")

    n_fft      = 2048
    hop_length = 512

    # --- Melody chroma (12 dim) ---
    melody_chroma_matrices = []
    melody_signals = []
    if has_melody:
        for y_melody in main_melody:
            y_melody = np.array(y_melody, dtype=np.float32)
            y_harm   = librosa.effects.hpss(y_melody, margin=3.0, kernel_size=n_fft)[0]
            chroma   = librosa.feature.chroma_cens(y=y_harm, sr=sr, hop_length=hop_length)
            melody_chroma_matrices.append(chroma)
            melody_signals.append(y_melody)

    # --- Harmony chroma (12 dim) ---
    harmony_chroma_matrices = []
    harmony_signals = []
    if has_harmony:
        for y_clip in harmonic_clips:
            y_clip = np.array(y_clip, dtype=np.float32)
            y_harm = librosa.effects.hpss(y_clip, margin=3.0, kernel_size=n_fft)[0]
            chroma = librosa.feature.chroma_cens(y=y_harm, sr=sr, hop_length=hop_length)
            harmony_chroma_matrices.append(chroma)
            harmony_signals.append(y_clip)

    # Align frame counts
    all_chromas = melody_chroma_matrices + harmony_chroma_matrices
    min_frames  = min(m.shape[1] for m in all_chromas)

    chroma_mel  = (
        np.mean([m[:, :min_frames] for m in melody_chroma_matrices], axis=0)
        if has_melody else np.zeros((12, min_frames))
    )
    chroma_harm = (
        np.mean([m[:, :min_frames] for m in harmony_chroma_matrices], axis=0)
        if has_harmony else np.zeros((12, min_frames))
    )

    # --- Rhythmic density (1 dim) ---
    all_signals     = melody_signals + harmony_signals
    min_len         = min(s.shape[0] for s in all_signals)
    y_mix           = np.sum([s[:min_len] for s in all_signals], axis=0)
    onset_env       = librosa.onset.onset_strength(y=y_mix, sr=sr, hop_length=hop_length)
    rhythm_matrix   = onset_env[:min_frames][np.newaxis, :]

    return np.vstack([chroma_mel, chroma_harm, rhythm_matrix])  # (25, T_frames)


def extract_bar_feature_vector(
    melody_clips: Optional[List[np.ndarray]],
    harmony_clips: Optional[List[np.ndarray]],
    sr: int = 22050,
) -> np.ndarray:
    """
    Extract a single 25-dim feature vector for one bar position.

    Mean-pools the time-series matrix along the frame axis so the result
    is a fixed-size descriptor regardless of bar length. This is the input
    format expected by the VQ-VAE encoder.

    Args:
        melody_clips:   List of bar-length audio arrays for melody stems
        harmony_clips:  List of bar-length audio arrays for harmony stems
        sr:             Sample rate

    Returns:
        (25,) float32 numpy array
    """
    matrix = extract_feature_matrix(
        main_melody    = np.array(melody_clips,  dtype=object) if melody_clips  else None,
        harmonic_clips = np.array(harmony_clips, dtype=object) if harmony_clips else None,
        sr             = sr,
    )
    return matrix.mean(axis=1).astype(np.float32)  # (25, T) → (25,)
