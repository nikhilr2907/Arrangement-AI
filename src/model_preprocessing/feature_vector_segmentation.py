"""
Feature extraction from bar-level audio clips.

Entry points:
    extract_bar_feature_vector() → (25,)  mean-pooled, used for VQ-VAE training

Feature layout (25 dims, no melody/harmony split needed):
    0–11   chroma_cens        (12)  pitch-class content, HPSS harmonic component
    12–17  spectral_contrast  ( 6)  texture / fullness across frequency bands
    18–21  MFCC 1–4           ( 4)  coarse timbral shape
    22     onset_strength     ( 1)  rhythmic density
    23     RMS                ( 1)  energy / dynamics
    24     ZCR                ( 1)  noisiness / percussiveness
"""

from typing import List

import librosa
import numpy as np


def extract_bar_feature_vector(
    clips: List[np.ndarray],
    sr: int = 22050,
) -> np.ndarray:
    """
    Extract a single 25-dim feature vector for one bar position.

    Mixes all stem clips together, then computes features on the mix.
    Per-clip chroma is averaged before mixing to avoid louder stems dominating
    the pitch content.

    Args:
        clips: List of bar-length audio arrays (one per stem, any number)
        sr:    Sample rate

    Returns:
        (25,) float32 numpy array
    """
    if not clips:
        return np.zeros(25, dtype=np.float32)

    n_fft      = 2048
    hop_length = 512

    # Align all clips to the same length
    min_len = min(len(c) for c in clips)
    aligned = [np.array(c[:min_len], dtype=np.float32) for c in clips]

    # Mixed signal for onset / RMS / ZCR (sum then normalise)
    mix = np.sum(aligned, axis=0)
    peak = np.abs(mix).max()
    if peak > 0:
        mix = mix / peak

    # --- Chroma (12 dims): average per-clip HPSS chroma, then mean over time ---
    chroma_matrices = []
    for y in aligned:
        y_harm = librosa.effects.hpss(y, margin=3.0, kernel_size=n_fft)[0]
        chroma = librosa.feature.chroma_cens(y=y_harm, sr=sr, hop_length=hop_length)
        chroma_matrices.append(chroma)

    min_frames  = min(m.shape[1] for m in chroma_matrices)
    chroma_mean = np.mean([m[:, :min_frames] for m in chroma_matrices], axis=0)  # (12, T)
    feat_chroma = chroma_mean.mean(axis=1)  # (12,)

    # --- Spectral contrast (6 dims): 5 sub-bands + 1 ---
    contrast = librosa.feature.spectral_contrast(
        y=mix, sr=sr, hop_length=hop_length, n_bands=5
    )  # (6, T)
    feat_contrast = contrast.mean(axis=1)  # (6,)

    # --- MFCC 1–4 (4 dims) ---
    mfcc = librosa.feature.mfcc(y=mix, sr=sr, n_mfcc=4, hop_length=hop_length)  # (4, T)
    feat_mfcc = mfcc.mean(axis=1)  # (4,)

    # --- Onset strength (1 dim) ---
    onset_env   = librosa.onset.onset_strength(y=mix, sr=sr, hop_length=hop_length)
    feat_onset  = np.array([onset_env.mean()], dtype=np.float32)

    # --- RMS (1 dim) ---
    rms        = librosa.feature.rms(y=mix, hop_length=hop_length)
    feat_rms   = np.array([rms.mean()], dtype=np.float32)

    # --- ZCR (1 dim) ---
    zcr        = librosa.feature.zero_crossing_rate(y=mix, hop_length=hop_length)
    feat_zcr   = np.array([zcr.mean()], dtype=np.float32)

    return np.concatenate([
        feat_chroma,    # 12
        feat_contrast,  # 6
        feat_mfcc,      # 4
        feat_onset,     # 1
        feat_rms,       # 1
        feat_zcr,       # 1
    ]).astype(np.float32)  # (25,)