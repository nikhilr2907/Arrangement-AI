import librosa
import numpy as np


def extract_feature_matrix(
    main_melody: np.ndarray = None,
    harmonic_clips: np.ndarray = None,
    sr: int = 22050
) -> np.ndarray:
    """
    Extracts the combined, time-series feature matrix for arrangement analysis.

    The output matrix F has the shape (N_features x N_frames), where each column
    F[:, t] is the synchronized feature vector for time step t (25 features:
    12 Melody Chroma + 12 Harmony Chroma + 1 Rhythmic Density).

    Args:
        main_melody: Array of audio arrays for leading melody clips (optional)
                    If None or empty, melody chroma features will be zeros
        harmonic_clips: Array of audio arrays for harmony parts (optional)
                       If None or empty, harmony chroma features will be zeros
        sr: Sample rate (default 22050)

    Returns:
        A 2D numpy array (Feature Matrix) where columns are time steps (frames).
        Always returns shape (25, n_frames).

    Note:
        At least one of main_melody or harmonic_clips must be provided.
    """

    # Validate that at least one input is provided
    has_melody = main_melody is not None and len(main_melody) > 0
    has_harmony = harmonic_clips is not None and len(harmonic_clips) > 0

    if not has_melody and not has_harmony:
        raise ValueError("At least one of main_melody or harmonic_clips must be provided.")
    print("shape of main melody:", main_melody.shape)
    print("shape of harmonic clips:", harmonic_clips.shape)
 

    n_fft = 2048
    hop_length = 512

    # --- 1. Calculate Melody Chroma (12 dim) ---
    if has_melody:
        # Has melody - compute normally
        melody_chroma_matrices = []
        melody_signals = []

        for y_melody in main_melody:
            y_melody = np.array(y_melody, dtype=np.float32)
            y_harm_mel = librosa.effects.hpss(y_melody, margin=3.0, kernel_size=n_fft)[0]
            chroma_mel = librosa.feature.chroma_cens(
                y=y_harm_mel, sr=sr, hop_length=hop_length
            )
            melody_chroma_matrices.append(chroma_mel)
            melody_signals.append(y_melody)
    else:
        # No melody - will use zeros later
        melody_chroma_matrices = []
        melody_signals = []

    # --- 2. Calculate and Average Harmony Chroma (12 dim) ---
    if has_harmony:
        # Has harmony - compute normally
        harmony_chroma_matrices = []
        harmony_signals = []

        for y_clip in harmonic_clips:
            y_clip = np.array(y_clip, dtype=np.float32)
            y_harm = librosa.effects.hpss(y_clip, margin=3.0, kernel_size=n_fft)[0]
            chroma_matrix = librosa.feature.chroma_cens(
                y=y_harm, sr=sr, hop_length=hop_length
            )
            harmony_chroma_matrices.append(chroma_matrix)
            harmony_signals.append(y_clip)
    else:
        # No harmony - will use zeros later
        harmony_chroma_matrices = []
        harmony_signals = []

    # --- 3. Determine dimensions and compute averaged features ---
    # Determine minimum number of frames across all available clips
    all_chroma_matrices = melody_chroma_matrices + harmony_chroma_matrices
    min_frames = min(m.shape[1] for m in all_chroma_matrices)

    # Compute melody chroma matrix
    if has_melody:
        # Average melody chroma matrices (if multiple leading melodies)
        chroma_mel_matrix = np.mean(
            [m[:, :min_frames] for m in melody_chroma_matrices], axis=0
        )
    else:
        # No melody - use zeros for melody chroma
        chroma_mel_matrix = np.zeros((12, min_frames))

    # Compute harmony chroma matrix
    if has_harmony:
        # Average harmony chroma matrices
        chroma_harm_matrix = np.mean(
            [m[:, :min_frames] for m in harmony_chroma_matrices], axis=0
        )
    else:
        # No harmony - use zeros for harmony chroma
        chroma_harm_matrix = np.zeros((12, min_frames))

    # --- 4. Prepare signals for rhythm calculation ---
    # Mix all available signals
    all_signals = melody_signals + harmony_signals
    min_signal_length = min(sig.shape[0] for sig in all_signals)
    mixed_signals = [sig[:min_signal_length] for sig in all_signals]

    # --- 5. Calculate Rhythmic Density (1 dim) ---
    # Create unified mixed signal from available clips
    y_mix = np.sum(mixed_signals, axis=0)

    # Calculate Onset Strength Function
    o_env = librosa.onset.onset_strength(y=y_mix, sr=sr, hop_length=hop_length)

    # Truncate and reshape density vector for concatenation
    rhythm_density_vector = o_env[:min_frames]
    rhythm_density_matrix = rhythm_density_vector[np.newaxis, :]

    # --- 6. Concatenate Features ---
    feature_matrix = np.vstack(
        [chroma_mel_matrix, chroma_harm_matrix, rhythm_density_matrix]
    )

    return feature_matrix

