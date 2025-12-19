"""Audio breakdown and processing pipeline for musical stems."""

from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import numpy as np

from src.latent_preprocessing.melodic_candidates import find_melodic_candidates
from src.model_preprocessing.feature_vector_segmentation import extract_feature_matrix
from src.model_preprocessing.chunking_transformation import chunking_transformation


def process_audio_stems(
    stems: List[str],
    tempo_hint: float = None,
    use_manual_tempo: bool = True
) -> Dict[str, np.ndarray]:
    """
    Process audio stems into bars.

    Args:
        stems: List of file paths to audio stems
        tempo_hint: Optional tempo hint in BPM for beat detection
        use_manual_tempo: If True and tempo_hint provided, use manual tempo instead of detection

    Returns:
        Dictionary mapping stem filename to array of bar audio segments
    """
    loaded_audio = {Path(stem).name: librosa.load(stem) for stem in stems}
    bars_dict = {}

    for stem_filename, (audio_array, sr) in loaded_audio.items():
        filename = Path(stem_filename).stem

        # Detect or calculate beats
        if use_manual_tempo and tempo_hint is not None:
            beat_frames = _generate_beats_from_tempo(audio_array, sr, tempo_hint)
            print(f"{filename}: Using manual tempo = {tempo_hint:.1f} BPM, {len(beat_frames)} beats")
        else:
            tempo, beat_frames = _detect_beats(audio_array, sr, tempo_hint)
            print(f"{filename}: Detected tempo = {tempo:.1f} BPM, {len(beat_frames)} beats")

        # Convert beats to bars (4/4 time signature)
        bars = _extract_bars(audio_array, beat_frames, beats_per_bar=4)

        if bars:
            bars_dict[filename] = np.array(bars, dtype=object)

    return bars_dict


def _generate_beats_from_tempo(audio_array: np.ndarray, sr: int, tempo: float) -> np.ndarray:
    """Generate beat frames directly from known BPM."""
    audio_duration = len(audio_array) / sr
    beat_duration = 60.0 / tempo
    beat_times = np.arange(0, audio_duration, beat_duration)
    return librosa.time_to_frames(beat_times, sr=sr, hop_length=512)


def _detect_beats(audio_array: np.ndarray, sr: int, tempo_hint: float = None) -> Tuple[float, np.ndarray]:
    """Detect beats using onset envelope."""
    onset_env = librosa.onset.onset_strength(y=audio_array, sr=sr)

    kwargs = {
        'onset_envelope': onset_env,
        'sr': sr,
        'trim': False,
        'units': 'frames'
    }

    if tempo_hint is not None:
        kwargs['start_bpm'] = tempo_hint

    return librosa.beat.beat_track(**kwargs)


def _extract_bars(
    audio_array: np.ndarray,
    beat_frames: np.ndarray,
    beats_per_bar: int = 4
) -> List[np.ndarray]:
    """Extract bar-length audio segments from beat frames."""
    bar_indices = beat_frames[::beats_per_bar]
    bars = []

    for i in range(len(bar_indices) - 1):
        start_sample = librosa.frames_to_samples(bar_indices[i])
        end_sample = librosa.frames_to_samples(bar_indices[i + 1])
        bars.append(audio_array[start_sample:end_sample])

    return bars


def sort_structure(bars: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Sort stems into melodic and harmonic groups.

    Args:
        bars: Dictionary mapping filename to array of bars

    Returns:
        Tuple of (melodic_stems, harmonic_stems) dictionaries
    """
    melodic_stems = find_melodic_candidates(bars, activity_threshold=0.5)
    harmonic_stems = {k: v for k, v in bars.items() if k not in melodic_stems}
    return melodic_stems, harmonic_stems


def _build_pattern_structure(bars: Dict[str, np.ndarray]) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Organize bars into time-step structure.

    Args:
        bars: Dictionary mapping filename to array of bars

    Returns:
        Dictionary mapping time step to stem audio for that bar
    """
    if not bars:
        return {}

    max_bars = max(len(bar_array) for bar_array in bars.values())
    structure = {}

    for time_step in range(max_bars):
        time_step_dict = {}
        for stem_name, bar_array in bars.items():
            if time_step < len(bar_array) and np.sum(bar_array[time_step]) > 0:
                time_step_dict[stem_name] = bar_array[time_step]
        structure[time_step] = time_step_dict

    return structure


def process_melodic_harmony_groups(
    melodic_stems: Dict[str, np.ndarray],
    harmonic_stems: Dict[str, np.ndarray],
    bars: Dict[str, np.ndarray]
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    Process melodic and harmonic groups into time-step vectors.

    Args:
        melodic_stems: Dictionary of melodic stem bars
        harmonic_stems: Dictionary of harmonic stem bars
        bars: Full dictionary of all bars

    Returns:
        Dictionary mapping time step to (melodic_clips, harmonic_clips) tuple
    """
    structure = _build_pattern_structure(bars)
    overall_vectors = {}

    for time_step, stem_dict in structure.items():
        melodic_clips = []
        harmonic_clips = []

        for stem_name, bar_audio in stem_dict.items():
            if stem_name in melodic_stems:
                melodic_clips.append(bar_audio)
            elif stem_name in harmonic_stems:
                harmonic_clips.append(bar_audio)

        overall_vectors[time_step] = (
            np.array(melodic_clips, dtype=object),
            np.array(harmonic_clips, dtype=object)
        )

    return overall_vectors


def convert_to_feature_matrices(overall_vectors: Dict[int, Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """
    Convert time-step vectors into feature matrices.

    Args:
        overall_vectors: Dictionary mapping time step to (melodic, harmonic) clip tuples

    Returns:
        Array of feature matrices
    """
    feature_matrices = []

    for melodic_clips, harmonic_clips in overall_vectors.values():
        if len(melodic_clips) == 0:
            continue

        feature_matrix = extract_feature_matrix(harmonic_clips, melodic_clips)
        feature_matrices.append(feature_matrix)

    return np.array(feature_matrices)


def chunk_into_training_segments(feature_matrices: np.ndarray, chunk_size: int = 32, overlap: float = 0.5) -> np.ndarray:
    """
    Chunk feature matrices into training segments.

    Args:
        feature_matrices: Array of feature matrices
        chunk_size: Size of each chunk
        overlap: Overlap ratio between chunks

    Returns:
        Array of chunked training segments
    """
    return chunking_transformation(feature_matrices, chunk_size=chunk_size, overlap=overlap)


def run(
    stem_paths: List[str],
    tempo: float = None,
    use_manual_tempo: bool = True
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Run the complete audio breakdown and processing pipeline.

    Args:
        stem_paths: List of file paths to audio stems
        tempo: Optional tempo in BPM
        use_manual_tempo: Whether to use manual tempo if provided

    Returns:
        Tuple of (bars_dict, training_segments)
    """
    # Process stems into bars
    bars = process_audio_stems(stem_paths, tempo_hint=tempo, use_manual_tempo=use_manual_tempo)
    print(f"Processed {len(bars)} stems into bars.")

    # Sort into melodic and harmonic groups
    melodic_stems, harmonic_stems = sort_structure(bars)
    print(f"Identified {len(melodic_stems)} melodic stems and {len(harmonic_stems)} harmonic stems.")

    # Process groups into feature vectors
    overall_vectors = process_melodic_harmony_groups(melodic_stems, harmonic_stems, bars)
    print(f"Processed overall vectors for {len(overall_vectors)} time steps.")

    # Convert to feature matrices
    feature_matrices = convert_to_feature_matrices(overall_vectors)
    print(f"Converted to feature matrices with shape {feature_matrices.shape}.")

    # Chunk into training segments
    training_segments = chunk_into_training_segments(feature_matrices)
    print(f"Chunked into {len(training_segments)} training segments.")

    return bars, training_segments
