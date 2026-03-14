"""
Audio stem segmentation into bars.

Core functions for in-memory audio processing:
  - Beat detection and frame extraction
  - Bar segmentation from beat frames
  - Stem classification (melody vs harmony)
"""

from typing import Dict, List, Tuple

import librosa
import numpy as np

from src.latent_preprocessing.melodic_candidates import find_melodic_candidates


# ---------------------------------------------------------------------------
# Beat and bar extraction (work with numpy arrays, no file I/O)
# ---------------------------------------------------------------------------

def generate_beats_from_tempo(
    audio_array: np.ndarray,
    sr: int,
    tempo: float,
) -> np.ndarray:
    """
    Generate beat frames directly from known BPM.

    Args:
        audio_array: audio samples
        sr:          sample rate
        tempo:       BPM

    Returns:
        beat_frames: indices of beat frames
    """
    audio_duration = len(audio_array) / sr
    beat_duration  = 60.0 / tempo
    beat_times     = np.arange(0, audio_duration, beat_duration)
    return librosa.time_to_frames(beat_times, sr=sr, hop_length=512)


def detect_beats(
    audio_array: np.ndarray,
    sr: int,
    tempo_hint: float = None,
) -> Tuple[float, np.ndarray]:
    """
    Detect beats using onset envelope (librosa).

    Args:
        audio_array: audio samples
        sr:          sample rate
        tempo_hint:  optional BPM hint for beat_track

    Returns:
        (detected_tempo, beat_frames)
    """
    onset_env = librosa.onset.onset_strength(y=audio_array, sr=sr)

    kwargs = {
        "onset_envelope": onset_env,
        "sr": sr,
        "trim": False,
        "units": "frames",
    }

    if tempo_hint is not None:
        kwargs["start_bpm"] = tempo_hint

    return librosa.beat.beat_track(**kwargs)


def extract_bars(
    audio_array: np.ndarray,
    beat_frames: np.ndarray,
    beats_per_bar: int = 4,
) -> List[np.ndarray]:
    """
    Extract bar-length audio segments from beat frames.

    Args:
        audio_array:  audio samples
        beat_frames:  frame indices of beats
        beats_per_bar: beats per measure (typically 4)

    Returns:
        list of bar audio arrays
    """
    bar_indices = beat_frames[::beats_per_bar]
    bars = []

    for i in range(len(bar_indices) - 1):
        start_sample = librosa.frames_to_samples(bar_indices[i])
        end_sample = librosa.frames_to_samples(bar_indices[i + 1])
        bars.append(audio_array[start_sample:end_sample])

    return bars


# ---------------------------------------------------------------------------
# Stem classification
# ---------------------------------------------------------------------------

def classify_stems(
    stem_bars: Dict[str, List[np.ndarray]],
    stem_types: Dict[str, str] = None,
    activity_threshold: float = 0.5,
) -> Tuple[Dict[str, List[np.ndarray]], Dict[str, List[np.ndarray]]]:
    """
    Separate stems into melody and harmony groups.

    Args:
        stem_bars:          {stem_name: [bar_1, bar_2, ...]}
        stem_types:         optional {stem_name: 'melody' or 'harmony'} from metadata
        activity_threshold: fallback threshold for auto-classification (0-1)

    Returns:
        (melody_stems, harmony_stems) dicts with same structure as stem_bars
    """
    melody_stems = {}
    harmony_stems = {}

    # If stem types are provided in metadata, use those
    if stem_types:
        for stem_name, bars in stem_bars.items():
            if stem_types.get(stem_name) == "melody":
                melody_stems[stem_name] = bars
            else:
                harmony_stems[stem_name] = bars
    else:
        # Fallback: auto-classify by activity
        active_stems = find_melodic_candidates(
            {k: np.array(v, dtype=object) for k, v in stem_bars.items()},
            activity_threshold=activity_threshold,
        )
        for stem_name, bars in stem_bars.items():
            if stem_name in active_stems:
                melody_stems[stem_name] = bars
            else:
                harmony_stems[stem_name] = bars

    return melody_stems, harmony_stems