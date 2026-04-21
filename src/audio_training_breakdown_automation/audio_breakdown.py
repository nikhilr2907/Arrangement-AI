"""
Audio stem segmentation into bars.

Core functions for in-memory audio processing:
  - Beat detection and frame extraction
  - Bar segmentation from beat frames
"""

from typing import Dict, List, Tuple

import librosa
import numpy as np


def score_drum_likelihood(audio_array: np.ndarray, sr: int) -> float:
    """
    Score how drum-like a stem is using percussive energy ratio and onset density.

    Returns a value in [0, 1] — higher means more drum-like.
    """
    _, percussive = librosa.effects.hpss(audio_array)

    total_energy      = np.sum(audio_array ** 2) + 1e-8
    percussive_energy = np.sum(percussive ** 2)
    percussive_ratio  = percussive_energy / total_energy

    duration_s    = len(audio_array) / sr
    onsets        = librosa.onset.onset_detect(y=audio_array, sr=sr)
    onset_density = min(len(onsets) / (duration_s + 1e-8) / 10.0, 1.0)

    return 0.7 * percussive_ratio + 0.3 * onset_density


def find_drum_stem(
    stems: Dict[str, np.ndarray],
    sr: int,
) -> Tuple[str, np.ndarray]:
    """
    Identify the most drum-like stem from a dict of {stem_name: audio_array}.

    Returns (stem_name, audio_array) of the highest-scoring stem.
    """
    best_name  = None
    best_audio = None
    best_score = -1.0

    for name, audio in stems.items():
        score = score_drum_likelihood(audio, sr)
        if score > best_score:
            best_score = score
            best_name  = name
            best_audio = audio

    return best_name, best_audio


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
        audio_array:   audio samples
        beat_frames:   frame indices of beats
        beats_per_bar: beats per measure (typically 4)

    Returns:
        list of bar audio arrays
    """
    bar_indices = beat_frames[::beats_per_bar]
    bars = []

    for i in range(len(bar_indices) - 1):
        start_sample = librosa.frames_to_samples(bar_indices[i])
        end_sample   = librosa.frames_to_samples(bar_indices[i + 1])
        bars.append(audio_array[start_sample:end_sample])

    return bars