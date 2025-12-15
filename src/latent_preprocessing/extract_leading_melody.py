from itertools import combinations
import librosa
import numpy as np
from numpy.typing import NDArray


MELODY_WEIGHTS = {
    'pitch_clarity': 0.35,
    'harmonic_ratio': 0.30,
    'melodic_complexity': 0.20,
    'pitch_presence': 0.15,
}


def find_leading_melody_feature(audio_clip: NDArray[np.float32], sr: int = 22050) -> float:
    """Compute composite melody strength score from audio."""
    y_harmonic, _ = librosa.effects.hpss(audio_clip)
    pitches, magnitudes = librosa.piptrack(y=y_harmonic, sr=sr)

    # Extract leading pitch per time frame
    leading_pitches = pitches[magnitudes.argmax(axis=0), np.arange(pitches.shape[1])]
    pitch_magnitudes = magnitudes.max(axis=0)

    # Pitch clarity: average magnitude of detected pitches
    pitch_clarity = (
        np.mean(pitch_magnitudes[pitch_magnitudes > 0])
        if np.any(pitch_magnitudes > 0)
        else 0.0
    )

    # Melodic complexity: variation in pitch movement
    non_zero_pitches = leading_pitches[leading_pitches > 0]
    melodic_complexity = (
        np.std(np.diff(non_zero_pitches))
        if len(non_zero_pitches) > 1
        else 0.0
    )

    # Harmonic ratio: harmonic energy vs total energy
    harmonic_energy = np.mean(librosa.feature.rms(y=y_harmonic))
    total_energy = np.mean(librosa.feature.rms(y=audio_clip))
    harmonic_ratio = harmonic_energy / (total_energy + 1e-10)

    # Normalize features
    pitch_clarity_norm = np.clip(pitch_clarity / 0.5, 0, 1)
    harmonic_ratio_norm = np.clip(harmonic_ratio, 0, 1)
    melodic_complexity_norm = np.clip(melodic_complexity / 100, 0, 1)
    pitch_presence = np.mean(leading_pitches > 0)

    # Weighted combination
    return sum([
        MELODY_WEIGHTS['pitch_clarity'] * pitch_clarity_norm,
        MELODY_WEIGHTS['harmonic_ratio'] * harmonic_ratio_norm,
        MELODY_WEIGHTS['melodic_complexity'] * melodic_complexity_norm,
        MELODY_WEIGHTS['pitch_presence'] * pitch_presence,
    ])


def compute_group_compatibility(
    waveforms: list[NDArray[np.float32]], sr: int = 22050
) -> float:
    """
    Check harmonic compatibility between multiple waveforms.
    """
    if len(waveforms) == 1:
        return 1.0

    # Compute chromagrams for each waveform
    chromas = [librosa.feature.chroma_cqt(y=y, sr=sr) for y in waveforms]

    # Calculate pairwise correlations
    correlations = [
        np.corrcoef(chromas[i].flatten(), chromas[j].flatten())[0, 1]
        for i in range(len(chromas))
        for j in range(i + 1, len(chromas))
    ]

    return np.mean(correlations) if correlations else 0.0


AudioClip = tuple[NDArray[np.float32], int]


def is_leading_melody(
    audio_clips: dict[str, AudioClip],
    max_group_size: int = 3,
    compatibility_threshold: float = 0.3,
) -> dict[str, AudioClip]:
    """
    Return the leading melody group (1-3 clips) with filename tracking.

    Args:
        audio_clips: dictionary mapping filename to (audio_array, sample_rate) tuples
        max_group_size: Maximum number of clips in a group
        compatibility_threshold: Minimum compatibility score for grouping

    Returns:
        dictionary mapping filename to (audio_array, sample_rate) for leading melody group
    """
    if len(audio_clips) == 0:
        return {}

    if len(audio_clips) == 1:
        return audio_clips

    # Score each clip individually
    clip_scores = [
        (filename, clip, find_leading_melody_feature(clip[0], clip[1]))
        for filename, clip in audio_clips.items()
    ]

    # Sort by melody strength (descending)
    clip_scores.sort(key=lambda x: x[2], reverse=True)

    # Start with the strongest clip as fallback
    best_group = {clip_scores[0][0]: clip_scores[0][1]}
    best_group_score = clip_scores[0][2]

    # Test all combinations of top clips
    num_candidates = min(len(clip_scores), max_group_size * 2)
    top_clips = clip_scores[:num_candidates]

    for group_size in range(2, min(max_group_size + 1, len(top_clips) + 1)):
        for combo in combinations(range(len(top_clips)), group_size):
            candidate_group = {top_clips[i][0]: top_clips[i][1] for i in combo}

            # Extract waveforms and sample rate
            waveforms = [clip[0] for clip in candidate_group.values()]
            sr = list(candidate_group.values())[0][1]

            # Check compatibility
            compatibility = compute_group_compatibility(waveforms, sr)

            # If compatible, compute group score with bonus
            if compatibility >= compatibility_threshold:
                individual_scores = [
                    find_leading_melody_feature(clip[0], clip[1])
                    for clip in candidate_group.values()
                ]
                group_score = np.mean(individual_scores) * (1 + compatibility * 0.5)

                # Update best group if this is better
                if group_score > best_group_score:
                    best_group = candidate_group
                    best_group_score = group_score

    return best_group

