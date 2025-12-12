import librosa
import numpy as np


MELODY_WEIGHTS = {
    'pitch_clarity':      0.35,
    'harmonic_ratio':     0.30,
    'melodic_complexity': 0.20,
    'pitch_presence':     0.15,
}


def find_leading_melody_feature(audio_clip, sr=22050):
    """Compute composite melody strength score from audio."""
    y_harmonic, _ = librosa.effects.hpss(audio_clip)
    pitches, magnitudes = librosa.piptrack(y=y_harmonic, sr=sr)

    leading_pitches = np.array([pitches[magnitudes[:, i].argmax(), i]
                                for i in range(pitches.shape[1])])

    pitch_magnitudes = magnitudes.max(axis=0)
    pitch_clarity = np.mean(pitch_magnitudes[pitch_magnitudes > 0]) if np.any(pitch_magnitudes > 0) else 0

    non_zero_pitches = leading_pitches[leading_pitches > 0]
    melodic_complexity = np.std(np.diff(non_zero_pitches)) if len(non_zero_pitches) > 1 else 0

    harmonic_energy = np.mean(librosa.feature.rms(y=y_harmonic))
    total_energy = np.mean(librosa.feature.rms(y=audio_clip))
    harmonic_ratio = harmonic_energy / (total_energy + 1e-10)

    pitch_clarity_norm = np.clip(pitch_clarity / 0.5, 0, 1)
    harmonic_ratio_norm = np.clip(harmonic_ratio, 0, 1)
    melodic_complexity_norm = np.clip(melodic_complexity / 100, 0, 1)
    pitch_presence = np.sum(leading_pitches > 0) / len(leading_pitches)

    return (
        MELODY_WEIGHTS['pitch_clarity'] * pitch_clarity_norm +
        MELODY_WEIGHTS['harmonic_ratio'] * harmonic_ratio_norm +
        MELODY_WEIGHTS['melodic_complexity'] * melodic_complexity_norm +
        MELODY_WEIGHTS['pitch_presence'] * pitch_presence
    )


def is_leading_melody(audio_clips):
    """Return clip with strongest melody."""
    return max(audio_clips, key=find_leading_melody_feature)
