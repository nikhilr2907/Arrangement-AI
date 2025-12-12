import librosa


def extract_stem_feature(audio_clip, sr=22050):
    """Extract musical features from audio clip."""
    return {
        "tempo":              librosa.beat.tempo(y=audio_clip, sr=sr)[0],
        "chroma_stft":        librosa.feature.chroma_stft(y=audio_clip, sr=sr).mean(),
        "chroma_cqt":         librosa.feature.chroma_cqt(y=audio_clip, sr=sr).mean(),
        "chroma_cens":        librosa.feature.chroma_cens(y=audio_clip, sr=sr).mean(),
        "rmse":               librosa.feature.rms(y=audio_clip).mean(),
        "spectral_centroid":  librosa.feature.spectral_centroid(y=audio_clip, sr=sr).mean(),
        "spectral_bandwidth": librosa.feature.spectral_bandwidth(y=audio_clip, sr=sr).mean(),
        "rolloff":            librosa.feature.spectral_rolloff(y=audio_clip, sr=sr).mean(),
        "onset_strength":     librosa.onset.onset_strength(y=audio_clip, sr=sr).mean(),
        "zcr":                librosa.feature.zero_crossing_rate(y=audio_clip).mean(),
        "key_signature":      librosa.key_signature(audio_clip),
    }
