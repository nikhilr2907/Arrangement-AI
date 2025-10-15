import librosa

def extract_stem_feature(audio_clip):
    """
    Extract musical features from stems themselves
    """
    tempo = librosa.beat.tempo(y=audio_clip, sr=22050)[0]
    chroma_stft = librosa.feature.chroma_stft(y=audio_clip, sr=22050).mean()
    rmse = librosa.feature.rms(y=audio_clip).mean()
    spectral = librosa.feature.spectral_centroid(y=audio_clip, sr=22050).mean()
    spectral_bw = librosa.feature.spectral_bandwidth(y=audio_clip, sr=22050).mean()
    rolloff = librosa.feature.spectral_rolloff(y=audio_clip, sr=22050).mean()
    onset_strength = librosa.onset.onset_strength(y=audio_clip, sr=22050).mean()
    zcr = librosa.feature.zero_crossing_rate(y=audio_clip).mean()

    return {
        "tempo": tempo,
        "chroma_stft": chroma_stft,
        "rmse": rmse,
        "spectral_centroid": spectral,
        "spectral_bandwidth": spectral_bw,
        "rolloff": rolloff,
        "onset_strength": onset_strength,
        "zcr": zcr,
        "key signature": librosa.key_signature(audio_clip)

    }
