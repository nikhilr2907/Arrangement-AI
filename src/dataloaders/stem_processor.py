"""
StemProcessor — core in-memory stem processing pipeline.

Takes raw audio stems in memory and produces a SongData object.
No file I/O, no temporary files. No melody/harmony classification —
all stems are treated uniformly.

Stages:
  1. Beat detection and bar extraction (using BPM if available)
  2. SongData assembly
"""

from typing import Dict, Optional

import numpy as np

from src.audio_training_breakdown_automation.audio_breakdown import (
    generate_beats_from_tempo,
    detect_beats,
    extract_bars,
)
from src.song_data import SongData


class StemProcessor:
    """
    Process raw audio stems into SongData.

    Usage:
        processor = StemProcessor()
        song_data = processor.process(
            stems  = {"guitar": audio_array, "bass": audio_array, ...},
            song_id = "song_001",
            sr      = 22050,
            bpm     = 120.0,
        )
    """

    def __init__(self, beats_per_bar: int = 4):
        self.beats_per_bar = beats_per_bar

    def process(
        self,
        stems:   Dict[str, np.ndarray],
        song_id: str,
        sr:      int,
        bpm:     float,
    ) -> SongData:
        """
        Process stems into SongData.

        Args:
            stems:   {stem_name: audio_array (mono or stereo)}
            song_id: unique song identifier
            sr:      sample rate
            bpm:     tempo in beats per minute (pass 0 or None to use librosa detection)

        Returns:
            SongData object ready for training/inference
        """
        # Normalise to mono if needed
        stems_mono = {}
        for name, audio in stems.items():
            if audio.ndim > 1:
                audio = np.mean(audio, axis=0)
            stems_mono[name] = audio.astype(np.float32)

        # Beat detection
        beat_frames = self._extract_beat_frames(stems_mono, sr, bpm)

        # Bar extraction per stem
        all_stems = []
        for name, audio in stems_mono.items():
            bars = extract_bars(audio, beat_frames, self.beats_per_bar)
            if bars:
                all_stems.append(bars)

        if not all_stems:
            raise ValueError(f"No bars extracted from stems for {song_id}")

        return SongData(
            song_id = song_id,
            stems   = all_stems,
            sr      = sr,
        )

    def _extract_beat_frames(
        self, stems: Dict[str, np.ndarray], sr: int, bpm: float
    ) -> np.ndarray:
        if bpm is not None and bpm > 0:
            first_stem = next(iter(stems.values()))
            return generate_beats_from_tempo(first_stem, sr, bpm)
        else:
            first_stem = next(iter(stems.values()))
            _, beat_frames = detect_beats(first_stem, sr)
            return beat_frames
