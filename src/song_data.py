"""
SongData - the contract between data loading and training.

The data loading layer (HuggingFace / local, implemented separately) produces
SongData objects. Everything downstream - feature extraction, VQ-VAE training,
transformer training - consumes this interface.

A SongData represents ONE song with its stems already segmented into bars.
Each stem is a list of bar-length audio clips (numpy arrays).
Melody and harmony stems are separated at load time.
"""

from dataclasses import dataclass, field
from typing import List
import numpy as np


@dataclass
class SongData:
    """
    One song's worth of bar-segmented audio stems.

    Attributes:
        song_id:        Unique identifier (e.g. HuggingFace dataset row id or filename)
        melody_stems:   List of stems classified as melodic.
                        melody_stems[stem_idx][bar_idx] → np.ndarray (audio samples)
        harmony_stems:  List of stems classified as harmonic.
                        harmony_stems[stem_idx][bar_idx] → np.ndarray (audio samples)
        sr:             Sample rate shared across all clips
    """

    song_id:       str
    melody_stems:  List[List[np.ndarray]] = field(default_factory=list)
    harmony_stems: List[List[np.ndarray]] = field(default_factory=list)
    sr:            int = 22050

    @property
    def num_bars(self) -> int:
        """Minimum bar count across all stems (shortest stem sets the length)."""
        all_stems = self.melody_stems + self.harmony_stems
        if not all_stems:
            return 0
        return min(len(stem) for stem in all_stems)

    @property
    def num_melody_stems(self) -> int:
        return len(self.melody_stems)

    @property
    def num_harmony_stems(self) -> int:
        return len(self.harmony_stems)

    def is_valid(self) -> bool:
        """At least one melody stem and at least two bars."""
        return self.num_melody_stems >= 1 and self.num_bars >= 2

    def melody_clips_at(self, bar_idx: int) -> List[np.ndarray]:
        """All melody stem clips at a given bar position."""
        return [
            stem[bar_idx]
            for stem in self.melody_stems
            if bar_idx < len(stem)
        ]

    def harmony_clips_at(self, bar_idx: int) -> List[np.ndarray]:
        """All harmony stem clips at a given bar position."""
        return [
            stem[bar_idx]
            for stem in self.harmony_stems
            if bar_idx < len(stem)
        ]
