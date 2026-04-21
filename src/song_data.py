"""
SongData - the contract between data loading and training.

The data loading layer produces SongData objects. Everything downstream —
feature extraction, VQ-VAE training, transformer training — consumes this
interface.

A SongData represents ONE song with its stems already segmented into bars.
Each stem is a list of bar-length audio clips (numpy arrays).
No melody/harmony classification — all stems are treated uniformly.
"""

from dataclasses import dataclass, field
from typing import List
import numpy as np


@dataclass
class SongData:
    """
    One song's worth of bar-segmented audio stems.

    Attributes:
        song_id:  Unique identifier (e.g. HuggingFace dataset row id or filename)
        stems:    List of stems. stems[stem_idx][bar_idx] → np.ndarray (audio samples)
        sr:       Sample rate shared across all clips
    """

    song_id: str
    stems:   List[List[np.ndarray]] = field(default_factory=list)
    sr:      int = 22050

    @property
    def num_bars(self) -> int:
        """Minimum bar count across all stems (shortest stem sets the length)."""
        if not self.stems:
            return 0
        return min(len(stem) for stem in self.stems)

    @property
    def num_stems(self) -> int:
        return len(self.stems)

    def is_valid(self) -> bool:
        """At least one stem and at least two bars."""
        return self.num_stems >= 1 and self.num_bars >= 2

    def clips_at(self, bar_idx: int) -> List[np.ndarray]:
        """All stem clips at a given bar position."""
        return [
            stem[bar_idx]
            for stem in self.stems
            if bar_idx < len(stem)
        ]
