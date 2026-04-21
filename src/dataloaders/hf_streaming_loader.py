"""
HuggingFace Streaming Loader - stream SongData from parquet datasets.

Loads one song at a time from HF Hub without downloading entire dataset.
No melody/harmony classification — all stems are treated uniformly.
"""

from typing import Iterator
import numpy as np
from datasets import load_dataset

from src.dataloaders.stem_processor import StemProcessor
from src.song_data import SongData


class HFStreamingLoader:
    """
    Stream SongData objects from HuggingFace parquet dataset.

    Dynamically discovers all stem columns. Expected parquet structure:
      Required columns:
        - song_id: str
        - sr:      int  (sample rate)
        - bpm:     float (tempo in BPM; use 0 to trigger librosa detection)

      Audio stem columns (any number, named flexibly):
        - *_audio: bytes (float32 audio data)
          Examples: guitar_audio, bass_audio, drums_audio, synth_audio, etc.

    Usage:
        loader = HFStreamingLoader("username/dataset_name", split="train")
        for song in loader.stream():
            train_on_song(song)
    """

    def __init__(self, repo_id: str, split: str = "train"):
        """
        Args:
            repo_id: HuggingFace dataset repo (e.g., "username/music_stems")
            split:   Dataset split to load (e.g., "train", "test")
        """
        self.repo_id   = repo_id
        self.split     = split
        self.processor = StemProcessor()

    def stream(self) -> Iterator[SongData]:
        """
        Stream SongData objects one at a time from HF dataset.

        Dynamically discovers all *_audio columns and loads them as stems.

        Yields:
            SongData: Processed song with bars extracted
        """
        dataset = load_dataset(
            self.repo_id,
            split=self.split,
            streaming=True,
        )

        for row in dataset:
            try:
                stems = {}

                for key in row.keys():
                    if not key.endswith("_audio"):
                        continue

                    audio_bytes = row[key]
                    if audio_bytes is None:
                        continue

                    stem_name = key.replace("_audio", "")
                    stems[stem_name] = np.frombuffer(
                        audio_bytes, dtype=np.float32
                    ).copy()

                if not stems:
                    print(f"  [WARNING] {row.get('song_id', 'unknown')}: no audio stems found")
                    continue

                song_data = self.processor.process(
                    stems   = stems,
                    song_id = row["song_id"],
                    sr      = row["sr"],
                    bpm     = row["bpm"],
                )

                if song_data.is_valid():
                    yield song_data

            except Exception as e:
                print(f"  [WARNING] {row.get('song_id', 'unknown')}: {e}")
                continue