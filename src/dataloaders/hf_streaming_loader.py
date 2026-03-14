"""
HuggingFace Streaming Loader - stream SongData from parquet datasets.

Loads one song at a time from HF Hub without downloading entire dataset.
"""

from typing import Iterator
import numpy as np
from datasets import load_dataset

from src.dataloaders.stem_processor import StemProcessor
from src.song_data import SongData


class HFStreamingLoader:
    """
    Stream SongData objects from HuggingFace parquet dataset.

    Expected parquet columns:
      - song_id: str
      - melody_audio: bytes (float32 audio data)
      - harmony_audio: bytes (float32 audio data)
      - sr: int (sample rate)
      - bpm: float (tempo in BPM)

    Usage:
        loader = HFStreamingLoader("username/dataset_name", split="train")
        for song in loader.stream():
            # Process one song at a time (not loading entire dataset)
            train_on_song(song)
    """

    def __init__(self, repo_id: str, split: str = "train"):
        """
        Args:
            repo_id: HuggingFace dataset repo (e.g., "username/music_stems")
            split: Dataset split to load (e.g., "train", "test", "validation")
        """
        self.repo_id = repo_id
        self.split = split
        self.processor = StemProcessor()

    def stream(self) -> Iterator[SongData]:
        """
        Stream SongData objects one at a time from HF dataset.

        Yields:
            SongData: Processed song with bars extracted
        """
        # Load with streaming=True to avoid downloading entire dataset
        dataset = load_dataset(
            self.repo_id,
            split=self.split,
            streaming=True
        )

        for row in dataset:
            try:
                # Convert bytes back to numpy arrays
                melody_audio = np.frombuffer(row["melody_audio"], dtype=np.float32)
                harmony_audio = np.frombuffer(row["harmony_audio"], dtype=np.float32)

                stems = {
                    "melody": melody_audio,
                    "harmony": harmony_audio,
                }

                song_data = self.processor.process(
                    stems=stems,
                    song_id=row["song_id"],
                    sr=row["sr"],
                    bpm=row["bpm"],
                    stem_types={"melody": "melody", "harmony": "harmony"}
                )

                if song_data.is_valid():
                    yield song_data
            except Exception as e:
                print(f"  [WARNING] Error processing {row.get('song_id', 'unknown')}: {e}")
                continue
