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

    Dynamically discovers all stem columns. Expected parquet structure:
      Required columns:
        - song_id: str
        - sr: int (sample rate)
        - bpm: float (tempo in BPM)

      Audio stem columns (any number, named flexibly):
        - *_audio: bytes (float32 audio data)
          Examples: melody_audio, harmony_1_audio, drums_audio, bass_audio, etc.

      Optional type columns (explicit stem classification):
        - *_type: str (melody/harmony/drums)
          Examples: melody_type, harmony_1_type, drums_type, etc.

    If *_type columns are not provided, stem types are inferred from names:
      - "melody", "vocal", "lead", "voice" → melody
      - "drum", "percussion", "kick", "snare" → drums
      - "harmony", "bass", "synth", "pad", "guitar" → harmony

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

        Dynamically discovers all *_audio columns and loads them as stems.
        Stem types are either read from *_type columns or inferred from names.

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
                # Dynamically discover all *_audio columns (not hardcoded)
                stems = {}
                stem_types = {}

                for key in row.keys():
                    if key.endswith("_audio"):
                        stem_name = key.replace("_audio", "")

                        # Convert bytes to numpy array
                        audio_bytes = row[key]
                        if audio_bytes is None:
                            continue

                        stems[stem_name] = np.frombuffer(
                            audio_bytes, dtype=np.float32
                        ).copy()

                        # Try to get explicit type from *_type column
                        type_key = f"{stem_name}_type"
                        if type_key in row and row[type_key] is not None:
                            stem_types[stem_name] = row[type_key]
                        else:
                            # Fall back to inferring from stem name
                            stem_types[stem_name] = self._infer_stem_type(stem_name)

                if not stems:
                    print(f"  [WARNING] {row.get('song_id', 'unknown')}: No audio stems found")
                    continue

                song_data = self.processor.process(
                    stems=stems,
                    song_id=row["song_id"],
                    sr=row["sr"],
                    bpm=row["bpm"],
                    stem_types=stem_types if stem_types else None
                )

                if song_data.is_valid():
                    yield song_data
            except Exception as e:
                print(f"  [WARNING] Error processing {row.get('song_id', 'unknown')}: {e}")
                continue

    def _infer_stem_type(self, stem_name: str) -> str:
        """
        Infer stem type from stem name if not provided explicitly.

        Args:
            stem_name: Name of the stem (e.g., "melody", "harmony_1", "drums")

        Returns:
            "melody", "harmony", or "drums" based on naming conventions
        """
        name_lower = stem_name.lower()

        # Melody detection
        if any(x in name_lower for x in ["melody", "vocal", "lead", "main", "voice", "singer"]):
            return "melody"

        # Drums/percussion detection
        if any(x in name_lower for x in ["drum", "kick", "percussion", "beat", "snare", "cymbal"]):
            return "drums"

        # Harmony detection (includes bass, synth, guitar, strings, etc.)
        if any(x in name_lower for x in ["harmony", "bass", "synth", "pad", "chord", "guitar", "strings", "instrumental"]):
            return "harmony"

        # Default to harmony for anything else (safe fallback)
        return "harmony"
