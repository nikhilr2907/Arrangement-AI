"""
ManifestLoader — load stems from a local metadata DataFrame.

The manifest is a CSV/parquet with columns:
  song_id:   str      (unique song identifier)
  stem_path: str      (path to stem file: HF path, DVC path, or local path)
  stem_type: str      ('melody' or 'harmony')
  bpm:       float    (tempo in BPM)
  sr:        int      (sample rate)
  stem_name: str      (optional: name of stem; derived from stem_path if missing)

The loader groups rows by song_id, downloads stems if needed, and builds SongData objects.

Download logic is pluggable — override the download_fn to use HF, DVC, or local paths.
"""

from pathlib import Path
from typing import Callable, List, Optional, Dict

import soundfile as sf
import numpy as np
import pandas as pd

from src.song_data import SongData
from src.dataloaders.stem_processor import StemProcessor


class ManifestLoader:
    """
    Load stems from a metadata DataFrame and build SongData objects.

    Usage:
        loader = ManifestLoader("manifest.csv", download_fn=my_download_func)
        songs = loader.load()  # List[SongData]
    """

    def __init__(
        self,
        manifest_path: str,
        download_fn: Optional[Callable[[str], np.ndarray]] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Args:
            manifest_path: Path to CSV/parquet manifest file
            download_fn:   Function to download audio from a stem_path.
                           Signature: (stem_path: str) -> audio_array: np.ndarray
                           If None, assumes all stem_paths are local file paths.
            cache_dir:     Optional directory to cache downloaded audio (avoids re-downloads)
        """
        self.manifest_path = Path(manifest_path)
        self.download_fn = download_fn or self._load_local_audio
        self.cache_dir = Path(cache_dir) if cache_dir else None

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load manifest
        if str(self.manifest_path).endswith(".parquet"):
            self.df = pd.read_parquet(manifest_path)
        else:
            self.df = pd.read_csv(manifest_path)

        self._validate_manifest()

    def _validate_manifest(self) -> None:
        """Check that manifest has required columns."""
        required = {"song_id", "stem_path", "stem_type", "bpm", "sr"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Manifest missing columns: {missing}")

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> List[SongData]:
        """
        Load all songs from the manifest.

        Returns:
            List of SongData objects, one per unique song_id
        """
        processor = StemProcessor()
        songs = []

        for song_id, group in self.df.groupby("song_id"):
            try:
                song_data = self._process_song(processor, song_id, group)
                songs.append(song_data)
            except Exception as e:
                print(f"  Warning: Failed to load {song_id}: {e}")
                continue

        print(f"Loaded {len(songs)} songs from manifest")
        return songs

    def load_single(self, song_id: str) -> SongData:
        """Load a specific song by ID."""
        group = self.df[self.df["song_id"] == song_id]
        if group.empty:
            raise ValueError(f"Song {song_id} not found in manifest")

        processor = StemProcessor()
        return self._process_song(processor, song_id, group)

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------

    def _process_song(
        self, processor: StemProcessor, song_id: str, group: pd.DataFrame
    ) -> SongData:
        """
        Process one song's rows from the manifest into a SongData.

        Args:
            processor: StemProcessor instance
            song_id:   unique song identifier
            group:     DataFrame rows for this song

        Returns:
            SongData object
        """
        # Load each stem
        stems: Dict[str, np.ndarray] = {}
        stem_types: Dict[str, str] = {}

        for _, row in group.iterrows():
            stem_path = row["stem_path"]
            stem_type = row["stem_type"]
            stem_name = row.get("stem_name") or Path(stem_path).stem

            # Load audio (with optional caching)
            audio = self._load_audio_cached(stem_path)
            stems[stem_name] = audio
            stem_types[stem_name] = stem_type

        # Extract BPM and SR (should be consistent across rows for a song)
        bpm = group["bpm"].iloc[0]
        sr = group["sr"].iloc[0]

        # Process
        song_data = processor.process(
            stems=stems,
            song_id=song_id,
            sr=sr,
            bpm=bpm,
            stem_types=stem_types,
        )

        return song_data

    def _load_audio_cached(self, stem_path: str) -> np.ndarray:
        """Load audio from path, with optional caching."""
        if self.cache_dir:
            cache_file = self.cache_dir / f"{stem_path.replace('/', '_')}.npy"
            if cache_file.exists():
                return np.load(cache_file)

        # Download or load
        audio = self.download_fn(stem_path)

        # Cache
        if self.cache_dir:
            np.save(cache_file, audio)

        return audio

    # ------------------------------------------------------------------
    # Default download function (local file loading)
    # ------------------------------------------------------------------

    @staticmethod
    def _load_local_audio(stem_path: str) -> np.ndarray:
        """
        Load audio from a local file path.

        Supports .wav, .mp3, .flac, etc. (via soundfile/librosa).
        Returns mono audio as float32.
        """
        try:
            audio, sr = sf.read(stem_path, dtype="float32")
        except Exception:
            # Fallback to librosa if soundfile fails
            import librosa

            audio, sr = librosa.load(stem_path, sr=None, mono=True)
            audio = audio.astype(np.float32)

        return audio

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def songs_in_manifest(self) -> List[str]:
        """List all unique song_ids in the manifest."""
        return sorted(self.df["song_id"].unique().tolist())

    def manifest_stats(self) -> dict:
        """Get summary stats about the manifest."""
        return {
            "num_rows": len(self.df),
            "num_songs": self.df["song_id"].nunique(),
            "num_stems": len(self.df),
            "stem_types": self.df["stem_type"].value_counts().to_dict(),
            "bpm_range": (self.df["bpm"].min(), self.df["bpm"].max()),
            "sample_rates": sorted(self.df["sr"].unique().tolist()),
        }
