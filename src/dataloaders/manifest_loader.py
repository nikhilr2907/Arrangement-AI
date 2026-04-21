"""
ManifestLoader — load stems from a local metadata DataFrame.

The manifest is a CSV/parquet with columns:
  song_id:   str      (unique song identifier)
  stem_path: str      (path to stem file: local path, HF path, or DVC path)
  bpm:       float    (tempo in BPM; use 0 to trigger librosa beat detection)
  sr:        int      (sample rate)
  stem_name: str      (optional: name of stem; derived from stem_path if missing)

No stem_type column needed — all stems are treated uniformly.
The loader groups rows by song_id, loads stems, and builds SongData objects.
Download logic is pluggable — override download_fn to use HF, DVC, or local paths.
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
        loader = ManifestLoader("manifest.csv")
        songs  = loader.load()  # List[SongData]
    """

    def __init__(
        self,
        manifest_path: str,
        download_fn:   Optional[Callable[[str], np.ndarray]] = None,
        cache_dir:     Optional[str] = None,
    ):
        """
        Args:
            manifest_path: Path to CSV/parquet manifest file
            download_fn:   (stem_path: str) -> audio_array: np.ndarray
                           If None, assumes all stem_paths are local file paths.
            cache_dir:     Optional directory to cache downloaded audio
        """
        self.manifest_path = Path(manifest_path)
        self.download_fn   = download_fn or self._load_local_audio
        self.cache_dir     = Path(cache_dir) if cache_dir else None

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        if str(self.manifest_path).endswith(".parquet"):
            self.df = pd.read_parquet(manifest_path)
        else:
            self.df = pd.read_csv(manifest_path)

        self._validate_manifest()

    def _validate_manifest(self) -> None:
        required = {"song_id", "stem_path", "bpm", "sr"}
        missing  = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Manifest missing columns: {missing}")

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> List[SongData]:
        processor = StemProcessor()
        songs     = []

        for song_id, group in self.df.groupby("song_id"):
            try:
                songs.append(self._process_song(processor, song_id, group))
            except Exception as e:
                print(f"  Warning: Failed to load {song_id}: {e}")

        print(f"Loaded {len(songs)} songs from manifest")
        return songs

    def load_single(self, song_id: str) -> SongData:
        group = self.df[self.df["song_id"] == song_id]
        if group.empty:
            raise ValueError(f"Song {song_id} not found in manifest")
        return self._process_song(StemProcessor(), song_id, group)

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------

    def _process_song(
        self, processor: StemProcessor, song_id: str, group: pd.DataFrame
    ) -> SongData:
        stems: Dict[str, np.ndarray] = {}

        for _, row in group.iterrows():
            stem_path = row["stem_path"]
            stem_name = row.get("stem_name") or Path(stem_path).stem
            stems[stem_name] = self._load_audio_cached(stem_path)

        bpm = group["bpm"].iloc[0]
        sr  = group["sr"].iloc[0]

        return processor.process(
            stems   = stems,
            song_id = song_id,
            sr      = sr,
            bpm     = bpm,
        )

    def _load_audio_cached(self, stem_path: str) -> np.ndarray:
        if self.cache_dir:
            cache_file = self.cache_dir / f"{stem_path.replace('/', '_')}.npy"
            if cache_file.exists():
                return np.load(cache_file)

        audio = self.download_fn(stem_path)

        if self.cache_dir:
            np.save(cache_file, audio)

        return audio

    # ------------------------------------------------------------------
    # Default download function
    # ------------------------------------------------------------------

    @staticmethod
    def _load_local_audio(stem_path: str) -> np.ndarray:
        try:
            audio, sr = sf.read(stem_path, dtype="float32")
        except Exception:
            import librosa
            audio, sr = librosa.load(stem_path, sr=None, mono=True)
            audio = audio.astype(np.float32)
        return audio

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def songs_in_manifest(self) -> List[str]:
        return sorted(self.df["song_id"].unique().tolist())

    def manifest_stats(self) -> dict:
        return {
            "num_rows":    len(self.df),
            "num_songs":   self.df["song_id"].nunique(),
            "num_stems":   len(self.df),
            "bpm_range":   (self.df["bpm"].min(), self.df["bpm"].max()),
            "sample_rates": sorted(self.df["sr"].unique().tolist()),
        }