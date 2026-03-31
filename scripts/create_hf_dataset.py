"""
Convert local stem folders to HuggingFace parquet dataset.

Expected folder structure:
  data/
    song_001/
      melody.wav          ← required (at least one melody)
      harmony_1.wav       ← optional
      harmony_2.wav       ← optional
      drums.wav           ← optional
      bass.wav            ← optional
      metadata.json       ← optional {"bpm": 120.0, "sr": 22050}
    song_002/
      melody.wav
      drums.wav
      harmony_1.wav
      metadata.json
    ...

Stem type is inferred from filename:
  "melody*"       → melody
  "drum*", "kick*", "snare*", "perc*"  → drums
  everything else → harmony

Produces parquet with columns:
  song_id | sr | bpm | <stem_name>_audio | <stem_name>_type | ...

Stems missing for a song are stored as None (nullable).

Usage:
    python scripts/create_hf_dataset.py ./data --repo username/music_stems
    python scripts/create_hf_dataset.py ./data --repo username/music_stems --bpm 128 --private
"""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import Dataset


# ---------------------------------------------------------------------------
# Stem type inference from filename
# ---------------------------------------------------------------------------

_MELODY_KEYWORDS  = {"melody", "vocal", "lead", "voice", "main", "singer", "vox"}
_DRUMS_KEYWORDS   = {"drum", "kick", "snare", "perc", "cymbal", "hat", "clap", "beat"}
_HARMONY_KEYWORDS = {"harmony", "bass", "synth", "pad", "chord", "guitar", "strings",
                     "keys", "piano", "organ", "brass", "instrumental", "bgv", "backing"}


def infer_stem_type(stem_name: str) -> str:
    """
    Infer melody / harmony / drums from stem filename.

    Args:
        stem_name: Stem file name without extension (e.g. "drums", "harmony_1")

    Returns:
        "melody", "harmony", or "drums"
    """
    name = stem_name.lower()

    for keyword in _MELODY_KEYWORDS:
        if keyword in name:
            return "melody"

    for keyword in _DRUMS_KEYWORDS:
        if keyword in name:
            return "drums"

    for keyword in _HARMONY_KEYWORDS:
        if keyword in name:
            return "harmony"

    return "harmony"  # safe default


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".aiff", ".aif"}


def load_audio_mono(path: Path) -> tuple:
    """
    Load audio file as mono float32.

    Returns:
        (audio: np.ndarray, sr: int)
    """
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    # Mix down to mono
    audio = np.mean(audio, axis=1)
    return audio.astype(np.float32), int(sr)


def load_song_metadata(song_dir: Path, bpm_default: float) -> dict:
    """
    Load BPM and SR from metadata.json if it exists.

    metadata.json format:
        {"bpm": 120.0, "sr": 22050}

    Returns:
        {"bpm": float, "sr": int or None}
    """
    meta_path = song_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            data = json.load(f)
        return {
            "bpm": float(data.get("bpm", bpm_default)),
            "sr": int(data["sr"]) if "sr" in data else None,
        }
    return {"bpm": float(bpm_default), "sr": None}


# ---------------------------------------------------------------------------
# Dataset creation
# ---------------------------------------------------------------------------

def create_hf_dataset_from_stems(
    stem_root_dir: str,
    output_repo: str,
    bpm_default: float = 120.0,
    private: bool = False,
    split: str = "train",
):
    """
    Scan local stem folders and push to HuggingFace Hub as parquet dataset.

    Args:
        stem_root_dir: Root folder containing one subfolder per song
        output_repo:   HF repo ID (e.g. "username/my_dataset")
        bpm_default:   Fallback BPM if no metadata.json found
        private:       Whether to make the HF dataset private
        split:         Dataset split name (default: "train")
    """
    stem_root = Path(stem_root_dir)
    print(f"Scanning {stem_root} for stems...\n")

    # ------------------------------------------------------------------
    # Pass 1: collect all rows and discover all unique stem names
    # ------------------------------------------------------------------

    raw_rows = []
    all_stem_names = set()

    for song_dir in sorted(stem_root.iterdir()):
        if not song_dir.is_dir():
            continue

        song_id = song_dir.name
        meta = load_song_metadata(song_dir, bpm_default)

        # Find all audio files in this song folder
        audio_files = sorted(
            f for f in song_dir.iterdir()
            if f.suffix.lower() in AUDIO_EXTENSIONS
        )

        if not audio_files:
            print(f"  ⚠ Skipping {song_id}: no audio files found")
            continue

        song_stems = {}  # stem_name → audio bytes

        for audio_path in audio_files:
            stem_name = audio_path.stem  # filename without extension

            try:
                audio, file_sr = load_audio_mono(audio_path)

                # Use file's SR if not in metadata
                if meta["sr"] is None:
                    meta["sr"] = file_sr

                song_stems[stem_name] = audio.tobytes()
                all_stem_names.add(stem_name)
                print(f"  ✓ {song_id}/{stem_name}.{audio_path.suffix.lstrip('.')} "
                      f"({len(audio)} samples, SR={file_sr})")

            except Exception as e:
                print(f"  ✗ {song_id}/{audio_path.name}: {e}")

        # Skip songs with no stems loaded
        if not song_stems:
            print(f"  ⚠ Skipping {song_id}: all stems failed to load")
            continue

        # Check at least one melody stem exists
        melody_stems = [k for k in song_stems if infer_stem_type(k) == "melody"]
        if not melody_stems:
            print(f"  ⚠ Skipping {song_id}: no melody stem detected "
                  f"(stems: {list(song_stems.keys())})")
            continue

        raw_rows.append({
            "song_id": song_id,
            "sr": meta["sr"] or 22050,
            "bpm": meta["bpm"],
            "stems": song_stems,
        })

    if not raw_rows:
        print("\nERROR: No valid songs found. Check folder structure.")
        return

    print(f"\nFound {len(raw_rows)} valid songs")
    print(f"Unique stem names: {sorted(all_stem_names)}\n")

    # ------------------------------------------------------------------
    # Pass 2: build flat rows — one column per stem (None if absent)
    # ------------------------------------------------------------------
    # Columns: song_id, sr, bpm,
    #          <stem>_audio (bytes or None), <stem>_type (str or None)
    #          for each stem in all_stem_names

    all_stem_names = sorted(all_stem_names)  # stable ordering

    flat_rows = []
    for raw in raw_rows:
        row = {
            "song_id": raw["song_id"],
            "sr": raw["sr"],
            "bpm": raw["bpm"],
        }
        for stem_name in all_stem_names:
            if stem_name in raw["stems"]:
                row[f"{stem_name}_audio"] = raw["stems"][stem_name]
                row[f"{stem_name}_type"]  = infer_stem_type(stem_name)
            else:
                row[f"{stem_name}_audio"] = None
                row[f"{stem_name}_type"]  = None

        flat_rows.append(row)

    # ------------------------------------------------------------------
    # Build and push dataset
    # ------------------------------------------------------------------

    print(f"[1/2] Building dataset ({len(flat_rows)} rows, "
          f"{len(all_stem_names)} stem columns)...")

    dataset_dict = {k: [r[k] for r in flat_rows] for k in flat_rows[0]}
    dataset = Dataset.from_dict(dataset_dict)

    print(f"[2/2] Pushing to {output_repo} (split={split}, private={private})...")
    dataset.push_to_hub(output_repo, split=split, private=private)

    print(f"\n✓ Done!")
    print(f"Dataset: https://huggingface.co/datasets/{output_repo}")
    print(f"Columns: {list(dataset.column_names)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert local stem folders to HF parquet dataset"
    )
    parser.add_argument(
        "stem_root",
        help="Path to folder containing one subfolder per song",
    )
    parser.add_argument(
        "--repo",
        required=True,
        help="HuggingFace dataset repo ID (e.g. username/my_dataset)",
    )
    parser.add_argument(
        "--bpm",
        type=float,
        default=120.0,
        help="Default BPM if no metadata.json found (default: 120)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split name (default: train)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make HF dataset private",
    )
    args = parser.parse_args()

    create_hf_dataset_from_stems(
        stem_root_dir=args.stem_root,
        output_repo=args.repo,
        bpm_default=args.bpm,
        private=args.private,
        split=args.split,
    )
