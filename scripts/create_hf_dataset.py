"""
Convert local stem folders to HuggingFace parquet dataset.

Local structure expected:
  data/
    song_001/
      melody.wav
      harmony1.wav
      harmony2.wav
    song_002/
      melody.wav
      harmony1.wav
      ...

Output: Parquet dataset pushed to HF Hub

Usage:
    python scripts/create_hf_dataset.py ./data --repo username/music_stems
"""

from pathlib import Path
import soundfile as sf
import numpy as np
from datasets import Dataset
import argparse


def create_hf_dataset_from_stems(
    stem_root_dir: str,
    output_repo: str,
    bpm_default: float = 120.0,
    sr_default: int = 22050,
):
    """
    Create HF dataset from local stems.

    Args:
        stem_root_dir: Path to folder containing song subdirectories
        output_repo: HuggingFace repo (e.g., "username/music_stems")
        bpm_default: Default BPM if not specified
        sr_default: Default sample rate
    """
    rows = []

    stem_root = Path(stem_root_dir)
    print(f"Scanning {stem_root} for stems...")

    for song_dir in sorted(stem_root.iterdir()):
        if not song_dir.is_dir():
            continue

        song_id = song_dir.name

        # Load stems (melody.wav, harmony1.wav, etc.)
        melody_path = song_dir / "melody.wav"
        harmony_path = song_dir / "harmony1.wav"

        if not melody_path.exists() or not harmony_path.exists():
            print(f"  Skipping {song_id} (missing melody.wav or harmony1.wav)")
            continue

        try:
            melody, sr = sf.read(melody_path, dtype="float32")
            harmony, _ = sf.read(harmony_path, dtype="float32")

            # Stereo → mono
            if melody.ndim > 1:
                melody = np.mean(melody, axis=0)
            if harmony.ndim > 1:
                harmony = np.mean(harmony, axis=0)

            # Store as bytes for parquet
            rows.append({
                "song_id": song_id,
                "melody_audio": melody.astype(np.float32).tobytes(),
                "harmony_audio": harmony.astype(np.float32).tobytes(),
                "sr": int(sr),
                "bpm": float(bpm_default),
            })
            print(f"  ✓ Added {song_id} ({len(melody)} samples)")
        except Exception as e:
            print(f"  ✗ Error loading {song_id}: {e}")

    if not rows:
        print("ERROR: No songs processed!")
        return

    # Create dataset
    print(f"\n[1/2] Creating dataset with {len(rows)} songs...")
    dataset_dict = {k: [r[k] for r in rows] for k in rows[0]}
    dataset = Dataset.from_dict(dataset_dict)

    # Push to Hub (requires HF auth token)
    print(f"[2/2] Pushing to {output_repo}...")
    dataset.push_to_hub(output_repo, split="train", private=False)
    print("✓ Done!")
    print(f"\nDataset available at: https://huggingface.co/datasets/{output_repo}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert local stems to HF parquet dataset"
    )
    parser.add_argument("stem_root", help="Path to folder with song subdirectories")
    parser.add_argument("--repo", required=True, help="HF repo (username/dataset)")
    parser.add_argument("--bpm", type=float, default=120.0, help="Default BPM")
    parser.add_argument("--sr", type=int, default=22050, help="Default sample rate")

    args = parser.parse_args()

    create_hf_dataset_from_stems(
        args.stem_root,
        args.repo,
        bpm_default=args.bpm,
        sr_default=args.sr
    )
