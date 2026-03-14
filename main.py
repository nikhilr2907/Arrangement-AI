"""
Arrangement model — training entry point.

Three ways to load data:
  1. HuggingFace dataset (streaming):
       python main.py --data-source hf --hf-dataset username/dataset_name
  2. Local manifest CSV/parquet:
       python main.py --data-source local --manifest path/to/manifest.csv
  3. Synthetic data (development, default):
       python main.py
"""

import argparse
from pathlib import Path

import numpy as np

from src.song_data import SongData
from src.dataloaders.manifest_loader import ManifestLoader
from src.dataloaders.hf_streaming_loader import HFStreamingLoader
from src.training_code.arrangement_trainer import ArrangementTrainer, TrainingConfig


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

def load_songs_from_manifest(manifest_path: str) -> list[SongData]:
    """
    Load from a manifest CSV/parquet.

    Manifest columns: song_id, stem_path, stem_type, bpm, sr, [stem_name]

    To use HuggingFace or DVC paths, pass a custom download_fn:
        loader = ManifestLoader(
            manifest_path,
            download_fn=your_hf_or_dvc_download_func
        )
    """
    loader = ManifestLoader(manifest_path, cache_dir=".cache/stems")
    print(f"Loading from manifest: {manifest_path}")
    print(f"  {loader.manifest_stats()}")
    songs = loader.load()
    return songs


def load_songs_from_hf(dataset_name: str, split: str = "train"):
    """
    Load songs from HuggingFace dataset (streaming).

    Does not download entire dataset — streams one song at a time.

    Args:
        dataset_name: HF repo (e.g., "username/music_stems")
        split: Dataset split to load (default: "train")

    Returns:
        Iterator[SongData]: Streaming songs
    """
    loader = HFStreamingLoader(dataset_name, split=split)
    print(f"Streaming from HuggingFace: {dataset_name}/{split}")
    return loader.stream()  # Returns Iterator[SongData]


def load_songs_synthetic() -> list[SongData]:
    """
    Placeholder synthetic data for development.
    Replace load_songs() call with load_songs_from_manifest(path) when ready.
    """
    rng      = np.random.default_rng(42)
    sr       = 22050
    bar_len  = sr * 2          # ~2 seconds per bar at 120 BPM
    num_bars = 16

    songs = []
    for i in range(40):
        melody_stem  = [rng.standard_normal(bar_len).astype(np.float32) for _ in range(num_bars)]
        harmony_stem1 = [rng.standard_normal(bar_len).astype(np.float32) for _ in range(num_bars)]
        harmony_stem2 = [rng.standard_normal(bar_len).astype(np.float32) for _ in range(num_bars)]

        songs.append(SongData(
            song_id       = f"synthetic_{i:03d}",
            melody_stems  = [melody_stem],
            harmony_stems = [harmony_stem1, harmony_stem2],
            sr            = sr,
        ))

    return songs


# ---------------------------------------------------------------------------
# TRAINING CONFIG
# ---------------------------------------------------------------------------

config = TrainingConfig(
    # VQ-VAE
    input_dim      = 25,
    hidden_dim     = 128,
    codebook_size  = 64,
    emb_dim        = 32,
    vq_lr          = 1e-3,
    vq_epochs      = 50,
    vq_batch_size  = 64,

    # Transformer
    model_dim              = 256,
    num_heads              = 4,
    num_layers             = 4,
    max_seq_len            = 512,
    transformer_lr         = 1e-3,
    transformer_epochs     = 100,
    transformer_batch_size = 32,

    checkpoint_dir      = "checkpoints",
    save_every_n_epochs = 25,
)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train arrangement model")
    parser.add_argument(
        "--data-source",
        choices=["synthetic", "local", "hf"],
        default="synthetic",
        help="Data source: synthetic (default), local manifest, or HuggingFace dataset",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Path to manifest CSV/parquet for --data-source local",
    )
    parser.add_argument(
        "--hf-dataset",
        type=str,
        default=None,
        help="HuggingFace dataset repo for --data-source hf (e.g., username/dataset_name)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="HF dataset split (default: train)",
    )
    args = parser.parse_args()

    # Load data
    if args.data_source == "hf":
        if not args.hf_dataset:
            parser.error("--hf-dataset required for --data-source hf")
        songs = load_songs_from_hf(args.hf_dataset, split=args.split)
        use_streaming = True
    elif args.data_source == "local":
        if not args.manifest:
            parser.error("--manifest required for --data-source local")
        songs = load_songs_from_manifest(args.manifest)
        use_streaming = False
    else:
        print("Using synthetic data. To use real data, provide --data-source local/hf with --manifest/--hf-dataset")
        songs = load_songs_synthetic()
        use_streaming = False

    # Train
    trainer = ArrangementTrainer(config)
    if use_streaming:
        print("Training with streaming data (two passes)...")
        trainer.train_streaming(songs)
    else:
        print(f"Training on {len(songs)} songs")
        trainer.train(songs)

    # Save
    trainer.save("checkpoints/arrangement_model.pt")
    print("Training complete. Model saved to checkpoints/arrangement_model.pt")
