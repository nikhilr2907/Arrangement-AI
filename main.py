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

import numpy as np

from src.song_data import SongData
from src.dataloaders.manifest_loader import ManifestLoader
from src.dataloaders.hf_streaming_loader import HFStreamingLoader
from src.training_code.arrangement_trainer import ArrangementTrainer, TrainingConfig


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

def load_songs_from_manifest(manifest_path: str) -> list[SongData]:
    loader = ManifestLoader(manifest_path, cache_dir=".cache/stems")
    print(f"Loading from manifest: {manifest_path}")
    print(f"  {loader.manifest_stats()}")
    return loader.load()


def load_songs_from_hf(dataset_name: str, split: str = "train"):
    loader = HFStreamingLoader(dataset_name, split=split)
    print(f"Streaming from HuggingFace: {dataset_name}/{split}")
    return loader.stream()


def load_songs_synthetic() -> list[SongData]:
    """Placeholder synthetic data for development."""
    rng     = np.random.default_rng(42)
    sr      = 22050
    bar_len = sr * 2
    n_bars  = 16

    songs = []
    for i in range(40):
        stem1 = [rng.standard_normal(bar_len).astype(np.float32) for _ in range(n_bars)]
        stem2 = [rng.standard_normal(bar_len).astype(np.float32) for _ in range(n_bars)]
        stem3 = [rng.standard_normal(bar_len).astype(np.float32) for _ in range(n_bars)]

        songs.append(SongData(
            song_id = f"synthetic_{i:03d}",
            stems   = [stem1, stem2, stem3],
            sr      = sr,
        ))

    return songs


# ---------------------------------------------------------------------------
# TRAINING CONFIG
# ---------------------------------------------------------------------------

config = TrainingConfig(
    input_dim      = 25,
    hidden_dim     = 128,
    codebook_size  = 64,
    emb_dim        = 32,
    vq_lr          = 1e-3,
    vq_epochs      = 50,
    vq_batch_size  = 64,

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
    )
    parser.add_argument("--manifest",   type=str, default=None)
    parser.add_argument("--hf-dataset", type=str, default=None)
    parser.add_argument("--split",      type=str, default="train")
    args = parser.parse_args()

    if args.data_source == "hf":
        if not args.hf_dataset:
            parser.error("--hf-dataset required for --data-source hf")
        songs         = load_songs_from_hf(args.hf_dataset, split=args.split)
        use_streaming = True
    elif args.data_source == "local":
        if not args.manifest:
            parser.error("--manifest required for --data-source local")
        songs         = load_songs_from_manifest(args.manifest)
        use_streaming = False
    else:
        print("Using synthetic data.")
        songs         = load_songs_synthetic()
        use_streaming = False

    trainer = ArrangementTrainer(config)
    if use_streaming:
        trainer.train_streaming(songs)
    else:
        print(f"Training on {len(songs)} songs")
        trainer.train(songs)

    trainer.save("checkpoints/arrangement_model.pt")
    print("Training complete. Model saved to checkpoints/arrangement_model.pt")
