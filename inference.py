"""
Arrangement model — inference entry point.

Generates a musical arrangement from available stems.

Usage:
    python inference.py --model checkpoints/arrangement_model.pt --manifest path/to/stems.csv
    python inference.py --model checkpoints/arrangement_model.pt  # synthetic stems
"""

import argparse
from pathlib import Path

import numpy as np

from src.song_data import SongData
from src.dataloaders.manifest_loader import ManifestLoader
from src.dataloaders.hf_streaming_loader import HFStreamingLoader
from src.training_code.arrangement_trainer import ArrangementTrainer
from src.inference.stem_library import StemLibrary
from src.inference.arrangement_generator import ArrangementGenerator


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

def load_inference_stems_from_manifest(manifest_path: str) -> list[SongData]:
    """
    Load inference stems from a manifest CSV/parquet.

    Same format as training manifest.
    """
    loader = ManifestLoader(manifest_path, cache_dir=".cache/inference_stems")
    print(f"Loading stems from: {manifest_path}")
    print(f"  {loader.manifest_stats()}")
    songs = loader.load()
    return songs


def load_inference_stems_from_hf(dataset_name: str, split: str = "test") -> list[SongData]:
    """
    Load inference stems from HuggingFace dataset.

    Materializes all stems into memory (needed for indexing during inference).

    Args:
        dataset_name: HF repo (e.g., "username/music_stems")
        split: Dataset split to load (default: "test")

    Returns:
        List[SongData]: All stems from dataset
    """
    loader = HFStreamingLoader(dataset_name, split=split)
    print(f"Loading stems from HuggingFace: {dataset_name}/{split}")
    return list(loader.stream())  # Materialize iterator to list


def load_inference_stems_synthetic() -> list[SongData]:
    """Placeholder synthetic stems for development."""
    rng     = np.random.default_rng(0)
    sr      = 22050
    bar_len = sr * 2
    n_bars  = 16

    songs = []
    for i in range(5):
        melody  = [rng.standard_normal(bar_len).astype(np.float32) for _ in range(n_bars)]
        harm1   = [rng.standard_normal(bar_len).astype(np.float32) for _ in range(n_bars)]
        harm2   = [rng.standard_normal(bar_len).astype(np.float32) for _ in range(n_bars)]

        songs.append(SongData(
            song_id       = f"inference_stem_{i:02d}",
            melody_stems  = [melody],
            harmony_stems = [harm1, harm2],
            sr            = sr,
        ))

    return songs


# ---------------------------------------------------------------------------
# INFERENCE CONFIG
# ---------------------------------------------------------------------------

GENERATE_BARS = 16      # how many bars to generate
TEMPERATURE   = 0.85    # higher = more varied, lower = more predictable
TOP_K         = 8       # restrict each step to top-k available tokens


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate arrangement from stems")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model checkpoint",
    )
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
        help="HuggingFace dataset repo for --data-source hf",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="HF dataset split (default: test)",
    )
    parser.add_argument(
        "--bars",
        type=int,
        default=GENERATE_BARS,
        help="Number of bars to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=TEMPERATURE,
        help="Generation temperature (higher = more varied)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help="Top-k tokens to sample from at each step",
    )
    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.model} ...")
    trainer = ArrangementTrainer.load(args.model)

    # Load stems
    if args.data_source == "hf":
        if not args.hf_dataset:
            parser.error("--hf-dataset required for --data-source hf")
        stems = load_inference_stems_from_hf(args.hf_dataset, split=args.split)
    elif args.data_source == "local":
        if not args.manifest:
            parser.error("--manifest required for --data-source local")
        stems = load_inference_stems_from_manifest(args.manifest)
    else:
        print("Using synthetic stems. To use real stems, provide --data-source local/hf")
        stems = load_inference_stems_synthetic()

    # Index stems into token space
    print("\nIndexing stems into token space ...")
    library = StemLibrary(trainer.vq_vae, trainer.config)
    library.index(stems)

    # Generate arrangement
    print(f"\nGenerating {args.bars}-bar arrangement ...")
    generator = ArrangementGenerator(trainer.transformer, trainer.config, library)
    arrangement = generator.generate(
        num_bars    = args.bars,
        temperature = args.temperature,
        top_k       = args.top_k,
    )

    # Print plan
    generator.print_arrangement(arrangement)

    print("\nDone. To render audio, mix arrangement bars and output to file.")
    print("Each ArrangementBar has .melody_clips and .harmony_clips (List[np.ndarray])")
