"""
Arrangement model — inference entry point.

Generates a musical arrangement from available stems.

Usage:
    python inference.py --model checkpoints/arrangement_model.pt --data-source hf --hf-dataset username/stems
    python inference.py --model checkpoints/arrangement_model.pt --data-source local --manifest stems.csv
    python inference.py --model checkpoints/arrangement_model.pt  # synthetic stems
"""

import argparse

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
    loader = ManifestLoader(manifest_path, cache_dir=".cache/inference_stems")
    print(f"Loading stems from: {manifest_path}")
    return loader.load()


def load_inference_stems_from_hf(dataset_name: str, split: str = "test") -> list[SongData]:
    loader = HFStreamingLoader(dataset_name, split=split)
    print(f"Loading stems from HuggingFace: {dataset_name}/{split}")
    return list(loader.stream())


def load_inference_stems_synthetic() -> list[SongData]:
    rng     = np.random.default_rng(0)
    sr      = 22050
    bar_len = sr * 2
    n_bars  = 16

    songs = []
    for i in range(5):
        stem1 = [rng.standard_normal(bar_len).astype(np.float32) for _ in range(n_bars)]
        stem2 = [rng.standard_normal(bar_len).astype(np.float32) for _ in range(n_bars)]
        stem3 = [rng.standard_normal(bar_len).astype(np.float32) for _ in range(n_bars)]

        songs.append(SongData(
            song_id = f"inference_stem_{i:02d}",
            stems   = [stem1, stem2, stem3],
            sr      = sr,
        ))

    return songs


# ---------------------------------------------------------------------------
# INFERENCE CONFIG
# ---------------------------------------------------------------------------

GENERATE_BARS = 16
TEMPERATURE   = 0.85
TOP_K         = 8


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate arrangement from stems")
    parser.add_argument("--model",       type=str, required=True)
    parser.add_argument("--data-source", choices=["synthetic", "local", "hf"], default="synthetic")
    parser.add_argument("--manifest",    type=str, default=None)
    parser.add_argument("--hf-dataset",  type=str, default=None)
    parser.add_argument("--split",       type=str, default="test")
    parser.add_argument("--bars",        type=int,   default=GENERATE_BARS)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--top-k",       type=int,   default=TOP_K)
    args = parser.parse_args()

    print(f"Loading model from {args.model} ...")
    trainer = ArrangementTrainer.load(args.model)

    if args.data_source == "hf":
        if not args.hf_dataset:
            parser.error("--hf-dataset required for --data-source hf")
        stems = load_inference_stems_from_hf(args.hf_dataset, split=args.split)
    elif args.data_source == "local":
        if not args.manifest:
            parser.error("--manifest required for --data-source local")
        stems = load_inference_stems_from_manifest(args.manifest)
    else:
        print("Using synthetic stems.")
        stems = load_inference_stems_synthetic()

    print("\nIndexing stems into token space ...")
    library = StemLibrary(trainer.vq_vae, trainer.config)
    library.index(stems)

    print(f"\nGenerating {args.bars}-bar arrangement ...")
    generator   = ArrangementGenerator(trainer.transformer, trainer.config, library)
    arrangement = generator.generate(
        num_bars    = args.bars,
        temperature = args.temperature,
        top_k       = args.top_k,
    )

    generator.print_arrangement(arrangement)

    print("\nDone. Each ArrangementBar has .clips (List[np.ndarray]) — all stems for that bar.")
