"""
Arrangement model — inference entry point.

Loads a trained model, indexes available stems into the token space,
then generates a musical arrangement via constrained autoregressive decoding.

Data loading (HuggingFace / local) is stubbed below — same pattern as main.py.
Replace load_inference_songs() with the real loader when ready.
"""

import numpy as np

from src.song_data import SongData
from src.training_code.arrangement_trainer import ArrangementTrainer
from src.inference.stem_library import StemLibrary
from src.inference.arrangement_generator import ArrangementGenerator


# ---------------------------------------------------------------------------
# DATA STUB  —  replace with real HF / local loader
# ---------------------------------------------------------------------------

def load_inference_songs() -> list[SongData]:
    """
    Placeholder. Returns synthetic SongData objects.
    Real implementation pulls from HuggingFace or local files.

    In production this would be the stems you want to arrange —
    could be a single song's stems or a pool drawn from multiple songs.
    """
    rng     = np.random.default_rng(0)
    sr      = 22050
    bar_len = sr * 2    # ~2 s per bar at 120 BPM
    n_bars  = 16

    songs = []
    for i in range(5):   # small pool of stems to arrange
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
# INFERENCE CONFIG  —  tweak generation behaviour here
# ---------------------------------------------------------------------------

CHECKPOINT = "checkpoints/arrangement_model.pt"

GENERATE_BARS = 16      # how many bars to generate
TEMPERATURE   = 0.85    # higher = more varied, lower = more predictable
TOP_K         = 8       # restrict each step to top-k available tokens


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 1. Load trained model
    print(f"Loading model from {CHECKPOINT} ...")
    trainer = ArrangementTrainer.load(CHECKPOINT)

    # 2. Load available stems and index into token space
    print("Loading inference stems ...")
    songs   = load_inference_songs()
    library = StemLibrary(trainer.vq_vae, trainer.config)
    library.index(songs)

    # 3. Generate arrangement
    print(f"\nGenerating {GENERATE_BARS}-bar arrangement ...")
    generator   = ArrangementGenerator(trainer.transformer, trainer.config, library)
    arrangement = generator.generate(
        num_bars    = GENERATE_BARS,
        temperature = TEMPERATURE,
        top_k       = TOP_K,
    )

    # 4. Print plan
    generator.print_arrangement(arrangement)

    # 5. TODO: render audio from arrangement clips
    #    Each ArrangementBar has .melody_clips and .harmony_clips (List[np.ndarray])
    #    Mix and concatenate to produce output audio.
    print("\nDone. Clip retrieval complete — audio rendering not yet implemented.")
