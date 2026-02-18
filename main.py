"""
Arrangement model — training entry point.

Data loading (HuggingFace / local) is implemented in src/dataloaders/ and
plugged in below. For now a synthetic stub stands in so the training loop
can be exercised end-to-end.
"""

import numpy as np

from src.song_data import SongData
from src.training_code.arrangement_trainer import ArrangementTrainer, TrainingConfig


# ---------------------------------------------------------------------------
# DATA STUB  —  replace with real HF / local loader when ready
# ---------------------------------------------------------------------------

def load_songs() -> list[SongData]:
    """
    Placeholder.  Returns synthetic SongData objects with random audio clips.
    Real implementation will live in src/dataloaders/ and pull from HuggingFace.

    Each fake stem is bar-length white noise at 22050 Hz.
    A "bar" at 120 BPM ≈ 2 seconds ≈ 44100 samples.
    """
    rng       = np.random.default_rng(42)
    sr        = 22050
    bar_len   = sr * 2          # samples per bar
    num_songs = 40
    num_bars  = 16

    songs = []
    for i in range(num_songs):
        # 1 melody stem, 2 harmony stems  (variable counts are fine)
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
    print("Loading songs...")
    songs = load_songs()
    print(f"  {len(songs)} songs loaded")

    trainer = ArrangementTrainer(config)
    trainer.train(songs)

    trainer.save("checkpoints/arrangement_model.pt")
