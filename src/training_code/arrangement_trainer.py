"""
ArrangementTrainer - end-to-end training orchestrator.

Stage 1 (VQ-VAE):
    For each song, for each bar position:
        extract_bar_feature_vector(all melody clips, all harmony clips) → (25,)
    Train VQ-VAE on all these vectors → global codebook

Stage 2 (Token encoding):
    Re-encode all bar vectors → integer token per bar
    Group by song → one token sequence per song

Stage 3 (Transformer):
    Train ImprovedMusicalTransformer on token sequences (next-token prediction)
"""

import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset

from src.model_preprocessing.feature_vector_segmentation import extract_bar_feature_vector
from src.model_preprocessing.quantization.vq_vae import VQ_VAE, VQ_VAE_TRAINER
from src.main_model.transformers_improved import ImprovedMusicalTransformer
from src.song_data import SongData


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    # VQ-VAE
    input_dim:      int   = 25
    hidden_dim:     int   = 128
    codebook_size:  int   = 64
    emb_dim:        int   = 32
    vq_coef:        float = 0.2
    commit_coef:    float = 0.4
    vq_lr:          float = 1e-3
    vq_epochs:      int   = 50
    vq_batch_size:  int   = 64

    # Transformer
    model_dim:               int   = 256
    num_heads:               int   = 4
    num_layers:              int   = 4
    max_seq_len:             int   = 512
    dropout:                 float = 0.1
    transformer_lr:          float = 1e-3
    transformer_epochs:      int   = 100
    transformer_batch_size:  int   = 32

    # Special tokens  (real music tokens are offset by num_special_tokens)
    pad_idx:           int = 0
    bos_idx:           int = 1
    eos_idx:           int = 2
    num_special_tokens: int = 3

    # Checkpointing
    checkpoint_dir:      str = "checkpoints"
    save_every_n_epochs: int = 10

    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    @property
    def vocab_size(self) -> int:
        return self.codebook_size + self.num_special_tokens


# ---------------------------------------------------------------------------
# ArrangementTrainer
# ---------------------------------------------------------------------------

class ArrangementTrainer:
    """
    Full training pipeline: SongData list → trained VQ-VAE + Transformer.

    Usage:
        config  = TrainingConfig()
        trainer = ArrangementTrainer(config)
        trainer.train(songs)          # songs: List[SongData]
        trainer.save("model.pt")
    """

    def __init__(self, config: TrainingConfig):
        self.config  = config
        self.device  = torch.device(config.device)
        self.vq_vae: Optional[VQ_VAE] = None
        self.transformer: Optional[ImprovedMusicalTransformer] = None

    # ------------------------------------------------------------------
    # Stage 1 helpers — feature extraction
    # ------------------------------------------------------------------

    def _extract_features(
        self, songs: List[SongData]
    ) -> Tuple[np.ndarray, List[Tuple[str, int]]]:
        """
        Extract one 25-dim feature vector per (song, bar) pair.

        Returns:
            features: (N, 25) float32 array
            metadata: list of (song_id, bar_idx) — one entry per row in features
        """
        features = []
        metadata = []

        for song in songs:
            if not song.is_valid():
                continue

            for bar_idx in range(song.num_bars):
                mel  = song.melody_clips_at(bar_idx)
                harm = song.harmony_clips_at(bar_idx)

                vec = extract_bar_feature_vector(
                    melody_clips  = mel  if mel  else None,
                    harmony_clips = harm if harm else None,
                    sr            = song.sr,
                )
                features.append(vec)
                metadata.append((song.song_id, bar_idx))

        return np.array(features, dtype=np.float32), metadata

    # ------------------------------------------------------------------
    # Stage 1 — VQ-VAE training
    # ------------------------------------------------------------------

    def _build_vq_vae(self) -> VQ_VAE:
        cfg = self.config
        return VQ_VAE(
            input_dim     = cfg.input_dim,
            hidden_dim    = cfg.hidden_dim,
            codebook_size = cfg.codebook_size,
            emb_dim       = cfg.emb_dim,
            vq_coef       = cfg.vq_coef,
            commit_coef   = cfg.commit_coef,
        )

    def train_vq_vae(self, features: np.ndarray) -> None:
        """
        Train VQ-VAE on (N, 25) feature matrix.
        Populates self.vq_vae and saves checkpoint.
        """
        cfg = self.config
        self.vq_vae = self._build_vq_vae().to(self.device)
        trainer = VQ_VAE_TRAINER(self.vq_vae, lr=cfg.vq_lr, device=self.device)

        dataset = TensorDataset(torch.from_numpy(features))
        loader  = DataLoader(
            dataset,
            batch_size = cfg.vq_batch_size,
            shuffle    = True,
            drop_last  = True,
        )

        print(f"  VQ-VAE: {len(features)} samples, {cfg.vq_epochs} epochs")
        for epoch in range(cfg.vq_epochs):
            epoch_loss = defaultdict(float)

            for (x,) in loader:
                breakdown = trainer.step(x)
                for k, v in breakdown.items():
                    epoch_loss[k] += v

            if (epoch + 1) % 10 == 0:
                n = len(loader)
                print(
                    f"    epoch {epoch+1:>3}/{cfg.vq_epochs}"
                    f"  total={epoch_loss['total']/n:.4f}"
                    f"  recon={epoch_loss['recon']/n:.4f}"
                    f"  vq={epoch_loss['vq']/n:.4f}"
                    f"  commit={epoch_loss['commit']/n:.4f}"
                )

        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        trainer.save(os.path.join(cfg.checkpoint_dir, "vq_vae.pt"))
        print(f"  Saved → {cfg.checkpoint_dir}/vq_vae.pt")

    # ------------------------------------------------------------------
    # Stage 2 — encode bars to token sequences
    # ------------------------------------------------------------------

    def _encode_to_token_sequences(
        self,
        features: np.ndarray,
        metadata: List[Tuple[str, int]],
    ) -> Dict[str, List[int]]:
        """
        Encode feature vectors to codebook indices, group by song.

        Tokens are offset by num_special_tokens so that:
            0 = PAD,  1 = BOS,  2 = EOS,  3..N+2 = music tokens

        Returns:
            {song_id: [tok_bar0, tok_bar1, ...]}  ordered by bar_idx
        """
        assert self.vq_vae is not None, "Train VQ-VAE first."

        features_t = torch.from_numpy(features).to(self.device)
        indices    = self.vq_vae.encode_to_indices(features_t).cpu().numpy()

        song_tokens: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        for (song_id, bar_idx), raw_tok in zip(metadata, indices):
            music_tok = int(raw_tok) + self.config.num_special_tokens
            song_tokens[song_id].append((bar_idx, music_tok))

        # Sort each song's tokens by bar position
        result = {}
        for song_id, bar_toks in song_tokens.items():
            bar_toks.sort(key=lambda x: x[0])
            result[song_id] = [t for _, t in bar_toks]

        return result

    # ------------------------------------------------------------------
    # Stage 3 — Transformer training
    # ------------------------------------------------------------------

    def _build_transformer(self) -> ImprovedMusicalTransformer:
        cfg = self.config
        return ImprovedMusicalTransformer(
            vocab_size  = cfg.vocab_size,
            model_dim   = cfg.model_dim,
            num_heads   = cfg.num_heads,
            num_layers  = cfg.num_layers,
            max_seq_len = cfg.max_seq_len,
            dropout     = cfg.dropout,
            pad_idx     = cfg.pad_idx,
            bos_idx     = cfg.bos_idx,
            eos_idx     = cfg.eos_idx,
        )

    def train_transformer(self, song_sequences: Dict[str, List[int]]) -> None:
        """
        Train transformer on token sequences.
        Populates self.transformer and saves checkpoints.
        """
        cfg = self.config
        self.transformer = self._build_transformer().to(self.device)
        optimizer = torch.optim.Adam(self.transformer.parameters(), lr=cfg.transformer_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.transformer_epochs
        )

        # Build padded dataset: prepend BOS, append EOS
        sequences = []
        for tokens in song_sequences.values():
            seq = [cfg.bos_idx] + tokens + [cfg.eos_idx]
            sequences.append(torch.tensor(seq, dtype=torch.long))

        padded  = pad_sequence(sequences, batch_first=True, padding_value=cfg.pad_idx)
        masks   = (padded != cfg.pad_idx).long()
        dataset = TensorDataset(padded, masks)
        loader  = DataLoader(
            dataset,
            batch_size = cfg.transformer_batch_size,
            shuffle    = True,
        )

        print(f"  Transformer: {len(sequences)} sequences, {cfg.transformer_epochs} epochs")
        for epoch in range(cfg.transformer_epochs):
            self.transformer.train()
            total_loss = 0.0

            for seqs, attn_masks in loader:
                seqs       = seqs.to(self.device)
                attn_masks = attn_masks.to(self.device)

                # Targets = same sequence; mask padding positions with -100
                targets = seqs.clone()
                targets[attn_masks == 0] = -100

                optimizer.zero_grad()
                loss = self.transformer.compute_loss(seqs, targets, attn_masks)
                loss.backward()
                nn.utils.clip_grad_norm_(self.transformer.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            scheduler.step()

            if (epoch + 1) % 10 == 0:
                print(
                    f"    epoch {epoch+1:>3}/{cfg.transformer_epochs}"
                    f"  loss={total_loss/len(loader):.4f}"
                )

            if (epoch + 1) % cfg.save_every_n_epochs == 0:
                self._save_transformer_checkpoint(epoch + 1)

        self._save_transformer_checkpoint("final")

    def _save_transformer_checkpoint(self, tag) -> None:
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.config.checkpoint_dir, f"transformer_{tag}.pt")
        torch.save(
            {"model_state": self.transformer.state_dict(), "config": self.config},
            path,
        )
        print(f"  Saved → {path}")

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def train(self, songs: List[SongData]) -> None:
        """
        Run the full training pipeline on a list of SongData objects.

        Args:
            songs: produced by the data loading layer (HF or local)
        """
        valid = [s for s in songs if s.is_valid()]
        print(f"Training on {len(valid)}/{len(songs)} valid songs.")

        print("\n[1/3] Extracting bar features...")
        features, metadata = self._extract_features(valid)
        print(f"      {len(features)} bar feature vectors")

        print("\n[2/3] Training VQ-VAE...")
        self.train_vq_vae(features)

        print("\n[  ] Encoding token sequences...")
        song_sequences = self._encode_to_token_sequences(features, metadata)
        lengths = [len(s) for s in song_sequences.values()]
        print(f"      {len(song_sequences)} sequences, avg length {np.mean(lengths):.1f}")

        print("\n[3/3] Training Transformer...")
        self.train_transformer(song_sequences)

        print("\nTraining complete.")

    # ------------------------------------------------------------------
    # Save / load full model
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save both models and config to a single checkpoint file."""
        assert self.vq_vae is not None and self.transformer is not None
        torch.save(
            {
                "vq_vae":      self.vq_vae.state_dict(),
                "transformer": self.transformer.state_dict(),
                "config":      self.config,
            },
            path,
        )
        print(f"Saved full model → {path}")

    @classmethod
    def load(cls, path: str) -> "ArrangementTrainer":
        """Load a previously saved ArrangementTrainer from checkpoint."""
        checkpoint = torch.load(path, map_location="cpu")
        config     = checkpoint["config"]

        trainer = cls(config)
        trainer.vq_vae = trainer._build_vq_vae()
        trainer.vq_vae.load_state_dict(checkpoint["vq_vae"])
        trainer.vq_vae.to(trainer.device)

        trainer.transformer = trainer._build_transformer()
        trainer.transformer.load_state_dict(checkpoint["transformer"])
        trainer.transformer.to(trainer.device)

        return trainer
