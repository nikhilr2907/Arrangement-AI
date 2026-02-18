"""
StemLibrary — indexes available stem clips into the token space.

For each (song, bar_position) in the provided stems, extracts a combined
25-dim feature vector, encodes it through the trained VQ-VAE, and stores
the clip combination indexed by its token.

At generation time ArrangementGenerator queries the library to retrieve
the actual audio clips corresponding to each predicted token.

Fallback: if the transformer predicts a token with no clips in the library,
the nearest token by codebook L2 distance is used instead.
"""

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import numpy as np
import torch

from src.model_preprocessing.feature_vector_segmentation import extract_bar_feature_vector
from src.model_preprocessing.quantization.vq_vae import VQ_VAE
from src.song_data import SongData
from src.training_code.arrangement_trainer import TrainingConfig


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ClipEntry:
    """One available clip combination at a specific bar position."""
    song_id:       str
    bar_idx:       int
    melody_clips:  List[np.ndarray]
    harmony_clips: List[np.ndarray]
    feature_vec:   np.ndarray        # (25,) for fallback distance lookup
    token:         int               # offset token (>= num_special_tokens)


@dataclass
class ArrangementBar:
    """One bar of a generated arrangement."""
    bar_position:  int
    token:         int
    melody_clips:  List[np.ndarray]
    harmony_clips: List[np.ndarray]
    source_song_id: str
    source_bar_idx: int


# ---------------------------------------------------------------------------
# StemLibrary
# ---------------------------------------------------------------------------

class StemLibrary:
    """
    Indexes available stem clips into the learned token space.

    Usage:
        library = StemLibrary(vq_vae, config)
        library.index(songs)          # songs: List[SongData]
        entry  = library.query(token) # → ClipEntry
    """

    def __init__(self, vq_vae: VQ_VAE, config: TrainingConfig):
        self.vq_vae  = vq_vae
        self.config  = config
        self._store: Dict[int, List[ClipEntry]] = defaultdict(list)
        self._codebook_vecs: Optional[torch.Tensor] = None  # cached (codebook_size, emb_dim)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index(self, songs: List[SongData]) -> None:
        """
        Encode every bar from every song and store in the library.
        Can be called multiple times to add more clips.
        """
        self.vq_vae.eval()
        device = next(self.vq_vae.parameters()).device

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

                with torch.no_grad():
                    vec_t   = torch.tensor(vec, dtype=torch.float32).unsqueeze(0).to(device)
                    raw_tok = self.vq_vae.encode_to_indices(vec_t).item()

                token = int(raw_tok) + self.config.num_special_tokens

                self._store[token].append(ClipEntry(
                    song_id       = song.song_id,
                    bar_idx       = bar_idx,
                    melody_clips  = mel,
                    harmony_clips = harm,
                    feature_vec   = vec,
                    token         = token,
                ))

        self._codebook_vecs = None  # invalidate cache after re-indexing
        print(f"StemLibrary: {self.total_clips} clips across {len(self.available_tokens)} token types")

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, token: int) -> ClipEntry:
        """
        Return one clip entry for the given token.
        Falls back to the nearest available token if the exact one is missing.
        """
        if token not in self._store:
            token = self._nearest_token(token)
        return random.choice(self._store[token])

    def query_all(self, token: int) -> List[ClipEntry]:
        """Return all clip entries for the given token (or nearest fallback)."""
        if token not in self._store:
            token = self._nearest_token(token)
        return list(self._store[token])

    # ------------------------------------------------------------------
    # Fallback — nearest codebook neighbour
    # ------------------------------------------------------------------

    def _nearest_token(self, token: int) -> int:
        """
        Find the token in the library whose codebook vector is closest
        (L2 distance in emb_dim space) to the requested token.
        """
        codebook = self._get_codebook_vecs()
        raw_tok  = token - self.config.num_special_tokens
        raw_tok  = max(0, min(raw_tok, codebook.shape[0] - 1))  # clamp
        query_v  = codebook[raw_tok]

        best_token = None
        best_dist  = float("inf")

        for avail_token in self._store:
            raw_avail = avail_token - self.config.num_special_tokens
            avail_v   = codebook[raw_avail]
            dist      = float(torch.norm(query_v - avail_v))
            if dist < best_dist:
                best_dist  = dist
                best_token = avail_token

        return best_token

    def _get_codebook_vecs(self) -> torch.Tensor:
        """Cached codebook vectors (codebook_size, emb_dim)."""
        if self._codebook_vecs is None:
            self._codebook_vecs = self.vq_vae.get_codebook_vectors().cpu()
        return self._codebook_vecs

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def available_tokens(self) -> Set[int]:
        return set(self._store.keys())

    @property
    def total_clips(self) -> int:
        return sum(len(v) for v in self._store.values())

    def token_counts(self) -> Dict[int, int]:
        """Number of clips available per token."""
        return {tok: len(entries) for tok, entries in self._store.items()}
