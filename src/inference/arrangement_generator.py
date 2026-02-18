"""
ArrangementGenerator — generates a musical arrangement from a StemLibrary.

Steps:
  1. Constrained autoregressive generation:
       transformer predicts next token, but only tokens present in the
       StemLibrary are allowed. This guarantees every predicted token
       maps to at least one real clip.
  2. Retrieval:
       for each token in the predicted sequence, query the library for
       a matching clip combination (melody + harmony stems).
  3. Output:
       ordered list of ArrangementBar — one per bar of the arrangement.
"""

from typing import List, Optional

import torch
import torch.nn.functional as F

from src.inference.stem_library import ArrangementBar, ClipEntry, StemLibrary
from src.main_model.transformers_improved import ImprovedMusicalTransformer
from src.training_code.arrangement_trainer import TrainingConfig


class ArrangementGenerator:
    """
    Generates an arrangement from a trained transformer + populated StemLibrary.

    Usage:
        generator   = ArrangementGenerator(transformer, config, library)
        arrangement = generator.generate(num_bars=16)
        # → List[ArrangementBar]
    """

    def __init__(
        self,
        transformer: ImprovedMusicalTransformer,
        config:      TrainingConfig,
        library:     StemLibrary,
    ):
        self.transformer = transformer
        self.config      = config
        self.library     = library
        self.device      = next(transformer.parameters()).device

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        num_bars:    int,
        seed_tokens: Optional[List[int]] = None,
        temperature: float = 1.0,
        top_k:       Optional[int] = None,
    ) -> List[ArrangementBar]:
        """
        Generate a full arrangement.

        Args:
            num_bars:    Number of bars to generate
            seed_tokens: Optional list of music tokens (offset, >= num_special_tokens)
                         to condition generation on. E.g. the first N bars of a known
                         arrangement.
            temperature: Sampling temperature (lower = more conservative)
            top_k:       If set, restrict sampling to top-k available tokens at each step

        Returns:
            List[ArrangementBar] of length num_bars
        """
        token_sequence = self._constrained_generate(num_bars, seed_tokens, temperature, top_k)
        return self._retrieve(token_sequence)

    # ------------------------------------------------------------------
    # Constrained generation
    # ------------------------------------------------------------------

    def _constrained_generate(
        self,
        num_bars:    int,
        seed_tokens: Optional[List[int]],
        temperature: float,
        top_k:       Optional[int],
    ) -> List[int]:
        """
        Autoregressively predict tokens, masking out anything not in the library.

        Returns a list of raw music token integers (offset, >= num_special_tokens).
        """
        self.transformer.eval()

        # Token ids available in the library — used to build the allow-mask
        available = torch.tensor(
            sorted(self.library.available_tokens),
            dtype=torch.long,
            device=self.device,
        )

        # Seed sequence: [BOS] + optional seed tokens
        ids = [self.config.bos_idx]
        if seed_tokens:
            ids.extend(seed_tokens)

        generated_ids = torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)
        music_tokens: List[int] = list(seed_tokens) if seed_tokens else []

        bars_to_generate = num_bars - len(music_tokens)

        with torch.no_grad():
            for _ in range(bars_to_generate):
                if generated_ids.shape[1] >= self.config.max_seq_len:
                    break

                logits     = self.transformer(generated_ids)        # (1, seq, vocab)
                next_logits = logits[0, -1, :].clone()              # (vocab,)

                # Zero out all tokens not present in the library
                allow_mask = torch.full(
                    (self.config.vocab_size,), float("-inf"), device=self.device
                )
                allow_mask[available] = 0.0
                next_logits = next_logits + allow_mask

                # Optional top-k within the available set
                if top_k is not None:
                    n_avail  = len(available)
                    k        = min(top_k, n_avail)
                    topk_val = torch.topk(next_logits, k).values[-1]
                    next_logits[next_logits < topk_val] = float("-inf")

                probs      = F.softmax(next_logits / max(temperature, 1e-8), dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # (1,)

                if next_token.item() == self.config.eos_idx:
                    break

                generated_ids = torch.cat(
                    [generated_ids, next_token.unsqueeze(0)], dim=1
                )
                music_tokens.append(next_token.item())

        return music_tokens

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def _retrieve(self, token_sequence: List[int]) -> List[ArrangementBar]:
        """
        Map each token in the sequence to a clip combination from the library.
        """
        arrangement: List[ArrangementBar] = []

        for bar_pos, token in enumerate(token_sequence):
            entry: ClipEntry = self.library.query(token)

            arrangement.append(ArrangementBar(
                bar_position   = bar_pos,
                token          = token,
                melody_clips   = entry.melody_clips,
                harmony_clips  = entry.harmony_clips,
                source_song_id = entry.song_id,
                source_bar_idx = entry.bar_idx,
            ))

        return arrangement

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def print_arrangement(self, arrangement: List[ArrangementBar]) -> None:
        """Pretty-print the generated arrangement plan."""
        cfg = self.config
        print(f"\nArrangement ({len(arrangement)} bars):")
        print(f"  {'Bar':>3}  {'Token':>6}  {'Code':>5}  Source")
        print(f"  {'-'*3}  {'-'*6}  {'-'*5}  {'-'*30}")
        for bar in arrangement:
            code = bar.token - cfg.num_special_tokens
            print(
                f"  {bar.bar_position:>3}  {bar.token:>6}  c{code:<4}  "
                f"{bar.source_song_id}[bar {bar.source_bar_idx}]  "
                f"({len(bar.melody_clips)}m + {len(bar.harmony_clips)}h stems)"
            )
