"""
Improved transformer for handling variable-length chunks and next-sequence prediction.

Key improvements:
1. Padding and attention masking for variable-length sequences
2. Special tokens (PAD, BOS, EOS)
3. Multiple training strategies for variable-length chunks
4. Next-sequence prediction (NSP) objectives
"""

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from typing import Optional, Tuple, List
import math


class ImprovedMusicalTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        model_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        pad_idx: int = 0,
        bos_idx: int = 1,
        eos_idx: int = 2,
    ):
        """
        Args:
            vocab_size: Total vocabulary size (including special tokens)
            pad_idx: Padding token index
            bos_idx: Beginning of sequence token index
            eos_idx: End of sequence token index
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.model_dim = model_dim
        self.max_seq_len = max_seq_len
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, model_dim, padding_idx=pad_idx)

        # Learned positional encoding
        self.positional_encoding = nn.Embedding(max_seq_len, model_dim)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )

        # Output projection
        self.output_linear = nn.Linear(model_dim, vocab_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def _create_padding_mask(self, seq: torch.Tensor) -> torch.Tensor:
        """Create mask for padded positions (True = masked/ignored)."""
        return seq == self.pad_idx

    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal mask preventing attention to future positions."""
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=device)
        return mask

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len) - token indices
            attention_mask: (batch, seq_len) - 1 for real tokens, 0 for padding

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Create position indices
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        # Embed tokens and add positional encoding
        token_embeds = self.embedding(input_ids)  # (batch, seq_len, model_dim)
        pos_embeds = self.positional_encoding(positions)  # (batch, seq_len, model_dim)
        embedded = self.dropout(token_embeds + pos_embeds)

        # Create masks
        causal_mask = self._create_causal_mask(seq_len, device)

        if attention_mask is None:
            padding_mask = self._create_padding_mask(input_ids)
        else:
            padding_mask = ~attention_mask.bool()

        # Transformer forward pass
        output = self.transformer_decoder(
            embedded,
            embedded,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=padding_mask,
        )

        # Project to vocabulary
        logits = self.output_linear(output)

        return logits

    def compute_loss(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss for next-token prediction.

        Args:
            input_ids: (batch, seq_len) - input sequence
            target_ids: (batch, seq_len) - target sequence (shifted by 1)
            attention_mask: (batch, seq_len) - mask for valid positions
        """
        logits = self.forward(input_ids, attention_mask)

        # Shift logits and targets for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_targets = target_ids[:, 1:].contiguous()

        # Flatten for loss computation
        loss_fct = nn.CrossEntropyLoss(ignore_index=ignore_index)
        loss = loss_fct(
            shift_logits.view(-1, self.vocab_size),
            shift_targets.view(-1),
        )

        return loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        stop_at_eos: bool = True,
    ) -> torch.Tensor:
        """
        Autoregressive generation with various sampling strategies.

        Args:
            input_ids: (batch, seq_len) - context tokens
            max_length: Maximum length to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top k tokens
            top_p: Nucleus sampling threshold
            stop_at_eos: Stop generation when EOS is produced
        """
        self.eval()
        generated = input_ids.clone()

        for _ in range(max_length):
            if generated.size(1) >= self.max_seq_len:
                break

            logits = self.forward(generated)
            next_token_logits = logits[:, -1, :] / temperature

            # Apply top-k filtering
            if top_k is not None:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('inf')

            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = -float('inf')

            # Sample next token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)

            # Stop if EOS token generated
            if stop_at_eos and (next_token == self.eos_idx).all():
                break

        return generated


def collate_variable_length_sequences(
    sequences: List[torch.Tensor],
    pad_idx: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate variable-length sequences into a padded batch.

    Args:
        sequences: List of 1D tensors of varying lengths
        pad_idx: Padding token index

    Returns:
        padded_batch: (batch, max_len) padded sequences
        attention_mask: (batch, max_len) 1 for real tokens, 0 for padding
    """
    padded_batch = pad_sequence(sequences, batch_first=True, padding_value=pad_idx)

    # Create attention mask
    attention_mask = (padded_batch != pad_idx).long()

    return padded_batch, attention_mask


class VariableLengthTrainer:
    """Training strategies for variable-length sequence prediction."""

    def __init__(
        self,
        model: ImprovedMusicalTransformer,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device

    def train_step_next_token_prediction(
        self,
        sequences: List[torch.Tensor],
    ) -> float:
        """
        Standard next-token prediction with variable-length sequences.

        Args:
            sequences: List of variable-length token sequences
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Collate sequences
        input_ids, attention_mask = collate_variable_length_sequences(
            sequences, pad_idx=self.model.pad_idx
        )
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        # Create targets (shifted input)
        target_ids = input_ids.clone()
        target_ids[attention_mask == 0] = -100  # Ignore padding in loss

        # Compute loss
        loss = self.model.compute_loss(input_ids, target_ids, attention_mask)

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train_step_prefix_suffix_prediction(
        self,
        sequences: List[torch.Tensor],
        prefix_ratio: float = 0.7,
    ) -> float:
        """
        Split sequences into prefix (context) and suffix (target).
        Train to predict suffix given prefix.

        Args:
            sequences: List of variable-length sequences
            prefix_ratio: Proportion of sequence to use as context
        """
        self.model.train()
        self.optimizer.zero_grad()

        total_loss = 0
        num_sequences = 0

        for seq in sequences:
            seq_len = seq.size(0)
            split_point = max(1, int(seq_len * prefix_ratio))

            # Split into prefix and suffix
            full_seq = seq.unsqueeze(0).to(self.device)

            # Create targets with masked prefix
            target_ids = full_seq.clone()
            target_ids[:, :split_point] = -100  # Don't compute loss on prefix

            # Compute loss
            loss = self.model.compute_loss(full_seq, target_ids)

            total_loss += loss.item()
            num_sequences += 1

            loss.backward()

        self.optimizer.step()

        return total_loss / num_sequences if num_sequences > 0 else 0

    def train_step_multi_horizon_prediction(
        self,
        sequences: List[torch.Tensor],
        prediction_lengths: List[int] = [1, 2, 4, 8, 16],
    ) -> float:
        """
        Train to predict multiple future horizons simultaneously.
        Helps model learn at different timescales.

        Args:
            sequences: List of sequences
            prediction_lengths: List of different prediction horizons
        """
        self.model.train()
        self.optimizer.zero_grad()

        total_loss = 0

        for pred_len in prediction_lengths:
            # Create shifted targets for different horizons
            horizon_loss = 0
            count = 0

            for seq in sequences:
                if seq.size(0) > pred_len:
                    # Input: predict token at position i+pred_len given positions up to i
                    input_seq = seq[:-pred_len].unsqueeze(0).to(self.device)
                    target_seq = seq.clone().unsqueeze(0).to(self.device)

                    # Mask early positions
                    target_seq[:, :-pred_len] = -100

                    loss = self.model.compute_loss(input_seq, target_seq)
                    horizon_loss += loss.item()
                    count += 1

                    loss.backward()

            if count > 0:
                total_loss += horizon_loss / count

        self.optimizer.step()

        return total_loss / len(prediction_lengths)
