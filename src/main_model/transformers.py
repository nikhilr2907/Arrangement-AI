"""
Basic transformer for next-token prediction with essential variable-length support.
"""

import torch
from torch import nn
from typing import Optional


class MusicalTransformer(nn.Module):
    """Basic transformer for next-token prediction with variable-length sequences."""

    def __init__(
        self,
        vocab_size: int,
        model_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        """
        Args:
            vocab_size: Total vocabulary size (including PAD token)
            model_dim: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
            pad_idx: Padding token index (default: 0)
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.model_dim = model_dim
        self.max_seq_len = max_seq_len
        self.pad_idx = pad_idx

        # Token embedding (padding_idx zeros out padding tokens)
        self.embedding = nn.Embedding(vocab_size, model_dim, padding_idx=pad_idx)

        # Positional encoding (learnable)
        self.positional_encoding = nn.Embedding(max_seq_len, model_dim)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output projection to vocabulary
        self.output_linear = nn.Linear(model_dim, vocab_size)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

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
        token_embeds = self.embedding(input_ids)
        pos_embeds = self.positional_encoding(positions)
        embedded = self.dropout(token_embeds + pos_embeds)

        # Create causal mask (prevents attending to future positions)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=device
        )

        # Create padding mask (True = masked/ignored)
        if attention_mask is None:
            padding_mask = (input_ids == self.pad_idx)
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
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss for next-token prediction.

        Args:
            input_ids: (batch, seq_len) - input sequence
            attention_mask: (batch, seq_len) - mask for valid positions
        """
        logits = self.forward(input_ids, attention_mask)

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        # Mask padding in labels
        if attention_mask is not None:
            shift_mask = attention_mask[:, 1:].contiguous()
            shift_labels = shift_labels.masked_fill(~shift_mask.bool(), -100)
        else:
            shift_labels = shift_labels.masked_fill(
                shift_labels == self.pad_idx, -100
            )

        # Compute loss (ignore_index=-100 skips padding)
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(
            shift_logits.view(-1, self.vocab_size),
            shift_labels.view(-1),
        )

        return loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Autoregressive generation.

        Args:
            input_ids: (batch, seq_len) - context tokens
            max_length: Maximum number of new tokens to generate
            temperature: Sampling temperature (higher = more random)

        Returns:
            generated: (batch, seq_len + generated_length)
        """
        self.eval()
        generated = input_ids.clone()

        for _ in range(max_length):
            if generated.size(1) >= self.max_seq_len:
                break

            # Get logits for all positions
            logits = self.forward(generated)

            # Get logits for last position only
            next_token_logits = logits[:, -1, :] / temperature

            # Sample next token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)

        return generated
