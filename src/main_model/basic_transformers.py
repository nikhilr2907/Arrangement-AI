import torch
from torch import nn
import random




class MusicalTransformer(nn.Module):
    def __init__(self, vocab_size, model_dim, num_heads, num_layers, max_seq_len=120):
        super(MusicalTransformer, self).__init__()
        # For tokenized inputs: vocab_size = number of unique tokens
        self.embedding = nn.Embedding(vocab_size, model_dim)
        self.positional_encoding = nn.Parameter(torch.randn(max_seq_len, model_dim))

        decoder_layer = nn.TransformerDecoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Project back to vocabulary for next-token prediction
        self.output_linear = nn.Linear(model_dim, vocab_size)
        self.max_seq_len = max_seq_len

    def forward(self, src, tgt):
        # src and tgt are token IDs: (batch, src_len) and (batch, tgt_len)
        # Concatenate along sequence dimension
        combined = torch.cat([src, tgt], dim=1)  # Shape: (batch, src_len + tgt_len)
        # Embed tokens and add positional encoding
        embedded = self.embedding(combined)  # Shape: (batch, seq_len, model_dim)
        seq_len = embedded.size(1)
        embedded = embedded + self.positional_encoding[:seq_len, :]
        # Create causal mask (prevents attending to future tokens)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=embedded.device)
        # For decoder-only, we use the embedded sequence as both input and memory
        output = self.transformer_decoder(embedded, embedded, tgt_mask=causal_mask)
        # Project back to output dimension
        output = self.output_linear(output)

        return output

    def autoregressive_generate(self, src, max_length):
        # src: (batch, src_len)
        generated = src

        for _ in range(max_length):
            output = self.forward(generated, generated)
            next_token_logits = output[:, -1, :]  # Get logits for the last time step
            next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # Greedy decoding
            generated = torch.cat([generated, next_tokens], dim=1)

        return generated
    
    def train_step_tf(self, src, tgt, criterion, optimizer):
        """Teacher Forcing: Use ground truth at each step
        """
        self.train()
        optimizer.zero_grad()

        # Predict all positions using ground truth input
        output = self.forward(src, tgt[:, :-1])  # Input all but last token
        # output shape: (batch, src_len + tgt_len - 1, vocab_size)

        # We only compute loss on the tgt portion
        # Take predictions for positions corresponding to tgt
        tgt_predictions = output[:, -tgt.size(1)+1:, :]  # (batch, tgt_len-1, vocab_size)

        # Compare with actual next tokens
        loss = criterion(
            tgt_predictions.reshape(-1, tgt_predictions.size(-1)),  # (batch*(tgt_len-1), vocab_size)
            tgt[:, 1:].reshape(-1)  # (batch*(tgt_len-1),)
        )

        loss.backward()
        optimizer.step()
        return loss.item()

    def train_step_scheduled_sampling(self, src, tgt, criterion, optimizer, sampling_prob=0.5):
        """Scheduled Sampling: Mix teacher forcing and autoregressive generation

        At each step, use ground truth with probability (1 - sampling_prob),
        or use model's own prediction with probability sampling_prob.

        Args:
            src: Context tokens (batch, src_len)
            tgt: Target tokens (batch, tgt_len)
            sampling_prob: Probability of using model's prediction (0=pure teacher forcing, 1=pure autoregressive)
        """
        self.train()
        optimizer.zero_grad()

        tgt_len = tgt.size(1)

        # Start with src
        current_input_src = src
        current_input_tgt = tgt[:, :1]  # First token of target

        all_logits = []

        for i in range(1, tgt_len):
            # Get predictions
            output = self.forward(current_input_src, current_input_tgt)
            next_logits = output[:, -1:, :]  # Last position logits: (batch, 1, vocab_size)
            all_logits.append(next_logits)

            # Decide: use ground truth or model prediction?
            if torch.rand(1).item() > sampling_prob:
                # Use ground truth (teacher forcing)
                next_token = tgt[:, i:i+1]
            else:
                # Use model's prediction
                next_token = torch.argmax(next_logits, dim=-1)  # (batch, 1)

            # Append to input for next step
            current_input_tgt = torch.cat([current_input_tgt, next_token], dim=1)

        # Compute loss
        all_logits = torch.cat(all_logits, dim=1)  # (batch, tgt_len-1, vocab_size)
        loss = criterion(
            all_logits.reshape(-1, all_logits.size(-1)),
            tgt[:, 1:].reshape(-1)
        )

        loss.backward()
        optimizer.step()
        return loss.item()
    
    def inference_testing(self, src, max_length):
        """Generate sequence given src context for testing/inference."""
        self.eval()
        with torch.no_grad():
            generated = self.autoregressive_generate(src, max_length)
        return generated
