"""
Vector Quantized Variational Autoencoder.

Maps 25-dim bar feature vectors to discrete codebook indices.
One index per bar = one token in the arrangement sequence.

Architecture:
    Encoder: MLP  input_dim → hidden → emb_dim
    Codebook: NearestEmbed  (codebook_size entries of emb_dim each)
    Decoder: MLP  emb_dim → hidden → input_dim
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

from src.model_preprocessing.quantization.nearest_embed import NearestEmbed


class VQ_VAE(nn.Module):

    def __init__(
        self,
        input_dim: int = 25,
        hidden_dim: int = 128,
        codebook_size: int = 64,
        emb_dim: int = 32,
        vq_coef: float = 0.2,
        commit_coef: float = 0.4,
    ):
        """
        Args:
            input_dim:      Feature vector size (25 = 12 mel chroma + 12 harm chroma + 1 onset)
            hidden_dim:     MLP hidden layer width
            codebook_size:  Number of discrete codes (vocabulary size before special tokens)
            emb_dim:        Codebook entry dimensionality (latent bottleneck)
            vq_coef:        Weight for VQ loss term
            commit_coef:    Weight for commitment loss term
        """
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.codebook_size = codebook_size
        self.vq_coef = vq_coef
        self.commit_coef = commit_coef

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, emb_dim),
        )

        self.codebook = NearestEmbed(codebook_size, emb_dim)

        self.decoder = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    # ------------------------------------------------------------------
    # Core forward passes
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, input_dim) → z_e: (batch, emb_dim)"""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: (batch, emb_dim) → x_recon: (batch, input_dim)"""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass used during training.

        Returns:
            x_recon:  (batch, input_dim)  reconstructed input
            z_e:      (batch, emb_dim)    encoder output (pre-quantization)
            emb:      (batch, emb_dim)    quantized vector (straight-through, detached codebook)
        """
        z_e = self.encode(x)
        z_q, _ = self.codebook(z_e, weight_sg=True)   # straight-through: grad flows to encoder
        emb, _ = self.codebook(z_e.detach())           # for VQ loss: codebook learns toward z_e
        x_recon = self.decode(z_q)
        return x_recon, z_e, emb

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def loss(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        z_e: torch.Tensor,
        emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        VQ-VAE loss = reconstruction + VQ + commitment.

        Returns:
            total_loss: scalar tensor
            breakdown:  dict with individual loss values for logging
        """
        recon_loss  = F.mse_loss(x_recon, x)
        vq_loss     = F.mse_loss(emb, z_e.detach())
        commit_loss = F.mse_loss(z_e, emb.detach())
        total       = recon_loss + self.vq_coef * vq_loss + self.commit_coef * commit_loss

        return total, {
            "recon":  recon_loss.item(),
            "vq":     vq_loss.item(),
            "commit": commit_loss.item(),
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_to_indices(self, x: torch.Tensor, batch_size: int = 512) -> torch.Tensor:
        """
        Encode a batch of feature vectors to integer codebook indices.

        Args:
            x:          (N, input_dim) feature vectors
            batch_size: chunk size to avoid OOM on large datasets

        Returns:
            indices: (N,) long tensor, values in [0, codebook_size)
        """
        self.eval()
        all_indices = []
        for i in range(0, len(x), batch_size):
            batch = x[i : i + batch_size]
            z_e = self.encode(batch)
            _, idx = self.codebook(z_e)
            all_indices.append(idx)
        return torch.cat(all_indices)

    def get_codebook_vectors(self) -> torch.Tensor:
        """Returns (codebook_size, emb_dim) codebook weight matrix."""
        return self.codebook.weight.detach().t()


class VQ_VAE_TRAINER:
    """Training wrapper for VQ_VAE."""

    def __init__(self, model: VQ_VAE, lr: float = 1e-3, device: torch.device = None):
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def step(self, x: torch.Tensor) -> Dict[str, float]:
        """Single training step. Returns loss breakdown."""
        self.model.train()
        x = x.to(self.device)

        x_recon, z_e, emb = self.model(x)
        loss, breakdown = self.model.loss(x, x_recon, z_e, emb)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        breakdown["total"] = loss.item()
        return breakdown

    def save(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)

    def load(self, path: str) -> None:
        self.model.load_state_dict(torch.load(path, map_location=self.device))
