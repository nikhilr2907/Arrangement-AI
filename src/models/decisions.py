from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class DecisionLogits:
    change: Tensor
    destinations: Tensor


def action_log_probabilities(
    logits: DecisionLogits, keep_valid: Tensor, destination_valid: Tensor
) -> Tensor:
    # Input: gate float[B], destination float[B,J], legal masks bool[B], bool[B,J].
    # Output: normalised log-probabilities float[B,1+J], KEEP at column zero.
    has_change = destination_valid.any(dim=-1)
    if torch.any(~keep_valid & ~has_change):
        raise ValueError("Every decision requires at least one legal action")
    scores = logits.destinations.masked_fill(~destination_valid, -torch.inf)
    safe = scores.masked_fill(~has_change.unsqueeze(-1), 0)
    destinations = F.log_softmax(safe, dim=-1).masked_fill(~destination_valid, -torch.inf)
    gate = torch.where(keep_valid, F.logsigmoid(logits.change), 0)
    keep = torch.where(has_change, F.logsigmoid(-logits.change), 0).masked_fill(
        ~keep_valid, -torch.inf
    )
    return torch.cat((keep.unsqueeze(-1), destinations + gate.unsqueeze(-1)), dim=-1)


@dataclass(frozen=True)
class LossBreakdown:
    total: Tensor
    gate_sum: Tensor
    destination_sum: Tensor
    gate_count: int
    destination_count: int


class ArrangementLoss(nn.Module):
    def __init__(self, destination_weight: float = 1.0) -> None:
        super().__init__()
        self.destination_weight = destination_weight

    # Input: raw DecisionLogits, legal masks and independently supplied supervised targets.
    # Output: scalar loss plus component sums/counts for correctly weighted logging.
    def forward(
        self,
        logits: DecisionLogits,
        keep_valid: Tensor,
        destination_valid: Tensor,
        change_target: Tensor,
        gate_valid: Tensor,
        acceptable: Tensor,
        labelled_change: Tensor,
    ) -> LossBreakdown:
        zero = logits.change.sum() * 0 + logits.destinations.sum() * 0
        gate_mask = gate_valid & keep_valid & destination_valid.any(-1)
        gate_count = int(gate_mask.sum().item())
        gate_sum = (
            F.binary_cross_entropy_with_logits(
                logits.change[gate_mask], change_target[gate_mask], reduction="sum"
            )
            if gate_count
            else zero
        )
        change_mask = labelled_change & (change_target > 0.5)
        destination_count = int(change_mask.sum().item())
        destination_sum = zero
        if destination_count:
            valid = destination_valid[change_mask]
            accepted = acceptable[change_mask] & valid
            if not accepted.any(-1).all():
                raise ValueError("Supervised destination is outside the legal candidate set")
            selected = logits.destinations[change_mask].masked_fill(~valid, -torch.inf)
            log_probs = F.log_softmax(selected, dim=-1)
            destination_sum = -torch.logsumexp(
                log_probs.masked_fill(~accepted, -torch.inf), dim=-1
            ).sum()
        total = gate_sum / max(gate_count, 1) + self.destination_weight * destination_sum / max(
            destination_count, 1
        )
        return LossBreakdown(total, gate_sum, destination_sum, gate_count, destination_count)
