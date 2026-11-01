from dataclasses import fields, replace

import pytest
import torch

from src.config import ModelConfig
from src.contracts import ActionKind, PlaybackAction, PreparedEpisode
from src.data.collate import DecisionCollator
from src.data.dataset import ArrangementDecisionDataset
from src.features.cache import ClipFeatures
from src.models.arrangement import ArrangementModel
from src.models.decisions import ArrangementLoss, DecisionLogits, action_log_probabilities
from src.models.encoders import SetPool


@pytest.mark.parametrize(
    "keep,candidates", [(True, [True, True]), (False, [True, False]), (True, [False, False])]
)
def test_action_distribution_normalises(keep: bool, candidates: list[bool]) -> None:
    logits = DecisionLogits(torch.tensor([2.0]), torch.tensor([[1.0, -1.0]]))
    distribution = action_log_probabilities(
        logits, torch.tensor([keep]), torch.tensor([candidates])
    ).exp()
    assert distribution.sum().item() == pytest.approx(1)
    assert torch.isfinite(distribution).all()
    assert (distribution[0, 1:][~torch.tensor(candidates)] == 0).all()


def test_no_legal_action_raises() -> None:
    with pytest.raises(ValueError, match="legal action"):
        action_log_probabilities(
            DecisionLogits(torch.zeros(1), torch.zeros(1, 1)),
            torch.tensor([False]),
            torch.tensor([[False]]),
        )


def test_loss_ignores_unknown_and_keep_destinations() -> None:
    gate = torch.zeros(2, requires_grad=True)
    destinations = torch.zeros(2, 2, requires_grad=True)
    loss = ArrangementLoss()(
        DecisionLogits(gate, destinations),
        torch.ones(2, dtype=torch.bool),
        torch.ones(2, 2, dtype=torch.bool),
        torch.tensor([0.0, 1.0]),
        torch.tensor([True, False]),
        torch.tensor([[False, False], [False, True]]),
        torch.tensor([False, True]),
    )
    loss.total.backward()
    assert destinations.grad is not None and torch.equal(destinations.grad[0], torch.zeros(2))
    assert gate.grad is not None and gate.grad[1] == 0


def test_empty_loss_masks_are_differentiable() -> None:
    gate = torch.zeros(1, requires_grad=True)
    destination = torch.zeros(1, 1, requires_grad=True)
    loss = ArrangementLoss()(
        DecisionLogits(gate, destination),
        torch.tensor([True]),
        torch.tensor([[False]]),
        torch.zeros(1),
        torch.tensor([False]),
        torch.tensor([[False]]),
        torch.tensor([False]),
    )
    loss.total.backward()
    assert loss.total.item() == 0 and gate.grad is not None


def test_future_targets_do_not_change_inputs_or_logits(
    episode: PreparedEpisode,
    features: dict[tuple[str, str], ClipFeatures],
    model_config: ModelConfig,
) -> None:
    changed = replace(
        episode,
        actions=episode.actions[:3]
        + ((PlaybackAction(ActionKind.SILENCE), PlaybackAction(ActionKind.KEEP)),),
    )
    collator = DecisionCollator(features, model_config)
    a = collator([ArrangementDecisionDataset([episode], 3)[4]]).inputs
    b = collator([ArrangementDecisionDataset([changed], 3)[4]]).inputs
    for field in fields(a):
        assert torch.equal(getattr(a, field.name), getattr(b, field.name))
    model = ArrangementModel(model_config).eval()
    assert torch.equal(model(a).change, model(b).change)


def test_undecided_current_target_is_hidden(
    episode: PreparedEpisode,
    features: dict[tuple[str, str], ClipFeatures],
    model_config: ModelConfig,
) -> None:
    dataset = ArrangementDecisionDataset([episode], 3)
    first = dataset[4]
    other = 1 - first.channel_index
    row = list(episode.actions[2])
    row[other] = (
        PlaybackAction(ActionKind.START, 0)
        if row[other].kind == ActionKind.SILENCE
        else PlaybackAction(ActionKind.SILENCE)
    )
    last = list(episode.actions[3])
    last[other] = PlaybackAction(ActionKind.SILENCE)
    changed = replace(episode, actions=episode.actions[:2] + (tuple(row), tuple(last)))
    collator = DecisionCollator(features, model_config)
    a, b = collator([first]).inputs, collator([ArrangementDecisionDataset([changed], 3)[4]]).inputs
    assert torch.equal(a.current, b.current) and torch.equal(a.current_status, b.current_status)


def test_candidate_permutation_equivariance(
    episode: PreparedEpisode,
    features: dict[tuple[str, str], ClipFeatures],
    model_config: ModelConfig,
) -> None:
    inputs = DecisionCollator(features, model_config)(
        [ArrangementDecisionDataset([episode], 3)[0]]
    ).inputs
    permutation = torch.arange(inputs.destination.size(1) - 1, -1, -1)
    permuted = replace(
        inputs,
        destination=inputs.destination[:, permutation],
        destination_meta=inputs.destination_meta[:, permutation],
        destination_valid=inputs.destination_valid[:, permutation],
    )
    model = ArrangementModel(model_config).eval()
    assert torch.allclose(
        model(inputs).destinations[:, permutation], model(permuted).destinations, atol=1e-6
    )


def test_current_set_pool_ignores_order() -> None:
    pool = SetPool(16, 2).eval()
    values = torch.randn(2, 4, 16)
    valid = torch.ones(2, 4, dtype=torch.bool)
    assert torch.allclose(pool(values, valid), pool(values.flip(1), valid), atol=1e-6)


def test_small_batch_and_empty_context(
    episode: PreparedEpisode,
    features: dict[tuple[str, str], ClipFeatures],
    model_config: ModelConfig,
) -> None:
    request = replace(episode.request, channels=episode.request.channels[:1])
    single = replace(episode, request=request, actions=tuple((row[0],) for row in episode.actions))
    batch = DecisionCollator(features, model_config)([ArrangementDecisionDataset([single], 3)[0]])
    output = ArrangementModel(model_config)(batch.inputs)
    assert output.change.shape == (1,) and output.destinations.ndim == 2
    assert torch.isfinite(output.destinations).all()
