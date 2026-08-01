from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from circuits.tracing.attribution import _get_grad_attributions_from_logits
from circuits.tracing.candidates import (
    CandidateLogit,
    CandidateLogitAxis,
    build_joint_objective,
    reduce_candidate_contributions,
    select_candidate_logits,
)


def _decode(token_id: int) -> str:
    return f"token-{token_id}"


def test_model_top5_uses_token_id_as_stable_tie_break() -> None:
    logits = torch.tensor([0.0, 3.0, 3.0, 2.0, 4.0, -1.0])

    selection = select_candidate_logits(
        logits,
        observed_token_id=3,
        policy_id="model_top5",
        candidate_count=5,
        decode_token=_decode,
    )

    assert [candidate.token_id for candidate in selection.candidates] == [4, 1, 2, 3, 0]
    assert [candidate.candidate_index for candidate in selection.candidates] == list(
        range(5)
    )
    assert selection.observed_token_rank == 4
    assert selection.candidates[3].is_observed is True
    assert selection.ordering_rule == "descending_logit_then_ascending_token_id"


def test_observed_plus_alternatives_keeps_observed_at_index_zero() -> None:
    logits = torch.tensor([9.0, 8.0, 7.0, 6.0, 5.0, -4.0])

    selection = select_candidate_logits(
        logits,
        observed_token_id=5,
        policy_id="observed_plus_top4_alternatives",
        candidate_count=5,
        decode_token=_decode,
    )

    assert [candidate.token_id for candidate in selection.candidates] == [5, 0, 1, 2, 3]
    assert selection.observed_token_rank == 6
    assert [candidate.is_observed for candidate in selection.candidates] == [
        True,
        False,
        False,
        False,
        False,
    ]
    assert math.isclose(
        sum(candidate.probability for candidate in selection.candidates),
        torch.softmax(logits, dim=-1)[[5, 0, 1, 2, 3]].sum().item(),
        rel_tol=1e-6,
    )


@pytest.mark.parametrize(
    ("observed_token_id", "expected_ids", "expected_count"),
    [
        (1, [1, 0, 2, 3, 4], 5),
        (6, [6, 0, 1, 2, 3, 4], 6),
    ],
)
def test_model_top5_plus_observed_realizes_width_five_or_six(
    observed_token_id: int,
    expected_ids: list[int],
    expected_count: int,
) -> None:
    logits = torch.tensor([9.0, 8.0, 7.0, 6.0, 5.0, 4.0, -4.0])

    selection = select_candidate_logits(
        logits,
        observed_token_id=observed_token_id,
        policy_id="model_top5_plus_observed",
        candidate_count=6,
        decode_token=_decode,
    )

    assert [candidate.token_id for candidate in selection.candidates] == expected_ids
    assert len(selection.candidates) == expected_count
    assert selection.candidates[0].is_observed is True
    assert [
        candidate.full_distribution_rank for candidate in selection.candidates[1:]
    ] == [rank for rank in range(1, 6) if rank != selection.observed_token_rank]
    assert selection.ordering_rule == (
        "observed_first_then_model_top5_descending_logit_then_ascending_token_id"
    )


def test_observed_token_policy_is_the_explicit_k1_compatibility_mode() -> None:
    selection = select_candidate_logits(
        torch.tensor([3.0, 1.0, 2.0]),
        observed_token_id=2,
        policy_id="observed_token",
        candidate_count=1,
        decode_token=_decode,
    )

    assert [candidate.token_id for candidate in selection.candidates] == [2]
    assert selection.candidates[0].full_distribution_rank == 2
    with pytest.raises(ValueError, match="candidate_count=1"):
        select_candidate_logits(
            torch.tensor([3.0, 1.0, 2.0, 0.0, -1.0]),
            observed_token_id=2,
            policy_id="observed_token",
            candidate_count=5,
            decode_token=_decode,
        )


def test_specified_token_policy_supports_c0_independent_alternatives() -> None:
    selection = select_candidate_logits(
        torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0]),
        observed_token_id=0,
        policy_id="specified_token",
        candidate_count=1,
        specified_token_id=3,
        decode_token=_decode,
    )

    assert [candidate.token_id for candidate in selection.candidates] == [3]
    assert selection.candidates[0].full_distribution_rank == 4
    assert selection.candidates[0].is_observed is False
    assert selection.observed_token_id == 0
    with pytest.raises(ValueError, match="specified_token_id"):
        select_candidate_logits(
            torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0]),
            observed_token_id=0,
            policy_id="specified_token",
            candidate_count=1,
            decode_token=_decode,
        )


def test_joint_objectives_define_explicit_candidate_weights() -> None:
    candidates = tuple(
        CandidateLogit(
            candidate_index=index,
            full_distribution_rank=index + 1,
            token_id=100 + index,
            token_text=f"candidate-{index}",
            logit=float(5 - index),
            probability=0.1,
            is_observed=index == 0,
        )
        for index in range(5)
    )

    raw = build_joint_objective("raw_logit_sum", candidates)
    contrast = build_joint_objective("observed_vs_alternatives", candidates)

    assert raw.candidate_weights == (1.0, 1.0, 1.0, 1.0, 1.0)
    assert raw.percentage_threshold_reference == "signed_joint_objective"
    assert contrast.candidate_weights == (1.0, -0.25, -0.25, -0.25, -0.25)
    assert contrast.percentage_threshold_reference == (
        "absolute_joint_objective_magnitude"
    )
    assert sum(contrast.candidate_weights) == 0.0
    assert contrast.formula == (
        "logit[candidate_0] - mean(logit[alternative_candidates])"
    )


def test_contrastive_objective_rejects_missing_observed_candidate() -> None:
    candidates = tuple(
        CandidateLogit(
            candidate_index=index,
            full_distribution_rank=index + 1,
            token_id=index,
            token_text=str(index),
            logit=float(index),
            probability=0.1,
            is_observed=False,
        )
        for index in range(5)
    )

    with pytest.raises(ValueError, match="exactly one observed candidate"):
        build_joint_objective("observed_vs_alternatives", candidates)


def test_candidate_axis_rejects_duplicates_and_width_mismatch() -> None:
    with pytest.raises(ValueError, match="unique"):
        CandidateLogitAxis(
            prediction_position=7,
            token_ids_by_batch=((1, 1),),
            objective_weights=(1.0, 1.0),
            use_absolute_goal_for_percentage_threshold=False,
        )
    with pytest.raises(ValueError, match="weight width"):
        CandidateLogitAxis(
            prediction_position=7,
            token_ids_by_batch=((1, 2),),
            objective_weights=(1.0,),
            use_absolute_goal_for_percentage_threshold=False,
        )


def test_opposing_candidate_effects_remain_visible_when_raw_sum_cancels() -> None:
    contributions = torch.tensor([[1.5, -1.5]])

    raw_sum = reduce_candidate_contributions(contributions, (1.0, 1.0))
    contrast = reduce_candidate_contributions(contributions, (1.0, -1.0))

    assert contributions.norm().item() > 0
    assert raw_sum.tolist() == [[0.0]]
    assert contrast.tolist() == [[3.0]]


class _TinyLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.down_proj = nn.Linear(1, 1, bias=False)
        self.mlp.down_proj.weight.data.fill_(1.0)


class _TinyCandidateModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(2, 1)
        self.model.embed_tokens.weight.data.copy_(torch.tensor([[2.0], [1.0]]))
        self.model.layers = nn.ModuleList([_TinyLayer()])
        self.lm_head = nn.Linear(1, 2, bias=False)
        self.lm_head.weight.data.copy_(torch.tensor([[1.0], [-1.0]]))

    def forward(self, *, inputs_embeds, attention_mask):
        del attention_mask
        hidden = self.model.layers[0].mlp.down_proj(inputs_embeds)
        return type("Output", (), {"logits": self.lm_head(hidden)})()


def test_initial_attribution_uses_named_objective_weights() -> None:
    model = _TinyCandidateModel()
    common = {
        "model": model,
        "input_ids": torch.tensor([[0]]),
        "keep_tokens": [0],
        "focus_positions": [0, 0],
        "focus_logits": [[0, 1]],
        "attention_masks": [[1]],
        "disable_stop_grad": True,
    }

    raw = _get_grad_attributions_from_logits(
        **common,
        objective_weights=(1.0, 1.0),
    )
    contrast = _get_grad_attributions_from_logits(
        **common,
        objective_weights=(1.0, -1.0),
    )

    raw_mlp, raw_embed, raw_goal, *_ = raw
    contrast_mlp, contrast_embed, contrast_goal, *_ = contrast
    assert raw_goal.tolist() == [0.0]
    assert torch.count_nonzero(raw_mlp) == 0
    assert torch.count_nonzero(raw_embed) == 0
    assert contrast_goal.tolist() == [4.0]
    assert contrast_mlp.flatten().tolist() == [4.0]
    assert contrast_embed.flatten().tolist() == [4.0]
