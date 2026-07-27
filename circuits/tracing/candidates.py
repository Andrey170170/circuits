"""Contracts for tracing several candidate logits at one prediction position."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Literal

import torch

CandidatePolicyId = Literal[
    "observed_token",
    "model_top5",
    "observed_plus_top4_alternatives",
]
JointObjectiveId = Literal[
    "raw_logit_sum",
    "observed_vs_alternatives",
]

CANDIDATE_POLICY_VERSION = "1"
JOINT_OBJECTIVE_VERSION = "1"


@dataclass(frozen=True)
class CandidateLogit:
    """One deterministically ordered candidate at a shared prediction position.

    ``full_distribution_rank`` is one-based. ``candidate_index`` is zero-based
    and indexes contribution-profile vectors stored in top-k trace artifacts.
    """

    candidate_index: int
    full_distribution_rank: int
    token_id: int
    token_text: str
    logit: float
    probability: float
    is_observed: bool

    def to_dict(self) -> dict[str, int | float | str | bool]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateSelection:
    """A candidate vector selected from one full next-token distribution."""

    policy_id: CandidatePolicyId
    policy_version: str
    ordering_rule: str
    observed_token_id: int
    observed_token_text: str
    observed_token_rank: int
    candidates: tuple[CandidateLogit, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "ordering_rule": self.ordering_rule,
            "observed_token_id": self.observed_token_id,
            "observed_token_text": self.observed_token_text,
            "observed_token_rank": self.observed_token_rank,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


@dataclass(frozen=True)
class JointLogitObjective:
    """Named scalar topology objective over an ordered candidate vector."""

    objective_id: JointObjectiveId
    objective_version: str
    formula: str
    candidate_weights: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "objective_id": self.objective_id,
            "objective_version": self.objective_version,
            "formula": self.formula,
            "candidate_weights": list(self.candidate_weights),
        }


@dataclass(frozen=True)
class CandidateLogitAxis:
    """Validated CLJA input for candidates sharing one prediction position."""

    prediction_position: int
    token_ids_by_batch: tuple[tuple[int, ...], ...]
    objective_weights: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.prediction_position < 0:
            raise ValueError("candidate prediction_position cannot be negative")
        if not self.token_ids_by_batch:
            raise ValueError("candidate axis requires at least one batch item")
        widths = {len(row) for row in self.token_ids_by_batch}
        if len(widths) != 1 or not widths or next(iter(widths)) < 1:
            raise ValueError("candidate token rows must have one common positive width")
        width = next(iter(widths))
        if len(self.objective_weights) != width:
            raise ValueError(
                "candidate objective weight width must match candidate token width"
            )
        for batch_index, row in enumerate(self.token_ids_by_batch):
            if len(set(row)) != len(row):
                raise ValueError(
                    f"candidate token IDs must be unique within batch item {batch_index}"
                )
            if any(isinstance(token_id, bool) or token_id < 0 for token_id in row):
                raise ValueError("candidate token IDs must be non-negative integers")
        if any(not math.isfinite(weight) for weight in self.objective_weights):
            raise ValueError("candidate objective weights must be finite")

    @property
    def candidate_count(self) -> int:
        return len(self.objective_weights)


def select_candidate_logits(
    position_logits: torch.Tensor,
    *,
    observed_token_id: int,
    policy_id: CandidatePolicyId,
    candidate_count: int,
    decode_token: Callable[[int], str],
) -> CandidateSelection:
    """Select and score candidates with a deterministic full-vocabulary order.

    Stable descending-logit sorting preserves the original ascending token-ID
    order for exact ties, giving the required token-ID tie-break.
    """

    if position_logits.ndim != 1:
        raise ValueError("position_logits must be a one-dimensional vocabulary vector")
    vocab_size = int(position_logits.shape[0])
    if isinstance(observed_token_id, bool) or not 0 <= observed_token_id < vocab_size:
        raise ValueError("observed_token_id is outside the vocabulary")
    if isinstance(candidate_count, bool) or not 1 <= candidate_count <= vocab_size:
        raise ValueError("candidate_count must be between one and vocabulary size")
    if policy_id == "observed_token" and candidate_count != 1:
        raise ValueError("observed_token policy requires candidate_count=1")
    if policy_id == "observed_plus_top4_alternatives" and candidate_count != 5:
        raise ValueError(
            "observed_plus_top4_alternatives policy requires candidate_count=5"
        )
    if policy_id == "model_top5" and candidate_count != 5:
        raise ValueError("model_top5 policy requires candidate_count=5")

    ordered_ids = torch.argsort(
        position_logits, descending=True, stable=True
    ).detach()
    ordered_cpu = [int(token_id) for token_id in ordered_ids.cpu().tolist()]
    rank_by_token_id = {
        token_id: rank for rank, token_id in enumerate(ordered_cpu, start=1)
    }

    if policy_id == "observed_token":
        selected_ids = [observed_token_id]
    elif policy_id == "model_top5":
        selected_ids = ordered_cpu[:candidate_count]
    elif policy_id == "observed_plus_top4_alternatives":
        alternatives = [
            token_id for token_id in ordered_cpu if token_id != observed_token_id
        ]
        selected_ids = [observed_token_id, *alternatives[: candidate_count - 1]]
    else:
        raise ValueError(f"unsupported candidate policy: {policy_id!r}")

    probabilities = torch.softmax(position_logits.float(), dim=-1)
    candidates = tuple(
        CandidateLogit(
            candidate_index=index,
            full_distribution_rank=rank_by_token_id[token_id],
            token_id=token_id,
            token_text=decode_token(token_id),
            logit=float(position_logits[token_id].detach().float().cpu().item()),
            probability=float(probabilities[token_id].detach().cpu().item()),
            is_observed=token_id == observed_token_id,
        )
        for index, token_id in enumerate(selected_ids)
    )
    return CandidateSelection(
        policy_id=policy_id,
        policy_version=CANDIDATE_POLICY_VERSION,
        ordering_rule="descending_logit_then_ascending_token_id",
        observed_token_id=observed_token_id,
        observed_token_text=decode_token(observed_token_id),
        observed_token_rank=rank_by_token_id[observed_token_id],
        candidates=candidates,
    )


def build_joint_objective(
    objective_id: JointObjectiveId,
    candidates: Sequence[CandidateLogit],
) -> JointLogitObjective:
    """Build the named scalar objective used to select and prune topology."""

    if not candidates:
        raise ValueError("joint objective requires at least one candidate")
    if [candidate.candidate_index for candidate in candidates] != list(
        range(len(candidates))
    ):
        raise ValueError("candidate indices must be contiguous and zero-based")

    if objective_id == "raw_logit_sum":
        weights = (1.0,) * len(candidates)
        formula = " + ".join(
            f"logit[candidate_{index}]" for index in range(len(candidates))
        )
    elif objective_id == "observed_vs_alternatives":
        observed_indices = [
            candidate.candidate_index
            for candidate in candidates
            if candidate.is_observed
        ]
        if len(observed_indices) != 1 or len(candidates) < 2:
            raise ValueError(
                "observed_vs_alternatives requires exactly one observed candidate "
                "and at least one alternative"
            )
        alternative_weight = -1.0 / (len(candidates) - 1)
        weights = tuple(
            1.0 if index == observed_indices[0] else alternative_weight
            for index in range(len(candidates))
        )
        formula = (
            f"logit[candidate_{observed_indices[0]}] - "
            "mean(logit[alternative_candidates])"
        )
    else:
        raise ValueError(f"unsupported joint objective: {objective_id!r}")

    return JointLogitObjective(
        objective_id=objective_id,
        objective_version=JOINT_OBJECTIVE_VERSION,
        formula=formula,
        candidate_weights=weights,
    )


def reduce_candidate_contributions(
    contributions: torch.Tensor,
    objective_weights: Sequence[float] | torch.Tensor,
) -> torch.Tensor:
    """Reduce a raw candidate vector to a scalar joint-objective contribution."""

    if contributions.ndim < 1:
        raise ValueError("candidate contributions require at least one dimension")
    if contributions.shape[-1] != len(objective_weights):
        raise ValueError(
            "candidate contribution width must match objective weight width"
        )
    if isinstance(objective_weights, torch.Tensor):
        weights = objective_weights.to(
            device=contributions.device, dtype=contributions.dtype
        )
    else:
        weights = contributions.new_tensor(objective_weights)
    return (contributions * weights).sum(dim=-1, keepdim=True)
