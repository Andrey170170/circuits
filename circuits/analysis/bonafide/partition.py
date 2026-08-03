"""Discovery/holdout firewall and hierarchical fit weights."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum


class CorpusRole(StrEnum):
    DENSE_DISCOVERY = "dense_discovery"
    BROAD_DISCOVERY = "broad_discovery"
    BROAD_CONFIRMATORY_HOLDOUT = "broad_confirmatory_holdout"

    @property
    def expected_cluster_fit_eligible(self) -> bool:
        return self is not CorpusRole.BROAD_CONFIRMATORY_HOLDOUT


@dataclass(frozen=True, order=True)
class AnalysisTarget:
    source_artifact_id: str
    base_question_id: str
    response_id: str
    response_position: int
    corpus_role: CorpusRole
    cluster_fit_eligible: bool

    def __post_init__(self) -> None:
        for field_name in (
            "source_artifact_id",
            "base_question_id",
            "response_id",
        ):
            value = getattr(self, field_name)
            if not value:
                raise ValueError(f"{field_name} must be non-empty")
        if self.response_position < 0:
            raise ValueError("response_position must be nonnegative")
        validate_partition_contract(
            self.corpus_role,
            cluster_fit_eligible=self.cluster_fit_eligible,
        )


def validate_partition_contract(
    corpus_role: CorpusRole | str,
    *,
    cluster_fit_eligible: bool,
) -> CorpusRole:
    try:
        role = CorpusRole(corpus_role)
    except ValueError as error:
        raise ValueError(f"unsupported corpus role: {corpus_role!r}") from error
    if not isinstance(cluster_fit_eligible, bool):
        raise ValueError("cluster_fit_eligible must be boolean")
    if cluster_fit_eligible is not role.expected_cluster_fit_eligible:
        raise ValueError(
            f"partition contract mismatch for {role.value}: "
            f"cluster_fit_eligible={cluster_fit_eligible}"
        )
    return role


def assert_fit_partition(
    records: Iterable[AnalysisTarget],
) -> tuple[AnalysisTarget, ...]:
    materialized = tuple(records)
    holdout = [
        record.source_artifact_id
        for record in materialized
        if not record.cluster_fit_eligible
        or record.corpus_role is CorpusRole.BROAD_CONFIRMATORY_HOLDOUT
    ]
    if holdout:
        raise ValueError(
            "holdout firewall: fit input contains confirmatory targets "
            f"{sorted(holdout)[:5]}"
        )
    return materialized


def hierarchical_fit_weights(
    records: Iterable[AnalysisTarget],
) -> dict[str, float]:
    """Give equal mass to families, responses within family, and their targets."""

    fit_records = assert_fit_partition(records)
    if not fit_records:
        raise ValueError("fit weighting requires at least one discovery target")
    source_ids = [record.source_artifact_id for record in fit_records]
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("fit weighting received duplicate source_artifact_id values")

    families: dict[str, dict[str, list[AnalysisTarget]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in fit_records:
        families[record.base_question_id][record.response_id].append(record)

    family_mass = 1.0 / len(families)
    weights: dict[str, float] = {}
    for responses in families.values():
        response_mass = family_mass / len(responses)
        for targets in responses.values():
            target_mass = response_mass / len(targets)
            for target in targets:
                weights[target.source_artifact_id] = target_mass

    if not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-12):
        raise AssertionError("hierarchical weights do not sum to one")
    return dict(sorted(weights.items()))
