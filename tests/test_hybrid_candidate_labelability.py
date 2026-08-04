from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import circuits.analysis.bonafide.hybrid_candidate_labelability as module
from circuits.analysis.bonafide.hybrid_candidate_labelability import (
    _bootstrap,
    _target_effects,
    authorization_report,
    build_witness_inventory,
    generation_family_jackknife,
    run_hybrid_candidate_labelability,
)


def _target(case: str, family: str, partition: str) -> dict[str, object]:
    return {
        "case_id": case,
        "response_id": f"response-{case}",
        "base_question_id": family,
        "family_partition": partition,
        "model_top5_indices": [0, 1, 2, 3, 4],
        "observed_candidate_index": 0,
    }


def _occurrence(
    case: str,
    family: str,
    partition: str,
    basis: int,
    *,
    input_values: list[float] | None = None,
    candidate_values: list[float] | None = None,
) -> dict[str, object]:
    return {
        "target_id": case,
        "response_id": f"response-{case}",
        "family_id": family,
        "partition": partition,
        "basis_index": basis,
        "input_values": input_values or [1.0, float(basis + 1)],
        "input_support": [True, True],
        "paper_input_values": input_values or [1.0, float(basis + 1)],
        "raw_candidate_values": candidate_values or [1.0, float(basis + 1), 1, 1, 1],
        "paper_candidate_values": candidate_values or [1.0, float(basis + 1), 1, 1, 1],
        "occurrence_count": 1,
    }


def test_common_pair_pool_is_intersected_before_state_split() -> None:
    target = _target("case", "family", "audit")
    rows = [
        _occurrence("case", "family", "audit", index)
        for index in range(4)
    ]
    assignments = {
        "primary": np.asarray([0, 0, 1, 1]),
        "alternative": np.asarray([0, 1, 0, 1]),
    }
    result = _target_effects(rows, target, assignments, view="candidate")
    assert result["primary"] is not None
    assert result["alternative"] is not None
    assert result["primary"]["common_pair_pool_count"] == 6
    assert result["alternative"]["common_pair_pool_count"] == 6


def test_bootstrap_seed_and_intervals_are_deterministic() -> None:
    effects = {f"family-{index}": index / 10 for index in range(8)}
    left = _bootstrap(
        effects,
        protocol_sha256="a" * 64,
        partition="audit",
        role="primary",
        view="input",
    )
    right = _bootstrap(
        effects,
        protocol_sha256="a" * 64,
        partition="audit",
        role="primary",
        view="input",
    )
    assert left == right
    assert left["replicates"] == 10_000


def test_readiness_requires_joint_same_occurrence_and_freezes_8_4_4() -> None:
    targets: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    shapes = {"generation": (8, 4), "selection_scoring": (4, 2), "audit": (4, 2)}
    for partition, (target_count, family_count) in shapes.items():
        for index in range(target_count):
            case = f"{partition}-{index}"
            family = f"{partition}-family-{index % family_count}"
            targets.append(_target(case, family, partition))
            rows.append(_occurrence(case, family, partition, 0))
    assignments = {
        "primary": np.asarray([0]),
        "alternative": np.asarray([0]),
    }
    report = build_witness_inventory(
        rows, targets, assignments, protocol_sha256="b" * 64
    )
    cluster = report["states"]["primary"]["clusters"][0]
    assert cluster["ready"] is True
    assert len(cluster["joint_witnesses"]["generation"]["frozen_target_ids"]) == 8
    assert len(cluster["joint_witnesses"]["selection_scoring"]["frozen_target_ids"]) == 4
    assert len(cluster["joint_witnesses"]["audit"]["frozen_target_ids"]) == 4

    rows[0]["raw_candidate_values"] = [0.0] * 5
    rows[0]["paper_candidate_values"] = [0.0] * 5
    reduced = build_witness_inventory(
        rows, targets, assignments, protocol_sha256="b" * 64
    )
    assert (
        reduced["states"]["primary"]["clusters"][0]["joint_witnesses"]
        ["generation"]["target_count"]
        == 7
    )


def test_candidate_coherence_is_an_authorization_gate() -> None:
    view = {
        "effect": 0.2,
        "per_family_effect": {f"family-{index}": 0.1 for index in range(8)},
        "bootstrap": {"ci_95_lower": 0.01},
    }
    coherence = {
        "partitions": {
            partition: {
                role: {"input": dict(view), "candidate": dict(view)}
                for role in ("primary", "alternative")
            }
            for partition in ("selection_scoring", "audit")
        }
    }
    coherence["partitions"]["audit"]["primary"]["candidate"] = {
        **view,
        "effect": -0.1,
    }
    structural = {role: {"passed": True} for role in ("primary", "alternative")}
    jackknife = {role: {"passed": True} for role in ("primary", "alternative")}
    witness = {
        "states": {
            role: {
                "passed": True,
                "cluster_count": 64,
                "ready_cluster_count": 64,
                "ready_cluster_fraction": 1.0,
                "required_ready_cluster_count": 52,
            }
            for role in ("primary", "alternative")
        }
    }
    result = authorization_report(
        structural=structural,
        jackknife=jackknife,
        witness=witness,
        coherence=coherence,
    )
    assert result["states"]["primary"]["exploratory_labeling_authorized"] is False
    assert result["states"]["alternative"]["exploratory_labeling_authorized"] is True
    assert result["scientific_promotion_authorized"] is False


def test_jackknife_coverage_below_80_percent_fails_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = SimpleNamespace(
        target_rows=tuple(
            {"base_question_id": f"family-{index}"} for index in range(18)
        )
    )
    monkeypatch.setattr(module, "_jackknife_evidence", lambda *args, **kwargs: object())
    labels = np.asarray([0] * 7 + [-1] * 3)
    fit = SimpleNamespace(
        medoid_seed=17,
        seeds={17: SimpleNamespace(result=SimpleNamespace(labels=labels))},
    )
    monkeypatch.setattr(module, "_fit_one", lambda *args, **kwargs: fit)
    assignments = {
        "primary": np.zeros(10, dtype=np.int64),
        "alternative": np.zeros(10, dtype=np.int64),
    }
    result = generation_family_jackknife(bundle, assignments)
    assert result["primary"]["valid_replicate_count"] == 0
    assert result["primary"]["passed"] is False


def test_runner_refuses_overwrite_before_opening_sources(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        run_hybrid_candidate_labelability(
            input_root=tmp_path / "missing-input",
            fit_root=tmp_path / "missing-fit",
            output_root=output,
            repo_root=tmp_path / "missing-repo",
        )
