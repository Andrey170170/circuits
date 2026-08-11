from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from circuits.analysis.bonafide.canonical import file_sha256
from circuits.analysis.bonafide.hybrid_candidate_labeling import (
    _candidate_summary,
    _input_attribution_summary,
    _selected_assignments,
)
from circuits.labeling import profiles as profiles_module
from circuits.labeling.profiles import build_cluster_profile


def test_selected_assignments_join_exact_basis_identity(tmp_path: Path) -> None:
    rows = []
    for representation in ("raw_top5_plus_observed.v1", "control"):
        for index, cluster in enumerate((3, None)):
            rows.append(
                {
                    "representation": representation,
                    "affinity_mode": "full_positive",
                    "n_clusters": 64,
                    "seed": 17,
                    "is_medoid": representation == "raw_top5_plus_observed.v1",
                    "signed_basis_index": index,
                    "assigned": cluster is not None,
                    "cluster_id": cluster,
                }
            )
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "assignments.parquet")
    basis = [
        {
            "signed_basis_index": index,
            "model_id": "model",
            "model_revision": "revision",
            "layer": index + 1,
            "neuron_index": index + 10,
            "polarity": "+" if index == 0 else "-",
        }
        for index in range(2)
    ]
    result = _selected_assignments(
        fit_root=tmp_path,
        state={
            "representation": "raw_top5_plus_observed.v1",
            "affinity_mode": "full_positive",
            "n_clusters": 64,
            "seed": 17,
        },
        basis_rows=basis,
    )

    assert result[0]["layer"] == 1
    assert result[0]["neuron_index"] == 10
    assert result[0]["polarity"] == "+"
    assert result[0]["cluster_id"] == 3
    assert result[1]["assigned"] is False


def test_paper_candidate_summary_uses_model_top_five_from_width_six() -> None:
    candidates = [
        {
            "candidate_index": index,
            "full_distribution_rank": rank,
            "token_id": 100 + index,
            "token_text": f"t{index}",
            "logit": 10.0 - index,
            "probability": 0.1,
            "is_observed": index == 0,
        }
        for index, rank in enumerate((9, 1, 2, 3, 4, 5))
    ]
    summary = _candidate_summary(
        occurrence_rows=[
            {
                "basis_index": 4,
                "paper_candidate_values": [100.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                "raw_candidate_values": [100.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                "occurrence_count": 2,
            }
        ],
        member_indices={4},
        state={"representation": "paper_normalized_model_top5.v1"},
        target={
            "candidate_count": 6,
            "candidate_selection_json": json.dumps({"candidates": candidates}),
        },
    )

    assert summary["candidate_width"] == 5
    assert summary["signed_contribution_sum"] == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert [row["full_distribution_rank"] for row in summary["candidate_axis"]] == [
        1,
        2,
        3,
        4,
        5,
    ]
    assert summary["signed_cancellation_preserved"] is True


def test_fixed_union_input_summary_is_missing_aware_and_occurrence_weighted() -> None:
    summary = _input_attribution_summary(
        occurrence_rows=[
            {
                "basis_index": 4,
                "input_values": [3.0, 99.0, -2.0],
                "paper_input_values": [6.0, 99.0, -4.0],
                "input_support": [True, False, True],
                "occurrence_count": 2,
            },
            {
                "basis_index": 5,
                "input_values": [1.0, 7.0, 8.0],
                "paper_input_values": [2.0, 7.0, 16.0],
                "input_support": [True, True, False],
                "occurrence_count": 1,
            },
        ],
        member_indices={4},
        state={"representation": "paper_normalized_model_top5.v1"},
    )
    assert summary["representation"] == "paper_normalized_input_attribution"
    assert summary["signed_sum_by_source_token"] == [6.0, None, -4.0]
    assert summary["support_occurrence_count_by_source_token"] == [2, 0, 2]
    assert summary["mean_by_member_occurrence"] == [3.0, None, -2.0]


def _profile_exemplar(tmp_path: Path, summary: dict) -> dict:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    return {
        "trace_unit_id": "trace-b8a41-production-shape",
        "hybrid_target_id": "c2-r32-p3",
        "family_partition": "generation",
        "artifact_manifest_path": str(manifest_path),
        "artifact_manifest_sha256": file_sha256(manifest_path),
        "artifact_payload_sha256": "payload-sha",
        "fixed_union_input_summary": summary,
    }


def _valid_profile_summary() -> dict:
    return {
        "schema_version": "adag.bonafide.fixed-union-input-summary.v1",
        "source": "observed_candidate_fixed_union_refinement",
        "representation": "raw_input_attribution",
        "member_basis_count": 2,
        "member_occurrence_count": 3,
        "signed_sum_by_source_token": [3.0, None],
        "mean_by_member_occurrence": [1.0, None],
        "support_occurrence_count_by_source_token": [3, 0],
    }


def test_fixed_union_profile_survives_width_one_topology_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Production regression shape: cluster 37 target c2-r32-p3 has a 632-wide
    # fixed-union vector but 633 authenticated source tokens, requiring left padding.
    summary = _valid_profile_summary()
    summary["signed_sum_by_source_token"] = [3.0, *([None] * 631)]
    summary["mean_by_member_occurrence"] = [1.0, *([None] * 631)]
    summary["support_occurrence_count_by_source_token"] = [3, *([0] * 631)]
    artifact = SimpleNamespace(
        manifest={
            "data_sha256": "payload-sha",
            "artifact_id": "trace-b8a41-production-shape",
        },
        circuit_data=SimpleNamespace(
            cis=[list(range(633))],
            df_node=pd.DataFrame(
                [{"layer": 99, "neuron": 99, "attr_map": [9.0, 9.0, 9.0]}]
            ),
        ),
    )
    monkeypatch.setattr(profiles_module, "load_compact_trace", lambda _path: artifact)
    tokenizer = SimpleNamespace(decode=lambda ids, **_kwargs: f"<{ids[0]}>")
    profile = build_cluster_profile(
        _profile_exemplar(tmp_path, summary),
        cluster_members={(9841, 9865): [1]},
        source_tokenizer=tokenizer,
    )
    assert profile.matched_signed_basis_count == 2
    assert len(profile.record.token_ids) == 633
    assert len(profile.record.activations) == 633
    assert profile.record.activations[:3] == [0.0, 1.0, 0.0]


@pytest.mark.parametrize(
    "failure", ["no_support", "all_zero", "nonfinite", "malformed"]
)
def test_fixed_union_profile_rejects_invalid_summary(
    failure: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    artifact = SimpleNamespace(
        manifest={
            "data_sha256": "payload-sha",
            "artifact_id": "trace-b8a41-production-shape",
        },
        circuit_data=SimpleNamespace(cis=[[10, 11]], df_node=pd.DataFrame()),
    )
    monkeypatch.setattr(profiles_module, "load_compact_trace", lambda _path: artifact)
    summary = _valid_profile_summary()
    if failure == "no_support":
        summary.update(
            {
                "signed_sum_by_source_token": [None, None],
                "mean_by_member_occurrence": [None, None],
                "support_occurrence_count_by_source_token": [0, 0],
            }
        )
    elif failure == "all_zero":
        summary.update(
            {
                "signed_sum_by_source_token": [0.0, None],
                "mean_by_member_occurrence": [0.0, None],
            }
        )
    elif failure == "nonfinite":
        summary.update(
            {
                "signed_sum_by_source_token": [float("inf"), None],
                "mean_by_member_occurrence": [float("inf"), None],
            }
        )
    else:
        summary.pop("source")
    with pytest.raises(ValueError, match="frozen fixed-union"):
        build_cluster_profile(
            _profile_exemplar(tmp_path, summary),
            cluster_members={},
            source_tokenizer=SimpleNamespace(decode=lambda ids, **_kwargs: str(ids)),
        )
