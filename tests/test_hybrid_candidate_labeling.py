from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from circuits.analysis.bonafide.hybrid_candidate_labeling import (
    _candidate_summary,
    _selected_assignments,
)


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
