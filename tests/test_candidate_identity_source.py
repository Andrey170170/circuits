from __future__ import annotations

import gzip
import hashlib
import pickle
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from circuits.analysis.bonafide import candidate_identity_source as source
from circuits.analysis.bonafide.candidate_profiles import (
    TARGET_SCHEMA as C2_TARGET_SCHEMA,
)
from circuits.analysis.bonafide.canonical import file_sha256


def _target_row(case: str, partition: str) -> dict[str, object]:
    sentinel = "AUDIT_SENTINEL" if partition == "audit" else partition
    return {
        "case_id": case,
        "source_width1_artifact_id": f"source-{case}",
        "width1_artifact_id": f"width-{case}",
        "width1_payload_sha256": "1" * 64,
        "candidate_union_artifact_id": f"union-{case}",
        "candidate_union_payload_sha256": "2" * 64,
        "candidate_union_topology_sha256": "3" * 64,
        "base_question_id": f"family-{case}",
        "response_id": f"response-{case}",
        "phase_bin": 0,
        "response_position": 1,
        "family_partition": partition,
        "partition_hierarchical_weight": 1.0,
        "candidate_count": 5,
        "observed_token_id": 7,
        "observed_token_text": sentinel,
        "candidate_selection_json": sentinel,
        "example_json": sentinel,
        "width_signed_basis_count": 1,
        "candidate_signed_basis_count": 1,
        "zero_activation_width_occurrence_count": 0,
        "zero_activation_candidate_occurrence_count": 0,
        "candidate_activation_invariance_max_abs_deviation": 0.0,
        "candidate_activation_invariance_max_relative_deviation": 0.0,
        "candidate_activation_invariance_violation_count": 0,
        "candidate_activation_invariance_comparison_count": 1,
        "width_polarity_crosswalk_json": "{}",
        "candidate_polarity_crosswalk_json": "{}",
    }


def test_frozen_authorization_self_hash_and_contract() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    authorization = source._load_authorization(repo_root)
    assert authorization["authorization_sha256"] == source.AUTHORIZATION_SHA256
    assert authorization["projected_target_columns"] == list(
        source.PROJECTED_TARGET_COLUMNS
    )
    assert authorization["exposure_contract"] == source.EXPOSURE_CONTRACT


def test_structural_projection_never_reads_or_materializes_audit_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "targets.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                _target_row("g", "generation"),
                _target_row("s", "selection_scoring"),
                _target_row("a", "audit"),
            ],
            schema=C2_TARGET_SCHEMA,
        ),
        path,
    )
    manifest = {
        "files": [{"path": path.name, "sha256": file_sha256(path)}],
    }
    monkeypatch.setattr(
        source,
        "EXPECTED_TARGET_COUNTS",
        {"generation": 1, "selection_scoring": 1},
    )
    monkeypatch.setattr(source, "EXPECTED_AUDIT_TARGET_COUNT", 1)
    original = source.pq.read_table
    projected_columns: list[str] = []

    def _read_table(
        parquet_path: Path, *, columns: list[str]
    ) -> pa.Table:
        projected_columns.extend(columns)
        return original(parquet_path, columns=columns)

    monkeypatch.setattr(source.pq, "read_table", _read_table)
    rows, counts = source._project_executable_targets(tmp_path, manifest)

    assert projected_columns == list(source.PROJECTED_TARGET_COLUMNS)
    assert counts == {"generation": 1, "selection_scoring": 1, "audit": 1}
    assert [row["case_id"] for row in rows] == ["g", "s"]
    assert all(set(row) == set(source.PROJECTED_TARGET_COLUMNS) for row in rows)
    assert "AUDIT_SENTINEL" not in repr(rows)


def test_payload_hash_is_checked_before_deserialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = tmp_path / source.DATA_FILENAME
    with gzip.open(payload, "wb") as handle:
        pickle.dump({"trace": "sentinel"}, handle)

    def _forbidden_open(*args: object, **kwargs: object) -> object:
        raise AssertionError("payload was opened before its hash was accepted")

    monkeypatch.setattr(source.gzip, "open", _forbidden_open)
    with pytest.raises(ValueError, match="payload hash drift"):
        source.load_hash_bound_candidate_union_trace(
            payload,
            expected_sha256="0" * 64,
        )


def test_hash_bound_deserializer_does_not_open_adjacent_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = tmp_path / source.DATA_FILENAME
    trace = {"trace": "sentinel"}
    with gzip.open(payload, "wb") as handle:
        pickle.dump(trace, handle)
    (tmp_path / "manifest.json").write_text("AUDIT_MANIFEST_SENTINEL")
    (tmp_path / "metrics.json").write_text("AUDIT_METRICS_SENTINEL")
    validated: list[object] = []
    monkeypatch.setattr(
        source,
        "validate_candidate_union_trace",
        lambda candidate: validated.append(candidate),
    )

    loaded = source.load_hash_bound_candidate_union_trace(
        payload,
        expected_sha256=hashlib.sha256(payload.read_bytes()).hexdigest(),
    )

    assert loaded == trace
    assert validated == [trace]


def test_payload_resolution_is_limited_to_exact_selected_artifact(
    tmp_path: Path,
) -> None:
    family = tmp_path / source.CANDIDATE_UNION_TRACE_FAMILY_ID
    selected = family / "wave-a" / "selected" / source.DATA_FILENAME
    audit = family / "wave-b" / "audit-sentinel" / source.DATA_FILENAME
    selected.parent.mkdir(parents=True)
    audit.parent.mkdir(parents=True)
    selected.touch()
    audit.touch()

    assert source._resolve_payload(tmp_path, "selected") == selected.resolve()


def test_non_w64_baseline_is_rejected_before_assignment_extraction() -> None:
    baseline = SimpleNamespace(manifest={"chosen_cluster_count": 32})
    with pytest.raises(ValueError, match="not W64"):
        source._extract_w64_assignments(baseline, basis_count=1)


def test_selected_trace_identity_mismatch_fails_closed() -> None:
    trace = SimpleNamespace(
        source_width1_artifact_id="source",
        shared_response_position=3,
        topology_sha256="wrong-topology",
    )
    target = {
        "source_width1_artifact_id": "source",
        "response_position": 3,
        "candidate_union_topology_sha256": "expected-topology",
    }
    with pytest.raises(ValueError, match="trace binding drift"):
        source._validate_selected_trace_binding(
            trace,
            target,
            artifact_id="artifact",
        )


def test_published_parquet_hash_is_checked_before_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "published.parquet"
    path.write_bytes(b"not trusted parquet")

    def _forbidden_read(*args: object, **kwargs: object) -> object:
        raise AssertionError("Parquet was decoded before its hash was accepted")

    monkeypatch.setattr(source.pq, "read_table", _forbidden_read)
    with pytest.raises(ValueError, match="source file drift"):
        source._exact_table(
            path,
            source.TARGET_SCHEMA,
            {
                "size_bytes": path.stat().st_size,
                "sha256": "0" * 64,
                "row_count": 1,
            },
        )


def test_source_no_overwrite_fails_before_revision_or_input_access(
    tmp_path: Path,
) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError, match="refusing to replace"):
        source.build_candidate_identity_source(
            output_root=output,
            repo_root=tmp_path / "missing-repository",
        )
