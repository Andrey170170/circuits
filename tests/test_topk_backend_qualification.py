from __future__ import annotations

import hashlib
import json
from copy import deepcopy

import pytest
from circuits.tracing.artifact import save_topk_compact_trace
from circuits.tracing.backend_qualification import (
    CUDA_ALLOCATOR_AB_IDENTITY_PATHS,
    EMBEDDING_EDGE_AB_IDENTITY_PATHS,
    TOLERANCE_GROUPS,
    NumericTolerance,
    compare_attention_backend_artifacts,
    compare_execution_artifacts,
)
from scripts.bonafide.topk_backend_qualification import (
    build_parser,
    comparison_options,
    save_qualification_report,
)
from tests.test_teacher_forced_trace import _topk_trace


def _manifest(
    backend: str,
    *,
    source_id: str = "source-width1",
    contribution_execution: str | None = "full_graph_v1",
    allocator_policy: str | None = None,
    embedding_edge_materialization: str | None = None,
    code_revision: str | None = None,
) -> dict:
    manifest = {
        "source_width1_artifact_id": source_id,
        "source_width1_manifest_sha256": "a" * 64,
        "source_target_selection": {"response_token_positions": [0]},
        "bonafide_example": {
            "example_id": "row-1",
            "prompt": "prompt",
            "response": "response",
        },
        "model_revision": "revision-1",
        "gpu": {
            "name": "NVIDIA A100 80GB PCIe",
            "total_memory_bytes": 80_000_000_000,
            "compute_capability": [8, 0],
        },
        "runtime_environment": {
            "python": "3.12.12",
            "gpu_runtime": {"devices": [{"name": "NVIDIA A100 80GB PCIe"}]},
        },
        "artifact_identity": {
            "source_width1_artifact_id": source_id,
            "source_width1_manifest_sha256": "a" * 64,
            "source_target_selection": {"response_token_positions": [0]},
            "trace_family": {"trace_family_id": "bonafide.topk-position.v1"},
            "model": {
                "model_id": "fake/model",
                "revision": "revision-1",
                "device": "cuda:0",
                "dtype": "bfloat16",
            },
            "adag_config": {
                "percentage_threshold": 0.05,
                "stop_gradient_attention_backend": backend,
            },
            "trace_warmup": {"enabled": False},
            "batch_size": 1,
            "wave_id": "wave-1",
            "code_revision": {
                "git_commit": code_revision
                or ("commit-eager" if backend == "eager" else "commit-sdpa"),
                "source_tree_sha256": "b" * 64 if backend == "eager" else "c" * 64,
            },
            "runtime_environment": {
                "python": "3.12.12",
                "packages": {"torch": "2.5.1"},
            },
        },
    }
    if contribution_execution is not None:
        manifest["artifact_identity"]["adag_config"][
            "stop_gradient_contribution_execution"
        ] = contribution_execution
    if embedding_edge_materialization is not None:
        manifest["artifact_identity"]["adag_config"][
            "embedding_edge_materialization"
        ] = embedding_edge_materialization
    if allocator_policy is not None:
        allocator_value = (
            None if allocator_policy == "default_v1" else "expandable_segments:True"
        )
        allocator_receipt = {
            "intended_policy_id": allocator_policy,
            "observed_environment": {
                "name": "PYTORCH_CUDA_ALLOC_CONF",
                "value": allocator_value,
                "is_set": allocator_value is not None,
            },
            "observed_allocator_backend": "native",
        }
        manifest["artifact_identity"]["cuda_allocator_policy"] = allocator_policy
        manifest["artifact_identity"]["runtime_environment"][
            "cuda_allocator_policy"
        ] = allocator_receipt
        manifest["runtime_environment"]["cuda_allocator_policy"] = allocator_receipt
    identity = manifest["artifact_identity"]
    identity_sha256 = hashlib.sha256(
        json.dumps(
            identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()
    identity["sha256"] = identity_sha256
    manifest["artifact_id"] = f"topk-trace-{identity_sha256[:24]}"
    return manifest


def _complete_trace_metadata(trace) -> None:
    trace.circuit_data.trace_metadata.update(
        {
            "prompt": "prompt",
            "prompt_sha256": "1" * 64,
            "response": "response",
            "response_sha256": "2" * 64,
            "system_prompt": None,
            "system_prompt_sha256": None,
            "teacher_forced_serialization_mode": "assistant_turn",
            "teacher_forced_token_identity": {
                "assistant_prefix_ids_sha256": "3" * 64,
                "response_ids_sha256": "4" * 64,
            },
            "assistant_prefix_token_count": 3,
            "response_token_count": 1,
            "included_response_token_count": 1,
            "input_token_count": 3,
            "chat_template_sha256": "5" * 64,
        }
    )


def _rehash_manifest(manifest: dict) -> dict:
    identity = manifest["artifact_identity"]
    identity_without_hash = {
        key: value for key, value in identity.items() if key != "sha256"
    }
    identity_sha256 = hashlib.sha256(
        json.dumps(
            identity_without_hash,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    identity["sha256"] = identity_sha256
    manifest["artifact_id"] = f"topk-trace-{identity_sha256[:24]}"
    return manifest


def _save_pair(
    tmp_path,
    *,
    reference_execution: str | None = "full_graph_v1",
    candidate_execution: str | None = "full_graph_v1",
):
    reference = _topk_trace()
    candidate = deepcopy(reference)
    _complete_trace_metadata(reference)
    _complete_trace_metadata(candidate)
    reference_path = tmp_path / "reference"
    candidate_path = tmp_path / "candidate"
    save_topk_compact_trace(
        reference_path,
        reference,
        metrics={
            "trace_wall_seconds": 10.0,
            "cuda_peak_allocated_bytes": 100,
            "cuda_peak_reserved_bytes": 120,
            "rss_peak_after_bytes": 200,
        },
        manifest=_manifest("eager", contribution_execution=reference_execution),
    )
    save_topk_compact_trace(
        candidate_path,
        candidate,
        metrics={
            "trace_wall_seconds": 6.0,
            "cuda_peak_allocated_bytes": 70,
            "cuda_peak_reserved_bytes": 80,
            "rss_peak_after_bytes": 190,
        },
        manifest=_manifest("sdpa_ov_only", contribution_execution=candidate_execution),
    )
    return reference_path, candidate_path


def _allowed_paths() -> list[str]:
    return [
        "artifact_identity.adag_config.stop_gradient_attention_backend",
        "artifact_identity.code_revision.*",
    ]


def test_backend_qualification_passes_explicit_gates_and_reports_resources(
    tmp_path,
) -> None:
    reference, candidate = _save_pair(tmp_path)

    report = compare_execution_artifacts(
        reference,
        candidate,
        allowed_identity_difference_paths=_allowed_paths(),
        tolerances={
            group: NumericTolerance(absolute=0.0, relative=0.0)
            for group in ("target", "node", "edge", "candidate_profile")
        },
        require_same_gpu_model=True,
        require_exact_node_topology=True,
        require_exact_edge_topology=True,
    )

    assert report["qualification_passed"] is True
    assert report["validation_passed"] is True
    assert report["diagnostic_only"] is False
    assert report["scientific_parity_claimed"] is False
    assert report["hardware"]["reference"]["family"] == "A100"
    assert report["topology"]["nodes"]["jaccard"] == 1.0
    assert report["target_values"]["candidate_logits"]["max_absolute_error"] == 0.0
    assert report["resources"]["trace_wall_seconds"]["candidate_over_reference"] == 0.6
    assert (
        report["resources"]["cuda_peak_reserved_bytes"]["candidate_minus_reference"]
        == -40
    )


def test_backend_qualification_can_allow_contribution_execution_difference(
    tmp_path,
) -> None:
    reference, candidate = _save_pair(
        tmp_path,
        reference_execution=None,
        candidate_execution="source_leaf_v1",
    )

    report = compare_execution_artifacts(
        reference,
        candidate,
        allowed_identity_difference_paths=[
            *_allowed_paths(),
            "artifact_identity.adag_config.stop_gradient_contribution_execution",
        ],
        require_exact_node_topology=True,
        require_exact_edge_topology=True,
    )

    assert report["validation_passed"] is True
    assert report["schema_version"] == "bonafide-execution-qualification/v1"
    allowed = report["identity"]["artifact_identity"]["allowed_differences"]
    assert {difference["path"] for difference in allowed} >= {
        "artifact_identity.adag_config.stop_gradient_attention_backend",
        "artifact_identity.adag_config.stop_gradient_contribution_execution",
    }


def test_backend_qualification_rejects_wildcard_for_scalar_strategy(
    tmp_path,
) -> None:
    reference, candidate = _save_pair(tmp_path)

    with pytest.raises(ValueError, match="exact path"):
        compare_execution_artifacts(
            reference,
            candidate,
            allowed_identity_difference_paths=[
                "artifact_identity.adag_config.stop_gradient_contribution_execution.*"
            ],
        )


def test_execution_qualification_allows_exact_allocator_policy_only(
    tmp_path,
) -> None:
    reference = _topk_trace()
    candidate = deepcopy(reference)
    _complete_trace_metadata(reference)
    _complete_trace_metadata(candidate)
    reference_path = tmp_path / "allocator-default"
    candidate_path = tmp_path / "allocator-expandable"
    save_topk_compact_trace(
        reference_path,
        reference,
        manifest=_manifest("eager", allocator_policy="default_v1"),
    )
    save_topk_compact_trace(
        candidate_path,
        candidate,
        manifest=_manifest("eager", allocator_policy="expandable_segments_v1"),
    )

    report = compare_execution_artifacts(
        reference_path,
        candidate_path,
        allowed_identity_difference_paths=CUDA_ALLOCATOR_AB_IDENTITY_PATHS,
        tolerances={
            group: NumericTolerance(absolute=0.0, relative=0.0)
            for group in TOLERANCE_GROUPS
        },
        require_same_gpu_model=True,
        require_exact_node_topology=True,
        require_exact_edge_topology=True,
        require_canonical_cuda_allocator_ab=True,
    )
    assert report["validation_passed"] is True
    assert report["schema_version"] == "bonafide-execution-qualification/v1"
    assert {gate["gate"] for gate in report["gates"]} >= {
        "same_gpu_model",
        "canonical_cuda_allocator_ab_pair",
        "exact_node_topology",
        "exact_edge_topology",
        *(f"{group}_numeric_tolerance" for group in TOLERANCE_GROUPS),
    }
    allowed = report["identity"]["artifact_identity"]["allowed_differences"]
    assert {difference["path"] for difference in allowed} == set(
        CUDA_ALLOCATOR_AB_IDENTITY_PATHS
    )

    with pytest.raises(ValueError, match="exact path"):
        compare_execution_artifacts(
            reference_path,
            candidate_path,
            allowed_identity_difference_paths=[
                "artifact_identity.cuda_allocator_policy.*"
            ],
        )


def test_cuda_allocator_ab_cli_resolves_strict_non_overridable_profile() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--reference",
            "reference",
            "--candidate",
            "candidate",
            "--output",
            "report.json",
            "--cuda-allocator-ab",
        ]
    )

    options = comparison_options(args)

    assert options["allowed_identity_difference_paths"] == (
        CUDA_ALLOCATOR_AB_IDENTITY_PATHS
    )
    assert options["require_same_gpu_model"] is True
    assert options["require_same_gpu_family"] is True
    assert options["require_exact_node_topology"] is True
    assert options["require_exact_edge_topology"] is True
    assert options["require_canonical_cuda_allocator_ab"] is True
    assert options["tolerances"] == {
        group: NumericTolerance(absolute=0.0, relative=0.0)
        for group in TOLERANCE_GROUPS
    }

    args.allow_identity_difference = ["artifact_identity.code_revision.*"]
    with pytest.raises(ValueError, match="cannot be combined"):
        comparison_options(args)


def test_cuda_allocator_ab_rejects_same_target_code_revision_drift(tmp_path) -> None:
    reference = _topk_trace()
    candidate = deepcopy(reference)
    _complete_trace_metadata(reference)
    _complete_trace_metadata(candidate)
    reference_path = tmp_path / "allocator-default"
    candidate_path = tmp_path / "allocator-expandable-code-drift"
    save_topk_compact_trace(
        reference_path,
        reference,
        manifest=_manifest(
            "eager", allocator_policy="default_v1", code_revision="same-commit"
        ),
    )
    save_topk_compact_trace(
        candidate_path,
        candidate,
        manifest=_manifest(
            "eager",
            allocator_policy="expandable_segments_v1",
            code_revision="different-commit",
        ),
    )

    report = compare_execution_artifacts(
        reference_path,
        candidate_path,
        **comparison_options(
            build_parser().parse_args(
                [
                    "--reference",
                    str(reference_path),
                    "--candidate",
                    str(candidate_path),
                    "--output",
                    str(tmp_path / "unused.json"),
                    "--cuda-allocator-ab",
                ]
            )
        ),
    )

    assert report["validation_passed"] is False
    disallowed = report["identity"]["artifact_identity"]["unallowed_differences"]
    assert {difference["path"] for difference in disallowed} == {
        "artifact_identity.code_revision.git_commit"
    }


@pytest.mark.parametrize(
    ("reference_policy", "candidate_policy"),
    [
        ("default_v1", "default_v1"),
        ("expandable_segments_v1", "default_v1"),
    ],
)
def test_cuda_allocator_ab_rejects_equal_or_reversed_lanes(
    tmp_path, reference_policy, candidate_policy
) -> None:
    reference = _topk_trace()
    candidate = deepcopy(reference)
    _complete_trace_metadata(reference)
    _complete_trace_metadata(candidate)
    reference_path = tmp_path / "allocator-reference"
    candidate_path = tmp_path / "allocator-candidate"
    save_topk_compact_trace(
        reference_path,
        reference,
        manifest=_manifest("eager", allocator_policy=reference_policy),
    )
    save_topk_compact_trace(
        candidate_path,
        candidate,
        manifest=_manifest("eager", allocator_policy=candidate_policy),
    )

    report = compare_execution_artifacts(
        reference_path,
        candidate_path,
        allowed_identity_difference_paths=CUDA_ALLOCATOR_AB_IDENTITY_PATHS,
        require_canonical_cuda_allocator_ab=True,
    )

    assert report["validation_passed"] is False
    assert report["cuda_allocator_ab_contract"]["passed"] is False


def test_cuda_allocator_ab_rejects_equally_malformed_receipts(tmp_path) -> None:
    reference = _topk_trace()
    candidate = deepcopy(reference)
    _complete_trace_metadata(reference)
    _complete_trace_metadata(candidate)
    reference_path = tmp_path / "allocator-default-malformed"
    candidate_path = tmp_path / "allocator-expandable-malformed"
    manifests = [
        _manifest("eager", allocator_policy="default_v1"),
        _manifest("eager", allocator_policy="expandable_segments_v1"),
    ]
    for manifest in manifests:
        for runtime in (
            manifest["runtime_environment"],
            manifest["artifact_identity"]["runtime_environment"],
        ):
            receipt = runtime["cuda_allocator_policy"]
            receipt["observed_environment"]["name"] = "WRONG_ALLOCATOR_VARIABLE"
            receipt["observed_allocator_backend"] = "wrong-backend"
        _rehash_manifest(manifest)
    save_topk_compact_trace(reference_path, reference, manifest=manifests[0])
    save_topk_compact_trace(candidate_path, candidate, manifest=manifests[1])

    report = compare_execution_artifacts(
        reference_path,
        candidate_path,
        allowed_identity_difference_paths=CUDA_ALLOCATOR_AB_IDENTITY_PATHS,
        require_canonical_cuda_allocator_ab=True,
    )

    assert report["validation_passed"] is False
    assert report["identity"]["artifact_identity"]["passed"] is True
    assert report["cuda_allocator_ab_contract"]["passed"] is False


def _save_embedding_edge_pair(
    tmp_path,
    *,
    reference_strategy: str = "scalar_v1",
    candidate_strategy: str = "vectorized_v1",
    candidate_code_revision: str = "same-commit",
):
    reference_trace = _topk_trace()
    candidate_trace = deepcopy(reference_trace)
    _complete_trace_metadata(reference_trace)
    _complete_trace_metadata(candidate_trace)
    reference_path = tmp_path / "embedding-edge-scalar"
    candidate_path = tmp_path / "embedding-edge-vectorized"
    save_topk_compact_trace(
        reference_path,
        reference_trace,
        manifest=_manifest(
            "eager",
            embedding_edge_materialization=reference_strategy,
            code_revision="same-commit",
        ),
    )
    save_topk_compact_trace(
        candidate_path,
        candidate_trace,
        manifest=_manifest(
            "eager",
            embedding_edge_materialization=candidate_strategy,
            code_revision=candidate_code_revision,
        ),
    )
    return reference_path, candidate_path


def test_embedding_edge_ab_passes_strict_canonical_profile(tmp_path) -> None:
    reference, candidate = _save_embedding_edge_pair(tmp_path)

    report = compare_execution_artifacts(
        reference,
        candidate,
        **comparison_options(
            build_parser().parse_args(
                [
                    "--reference",
                    str(reference),
                    "--candidate",
                    str(candidate),
                    "--output",
                    str(tmp_path / "unused.json"),
                    "--embedding-edge-ab",
                ]
            )
        ),
    )

    assert report["validation_passed"] is True
    assert report["schema_version"] == "bonafide-execution-qualification/v1"
    assert report["embedding_edge_ab_contract"]["passed"] is True
    assert {gate["gate"] for gate in report["gates"]} >= {
        "same_gpu_family",
        "same_gpu_model",
        "canonical_embedding_edge_ab_pair",
        "exact_node_topology",
        "exact_edge_topology",
        *(f"{group}_numeric_tolerance" for group in TOLERANCE_GROUPS),
    }
    allowed = report["identity"]["artifact_identity"]["allowed_differences"]
    assert {difference["path"] for difference in allowed} == set(
        EMBEDDING_EDGE_AB_IDENTITY_PATHS
    )


def test_embedding_edge_ab_cli_resolves_strict_non_overridable_profile() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--reference",
            "reference",
            "--candidate",
            "candidate",
            "--output",
            "report.json",
            "--embedding-edge-ab",
        ]
    )

    options = comparison_options(args)

    assert options["allowed_identity_difference_paths"] == (
        EMBEDDING_EDGE_AB_IDENTITY_PATHS
    )
    assert options["require_same_gpu_model"] is True
    assert options["require_same_gpu_family"] is True
    assert options["require_exact_node_topology"] is True
    assert options["require_exact_edge_topology"] is True
    assert options["require_canonical_cuda_allocator_ab"] is False
    assert options["require_canonical_embedding_edge_ab"] is True
    assert options["tolerances"] == {
        group: NumericTolerance(absolute=0.0, relative=0.0)
        for group in TOLERANCE_GROUPS
    }

    args.allow_identity_difference = ["artifact_identity.code_revision.*"]
    with pytest.raises(ValueError, match="cannot be combined"):
        comparison_options(args)

    args.allow_identity_difference = []
    args.target_atol = 1e-6
    with pytest.raises(ValueError, match="fixes every numerical tolerance at zero"):
        comparison_options(args)

    args.target_atol = None
    args.cuda_allocator_ab = True
    with pytest.raises(ValueError, match="mutually exclusive"):
        comparison_options(args)


def test_embedding_edge_ab_rejects_code_revision_drift(tmp_path) -> None:
    reference, candidate = _save_embedding_edge_pair(
        tmp_path, candidate_code_revision="different-commit"
    )

    report = compare_execution_artifacts(
        reference,
        candidate,
        **comparison_options(
            build_parser().parse_args(
                [
                    "--reference",
                    str(reference),
                    "--candidate",
                    str(candidate),
                    "--output",
                    str(tmp_path / "unused.json"),
                    "--embedding-edge-ab",
                ]
            )
        ),
    )

    assert report["validation_passed"] is False
    disallowed = report["identity"]["artifact_identity"]["unallowed_differences"]
    assert {difference["path"] for difference in disallowed} == {
        "artifact_identity.code_revision.git_commit"
    }


@pytest.mark.parametrize(
    ("reference_strategy", "candidate_strategy"),
    [
        ("scalar_v1", "scalar_v1"),
        ("vectorized_v1", "scalar_v1"),
    ],
)
def test_embedding_edge_ab_rejects_equal_or_reversed_lanes(
    tmp_path, reference_strategy, candidate_strategy
) -> None:
    reference, candidate = _save_embedding_edge_pair(
        tmp_path,
        reference_strategy=reference_strategy,
        candidate_strategy=candidate_strategy,
    )

    report = compare_execution_artifacts(
        reference,
        candidate,
        allowed_identity_difference_paths=EMBEDDING_EDGE_AB_IDENTITY_PATHS,
        require_canonical_embedding_edge_ab=True,
    )

    assert report["validation_passed"] is False
    assert report["embedding_edge_ab_contract"]["passed"] is False


def test_embedding_edge_materialization_requires_exact_allowlist_path(
    tmp_path,
) -> None:
    reference, candidate = _save_embedding_edge_pair(tmp_path)

    with pytest.raises(ValueError, match="exact path"):
        compare_execution_artifacts(
            reference,
            candidate,
            allowed_identity_difference_paths=[
                "artifact_identity.adag_config.embedding_edge_materialization.*"
            ],
        )


def test_backend_qualification_fails_hard_source_identity_mismatch(tmp_path) -> None:
    reference, candidate = _save_pair(tmp_path)
    candidate_manifest_path = candidate / "manifest.json"
    # Re-save is unnecessary: source identity is present in both the top-level
    # manifest and the hashed payload identity, so create a second valid pair.
    _ = candidate_manifest_path
    mismatched = tmp_path / "candidate-mismatched"
    trace = _topk_trace()
    _complete_trace_metadata(trace)
    save_topk_compact_trace(
        mismatched,
        trace,
        manifest=_manifest("sdpa_ov_only", source_id="wrong-source"),
    )

    report = compare_attention_backend_artifacts(
        reference,
        mismatched,
        allowed_identity_difference_paths=_allowed_paths(),
    )

    assert report["validation_passed"] is False
    assert report["qualification_passed"] is None
    assert report["diagnostic_only"] is True
    hard = {item["field"]: item for item in report["identity"]["hard_checks"]}
    assert hard["manifest.source_width1_artifact_id"]["reason"] == "mismatch"
    assert report["identity"]["artifact_identity"]["passed"] is False


def test_backend_qualification_reports_topology_and_numeric_drift(tmp_path) -> None:
    reference_trace = _topk_trace()
    candidate_trace = deepcopy(reference_trace)
    _complete_trace_metadata(reference_trace)
    _complete_trace_metadata(candidate_trace)
    candidate_trace.circuit_data.df_node.loc[0, "attribution"] += 0.25
    candidate_trace.circuit_data.df_node.at[0, "contrib_map"] = [
        0.4,
        0.1,
        0.2,
        0.3,
        0.4,
    ]
    candidate_trace.circuit_data.df_edge = candidate_trace.circuit_data.df_edge.iloc[
        0:0
    ].copy()
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    save_topk_compact_trace(reference, reference_trace, manifest=_manifest("eager"))
    save_topk_compact_trace(
        candidate, candidate_trace, manifest=_manifest("sdpa_ov_only")
    )

    report = compare_attention_backend_artifacts(
        reference,
        candidate,
        allowed_identity_difference_paths=_allowed_paths(),
        tolerances={
            "node": NumericTolerance(absolute=0.01, relative=0.0),
            "candidate_profile": NumericTolerance(absolute=0.01, relative=0.0),
        },
        require_exact_edge_topology=True,
    )

    assert report["qualification_passed"] is False
    assert report["topology"]["edges"]["jaccard"] == 0.0
    assert report["node_values_on_intersection"]["attribution"][
        "max_absolute_error"
    ] == pytest.approx(0.25)
    assert report["candidate_profiles_on_node_intersection"]["overall"][
        "max_absolute_error"
    ] == pytest.approx(0.4)
    failed_gates = {gate["gate"] for gate in report["gates"] if not gate["passed"]}
    assert "exact_edge_topology" in failed_gates
    assert "node_numeric_tolerance" in failed_gates
    assert "candidate_profile_numeric_tolerance" in failed_gates


def test_backend_qualification_rejects_allowlisting_scientific_identity(
    tmp_path,
) -> None:
    reference, candidate = _save_pair(tmp_path)

    with pytest.raises(ValueError, match="may only allow"):
        compare_attention_backend_artifacts(
            reference,
            candidate,
            allowed_identity_difference_paths=["artifact_identity.model.revision"],
        )

    with pytest.raises(ValueError, match="stop-gradient attention backend"):
        compare_attention_backend_artifacts(
            reference,
            candidate,
            allowed_identity_difference_paths=[
                "artifact_identity.adag_config.percentage_threshold"
            ],
        )


def test_backend_qualification_rejects_invalid_identity_hash(tmp_path) -> None:
    reference, candidate = _save_pair(tmp_path)
    manifest_path = candidate / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_identity"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = compare_attention_backend_artifacts(
        reference,
        candidate,
        allowed_identity_difference_paths=_allowed_paths(),
    )

    assert report["validation_passed"] is False
    assert report["identity"]["candidate_integrity"]["passed"] is False


def test_qualification_report_is_atomic_and_never_overwritten(tmp_path) -> None:
    reference, candidate = _save_pair(tmp_path)
    report = compare_attention_backend_artifacts(
        reference,
        candidate,
        allowed_identity_difference_paths=_allowed_paths(),
    )
    output = tmp_path / "reports" / "qualification.json"

    save_qualification_report(output, report)

    assert output.is_file()
    assert not list(output.parent.glob(".qualification.json.tmp-*"))
    with pytest.raises(FileExistsError):
        save_qualification_report(output, report)
