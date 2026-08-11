"""Versioned inputs for union-trace, paper-style multiview clustering.

This is deliberately a derived artifact.  It re-opens the immutable v1
candidate-union and fixed-topology refinement payloads rather than changing the
frozen candidate-aware input schema.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from circuits.analysis.bonafide.candidate_clustering import (
    CandidateClusterInputBundle,
    load_candidate_cluster_input_bundle,
)
from circuits.analysis.bonafide.canonical import canonical_sha256, file_sha256
from circuits.analysis.bonafide.identity import SignedBasisKey
from circuits.tracing.artifact import load_topk_compact_trace
from circuits.tracing.candidate_union import (
    CandidateUnionArtifact,
    load_candidate_union_artifact,
)

HYBRID_INPUT_SCHEMA = "adag.bonafide.hybrid-candidate-inputs.v3"
PAPER_NORMALIZATION_EPSILON = 1e-10
HYBRID_PROTOCOL_PATH = "docs/HYBRID_CANDIDATE_CLUSTERING_PROTOCOL.md"
HYBRID_SOURCE_PATHS = (
    "circuits/analysis/bonafide/canonical.py",
    "circuits/analysis/bonafide/candidate_clustering.py",
    "circuits/analysis/bonafide/clustering.py",
    "circuits/analysis/bonafide/clustering_evaluation.py",
    "circuits/analysis/bonafide/hybrid_candidate_inputs.py",
    "circuits/analysis/bonafide/hybrid_candidate_clustering.py",
    "circuits/analysis/bonafide/hybrid_candidate_clustering_execution.py",
    "circuits/analysis/bonafide/identity.py",
    "circuits/tracing/artifact.py",
    "circuits/tracing/candidate_union.py",
    "circuits/tracing/candidates.py",
    HYBRID_PROTOCOL_PATH,
    "scripts/bonafide/build_hybrid_candidate_inputs.py",
    "scripts/bonafide/hybrid_candidate_clustering_fit.py",
    "pyproject.toml",
    "uv.lock",
)
PROFILE_SCHEMA = pa.schema(
    [
        pa.field("case_id", pa.string(), nullable=False),
        pa.field("signed_basis_index", pa.int64(), nullable=False),
        pa.field("model_id", pa.string(), nullable=False),
        pa.field("model_revision", pa.string(), nullable=False),
        pa.field("layer", pa.int32(), nullable=False),
        pa.field("neuron_index", pa.int64(), nullable=False),
        pa.field("polarity", pa.string(), nullable=False),
        pa.field("input_attribution_profile", pa.list_(pa.float64()), nullable=False),
        pa.field("input_attribution_support", pa.list_(pa.bool_()), nullable=False),
        pa.field("raw_candidate_contribution", pa.list_(pa.float64()), nullable=False),
        pa.field(
            "paper_normalized_input_attribution_profile",
            pa.list_(pa.float64()),
            nullable=False,
        ),
        pa.field(
            "paper_normalized_candidate_contribution",
            pa.list_(pa.float64()),
            nullable=False,
        ),
        pa.field("occurrence_count", pa.int32(), nullable=False),
    ]
)
BASIS_SCHEMA = pa.schema(
    [
        pa.field("signed_basis_index", pa.int64(), nullable=False),
        pa.field("model_id", pa.string(), nullable=False),
        pa.field("model_revision", pa.string(), nullable=False),
        pa.field("layer", pa.int32(), nullable=False),
        pa.field("neuron_index", pa.int64(), nullable=False),
        pa.field("polarity", pa.string(), nullable=False),
    ]
)


@dataclass(frozen=True)
class HybridInputBundle:
    root: Path
    manifest: Mapping[str, Any]
    source_bundle: CandidateClusterInputBundle
    basis_rows: tuple[Mapping[str, Any], ...]
    target_rows: tuple[Mapping[str, Any], ...]
    profile_rows: tuple[Mapping[str, Any], ...]


def _git(repo_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def collect_hybrid_code_revision(repo_root: Path) -> dict[str, Any]:
    """Bind every executable hybrid source and require committed contents."""

    repo_root = repo_root.resolve()
    if Path(_git(repo_root, "rev-parse", "--show-toplevel")).resolve() != repo_root:
        raise ValueError("repo_root is not the Git worktree root")
    tracked_status = _git(repo_root, "status", "--porcelain=v1", "--untracked-files=no")
    digest = hashlib.sha256()
    files: list[dict[str, str]] = []
    for relative in HYBRID_SOURCE_PATHS:
        path = repo_root / relative
        if not path.is_file():
            raise ValueError(f"hybrid source is missing: {relative}")
        try:
            tracked = _git(repo_root, "ls-files", "--error-unmatch", "--", relative)
        except subprocess.CalledProcessError as error:
            raise ValueError(f"hybrid source is not tracked: {relative}") from error
        if tracked != relative:
            raise ValueError(f"hybrid source tracking is ambiguous: {relative}")
        if _git(repo_root, "hash-object", relative) != _git(
            repo_root, "rev-parse", f"HEAD:{relative}"
        ):
            raise ValueError(f"hybrid source differs from HEAD: {relative}")
        encoded = relative.encode("utf-8")
        content = path.read_bytes()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
        files.append({"path": relative, "sha256": file_sha256(path)})
    return {
        "repo_root": str(repo_root),
        "git_commit": _git(repo_root, "rev-parse", "HEAD"),
        "git_tree": _git(repo_root, "rev-parse", "HEAD^{tree}"),
        "git_dirty": bool(tracked_status),
        "git_status_sha256": hashlib.sha256(tracked_status.encode("utf-8")).hexdigest(),
        "source_tree_sha256": digest.hexdigest(),
        "files": files,
    }


def paper_normalize_occurrence(
    attr_map: Sequence[float | None],
    *,
    activation: float,
    candidate_contribution: Sequence[float],
    candidate_logits: Sequence[float],
    epsilon: float = PAPER_NORMALIZATION_EPSILON,
) -> tuple[list[float | None], list[float]]:
    """Apply the upstream normalization before token-position aggregation."""

    activation = _finite(activation, "activation")
    contributions = [
        _finite(value, "candidate contribution") for value in candidate_contribution
    ]
    logits = [_finite(value, "candidate logit") for value in candidate_logits]
    if len(contributions) != len(logits) or not contributions:
        raise ValueError("candidate contribution/logit widths disagree")
    activation_denominator = activation if abs(activation) > epsilon else 1.0
    normalized_attr = [
        None if value is None else _finite(value, "attr_map") / activation_denominator
        for value in attr_map
    ]
    normalized_contribution = [
        contribution / (logit if abs(logit) > epsilon else 1.0)
        for contribution, logit in zip(contributions, logits, strict=True)
    ]
    return normalized_attr, normalized_contribution


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _vector(value: object, field: str) -> list[Any]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{field} must be a sequence")
    try:
        return list(cast(Iterable[Any], value))
    except TypeError as error:
        raise TypeError(f"{field} must be a sequence") from error


def _node_key(row: Any) -> tuple[int, int, int]:
    return int(row.layer), int(row.token), int(row.neuron)


def _polarity(activation: float) -> Literal["+", "-"] | None:
    if activation > 0:
        return "+"
    if activation < 0:
        return "-"
    return None


def _locate_unions(root: Path) -> dict[str, Path]:
    family = root / "bonafide.candidate-union.v1"
    if not family.is_dir():
        raise ValueError(f"candidate-union family is missing: {family}")
    result: dict[str, Path] = {}
    for manifest_path in sorted(family.rglob("manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        artifact_id = manifest.get("artifact_id")
        if not isinstance(artifact_id, str) or not artifact_id:
            raise ValueError(f"candidate-union artifact ID is invalid: {manifest_path}")
        if artifact_id in result:
            raise ValueError(f"duplicate candidate-union artifact ID: {artifact_id}")
        result[artifact_id] = manifest_path.parent
    return result


def _resolve_artifact_location(location: object, *, union_path: Path) -> Path:
    if not isinstance(location, str) or not location:
        raise ValueError("refinement artifact location is invalid")
    path = Path(location)
    if path.is_dir():
        return path
    # Snapshots can move as a unit.  Permit only an exact artifact basename
    # beneath the union family root, then let payload hashes authenticate it.
    matches = list(union_path.parents[2].rglob(path.name))
    matches = [candidate for candidate in matches if candidate.is_dir()]
    if len(matches) != 1:
        raise ValueError(f"refinement artifact cannot be resolved uniquely: {location}")
    return matches[0]


def extract_hybrid_target(
    union: CandidateUnionArtifact,
    *,
    target: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Recover observed-target attr maps and all candidate contributions."""

    trace = union.trace
    if union.manifest.get("artifact_id") != target["candidate_union_artifact_id"]:
        raise ValueError("candidate-union artifact ID disagrees with target")
    if union.manifest.get("data_sha256") != target["candidate_union_payload_sha256"]:
        raise ValueError("candidate-union payload hash disagrees with target")
    if trace.topology_sha256 != target["candidate_union_topology_sha256"]:
        raise ValueError("candidate-union topology hash disagrees with target")
    if trace.source_width1_artifact_id != target["source_width1_artifact_id"]:
        raise ValueError("candidate-union source identity disagrees with target")
    if trace.shared_response_position != int(target["response_position"]):
        raise ValueError("candidate-union response position disagrees with target")

    candidates = tuple(trace.candidate_selection.candidates)
    if len(candidates) not in {5, 6} or len(candidates) != int(
        target["candidate_count"]
    ):
        raise ValueError("candidate width disagrees with target")
    if not candidates[0].is_observed or any(
        item.is_observed for item in candidates[1:]
    ):
        raise ValueError("candidate union must place exactly one observed token first")
    ranks = [int(item.full_distribution_rank) for item in candidates]
    if len(set(ranks)) != len(ranks) or not set(range(1, 6)).issubset(ranks):
        raise ValueError("candidate axis does not contain unique model top-five ranks")

    records = trace.refinement_artifacts
    if not records:
        raise ValueError("candidate union has no refinement provenance")
    observed_record = records[0]
    refinement_path = _resolve_artifact_location(
        observed_record.get("location"), union_path=union.path
    )
    refinement = load_topk_compact_trace(refinement_path)
    if refinement.manifest.get("artifact_id") != observed_record.get("artifact_id"):
        raise ValueError("refinement artifact ID mismatch")
    if refinement.manifest.get("data_sha256") != observed_record.get("payload_sha256"):
        raise ValueError("refinement artifact payload hash mismatch")
    refinement_candidate = refinement.topk_trace.candidate_selection.candidates
    if len(refinement_candidate) != 1 or (
        refinement_candidate[0].token_id != candidates[0].token_id
    ):
        raise ValueError("refinement is not bound to the observed candidate")
    trace_metadata = refinement.manifest.get("trace_metadata")
    frozen_topology = (
        trace_metadata.get("frozen_topology")
        if isinstance(trace_metadata, Mapping)
        else None
    )
    if (
        not isinstance(frozen_topology, Mapping)
        or frozen_topology.get("sha256") != trace.topology_sha256
    ):
        raise ValueError("refinement topology binding mismatch")

    refinement_nodes = {
        _node_key(row): row
        for row in refinement.topk_trace.circuit_data.df_node.itertuples(index=False)
    }
    if len(refinement_nodes) != len(refinement.topk_trace.circuit_data.df_node):
        raise ValueError("refinement contains duplicate nodes")
    final_layer = int(trace.df_node["layer"].max())
    # v1 stores the authoritative model identity on its basis rows, and callers
    # fill these two private target fields before extraction.
    model_id = str(target["_model_id"])
    model_revision = str(target["_model_revision"])
    candidate_logits = [_finite(item.logit, "candidate logit") for item in candidates]
    sums: dict[SignedBasisKey, dict[str, Any]] = {}
    comparisons = 0
    for raw_row in trace.df_node.itertuples(index=False):
        row: Any = raw_row
        layer = int(row.layer)
        if not 0 <= layer < final_layer:
            continue
        activations = [
            _finite(item, "candidate activation")
            for item in _vector(row.candidate_activation, "candidate activation")
        ]
        contributions = [
            _finite(item, "candidate contribution")
            for item in _vector(row.candidate_contribution, "candidate contribution")
        ]
        applicable = [
            bool(item)
            for item in _vector(row.applicable_by_candidate, "candidate applicability")
        ]
        if len(activations) != len(candidates) or len(contributions) != len(candidates):
            raise ValueError("candidate measurement width mismatch")
        if applicable != [True] * len(candidates):
            raise ValueError("internal union node is not applicable to every candidate")
        for activation in activations[1:]:
            comparisons += 1
            if not np.isclose(activation, activations[0], rtol=1e-6, atol=1e-7):
                raise ValueError("candidate activation invariance failed")
        polarity = _polarity(activations[0])
        if polarity is None:
            continue
        key = _node_key(row)
        refinement_row: Any = refinement_nodes.get(key)
        if refinement_row is None:
            raise ValueError(f"observed refinement omitted union node: {key}")
        if not np.isclose(
            float(refinement_row.activation), activations[0], rtol=1e-6, atol=1e-7
        ):
            raise ValueError("refinement/union activation mismatch")
        raw_attr = _vector(refinement_row.attr_map, "refinement attr_map")
        attr = [
            None if value is None else _finite(value, "refinement attr_map")
            for value in raw_attr
        ]
        basis = SignedBasisKey(
            model_id, model_revision, layer, int(row.neuron), polarity
        )  # type: ignore[arg-type]
        accumulator = sums.setdefault(
            basis,
            {
                "attr": [0.0] * len(attr),
                "support": [False] * len(attr),
                "contribution": np.zeros(len(candidates), dtype=np.float64),
                "paper_attr": [0.0] * len(attr),
                "paper_contribution": np.zeros(len(candidates), dtype=np.float64),
                "occurrence_count": 0,
            },
        )
        if len(accumulator["attr"]) != len(attr):
            raise ValueError("attribution-map width varies within target")
        for index, value in enumerate(attr):
            if value is not None:
                accumulator["attr"][index] += value
                accumulator["support"][index] = True
        paper_attr, paper_contribution = paper_normalize_occurrence(
            attr,
            activation=activations[0],
            candidate_contribution=contributions,
            candidate_logits=candidate_logits,
        )
        for index, value in enumerate(paper_attr):
            if value is not None:
                accumulator["paper_attr"][index] += value
        accumulator["contribution"] += np.asarray(contributions, dtype=np.float64)
        accumulator["paper_contribution"] += np.asarray(
            paper_contribution, dtype=np.float64
        )
        accumulator["occurrence_count"] += 1
    if comparisons == 0 or not sums:
        raise ValueError("target has no candidate-invariant internal MLP evidence")

    rows: list[dict[str, Any]] = []
    for basis, accumulator in sorted(sums.items()):
        rows.append(
            {
                "case_id": str(target["case_id"]),
                "basis": basis,
                "input_attribution_profile": [
                    value if supported else None
                    for value, supported in zip(
                        accumulator["attr"], accumulator["support"], strict=True
                    )
                ],
                "input_attribution_support": list(accumulator["support"]),
                "raw_candidate_contribution": [
                    float(value) for value in accumulator["contribution"]
                ],
                "paper_normalized_input_attribution_profile": [
                    value if supported else None
                    for value, supported in zip(
                        accumulator["paper_attr"],
                        accumulator["support"],
                        strict=True,
                    )
                ],
                "paper_normalized_candidate_contribution": [
                    float(value) for value in accumulator["paper_contribution"]
                ],
                "occurrence_count": int(accumulator["occurrence_count"]),
            }
        )
    metadata = {
        "case_id": str(target["case_id"]),
        "response_id": str(target["response_id"]),
        "base_question_id": str(target["base_question_id"]),
        "family_partition": str(target["family_partition"]),
        "partition_hierarchical_weight": float(target["partition_hierarchical_weight"]),
        "candidate_count": len(candidates),
        "candidate_ordering": "observed_first_then_model_top5_descending_logit_then_ascending_token_id",
        "candidate_axis": [item.to_dict() for item in candidates],
        "candidate_logits": candidate_logits,
        "observed_candidate_index": 0,
        "model_top5_indices": [ranks.index(rank) for rank in range(1, 6)],
        "candidate_union_artifact_id": union.manifest["artifact_id"],
        "candidate_union_payload_sha256": union.manifest["data_sha256"],
        "candidate_union_topology_sha256": trace.topology_sha256,
        "refinement_artifact_id": refinement.manifest["artifact_id"],
        "refinement_payload_sha256": refinement.manifest["data_sha256"],
    }
    return rows, metadata


def _write_parquet(
    path: Path, rows: Sequence[Mapping[str, Any]], schema: pa.Schema
) -> None:
    pq.write_table(
        pa.Table.from_pylist(list(rows), schema=schema), path, compression="zstd"
    )


def build_hybrid_input_bundle(
    *, source_root: Path, output_root: Path, repo_root: Path
) -> dict[str, Any]:
    """Build one atomic, hash-bound derived input bundle."""

    if output_root.exists():
        raise FileExistsError(f"hybrid input destination already exists: {output_root}")
    code_revision = collect_hybrid_code_revision(repo_root)
    if code_revision["git_dirty"]:
        raise ValueError("refuse to publish hybrid inputs from dirty tracked source")
    source = load_candidate_cluster_input_bundle(source_root)
    source_manifest_hash = str(source.manifest["manifest_sha256"])
    union_root = Path(str(source.manifest["inputs"]["candidate_union_root"]))
    unions = _locate_unions(union_root)
    identity_pairs = {
        (str(row["model_id"]), str(row["model_revision"])) for row in source.basis_rows
    }
    if len(identity_pairs) != 1:
        raise ValueError("source bundle mixes model identities")
    model_id, model_revision = next(iter(identity_pairs))
    extracted: list[dict[str, Any]] = []
    target_metadata: list[dict[str, Any]] = []
    generation_targets = [
        row for row in source.target_rows if row.get("family_partition") == "generation"
    ]
    if not generation_targets:
        raise ValueError("source bundle has no generation targets")
    for raw_target in sorted(generation_targets, key=lambda row: str(row["case_id"])):
        target = dict(raw_target)
        target["_model_id"] = model_id
        target["_model_revision"] = model_revision
        artifact_id = str(target["candidate_union_artifact_id"])
        if artifact_id not in unions:
            raise ValueError(f"candidate-union artifact is missing: {artifact_id}")
        union = load_candidate_union_artifact(unions[artifact_id])
        rows, metadata = extract_hybrid_target(union, target=target)
        extracted.extend(rows)
        target_metadata.append(metadata)

    bases = sorted({row["basis"] for row in extracted})
    basis_index = {basis: index for index, basis in enumerate(bases)}
    basis_rows = [
        {
            "signed_basis_index": index,
            "model_id": basis.model_id,
            "model_revision": basis.model_revision,
            "layer": basis.layer,
            "neuron_index": basis.neuron_index,
            "polarity": basis.polarity,
        }
        for index, basis in enumerate(bases)
    ]
    profile_rows = []
    for row in sorted(
        extracted, key=lambda item: (str(item["case_id"]), basis_index[item["basis"]])
    ):
        basis = row.pop("basis")
        profile_rows.append(
            {
                "case_id": row["case_id"],
                "signed_basis_index": basis_index[basis],
                "model_id": basis.model_id,
                "model_revision": basis.model_revision,
                "layer": basis.layer,
                "neuron_index": basis.neuron_index,
                "polarity": basis.polarity,
                "input_attribution_profile": row["input_attribution_profile"],
                "input_attribution_support": row["input_attribution_support"],
                "raw_candidate_contribution": row["raw_candidate_contribution"],
                "paper_normalized_input_attribution_profile": row[
                    "paper_normalized_input_attribution_profile"
                ],
                "paper_normalized_candidate_contribution": row[
                    "paper_normalized_candidate_contribution"
                ],
                "occurrence_count": row["occurrence_count"],
            }
        )

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_root.parent / f".{output_root.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        _write_parquet(temporary / "basis-index.parquet", basis_rows, BASIS_SCHEMA)
        _write_parquet(temporary / "profiles.parquet", profile_rows, PROFILE_SCHEMA)
        targets_payload = {
            "schema_version": HYBRID_INPUT_SCHEMA,
            "targets": target_metadata,
        }
        (temporary / "targets.json").write_text(
            json.dumps(targets_payload, indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        files = [
            {"path": name, "sha256": file_sha256(temporary / name)}
            for name in ("basis-index.parquet", "profiles.parquet", "targets.json")
        ]
        manifest: dict[str, Any] = {
            "schema_version": HYBRID_INPUT_SCHEMA,
            "purpose": "exploratory_union_measurements_paper_style_clustering",
            "source_candidate_cluster_input": {
                "path": str(source_root.resolve()),
                "schema_version": source.manifest["schema_version"],
                "manifest_sha256": source_manifest_hash,
            },
            "code_revision": code_revision,
            "protocol": {
                "path": HYBRID_PROTOCOL_PATH,
                "sha256": next(
                    record["sha256"]
                    for record in code_revision["files"]
                    if record["path"] == HYBRID_PROTOCOL_PATH
                ),
            },
            "representation_contract": {
                "primary": "raw_top5_plus_observed.v1",
                "sensitivities": [
                    "paper_normalized_model_top5.v1",
                    "raw_model_top5.v1",
                    "top5_minus_observed.v1",
                ],
                "input_attribution": "observed_candidate_fixed_union_attr_map_sum_by_signed_basis.v1",
                "paper_normalization": {
                    "id": "upstream_attr_activation_and_contribution_logit.v1",
                    "epsilon": PAPER_NORMALIZATION_EPSILON,
                    "small_denominator_fallback": 1.0,
                    "aggregation_order": "normalize_per_occurrence_then_sum_by_signed_basis",
                },
                "candidate_axis": "target_local_observed_first_width_5_or_6.v1",
            },
            "artifact_payloads": {
                "candidate_union_set_sha256": canonical_sha256(
                    sorted(
                        (
                            {
                                "case_id": row["case_id"],
                                "artifact_id": row["candidate_union_artifact_id"],
                                "payload_sha256": row["candidate_union_payload_sha256"],
                                "topology_sha256": row[
                                    "candidate_union_topology_sha256"
                                ],
                            }
                            for row in target_metadata
                        ),
                        key=lambda record: record["case_id"],
                    )
                ),
                "refinement_set_sha256": canonical_sha256(
                    sorted(
                        (
                            {
                                "case_id": row["case_id"],
                                "artifact_id": row["refinement_artifact_id"],
                                "payload_sha256": row["refinement_payload_sha256"],
                            }
                            for row in target_metadata
                        ),
                        key=lambda record: record["case_id"],
                    )
                ),
            },
            "fit_partition": {
                "name": "generation",
                "case_set_sha256": canonical_sha256(
                    sorted(row["case_id"] for row in target_metadata)
                ),
                "family_set_sha256": canonical_sha256(
                    sorted({row["base_question_id"] for row in target_metadata})
                ),
                "confirmatory_holdout_opened": False,
            },
            "counts": {
                "target_count": len(target_metadata),
                "basis_count": len(basis_rows),
                "profile_row_count": len(profile_rows),
                "candidate_width_counts": {
                    str(width): sum(
                        row["candidate_count"] == width for row in target_metadata
                    )
                    for width in (5, 6)
                },
            },
            "exploratory": True,
            "labeling_authorized": False,
            "confirmatory_holdout_opened": False,
            "files": files,
        }
        manifest["manifest_sha256"] = canonical_sha256(manifest)
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output_root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


_INPUT_FILES = {"basis-index.parquet", "profiles.parquet", "targets.json"}
_TARGET_FIELDS = {
    "case_id",
    "response_id",
    "base_question_id",
    "family_partition",
    "partition_hierarchical_weight",
    "candidate_count",
    "candidate_ordering",
    "candidate_axis",
    "candidate_logits",
    "observed_candidate_index",
    "model_top5_indices",
    "candidate_union_artifact_id",
    "candidate_union_payload_sha256",
    "candidate_union_topology_sha256",
    "refinement_artifact_id",
    "refinement_payload_sha256",
}
_BASIS_IDENTITY_FIELDS = (
    "model_id",
    "model_revision",
    "layer",
    "neuron_index",
    "polarity",
)


def _sha256_string(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _validate_file_inventory(
    root: Path, records: object, *, expected: set[str]
) -> None:
    if not isinstance(records, list):
        raise TypeError("hybrid input file inventory is invalid")
    by_name: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
            raise TypeError("hybrid input file record is invalid")
        name = record.get("path")
        if not isinstance(name, str) or Path(name).name != name or name in by_name:
            raise ValueError("hybrid input file path is unsafe or duplicate")
        _sha256_string(record.get("sha256"), "hybrid input file hash")
        by_name[name] = cast(Mapping[str, Any], record)
    if set(by_name) != expected:
        raise ValueError("hybrid input file inventory is incomplete")
    for name, record in by_name.items():
        path = root / name
        if not path.is_file() or file_sha256(path) != record["sha256"]:
            raise ValueError(f"hybrid input file hash mismatch: {name}")


def _validate_code_revision(value: object) -> None:
    if not isinstance(value, Mapping) or value.get("git_dirty") is not False:
        raise ValueError("hybrid input code revision is absent or dirty")
    for field in ("git_commit", "git_tree"):
        raw = value.get(field)
        if not isinstance(raw, str) or len(raw) != 40:
            raise ValueError(f"hybrid input code revision {field} is invalid")
    _sha256_string(value.get("source_tree_sha256"), "hybrid source tree hash")
    _sha256_string(value.get("git_status_sha256"), "hybrid Git status hash")
    records = value.get("files")
    if not isinstance(records, list):
        raise TypeError("hybrid source file inventory is invalid")
    by_path: dict[str, str] = {}
    for record in records:
        if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
            raise TypeError("hybrid source file record is invalid")
        path = record.get("path")
        if not isinstance(path, str) or path in by_path:
            raise ValueError("hybrid source file path is invalid or duplicate")
        by_path[path] = _sha256_string(record.get("sha256"), "hybrid source hash")
    if set(by_path) != set(HYBRID_SOURCE_PATHS):
        raise ValueError("hybrid source file inventory drift")


def _validate_target_rows(
    target_rows: Sequence[Mapping[str, Any]],
    *,
    source: CandidateClusterInputBundle,
    manifest: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    source_generation = {
        str(row["case_id"]): row
        for row in source.target_rows
        if row["family_partition"] == "generation"
    }
    targets: dict[str, Mapping[str, Any]] = {}
    for row in target_rows:
        if not isinstance(row, Mapping) or set(row) != _TARGET_FIELDS:
            raise ValueError("hybrid target metadata fields drift")
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not case_id or case_id in targets:
            raise ValueError("hybrid target case ID is invalid or duplicate")
        if row.get("family_partition") != "generation":
            raise ValueError("hybrid input contains a non-generation target")
        source_row = source_generation.get(case_id)
        if source_row is None:
            raise ValueError("hybrid target is outside the generation partition")
        for field in (
            "response_id",
            "base_question_id",
            "candidate_count",
            "candidate_union_artifact_id",
            "candidate_union_payload_sha256",
            "candidate_union_topology_sha256",
        ):
            if row[field] != source_row[field]:
                raise ValueError(f"hybrid target/source binding drift: {field}")
        weight = _finite(row["partition_hierarchical_weight"], "target weight")
        if weight <= 0 or weight != float(source_row["partition_hierarchical_weight"]):
            raise ValueError("hybrid target hierarchical weight drift")
        count = row["candidate_count"]
        if isinstance(count, bool) or not isinstance(count, int) or count not in {5, 6}:
            raise ValueError("hybrid target candidate count is invalid")
        observed_index = row["observed_candidate_index"]
        if observed_index != 0:
            raise ValueError("hybrid target observed candidate index drift")
        indices = row["model_top5_indices"]
        if (
            not isinstance(indices, list)
            or len(indices) != 5
            or any(
                isinstance(index, bool) or not isinstance(index, int)
                for index in indices
            )
            or len(set(indices)) != 5
            or any(not 0 <= index < count for index in indices)
        ):
            raise ValueError("hybrid target model-top-five indices are invalid")
        axis = row["candidate_axis"]
        logits = row["candidate_logits"]
        if not isinstance(axis, list) or len(axis) != count:
            raise ValueError("hybrid target candidate axis width drift")
        if not isinstance(logits, list) or len(logits) != count:
            raise ValueError("hybrid target candidate logits width drift")
        ranks: list[int] = []
        observed: list[int] = []
        for index, raw_candidate in enumerate(axis):
            candidate = cast(Mapping[str, Any], raw_candidate)
            if not isinstance(candidate, Mapping) or set(candidate) != {
                "candidate_index",
                "full_distribution_rank",
                "token_id",
                "token_text",
                "logit",
                "probability",
                "is_observed",
            }:
                raise ValueError("hybrid target candidate-axis fields drift")
            if candidate["candidate_index"] != index:
                raise ValueError("hybrid target candidate axis is not contiguous")
            rank = candidate["full_distribution_rank"]
            token_id = candidate["token_id"]
            if (
                isinstance(rank, bool)
                or not isinstance(rank, int)
                or rank <= 0
                or isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or token_id < 0
            ):
                raise ValueError("hybrid target candidate identity is invalid")
            ranks.append(rank)
            if candidate["is_observed"]:
                observed.append(index)
            logit = _finite(candidate["logit"], "candidate-axis logit")
            probability = _finite(candidate["probability"], "candidate probability")
            if not 0 <= probability <= 1 or logit != _finite(
                logits[index], "candidate logit"
            ):
                raise ValueError("hybrid target candidate score drift")
        if observed != [0] or len(set(ranks)) != count:
            raise ValueError("hybrid target observed/rank axis is invalid")
        if [ranks.index(rank) for rank in range(1, 6)] != indices:
            raise ValueError("hybrid target top-five rank ordering drift")
        for field in (
            "candidate_union_payload_sha256",
            "candidate_union_topology_sha256",
            "refinement_payload_sha256",
        ):
            _sha256_string(row[field], field)
        for field in ("candidate_union_artifact_id", "refinement_artifact_id"):
            if not isinstance(row[field], str) or not row[field]:
                raise ValueError(f"hybrid target {field} is invalid")
        targets[case_id] = row
    if set(targets) != set(source_generation):
        raise ValueError("hybrid targets do not exactly cover the generation partition")
    fit_partition = manifest.get("fit_partition")
    families = sorted({str(row["base_question_id"]) for row in target_rows})
    if not isinstance(fit_partition, Mapping) or fit_partition != {
        "name": "generation",
        "case_set_sha256": canonical_sha256(sorted(targets)),
        "family_set_sha256": canonical_sha256(families),
        "confirmatory_holdout_opened": False,
    }:
        raise ValueError("hybrid generation fit-partition binding drift")
    return targets


def _validate_profile_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    targets: Mapping[str, Mapping[str, Any]],
    basis_rows: Sequence[Mapping[str, Any]],
) -> None:
    seen: set[tuple[str, int]] = set()
    covered_cases: set[str] = set()
    covered_bases: set[int] = set()
    attr_widths: dict[str, int] = {}
    for row in rows:
        case_id = str(row["case_id"])
        if case_id not in targets:
            raise ValueError("hybrid profile references an unknown target")
        index = row["signed_basis_index"]
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < len(basis_rows)
        ):
            raise ValueError("hybrid profile basis index is invalid")
        if any(
            row[field] != basis_rows[index][field] for field in _BASIS_IDENTITY_FIELDS
        ):
            raise ValueError("hybrid profile identity disagrees with basis index")
        key = (case_id, index)
        if key in seen:
            raise ValueError("hybrid profile target/basis row is duplicate")
        seen.add(key)
        covered_cases.add(case_id)
        covered_bases.add(index)
        count = row["occurrence_count"]
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("hybrid profile occurrence count is invalid")
        support = row["input_attribution_support"]
        raw_attr = row["input_attribution_profile"]
        normalized_attr = row["paper_normalized_input_attribution_profile"]
        if (
            not isinstance(support, list)
            or not support
            or not all(isinstance(value, bool) for value in support)
            or not isinstance(raw_attr, list)
            or not isinstance(normalized_attr, list)
            or len(raw_attr) != len(support)
            or len(normalized_attr) != len(support)
        ):
            raise ValueError("hybrid attribution profile shape is invalid")
        previous_width = attr_widths.setdefault(case_id, len(support))
        if previous_width != len(support):
            raise ValueError("hybrid attribution width varies within target")
        for raw, normalized, is_supported in zip(
            raw_attr, normalized_attr, support, strict=True
        ):
            if is_supported:
                _finite(raw, "supported raw attribution")
                _finite(normalized, "supported normalized attribution")
            elif raw is not None or normalized is not None:
                raise ValueError("unsupported hybrid attribution is not null")
        candidate_count = int(targets[case_id]["candidate_count"])
        for field in (
            "raw_candidate_contribution",
            "paper_normalized_candidate_contribution",
        ):
            values = row[field]
            if not isinstance(values, list) or len(values) != candidate_count:
                raise ValueError("hybrid candidate profile width is invalid")
            for value in values:
                _finite(value, field)
    if covered_cases != set(targets):
        raise ValueError("hybrid profiles do not cover every generation target")
    if covered_bases != set(range(len(basis_rows))):
        raise ValueError("hybrid basis index contains an unreferenced basis")


def _artifact_payload_hashes(
    target_rows: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    ordered = sorted(target_rows, key=lambda row: str(row["case_id"]))
    return {
        "candidate_union_set_sha256": canonical_sha256(
            [
                {
                    "case_id": row["case_id"],
                    "artifact_id": row["candidate_union_artifact_id"],
                    "payload_sha256": row["candidate_union_payload_sha256"],
                    "topology_sha256": row["candidate_union_topology_sha256"],
                }
                for row in ordered
            ]
        ),
        "refinement_set_sha256": canonical_sha256(
            [
                {
                    "case_id": row["case_id"],
                    "artifact_id": row["refinement_artifact_id"],
                    "payload_sha256": row["refinement_payload_sha256"],
                }
                for row in ordered
            ]
        ),
    }


def load_hybrid_input_bundle(root: Path) -> HybridInputBundle:
    root = root.resolve()
    manifest_value = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest_value, dict):
        raise TypeError("hybrid input manifest must be an object")
    manifest = manifest_value
    core = dict(manifest)
    recorded = core.pop("manifest_sha256", None)
    if manifest.get(
        "schema_version"
    ) != HYBRID_INPUT_SCHEMA or recorded != canonical_sha256(core):
        raise ValueError("hybrid input manifest is invalid")
    if (
        manifest.get("purpose")
        != "exploratory_union_measurements_paper_style_clustering"
        or manifest.get("exploratory") is not True
        or manifest.get("labeling_authorized") is not False
        or manifest.get("confirmatory_holdout_opened") is not False
    ):
        raise ValueError("hybrid input scientific status drift")
    _validate_file_inventory(root, manifest.get("files"), expected=_INPUT_FILES)
    _validate_code_revision(manifest.get("code_revision"))
    source_hashes = {
        record["path"]: record["sha256"]
        for record in manifest["code_revision"]["files"]
    }
    if manifest.get("protocol") != {
        "path": HYBRID_PROTOCOL_PATH,
        "sha256": source_hashes[HYBRID_PROTOCOL_PATH],
    }:
        raise ValueError("hybrid protocol provenance drift")
    representation_contract = manifest.get("representation_contract")
    if not isinstance(representation_contract, Mapping) or (
        representation_contract.get("primary") != "raw_top5_plus_observed.v1"
        or representation_contract.get("sensitivities")
        != [
            "paper_normalized_model_top5.v1",
            "raw_model_top5.v1",
            "top5_minus_observed.v1",
        ]
        or representation_contract.get("candidate_axis")
        != "target_local_observed_first_width_5_or_6.v1"
        or representation_contract.get("paper_normalization")
        != {
            "id": "upstream_attr_activation_and_contribution_logit.v1",
            "epsilon": PAPER_NORMALIZATION_EPSILON,
            "small_denominator_fallback": 1.0,
            "aggregation_order": ("normalize_per_occurrence_then_sum_by_signed_basis"),
        }
    ):
        raise ValueError("hybrid representation contract drift")
    source_record = manifest.get("source_candidate_cluster_input")
    if not isinstance(source_record, Mapping) or set(source_record) != {
        "path",
        "schema_version",
        "manifest_sha256",
    }:
        raise TypeError("hybrid input lacks source bundle binding")
    source = load_candidate_cluster_input_bundle(Path(str(source_record["path"])))
    if source.manifest.get("manifest_sha256") != source_record.get(
        "manifest_sha256"
    ) or source.manifest.get("schema_version") != source_record.get("schema_version"):
        raise ValueError("source candidate input manifest hash drift")
    basis_table = pq.read_table(root / "basis-index.parquet")
    profiles_table = pq.read_table(root / "profiles.parquet")
    if not basis_table.schema.equals(BASIS_SCHEMA, check_metadata=False):
        raise ValueError("hybrid basis schema drift")
    if not profiles_table.schema.equals(PROFILE_SCHEMA, check_metadata=False):
        raise ValueError("hybrid profile schema drift")
    targets_value = json.loads((root / "targets.json").read_text(encoding="utf-8"))
    if targets_value.get("schema_version") != HYBRID_INPUT_SCHEMA:
        raise ValueError("hybrid target metadata schema drift")
    basis_rows = tuple(basis_table.to_pylist())
    targets_raw = targets_value.get("targets")
    if not isinstance(targets_raw, list):
        raise TypeError("hybrid targets payload is invalid")
    target_rows = tuple(targets_raw)
    profile_rows = tuple(profiles_table.to_pylist())
    if [row["signed_basis_index"] for row in basis_rows] != list(
        range(len(basis_rows))
    ):
        raise ValueError("hybrid basis index is not canonical and contiguous")
    identities = [
        tuple(row[field] for field in _BASIS_IDENTITY_FIELDS) for row in basis_rows
    ]
    if len(set(identities)) != len(identities):
        raise ValueError("hybrid basis identity is duplicate")
    if any(
        not row["model_id"]
        or not row["model_revision"]
        or row["layer"] < 0
        or row["neuron_index"] < 0
        or row["polarity"] not in {"+", "-"}
        for row in basis_rows
    ):
        raise ValueError("hybrid basis identity is invalid")
    targets = _validate_target_rows(target_rows, source=source, manifest=manifest)
    _validate_profile_rows(profile_rows, targets=targets, basis_rows=basis_rows)
    if manifest.get("artifact_payloads") != _artifact_payload_hashes(target_rows):
        raise ValueError("hybrid artifact payload-set binding drift")
    counts = manifest.get("counts")
    if not isinstance(counts, Mapping):
        raise TypeError("hybrid input counts are invalid")
    if (len(basis_rows), len(target_rows), len(profile_rows)) != (
        counts.get("basis_count"),
        counts.get("target_count"),
        counts.get("profile_row_count"),
    ):
        raise ValueError("hybrid input counts drift")
    width_counts = counts.get("candidate_width_counts")
    expected_width_counts = {
        str(width): sum(row["candidate_count"] == width for row in target_rows)
        for width in (5, 6)
    }
    if width_counts != expected_width_counts:
        raise ValueError("hybrid candidate-width counts drift")
    return HybridInputBundle(
        root, manifest, source, basis_rows, target_rows, profile_rows
    )
