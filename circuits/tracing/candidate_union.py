"""Candidate-specific topology unions with dense fixed-topology measurements."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import pickle
import shutil
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from circuits.tracing.artifact import TopKCompactTraceArtifact
from circuits.tracing.candidates import CandidateLogit, CandidateSelection
from circuits.tracing.clja import FrozenGraphTopology
from circuits.tracing.trace import TopKPositionTrace
from circuits.tracing.utils import NeuronIdx

CANDIDATE_UNION_SCHEMA_VERSION = "adag.candidate-union-trace.v1"
CANDIDATE_UNION_ARTIFACT_SCHEMA_VERSION = "adag.compact-candidate-union.v1"
CANDIDATE_UNION_TRACE_FAMILY_ID = "bonafide.candidate-union.v1"
DATA_FILENAME = "candidate_union.pkl.gz"
MANIFEST_FILENAME = "manifest.json"
METRICS_FILENAME = "metrics.json"

NodeKey = tuple[int, int, int]
EdgeKey = tuple[NodeKey, NodeKey]


def _node_key(row: pd.Series) -> NodeKey:
    return int(row["layer"]), int(row["token"]), int(row["neuron"])


def _edge_key(row: pd.Series) -> EdgeKey:
    values = []
    for column in ("layer", "token", "neuron"):
        parts = str(row[column]).split("->")
        if len(parts) != 2:
            raise ValueError(f"invalid candidate-union edge {column}: {row[column]!r}")
        values.append((int(parts[0]), int(parts[1])))
    return (
        (values[0][0], values[1][0], values[2][0]),
        (values[0][1], values[1][1], values[2][1]),
    )


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


@dataclass(frozen=True)
class CandidateUnionTrace:
    """Exact independent topology union plus dense candidate measurements."""

    trace_family_id: str
    shared_response_position: int
    shared_prediction_position: int
    candidate_selection: CandidateSelection
    df_node: pd.DataFrame
    df_edge: pd.DataFrame
    topology_sha256: str
    source_width1_artifact_id: str
    reference_artifacts: tuple[dict[str, Any], ...]
    refinement_artifacts: tuple[dict[str, Any], ...]

    @property
    def candidate_count(self) -> int:
        return len(self.candidate_selection.candidates)

    def contract_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CANDIDATE_UNION_SCHEMA_VERSION,
            "trace_family_id": self.trace_family_id,
            "shared_response_position": self.shared_response_position,
            "shared_prediction_position": self.shared_prediction_position,
            "candidate_count": self.candidate_count,
            "candidate_selection": self.candidate_selection.to_dict(),
            "topology_sha256": self.topology_sha256,
            "source_width1_artifact_id": self.source_width1_artifact_id,
            "topology_semantics": "exact_union_of_independent_candidate_k1_graphs",
            "measurement_semantics": (
                "candidate_specific_fixed_topology_node_and_edge_rescore"
            ),
        }


@dataclass(frozen=True)
class CandidateUnionArtifact:
    path: Path
    trace: CandidateUnionTrace
    manifest: dict[str, Any]
    metrics: dict[str, Any]


def candidate_selection_from_references(
    references: Sequence[TopKCompactTraceArtifact],
) -> CandidateSelection:
    """Recover the frozen observed-first model-top-five-plus-observed order."""

    if len(references) not in {5, 6}:
        raise ValueError("candidate union requires five or six independent references")
    candidates = []
    observed_ids = set()
    observed_ranks = set()
    for index, artifact in enumerate(references):
        trace = artifact.topk_trace
        if trace.candidate_count != 1:
            raise ValueError("candidate-union references must be independent k1 traces")
        candidate = trace.candidate_selection.candidates[0]
        candidates.append(
            CandidateLogit(
                candidate_index=index,
                full_distribution_rank=candidate.full_distribution_rank,
                token_id=candidate.token_id,
                token_text=candidate.token_text,
                logit=candidate.logit,
                probability=candidate.probability,
                is_observed=candidate.is_observed,
            )
        )
        observed_ids.add(trace.candidate_selection.observed_token_id)
        observed_ranks.add(trace.candidate_selection.observed_token_rank)
    if len(observed_ids) != 1 or len(observed_ranks) != 1:
        raise ValueError("candidate-union references disagree on the observed token")
    if not candidates[0].is_observed or any(
        candidate.is_observed for candidate in candidates[1:]
    ):
        raise ValueError(
            "candidate-union references must place the observed token first"
        )
    if len({candidate.token_id for candidate in candidates}) != len(candidates):
        raise ValueError("candidate-union references contain duplicate tokens")
    expected_alternative_ranks = sorted(
        candidate.full_distribution_rank for candidate in candidates[1:]
    )
    actual_alternative_ranks = [
        candidate.full_distribution_rank for candidate in candidates[1:]
    ]
    if actual_alternative_ranks != expected_alternative_ranks:
        raise ValueError("candidate-union alternatives are not rank ordered")
    observed = candidates[0]
    return CandidateSelection(
        policy_id="model_top5_plus_observed",
        policy_version="1",
        ordering_rule=(
            "observed_first_then_model_top5_descending_logit_then_ascending_token_id"
        ),
        observed_token_id=observed.token_id,
        observed_token_text=observed.token_text,
        observed_token_rank=next(iter(observed_ranks)),
        candidates=tuple(candidates),
    )


def frozen_union_topologies(
    references: Sequence[TopKCompactTraceArtifact],
) -> tuple[str, tuple[FrozenGraphTopology, ...]]:
    """Freeze one exact union and candidate-applicable terminal-edge views."""

    selection = candidate_selection_from_references(references)
    shared_positions = {
        (
            artifact.topk_trace.shared_response_position,
            artifact.topk_trace.shared_prediction_position,
        )
        for artifact in references
    }
    if len(shared_positions) != 1:
        raise ValueError("candidate-union references do not share one position")
    node_union: set[NodeKey] = set()
    edge_union: set[EdgeKey] = set()
    final_layers = set()
    for artifact in references:
        frame = artifact.topk_trace.circuit_data.df_node
        if frame.empty:
            raise ValueError("candidate-union reference graph cannot be empty")
        final_layers.add(int(frame["layer"].max()))
        node_union.update(_node_key(row) for _, row in frame.iterrows())
        edge_union.update(
            _edge_key(row)
            for _, row in artifact.topk_trace.circuit_data.df_edge.iterrows()
        )
    if len(final_layers) != 1:
        raise ValueError("candidate-union references disagree on the final layer")
    final_layer = next(iter(final_layers))
    mlp_nodes = frozenset(
        NeuronIdx(*node) for node in node_union if 0 <= node[0] < final_layer
    )
    topology_value = {
        "nodes": [list(node) for node in sorted(node_union)],
        "edges": [
            [list(source), list(target)] for source, target in sorted(edge_union)
        ],
        "candidate_token_ids": [
            candidate.token_id for candidate in selection.candidates
        ],
    }
    topology_sha256 = _canonical_sha256(topology_value)
    topologies = []
    for candidate in selection.candidates:
        applicable_edges = frozenset(
            (NeuronIdx(*source), NeuronIdx(*target))
            for source, target in edge_union
            if target[0] < final_layer or target[2] == candidate.token_id
        )
        topologies.append(
            FrozenGraphTopology(mlp_nodes=mlp_nodes, edges=applicable_edges)
        )
    return topology_sha256, tuple(topologies)


def _frame_by_node(trace: TopKPositionTrace) -> dict[NodeKey, pd.Series]:
    result = {}
    for _, row in trace.circuit_data.df_node.iterrows():
        key = _node_key(row)
        if key in result:
            raise ValueError(f"duplicate refinement node: {key}")
        result[key] = row
    return result


def _frame_by_edge(trace: TopKPositionTrace) -> dict[EdgeKey, pd.Series]:
    result = {}
    for _, row in trace.circuit_data.df_edge.iterrows():
        key = _edge_key(row)
        if key in result:
            raise ValueError(f"duplicate refinement edge: {key}")
        result[key] = row
    return result


def assemble_candidate_union(
    references: Sequence[TopKCompactTraceArtifact],
    refinements: Sequence[TopKCompactTraceArtifact],
    *,
    topology_sha256: str,
    source_width1_artifact_id: str,
) -> CandidateUnionTrace:
    """Assemble dense node/edge vectors over the exact independent union."""

    if len(references) != len(refinements):
        raise ValueError("candidate-union reference/refinement widths disagree")
    selection = candidate_selection_from_references(references)
    candidate_count = len(selection.candidates)
    ref_nodes = [_frame_by_node(artifact.topk_trace) for artifact in references]
    ref_edges = [_frame_by_edge(artifact.topk_trace) for artifact in references]
    measured_nodes = [_frame_by_node(artifact.topk_trace) for artifact in refinements]
    measured_edges = [_frame_by_edge(artifact.topk_trace) for artifact in refinements]
    node_union = set().union(*(set(frame) for frame in ref_nodes))
    edge_union = set().union(*(set(frame) for frame in ref_edges))
    final_layer = max(node[0] for node in node_union)

    node_rows = []
    for key in sorted(node_union):
        applicable = [
            key[0] < final_layer or key[2] == selection.candidates[index].token_id
            for index in range(candidate_count)
        ]
        attribution = []
        contribution = []
        activation = []
        for index, is_applicable in enumerate(applicable):
            row = measured_nodes[index].get(key)
            if not is_applicable:
                if row is not None:
                    raise ValueError("non-applicable final node was measured")
                attribution.append(None)
                contribution.append(None)
                activation.append(None)
                continue
            if row is None:
                raise ValueError(f"refinement omitted applicable union node: {key}")
            cmap = row["contrib_map"]
            if not isinstance(cmap, (list, tuple)) or len(cmap) != 1:
                raise ValueError("refinement node contribution must have width one")
            attribution.append(float(row["attribution"]))
            contribution.append(float(cmap[0]))
            activation.append(float(row["activation"]))
        node_rows.append(
            {
                "layer": key[0],
                "token": key[1],
                "neuron": key[2],
                "candidate_attribution": attribution,
                "candidate_contribution": contribution,
                "candidate_activation": activation,
                "applicable_by_candidate": applicable,
                "selected_by_candidate": [
                    key in ref_nodes[index] for index in range(candidate_count)
                ],
            }
        )

    edge_rows = []
    for source, target in sorted(edge_union):
        applicable = [
            target[0] < final_layer or target[2] == selection.candidates[index].token_id
            for index in range(candidate_count)
        ]
        attribution = []
        weight = []
        for index, is_applicable in enumerate(applicable):
            row = measured_edges[index].get((source, target))
            if not is_applicable:
                if row is not None:
                    raise ValueError("non-applicable terminal edge was measured")
                attribution.append(None)
                weight.append(None)
                continue
            if row is None:
                raise ValueError(
                    f"refinement omitted applicable union edge: {(source, target)}"
                )
            attribution.append(float(row["attribution"]))
            weight.append(float(row["weight"]))
        edge_rows.append(
            {
                "layer": f"{source[0]}->{target[0]}",
                "token": f"{source[1]}->{target[1]}",
                "neuron": f"{source[2]}->{target[2]}",
                "candidate_attribution": attribution,
                "candidate_weight": weight,
                "applicable_by_candidate": applicable,
                "selected_by_candidate": [
                    (source, target) in ref_edges[index]
                    for index in range(candidate_count)
                ],
            }
        )

    def artifact_record(artifact: TopKCompactTraceArtifact) -> dict[str, Any]:
        return {
            "artifact_id": artifact.manifest.get("artifact_id"),
            "payload_sha256": artifact.manifest["data_sha256"],
            "location": str(artifact.path),
        }

    trace = CandidateUnionTrace(
        trace_family_id=CANDIDATE_UNION_TRACE_FAMILY_ID,
        shared_response_position=references[0].topk_trace.shared_response_position,
        shared_prediction_position=references[0].topk_trace.shared_prediction_position,
        candidate_selection=selection,
        df_node=pd.DataFrame(node_rows),
        df_edge=pd.DataFrame(edge_rows),
        topology_sha256=topology_sha256,
        source_width1_artifact_id=source_width1_artifact_id,
        reference_artifacts=tuple(artifact_record(item) for item in references),
        refinement_artifacts=tuple(artifact_record(item) for item in refinements),
    )
    validate_candidate_union_trace(trace)
    return trace


def _validate_vector(
    value: object,
    applicable: list[bool],
    *,
    field: str,
) -> None:
    if not isinstance(value, (list, tuple)) or len(value) != len(applicable):
        raise ValueError(f"{field} width does not match candidate count")
    for index, (item, is_applicable) in enumerate(zip(value, applicable, strict=True)):
        if not is_applicable:
            if item is not None:
                raise ValueError(f"{field}[{index}] must be null when inapplicable")
            continue
        if (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
        ):
            raise ValueError(f"{field}[{index}] must be finite when applicable")


def validate_candidate_union_trace(trace: CandidateUnionTrace) -> int:
    if not isinstance(trace, CandidateUnionTrace):
        raise TypeError("candidate-union payload has the wrong type")
    candidate_count = trace.candidate_count
    if candidate_count not in {5, 6}:
        raise ValueError("candidate-union width must be five or six")
    if trace.trace_family_id != CANDIDATE_UNION_TRACE_FAMILY_ID:
        raise ValueError("candidate-union trace family is unsupported")
    if trace.candidate_selection.policy_id != "model_top5_plus_observed":
        raise ValueError("candidate-union policy is unsupported")
    if len(trace.topology_sha256) != 64:
        raise ValueError("candidate-union topology hash is invalid")
    if len(trace.reference_artifacts) != candidate_count:
        raise ValueError("candidate-union reference width mismatch")
    if len(trace.refinement_artifacts) != candidate_count:
        raise ValueError("candidate-union refinement width mismatch")

    for frame_name, frame, value_fields in (
        (
            "df_node",
            trace.df_node,
            (
                "candidate_attribution",
                "candidate_contribution",
                "candidate_activation",
            ),
        ),
        (
            "df_edge",
            trace.df_edge,
            ("candidate_attribution", "candidate_weight"),
        ),
    ):
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise ValueError(f"candidate-union {frame_name} must be non-empty")
        required = {
            "layer",
            "token",
            "neuron",
            "applicable_by_candidate",
            "selected_by_candidate",
            *value_fields,
        }
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"candidate-union {frame_name} lacks {sorted(missing)}")
        for row_index, row in frame.iterrows():
            applicable = row["applicable_by_candidate"]
            selected = row["selected_by_candidate"]
            if (
                not isinstance(applicable, (list, tuple))
                or len(applicable) != candidate_count
                or any(not isinstance(value, bool) for value in applicable)
            ):
                raise ValueError(f"{frame_name} applicability mask is invalid")
            if (
                not isinstance(selected, (list, tuple))
                or len(selected) != candidate_count
                or any(not isinstance(value, bool) for value in selected)
            ):
                raise ValueError(f"{frame_name} selection mask is invalid")
            if any(
                was_selected and not is_applicable
                for was_selected, is_applicable in zip(
                    selected, applicable, strict=True
                )
            ):
                raise ValueError(f"{frame_name} selected an inapplicable candidate")
            for field in value_fields:
                _validate_vector(
                    row[field],
                    list(applicable),
                    field=f"{frame_name}.{field}[{row_index}]",
                )
    if trace.df_node.duplicated(["layer", "token", "neuron"]).any():
        raise ValueError("candidate-union nodes are not unique")
    if trace.df_edge.duplicated(["layer", "token", "neuron"]).any():
        raise ValueError("candidate-union edges are not unique")
    return candidate_count


def save_candidate_union_artifact(
    path: str | os.PathLike[str],
    trace: CandidateUnionTrace,
    *,
    manifest: Mapping[str, Any] | None = None,
    metrics: Mapping[str, Any] | None = None,
) -> Path:
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"candidate-union artifact already exists: {target}")
    candidate_count = validate_candidate_union_trace(trace)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        data_path = temporary / DATA_FILENAME
        with gzip.open(data_path, "wb", compresslevel=6) as handle:
            pickle.dump(trace, handle, protocol=pickle.HIGHEST_PROTOCOL)
        payload_sha256 = _sha256_file(data_path)
        canonical_manifest = dict(manifest or {})
        canonical_manifest.update(
            {
                "schema_version": CANDIDATE_UNION_ARTIFACT_SCHEMA_VERSION,
                "created_at": datetime.now(UTC).isoformat(),
                "data_file": DATA_FILENAME,
                "data_sha256": payload_sha256,
                "data_size_bytes": data_path.stat().st_size,
                "candidate_count": candidate_count,
                "node_count": len(trace.df_node),
                "edge_count": len(trace.df_edge),
                "candidate_union_contract": trace.contract_dict(),
                "reference_artifacts": list(trace.reference_artifacts),
                "refinement_artifacts": list(trace.refinement_artifacts),
                "numerically_valid": True,
                "scientifically_reusable": True,
            }
        )
        _write_json(temporary / MANIFEST_FILENAME, canonical_manifest)
        _write_json(temporary / METRICS_FILENAME, dict(metrics or {}))
        os.replace(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target


def load_candidate_union_artifact(
    path: str | os.PathLike[str],
) -> CandidateUnionArtifact:
    artifact_path = Path(path)
    manifest = json.loads((artifact_path / MANIFEST_FILENAME).read_text())
    metrics = json.loads((artifact_path / METRICS_FILENAME).read_text())
    if manifest.get("schema_version") != CANDIDATE_UNION_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("unsupported candidate-union artifact schema")
    data_path = artifact_path / DATA_FILENAME
    if data_path.stat().st_size != manifest.get("data_size_bytes"):
        raise ValueError("candidate-union payload size mismatch")
    if _sha256_file(data_path) != manifest.get("data_sha256"):
        raise ValueError("candidate-union payload checksum mismatch")
    with gzip.open(data_path, "rb") as handle:
        trace = pickle.load(handle)
    validate_candidate_union_trace(trace)
    if manifest.get("candidate_union_contract") != trace.contract_dict():
        raise ValueError("candidate-union manifest contract mismatch")
    return CandidateUnionArtifact(
        path=artifact_path,
        trace=trace,
        manifest=manifest,
        metrics=metrics,
    )
