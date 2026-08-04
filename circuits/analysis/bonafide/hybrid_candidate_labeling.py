"""Freeze evaluated hybrid candidate states for the provider-neutral labeling runtime."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.sparse import load_npz

from circuits.analysis.bonafide.candidate_labeling_comparison import select_w_anchors
from circuits.analysis.bonafide.canonical import canonical_sha256, file_sha256
from circuits.analysis.bonafide.hybrid_candidate_clustering import REPRESENTATION_IDS
from circuits.analysis.bonafide.hybrid_candidate_clustering_execution import (
    load_hybrid_clustering_manifest,
)
from circuits.analysis.bonafide.hybrid_candidate_inputs import load_hybrid_input_bundle
from circuits.analysis.bonafide.hybrid_candidate_labelability import (
    load_hybrid_candidate_labelability,
)
from circuits.labeling.config import (
    HYBRID_CANDIDATE_RECIPE_ID,
    HYBRID_CANDIDATE_RECIPE_PATH,
    load_recipe,
)
from circuits.labeling.evidence import (
    EVIDENCE_SCHEMA,
    MASTER_SCHEMA,
    STATE_SCHEMA,
    load_frozen_bundle,
)

HYBRID_LABELING_BUNDLE_SCHEMA = MASTER_SCHEMA
WITNESS_INVENTORY_SCHEMA = "adag.bonafide.hybrid-candidate-witness-inventory.v1"
STATE_ROLES = ("primary", "alternative")
BRIDGE_SOURCE_PATHS = (
    "circuits/analysis/bonafide/hybrid_candidate_labeling.py",
    "circuits/labeling/api.py",
    "circuits/labeling/batch.py",
    "circuits/labeling/batch_runtime.py",
    "circuits/labeling/config.py",
    "circuits/labeling/cost_guard.py",
    "circuits/labeling/evidence.py",
    "circuits/labeling/io.py",
    "circuits/labeling/pricing.py",
    "circuits/labeling/profiles.py",
    "circuits/labeling/provenance.py",
    "circuits/labeling/quality.py",
    "circuits/labeling/runtime.py",
    "circuits/labeling/scoring.py",
    "circuits/labeling/schema.py",
    "docs/HYBRID_CANDIDATE_LABELABILITY_PROTOCOL.md",
    "scripts/bonafide/hybrid_candidate_labeling.py",
    "scripts/bonafide/labeling_pipeline.py",
    "scripts/bonafide/configs/labeling/openai-hybrid-candidate-v1.json",
)
ASSIGNMENT_SCHEMA = pa.schema(
    [
        pa.field("signed_basis_index", pa.int64(), nullable=False),
        pa.field("model_id", pa.string(), nullable=False),
        pa.field("model_revision", pa.string(), nullable=False),
        pa.field("layer", pa.int32(), nullable=False),
        pa.field("neuron_index", pa.int64(), nullable=False),
        pa.field("polarity", pa.string(), nullable=False),
        pa.field("assigned", pa.bool_(), nullable=False),
        pa.field("cluster_id", pa.int32(), nullable=True),
    ]
)


def collect_bridge_revision(repo_root: Path) -> dict[str, Any]:
    """Bind a clean commit and every executable bridge or prompt source."""

    repo_root = repo_root.resolve()

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    status = git("status", "--porcelain=v1", "--untracked-files=no")
    if status:
        raise ValueError("hybrid labeling bridge requires a clean tracked worktree")
    files: list[dict[str, str]] = []
    digest = hashlib.sha256()
    for relative in BRIDGE_SOURCE_PATHS:
        path = repo_root / relative
        if git("ls-files", "--error-unmatch", "--", relative) != relative:
            raise ValueError(f"hybrid labeling source is not tracked: {relative}")
        if git("hash-object", relative) != git("rev-parse", f"HEAD:{relative}"):
            raise ValueError(f"hybrid labeling source differs from HEAD: {relative}")
        content = path.read_bytes()
        encoded = relative.encode()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
        files.append({"path": relative, "sha256": file_sha256(path)})
    return {
        "repo_root": str(repo_root),
        "git_commit": git("rev-parse", "HEAD"),
        "git_tree": git("rev-parse", "HEAD^{tree}"),
        "tracked_worktree_clean": True,
        "source_tree_sha256": digest.hexdigest(),
        "files": files,
    }


def _expected_recipe_binding(repo_root: Path) -> dict[str, str]:
    recipe_path = repo_root / HYBRID_CANDIDATE_RECIPE_PATH
    recipe = load_recipe(recipe_path)
    if (
        recipe.recipe_id != HYBRID_CANDIDATE_RECIPE_ID
        or recipe.prompt_policy != "hybrid_candidate_v1"
    ):
        raise ValueError("frozen hybrid labeling recipe identity drift")
    return {
        "recipe_id": HYBRID_CANDIDATE_RECIPE_ID,
        "path": HYBRID_CANDIDATE_RECIPE_PATH,
        "sha256": file_sha256(recipe_path),
    }


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False))
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _load_witness_inventory(root: Path) -> Mapping[str, Any]:
    value = json.loads((root / "witness-inventory.json").read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or value.get("schema_version") != WITNESS_INVENTORY_SCHEMA:
        raise ValueError("hybrid witness inventory schema drift")
    states = value.get("states")
    if not isinstance(states, Mapping) or set(states) != set(STATE_ROLES):
        raise ValueError("hybrid witness inventory state roles drift")
    return value


def _state_identity(state: Mapping[str, Any]) -> dict[str, Any]:
    required = ("representation", "affinity_mode", "n_clusters", "seed")
    if any(field not in state for field in required):
        raise ValueError("hybrid state identity is incomplete")
    result = {field: state[field] for field in required}
    if result["n_clusters"] != 64:
        raise ValueError("hybrid labeling states must use K=64")
    return result


def _selected_assignments(
    *, fit_root: Path, state: Mapping[str, Any], basis_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    identity = _state_identity(state)
    table = pq.read_table(fit_root / "assignments.parquet")
    rows = [
        row
        for row in table.to_pylist()
        if row["representation"] == identity["representation"]
        and row["affinity_mode"] == identity["affinity_mode"]
        and row["n_clusters"] == identity["n_clusters"]
        and row["seed"] == identity["seed"]
    ]
    if len(rows) != len(basis_rows):
        raise ValueError("hybrid selected assignment does not cover the basis universe")
    by_index = {int(row["signed_basis_index"]): row for row in rows}
    if set(by_index) != set(range(len(basis_rows))):
        raise ValueError("hybrid selected assignment basis indices drift")
    result: list[dict[str, Any]] = []
    for index, basis in enumerate(basis_rows):
        assignment = by_index[index]
        result.append(
            {
                "signed_basis_index": index,
                "model_id": basis["model_id"],
                "model_revision": basis["model_revision"],
                "layer": basis["layer"],
                "neuron_index": basis["neuron_index"],
                "polarity": basis["polarity"],
                "assigned": assignment["assigned"],
                "cluster_id": assignment["cluster_id"],
            }
        )
    return result


def _width_one_manifests(
    target_rows: Sequence[Mapping[str, Any]], *, width_root: Path
) -> dict[str, Path]:
    wanted = {str(row["width1_artifact_id"]) for row in target_rows}
    found: dict[str, Path] = {}
    for path in width_root.rglob("manifest.json"):
        if path.parent.name in wanted:
            if path.parent.name in found:
                raise ValueError("duplicate width-one artifact ID")
            found[path.parent.name] = path
    if set(found) != wanted:
        missing = sorted(wanted - set(found))
        raise ValueError(f"width-one artifacts are missing: {missing[:3]}")
    return found


def _candidate_summary(
    *,
    occurrence_rows: Sequence[Mapping[str, Any]],
    member_indices: set[int],
    state: Mapping[str, Any],
    target: Mapping[str, Any],
) -> dict[str, Any]:
    matching = [
        row for row in occurrence_rows if int(row["basis_index"]) in member_indices
    ]
    field = (
        "paper_candidate_values"
        if state["representation"] == "paper_normalized_model_top5.v1"
        else "raw_candidate_values"
    )
    full_values = [list(map(float, row[field])) for row in matching]
    full_width = int(target["candidate_count"])
    if any(len(row) != full_width for row in full_values):
        raise ValueError("candidate summary width drift")
    axis = json.loads(str(target["candidate_selection_json"]))["candidates"]
    if state["representation"] == "paper_normalized_model_top5.v1":
        indices = [
            int(candidate["candidate_index"])
            for candidate in sorted(
                axis,
                key=lambda candidate: (
                    int(candidate["full_distribution_rank"]),
                    int(candidate["candidate_index"]),
                ),
            )[:5]
        ]
        values = [[row[index] for index in indices] for row in full_values]
        axis = [axis[index] for index in indices]
    else:
        values = full_values
    width = len(axis)
    signed_sum = np.sum(np.asarray(values), axis=0).tolist() if values else [0.0] * width
    occurrence_count = sum(int(row["occurrence_count"]) for row in matching)
    return {
        "representation": state["representation"],
        "axis_scope": "target_local_no_cross_target_rank_semantics",
        "candidate_width": width,
        "candidate_axis": axis,
        "matched_signed_basis_count": len(matching),
        "member_occurrence_count": occurrence_count,
        "signed_contribution_sum": signed_sum,
        "signed_contribution_mean_per_occurrence": [
            value / occurrence_count if occurrence_count else 0.0 for value in signed_sum
        ],
        "signed_cancellation_preserved": True,
    }


def _cluster_evidence(
    *,
    role: str,
    authorized: bool,
    state: Mapping[str, Any],
    cluster_record: Mapping[str, Any],
    assignments: Sequence[Mapping[str, Any]],
    target_by_id: Mapping[str, Mapping[str, Any]],
    occurrences_by_target: Mapping[str, Sequence[Mapping[str, Any]]],
    width_manifests: Mapping[str, Path],
    affinity: Any,
) -> dict[str, Any]:
    cluster_id = int(cluster_record["cluster_id"])
    members = [row for row in assignments if row["cluster_id"] == cluster_id]
    indices = np.asarray([int(row["signed_basis_index"]) for row in members], dtype=np.int64)
    strengths = np.asarray(affinity[indices][:, indices].sum(axis=1)).ravel()
    order = np.lexsort((indices, -strengths))[: min(5, len(indices))]
    prototypes = [
        {
            **members[int(local)],
            "internal_affinity_strength": float(strengths[int(local)]),
        }
        for local in order
    ]
    joint_witnesses = cast(Mapping[str, Any], cluster_record["joint_witnesses"])
    exemplars: list[dict[str, Any]] = []
    partition_counts: dict[str, int] = {}
    frozen_witnesses: dict[str, Any] = {}
    for partition in ("generation", "selection_scoring", "audit"):
        partition_record = cast(Mapping[str, Any], joint_witnesses[partition])
        selected = list(map(str, partition_record["frozen_target_ids"]))
        required_count = 8 if partition == "generation" else 4
        if bool(cluster_record["ready"]) and len(selected) != required_count:
            raise ValueError("ready hybrid cluster lacks its exact frozen witnesses")
        if canonical_sha256(selected) != partition_record["frozen_target_ids_sha256"]:
            raise ValueError("frozen hybrid witness hash drift")
        selection_hashes = list(partition_record["frozen_target_selection_hashes"])
        if len(selection_hashes) != len(selected):
            raise ValueError("frozen hybrid witness selection-hash count drift")
        partition_counts[partition] = len(selected)
        frozen_witnesses[partition] = {
            "ordered_target_ids": selected,
            "ordered_target_ids_sha256": partition_record[
                "frozen_target_ids_sha256"
            ],
            "ordered_target_selection_hashes": selection_hashes,
        }
        for target_id in selected:
            target = target_by_id[target_id]
            manifest_path = width_manifests[str(target["width1_artifact_id"])]
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("data_sha256") != target["width1_payload_sha256"]:
                raise ValueError("width-one payload binding drift")
            example = json.loads(str(target["example_json"]))
            exemplars.append(
                {
                    "trace_unit_id": target["width1_artifact_id"],
                    "hybrid_target_id": target_id,
                    "artifact_manifest_path": str(manifest_path),
                    "artifact_manifest_sha256": file_sha256(manifest_path),
                    "artifact_payload_sha256": target["width1_payload_sha256"],
                    "response_id": target["response_id"],
                    "base_question_id": target["base_question_id"],
                    "family_partition": partition,
                    "response_position": int(target["response_position"]),
                    "target_token_text": target["observed_token_text"],
                    "condition": {"diversity": example.get("diversity", {})},
                    "prompt": example["prompt"],
                    "question": example.get("question"),
                    "response": example["response"],
                    "cluster_projection": {
                        "matched_signed_basis_count": sum(
                            int(row["basis_index"]) in set(indices.tolist())
                            for row in occurrences_by_target[target_id]
                        ),
                        "evidence_scope": "input_width_one_plus_candidate_union_summary",
                    },
                    "candidate_union_summary": _candidate_summary(
                        occurrence_rows=occurrences_by_target[target_id],
                        member_indices=set(indices.tolist()),
                        state=state,
                        target=target,
                    ),
                }
            )
    ready = authorized and bool(cluster_record["ready"])
    return {
        "schema_version": EVIDENCE_SCHEMA,
        "state_role": role,
        "cluster_id": cluster_id,
        "labeling_status": "ready" if ready else "state_or_cluster_not_authorized",
        "partition_supported": bool(cluster_record["ready"]),
        "exemplar_partition_counts": partition_counts,
        "member_basis_count": len(members),
        "prototype_signed_bases": prototypes,
        "multiplex_summary": {
            "hybrid_state": _state_identity(state),
            "witness_readiness": cluster_record,
            "candidate_axis_scope": "target_local",
        },
        "frozen_witnesses": frozen_witnesses,
        "balanced_target_exemplars": exemplars,
        "top_recurrent_cluster_edges": [],
        "descriptions_generated": False,
    }


def build_hybrid_labeling_bundle(
    *, evaluation_root: Path, input_root: Path, fit_root: Path, output_root: Path
) -> dict[str, Any]:
    """Publish one standard FrozenBundle without opening any labeling provider."""

    if output_root.exists():
        raise FileExistsError(f"hybrid labeling bundle already exists: {output_root}")
    repo_root = Path(__file__).resolve().parents[3]
    code_revision = collect_bridge_revision(repo_root)
    recipe_binding = _expected_recipe_binding(repo_root)
    evaluation = load_hybrid_candidate_labelability(evaluation_root)
    inputs = load_hybrid_input_bundle(input_root)
    fit = load_hybrid_clustering_manifest(fit_root)
    expected_input_binding = evaluation.get("source_input_binding")
    expected_fit_binding = evaluation.get("source_fit_binding")
    if expected_input_binding != {
        "path": str(inputs.root),
        "schema_version": inputs.manifest["schema_version"],
        "manifest_sha256": inputs.manifest["manifest_sha256"],
    }:
        raise ValueError("passed hybrid input root differs from evaluation binding")
    if expected_fit_binding != {
        "path": str(fit_root.resolve()),
        "schema_version": fit["schema_version"],
        "manifest_sha256": fit["manifest_sha256"],
    }:
        raise ValueError("passed hybrid fit root differs from evaluation binding")
    witness = _load_witness_inventory(evaluation_root)
    state_reports = evaluation.get("states")
    if not isinstance(state_reports, Mapping) or set(state_reports) != set(STATE_ROLES):
        raise ValueError("hybrid evaluation state reports drift")
    witness_states = cast(Mapping[str, Any], witness["states"])

    source_targets = inputs.source_bundle.target_rows
    target_by_id = {str(row["case_id"]): row for row in source_targets}
    width_root = Path(str(inputs.source_bundle.manifest["inputs"]["width1_root"]))
    width_manifests = _width_one_manifests(source_targets, width_root=width_root)
    occurrence_rows = pq.read_table(evaluation_root / "occurrences.parquet").to_pylist()
    occurrences_by_target: dict[str, list[Mapping[str, Any]]] = {}
    for row in occurrence_rows:
        occurrences_by_target.setdefault(str(row["target_id"]), []).append(row)
    if set(occurrences_by_target) != set(target_by_id):
        raise ValueError("evaluation occurrences and frozen target inventory differ")

    temporary = output_root.parent / f".{output_root.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir(parents=True)
    selected_manifests: list[dict[str, Any]] = []
    try:
        for role in STATE_ROLES:
            report = cast(Mapping[str, Any], state_reports[role])
            inventory_state = cast(Mapping[str, Any], witness_states[role])
            state = cast(Mapping[str, Any], inventory_state["state"])
            authorized = report.get("exploratory_labeling_authorized") is True
            assignments = _selected_assignments(
                fit_root=fit_root, state=state, basis_rows=inputs.basis_rows
            )
            cluster_records = cast(Sequence[Mapping[str, Any]], inventory_state["clusters"])
            if sorted(int(row["cluster_id"]) for row in cluster_records) != list(range(64)):
                raise ValueError("hybrid witness cluster IDs are not contiguous")
            representation_key = next(
                key
                for key, value in REPRESENTATION_IDS.items()
                if value == state["representation"]
            )
            affinity_name = cast(Mapping[str, str], fit["affinity_files"])[
                f"{representation_key}:{state['affinity_mode']}"
            ]
            affinity = load_npz(fit_root / affinity_name).tocsr()
            evidence_rows = [
                _cluster_evidence(
                    role=role,
                    authorized=authorized,
                    state=state,
                    cluster_record=record,
                    assignments=assignments,
                    target_by_id=target_by_id,
                    occurrences_by_target=occurrences_by_target,
                    width_manifests=width_manifests,
                    affinity=affinity,
                )
                for record in cluster_records
            ]
            ready_records = [
                record
                for record in cluster_records
                if authorized and bool(record["ready"])
            ]
            recommended = (
                select_w_anchors(
                    member_counts={
                        int(record["cluster_id"]): sum(
                            row["cluster_id"] == int(record["cluster_id"])
                            for row in assignments
                        )
                        for record in ready_records
                    },
                    generation_target_counts={
                        int(record["cluster_id"]): int(
                            record["joint_witnesses"]["generation"]["target_count"]
                        )
                        for record in ready_records
                    },
                )
                if authorized
                else None
            )
            state_root = temporary / role
            state_root.mkdir()
            assignment_path = state_root / "assignments.parquet"
            pq.write_table(
                pa.Table.from_pylist(assignments, schema=ASSIGNMENT_SCHEMA),
                assignment_path,
                compression="zstd",
            )
            evidence_path = state_root / "labeling-evidence.jsonl"
            _write_jsonl(evidence_path, evidence_rows)
            inventory_path = state_root / "witness-inventory.json"
            inventory_path.write_text(
                json.dumps(witness, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            state_manifest: dict[str, Any] = {
                "schema_version": STATE_SCHEMA,
                "state_role": role,
                "selection_status": (
                    "exploratory_labeling_authorized"
                    if authorized
                    else "exploratory_labeling_not_authorized"
                ),
                "source_state": _state_identity(state),
                "source_evaluation": {
                    "path": str(evaluation_root.resolve()),
                    "manifest_sha256": evaluation["manifest_sha256"],
                },
                "source_fit": {
                    "path": str(fit_root.resolve()),
                    "manifest_sha256": fit["manifest_sha256"],
                },
                "source_inputs": {
                    "path": str(input_root.resolve()),
                    "manifest_sha256": inputs.manifest["manifest_sha256"],
                },
                "cluster_count": 64,
                "ready_cluster_count": sum(
                    row["labeling_status"] == "ready" for row in evidence_rows
                ),
                "recommended_cluster_selection": recommended,
                "exploratory_labeling_authorized": authorized,
                "scientific_promotion_authorized": False,
                "prompt_policy": "hybrid_candidate_v1",
                "files": [
                    _file_record(path)
                    for path in (assignment_path, evidence_path, inventory_path)
                ],
            }
            state_manifest["manifest_sha256"] = canonical_sha256(state_manifest)
            (state_root / "manifest.json").write_text(
                json.dumps(state_manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            selected_manifests.append(state_manifest)
        master: dict[str, Any] = {
            "schema_version": HYBRID_LABELING_BUNDLE_SCHEMA,
            "purpose": "provider_neutral_hybrid_candidate_labeling_evidence",
            "source_evaluation": {
                "path": str(evaluation_root.resolve()),
                "manifest_sha256": evaluation["manifest_sha256"],
            },
            "code_revision": code_revision,
            "labeling_recipe": recipe_binding,
            "selected_states": selected_manifests,
            "prompt_policy": "hybrid_candidate_v1",
            "partition_firewall": {
                "generation_prompt_eligible": True,
                "selection_scoring_summary_only": True,
                "audit_prompt_eligible": False,
                "model_calls_made": False,
            },
            "scientific_promotion_authorized": False,
        }
        master["manifest_sha256"] = canonical_sha256(master)
        (temporary / "manifest.json").write_text(
            json.dumps(master, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output_root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return master


def load_hybrid_labeling_bundle(root: Path) -> Mapping[str, Any]:
    """Deep-load a hybrid adapter artifact and all of its immutable sources."""

    bundle = load_frozen_bundle(root)
    manifest = bundle.manifest
    if (
        manifest.get("purpose")
        != "provider_neutral_hybrid_candidate_labeling_evidence"
        or manifest.get("prompt_policy") != "hybrid_candidate_v1"
        or manifest.get("scientific_promotion_authorized") is not False
        or manifest.get("partition_firewall")
        != {
            "generation_prompt_eligible": True,
            "selection_scoring_summary_only": True,
            "audit_prompt_eligible": False,
            "model_calls_made": False,
        }
    ):
        raise ValueError("hybrid labeling bundle contract drift")
    source_evaluation = cast(Mapping[str, Any], manifest["source_evaluation"])
    revision = manifest.get("code_revision")
    repo_root = Path(__file__).resolve().parents[3]
    if not isinstance(revision, Mapping) or revision != collect_bridge_revision(repo_root):
        raise ValueError("hybrid labeling bridge code revision drift")
    if manifest.get("labeling_recipe") != _expected_recipe_binding(repo_root):
        raise ValueError("hybrid labeling recipe binding drift")
    evaluation = load_hybrid_candidate_labelability(Path(str(source_evaluation["path"])))
    if evaluation["manifest_sha256"] != source_evaluation["manifest_sha256"]:
        raise ValueError("hybrid labeling evaluation binding drift")
    reports = cast(Mapping[str, Any], evaluation["states"])
    evaluation_input_binding = cast(
        Mapping[str, Any], evaluation["source_input_binding"]
    )
    evaluation_fit_binding = cast(
        Mapping[str, Any], evaluation["source_fit_binding"]
    )
    evaluation_inputs = load_hybrid_input_bundle(
        Path(str(cast(Mapping[str, Any], evaluation["source_input_binding"])["path"]))
    )
    target_by_id = {
        str(row["case_id"]): row for row in evaluation_inputs.source_bundle.target_rows
    }
    width_manifests = _width_one_manifests(
        evaluation_inputs.source_bundle.target_rows,
        width_root=Path(
            str(evaluation_inputs.source_bundle.manifest["inputs"]["width1_root"])
        ),
    )
    occurrence_rows = pq.read_table(
        Path(str(source_evaluation["path"])) / "occurrences.parquet"
    ).to_pylist()
    occurrences_by_target: dict[str, list[Mapping[str, Any]]] = {}
    for occurrence in occurrence_rows:
        occurrences_by_target.setdefault(str(occurrence["target_id"]), []).append(
            occurrence
        )
    for role in STATE_ROLES:
        state = bundle.states[role]
        state_manifest = state.manifest
        if (
            state_manifest.get("prompt_policy") != "hybrid_candidate_v1"
            or state_manifest.get("scientific_promotion_authorized") is not False
            or state_manifest.get("exploratory_labeling_authorized")
            is not (cast(Mapping[str, Any], reports[role]).get(
                "exploratory_labeling_authorized"
            ) is True)
            or state_manifest.get("ready_cluster_count")
            != len(state.ready_cluster_ids)
        ):
            raise ValueError(f"hybrid labeling {role} authorization drift")
        source_inputs = cast(Mapping[str, Any], state_manifest["source_inputs"])
        if source_inputs != {
            "path": evaluation_input_binding["path"],
            "manifest_sha256": evaluation_input_binding["manifest_sha256"],
        }:
            raise ValueError(
                f"hybrid labeling {role} input differs from evaluation binding"
            )
        inputs = load_hybrid_input_bundle(Path(str(source_inputs["path"])))
        if inputs.manifest["manifest_sha256"] != source_inputs["manifest_sha256"]:
            raise ValueError("hybrid labeling input binding drift")
        source_fit = cast(Mapping[str, Any], state_manifest["source_fit"])
        if source_fit != {
            "path": evaluation_fit_binding["path"],
            "manifest_sha256": evaluation_fit_binding["manifest_sha256"],
        }:
            raise ValueError(
                f"hybrid labeling {role} fit differs from evaluation binding"
            )
        fit = load_hybrid_clustering_manifest(Path(str(source_fit["path"])))
        if fit["manifest_sha256"] != source_fit["manifest_sha256"]:
            raise ValueError("hybrid labeling fit binding drift")
        assignment_table = pq.read_table(state.assignments_path)
        if not assignment_table.schema.equals(ASSIGNMENT_SCHEMA, check_metadata=False):
            raise ValueError("hybrid labeling assignment schema drift")
        source_state = cast(Mapping[str, Any], state_manifest["source_state"])
        expected_assignments = _selected_assignments(
            fit_root=Path(str(source_fit["path"])),
            state=source_state,
            basis_rows=inputs.basis_rows,
        )
        assignment_rows = assignment_table.to_pylist()
        if assignment_rows != expected_assignments:
            raise ValueError("hybrid labeling assignment content drift")
        inventory = _load_witness_inventory(state.root)
        inventory_state = cast(Mapping[str, Any], inventory["states"])[role]
        if inventory_state != _load_witness_inventory(
            Path(str(source_evaluation["path"]))
        )["states"][role]:
            raise ValueError("hybrid labeling witness inventory binding drift")
        authorized = state_manifest["exploratory_labeling_authorized"]
        ready_records = [
            record
            for record in cast(Sequence[Mapping[str, Any]], inventory_state["clusters"])
            if authorized and bool(record["ready"])
        ]
        expected_recommendation = (
            select_w_anchors(
                member_counts={
                    int(record["cluster_id"]): sum(
                        row["cluster_id"] == int(record["cluster_id"])
                        for row in assignment_rows
                    )
                    for record in ready_records
                },
                generation_target_counts={
                    int(record["cluster_id"]): int(
                        record["joint_witnesses"]["generation"]["target_count"]
                    )
                    for record in ready_records
                },
            )
            if authorized
            else None
        )
        if state_manifest.get("recommended_cluster_selection") != expected_recommendation:
            raise ValueError("hybrid labeling recommended cluster selection drift")
        representation_key = next(
            key
            for key, value in REPRESENTATION_IDS.items()
            if value == source_state["representation"]
        )
        affinity_name = cast(Mapping[str, str], fit["affinity_files"])[
            f"{representation_key}:{source_state['affinity_mode']}"
        ]
        affinity = load_npz(Path(str(source_fit["path"])) / affinity_name).tocsr()
        for cluster_id, row in state.evidence.items():
            cluster_record = cast(
                Sequence[Mapping[str, Any]], inventory_state["clusters"]
            )[cluster_id]
            source_ready = cluster_record["ready"]
            expected = "ready" if authorized and source_ready else "state_or_cluster_not_authorized"
            if row["labeling_status"] != expected:
                raise ValueError("hybrid labeling evidence authorization drift")
            recomputed = _cluster_evidence(
                role=role,
                authorized=bool(authorized),
                state=source_state,
                cluster_record=cluster_record,
                assignments=assignment_rows,
                target_by_id=target_by_id,
                occurrences_by_target=occurrences_by_target,
                width_manifests=width_manifests,
                affinity=affinity,
            )
            if row != recomputed:
                raise ValueError("hybrid labeling evidence content drift")
    return manifest
