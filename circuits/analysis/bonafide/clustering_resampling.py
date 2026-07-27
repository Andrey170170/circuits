"""Family-blocked and corpus-checkpoint stability for cluster candidates."""

from __future__ import annotations

import json
import math
import os
import shutil
import uuid
from pathlib import Path
from statistics import median
from typing import Any, Mapping

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from circuits.analysis.bonafide.canonical import (
    canonical_sha256,
    file_sha256,
    load_json_object,
    write_hashed_json,
)
from circuits.analysis.bonafide.cluster_execution import (
    collect_clustering_code_revision,
    collect_clustering_environment,
    fit_sparse_cluster_config,
)
from circuits.analysis.bonafide.clustering_evaluation import (
    _load_assignment_labels,
    _validate_source_plan,
    assignment_ari,
    load_cluster_states,
)
from circuits.analysis.bonafide.clustering_store import (
    FeatureStoreReader,
    load_pair_evidence,
)

RESAMPLE_PLAN_SCHEMA = "adag.bonafide.clustering-resample-plan.v1"
RESAMPLE_STATE_SCHEMA = "adag.bonafide.clustering-resample-state.v1"
RESAMPLE_REPORT_SCHEMA = "adag.bonafide.clustering-resample-report.v1"

RESAMPLE_ASSIGNMENT_SCHEMA = pa.schema(
    [
        pa.field("signed_basis_index", pa.int64(), nullable=False),
        pa.field("cluster_id", pa.int32(), nullable=True),
        pa.field("eligible", pa.bool_(), nullable=False),
        pa.field("assigned", pa.bool_(), nullable=False),
        pa.field("target_count", pa.int64(), nullable=False),
        pa.field("response_count", pa.int64(), nullable=False),
        pa.field("family_count", pa.int64(), nullable=False),
    ]
)


def _validated_hashed_object(
    path: Path,
    *,
    hash_field: str,
) -> dict[str, Any]:
    value = load_json_object(path)
    core = dict(value)
    recorded_hash = core.pop(hash_field, None)
    if recorded_hash != canonical_sha256(core):
        raise ValueError(f"{path.name} hash mismatch")
    return value


def _checkpoint_family_sets(
    target_rows: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    count_by_family: dict[str, int] = {}
    for row in target_rows:
        family_id = str(row["base_question_id"])
        count_by_family[family_id] = count_by_family.get(family_id, 0) + 1
    ordered_families = sorted(count_by_family)
    cumulative: list[int] = []
    total = 0
    for family_id in ordered_families:
        total += count_by_family[family_id]
        cumulative.append(total)
    checkpoints: list[dict[str, Any]] = []
    used_prefixes: set[int] = set()
    for requested_target_count in (500, 1000, 1500):
        prefix_count = min(
            range(2, len(ordered_families)),
            key=lambda count: (
                abs(cumulative[count - 1] - requested_target_count),
                count,
            ),
        )
        if prefix_count in used_prefixes:
            continue
        used_prefixes.add(prefix_count)
        checkpoints.append(
            {
                "kind": "checkpoint",
                "name": f"checkpoint-{requested_target_count}",
                "requested_target_count": requested_target_count,
                "included_family_ids": ordered_families[:prefix_count],
                "selected_target_count": cumulative[prefix_count - 1],
            }
        )
    return checkpoints


def build_resample_plan(
    *,
    repo_root: Path,
    source_plan_path: Path,
    structural_report_path: Path,
    projection_manifest_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    source_plan = _validate_source_plan(source_plan_path)
    structural = _validated_hashed_object(
        structural_report_path,
        hash_field="report_sha256",
    )
    projection = _validated_hashed_object(
        projection_manifest_path,
        hash_field="manifest_sha256",
    )
    if structural["source_plan"]["plan_sha256"] != source_plan["plan_sha256"]:
        raise ValueError("structural report source-plan drift")
    if projection["source_plan"]["plan_sha256"] != source_plan["plan_sha256"]:
        raise ValueError("projection source-plan drift")
    if (
        projection["source_structural_report"]["report_sha256"]
        != structural["report_sha256"]
    ):
        raise ValueError("projection structural-report drift")
    revision = collect_clustering_code_revision(repo_root)
    if revision["git_dirty"]:
        raise ValueError("refuse to freeze resample plan from dirty source")

    feature_reader = FeatureStoreReader(Path(str(source_plan["feature_store"]["path"])))
    target_rows = list(feature_reader.target_rows)
    family_ids = sorted({str(row["base_question_id"]) for row in target_rows})
    selections: list[dict[str, Any]] = [
        {
            "kind": "leave_one_family_out",
            "name": f"leave-out-{family_id}",
            "excluded_family_ids": [family_id],
        }
        for family_id in family_ids
    ]
    selections.extend(_checkpoint_family_sets(target_rows))
    evidence_tasks: list[dict[str, Any]] = []
    for task_index, selection in enumerate(selections):
        item = {
            "task_index": task_index,
            **selection,
            "output_path": str(
                output_root.resolve() / "pair-evidence" / f"task-{task_index:03d}"
            ),
        }
        item["selection_sha256"] = canonical_sha256(item)
        evidence_tasks.append(item)

    candidate_tasks = [
        int(candidate["task_index"]) for candidate in projection["candidates"]
    ]
    source_states = load_cluster_states(source_plan)
    fit_tasks: list[dict[str, Any]] = []
    for evidence_task in evidence_tasks:
        for source_task_index in candidate_tasks:
            task_index = len(fit_tasks)
            config = dict(source_states[source_task_index].config)
            item = {
                "task_index": task_index,
                "evidence_task_index": int(evidence_task["task_index"]),
                "source_task_index": source_task_index,
                "source_state_manifest_sha256": source_states[
                    source_task_index
                ].manifest["manifest_sha256"],
                "config": config,
                "output_path": str(
                    output_root.resolve() / "cluster-states" / f"task-{task_index:03d}"
                ),
            }
            item["fit_task_sha256"] = canonical_sha256(item)
            fit_tasks.append(item)
    plan: dict[str, Any] = {
        "schema_version": RESAMPLE_PLAN_SCHEMA,
        "repo_root": str(repo_root.resolve()),
        "source_plan": {
            "path": str(source_plan_path.resolve()),
            "plan_sha256": source_plan["plan_sha256"],
        },
        "source_structural_report": {
            "path": str(structural_report_path.resolve()),
            "report_sha256": structural["report_sha256"],
        },
        "source_projection": {
            "path": str(projection_manifest_path.resolve()),
            "manifest_sha256": projection["manifest_sha256"],
        },
        "feature_store": dict(source_plan["feature_store"]),
        "output_root": str(output_root.resolve()),
        "family_ids": family_ids,
        "candidate_source_task_indices": candidate_tasks,
        "evidence_tasks": evidence_tasks,
        "fit_tasks": fit_tasks,
        "code_revision": revision,
        "environment": collect_clustering_environment(),
        "descriptions_generated": False,
    }
    plan["plan_sha256"] = canonical_sha256(plan)
    return plan


def write_resample_plan(path: Path, plan: Mapping[str, Any]) -> None:
    write_hashed_json(path, plan, hash_field="plan_sha256")


def validate_resample_plan(
    plan: Mapping[str, Any],
    *,
    repo_root: Path,
    verify_code: bool,
) -> dict[str, Any]:
    validated = dict(plan)
    core = dict(validated)
    recorded_hash = core.pop("plan_sha256", None)
    if recorded_hash != canonical_sha256(core):
        raise ValueError("resample plan hash mismatch")
    if validated.get("schema_version") != RESAMPLE_PLAN_SCHEMA:
        raise ValueError("unsupported resample plan schema")
    if Path(str(validated["repo_root"])).resolve() != repo_root.resolve():
        raise ValueError("resample plan belongs to another worktree")
    source_plan = _validate_source_plan(Path(str(validated["source_plan"]["path"])))
    if source_plan["plan_sha256"] != validated["source_plan"]["plan_sha256"]:
        raise ValueError("resample source-plan drift")
    structural = _validated_hashed_object(
        Path(str(validated["source_structural_report"]["path"])),
        hash_field="report_sha256",
    )
    if (
        structural["report_sha256"]
        != validated["source_structural_report"]["report_sha256"]
    ):
        raise ValueError("resample structural-report drift")
    projection = _validated_hashed_object(
        Path(str(validated["source_projection"]["path"])),
        hash_field="manifest_sha256",
    )
    if (
        projection["manifest_sha256"]
        != validated["source_projection"]["manifest_sha256"]
    ):
        raise ValueError("resample projection drift")
    for task_index, task in enumerate(validated["evidence_tasks"]):
        if int(task["task_index"]) != task_index:
            raise ValueError("resample evidence task ordering is invalid")
        unhashed = dict(task)
        recorded = unhashed.pop("selection_sha256", None)
        if recorded != canonical_sha256(unhashed):
            raise ValueError("resample evidence task hash mismatch")
    for task_index, task in enumerate(validated["fit_tasks"]):
        if int(task["task_index"]) != task_index:
            raise ValueError("resample fit task ordering is invalid")
        unhashed = dict(task)
        recorded = unhashed.pop("fit_task_sha256", None)
        if recorded != canonical_sha256(unhashed):
            raise ValueError("resample fit task hash mismatch")
    if verify_code:
        if collect_clustering_code_revision(repo_root) != validated["code_revision"]:
            raise ValueError("resample executable source has drifted")
    return validated


def fit_resample_task(
    plan: Mapping[str, Any],
    *,
    repo_root: Path,
    task_index: int,
) -> dict[str, Any]:
    validated = validate_resample_plan(
        plan,
        repo_root=repo_root,
        verify_code=True,
    )
    if task_index < 0 or task_index >= len(validated["fit_tasks"]):
        raise ValueError("resample fit task index is out of range")
    task = validated["fit_tasks"][task_index]
    evidence_task = validated["evidence_tasks"][int(task["evidence_task_index"])]
    evidence, support = load_pair_evidence(Path(str(evidence_task["output_path"])))
    result, eligible = fit_sparse_cluster_config(
        evidence=evidence,
        support=support,
        config=task["config"],
    )
    output_path = Path(str(task["output_path"]))
    if output_path.exists():
        manifest = _validated_hashed_object(
            output_path / "manifest.json",
            hash_field="manifest_sha256",
        )
        if manifest.get("fit_task_sha256") != task["fit_task_sha256"]:
            raise ValueError("existing resample state belongs to another task")
        return manifest
    rows = [
        {
            "signed_basis_index": basis_index,
            "cluster_id": (
                int(result.labels[basis_index])
                if int(result.labels[basis_index]) >= 0
                else None
            ),
            "eligible": bool(eligible[basis_index]),
            "assigned": int(result.labels[basis_index]) >= 0,
            "target_count": int(support.target_counts[basis_index]),
            "response_count": int(support.response_counts[basis_index]),
            "family_count": int(support.family_counts[basis_index]),
        }
        for basis_index in range(evidence.basis_count)
    ]
    temporary = output_path.parent / f".{output_path.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir(parents=True)
    try:
        assignment_path = temporary / "assignments.parquet"
        pq.write_table(
            pa.Table.from_pylist(rows, schema=RESAMPLE_ASSIGNMENT_SCHEMA),
            assignment_path,
            compression="zstd",
            use_dictionary=True,
            write_statistics=True,
        )
        evidence_manifest = load_json_object(
            Path(str(evidence_task["output_path"])) / "manifest.json"
        )
        manifest: dict[str, Any] = {
            "schema_version": RESAMPLE_STATE_SCHEMA,
            "plan_sha256": validated["plan_sha256"],
            "fit_task_sha256": task["fit_task_sha256"],
            "task_index": task_index,
            "evidence_task_index": task["evidence_task_index"],
            "source_task_index": task["source_task_index"],
            "source_state_manifest_sha256": task["source_state_manifest_sha256"],
            "source_pair_evidence_manifest_sha256": evidence_manifest[
                "manifest_sha256"
            ],
            "config": task["config"],
            "eligible_basis_count": int(eligible.sum()),
            "assigned_basis_count": int(result.active_mask.sum()),
            "cluster_sizes": {
                str(key): value for key, value in result.cluster_sizes.items()
            },
            "connected_component_count": result.connected_component_count,
            "eigenvalues": result.eigenvalues.tolist(),
            "descriptions_generated": False,
            "assignment_file": {
                "path": assignment_path.name,
                "size_bytes": assignment_path.stat().st_size,
                "sha256": file_sha256(assignment_path),
                "row_count": len(rows),
            },
        }
        manifest["manifest_sha256"] = canonical_sha256(manifest)
        with (temporary / "manifest.json").open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        output_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, output_path)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def build_resample_report(plan_path: Path, *, repo_root: Path) -> dict[str, Any]:
    plan = validate_resample_plan(
        load_json_object(plan_path),
        repo_root=repo_root,
        verify_code=True,
    )
    source_plan = _validate_source_plan(Path(str(plan["source_plan"]["path"])))
    source_states = load_cluster_states(source_plan)
    comparisons: dict[int, dict[str, list[dict[str, Any]]]] = {
        task_index: {"leave_one_family_out": [], "checkpoint": []}
        for task_index in plan["candidate_source_task_indices"]
    }
    for task in plan["fit_tasks"]:
        output_path = Path(str(task["output_path"]))
        manifest = _validated_hashed_object(
            output_path / "manifest.json",
            hash_field="manifest_sha256",
        )
        if manifest["fit_task_sha256"] != task["fit_task_sha256"]:
            raise ValueError("resample state task drift")
        file_record = manifest["assignment_file"]
        assignment_path = output_path / str(file_record["path"])
        if file_sha256(assignment_path) != file_record["sha256"]:
            raise ValueError("resample assignment file hash drift")
        labels = _load_assignment_labels(
            assignment_path,
            basis_count=len(source_states[int(task["source_task_index"])].labels),
        )
        source_task_index = int(task["source_task_index"])
        evidence_task = plan["evidence_tasks"][int(task["evidence_task_index"])]
        comparisons[source_task_index][str(evidence_task["kind"])].append(
            {
                "fit_task_index": int(task["task_index"]),
                "evidence_task_index": int(task["evidence_task_index"]),
                "selection_name": evidence_task["name"],
                "ari_on_shared_assigned_bases": assignment_ari(
                    source_states[source_task_index].labels,
                    labels,
                ),
                "assigned_basis_count": int(manifest["assigned_basis_count"]),
            }
        )
    candidate_records: list[dict[str, Any]] = []
    for source_task_index in plan["candidate_source_task_indices"]:
        records = comparisons[int(source_task_index)]
        jackknife_values = [
            float(record["ari_on_shared_assigned_bases"])
            for record in records["leave_one_family_out"]
        ]
        checkpoint_values = [
            float(record["ari_on_shared_assigned_bases"])
            for record in records["checkpoint"]
        ]
        jackknife_median = median(jackknife_values)
        jackknife_p10 = float(np.quantile(jackknife_values, 0.1))
        candidate_records.append(
            {
                "source_task_index": int(source_task_index),
                "n_clusters": int(
                    source_states[int(source_task_index)].config["n_clusters"]
                ),
                "leave_one_family_out": records["leave_one_family_out"],
                "checkpoint": records["checkpoint"],
                "family_jackknife_median_ari": jackknife_median,
                "family_jackknife_p10_ari": jackknife_p10,
                "checkpoint_median_ari": (
                    median(checkpoint_values) if checkpoint_values else None
                ),
                "passes_family_jackknife_gate": (
                    jackknife_median >= 0.60 and jackknife_p10 >= 0.45
                ),
            }
        )
    report: dict[str, Any] = {
        "schema_version": RESAMPLE_REPORT_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "candidate_count": len(candidate_records),
        "candidates": candidate_records,
        "descriptions_generated": False,
    }
    report["report_sha256"] = canonical_sha256(report)
    return report


def write_resample_report(path: Path, report: Mapping[str, Any]) -> None:
    write_hashed_json(path, report, hash_field="report_sha256")
