"""Immutable execution plans for streaming dense downstream lanes."""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import subprocess
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, cast

from circuits.analysis.bonafide.canonical import (
    canonical_sha256,
    file_sha256,
    load_json_object,
    write_hashed_json,
)
from circuits.analysis.bonafide.inventory import INVENTORY_SCHEMA
from circuits.analysis.bonafide.partition import (
    AnalysisTarget,
    CorpusRole,
    hierarchical_fit_weights,
)

BUILD_PLAN_SCHEMA = "adag.bonafide.downstream-build-plan.v1"
BuildLane = Literal["dense_features", "dense_multiplex"]
BUILD_LANES: tuple[BuildLane, ...] = ("dense_features", "dense_multiplex")


def collect_downstream_code_revision(repo_root: Path) -> dict[str, Any]:
    """Fingerprint the executable downstream source and its scoped Git state."""

    repo_root = repo_root.resolve()

    def git(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    scoped_paths = (
        "circuits/analysis/bonafide",
        "scripts/bonafide/downstream_build.py",
        "scripts/bonafide/downstream_build_joint.py",
        "scripts/bonafide/downstream_build_plan.py",
        "scripts/bonafide/downstream_compact.py",
        "scripts/bonafide/downstream_dense_array.sbatch",
        "scripts/bonafide/downstream_dense_joint_array.sbatch",
        "scripts/bonafide/downstream_dense_compact.sbatch",
        "pyproject.toml",
        "uv.lock",
    )
    status = git(
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        *scoped_paths,
    )
    source_paths: list[Path] = []
    for relative in scoped_paths:
        path = repo_root / relative
        if path.is_dir():
            source_paths.extend(
                candidate
                for candidate in path.rglob("*")
                if candidate.is_file()
                and "__pycache__" not in candidate.parts
                and not candidate.name.endswith(".pyc")
            )
        elif path.is_file():
            source_paths.append(path)
    digest = hashlib.sha256()
    for path in sorted(set(source_paths)):
        relative = path.relative_to(repo_root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(status),
        "git_status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "source_tree_sha256": digest.hexdigest(),
    }


def collect_downstream_environment() -> dict[str, Any]:
    distributions = (
        "circuits",
        "numpy",
        "pandas",
        "pyarrow",
        "scipy",
        "torch",
    )
    packages: dict[str, str | None] = {}
    for distribution in distributions:
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
    }


def _validate_inventory(inventory: Mapping[str, Any]) -> None:
    if inventory.get("schema_version") != INVENTORY_SCHEMA:
        raise ValueError("unsupported downstream inventory schema")
    recorded_hash = inventory.get("inventory_sha256")
    core = dict(inventory)
    core.pop("inventory_sha256", None)
    if recorded_hash != canonical_sha256(core):
        raise ValueError("downstream inventory canonical hash mismatch")


def _dense_records(
    inventory: Mapping[str, Any],
) -> dict[str, list[Mapping[str, Any]]]:
    records_value = inventory.get("records")
    if not isinstance(records_value, list):
        raise ValueError("inventory records must be a list")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for value in records_value:
        if not isinstance(value, Mapping):
            raise ValueError("inventory target record must be an object")
        record = cast(Mapping[str, Any], value)
        if (
            record.get("status") == "discovery"
            and record.get("corpus_role") == CorpusRole.DENSE_DISCOVERY.value
        ):
            response_id = record.get("response_id")
            if not isinstance(response_id, str) or not response_id:
                raise ValueError("dense inventory record lacks response_id")
            grouped[response_id].append(record)
    if not grouped:
        raise ValueError("inventory contains no completed dense discovery records")
    for response_id, records in grouped.items():
        records.sort(
            key=lambda record: (
                int(record["response_position"]),
                str(record["source_artifact_id"]),
            )
        )
        positions = [int(record["response_position"]) for record in records]
        if positions != list(range(len(positions))):
            raise ValueError(
                f"dense response {response_id} positions are not contiguous from zero"
            )
        if len({str(record["base_question_id"]) for record in records}) != 1:
            raise ValueError(f"dense response {response_id} spans multiple families")
        if len({str(record["trace_unit_id"]) for record in records}) != len(records):
            raise ValueError(f"dense response {response_id} has duplicate trace IDs")
    return dict(grouped)


def _task_target_identity(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_artifact_id": record["source_artifact_id"],
        "trace_unit_id": record["trace_unit_id"],
        "artifact_manifest_sha256": record["artifact_manifest_sha256"],
        "artifact_payload_sha256": record["artifact_payload_sha256"],
        "response_position": record["response_position"],
        "target_token_id": record["target_token_id"],
    }


def build_downstream_plan(
    *,
    inventory_path: Path,
    output_root: Path,
    lane: BuildLane,
    repo_root: Path,
    allow_dirty_development: bool = False,
    require_frozen_dense: bool = False,
    development_targets_per_response: int | None = None,
    code_revision: Mapping[str, Any] | None = None,
    runtime_environment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if lane not in BUILD_LANES:
        raise ValueError(f"unsupported downstream lane: {lane!r}")
    inventory_path = inventory_path.resolve()
    output_root = output_root.resolve()
    repo_root = repo_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"downstream output root already exists: {output_root}")
    inventory = load_json_object(inventory_path)
    _validate_inventory(inventory)
    grouped = _dense_records(inventory)
    dense_count = sum(len(records) for records in grouped.values())
    if require_frozen_dense and (len(grouped), dense_count) != (11, 2083):
        raise ValueError(
            "frozen dense baseline must contain 11 responses and 2083 targets; "
            f"got {len(grouped)} responses and {dense_count} targets"
        )

    revision = dict(code_revision or collect_downstream_code_revision(repo_root))
    development = bool(revision.get("git_dirty")) or (
        development_targets_per_response is not None
    )
    if development and not allow_dirty_development:
        raise ValueError(
            "downstream production plan requires a clean scoped source tree"
        )
    if development_targets_per_response is not None:
        if not allow_dirty_development:
            raise ValueError("target-limited plans require explicit development mode")
        if development_targets_per_response < 1:
            raise ValueError("development_targets_per_response must be positive")
        grouped = {
            response_id: records[:development_targets_per_response]
            for response_id, records in grouped.items()
        }
        dense_count = sum(len(records) for records in grouped.values())
    environment = dict(runtime_environment or collect_downstream_environment())
    lock_path = repo_root / "uv.lock"
    if not lock_path.is_file():
        raise ValueError(f"uv.lock is missing: {lock_path}")

    tasks: list[dict[str, Any]] = []
    for task_index, response_id in enumerate(sorted(grouped)):
        records = grouped[response_id]
        target_identity = [_task_target_identity(record) for record in records]
        tasks.append(
            {
                "task_index": task_index,
                "response_id": response_id,
                "base_question_id": records[0]["base_question_id"],
                "target_count": len(records),
                "first_response_position": records[0]["response_position"],
                "last_response_position": records[-1]["response_position"],
                "target_identity_sha256": canonical_sha256(target_identity),
            }
        )
    plan: dict[str, Any] = {
        "schema_version": BUILD_PLAN_SCHEMA,
        "run_family": f"bonafide-{lane.replace('_', '-')}-width1-v1",
        "lane": lane,
        "development": development,
        "development_targets_per_response": development_targets_per_response,
        "source_inventory": {
            "path": str(inventory_path),
            "file_sha256": file_sha256(inventory_path),
            "inventory_sha256": inventory["inventory_sha256"],
            "validation_level": inventory.get("validation_level"),
        },
        "output_root": str(output_root),
        "repo_root": str(repo_root),
        "code_revision": revision,
        "runtime_environment": environment,
        "uv_lock_sha256": file_sha256(lock_path),
        "execution_contract": {
            "partition": CorpusRole.DENSE_DISCOVERY.value,
            "response_array": True,
            "array_task_count": len(tasks),
            "one_writer_per_response": True,
            "atomic_response_shards": True,
            "resume_requires_checksum_valid_shard": True,
            "source_artifacts_read_only": True,
        },
        "dense_summary": {
            "response_count": len(grouped),
            "target_count": dense_count,
        },
        "tasks": tasks,
    }
    plan["plan_sha256"] = canonical_sha256(plan)
    return plan


def validate_downstream_plan(
    plan: Mapping[str, Any],
    *,
    verify_inputs: bool,
    verify_code: bool,
) -> dict[str, Any]:
    if plan.get("schema_version") != BUILD_PLAN_SCHEMA:
        raise ValueError("unsupported downstream build-plan schema")
    core = dict(plan)
    recorded_hash = core.pop("plan_sha256", None)
    if recorded_hash != canonical_sha256(core):
        raise ValueError("downstream build-plan canonical hash mismatch")
    if plan.get("lane") not in BUILD_LANES:
        raise ValueError("downstream build-plan lane is invalid")
    tasks_value = plan.get("tasks")
    if not isinstance(tasks_value, list) or not tasks_value:
        raise ValueError("downstream build plan requires tasks")
    tasks = [cast(Mapping[str, Any], task) for task in tasks_value]
    if [task.get("task_index") for task in tasks] != list(range(len(tasks))):
        raise ValueError("downstream task indexes must be contiguous")
    if len({task.get("response_id") for task in tasks}) != len(tasks):
        raise ValueError("downstream tasks contain duplicate response IDs")
    contract = plan.get("execution_contract")
    if not isinstance(contract, Mapping) or any(
        contract.get(field) is not True
        for field in (
            "response_array",
            "one_writer_per_response",
            "atomic_response_shards",
            "resume_requires_checksum_valid_shard",
            "source_artifacts_read_only",
        )
    ):
        raise ValueError("downstream execution contract is incomplete")

    if verify_inputs:
        source = plan.get("source_inventory")
        if not isinstance(source, Mapping):
            raise ValueError("downstream plan source_inventory is invalid")
        inventory_path = Path(str(source.get("path"))).resolve()
        if file_sha256(inventory_path) != source.get("file_sha256"):
            raise ValueError("downstream source inventory file hash drift")
        inventory = load_json_object(inventory_path)
        _validate_inventory(inventory)
        if inventory.get("inventory_sha256") != source.get("inventory_sha256"):
            raise ValueError("downstream source inventory identity drift")
        grouped = _dense_records(inventory)
        for task in tasks:
            response_id = str(task["response_id"])
            records = grouped.get(response_id)
            if records is None:
                raise ValueError(
                    f"downstream task response is absent from inventory: {response_id}"
                )
            limit = plan.get("development_targets_per_response")
            if limit is not None:
                records = records[: int(limit)]
            identities = [_task_target_identity(record) for record in records]
            if canonical_sha256(identities) != task.get("target_identity_sha256"):
                raise ValueError(f"downstream target identity drift for {response_id}")

    if verify_code:
        repo_root = Path(str(plan.get("repo_root"))).resolve()
        current = collect_downstream_code_revision(repo_root)
        if current != plan.get("code_revision"):
            raise ValueError("downstream executable source revision drift")
        if file_sha256(repo_root / "uv.lock") != plan.get("uv_lock_sha256"):
            raise ValueError("downstream uv.lock hash drift")
        if collect_downstream_environment() != plan.get("runtime_environment"):
            raise ValueError("downstream runtime environment drift")
    return dict(plan)


def write_downstream_plan(path: Path, plan: Mapping[str, Any]) -> None:
    write_hashed_json(path, plan, hash_field="plan_sha256")


def task_records(
    plan: Mapping[str, Any],
    *,
    task_index: int,
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    validated = validate_downstream_plan(
        plan,
        verify_inputs=True,
        verify_code=False,
    )
    tasks = cast(Sequence[Mapping[str, Any]], validated["tasks"])
    if not 0 <= task_index < len(tasks):
        raise ValueError(f"downstream task index is out of range: {task_index}")
    task = tasks[task_index]
    inventory_path = Path(
        str(cast(Mapping[str, Any], validated["source_inventory"])["path"])
    )
    inventory = load_json_object(inventory_path)
    grouped = _dense_records(inventory)
    records = grouped[str(task["response_id"])]
    limit = plan.get("development_targets_per_response")
    if limit is not None:
        records = records[: int(limit)]
    return task, records


def dense_fit_weights(plan: Mapping[str, Any]) -> dict[str, float]:
    source = plan.get("source_inventory")
    if not isinstance(source, Mapping):
        raise ValueError("downstream plan source_inventory is invalid")
    inventory = load_json_object(Path(str(source["path"])))
    grouped = _dense_records(inventory)
    limit = plan.get("development_targets_per_response")
    if limit is not None:
        grouped = {
            response_id: records[: int(limit)]
            for response_id, records in grouped.items()
        }
    targets = [
        AnalysisTarget(
            source_artifact_id=str(record["source_artifact_id"]),
            base_question_id=str(record["base_question_id"]),
            response_id=str(record["response_id"]),
            response_position=int(record["response_position"]),
            corpus_role=CorpusRole(str(record["corpus_role"])),
            cluster_fit_eligible=bool(record["cluster_fit_eligible"]),
        )
        for records in grouped.values()
        for record in records
    ]
    return hierarchical_fit_weights(targets)
