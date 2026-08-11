"""Build an immutable, provenance-preserving T5 pass-one salvage plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from circuits.analysis.bonafide.canonical import (
    canonical_sha256,
    file_sha256,
    write_hashed_json,
)
from circuits.tracing.artifact import validate_topk_compact_trace_integrity

from scripts.bonafide.build_t5_corpus_bundle import BUNDLE_SCHEMA
from scripts.bonafide.runner import (
    _sha256,
    collect_code_revision,
    load_json,
    normalized_trace_warmup,
    validate_run_config,
)
from scripts.bonafide.topk_manifest import validate_topk_manifest
from scripts.bonafide.topk_runner import topk_runtime_artifact_identity

PLAN_SCHEMA = "bonafide-t5-pass1-salvage-plan/v1"
FAILURE_STATUSES = frozenset({"error", "oom"})


def _absolute_file(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not path.is_absolute() or not resolved.is_file():
        raise ValueError(f"{label} must be an existing absolute file: {path}")
    return resolved


def _absolute_dir(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not path.is_absolute() or not resolved.is_dir():
        raise ValueError(f"{label} must be an existing absolute directory: {path}")
    return resolved


def _git_output(source_tree: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=source_tree,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def validate_frozen_source_tree(
    source_tree: Path, expected_commit: str
) -> dict[str, Any]:
    """Verify a source tree without changing it and return its trace fingerprint."""

    source_tree = _absolute_dir(source_tree, "source tree")
    if len(expected_commit) != 40 or any(
        character not in "0123456789abcdef" for character in expected_commit
    ):
        raise ValueError("expected Git commit must be a lowercase 40-hex digest")
    if _git_output(source_tree, "rev-parse", "HEAD") != expected_commit:
        raise ValueError(f"source tree is not at expected commit: {source_tree}")
    if _git_output(source_tree, "status", "--porcelain=v1", "--untracked-files=no"):
        raise ValueError(f"source tree has tracked changes: {source_tree}")
    revision = collect_code_revision(source_tree)
    if revision.get("git_commit") != expected_commit or revision.get("git_dirty"):
        raise ValueError("trace-source code fingerprint is not clean and frozen")
    return revision


def _validate_hash(path: Path, expected: object, label: str) -> None:
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"{label} lacks a valid SHA-256 digest")
    if file_sha256(path) != expected:
        raise ValueError(f"{label} hash drift: {path}")


def _bundle_contracts(
    bundle_path: Path,
    expected_bundle_sha256: str,
    *,
    selected_task_indices: frozenset[int] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    _validate_hash(bundle_path, expected_bundle_sha256, "T5 pass-one bundle")
    bundle = load_json(bundle_path)
    if bundle.get("schema_version") != BUNDLE_SCHEMA:
        raise ValueError("unsupported T5 pass-one bundle schema")

    for name in ("selection", "source_manifest", "rank_screen"):
        path = _absolute_file(Path(str(bundle.get(f"{name}_path", ""))), name)
        _validate_hash(path, bundle.get(f"{name}_sha256"), name)

    manifest_records = bundle.get("manifests")
    if not isinstance(manifest_records, list) or len(manifest_records) != 6:
        raise ValueError("T5 pass-one bundle must bind six candidate manifests")
    manifests: dict[str, dict[str, Any]] = {}
    for record in manifest_records:
        if not isinstance(record, Mapping):
            raise TypeError("T5 pass-one manifest records must be objects")
        path = _absolute_file(Path(str(record.get("path", ""))), "candidate manifest")
        path_key = str(path)
        if path_key in manifests:
            raise ValueError(f"duplicate candidate manifest path: {path}")
        _validate_hash(path, record.get("sha256"), "candidate manifest")
        manifest = load_json(path)
        validate_topk_manifest(manifest)
        if record.get("canonical_sha256") != _sha256(manifest):
            raise ValueError(f"candidate manifest canonical hash drift: {path}")
        manifests[path_key] = manifest

    tasks_value = bundle.get("tasks")
    if not isinstance(tasks_value, list) or not tasks_value:
        raise ValueError("T5 pass-one bundle contains no task table")
    all_indices = set(range(len(tasks_value)))
    if selected_task_indices is not None and not selected_task_indices <= all_indices:
        invalid = sorted(selected_task_indices - all_indices)
        raise ValueError(
            f"selected original task indices are outside bundle: {invalid}"
        )
    selected: list[dict[str, Any]] = []
    for index, task_value in enumerate(tasks_value):
        if not isinstance(task_value, Mapping) or task_value.get("task_index") != index:
            raise ValueError(f"invalid T5 task-table row at index {index}")
        if selected_task_indices is not None and index not in selected_task_indices:
            continue
        task = dict(task_value)
        manifest_path = str(Path(str(task.get("manifest_path", ""))).resolve())
        manifest = manifests.get(manifest_path)
        if manifest is None:
            raise ValueError(f"task {index} points outside the bound manifests")
        if task.get("manifest_sha256") != file_sha256(Path(manifest_path)):
            raise ValueError(f"task {index} manifest hash drift")
        waves = [
            wave
            for wave in manifest["waves"]
            if wave.get("wave_id") == task.get("wave_id")
        ]
        if len(waves) != 1 or task.get("work_item_count") != len(waves[0]["items"]):
            raise ValueError(f"task {index} wave contract drift")
        selected.append(task)
    if not selected:
        raise ValueError("salvage scope selects no original pass-one tasks")
    return bundle, selected, manifests


def _artifact_manifest_candidates(
    artifact_root: Path,
    manifests: Iterable[Mapping[str, Any]],
    selected_wave_ids: frozenset[str],
) -> Iterable[Path]:
    family_ids = sorted(
        {str(manifest["trace_family"]["trace_family_id"]) for manifest in manifests}
    )
    for family_id in family_ids:
        family_root = artifact_root / family_id
        if family_root.is_dir():
            for wave_id in sorted(selected_wave_ids):
                wave_root = family_root / wave_id
                if wave_root.is_dir():
                    yield from (
                        path
                        for path in sorted(wave_root.glob("*/manifest.json"))
                        if not path.parent.name.startswith(".")
                    )


def discover_identity_execution_contract(
    artifact_root: Path,
    manifests: Iterable[Mapping[str, Any]],
    expected_code_revision: Mapping[str, Any],
    selected_wave_ids: frozenset[str],
) -> dict[str, Any]:
    """Recover the exact identity-defining contract from completed artifacts."""

    manifest_by_hash = {_sha256(manifest): manifest for manifest in manifests}
    contracts: dict[str, dict[str, Any]] = {}
    reference_artifacts: list[dict[str, Any]] = []
    metadata_artifact_count = 0
    for path in _artifact_manifest_candidates(
        artifact_root, manifests, selected_wave_ids
    ):
        wave_id = path.parent.parent.name
        value = validate_topk_compact_trace_integrity(path.parent)
        reference_artifacts.append(
            {
                "path": str(path.parent.resolve()),
                "manifest_sha256": file_sha256(path),
                "data_sha256": value["data_sha256"],
                "wave_id": wave_id,
            }
        )
        identity = value.get("artifact_identity")
        if not isinstance(identity, Mapping):
            raise TypeError(f"completed artifact lacks identity: {path}")
        manifest = manifest_by_hash.get(str(identity.get("topk_manifest_sha256")))
        if manifest is None:
            raise ValueError(
                f"artifact points outside bound candidate manifests: {path}"
            )
        claimed = identity.get("sha256")
        identity_core = dict(identity)
        identity_core.pop("sha256", None)
        if claimed != _sha256(identity_core):
            raise ValueError(f"existing artifact identity self-hash drift: {path}")
        expected_artifact_id = f"topk-trace-{str(claimed)[:24]}"
        if (
            value.get("artifact_id") != expected_artifact_id
            or path.parent.name != expected_artifact_id
            or path.parent.parent.name != identity.get("wave_id")
            or path.parent.parent.parent.name
            != manifest["trace_family"]["trace_family_id"]
            or identity.get("trace_family") != manifest["trace_family"]
            or identity.get("source_width1_manifest_sha256")
            != manifest["source"]["width1_manifest_sha256"]
        ):
            raise ValueError(f"completed artifact/manifest provenance drift: {path}")
        if identity.get("code_revision") != expected_code_revision:
            raise ValueError(
                f"existing artifact belongs to another code cohort: {path}"
            )
        wave_matches = [
            wave for wave in manifest["waves"] if wave["wave_id"] == identity["wave_id"]
        ]
        if len(wave_matches) != 1:
            raise ValueError(f"completed artifact wave is outside its manifest: {path}")
        source_items = {item["artifact_id"]: item for item in wave_matches[0]["items"]}
        source_item = source_items.get(identity.get("source_width1_artifact_id"))
        if source_item is None or identity.get(
            "source_width1_work_item_sha256"
        ) != _sha256(source_item):
            raise ValueError(f"completed artifact source-item drift: {path}")
        contract = {
            "model": identity.get("model"),
            "adag_config": identity.get("adag_config"),
            "trace_warmup": identity.get("trace_warmup"),
            "batch_size": identity.get("batch_size"),
            "code_revision": identity.get("code_revision"),
            "runtime_environment": identity.get("runtime_environment"),
        }
        if (
            not isinstance(contract["model"], Mapping)
            or not isinstance(contract["adag_config"], Mapping)
            or not isinstance(contract["trace_warmup"], Mapping)
            or contract["batch_size"] != 1
            or not isinstance(contract["runtime_environment"], Mapping)
        ):
            raise TypeError(f"completed artifact execution contract is invalid: {path}")
        contract = {
            **contract,
            "model": dict(contract["model"]),
            "adag_config": dict(contract["adag_config"]),
            "trace_warmup": dict(contract["trace_warmup"]),
            "code_revision": dict(contract["code_revision"]),
            "runtime_environment": dict(contract["runtime_environment"]),
        }
        contracts[canonical_sha256(contract)] = contract
        metadata_artifact_count += 1
    if metadata_artifact_count == 0 or not reference_artifacts:
        raise ValueError(
            "selected salvage waves contain no integrity-valid completed artifact "
            "from which to recover the original execution contract"
        )
    if len(contracts) != 1:
        raise ValueError(
            "selected pass-one artifacts contain multiple identity execution "
            "contracts; salvage must be planned separately for each cohort"
        )
    return {
        "contract": next(iter(contracts.values())),
        "metadata_artifact_count": metadata_artifact_count,
        "full_integrity_reference_artifacts": reference_artifacts,
    }


def _summary_failure_index(
    original_summary_root: Path,
    salvage_summary_root: Path,
    selected_wave_ids: frozenset[str],
) -> tuple[dict[tuple[str, str], list[dict[str, Any]]], list[dict[str, Any]]]:
    failures: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    file_records: list[dict[str, Any]] = []
    paths: list[Path] = []
    if original_summary_root.exists():
        if not original_summary_root.is_dir():
            raise ValueError(
                f"summary root is not a directory: {original_summary_root}"
            )
        # Original pass-one writers are partitioned by wave. Reading only the
        # selected, already-stopped waves avoids racing unrelated live tasks.
        for wave_id in sorted(selected_wave_ids):
            wave_root = original_summary_root / wave_id
            if wave_root.is_dir():
                paths.extend(wave_root.rglob("*.jsonl"))
    if salvage_summary_root.exists():
        if not salvage_summary_root.is_dir():
            raise ValueError(f"summary root is not a directory: {salvage_summary_root}")
        # A subsequent salvage plan is prepared only after prior salvage jobs
        # are quiescent; these summaries are not written by the main array.
        paths.extend(salvage_summary_root.rglob("*.jsonl"))
    for path in sorted(set(paths)):
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        file_records.append({"path": str(path.resolve()), "sha256": digest})
        for line_number, raw_line in enumerate(payload.splitlines(), start=1):
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid execution summary JSON: {path}:{line_number}"
                ) from error
            if not isinstance(row, dict):
                raise TypeError(f"non-object execution summary: {path}:{line_number}")
            if row.get("status") not in FAILURE_STATUSES:
                continue
            wave_id = row.get("wave_id")
            source_id = row.get("source_width1_artifact_id")
            if not isinstance(wave_id, str) or not isinstance(source_id, str):
                raise TypeError(
                    f"failure record lacks pair identity: {path}:{line_number}"
                )
            if wave_id not in selected_wave_ids:
                continue
            failures[(wave_id, source_id)].append(
                {
                    "status": row["status"],
                    "artifact_id": row.get("artifact_id"),
                    "error_type": row.get("error_type"),
                    "error": row.get("error"),
                    "summary_path": str(path.resolve()),
                    "summary_sha256": digest,
                    "line_number": line_number,
                }
            )
    return failures, file_records


def _task_items(
    selected_tasks: Sequence[Mapping[str, Any]],
    manifests: Mapping[str, Mapping[str, Any]],
) -> Iterable[tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]]:
    seen: set[tuple[str, str]] = set()
    for task in selected_tasks:
        manifest = manifests[str(Path(str(task["manifest_path"])).resolve())]
        wave = next(
            value for value in manifest["waves"] if value["wave_id"] == task["wave_id"]
        )
        for item in wave["items"]:
            key = (str(wave["wave_id"]), str(item["artifact_id"]))
            if key in seen:
                raise ValueError(f"duplicate expected pass-one pair: {key}")
            seen.add(key)
            yield task, manifest, item


def _chunked(values: Sequence[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [list(values[index : index + size]) for index in range(0, len(values), size)]


def build_salvage_plan(
    *,
    bundle_path: Path,
    bundle_sha256: str,
    artifact_root: Path,
    frozen_source_tree: Path,
    frozen_git_commit: str,
    orchestration_source_tree: Path,
    orchestration_git_commit: str,
    config_path: Path,
    python_bin: Path,
    selected_task_indices: Sequence[int] | None = None,
    allow_quiescent_missing_scan: bool = False,
    max_items_per_task: int = 1,
    max_array_tasks: int = 1000,
) -> dict[str, Any]:
    """Inspect pass-one evidence and return only its missing exact identities."""

    if max_items_per_task < 1 or max_array_tasks < 1:
        raise ValueError("salvage task limits must be positive")
    if selected_task_indices is None and not allow_quiescent_missing_scan:
        raise ValueError(
            "an all-task missing scan requires --allow-quiescent-missing-scan "
            "after the pass-one array is globally quiescent"
        )
    bundle_path = _absolute_file(bundle_path, "T5 pass-one bundle")
    artifact_root = _absolute_dir(artifact_root, "artifact root")
    frozen_source_tree = _absolute_dir(frozen_source_tree, "frozen source tree")
    orchestration_source_tree = _absolute_dir(
        orchestration_source_tree, "orchestration source tree"
    )
    config_path = _absolute_file(config_path, "trace config")
    if not python_bin.is_absolute() or not python_bin.is_file():
        raise ValueError(
            f"frozen Python interpreter must be an existing absolute file: {python_bin}"
        )
    python_bin = python_bin.absolute()
    if not os.access(python_bin, os.X_OK):
        raise ValueError(f"frozen Python interpreter is not executable: {python_bin}")
    try:
        config_path.relative_to(frozen_source_tree)
    except ValueError as error:
        raise ValueError(
            "trace config must live inside the frozen source tree"
        ) from error

    frozen_revision = validate_frozen_source_tree(frozen_source_tree, frozen_git_commit)
    orchestration_revision = validate_frozen_source_tree(
        orchestration_source_tree, orchestration_git_commit
    )
    config = load_json(config_path)
    validate_run_config(config)
    selected_set = (
        frozenset(selected_task_indices) if selected_task_indices is not None else None
    )
    bundle, selected_tasks, manifests = _bundle_contracts(
        bundle_path,
        bundle_sha256,
        selected_task_indices=selected_set,
    )
    selected_wave_ids = frozenset(str(task["wave_id"]) for task in selected_tasks)
    failures, summary_files = _summary_failure_index(
        artifact_root / "execution-summaries",
        artifact_root / "salvage-execution-summaries",
        selected_wave_ids,
    )
    terminal_failure_wave_ids = frozenset(wave_id for wave_id, _ in failures)
    if not allow_quiescent_missing_scan:
        missing_terminal_evidence = sorted(
            selected_wave_ids - terminal_failure_wave_ids
        )
        if missing_terminal_evidence:
            raise ValueError(
                "selected waves lack terminal error/oom evidence; refuse an active-array "
                f"missing scan: {missing_terminal_evidence}"
            )

    discovered = discover_identity_execution_contract(
        artifact_root,
        manifests.values(),
        frozen_revision,
        selected_wave_ids,
    )
    identity_execution_contract = discovered["contract"]
    supplied_identity_config = {
        "model": dict(config["model"]),
        "adag_config": dict(config["adag_config"]),
        "trace_warmup": normalized_trace_warmup(config),
        "batch_size": 1,
    }
    for field, supplied_value in supplied_identity_config.items():
        if identity_execution_contract.get(field) != supplied_value:
            raise ValueError(
                "supplied trace config does not match the original artifact "
                f"identity execution contract: {field}"
            )
    if identity_execution_contract["code_revision"] != frozen_revision:
        raise ValueError(
            "discovered artifact code revision disagrees with frozen source"
        )
    runtime_environment = identity_execution_contract["runtime_environment"]

    complete_set: list[dict[str, str]] = []
    missing: list[dict[str, Any]] = []
    for task, manifest, item in _task_items(selected_tasks, manifests):
        source_manifest_sha256 = manifest["source"]["width1_manifest_sha256"]
        manifest_sha256 = _sha256(manifest)
        artifact_id, identity = topk_runtime_artifact_identity(
            item,
            config=config,
            trace_family=manifest["trace_family"],
            code_revision=frozen_revision,
            runtime_environment=runtime_environment,
            source_manifest_sha256=source_manifest_sha256,
            topk_manifest_sha256=manifest_sha256,
            wave_id=str(task["wave_id"]),
        )
        path = (
            artifact_root
            / str(manifest["trace_family"]["trace_family_id"])
            / str(task["wave_id"])
            / artifact_id
        )
        if path.exists():
            completed_manifest = validate_topk_compact_trace_integrity(path)
            if (
                completed_manifest.get("artifact_id") != artifact_id
                or completed_manifest.get("artifact_identity") != identity
            ):
                raise ValueError(f"completed artifact identity drift: {path}")
            complete_set.append(
                {
                    "artifact_id": artifact_id,
                    "artifact_identity_sha256": identity["sha256"],
                }
            )
            continue
        key = (str(task["wave_id"]), str(item["artifact_id"]))
        prior_failures = failures.get(key, [])
        for failure in prior_failures:
            if failure.get("artifact_id") not in {None, artifact_id}:
                raise ValueError(f"failure record artifact identity drift for {key}")
        missing.append(
            {
                "salvage_item_index": len(missing),
                "original_task_index": int(task["task_index"]),
                "candidate_index": int(task["candidate_index"]),
                "manifest_path": str(Path(str(task["manifest_path"])).resolve()),
                "manifest_sha256": str(task["manifest_sha256"]),
                "manifest_canonical_sha256": manifest_sha256,
                "wave_id": str(task["wave_id"]),
                "trace_family_id": str(manifest["trace_family"]["trace_family_id"]),
                "source_width1_artifact_id": str(item["artifact_id"]),
                "expected_artifact_id": artifact_id,
                "expected_artifact_identity_sha256": identity["sha256"],
                "expected_artifact_identity": identity,
                "expected_artifact_path": str(path.resolve()),
                "prior_failures": prior_failures,
            }
        )

    missing.sort(
        key=lambda row: (
            int(row["original_task_index"]),
            int(row["salvage_item_index"]),
        )
    )
    for index, item in enumerate(missing):
        item["salvage_item_index"] = index
    chunks = _chunked(missing, max_items_per_task)
    if len(chunks) > max_array_tasks:
        raise ValueError(
            "salvage plan exceeds array-task limit; increase --max-items-per-task: "
            f"tasks={len(chunks)}, limit={max_array_tasks}"
        )
    tasks = [
        {
            "task_index": index,
            "item_count": len(chunk),
            "salvage_item_indices": [item["salvage_item_index"] for item in chunk],
        }
        for index, chunk in enumerate(chunks)
    ]
    scoped_expected_count = len(complete_set) + len(missing)
    plan: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA,
        "source": {
            "bundle_path": str(bundle_path),
            "bundle_sha256": bundle_sha256,
            "cohort_id": bundle["cohort_id"],
            "artifact_root": str(artifact_root),
            "selected_original_task_indices": [
                int(task["task_index"]) for task in selected_tasks
            ],
            "frozen_source_tree": str(frozen_source_tree),
            "frozen_code_revision": frozen_revision,
            "orchestration_source_tree": str(orchestration_source_tree),
            "orchestration_code_revision": orchestration_revision,
            "config_path": str(config_path),
            "config_sha256": file_sha256(config_path),
            "config_canonical_sha256": _sha256(config),
            "python_bin": str(python_bin),
            "identity_execution_contract": identity_execution_contract,
            "runtime_environment": runtime_environment,
            "runtime_environment_sha256": canonical_sha256(runtime_environment),
        },
        "scan": {
            "mode": (
                "quiescent_missing_scan"
                if allow_quiescent_missing_scan
                else "terminal_failure_repair"
            ),
            "allow_quiescent_missing_scan": allow_quiescent_missing_scan,
            "terminal_failure_wave_ids": sorted(terminal_failure_wave_ids),
            "summary_files": summary_files,
            "completed_artifact_set_sha256": canonical_sha256(
                sorted(complete_set, key=lambda row: row["artifact_id"])
            ),
            "completed_artifact_classification": (
                "manifest-and-metrics-json, identity-self-hash, payload-sha256"
            ),
            "identity_contract_metadata_artifact_count": discovered[
                "metadata_artifact_count"
            ],
            "full_integrity_reference_artifacts": discovered[
                "full_integrity_reference_artifacts"
            ],
        },
        "counts": {
            "scoped_original_tasks": len(selected_tasks),
            "scoped_expected_artifacts": scoped_expected_count,
            "completed_artifacts": len(complete_set),
            "missing_artifacts": len(missing),
            "missing_with_prior_failures": sum(
                bool(item["prior_failures"]) for item in missing
            ),
            "known_failure_events": sum(
                len(item["prior_failures"]) for item in missing
            ),
            "salvage_tasks": len(tasks),
        },
        "execution": {
            "runner_module": "scripts.bonafide.topk_runner",
            "one_frozen_runner_subprocess_per_artifact": True,
            "continue_after_artifact_failure": True,
            "max_items_per_task": max_items_per_task,
            "max_array_tasks": max_array_tasks,
            "submission_authorized": False,
        },
        "items": missing,
        "tasks": tasks,
    }
    plan["manifest_sha256"] = canonical_sha256(plan)
    return plan


def write_salvage_plan(output_dir: Path, plan: Mapping[str, Any]) -> Path:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"salvage output directory exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    try:
        destination = staging / "t5-pass1-salvage-plan.json"
        write_hashed_json(destination, plan, hash_field="manifest_sha256")
        os.replace(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output_dir / "t5-pass1-salvage-plan.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--bundle-sha256", required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--frozen-source-tree", type=Path, required=True)
    parser.add_argument("--frozen-git-commit", required=True)
    parser.add_argument("--orchestration-source-tree", type=Path, required=True)
    parser.add_argument("--orchestration-git-commit", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--python-bin", type=Path, required=True)
    parser.add_argument("--original-task-index", type=int, action="append")
    parser.add_argument("--allow-quiescent-missing-scan", action="store_true")
    parser.add_argument("--max-items-per-task", type=int, default=1)
    parser.add_argument("--max-array-tasks", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plan = build_salvage_plan(
        bundle_path=args.bundle,
        bundle_sha256=args.bundle_sha256,
        artifact_root=args.artifact_root,
        frozen_source_tree=args.frozen_source_tree,
        frozen_git_commit=args.frozen_git_commit,
        orchestration_source_tree=args.orchestration_source_tree,
        orchestration_git_commit=args.orchestration_git_commit,
        config_path=args.config,
        python_bin=args.python_bin,
        selected_task_indices=args.original_task_index,
        allow_quiescent_missing_scan=args.allow_quiescent_missing_scan,
        max_items_per_task=args.max_items_per_task,
        max_array_tasks=args.max_array_tasks,
    )
    destination = write_salvage_plan(args.output_dir, plan)
    print(
        json.dumps(
            {
                "path": str(destination),
                "file_sha256": file_sha256(destination),
                "manifest_sha256": plan["manifest_sha256"],
                "counts": plan["counts"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
