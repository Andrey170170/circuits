"""Execute one task from a provenance-preserving T5 pass-one salvage plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from circuits.analysis.bonafide.canonical import canonical_sha256, file_sha256
from circuits.tracing.artifact import validate_topk_compact_trace_integrity
from scripts.bonafide.build_t5_pass1_salvage import (
    PLAN_SCHEMA,
    validate_frozen_source_tree,
)
from scripts.bonafide.runner import (
    _sha256,
    load_json,
    normalized_trace_warmup,
    validate_run_config,
)

_ATTEMPT_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_OUTPUT_TAIL_LIMIT = 16_384


def load_salvage_plan(path: Path, expected_file_sha256: str) -> dict[str, Any]:
    if not path.is_absolute():
        raise ValueError("salvage plan must be an existing absolute file")
    path = path.resolve()
    if not path.is_file():
        raise ValueError("salvage plan must be an existing absolute file")
    if file_sha256(path) != expected_file_sha256:
        raise ValueError("salvage plan file hash drift")
    plan = load_json(path)
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("unsupported T5 pass-one salvage-plan schema")
    core = dict(plan)
    recorded = core.pop("manifest_sha256", None)
    if recorded != canonical_sha256(core):
        raise ValueError("salvage plan self-hash drift")
    return plan


def _validate_file(path_value: object, digest: object, label: str) -> Path:
    raw_path = Path(str(path_value))
    if not raw_path.is_absolute():
        raise ValueError(f"{label} is not an existing absolute file")
    path = raw_path.resolve()
    if not path.is_file():
        raise ValueError(f"{label} is not an existing absolute file")
    if not isinstance(digest, str) or file_sha256(path) != digest:
        raise ValueError(f"{label} hash drift")
    return path


def _child_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    return environment


def validate_execution_contract(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Fail before model load if any source, environment, or config has drifted."""

    source = plan.get("source")
    if not isinstance(source, Mapping):
        raise TypeError("salvage plan lacks source contract")
    repo_root = Path(__file__).resolve().parents[2]
    orchestration_value = Path(str(source.get("orchestration_source_tree", "")))
    if not orchestration_value.is_absolute():
        raise ValueError("orchestration source tree must be absolute")
    orchestration_tree = orchestration_value.resolve()
    if repo_root != orchestration_tree:
        raise ValueError("salvage executor is not running from its bound source tree")
    orchestration_revision = source.get("orchestration_code_revision")
    if not isinstance(orchestration_revision, Mapping):
        raise TypeError("salvage plan lacks orchestration code revision")
    actual_orchestration = validate_frozen_source_tree(
        orchestration_tree, str(orchestration_revision.get("git_commit", ""))
    )
    if actual_orchestration != orchestration_revision:
        raise ValueError("salvage orchestration source fingerprint drift")

    frozen_value = Path(str(source.get("frozen_source_tree", "")))
    if not frozen_value.is_absolute():
        raise ValueError("frozen trace source tree must be absolute")
    frozen_tree = frozen_value.resolve()
    frozen_revision = source.get("frozen_code_revision")
    if not isinstance(frozen_revision, Mapping):
        raise TypeError("salvage plan lacks frozen code revision")
    actual_frozen = validate_frozen_source_tree(
        frozen_tree, str(frozen_revision.get("git_commit", ""))
    )
    if actual_frozen != frozen_revision:
        raise ValueError("frozen trace source fingerprint drift")

    bundle_path = _validate_file(
        source.get("bundle_path"), source.get("bundle_sha256"), "pass-one bundle"
    )
    config_path = _validate_file(
        source.get("config_path"), source.get("config_sha256"), "trace config"
    )
    config = load_json(config_path)
    validate_run_config(config)
    if _sha256(config) != source.get("config_canonical_sha256"):
        raise ValueError("trace config canonical hash drift")
    identity_execution_contract = source.get("identity_execution_contract")
    if not isinstance(identity_execution_contract, Mapping):
        raise TypeError("salvage plan lacks original identity execution contract")
    supplied_identity_config = {
        "model": dict(config["model"]),
        "adag_config": dict(config["adag_config"]),
        "trace_warmup": normalized_trace_warmup(config),
        "batch_size": 1,
    }
    for field, supplied_value in supplied_identity_config.items():
        if identity_execution_contract.get(field) != supplied_value:
            raise ValueError(
                "trace config differs from original artifact identity contract: "
                f"{field}"
            )
    if identity_execution_contract.get("code_revision") != frozen_revision:
        raise ValueError("original identity contract code revision drift")
    python_bin = Path(str(source.get("python_bin", "")))
    if not python_bin.is_absolute():
        raise ValueError("bound frozen Python interpreter must be absolute")
    python_bin = python_bin.absolute()
    if not python_bin.is_file() or not os.access(python_bin, os.X_OK):
        raise ValueError("bound frozen Python interpreter is unavailable")

    probe_program = (
        "import json; "
        "from scripts.bonafide.runner import collect_code_revision, "
        "collect_runtime_environment; "
        "from pathlib import Path; "
        "print(json.dumps({'code_revision': collect_code_revision(Path.cwd()), "
        "'runtime_environment': collect_runtime_environment()}, "
        "sort_keys=True, allow_nan=False))"
    )
    probe = subprocess.run(
        [str(python_bin), "-c", probe_program],
        cwd=frozen_tree,
        env=_child_environment(),
        check=False,
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        raise RuntimeError(f"frozen runtime probe failed: {probe.stderr[-2000:]}")
    try:
        probe_value = json.loads(probe.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("frozen runtime probe returned invalid JSON") from error
    if probe_value.get("code_revision") != frozen_revision:
        raise ValueError("frozen child process reports another code revision")
    runtime_environment = source.get("runtime_environment")
    if identity_execution_contract.get("runtime_environment") != runtime_environment:
        raise ValueError("original identity contract runtime provenance drift")
    if probe_value.get(
        "runtime_environment"
    ) != runtime_environment or canonical_sha256(runtime_environment) != source.get(
        "runtime_environment_sha256"
    ):
        raise ValueError(
            "retry runtime differs from the original artifact cohort; refusing to "
            "create alternate artifact identities"
        )
    artifact_root_value = Path(str(source.get("artifact_root", "")))
    if not artifact_root_value.is_absolute():
        raise ValueError("bound artifact root must be absolute")
    return {
        "repo_root": repo_root,
        "frozen_tree": frozen_tree,
        "bundle_path": bundle_path,
        "config_path": config_path,
        "python_bin": python_bin,
        "artifact_root": artifact_root_value.resolve(),
    }


def _append_receipt(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = dict(value)
    record["event_sha256"] = canonical_sha256(record)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, allow_nan=False))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _tail_and_hash(value: str) -> tuple[str, str]:
    return value[-_OUTPUT_TAIL_LIMIT:], hashlib.sha256(value.encode()).hexdigest()


def _runner_record(stdout: str, expected_artifact_id: str) -> dict[str, Any] | None:
    matches: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("artifact_id") == expected_artifact_id:
            matches.append(value)
    if len(matches) != 1:
        return None
    return matches[0]


def _validate_expected_artifact(item: Mapping[str, Any]) -> None:
    path = Path(str(item["expected_artifact_path"]))
    manifest = validate_topk_compact_trace_integrity(path)
    identity = manifest.get("artifact_identity")
    if not isinstance(identity, Mapping):
        raise TypeError(f"salvaged artifact lacks identity: {path}")
    identity_core = dict(identity)
    claimed_identity_sha256 = identity_core.pop("sha256", None)
    if (
        manifest.get("artifact_id") != item["expected_artifact_id"]
        or claimed_identity_sha256 != _sha256(identity_core)
        or identity != item["expected_artifact_identity"]
        or claimed_identity_sha256 != item["expected_artifact_identity_sha256"]
    ):
        raise ValueError(f"salvaged artifact identity drift: {path}")


def _invoke_runner(
    command: list[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    signal_state: dict[str, Any],
) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=dict(environment),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    signal_state["process"] = process
    if signal_state["requested"]:
        _forward_usr1(process)
    try:
        stdout, stderr = process.communicate()
    finally:
        signal_state["process"] = None
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def _forward_usr1(process: Any) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        process.send_signal(signal.SIGUSR1)
    except ProcessLookupError:
        # The child may exit between poll() and send_signal().
        pass


def execute_salvage_task(
    *,
    plan: Mapping[str, Any],
    task_index: int,
    attempt_id: str,
) -> tuple[int, Path]:
    if not _ATTEMPT_RE.fullmatch(attempt_id):
        raise ValueError(
            "attempt ID may contain only letters, digits, dot, dash, underscore"
        )
    tasks = plan.get("tasks")
    items = plan.get("items")
    if not isinstance(tasks, list) or not isinstance(items, list):
        raise TypeError("salvage plan lacks task/item tables")
    if task_index < 0 or task_index >= len(tasks):
        raise ValueError(f"salvage task index is outside plan: {task_index}")
    task = tasks[task_index]
    if not isinstance(task, Mapping) or task.get("task_index") != task_index:
        raise ValueError(f"invalid salvage task-table row: {task_index}")
    indices = task.get("salvage_item_indices")
    if not isinstance(indices, list) or task.get("item_count") != len(indices):
        raise ValueError(f"invalid salvage item list for task {task_index}")
    selected: list[Mapping[str, Any]] = []
    for index in indices:
        if isinstance(index, bool) or not isinstance(index, int) or index >= len(items):
            raise ValueError(f"invalid salvage item index in task {task_index}")
        item = items[index]
        if not isinstance(item, Mapping) or item.get("salvage_item_index") != index:
            raise ValueError(f"salvage item-table drift at index {index}")
        selected.append(item)

    contract = validate_execution_contract(plan)
    artifact_root = contract["artifact_root"]
    if not artifact_root.is_dir():
        raise ValueError("bound artifact root is unavailable")
    plan_hash = str(plan["manifest_sha256"])
    receipt_root = artifact_root / "salvage-receipts" / plan_hash
    receipt_path = receipt_root / f"task-{task_index:04d}" / f"{attempt_id}.jsonl"
    if receipt_path.exists():
        raise FileExistsError(f"salvage attempt receipt already exists: {receipt_path}")
    runner_summary_root = artifact_root / "salvage-execution-summaries" / plan_hash
    environment = _child_environment()
    signal_state: dict[str, Any] = {"requested": False, "process": None}
    previous_handler = signal.getsignal(signal.SIGUSR1)

    def request_stop(_signum, _frame) -> None:
        signal_state["requested"] = True
        _forward_usr1(signal_state.get("process"))

    signal.signal(signal.SIGUSR1, request_stop)
    failures = 0
    stop_recorded = False

    def record_signal_stop(remaining_item_count: int) -> None:
        nonlocal failures, stop_recorded
        if stop_recorded:
            return
        _append_receipt(
            receipt_path,
            {
                "status": "task_stopped",
                "reason": "slurm_time_limit_signal",
                "plan_manifest_sha256": plan_hash,
                "task_index": task_index,
                "remaining_item_count": remaining_item_count,
            },
        )
        failures += 1
        stop_recorded = True

    try:
        for selected_index, item in enumerate(selected):
            if signal_state["requested"]:
                record_signal_stop(len(selected) - selected_index)
                break
            manifest_path = _validate_file(
                item["manifest_path"], item["manifest_sha256"], "candidate manifest"
            )
            runner_summary = (
                runner_summary_root
                / f"task-{task_index:04d}"
                / f"{attempt_id}-{item['expected_artifact_id']}.jsonl"
            )
            command = [
                str(contract["python_bin"]),
                "-m",
                str(plan["execution"]["runner_module"]),
                "--config",
                str(contract["config_path"]),
                "--manifest",
                str(manifest_path),
                "--wave",
                str(item["wave_id"]),
                "--artifact-root",
                str(artifact_root),
                "--summary-jsonl",
                str(runner_summary),
                "--only-artifact-id",
                str(item["source_width1_artifact_id"]),
                "--print-records",
            ]
            started_at = datetime.now(UTC).isoformat()
            started = time.perf_counter()
            if signal_state["requested"]:
                record_signal_stop(len(selected) - selected_index)
                break
            completed = _invoke_runner(
                command,
                cwd=contract["frozen_tree"],
                environment=environment,
                signal_state=signal_state,
            )
            stdout_tail, stdout_sha256 = _tail_and_hash(completed.stdout)
            stderr_tail, stderr_sha256 = _tail_and_hash(completed.stderr)
            runner_record = _runner_record(
                completed.stdout, str(item["expected_artifact_id"])
            )
            status = "failed"
            error: str | None = None
            if completed.returncode == 0 and runner_record is not None:
                if runner_record.get("status") not in {"complete", "skipped_complete"}:
                    error = "frozen runner returned an unexpected success status"
                else:
                    try:
                        _validate_expected_artifact(item)
                    except (OSError, TypeError, ValueError) as caught:
                        error = f"post-run artifact validation failed: {caught}"
                    else:
                        status = str(runner_record["status"])
            elif completed.returncode == 0:
                error = "frozen runner did not emit exactly one expected record"
            else:
                error = f"frozen runner exited with status {completed.returncode}"
            if status == "failed":
                failures += 1
            summary_sha256 = (
                file_sha256(runner_summary) if runner_summary.is_file() else None
            )
            _append_receipt(
                receipt_path,
                {
                    "status": status,
                    "error": error,
                    "plan_manifest_sha256": plan_hash,
                    "task_index": task_index,
                    "salvage_item_index": item["salvage_item_index"],
                    "original_task_index": item["original_task_index"],
                    "source_width1_artifact_id": item["source_width1_artifact_id"],
                    "expected_artifact_id": item["expected_artifact_id"],
                    "expected_artifact_identity_sha256": item[
                        "expected_artifact_identity_sha256"
                    ],
                    "started_at": started_at,
                    "wall_seconds": time.perf_counter() - started,
                    "runner_returncode": completed.returncode,
                    "runner_record_status": (
                        runner_record.get("status") if runner_record else None
                    ),
                    "runner_summary_path": str(runner_summary),
                    "runner_summary_sha256": summary_sha256,
                    "stdout_sha256": stdout_sha256,
                    "stdout_tail": stdout_tail,
                    "stderr_sha256": stderr_sha256,
                    "stderr_tail": stderr_tail,
                },
            )
            if signal_state["requested"]:
                record_signal_stop(len(selected) - selected_index - 1)
                break
        if signal_state["requested"]:
            record_signal_stop(0)
    finally:
        signal.signal(signal.SIGUSR1, previous_handler)
    return failures, receipt_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--attempt-id", required=True)
    args = parser.parse_args()
    plan = load_salvage_plan(args.plan, args.plan_sha256)
    failures, receipt_path = execute_salvage_task(
        plan=plan, task_index=args.task_index, attempt_id=args.attempt_id
    )
    print(
        json.dumps(
            {
                "task_index": args.task_index,
                "failure_count": failures,
                "receipt_path": str(receipt_path),
            },
            sort_keys=True,
        )
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
