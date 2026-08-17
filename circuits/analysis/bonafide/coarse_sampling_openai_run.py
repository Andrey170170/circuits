"""Receipt-bound direct OpenAI execution for the coarse qualification smoke."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from circuits.analysis.bonafide.canonical import canonical_sha256, file_sha256
from circuits.analysis.bonafide.coarse_sampling_annotation import (
    cost_plan,
    load_coarse_config,
    validate_decisions,
)
from circuits.labeling.api import openai_stop_reason, openai_usage
from circuits.labeling.io import atomic_write_json, atomic_write_jsonl, read_jsonl
from circuits.labeling.pricing import estimate_cost, load_price_snapshot

RUN_SCHEMA = "adag.process-witness.coarse-openai-run.v1"
EVENT_SCHEMA = "adag.process-witness.coarse-openai-event.v1"
INTENT_SCHEMA = "adag.process-witness.coarse-openai-attempt-intent.v1"
_BOUND_RUN_FILES = (
    "circuits/analysis/bonafide/coarse_sampling_openai_run.py",
    "circuits/analysis/bonafide/coarse_sampling_annotation.py",
    "circuits/analysis/bonafide/canonical.py",
    "circuits/labeling/api.py",
    "circuits/labeling/io.py",
    "circuits/labeling/pricing.py",
    "circuits/labeling/schema.py",
    "scripts/bonafide/run_process_witness_coarse_qualification.py",
)


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unreadable JSON object: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _verify_self_hash(value: Mapping[str, Any], field: str, label: str) -> None:
    payload = dict(value)
    recorded = payload.pop(field, None)
    if not isinstance(recorded, str) or recorded != canonical_sha256(payload):
        raise ValueError(f"{label} self-hash drift")


def load_offline_qualification(root: Path) -> dict[str, Any]:
    """Deep-validate the immutable offline request bundle before spend."""

    manifest = _load_object(root / "manifest.json")
    _verify_self_hash(manifest, "manifest_sha256", "qualification manifest")
    if (
        manifest.get("schema_version")
        != "adag.process-witness.coarse-qualification-bundle.v1"
        or manifest.get("status") != "prepared_offline_no_provider_calls"
        or manifest.get("network_calls_made") != 0
    ):
        raise ValueError("qualification bundle is not an offline prepared artifact")
    for binding in manifest["files"]:
        path = root / binding["path"]
        if (
            not path.is_file()
            or path.stat().st_size != binding["bytes"]
            or file_sha256(path) != binding["sha256"]
        ):
            raise ValueError(f"qualification payload drift: {path}")
    required_files = {
        "cost-plan.json",
        "human-review-template.jsonl",
        "requests.jsonl",
        "units.jsonl",
        "windows.json",
    }
    if {binding["path"] for binding in manifest["files"]} != required_files:
        raise ValueError("qualification payload membership drift")
    requests = read_jsonl(root / "requests.jsonl")
    bindings = manifest["request_bindings_in_order"]
    if len(requests) != 16 or len(bindings) != len(requests):
        raise ValueError("qualification request cardinality drift")
    for request, binding in zip(requests, bindings, strict=True):
        if request["request_id"] != binding["request_id"]:
            raise ValueError("qualification request order drift")
        if canonical_sha256(request["provider_body"]) != request["body_sha256"]:
            raise ValueError("qualification provider body hash drift")
        if any(request[key] != binding[key] for key in binding):
            raise ValueError("qualification request binding drift")
    plan = _load_object(root / "cost-plan.json")
    _verify_self_hash(plan, "cost_plan_sha256", "qualification cost plan")
    if plan.get("request_count") != len(requests) or plan.get("transport") != "live":
        raise ValueError("qualification cost plan contract drift")
    config_path = Path(manifest["config_path"])
    if file_sha256(config_path) != manifest["config_sha256"]:
        raise ValueError("qualification config drift")
    config = load_coarse_config(config_path)
    price_path = Path(plan["price_snapshot_path"])
    if file_sha256(price_path) != plan["price_snapshot_sha256"]:
        raise ValueError("qualification price snapshot drift")
    prices = load_price_snapshot(price_path)
    recomputed = cost_plan(requests, config, prices)
    for key, expected in recomputed.items():
        if plan.get(key) != expected:
            raise ValueError(f"qualification cost plan recomputation drift: {key}")
    return {"manifest": manifest, "requests": requests, "cost_plan": plan}


def _response_dict(response: Any) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        value = response.model_dump(mode="json")
    elif isinstance(response, dict):
        value = response
    else:
        raise TypeError("OpenAI response is not serializable")
    if not isinstance(value, dict):
        raise TypeError("OpenAI response serialization is not an object")
    return value


def _output_text(response: Any, raw: Mapping[str, Any]) -> str:
    direct = getattr(response, "output_text", None)
    if isinstance(direct, str):
        return direct
    texts: list[str] = []
    for item in raw.get("output", []):
        if not isinstance(item, Mapping):
            continue
        texts.extend(
            content["text"]
            for content in item.get("content", [])
            if isinstance(content, Mapping) and isinstance(content.get("text"), str)
        )
    return "".join(texts)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _source_revision() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    if Path(_git(root, "rev-parse", "--show-toplevel")) != root:
        raise ValueError("coarse OpenAI run repository root drift")
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=no")
    if status:
        raise ValueError("coarse OpenAI run requires a clean tracked worktree")
    commit = _git(root, "rev-parse", "HEAD")
    files = []
    for relative in _BOUND_RUN_FILES:
        if _git(root, "ls-files", "--error-unmatch", "--", relative) != relative:
            raise ValueError(f"coarse OpenAI run source is untracked: {relative}")
        path = root / relative
        blob = _git(root, "rev-parse", f"{commit}:{relative}")
        committed = subprocess.run(
            ["git", "cat-file", "blob", blob],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
        sha256 = hashlib.sha256(committed).hexdigest()
        if file_sha256(path) != sha256:
            raise ValueError(f"coarse OpenAI run source differs from HEAD: {relative}")
        files.append({"path": relative, "git_blob": blob, "sha256": sha256})
    return {
        "repo_root": str(root),
        "git_commit": commit,
        "git_tree": _git(root, "rev-parse", "HEAD^{tree}"),
        "tracked_worktree_clean": True,
        "files": files,
    }


def _readonly_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def _finalize_failed_run(
    *,
    output_root: Path,
    run_intent: Mapping[str, Any],
    events: list[dict[str, Any]],
    failure_kind: str,
    failure_message: str,
    known_priced_cost_usd: float,
    cost_complete: bool,
) -> None:
    atomic_write_jsonl(output_root / "events.jsonl", events)
    failed = dict(run_intent)
    failed.pop("run_intent_sha256")
    failed.update(
        {
            "status": "failed_closed_no_resume",
            "completed_at": _utc_now(),
            "failure_kind": failure_kind,
            "failure_message": failure_message[:2000],
            "event_count": len(events),
            "known_priced_cost_usd": known_priced_cost_usd,
            "cost_complete": cost_complete,
            "actual_total_cost_usd": (known_priced_cost_usd if cost_complete else None),
            "events_jsonl_sha256": file_sha256(output_root / "events.jsonl"),
            "record_bindings_in_order": [
                {
                    "request_id": event["request_id"],
                    "event_sha256": event["event_sha256"],
                }
                for event in events
            ],
        }
    )
    failed["run_manifest_sha256"] = canonical_sha256(failed)
    atomic_write_json(output_root / "run-manifest.json", failed)
    _readonly_tree(output_root)


def run_direct_qualification(
    *,
    qualification_root: Path,
    output_root: Path,
    maximum_authorized_cost_usd: float,
    authorization_note: str,
    client: Any | None = None,
) -> dict[str, Any]:
    """Execute each predeclared request once and retain every provider receipt."""

    loaded = load_offline_qualification(qualification_root)
    plan = loaded["cost_plan"]
    projected = float(plan["projected_upper_bound_usd"])
    if not 0 < maximum_authorized_cost_usd <= 10.0:
        raise ValueError("coarse qualification authorization must be in (0, 10]")
    if projected > maximum_authorized_cost_usd:
        raise ValueError("persisted cost upper bound exceeds authorization")
    if output_root.exists():
        raise FileExistsError(f"coarse OpenAI run already exists: {output_root}")
    source_revision = _source_revision()
    output_root.mkdir(parents=True)
    (output_root / "intents").mkdir()
    (output_root / "raw").mkdir()
    (output_root / "records").mkdir()

    price_path = Path(plan["price_snapshot_path"])
    if file_sha256(price_path) != plan["price_snapshot_sha256"]:
        raise ValueError("coarse OpenAI price snapshot drift")
    prices = load_price_snapshot(price_path)
    run_intent = {
        "schema_version": RUN_SCHEMA,
        "status": "intent_persisted_before_provider_calls",
        "created_at": _utc_now(),
        "qualification_root": str(qualification_root.resolve()),
        "qualification_manifest_sha256": loaded["manifest"]["manifest_sha256"],
        "cost_plan_sha256": plan["cost_plan_sha256"],
        "projected_upper_bound_usd": projected,
        "maximum_authorized_cost_usd": maximum_authorized_cost_usd,
        "authorization_note": authorization_note,
        "request_ids_in_order": [
            request["request_id"] for request in loaded["requests"]
        ],
        "environment": {
            "hostname": platform.node(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "openai_sdk_version": importlib.metadata.version("openai"),
            "endpoint_identity": "https://api.openai.com/v1",
            "openai_project_sha256": (
                hashlib.sha256(os.environ["OPENAI_PROJECT_ID"].encode()).hexdigest()
                if os.environ.get("OPENAI_PROJECT_ID")
                else None
            ),
            "openai_organization_sha256": (
                hashlib.sha256(os.environ["OPENAI_ORG_ID"].encode()).hexdigest()
                if os.environ.get("OPENAI_ORG_ID")
                else None
            ),
        },
        "source_revision": source_revision,
    }
    run_intent["run_intent_sha256"] = canonical_sha256(run_intent)
    atomic_write_json(output_root / "run-intent.json", run_intent)

    if client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is unavailable after run intent persistence"
            )
        from openai import OpenAI

        client = OpenAI(
            api_key=api_key,
            base_url="https://api.openai.com/v1",
            project=os.environ.get("OPENAI_PROJECT_ID"),
            organization=os.environ.get("OPENAI_ORG_ID"),
            timeout=180.0,
            max_retries=0,
        )

    events: list[dict[str, Any]] = []
    total_cost = 0.0
    for request in loaded["requests"]:
        intent = {
            "schema_version": INTENT_SCHEMA,
            "created_at": _utc_now(),
            "run_intent_sha256": run_intent["run_intent_sha256"],
            "request_id": request["request_id"],
            "body_sha256": request["body_sha256"],
            "repeat_of_request_id": request["repeat_of_request_id"],
            "attempt_number": 0,
        }
        intent["intent_sha256"] = canonical_sha256(intent)
        atomic_write_json(
            output_root / "intents" / f"{request['request_id']}.json", intent
        )
        started = time.monotonic()
        try:
            response = client.responses.create(**request["provider_body"])
        except Exception as error:
            failed = {
                "schema_version": EVENT_SCHEMA,
                "status": "transport_error_unknown_provider_state",
                "recorded_at": _utc_now(),
                "request_id": request["request_id"],
                "body_sha256": request["body_sha256"],
                "intent_sha256": intent["intent_sha256"],
                "latency_seconds": time.monotonic() - started,
                "error_type": type(error).__name__,
                "error_message": str(error)[:2000],
            }
            failed["event_sha256"] = canonical_sha256(failed)
            atomic_write_json(
                output_root / "records" / f"{request['request_id']}.json", failed
            )
            events.append(failed)
            _finalize_failed_run(
                output_root=output_root,
                run_intent=run_intent,
                events=events,
                failure_kind="transport_error_unknown_provider_state",
                failure_message=str(error),
                known_priced_cost_usd=total_cost,
                cost_complete=False,
            )
            raise RuntimeError(
                "provider call state is unknown; no automatic retry is permitted"
            ) from error

        raw_path = output_root / "raw" / f"{request['request_id']}.json"
        try:
            raw = _response_dict(response)
            atomic_write_json(raw_path, raw)
            raw_text = _output_text(response, raw)
            usage = openai_usage(getattr(response, "usage", None))
            cost = estimate_cost(
                prices,
                provider="openai",
                model=request["provider_body"]["model"],
                transport="live",
                usage=usage,
            )
        except Exception as error:
            failed = {
                "schema_version": EVENT_SCHEMA,
                "status": "post_response_processing_error",
                "recorded_at": _utc_now(),
                "request_id": request["request_id"],
                "body_sha256": request["body_sha256"],
                "intent_sha256": intent["intent_sha256"],
                "provider_request_id": getattr(response, "id", None),
                "model_requested": request["provider_body"]["model"],
                "model_resolved": getattr(response, "model", None),
                "response_status": getattr(response, "status", None),
                "latency_seconds": time.monotonic() - started,
                "raw_response_path": (
                    str(raw_path.relative_to(output_root))
                    if raw_path.exists()
                    else None
                ),
                "raw_response_sha256": (
                    file_sha256(raw_path) if raw_path.exists() else None
                ),
                "error_type": type(error).__name__,
                "error_message": str(error)[:2000],
            }
            failed["event_sha256"] = canonical_sha256(failed)
            atomic_write_json(
                output_root / "records" / f"{request['request_id']}.json", failed
            )
            events.append(failed)
            _finalize_failed_run(
                output_root=output_root,
                run_intent=run_intent,
                events=events,
                failure_kind="post_response_processing_error",
                failure_message=str(error),
                known_priced_cost_usd=total_cost,
                cost_complete=False,
            )
            raise RuntimeError(
                "billable provider response could not be processed; run frozen failed"
            ) from error
        provider_status = getattr(response, "status", None)
        validation_status = (
            "success" if provider_status == "completed" else "provider_incomplete"
        )
        decisions = None
        validation_error = None
        try:
            parsed = json.loads(raw_text)
            decisions = validate_decisions(
                parsed, focal_unit_ids=request["focal_unit_ids"]
            )
        except (json.JSONDecodeError, ValueError, TypeError) as error:
            if provider_status == "completed":
                validation_status = "invalid_output"
            validation_error = str(error)[:2000]
        billable_cost = cost.total_cost if cost.complete else None
        if billable_cost is not None:
            total_cost += billable_cost
        event = {
            "schema_version": EVENT_SCHEMA,
            "status": validation_status,
            "recorded_at": _utc_now(),
            "request_id": request["request_id"],
            "body_sha256": request["body_sha256"],
            "repeat_of_request_id": request["repeat_of_request_id"],
            "intent_sha256": intent["intent_sha256"],
            "provider_request_id": getattr(response, "id", None),
            "model_requested": request["provider_body"]["model"],
            "model_resolved": getattr(response, "model", None),
            "response_status": provider_status,
            "stop_reason": openai_stop_reason(response),
            "latency_seconds": time.monotonic() - started,
            "raw_response_path": str(raw_path.relative_to(output_root)),
            "raw_response_sha256": file_sha256(raw_path),
            "raw_text": raw_text,
            "raw_text_sha256": hashlib.sha256(raw_text.encode()).hexdigest(),
            "decisions": decisions,
            "validation_error": validation_error,
            "usage": usage.model_dump(mode="json"),
            "cost": cost.model_dump(mode="json"),
            "cumulative_cost_usd": total_cost,
        }
        event["event_sha256"] = canonical_sha256(event)
        record_path = output_root / "records" / f"{request['request_id']}.json"
        atomic_write_json(record_path, event)
        events.append(event)
        if billable_cost is None:
            _finalize_failed_run(
                output_root=output_root,
                run_intent=run_intent,
                events=events,
                failure_kind="usage_unpriceable",
                failure_message="provider usage could not be priced completely",
                known_priced_cost_usd=total_cost,
                cost_complete=False,
            )
            raise RuntimeError("provider usage could not be priced completely")
        if total_cost > maximum_authorized_cost_usd:
            _finalize_failed_run(
                output_root=output_root,
                run_intent=run_intent,
                events=events,
                failure_kind="authorization_exceeded",
                failure_message="actual cumulative API cost exceeded authorization",
                known_priced_cost_usd=total_cost,
                cost_complete=True,
            )
            raise RuntimeError("actual cumulative API cost exceeded authorization")

    atomic_write_jsonl(output_root / "events.jsonl", events)
    complete = {
        **run_intent,
        "status": "complete",
        "completed_at": _utc_now(),
        "event_count": len(events),
        "success_count": sum(event["status"] == "success" for event in events),
        "invalid_output_count": sum(
            event["status"] == "invalid_output" for event in events
        ),
        "provider_incomplete_count": sum(
            event["status"] == "provider_incomplete" for event in events
        ),
        "known_priced_cost_usd": total_cost,
        "cost_complete": True,
        "actual_total_cost_usd": total_cost,
        "events_jsonl_sha256": file_sha256(output_root / "events.jsonl"),
        "record_bindings_in_order": [
            {"request_id": event["request_id"], "event_sha256": event["event_sha256"]}
            for event in events
        ],
    }
    complete.pop("run_intent_sha256")
    complete["run_manifest_sha256"] = canonical_sha256(complete)
    atomic_write_json(output_root / "run-manifest.json", complete)
    _readonly_tree(output_root)
    return complete
