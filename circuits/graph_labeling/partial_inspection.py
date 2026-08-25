"""Build inspection-only overlays from saved OpenAI Batch collection attempts.

This module deliberately does not participate in canonical collection or label-set
finalization.  Its single public interface accepts one exact, already-downloaded
collection attempt and produces a provenance-bound observatory overlay in which
every selected occurrence is either independently validated or visibly invalid.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from circuits.analysis.bonafide.canonical import canonical_sha256
from circuits.graph_labeling.evidence import allowed_evidence_ids
from circuits.graph_labeling.openai_batch import (
    _batch_root,
    _context,
    _load_bound_plan,
    _load_bound_submission,
    _load_hashed,
    _response_text,
    _validate_remote,
)
from circuits.graph_labeling.runtime import (
    _packets_by_occurrence,
    _read_stable_json,
    normalize_structured_label,
)
from circuits.graph_labeling.schema import (
    OccurrenceRoleLabel,
    PromptRequest,
    require_safe_id,
)
from circuits.labeling.api import openai_usage
from circuits.labeling.io import atomic_write_json
from circuits.observatory import LABEL_SET_SCHEMA
from circuits.observatory.server import validate_site_bundle

PARTIAL_NORMALIZATION_POLICY = "append-first-seen-valid-claim-citations-to-top-level.v1"
PARTIAL_INSPECTION_SCHEMA = "adag.graph-labeling.openai-batch-partial-inspection.v1"
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


def _attempt_bundle(
    *,
    batch_root: Path,
    attempt_id: str,
    submission: Mapping[str, Any],
    request_count: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], bytes]:
    require_safe_id(attempt_id, "attempt_id")
    if not attempt_id.startswith("attempt-"):
        raise ValueError("collection attempt id must start with 'attempt-'")
    attempt_root = batch_root / "collection-attempts" / attempt_id
    intent = _load_hashed(attempt_root / "intent.json")
    snapshot_receipt = _load_hashed(attempt_root / "snapshot.json")
    download = _load_hashed(attempt_root / "download.json")
    output_path = attempt_root / "output.jsonl"
    try:
        first = output_path.read_bytes()
        second = output_path.read_bytes()
    except OSError as error:
        raise ValueError(
            "saved collection attempt has no readable provider output"
        ) from error
    if first != second:
        raise ValueError("saved provider output changed while being read")
    output_sha256 = hashlib.sha256(first).hexdigest()
    remote = snapshot_receipt.get("remote")
    if not isinstance(remote, Mapping):
        raise ValueError("saved collection attempt remote snapshot is malformed")
    if (
        intent.get("schema_version")
        != "adag.graph-labeling.openai-batch-collection-intent.v1"
        or intent.get("attempt_id") != attempt_id
        or intent.get("submission_sha256") != submission["content_hash"]
        or intent.get("batch_identity_sha256") != submission["batch_identity_sha256"]
        or intent.get("batch_id") != submission["batch_id"]
        or snapshot_receipt.get("schema_version")
        != "adag.graph-labeling.openai-batch-download-snapshot.v1"
        or snapshot_receipt.get("attempt_id") != attempt_id
        or snapshot_receipt.get("attempt_intent_sha256") != intent["content_hash"]
        or snapshot_receipt.get("submission_sha256") != submission["content_hash"]
        or download.get("schema_version")
        != "adag.graph-labeling.openai-batch-download.v1"
        or download.get("attempt_id") != attempt_id
        or download.get("attempt_intent_sha256") != intent["content_hash"]
        or download.get("snapshot_sha256") != snapshot_receipt["content_hash"]
        or download.get("output_file_id") != remote.get("output_file_id")
        or download.get("output_file_sha256") != output_sha256
        or download.get("error_file_id") != remote.get("error_file_id")
        or download.get("error_file_sha256") != _EMPTY_SHA256
    ):
        raise ValueError("saved collection attempt provenance binding drift")
    _validate_remote(remote, submission)
    counts = remote.get("request_counts")
    if (
        remote.get("status") != "completed"
        or not isinstance(counts, Mapping)
        or counts.get("total") != request_count
        or counts.get("completed") != request_count
        or counts.get("failed") != 0
        or not first
        or remote.get("error_file_id") is not None
    ):
        raise ValueError(
            "saved collection attempt is not a complete zero-failure Batch"
        )
    return intent, snapshot_receipt, download, first


def _normalize_redundant_citations(
    payload: dict[str, Any], valid_evidence_ids: set[str]
) -> tuple[dict[str, Any], list[str]]:
    """Apply the one permitted inspection normalization without typo repair."""

    normalized = dict(payload)
    cited = payload.get("cited_evidence_ids")
    claims = payload.get("claim_citations")
    if not isinstance(cited, list) or any(not isinstance(item, str) for item in cited):
        return normalized, []
    if not isinstance(claims, dict):
        return normalized, []
    appended: list[str] = []
    seen = set(cited)
    for evidence_ids in claims.values():
        if not isinstance(evidence_ids, list):
            continue
        for evidence_id in evidence_ids:
            if (
                isinstance(evidence_id, str)
                and evidence_id in valid_evidence_ids
                and evidence_id not in seen
            ):
                seen.add(evidence_id)
                appended.append(evidence_id)
    if appended:
        normalized["cited_evidence_ids"] = [*cited, *appended]
    return normalized, appended


def _row_outcomes(
    output: bytes,
    *,
    requests: dict[str, PromptRequest],
    packets: dict[str, Any],
    method: Any,
) -> tuple[list[dict[str, Any]], dict[str, OccurrenceRoleLabel], dict[str, list[str]]]:
    try:
        lines = output.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("saved OpenAI Batch output is not UTF-8") from error
    outcomes: list[dict[str, Any]] = []
    labels: dict[str, OccurrenceRoleLabel] = {}
    request_errors: dict[str, list[str]] = {}
    seen: set[str] = set()
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        outcome: dict[str, Any] = {
            "line_number": line_number,
            "request_id": None,
            "occurrence_id": None,
            "normalization_policy_id": PARTIAL_NORMALIZATION_POLICY,
            "raw_payload_bytes_sha256": None,
            "raw_payload_sha256": None,
            "normalized_payload_sha256": None,
            "appended_evidence_ids": [],
            "validation_outcome": "invalid",
            "validation_errors": [],
        }
        try:
            item = json.loads(line)
        except json.JSONDecodeError as error:
            outcome["validation_errors"] = [f"malformed Batch JSON row: {error.msg}"]
            outcomes.append(outcome)
            continue
        custom_id = item.get("custom_id") if isinstance(item, dict) else None
        outcome["request_id"] = custom_id if isinstance(custom_id, str) else None
        request = requests.get(custom_id) if isinstance(custom_id, str) else None
        if request is None:
            outcome["validation_errors"] = ["unknown or missing custom_id"]
            outcomes.append(outcome)
            continue
        outcome["occurrence_id"] = request.occurrence_id
        if custom_id in seen:
            error = "duplicate Batch row for custom_id"
            request_errors.setdefault(custom_id, []).append(error)
            labels.pop(custom_id, None)
            outcome["validation_errors"] = [error]
            outcomes.append(outcome)
            continue
        seen.add(custom_id)
        try:
            response = item.get("response")
            if (
                item.get("error") is not None
                or not isinstance(response, dict)
                or response.get("status_code") != 200
            ):
                raise ValueError("Batch row contains a failed response")
            body = response.get("body")
            if (
                not isinstance(body, dict)
                or body.get("status") != "completed"
                or body.get("error") is not None
            ):
                raise ValueError("Responses result is incomplete or failed")
            observed_model = body.get("model")
            if not isinstance(observed_model, str) or not (
                observed_model == request.generation.model
                or observed_model.startswith(request.generation.model + "-")
            ):
                raise ValueError("Responses result model drift")
            parsed_usage = openai_usage(body.get("usage"))
            if parsed_usage.input_tokens is None or parsed_usage.output_tokens is None:
                raise ValueError("Responses result usage is incomplete")
            raw_usage = body.get("usage")
            raw_total = (
                raw_usage.get("total_tokens") if isinstance(raw_usage, dict) else None
            )
            if raw_total is not None and raw_total != (
                parsed_usage.input_tokens + parsed_usage.output_tokens
            ):
                raise ValueError("Responses result total-token usage mismatch")
            response_id = body.get("id")
            provider_request_id = response.get("request_id")
            if not isinstance(response_id, str) or not response_id:
                raise ValueError("Responses result lacks response id")
            if not isinstance(provider_request_id, str) or not provider_request_id:
                raise ValueError("Responses Batch row lacks provider request id")
            outcome.update(
                {
                    "response_id": response_id,
                    "provider_request_id": provider_request_id,
                    "exact_model": observed_model,
                    "usage": parsed_usage.model_dump(mode="json"),
                }
            )
            text = _response_text(body)
            outcome["raw_payload_bytes_sha256"] = hashlib.sha256(
                text.encode("utf-8")
            ).hexdigest()
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"malformed structured payload: {error.msg}"
                ) from error
            if not isinstance(payload, dict):
                raise ValueError("structured payload is not an object")
            outcome["raw_payload_sha256"] = canonical_sha256(payload)
            packet = packets[request.occurrence_id]
            normalized, appended = _normalize_redundant_citations(
                payload, allowed_evidence_ids(packet)
            )
            normalized_sha256 = canonical_sha256(normalized)
            outcome["normalized_payload_sha256"] = normalized_sha256
            outcome["appended_evidence_ids"] = appended
            label = normalize_structured_label(
                normalized,
                packet=packet,
                method=method,
                logical_request_sha256=request.logical_request_sha256,
                result_sha256=outcome["raw_payload_sha256"],
            )
            labels[custom_id] = label
            outcome["validation_outcome"] = (
                "valid_normalized" if appended else "valid_unchanged"
            )
        except (KeyError, TypeError, ValueError) as error:
            message = str(error) or type(error).__name__
            request_errors.setdefault(custom_id, []).append(message)
            outcome["validation_errors"] = [message]
        outcomes.append(outcome)
    for request_id in sorted(set(requests) - seen):
        request_errors.setdefault(request_id, []).append(
            "provider output omitted frozen custom_id"
        )
    return outcomes, labels, request_errors


def _bound_viewer(
    *, site: Path, source_binding: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, str], str]:
    validate_site_bundle(site)
    _read_stable_json(
        site / "viewer-manifest.json",
        str(source_binding["viewer_manifest_sha256"]),
    )
    catalog = _read_stable_json(
        site / "catalog.json", str(source_binding["catalog_sha256"])
    )
    selected_artifact = str(source_binding["artifact_id"])
    _read_stable_json(
        site / "traces" / f"{selected_artifact}.json",
        str(source_binding["trace_file_sha256"]),
    )
    traces = {
        str(item["artifact_id"]): _read_stable_json(
            site / "traces" / f"{item['artifact_id']}.json"
        )
        for item in catalog["traces"]
    }
    source_hashes = {
        artifact_id: str(trace["artifact"]["source_hash"])
        for artifact_id, trace in traces.items()
    }
    basis_schemas = {
        str(node["basis"]["schema_version"])
        for trace in traces.values()
        for node in trace["nodes"]
    }
    if len(basis_schemas) != 1:
        raise ValueError("observatory contains multiple basis schemas")
    return catalog, traces, source_hashes, next(iter(basis_schemas))


def export_openai_batch_partial_overlay(
    run_root: Path,
    method_id: str,
    attempt_id: str,
    site_root: Path,
    destination: Path,
) -> dict[str, Any]:
    """Export one saved Batch attempt as a noncanonical inspection overlay.

    Every selected occurrence is independently classified as a validated label or
    ``invalid_result``. The only repair is the versioned redundant-citation union
    policy; unknown evidence identifiers are retained and therefore remain invalid.
    """

    root, manifest, method, request_values, canonical_label_set_id = _context(
        run_root, method_id
    )
    requests = {request.request_id: request for request in request_values}
    batch_root = _batch_root(root, canonical_label_set_id)
    plan = _load_bound_plan(
        batch_root, manifest, method, request_values, canonical_label_set_id
    )
    submission = _load_bound_submission(batch_root, plan)
    intent, snapshot, download, output = _attempt_bundle(
        batch_root=batch_root,
        attempt_id=attempt_id,
        submission=submission,
        request_count=len(requests),
    )
    packets = _packets_by_occurrence(root, manifest)
    outcomes, labels, request_errors = _row_outcomes(
        output, requests=requests, packets=packets, method=method
    )
    request_by_occurrence = {
        request.occurrence_id: request for request in request_values
    }
    if set(request_by_occurrence) != set(manifest["occurrence_ids"]):
        raise ValueError("frozen request coverage differs from selected occurrences")

    site = site_root.expanduser().resolve()
    catalog, traces, source_hashes, basis_schema = _bound_viewer(
        site=site, source_binding=manifest["source_binding"]
    )
    provider_output_sha256 = hashlib.sha256(output).hexdigest()
    partial_identity_sha256 = canonical_sha256(
        {
            "canonical_label_set_id": canonical_label_set_id,
            "normalization_policy_id": PARTIAL_NORMALIZATION_POLICY,
            "provider_output_sha256": provider_output_sha256,
            "collection_attempt_intent_sha256": intent["content_hash"],
            "download_snapshot_sha256": snapshot["content_hash"],
            "download_receipt_sha256": download["content_hash"],
        }
    )
    partial_label_set_id = (
        f"{canonical_label_set_id}-partial-v1-{partial_identity_sha256[:12]}"
    )
    outcome_by_request = {
        outcome["request_id"]: outcome
        for outcome in outcomes
        if outcome.get("request_id") in requests
    }
    valid_unchanged_count = sum(
        outcome["validation_outcome"] == "valid_unchanged" for outcome in outcomes
    )
    valid_normalized_count = sum(
        outcome["validation_outcome"] == "valid_normalized" for outcome in outcomes
    )
    invalid_row_count = sum(
        outcome["validation_outcome"] == "invalid" for outcome in outcomes
    )
    unexpected_row_count = sum(
        outcome.get("request_id") not in requests for outcome in outcomes
    )
    labels_by_trace: dict[str, list[dict[str, Any]]] = {}
    valid_count = 0
    invalid_count = 0
    unselected_count = 0
    selected_artifact = str(manifest["source_binding"]["artifact_id"])
    for artifact_id, trace in traces.items():
        records: list[dict[str, Any]] = []
        for node in trace["nodes"]:
            request = (
                request_by_occurrence.get(node["id"])
                if artifact_id == selected_artifact
                else None
            )
            if request is None:
                unselected_count += 1
                records.append(
                    {
                        "occurrence_id": node["id"],
                        "basis_id": node["basis_id"],
                        "label": None,
                        "status": "not_selected",
                        "confidence": None,
                    }
                )
                continue
            outcome = outcome_by_request.get(request.request_id)
            label = labels.get(request.request_id)
            errors = request_errors.get(request.request_id, [])
            if label is not None and not errors:
                valid_count += 1
                role = label.model_dump(mode="json")
                records.append(
                    {
                        "occurrence_id": node["id"],
                        "basis_id": node["basis_id"],
                        "label": label.label,
                        "description": label.label,
                        "status": label.status,
                        "confidence": label.confidence,
                        "role": role,
                        "inspection": outcome,
                    }
                )
            else:
                invalid_count += 1
                all_errors = errors or ["no independently valid provider row"]
                records.append(
                    {
                        "occurrence_id": node["id"],
                        "basis_id": node["basis_id"],
                        "label": "invalid_result",
                        "description": "invalid_result",
                        "status": "invalid_result",
                        "confidence": None,
                        "role": {
                            "apparent_role": "invalid_result",
                            "rationale": " | ".join(all_errors),
                            "limitations": [
                                "Partial inspection only; this result failed canonical validation."
                            ],
                        },
                        "inspection": {
                            **(outcome or {}),
                            "validation_outcome": "invalid",
                            "validation_errors": all_errors,
                        },
                    }
                )
        labels_by_trace[artifact_id] = records

    remote = snapshot["remote"]
    provenance = {
        "schema_version": PARTIAL_INSPECTION_SCHEMA,
        "normalization_policy_id": PARTIAL_NORMALIZATION_POLICY,
        "partial_identity_sha256": partial_identity_sha256,
        "canonical_label_set_id": canonical_label_set_id,
        "canonical_finalized": False,
        "run_manifest_sha256": manifest["content_hash"],
        "study_sha256": manifest["study_sha256"],
        "method_id": method.method_id,
        "method_sha256": method.identity_sha256,
        "batch_identity_sha256": submission["batch_identity_sha256"],
        "batch_id": submission["batch_id"],
        "input_file_id": submission["input_file_id"],
        "output_file_id": remote["output_file_id"],
        "submission_sha256": submission["content_hash"],
        "plan_sha256": plan["content_hash"],
        "collection_attempt_id": attempt_id,
        "collection_attempt_intent_sha256": intent["content_hash"],
        "download_snapshot_sha256": snapshot["content_hash"],
        "download_receipt_sha256": download["content_hash"],
        "provider_output_sha256": provider_output_sha256,
        "source_viewer_manifest_sha256": manifest["source_binding"][
            "viewer_manifest_sha256"
        ],
        "source_catalog_sha256": manifest["source_binding"]["catalog_sha256"],
        "source_trace_file_sha256": manifest["source_binding"]["trace_file_sha256"],
        "row_outcomes": outcomes,
    }
    model = catalog["model"]
    core = {
        "schema_version": LABEL_SET_SCHEMA,
        "label_set_id": partial_label_set_id,
        "name": f"{method.method_id} partial inspection",
        "synthetic": False,
        "inspection_only": True,
        "canonical_finalized": False,
        "warning": (
            "PARTIAL INSPECTION ONLY. Not a canonical/finalized scientific label set; "
            "invalid_result entries failed independent validation."
        ),
        "method": "openai_batch_partial_inspection_v1",
        "method_sha256": method.identity_sha256,
        "study_sha256": manifest["study_sha256"],
        "model_id": model["model_id"],
        "model_revision": model["model_revision"],
        "basis_schema": basis_schema,
        "polarity_derivation": "activation-sign-nonnegative-positive.v1",
        "source_trace_hashes": source_hashes,
        "partial_inspection": provenance,
        "labels_by_trace": labels_by_trace,
    }
    overlay = {**core, "content_hash": canonical_sha256(core)}
    output_path = destination.expanduser().resolve()
    if output_path.exists():
        try:
            existing = json.loads(output_path.read_bytes())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                "existing partial inspection overlay is unreadable"
            ) from error
        if existing != overlay:
            raise FileExistsError(
                "destination exists with different partial inspection content"
            )
        idempotent = True
    else:
        atomic_write_json(output_path, overlay)
        idempotent = False
    receipt_core = {
        "schema_version": "adag.graph-labeling.partial-inspection-export-receipt.v1",
        "destination": str(output_path),
        "label_set_id": partial_label_set_id,
        "content_sha256": overlay["content_hash"],
        "provider_output_sha256": provider_output_sha256,
        "normalization_policy_id": PARTIAL_NORMALIZATION_POLICY,
        "valid_count": valid_count,
        "valid_unchanged_count": valid_unchanged_count,
        "valid_normalized_count": valid_normalized_count,
        "invalid_count": invalid_count,
        "provider_row_count": len(outcomes),
        "invalid_provider_row_count": invalid_row_count,
        "unexpected_provider_row_count": unexpected_row_count,
        "unselected_count": unselected_count,
        "idempotent": idempotent,
    }
    return {**receipt_core, "receipt_hash": canonical_sha256(receipt_core)}


__all__ = [
    "PARTIAL_INSPECTION_SCHEMA",
    "PARTIAL_NORMALIZATION_POLICY",
    "export_openai_batch_partial_overlay",
]
