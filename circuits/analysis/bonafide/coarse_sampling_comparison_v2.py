"""Offline descriptive comparison for the frozen coarse-label qualification."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from circuits.analysis.bonafide.canonical import canonical_sha256, file_sha256
from circuits.analysis.bonafide.coarse_sampling_annotation import (
    BOUNDARY_CONCERNS,
    COARSE_TAGS,
    CONFIDENCE_VALUES,
)
from circuits.analysis.bonafide.coarse_sampling_annotation_v2 import (
    ARM_FULL_UNIT,
    ARM_IDS,
    ARM_TARGET_ONLY,
    load_v2_qualification,
)
from circuits.labeling.io import read_jsonl

REPORT_SCHEMA = "adag.process-witness.coarse-comparison-report.v1"
EXAMPLE_SCHEMA = "adag.process-witness.coarse-comparison-review-example.v1"
MANIFEST_SCHEMA = "adag.process-witness.coarse-comparison-bundle.v1"
USAGE_FIELDS = (
    "input_tokens",
    "uncached_input_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "output_tokens",
    "reasoning_tokens",
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
    observed = payload.pop(field, None)
    if not isinstance(observed, str) or observed != canonical_sha256(payload):
        raise ValueError(f"{label} self-hash drift")


def _validate_decisions(event: Mapping[str, Any], expected: Sequence[str]) -> None:
    decisions = event.get("decisions")
    if not isinstance(decisions, list) or len(decisions) != len(expected):
        raise ValueError("coarse comparison decision cardinality drift")
    if [item.get("unit_id") for item in decisions] != list(expected):
        raise ValueError("coarse comparison decision target order drift")
    for decision in decisions:
        if (
            decision.get("tag") not in COARSE_TAGS
            or decision.get("confidence") not in CONFIDENCE_VALUES
            or not isinstance(decision.get("boundary_concerns"), list)
            or any(
                value not in BOUNDARY_CONCERNS
                for value in decision["boundary_concerns"]
            )
            or len(set(decision["boundary_concerns"]))
            != len(decision["boundary_concerns"])
            or not isinstance(decision.get("boundary_note"), str)
        ):
            raise ValueError("coarse comparison decision schema drift")


def load_completed_comparison_inputs(
    *, run_root: Path, qualification_root: Path
) -> dict[str, Any]:
    """Load the completed v2 run and fail closed on any missing result."""

    qualification = load_v2_qualification(qualification_root)
    intent = _load_object(run_root / "run-intent.json")
    _verify_self_hash(intent, "run_intent_sha256", "coarse comparison run intent")
    collection = _load_object(run_root / "collection-manifest.json")
    _verify_self_hash(
        collection,
        "collection_manifest_sha256",
        "coarse comparison collection manifest",
    )
    if (
        intent.get("schema_version")
        != "adag.process-witness.coarse-openai-batch-run.v2"
        or intent.get("qualification_manifest_sha256")
        != qualification["manifest"]["manifest_sha256"]
        or Path(intent.get("qualification_root", "")).resolve()
        != qualification_root.resolve()
    ):
        raise ValueError("coarse comparison run/qualification binding drift")
    if (
        collection.get("schema_version")
        != "adag.process-witness.coarse-openai-batch-collection.v2"
        or collection.get("status") != "complete"
        or collection.get("qualification_decisions_ready") is not True
        or collection.get("cost_complete") is not True
        or collection.get("authorization_exceeded") is not False
        or collection.get("request_count") != 32
        or collection.get("success_count") != 32
        or collection.get("failure_count") != 0
        or collection.get("exact_target_coverage") is not True
        or collection.get("unique_arm_target_coverage") != 144
        or collection.get("run_intent_sha256") != intent["run_intent_sha256"]
    ):
        raise ValueError("coarse comparison collection is not complete and exact")
    events_path = run_root / "events.jsonl"
    if file_sha256(events_path) != collection.get("events_jsonl_sha256"):
        raise ValueError("coarse comparison events file drift")
    events = read_jsonl(events_path)
    requests = qualification["requests"]
    bindings = collection.get("event_bindings_in_order")
    if len(events) != 32 or not isinstance(bindings, list) or len(bindings) != 32:
        raise ValueError("coarse comparison event cardinality drift")
    for event, request, binding in zip(events, requests, bindings, strict=True):
        if not isinstance(binding, Mapping):
            raise ValueError("coarse comparison event binding is not an object")
        payload = dict(event)
        observed_hash = payload.pop("event_sha256", None)
        if (
            event.get("request_id") != request["request_id"]
            or event.get("request_id") != binding.get("request_id")
            or observed_hash != binding.get("event_sha256")
            or observed_hash != canonical_sha256(payload)
            or event.get("validation_status") != "success"
            or event.get("arm_id") != request["arm_id"]
            or event.get("source_v1_request_id") != request["source_v1_request_id"]
            or event.get("repeat_of_request_id") != request["repeat_of_request_id"]
        ):
            raise ValueError("coarse comparison event/request binding drift")
        _validate_decisions(event, request["focal_unit_ids"])
    return {
        "qualification": qualification,
        "run_intent": intent,
        "collection": collection,
        "events": events,
    }


def _decision_map(event: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["unit_id"]: dict(item) for item in event["decisions"]}


def _exact_decision(decision: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "tag": decision["tag"],
        "confidence": decision["confidence"],
        "boundary_concerns": decision["boundary_concerns"],
        "boundary_note": decision["boundary_note"],
    }


def _pair_record(
    *,
    comparison_id: str,
    stratum: str,
    request: Mapping[str, Any],
    unit: Mapping[str, Any],
    row_source: str,
    column_source: str,
    row_request_id: str,
    column_request_id: str,
    row_decision: Mapping[str, Any],
    column_decision: Mapping[str, Any],
    metrics: Sequence[str],
) -> dict[str, Any]:
    row_boundary = sorted(set(row_decision["boundary_concerns"]))
    column_boundary = sorted(set(column_decision["boundary_concerns"]))
    agreements = {
        "tag_agreement": row_decision["tag"] == column_decision["tag"],
        "confidence_agreement": (
            row_decision["confidence"] == column_decision["confidence"]
        ),
        "boundary_agreement": row_boundary == column_boundary,
        "exact_full_decision_agreement": (
            _exact_decision(row_decision) == _exact_decision(column_decision)
        ),
    }
    return {
        "comparison_id": comparison_id,
        "stratum": stratum,
        "window_index": request["window_index"],
        "response_id": request["response_id"],
        "unit_id": unit["unit_id"],
        "unit_text": unit["text"],
        "row_source": row_source,
        "column_source": column_source,
        "row_request_id": row_request_id,
        "column_request_id": column_request_id,
        "row_decision": _exact_decision(row_decision),
        "column_decision": _exact_decision(column_decision),
        "declared_metrics": list(metrics),
        "agreements": agreements,
    }


def _build_pair_groups(inputs: Mapping[str, Any]) -> list[dict[str, Any]]:
    qualification = inputs["qualification"]
    requests = qualification["requests"]
    events = inputs["events"]
    units = {unit["unit_id"]: unit for unit in qualification["focal_units"]}
    request_by_id = {item["request_id"]: item for item in requests}
    event_by_id = {item["request_id"]: item for item in events}
    v1_events = qualification["v1_comparison_baseline"]["events"]
    v1_by_id = {item["request_id"]: item for item in v1_events}
    primary_by_arm_source = {
        (item["arm_id"], item["source_v1_request_id"]): item
        for item in requests
        if item["repeat_of_request_id"] is None
    }
    groups: list[dict[str, Any]] = []

    for arm in ARM_IDS:
        pairs = []
        for repeat in (
            item
            for item in requests
            if item["arm_id"] == arm and item["repeat_of_request_id"] is not None
        ):
            primary = request_by_id[repeat["repeat_of_request_id"]]
            row = _decision_map(event_by_id[repeat["request_id"]])
            column = _decision_map(event_by_id[primary["request_id"]])
            pairs.extend(
                _pair_record(
                    comparison_id="within_arm_repeat",
                    stratum=arm,
                    request=repeat,
                    unit=units[unit_id],
                    row_source=f"{arm}:repeat",
                    column_source=f"{arm}:primary",
                    row_request_id=repeat["request_id"],
                    column_request_id=primary["request_id"],
                    row_decision=row[unit_id],
                    column_decision=column[unit_id],
                    metrics=("tag_agreement", "exact_full_decision_agreement"),
                )
                for unit_id in repeat["focal_unit_ids"]
            )
        groups.append(
            {
                "comparison_id": "within_arm_repeat",
                "stratum": arm,
                "expected_unit_pairs": 24,
                "pairs": pairs,
            }
        )

    cross_pairs = []
    target_primaries = [
        item
        for item in requests
        if item["arm_id"] == ARM_TARGET_ONLY and item["repeat_of_request_id"] is None
    ]
    for target in target_primaries:
        full = primary_by_arm_source[(ARM_FULL_UNIT, target["source_v1_request_id"])]
        row = _decision_map(event_by_id[target["request_id"]])
        column = _decision_map(event_by_id[full["request_id"]])
        cross_pairs.extend(
            _pair_record(
                comparison_id="cross_arm_primary",
                stratum="all_primary_targets",
                request=target,
                unit=units[unit_id],
                row_source=f"{ARM_TARGET_ONLY}:primary",
                column_source=f"{ARM_FULL_UNIT}:primary",
                row_request_id=target["request_id"],
                column_request_id=full["request_id"],
                row_decision=row[unit_id],
                column_decision=column[unit_id],
                metrics=(
                    "tag_agreement",
                    "exact_full_decision_agreement",
                    "confidence_agreement",
                    "boundary_agreement",
                ),
            )
            for unit_id in target["focal_unit_ids"]
        )
    groups.append(
        {
            "comparison_id": "cross_arm_primary",
            "stratum": "all_primary_targets",
            "expected_unit_pairs": 72,
            "pairs": cross_pairs,
        }
    )

    for comparison_id, is_repeat, expected in (
        ("each_arm_vs_v1_primary", False, 72),
        ("each_arm_vs_v1_repeat", True, 24),
    ):
        for arm in ARM_IDS:
            pairs = []
            selected = [
                item
                for item in requests
                if item["arm_id"] == arm
                and (item["repeat_of_request_id"] is not None) is is_repeat
            ]
            for request in selected:
                row = _decision_map(event_by_id[request["request_id"]])
                v1 = v1_by_id[request["source_v1_request_id"]]
                column = _decision_map(v1)
                for unit_id in request["focal_unit_ids"]:
                    pairs.append(
                        _pair_record(
                            comparison_id=comparison_id,
                            stratum=arm,
                            request=request,
                            unit=units[unit_id],
                            row_source=f"{arm}:{'repeat' if is_repeat else 'primary'}",
                            column_source=f"v1:{'repeat' if is_repeat else 'primary'}",
                            row_request_id=request["request_id"],
                            column_request_id=v1["request_id"],
                            row_decision=row[unit_id],
                            column_decision=column[unit_id],
                            metrics=("tag_agreement", "exact_full_decision_agreement"),
                        )
                    )
            groups.append(
                {
                    "comparison_id": comparison_id,
                    "stratum": arm,
                    "expected_unit_pairs": expected,
                    "pairs": pairs,
                }
            )
    return groups


def _matrix(order: Sequence[str]) -> dict[str, dict[str, int]]:
    return {row: dict.fromkeys(order, 0) for row in order}


def _summarize_group(group: Mapping[str, Any]) -> dict[str, Any]:
    pairs = group["pairs"]
    expected = group["expected_unit_pairs"]
    if len(pairs) != expected:
        raise ValueError("coarse comparison pair coverage is incomplete")
    metrics = pairs[0]["declared_metrics"] if pairs else []
    tag_confusion = _matrix(COARSE_TAGS)
    confidence_confusion = _matrix(CONFIDENCE_VALUES)
    boundary_confusions = {
        concern: _matrix(("absent", "present")) for concern in BOUNDARY_CONCERNS
    }
    agreements = dict.fromkeys(metrics, 0)
    for pair in pairs:
        if pair["declared_metrics"] != metrics:
            raise ValueError("coarse comparison metric drift within group")
        row = pair["row_decision"]
        column = pair["column_decision"]
        tag_confusion[row["tag"]][column["tag"]] += 1
        confidence_confusion[row["confidence"]][column["confidence"]] += 1
        for concern in BOUNDARY_CONCERNS:
            row_state = "present" if concern in row["boundary_concerns"] else "absent"
            column_state = (
                "present" if concern in column["boundary_concerns"] else "absent"
            )
            boundary_confusions[concern][row_state][column_state] += 1
        for metric in metrics:
            agreements[metric] += int(pair["agreements"][metric])
    return {
        "comparison_id": group["comparison_id"],
        "stratum": group["stratum"],
        "row_side": pairs[0]["row_source"] if pairs else None,
        "column_side": pairs[0]["column_source"] if pairs else None,
        "counts": {
            "expected": expected,
            "observed": len(pairs),
            "eligible": len(pairs),
            "missing": 0,
            "invalid": 0,
        },
        "agreement": {
            metric: {
                "agree": agreements[metric],
                "disagree": expected - agreements[metric],
                "rate": agreements[metric] / expected,
            }
            for metric in metrics
        },
        "tag_confusion": {
            "row_order": list(COARSE_TAGS),
            "column_order": list(COARSE_TAGS),
            "counts": tag_confusion,
        },
        "confidence_confusion": (
            {
                "row_order": list(CONFIDENCE_VALUES),
                "column_order": list(CONFIDENCE_VALUES),
                "counts": confidence_confusion,
            }
            if group["comparison_id"] == "cross_arm_primary"
            else None
        ),
        "boundary_concern_confusions": (
            {
                concern: {
                    "row_order": ["absent", "present"],
                    "column_order": ["absent", "present"],
                    "counts": boundary_confusions[concern],
                }
                for concern in BOUNDARY_CONCERNS
            }
            if group["comparison_id"] == "cross_arm_primary"
            else None
        ),
    }


def _usage_summary(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    totals = dict.fromkeys(USAGE_FIELDS, 0)
    missing = dict.fromkeys(USAGE_FIELDS, 0)
    total_cost = 0.0
    cost_missing = 0
    for event in events:
        usage = event.get("usage")
        if not isinstance(usage, Mapping):
            raise ValueError("coarse comparison usage is missing")
        for field in USAGE_FIELDS:
            value = usage.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                missing[field] += 1
            else:
                totals[field] += value
        cost = event.get("cost")
        value = cost.get("total_cost") if isinstance(cost, Mapping) else None
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
            cost_missing += 1
        else:
            total_cost += float(value)
    if any(missing.values()) or cost_missing:
        raise ValueError("coarse comparison usage/cost missingness is nonzero")
    return {
        "request_count": len(events),
        "token_totals": totals,
        "token_missing_request_counts": missing,
        "total_cost_usd": total_cost,
        "cost_missing_request_count": cost_missing,
    }


def _v1_usage_summary(
    events: Sequence[Mapping[str, Any]], audit: Mapping[str, Any]
) -> dict[str, Any]:
    events_by_id = {event["request_id"]: event for event in events}
    corrected = {item["request_id"]: item for item in audit["requests"]}
    normalized = []
    for request_id, event in events_by_id.items():
        item = corrected.get(request_id)
        if not isinstance(item, Mapping):
            raise ValueError("coarse comparison corrected v1 receipt is missing")
        normalized.append(
            {
                "request_id": request_id,
                "repeat_of_request_id": event["repeat_of_request_id"],
                "usage": item.get("corrected_usage_from_raw_receipt"),
                "cost": item.get("corrected_cost"),
            }
        )
    result = {
        "all": _usage_summary(normalized),
        "primary": _usage_summary(
            [item for item in normalized if item["repeat_of_request_id"] is None]
        ),
        "repeat": _usage_summary(
            [item for item in normalized if item["repeat_of_request_id"] is not None]
        ),
        "accounting_source": "pinned v1 cache-write cost-correction audit",
        "audit_sha256": audit["cost_correction_audit_sha256"],
    }
    corrected_total = audit.get("corrected_total_cost_usd")
    if (
        not isinstance(corrected_total, (int, float))
        or isinstance(corrected_total, bool)
        or abs(result["all"]["total_cost_usd"] - float(corrected_total)) > 1e-12
    ):
        raise ValueError("coarse comparison corrected v1 total-cost drift")
    return result


def _review_rows(
    groups: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    unblinding = []
    for group in groups:
        agreements_added = 0
        for pair in group["pairs"]:
            disagrees = any(
                not pair["agreements"][metric] for metric in pair["declared_metrics"]
            )
            if not disagrees and agreements_added >= 2:
                continue
            reason = (
                "declared_metric_disagreement"
                if disagrees
                else "deterministic_exact_agreement_example"
            )
            if not disagrees:
                agreements_added += 1
            identity = {
                key: pair[key]
                for key in (
                    "comparison_id",
                    "stratum",
                    "window_index",
                    "unit_id",
                    "row_request_id",
                    "column_request_id",
                )
            }
            review_id = f"pwcoarsereview-{canonical_sha256(identity)[:32]}"
            swap = int(hashlib.sha256(review_id.encode()).hexdigest()[0], 16) % 2 == 1
            decision_a = pair["column_decision"] if swap else pair["row_decision"]
            decision_b = pair["row_decision"] if swap else pair["column_decision"]
            source_a = pair["column_source"] if swap else pair["row_source"]
            source_b = pair["row_source"] if swap else pair["column_source"]
            rows.append(
                {
                    "schema_version": EXAMPLE_SCHEMA,
                    "review_pair_id": review_id,
                    "selection_reason": reason,
                    "comparison_id": pair["comparison_id"],
                    "stratum": pair["stratum"],
                    "window_index": pair["window_index"],
                    "response_id": pair["response_id"],
                    "unit_id": pair["unit_id"],
                    "unit_text": pair["unit_text"],
                    "decision_a": decision_a,
                    "decision_b": decision_b,
                    "declared_metric_agreement": {
                        metric: pair["agreements"][metric]
                        for metric in pair["declared_metrics"]
                    },
                }
            )
            unblinding.append(
                {
                    "review_pair_id": review_id,
                    "decision_a_source": source_a,
                    "decision_b_source": source_b,
                }
            )
    return rows, unblinding


def build_comparison_report(
    inputs: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compute the predeclared descriptive report without correctness judgments."""

    qualification = inputs["qualification"]
    plan = qualification["config"]["comparison_plan"]
    groups = _build_pair_groups(inputs)
    summaries = [_summarize_group(group) for group in groups]
    v2_usage = {}
    requests_by_id = {
        request["request_id"]: request for request in qualification["requests"]
    }
    for arm in ARM_IDS:
        arm_events = [event for event in inputs["events"] if event["arm_id"] == arm]
        v2_usage[arm] = {
            "all": _usage_summary(arm_events),
            "primary": _usage_summary(
                [
                    event
                    for event in arm_events
                    if requests_by_id[event["request_id"]]["repeat_of_request_id"]
                    is None
                ]
            ),
            "repeat": _usage_summary(
                [
                    event
                    for event in arm_events
                    if requests_by_id[event["request_id"]]["repeat_of_request_id"]
                    is not None
                ]
            ),
        }
    combined_tokens = {
        field: sum(v2_usage[arm]["all"]["token_totals"][field] for arm in ARM_IDS)
        for field in USAGE_FIELDS
    }
    collection_usage = inputs["collection"].get("usage_totals")
    combined_cost = sum(v2_usage[arm]["all"]["total_cost_usd"] for arm in ARM_IDS)
    collection_cost = inputs["collection"].get("actual_total_cost_usd")
    if (
        not isinstance(collection_usage, Mapping)
        or any(
            collection_usage.get(field) != combined_tokens[field]
            for field in USAGE_FIELDS
        )
        or not isinstance(collection_cost, (int, float))
        or isinstance(collection_cost, bool)
        or abs(float(collection_cost) - combined_cost) > 1e-12
    ):
        raise ValueError("coarse comparison v2 usage/cost reconciliation drift")
    review_rows, unblinding = _review_rows(groups)
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": "complete_descriptive_comparison_no_threshold",
        "claim_boundary": (
            "This report describes protocol agreement only. A disagreement is not a "
            "correctness, faithfulness, or model-computation error, and v1-v2 changes "
            "cannot be causally assigned to one protocol difference."
        ),
        "comparison_plan": plan,
        "comparison_plan_sha256": canonical_sha256(plan),
        "source_bindings": {
            "qualification_manifest_sha256": qualification["manifest"][
                "manifest_sha256"
            ],
            "run_intent_sha256": inputs["run_intent"]["run_intent_sha256"],
            "collection_manifest_sha256": inputs["collection"][
                "collection_manifest_sha256"
            ],
            "events_jsonl_sha256": inputs["collection"]["events_jsonl_sha256"],
            "v1_run_manifest_sha256": qualification["v1_comparison_baseline"][
                "manifest"
            ]["run_manifest_sha256"],
            "v1_events_sha256": qualification["config"]["source"][
                "v1_completed_events_sha256"
            ],
            "v1_cost_correction_audit_sha256": qualification[
                "v1_cost_correction_audit"
            ]["cost_correction_audit_sha256"],
        },
        "comparison_summaries": summaries,
        "usage_cost": {
            "v2_by_arm": v2_usage,
            "v2_collection_reconciliation": {
                "token_totals": combined_tokens,
                "total_cost_usd": combined_cost,
                "matches_collection_manifest": True,
            },
            "v1_baseline": _v1_usage_summary(
                qualification["v1_comparison_baseline"]["events"],
                qualification["v1_cost_correction_audit"],
            ),
        },
        "review_examples": {
            "file": "examples-disagreements.jsonl",
            "selection_rule": (
                "all pairs disagreeing on any declared metric, plus the first two "
                "exact-agreement pairs in frozen order per comparison stratum"
            ),
            "row_count": len(review_rows),
            "presentation": (
                "decisions are deterministically assigned to anonymous A/B sides; use "
                "review_unblinding_bindings only after blind review"
            ),
        },
        "review_unblinding_bindings": unblinding,
        "formal_pass_threshold": None,
        "human_blind_review_status": "deferred",
    }
    report["report_sha256"] = canonical_sha256(report)
    return report, review_rows


def load_comparison_bundle(root: Path) -> dict[str, Any]:
    """Validate a completed immutable comparison report bundle."""

    manifest = _load_object(root / "manifest.json")
    _verify_self_hash(manifest, "manifest_sha256", "coarse comparison bundle manifest")
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA
        or manifest.get("status") != "complete_offline_immutable_comparison"
        or manifest.get("network_calls_made") != 0
    ):
        raise ValueError("coarse comparison bundle is not complete and offline")
    expected_files = {"comparison-report.json", "examples-disagreements.jsonl"}
    if {item.get("path") for item in manifest.get("files", [])} != expected_files:
        raise ValueError("coarse comparison bundle membership drift")
    for binding in manifest["files"]:
        path = root / binding["path"]
        if (
            not path.is_file()
            or path.stat().st_size != binding.get("bytes")
            or file_sha256(path) != binding.get("sha256")
        ):
            raise ValueError("coarse comparison bundle payload drift")
    report = _load_object(root / "comparison-report.json")
    _verify_self_hash(report, "report_sha256", "coarse comparison report")
    if (
        report.get("schema_version") != REPORT_SCHEMA
        or report["report_sha256"] != manifest.get("report_sha256")
        or report.get("comparison_plan_sha256")
        != manifest.get("comparison_plan_sha256")
    ):
        raise ValueError("coarse comparison report binding drift")
    examples = read_jsonl(root / "examples-disagreements.jsonl")
    ids = [item.get("review_pair_id") for item in examples]
    unblinding_ids = [
        item.get("review_pair_id")
        for item in report.get("review_unblinding_bindings", [])
    ]
    if (
        len(examples) != manifest.get("example_row_count")
        or len(examples) != report.get("review_examples", {}).get("row_count")
        or any(item.get("schema_version") != EXAMPLE_SCHEMA for item in examples)
        or len(set(ids)) != len(ids)
        or ids != unblinding_ids
    ):
        raise ValueError("coarse comparison review-example binding drift")
    return {"manifest": manifest, "report": report, "examples": examples}
