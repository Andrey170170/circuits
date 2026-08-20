"""Frozen post-campaign analysis for coarse trace-target sampling metadata."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from circuits.analysis.bonafide.canonical import canonical_sha256, file_sha256
from circuits.analysis.bonafide.coarse_sampling_annotation import validate_decisions
from circuits.analysis.bonafide.coarse_sampling_openai_batch_continuation_v1 import (
    CONTINUATION_SCHEMA,
    _all_attempts,
    _validate_collected_attempt,
    _validate_inherited_calibration_evidence,
)
from circuits.analysis.bonafide.coarse_sampling_production_v1 import (
    BROAD_PROJECTION,
    load_production_bundle,
)

ANALYSIS_SCHEMA = "adag.process-witness.coarse-post-campaign-analysis.v1"
REPORT_SCHEMA = "adag.process-witness.coarse-post-campaign-report.v1"
INVENTORY_SCHEMA = "adag.process-witness.coarse-post-campaign-inventory.v1"
ANALYSIS_STATUS = "frozen_sampling_metadata_not_truth"
CLAIM_BOUNDARY = (
    "This artifact contains graph-blind coarse sampling metadata and sensitivity "
    "diagnostics. It is not semantic truth, a trace selection, an ADAG adequacy "
    "result, a motif or witness result, or a faithfulness judgment."
)
ROUTES = (
    "provider_process",
    "provider_contextual",
    "provider_uncertain_or_incomplete",
    "deterministic_terminal",
    "deterministic_control",
)
EXPECTED_CENSUS = {
    "physical_requests": 37_671,
    "effective_success": 37_656,
    "residual_invalid_output": 15,
    "responses": 188,
    "units": 94_546,
    "openai_pending_units": 74_860,
    "deterministic_surface_units": 19_500,
    "deterministic_terminal_units": 186,
}
EXPECTED_HEADLINES = {
    "strict_proposals": {
        "provider_vote_coverage": {"0": 6, "1": 6, "2": 60, "3": 74_788},
        "broad_counts": {
            "contextual": 55_966,
            "process_bearing": 38_366,
            "unresolved": 142,
            "missing_proposal": 72,
        },
        "fine_agreement": {
            "unanimous": 59_812,
            "two_one": 14_300,
            "one_one_one": 676,
        },
        "broad_agreement": {
            "unanimous": 67_113,
            "two_one": 7_647,
            "one_one_one": 28,
        },
    },
    "conservative_exact_id_salvage": {
        "provider_vote_coverage": {"0": 1, "1": 1, "2": 10, "3": 74_848},
        "broad_counts": {
            "contextual": 55_994,
            "process_bearing": 38_398,
            "unresolved": 142,
            "missing_proposal": 12,
        },
        "fine_agreement": {
            "unanimous": 59_864,
            "two_one": 14_308,
            "one_one_one": 676,
        },
    },
}


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _verify_self_hash(value: dict[str, Any], field: str, label: str) -> None:
    payload = dict(value)
    observed = payload.pop(field, None)
    if observed != canonical_sha256(payload):
        raise ValueError(f"{label} self-hash drift")


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in rows
    )


def _hashed(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(value)
    result[field] = canonical_sha256(result)
    return result


def _effective_events(
    *,
    inherited_events: Sequence[Mapping[str, Any]],
    primary_events: Sequence[Sequence[Mapping[str, Any]]],
    recovery_events: Sequence[Mapping[str, Any]],
    request_ids: Sequence[str],
) -> list[dict[str, Any]]:
    effective: dict[str, dict[str, Any]] = {}
    for event in inherited_events:
        effective[str(event["request_id"])] = dict(event)
    for attempt in primary_events:
        for event in attempt:
            request_id = str(event["request_id"])
            if request_id in effective:
                raise ValueError("primary event duplicated request identity")
            effective[request_id] = dict(event)
    for event in recovery_events:
        request_id = str(event["request_id"])
        prior = effective.get(request_id)
        if prior is None or prior.get("validation_status") == "success":
            raise ValueError("recovery event does not replace an earlier failure")
        effective[request_id] = dict(event)
    if set(effective) != set(request_ids):
        raise ValueError("effective event request coverage drift")
    return [effective[request_id] for request_id in request_ids]


def _validated_sources(
    *, bundle_root: Path, run_root: Path
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    bundle = load_production_bundle(bundle_root, load_units=True, strict_topology=True)
    continuation = _load_object(run_root / "continuation-manifest.json")
    _verify_self_hash(
        continuation, "continuation_manifest_sha256", "continuation manifest"
    )
    if (
        continuation.get("schema_version") != CONTINUATION_SCHEMA
        or Path(str(continuation.get("bundle_root"))).resolve() != bundle_root.resolve()
        or continuation.get("bundle_manifest_sha256")
        != bundle["manifest"]["manifest_sha256"]
    ):
        raise ValueError("continuation source binding drift")
    inherited_collection, inherited_events = _validate_inherited_calibration_evidence(
        run_root=run_root,
        bundle=bundle,
        manifest=continuation,
        bundle_root=bundle_root,
    )
    attempts = _all_attempts(run_root, continuation)
    if len(attempts) != len(continuation["attempts"]) + 1:
        raise ValueError("continuation aggregate recovery is absent")
    primary_events: list[list[dict[str, Any]]] = []
    recovery_events: list[dict[str, Any]] | None = None
    collection_bindings = [
        {
            "attempt_id": "inherited-calibration-shard-005",
            "collection_sha256": inherited_collection["collection_sha256"],
            "events_sha256": inherited_collection["events_sha256"],
            "request_count": inherited_collection["request_count"],
            "success_count": inherited_collection["success_count"],
            "failure_count": inherited_collection["failure_count"],
        }
    ]
    total_cost = float(continuation["calibration_known_priced_cost_usd"])
    for binding in attempts:
        collection, events = _validate_collected_attempt(
            run_root=run_root,
            manifest=continuation,
            bundle=bundle,
            binding=binding,
            bundle_root=bundle_root,
        )
        collection_bindings.append(
            {
                "attempt_id": binding["attempt_id"],
                "generation": binding["generation"],
                "collection_sha256": collection["collection_sha256"],
                "events_sha256": collection["events_sha256"],
                "request_count": collection["request_count"],
                "success_count": collection["success_count"],
                "failure_count": collection["failure_count"],
            }
        )
        total_cost += float(collection["known_priced_cost_usd"])
        if binding["generation"] == "failed-only-recovery":
            if recovery_events is not None:
                raise ValueError("multiple recovery waves are not supported")
            recovery_events = events
        else:
            primary_events.append(events)
    if recovery_events is None:
        raise ValueError("failed-only recovery events are absent")
    request_ids = [str(row["request_id"]) for row in bundle["request_index"]]
    effective = _effective_events(
        inherited_events=inherited_events,
        primary_events=primary_events,
        recovery_events=recovery_events,
        request_ids=request_ids,
    )
    latest_cost = _load_object(
        sorted((run_root / "cost-status").glob("receipt-*.json"))[-1]
    )
    _verify_self_hash(latest_cost, "cost_status_sha256", "final cost status")
    if (
        not math.isclose(total_cost, 30.283011425, rel_tol=0, abs_tol=1e-12)
        or not math.isclose(
            float(latest_cost["known_actual_cost_usd"]),
            total_cost,
            rel_tol=0,
            abs_tol=1e-12,
        )
        or latest_cost.get("hard_stop_crossed_after_inflight_attempt") is not False
    ):
        raise ValueError("campaign cost reconciliation drift")
    validation = {
        "schema_version": "adag.process-witness.coarse-source-validation.v1",
        "bundle_manifest_sha256": bundle["manifest"]["manifest_sha256"],
        "continuation_manifest_sha256": continuation["continuation_manifest_sha256"],
        "collection_bindings": collection_bindings,
        "actual_total_cost_usd": total_cost,
        "cost_status_sha256": latest_cost["cost_status_sha256"],
        "strict_loaders": [
            "load_production_bundle(strict_topology=True)",
            "_validate_inherited_calibration_evidence",
            "_validate_collected_attempt(all primary and recovery attempts)",
        ],
    }
    return bundle, continuation, effective, validation


def _salvage_exact_ids(
    events: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    votes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    diagnostics = []
    for event in events:
        if event.get("validation_status") != "invalid_output":
            continue
        raw = json.loads(str(event.get("raw_text")))
        decisions = raw.get("decisions") if isinstance(raw, Mapping) else None
        if not isinstance(decisions, list):
            raise ValueError("residual output is not a decisions list")
        expected = [str(value) for value in event["focal_unit_ids"]]
        accepted = []
        rejected = []
        seen = set()
        for decision in decisions:
            unit_id = decision.get("unit_id") if isinstance(decision, Mapping) else None
            if (
                not isinstance(unit_id, str)
                or unit_id not in expected
                or unit_id in seen
            ):
                rejected.append(
                    dict(decision) if isinstance(decision, Mapping) else decision
                )
                continue
            validated = validate_decisions(
                {"decisions": [dict(decision)]}, focal_unit_ids=[unit_id]
            )[0]
            seen.add(unit_id)
            accepted.append(validated)
            votes[unit_id].append(
                {
                    "request_id": event["request_id"],
                    "replica_index": event["replica_index"],
                    "vote_origin": "conservative_exact_id_salvage",
                    **validated,
                }
            )
        missing = [unit_id for unit_id in expected if unit_id not in seen]
        unknown_ids = [
            str(row.get("unit_id"))
            for row in rejected
            if isinstance(row, Mapping) and isinstance(row.get("unit_id"), str)
        ]
        diagnostics.append(
            {
                "schema_version": "adag.process-witness.coarse-exact-cover-diagnostic.v1",
                "request_id": event["request_id"],
                "window_id": event["window_id"],
                "response_id": event["response_id"],
                "replica_index": event["replica_index"],
                "expected_decision_count": len(expected),
                "raw_decision_count": len(decisions),
                "accepted_exact_expected_unique_count": len(accepted),
                "missing_expected_unit_ids": missing,
                "unknown_or_malformed_unit_ids": unknown_ids,
                "rejected_decision_count": len(rejected),
                "unique_one_extra_one_missing_predicate": (
                    len(decisions) == len(expected)
                    and len(accepted) == len(expected) - 1
                    and len(missing) == 1
                    and len(unknown_ids) == 1
                    and len(set(unknown_ids)) == 1
                ),
                "mapping_or_promotion_performed": False,
                "raw_text_sha256": hashlib.sha256(
                    str(event.get("raw_text")).encode()
                ).hexdigest(),
            }
        )
    return votes, diagnostics


def _provider_votes(
    events: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    votes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if event.get("validation_status") != "success":
            continue
        decisions = event.get("decisions")
        if not isinstance(decisions, list):
            raise ValueError("successful event decisions absent")
        for decision in decisions:
            unit_id = str(decision["unit_id"])
            votes[unit_id].append(
                {
                    "request_id": event["request_id"],
                    "replica_index": event["replica_index"],
                    "vote_origin": "provider_schema_valid",
                    **decision,
                }
            )
    return votes


def _proposal(
    unit: Mapping[str, Any], votes: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if unit["assignment_route"] != "openai_pending":
        tag = str(unit["deterministic_tag"])
        fine_votes = [tag, tag, tag]
        physical_votes: list[dict[str, Any]] = []
        source = "deterministic_rule"
        coverage = 3
    else:
        ordered = sorted(votes, key=lambda row: int(row["replica_index"]))
        replicas = [int(row["replica_index"]) for row in ordered]
        if len(replicas) != len(set(replicas)) or any(
            value not in (0, 1, 2) for value in replicas
        ):
            raise ValueError("proposal replica identity drift")
        fine_votes = [str(row["tag"]) for row in ordered]
        physical_votes = [dict(row) for row in ordered]
        source = "openai_replica_votes"
        coverage = len(fine_votes)
    broad_votes = [BROAD_PROJECTION[tag] for tag in fine_votes]
    fine_counts = Counter(fine_votes)
    broad_counts = Counter(broad_votes)
    complete = coverage == 3
    if complete:
        best = broad_counts.most_common()
        broad_majority = (
            best[0][0] if len(best) == 1 or best[0][1] > best[1][1] else "unresolved"
        )
        agreement = (
            "unanimous"
            if len(fine_counts) == 1
            else "one_one_one"
            if len(fine_counts) == 3
            else "two_one"
        )
        broad_agreement = (
            "unanimous"
            if len(broad_counts) == 1
            else "one_one_one"
            if len(broad_counts) == 3
            else "two_one"
        )
    else:
        broad_majority = "missing_proposal"
        agreement = "incomplete"
        broad_agreement = "incomplete"
    identity = {
        "schema_version": "adag.process-witness.coarse-post-campaign-proposal.v1",
        "unit_id": unit["unit_id"],
        "source": source,
        "fine_votes": fine_votes,
        "replica_coverage": coverage,
    }
    return {
        **identity,
        "proposal_id": f"pwcoarsepostv1-{canonical_sha256(identity)[:32]}",
        "response_id": unit["response_id"],
        "sequence_index": unit["sequence_index"],
        "assignment_route": unit["assignment_route"],
        "proposal_status": "complete" if complete else "incomplete",
        "missing_replica_indices": sorted(
            {0, 1, 2} - {int(v["replica_index"]) for v in physical_votes}
        ),
        "fine_vote_histogram": dict(sorted(fine_counts.items())),
        "broad_votes": broad_votes,
        "broad_vote_histogram": dict(sorted(broad_counts.items())),
        "broad_majority": broad_majority,
        "fine_agreement_pattern": agreement,
        "broad_agreement_pattern": broad_agreement,
        "physical_votes": physical_votes,
        "fragment_of": unit.get("fragment_of"),
        "token_span": unit["token_span"],
        "core_character_span": unit["core_character_span"],
        "covering_character_span": unit["covering_character_span"],
    }


def _proposal_route(unit: Mapping[str, Any], proposal: Mapping[str, Any]) -> str:
    assignment = unit["assignment_route"]
    if assignment == "deterministic_surface":
        return "deterministic_control"
    if assignment == "deterministic_terminal_serialization":
        return "deterministic_terminal"
    if proposal["broad_majority"] == "process_bearing":
        return "provider_process"
    if proposal["broad_majority"] == "contextual":
        return "provider_contextual"
    return "provider_uncertain_or_incomplete"


def _validate_readonly_modes(root: Path) -> None:
    if root.stat().st_mode & 0o777 != 0o555:
        raise ValueError("analysis root mode drift")
    for path in root.rglob("*"):
        expected = 0o555 if path.is_dir() else 0o444
        if path.stat().st_mode & 0o777 != expected:
            raise ValueError(f"analysis mode drift: {path.relative_to(root)}")


def load_frozen_post_campaign_analysis(root: Path) -> dict[str, Any]:
    """Validate a frozen analysis using only evidence beneath ``root``."""

    _validate_readonly_modes(root)
    manifest = _load_object(root / "manifest.json")
    _verify_self_hash(manifest, "manifest_sha256", "analysis manifest")
    if (
        manifest.get("schema_version") != ANALYSIS_SCHEMA
        or manifest.get("status") != ANALYSIS_STATUS
    ):
        raise ValueError("analysis manifest semantic drift")
    inventory = _load_object(root / "evidence-inventory.json")
    _verify_self_hash(inventory, "inventory_sha256", "analysis inventory")
    if inventory.get("schema_version") != INVENTORY_SCHEMA or inventory.get(
        "inventory_sha256"
    ) != manifest.get("inventory_sha256"):
        raise ValueError("analysis inventory binding drift")
    files = inventory.get("files")
    if not isinstance(files, list):
        raise ValueError("analysis inventory files absent")
    expected = {str(row["path"]): row for row in files}
    if len(expected) != len(files):
        raise ValueError("analysis inventory path collision")
    excluded = {root / "manifest.json", root / "evidence-inventory.json"}
    observed = {
        str(path.relative_to(root)): path
        for path in root.rglob("*")
        if path.is_file() and path not in excluded
    }
    if set(observed) != set(expected):
        raise ValueError("analysis inventory coverage drift")
    for relative, path in observed.items():
        row = expected[relative]
        if path.is_symlink():
            raise ValueError(f"analysis artifact contains symlink: {relative}")
        if path.stat().st_size != row.get("bytes") or file_sha256(path) != row.get(
            "sha256"
        ):
            raise ValueError(f"analysis evidence file drift: {relative}")
    report = _load_object(root / "completion-report.json")
    if report.get("schema_version") != REPORT_SCHEMA or file_sha256(
        root / "completion-report.json"
    ) != manifest.get("completion_report_sha256"):
        raise ValueError("analysis completion report drift")
    if report.get("census") != EXPECTED_CENSUS or any(
        report.get(lane) != expected for lane, expected in EXPECTED_HEADLINES.items()
    ):
        raise ValueError("analysis literal census drift")
    return {"manifest": manifest, "completion_report": report}
