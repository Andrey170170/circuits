"""Post-collection human-sealed compatibility gate for coarse v4."""

from __future__ import annotations

import json
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from circuits.analysis.bonafide.canonical import canonical_sha256, file_sha256
from circuits.analysis.bonafide.coarse_sampling_annotation import (
    BOUNDARY_CONCERNS,
    COARSE_TAGS,
)
from circuits.analysis.bonafide.coarse_sampling_annotation_v4 import (
    load_v4_qualification,
)
from circuits.analysis.bonafide.coarse_sampling_review_v4 import (
    EXPORT_SCHEMA,
    load_review_packet,
)
from circuits.labeling.io import read_jsonl

REPORT_SCHEMA = "adag.process-witness.coarse-compatibility-report.v4"
PROCESS_BEARING = {
    "active_task_work",
    "evaluation_or_revision",
    "intermediate_commitment",
    "final_answer",
}
GATED_ROLES = {"repair", "unchanged_short"}


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _verify_self_hash(value: Mapping[str, Any], field: str, label: str) -> None:
    payload = dict(value)
    observed = payload.pop(field, None)
    if not isinstance(observed, str) or observed != canonical_sha256(payload):
        raise ValueError(f"{label} self-hash drift")


def load_completed_v4_inputs(
    *,
    qualification_root: Path,
    run_root: Path,
    review_root: Path,
    human_ledger_path: Path,
) -> dict[str, Any]:
    qualification = load_v4_qualification(qualification_root)
    review = load_review_packet(review_root)
    if (
        review["packet"]["qualification_manifest_sha256"]
        != qualification["manifest"]["manifest_sha256"]
    ):
        raise ValueError("coarse v4 review/qualification binding drift")
    human = read_jsonl(human_ledger_path)
    collection = _load_object(run_root / "collection-manifest.json")
    _verify_self_hash(
        collection, "collection_manifest_sha256", "coarse v4 collection manifest"
    )
    intent = _load_object(run_root / "run-intent.json")
    _verify_self_hash(intent, "run_intent_sha256", "coarse v4 run intent")
    if (
        collection.get("status") != "complete"
        or collection.get("qualification_decisions_ready") is not True
        or collection.get("request_count") != 45
        or collection.get("success_count") != 45
        or collection.get("failure_count") != 0
        or intent.get("qualification_manifest_sha256")
        != qualification["manifest"]["manifest_sha256"]
        or intent.get("run_intent_sha256") != collection.get("run_intent_sha256")
        or file_sha256(run_root / "events.jsonl")
        != collection.get("events_jsonl_sha256")
    ):
        raise ValueError("coarse v4 completed run binding drift")
    events = read_jsonl(run_root / "events.jsonl")
    expected_bindings = collection.get("event_bindings_in_order")
    if (
        len(events) != 45
        or [event.get("request_id") for event in events]
        != [request["request_id"] for request in qualification["requests"]]
        or any(
            event.get("event_sha256")
            != canonical_sha256(
                {key: value for key, value in event.items() if key != "event_sha256"}
            )
            for event in events
        )
        or expected_bindings
        != [
            {
                "request_id": event["request_id"],
                "event_sha256": event["event_sha256"],
            }
            for event in events
        ]
    ):
        raise ValueError("coarse v4 event identity or hash drift")
    return {
        "qualification": qualification,
        "review": review,
        "human": human,
        "collection": collection,
        "events": events,
        "human_ledger_sha256": file_sha256(human_ledger_path),
    }


def _validate_human(
    human: Sequence[Mapping[str, Any]], review: Mapping[str, Any]
) -> dict[str, Mapping[str, Any]]:
    packet = review["packet"]
    items = {item["unit_id"]: item for item in review["items"]}
    if len(human) != 24 or len(items) != 24:
        raise ValueError("coarse v4 human gate requires exactly 24 items")
    seals = {row.get("global_seal_id") for row in human}
    times = {row.get("global_sealed_at") for row in human}
    output = {}
    required = {
        "schema_version",
        "packet_id",
        "packet_binding_sha256",
        "item_id",
        "unit_id",
        "primary_label",
        "defensible_alternatives",
        "boundary_concerns",
        "note",
        "globally_sealed",
        "global_seal_id",
        "global_sealed_at",
    }
    for row in human:
        item = items.get(row.get("unit_id"))
        primary = row.get("primary_label")
        alternatives = row.get("defensible_alternatives")
        boundaries = row.get("boundary_concerns")
        if (
            set(row) != required
            or row.get("schema_version") != EXPORT_SCHEMA
            or row.get("packet_id") != packet["packet_id"]
            or row.get("packet_binding_sha256") != packet["packet_binding_sha256"]
            or item is None
            or row.get("item_id") != item["item_id"]
            or row.get("globally_sealed") is not True
            or primary not in COARSE_TAGS
            or not isinstance(alternatives, list)
            or len(set(alternatives)) != len(alternatives)
            or any(
                value not in COARSE_TAGS or value == primary for value in alternatives
            )
            or not isinstance(boundaries, list)
            or len(set(boundaries)) != len(boundaries)
            or any(value not in BOUNDARY_CONCERNS for value in boundaries)
            or not isinstance(row.get("note"), str)
        ):
            raise ValueError("coarse v4 human decision binding or value drift")
        output[row["unit_id"]] = row
    if len(output) != 24 or len(seals) != 1 or len(times) != 1:
        raise ValueError("coarse v4 human global seal drift")
    try:
        uuid.UUID(next(iter(seals)))
        datetime.fromisoformat(next(iter(times)).replace("Z", "+00:00"))
    except (TypeError, ValueError) as error:
        raise ValueError("coarse v4 human global seal is invalid") from error
    return output


def build_compatibility_report(inputs: Mapping[str, Any]) -> dict[str, Any]:
    qualification = inputs["qualification"]
    human = _validate_human(inputs["human"], inputs["review"])
    decisions_by_unit: dict[str, list[Mapping[str, Any]]] = {
        unit["unit_id"]: [] for unit in qualification["focal_units"]
    }
    for event in inputs["events"]:
        if event.get("validation_status") != "success":
            raise ValueError("coarse v4 gate received a failed provider event")
        for decision in event["decisions"]:
            decisions_by_unit[decision["unit_id"]].append(decision)
    roles = {
        unit_id: role
        for window in qualification["windows"]
        for unit_id, role in window["target_roles"].items()
    }
    rows = []
    for unit_id, votes in decisions_by_unit.items():
        if len(votes) != 3:
            raise ValueError("coarse v4 gate requires three physical votes per target")
        histogram = Counter(vote["tag"] for vote in votes)
        majority = next((tag for tag, count in histogram.items() if count >= 2), None)
        pairwise = sum(
            votes[left]["tag"] == votes[right]["tag"]
            for left, right in ((0, 1), (0, 2), (1, 2))
        )
        majority_boundaries = sorted(
            concern
            for concern in BOUNDARY_CONCERNS
            if sum(concern in vote["boundary_concerns"] for vote in votes) >= 2
        )
        reference = human[unit_id]
        admissible = {reference["primary_label"], *reference["defensible_alternatives"]}
        rows.append(
            {
                "unit_id": unit_id,
                "role": roles[unit_id],
                "vote_histogram": dict(sorted(histogram.items())),
                "majority_label": majority,
                "one_one_one_disputed": majority is None,
                "pairwise_tag_agreements": pairwise,
                "majority_boundary_concerns": majority_boundaries,
                "human_primary_label": reference["primary_label"],
                "human_defensible_alternatives": reference["defensible_alternatives"],
                "human_boundary_concerns": reference["boundary_concerns"],
                "human_admissible_agreement": majority in admissible,
                "process_bearing_false_negative": bool(
                    admissible & PROCESS_BEARING and majority not in PROCESS_BEARING
                ),
            }
        )
    rows.sort(key=lambda row: row["unit_id"])
    gated = [row for row in rows if row["role"] in GATED_ROLES]
    diagnostics = [row for row in rows if row["role"] == "long_diagnostic"]
    mean_pairwise = sum(row["pairwise_tag_agreements"] for row in rows) / (24 * 3)
    gate = qualification["config"]["compatibility_gate"]
    human_merge_or_split = sum(
        bool(
            {"merge_previous", "merge_next", "split_needed"}
            & set(row["human_boundary_concerns"])
        )
        for row in gated
    )
    checks = {
        "all_24_targets_receive_three_valid_votes": all(
            len(decisions_by_unit[row["unit_id"]]) == 3 for row in rows
        ),
        "no_one_one_one_vote_patterns": not any(
            row["one_one_one_disputed"] for row in rows
        ),
        "minimum_mean_pairwise_tag_agreement": mean_pairwise
        >= gate["minimum_mean_pairwise_tag_agreement"],
        "maximum_merge_or_split_flags_on_20_gated_units": human_merge_or_split
        <= gate["maximum_merge_or_split_flags_on_20_gated_units"],
        "minimum_human_admissible_agreement_on_20_gated_units": sum(
            row["human_admissible_agreement"] for row in gated
        )
        >= gate["minimum_human_admissible_agreement_on_20_gated_units"],
        "maximum_process_bearing_false_negatives": sum(
            row["process_bearing_false_negative"] for row in gated
        )
        <= gate["maximum_process_bearing_false_negatives"],
        "long_diagnostic_units_excluded_from_pass_gate": len(diagnostics)
        == gate["long_diagnostic_units_excluded_from_pass_gate"],
        "human_review_required_before_pass": gate["human_review_required_before_pass"]
        is True,
    }
    if set(checks) != set(gate):
        raise ValueError("coarse v4 executable gate/config key drift")
    if len(rows) != 24 or len(gated) != 20 or len(diagnostics) != 4:
        raise ValueError("coarse v4 gate role census drift")
    return {
        "schema_version": REPORT_SCHEMA,
        "status": "passed" if all(checks.values()) else "failed_closed",
        "qualification_manifest_sha256": qualification["manifest"]["manifest_sha256"],
        "collection_manifest_sha256": inputs["collection"][
            "collection_manifest_sha256"
        ],
        "review_packet_manifest_sha256": inputs["review"]["manifest"][
            "manifest_sha256"
        ],
        "human_ledger_sha256": inputs["human_ledger_sha256"],
        "metrics": {
            "targets": 24,
            "gated_targets": 20,
            "long_diagnostic_targets": 4,
            "mean_pairwise_tag_agreement": mean_pairwise,
            "one_one_one_disputed": sum(row["one_one_one_disputed"] for row in rows),
            "gated_human_merge_or_split_flags": human_merge_or_split,
            "gated_model_merge_or_split_majorities": sum(
                bool(
                    {"merge_previous", "merge_next", "split_needed"}
                    & set(row["majority_boundary_concerns"])
                )
                for row in gated
            ),
            "gated_human_admissible_agreement": sum(
                row["human_admissible_agreement"] for row in gated
            ),
            "gated_process_bearing_false_negatives": sum(
                row["process_bearing_false_negative"] for row in gated
            ),
        },
        "gate_checks": checks,
        "long_diagnostic_interpretation": (
            "The four first/last residual chunks test the unavoidable 96-token cap. "
            "Their boundary concerns are reported but excluded from the 20-unit pass gate."
        ),
        "rows": rows,
    }
