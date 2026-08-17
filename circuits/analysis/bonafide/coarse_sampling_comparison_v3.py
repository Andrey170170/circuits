"""Vote stability, matched-arm comparison, and frozen human decision gate."""

from __future__ import annotations

import itertools
import json
import math
import random
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from circuits.analysis.bonafide.canonical import canonical_sha256, file_sha256
from circuits.analysis.bonafide.coarse_sampling_annotation import COARSE_TAGS
from circuits.analysis.bonafide.coarse_sampling_annotation_v3 import (
    ARM_FEW_SHOT,
    ARM_IDS,
    ARM_ZERO_SHOT,
    load_v3_qualification,
)
from circuits.analysis.bonafide.coarse_sampling_review_v3 import EXPORT_SCHEMA
from circuits.labeling.io import read_jsonl

REPORT_SCHEMA = "adag.process-witness.coarse-comparison-report.v3"
PROCESS_BEARING = {
    "active_task_work",
    "evaluation_or_revision",
    "intermediate_commitment",
    "final_answer",
}


def _selection_family(label: str | None) -> str:
    if label in PROCESS_BEARING:
        return "process_bearing"
    if label == "other_semantic_text":
        return "semantic_context"
    if label == "surface_or_control":
        return "surface_control"
    return "unresolved"


def _stability_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    patterns = Counter(row["vote_pattern"] for row in rows)
    return {
        "units": len(rows),
        "three_zero_stable": patterns["three_zero_stable"],
        "two_one_mixed": patterns["two_one_mixed"],
        "one_one_one_disputed": patterns["one_one_one_disputed"],
        "mean_pairwise_tag_agreement": (
            sum(row["pairwise_tag_agreements"] for row in rows) / (len(rows) * 3)
            if rows
            else None
        ),
        "mean_vote_entropy_bits": (
            sum(row["vote_entropy_bits"] for row in rows) / len(rows) if rows else None
        ),
    }


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


def load_completed_v3_inputs(
    *, qualification_root: Path, run_root: Path
) -> dict[str, Any]:
    qualification = load_v3_qualification(qualification_root)
    intent = _load_object(run_root / "run-intent.json")
    _verify_self_hash(intent, "run_intent_sha256", "coarse v3 run intent")
    collection = _load_object(run_root / "collection-manifest.json")
    _verify_self_hash(collection, "collection_manifest_sha256", "coarse v3 collection")
    if (
        intent.get("qualification_manifest_sha256")
        != qualification["manifest"]["manifest_sha256"]
        or collection.get("schema_version")
        != "adag.process-witness.coarse-openai-batch-collection.v3"
        or collection.get("status") != "complete"
        or collection.get("request_count") != 144
        or collection.get("success_count") != 144
        or collection.get("failure_count") != 0
        or collection.get("unique_arm_target_coverage") != 288
        or collection.get("exact_target_coverage") is not True
        or collection.get("qualification_decisions_ready") is not True
        or collection.get("distinct_provider_response_identity") is not True
        or collection.get("unique_provider_request_ids") != 144
        or collection.get("cost_complete") is not True
        or collection.get("authorization_exceeded") is not False
        or collection.get("run_intent_sha256") != intent["run_intent_sha256"]
    ):
        raise ValueError("coarse v3 completed run is not exact and usable")
    events_path = run_root / "events.jsonl"
    if file_sha256(events_path) != collection.get("events_jsonl_sha256"):
        raise ValueError("coarse v3 events hash drift")
    events = read_jsonl(events_path)
    requests = qualification["requests"]
    if len(events) != 144:
        raise ValueError("coarse v3 event cardinality drift")
    provider_request_ids = []
    for event, request in zip(events, requests, strict=True):
        payload = dict(event)
        event_hash = payload.pop("event_sha256", None)
        if (
            event_hash != canonical_sha256(payload)
            or event.get("request_id") != request["request_id"]
            or event.get("validation_status") != "success"
            or event.get("arm_id") != request["arm_id"]
            or event.get("replica_index") != request["replica_index"]
            or [decision.get("unit_id") for decision in event.get("decisions", [])]
            != request["focal_unit_ids"]
        ):
            raise ValueError("coarse v3 event/request binding drift")
        provider_request_ids.append(event.get("provider_request_id"))
    if (
        any(not isinstance(value, str) or not value for value in provider_request_ids)
        or len(set(provider_request_ids)) != 144
    ):
        raise ValueError("coarse v3 event provider-response identity drift")
    return {
        "qualification": qualification,
        "intent": intent,
        "collection": collection,
        "events": events,
    }


def _vote_pattern(histogram: Mapping[str, int]) -> tuple[str, str | None]:
    counts = sorted(histogram.values(), reverse=True)
    if counts == [3]:
        return "three_zero_stable", next(iter(histogram))
    if counts == [2, 1]:
        return "two_one_mixed", max(histogram, key=histogram.get)  # type: ignore[arg-type]
    if counts == [1, 1, 1]:
        return "one_one_one_disputed", None
    raise ValueError("coarse v3 vote histogram is not three-replica exact")


def build_comparison(inputs: Mapping[str, Any]) -> dict[str, Any]:
    qualification = inputs["qualification"]
    events = inputs["events"]
    windows = {window["window_index"]: window for window in qualification["windows"]}
    units = {unit["unit_id"]: unit for unit in qualification["focal_units"]}
    decisions: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for event in events:
        for decision in event["decisions"]:
            decisions.setdefault((event["arm_id"], decision["unit_id"]), []).append(
                {
                    **decision,
                    "request_id": event["request_id"],
                    "replica_index": event["replica_index"],
                }
            )
    request_lookup = {
        (request["arm_id"], unit_id): request
        for request in qualification["requests"]
        if request["replica_index"] == 0
        for unit_id in request["focal_unit_ids"]
    }
    vote_rows: list[dict[str, Any]] = []
    for arm_id in ARM_IDS:
        for unit_id in units:
            votes = sorted(
                decisions.get((arm_id, unit_id), []),
                key=lambda item: item["replica_index"],
            )
            if len(votes) != 3 or [item["replica_index"] for item in votes] != [
                0,
                1,
                2,
            ]:
                raise ValueError("coarse v3 unit does not have three exact votes")
            histogram = Counter(item["tag"] for item in votes)
            pattern, majority = _vote_pattern(histogram)
            request = request_lookup[(arm_id, unit_id)]
            window = windows[request["window_index"]]
            pairwise = [
                left["tag"] == right["tag"]
                for left, right in itertools.combinations(votes, 2)
            ]
            vote_rows.append(
                {
                    "arm_id": arm_id,
                    "window_index": request["window_index"],
                    "response_id": request["response_id"],
                    "unit_id": unit_id,
                    "unit_text": units[unit_id]["text"],
                    "hidden_sampling_stratum": {
                        "source_type": window["source_type_stratum"],
                        "position": window["position_stratum"],
                        "v9_hint": window["v9_hint_stratum_hidden_from_provider"],
                    },
                    "votes": votes,
                    "tag_vote_histogram": {
                        tag: histogram[tag] for tag in COARSE_TAGS if histogram[tag]
                    },
                    "vote_pattern": pattern,
                    "majority_label": majority,
                    "vote_entropy_bits": -sum(
                        (count / 3) * math.log2(count / 3)
                        for count in histogram.values()
                    ),
                    "pairwise_tag_agreements": sum(pairwise),
                    "pairwise_tag_agreement_by_replica_pair": {
                        "0-1": pairwise[0],
                        "0-2": pairwise[1],
                        "1-2": pairwise[2],
                    },
                    "majority_boundary_concerns": [
                        concern
                        for concern in (
                            "split_needed",
                            "merge_previous",
                            "merge_next",
                            "meaning_unclear",
                        )
                        if sum(concern in vote["boundary_concerns"] for vote in votes)
                        >= 2
                    ],
                    "stable_high_confidence": pattern == "three_zero_stable"
                    and all(vote["confidence"] == "high" for vote in votes),
                    "any_boundary_concern": any(
                        vote["boundary_concerns"] for vote in votes
                    ),
                }
            )

    arm_summaries = {}
    for arm_id in ARM_IDS:
        rows = [row for row in vote_rows if row["arm_id"] == arm_id]
        patterns = Counter(row["vote_pattern"] for row in rows)
        arm_events = [event for event in events if event["arm_id"] == arm_id]
        arm_summaries[arm_id] = {
            "units": len(rows),
            "physical_requests": len(arm_events),
            "three_zero_stable_count": patterns["three_zero_stable"],
            "two_one_mixed_count": patterns["two_one_mixed"],
            "one_one_one_disputed_count": patterns["one_one_one_disputed"],
            "majority_label_coverage": sum(
                row["majority_label"] is not None for row in rows
            ),
            "pairwise_tag_agreements": sum(
                row["pairwise_tag_agreements"] for row in rows
            ),
            "pairwise_tag_comparisons": len(rows) * 3,
            "mean_pairwise_tag_agreement": sum(
                row["pairwise_tag_agreements"] for row in rows
            )
            / (len(rows) * 3),
            "pairwise_tag_agreement_by_replica_pair": {
                pair: sum(
                    row["pairwise_tag_agreement_by_replica_pair"][pair] for row in rows
                )
                / len(rows)
                for pair in ("0-1", "0-2", "1-2")
            },
            "mean_vote_entropy_bits": sum(row["vote_entropy_bits"] for row in rows)
            / len(rows),
            "uncertain_vote_count": sum(
                vote["tag"] == "uncertain" for row in rows for vote in row["votes"]
            ),
            "units_with_any_boundary_concern": sum(
                row["any_boundary_concern"] for row in rows
            ),
            "usage": {
                field: sum(int(event["usage"].get(field) or 0) for event in arm_events)
                for field in (
                    "input_tokens",
                    "uncached_input_tokens",
                    "cache_read_tokens",
                    "cache_write_tokens",
                    "output_tokens",
                    "reasoning_tokens",
                )
            },
            "actual_cost_usd": sum(
                float(event["cost"]["total_cost"]) for event in arm_events
            ),
            "receipt_usage_and_cost_by_replica": {
                str(replica): {
                    "physical_requests": len(
                        [
                            event
                            for event in arm_events
                            if event["replica_index"] == replica
                        ]
                    ),
                    "usage": {
                        field: sum(
                            int(event["usage"].get(field) or 0)
                            for event in arm_events
                            if event["replica_index"] == replica
                        )
                        for field in (
                            "input_tokens",
                            "uncached_input_tokens",
                            "cache_read_tokens",
                            "cache_write_tokens",
                            "output_tokens",
                            "reasoning_tokens",
                        )
                    },
                    "actual_cost_usd": sum(
                        float(event["cost"]["total_cost"])
                        for event in arm_events
                        if event["replica_index"] == replica
                    ),
                }
                for replica in range(3)
            },
            "stability_by_hidden_factorial_cell": {
                key: _stability_summary(
                    [
                        row
                        for row in rows
                        if "|".join(row["hidden_sampling_stratum"].values()) == key
                    ]
                )
                for key in sorted(
                    {"|".join(row["hidden_sampling_stratum"].values()) for row in rows}
                )
            },
            "stability_by_majority_selection_family": {
                family: _stability_summary(
                    [
                        row
                        for row in rows
                        if _selection_family(row["majority_label"]) == family
                    ]
                )
                for family in (
                    "process_bearing",
                    "semantic_context",
                    "surface_control",
                    "unresolved",
                )
            },
        }

    rows_by_key = {(row["arm_id"], row["unit_id"]): row for row in vote_rows}
    cross = []
    stability_transition: dict[str, Counter[str]] = {}
    for unit_id in units:
        zero = rows_by_key[(ARM_ZERO_SHOT, unit_id)]
        few = rows_by_key[(ARM_FEW_SHOT, unit_id)]
        eligible = (
            zero["majority_label"] is not None and few["majority_label"] is not None
        )
        cross.append(
            {
                "unit_id": unit_id,
                "zero_shot_majority": zero["majority_label"],
                "few_shot_majority": few["majority_label"],
                "eligible": eligible,
                "agree": eligible and zero["majority_label"] == few["majority_label"],
            }
        )
        stability_transition.setdefault(zero["vote_pattern"], Counter())[
            few["vote_pattern"]
        ] += 1
    return {
        "schema_version": REPORT_SCHEMA,
        "status": "automated_vote_comparison_complete_human_gate_pending",
        "claim_boundary": (
            "Vote stability and cross-arm agreement are not semantic accuracy. "
            "No majority vote is ground truth and 1-1-1 remains disputed."
        ),
        "qualification_manifest_sha256": qualification["manifest"]["manifest_sha256"],
        "comparison_plan": qualification["config"]["comparison_plan"],
        "comparison_plan_sha256": canonical_sha256(
            qualification["config"]["comparison_plan"]
        ),
        "collection_manifest_sha256": inputs["collection"][
            "collection_manifest_sha256"
        ],
        "arm_summaries": arm_summaries,
        "cross_arm_majority": {
            "unit_count": len(cross),
            "eligible": sum(row["eligible"] for row in cross),
            "agreement": sum(row["agree"] for row in cross),
            "cross_arm_majority_tag_agreement": (
                sum(row["agree"] for row in cross)
                / sum(row["eligible"] for row in cross)
                if sum(row["eligible"] for row in cross)
                else None
            ),
            "disputed_or_missing": sum(not row["eligible"] for row in cross),
            "stability_transition_matrix": {
                row: dict(columns) for row, columns in stability_transition.items()
            },
        },
        "vote_rows": vote_rows,
        "human_gate": None,
    }


def apply_human_gate(
    report: Mapping[str, Any],
    human_decisions: Sequence[Mapping[str, Any]],
    review_packet: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the predeclared gate after one globally sealed 144-unit review."""

    packet = review_packet["packet"]
    packet_items = review_packet["items"]
    if (
        packet.get("qualification_manifest_sha256")
        != report.get("qualification_manifest_sha256")
        or len(packet_items) != 144
    ):
        raise ValueError("human gate review packet/qualification binding drift")
    if len(human_decisions) != 144:
        raise ValueError("human gate requires all 144 globally sealed decisions")
    expected_by_unit = {item["unit_id"]: item for item in packet_items}
    seal_ids = {item.get("global_seal_id") for item in human_decisions}
    seal_times = {item.get("global_sealed_at") for item in human_decisions}
    required_fields = {
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
    for item in human_decisions:
        expected = expected_by_unit.get(item.get("unit_id"))
        if (
            set(item) != required_fields
            or item.get("schema_version") != EXPORT_SCHEMA
            or item.get("packet_id") != packet["packet_id"]
            or item.get("packet_binding_sha256") != packet["packet_binding_sha256"]
            or expected is None
            or item.get("item_id") != expected["item_id"]
            or item.get("globally_sealed") is not True
        ):
            raise ValueError("human decision is not bound to the exact sealed packet")
    if (
        len(seal_ids) != 1
        or len(seal_times) != 1
        or not isinstance(next(iter(seal_ids)), str)
        or not next(iter(seal_ids))
        or not isinstance(next(iter(seal_times)), str)
        or not next(iter(seal_times))
    ):
        raise ValueError("human decisions do not share one global seal event")
    try:
        uuid.UUID(next(iter(seal_ids)))
        datetime.fromisoformat(next(iter(seal_times)).replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("human decisions have an invalid global seal event") from error
    human = {item["unit_id"]: item for item in human_decisions}
    if len(human) != 144:
        raise ValueError("human gate unit identity collision")
    rows = {(row["arm_id"], row["unit_id"]): row for row in report["vote_rows"]}
    metrics: dict[str, dict[str, Any]] = {}
    for arm_id in ARM_IDS:
        agreement = primary_agreement = process_false_negative = stable_high_error = 0
        human_process_count = 0
        confusion = {tag: dict.fromkeys(COARSE_TAGS, 0) for tag in COARSE_TAGS}
        boundary_tp = boundary_fp = boundary_fn = 0
        boundary_by_concern = {
            concern: {"true_positive": 0, "false_positive": 0, "false_negative": 0}
            for concern in (
                "split_needed",
                "merge_previous",
                "merge_next",
                "meaning_unclear",
            )
        }
        any_boundary_tp = any_boundary_fp = any_boundary_fn = 0
        clear_stable = clear_total = ambiguous_stable = ambiguous_total = 0
        clear_uncertain = clear_disputed = 0
        ambiguous_uncertain = ambiguous_disputed = 0
        for unit_id, item in human.items():
            primary = item.get("primary_label")
            alternatives = item.get("defensible_alternatives", [])
            if (
                primary not in COARSE_TAGS
                or not isinstance(alternatives, list)
                or len(set(alternatives)) != len(alternatives)
                or any(
                    value not in COARSE_TAGS or value == primary
                    for value in alternatives
                )
                or not isinstance(item.get("boundary_concerns"), list)
                or len(set(item["boundary_concerns"])) != len(item["boundary_concerns"])
                or any(
                    value
                    not in (
                        "split_needed",
                        "merge_previous",
                        "merge_next",
                        "meaning_unclear",
                    )
                    for value in item["boundary_concerns"]
                )
                or not isinstance(item.get("note"), str)
            ):
                raise ValueError("invalid human primary or defensible alternative")
            admissible = {primary, *alternatives}
            row = rows[(arm_id, unit_id)]
            majority = row["majority_label"]
            correct = majority in admissible if majority is not None else False
            agreement += int(correct)
            primary_agreement += int(majority == primary)
            if majority is not None:
                confusion[primary][majority] += 1
            human_process = bool(admissible & PROCESS_BEARING)
            model_process = majority in PROCESS_BEARING
            human_process_count += int(human_process)
            process_false_negative += int(human_process and not model_process)
            stable_high_error += int(row["stable_high_confidence"] and not correct)
            ambiguous = bool(alternatives)
            stable = row.get("vote_pattern") == "three_zero_stable"
            if ambiguous:
                ambiguous_total += 1
                ambiguous_stable += int(stable)
                ambiguous_uncertain += int(majority == "uncertain")
                ambiguous_disputed += int(
                    row.get("vote_pattern") == "one_one_one_disputed"
                )
            else:
                clear_total += 1
                clear_stable += int(stable)
                clear_uncertain += int(majority == "uncertain")
                clear_disputed += int(row.get("vote_pattern") == "one_one_one_disputed")
            human_boundary = set(item.get("boundary_concerns", []))
            model_boundary = set(row.get("majority_boundary_concerns", []))
            boundary_tp += len(human_boundary & model_boundary)
            boundary_fp += len(model_boundary - human_boundary)
            boundary_fn += len(human_boundary - model_boundary)
            any_boundary_tp += int(bool(human_boundary) and bool(model_boundary))
            any_boundary_fp += int(not human_boundary and bool(model_boundary))
            any_boundary_fn += int(bool(human_boundary) and not model_boundary)
            for concern, counts in boundary_by_concern.items():
                counts["true_positive"] += int(
                    concern in human_boundary and concern in model_boundary
                )
                counts["false_positive"] += int(
                    concern not in human_boundary and concern in model_boundary
                )
                counts["false_negative"] += int(
                    concern in human_boundary and concern not in model_boundary
                )
        for counts in boundary_by_concern.values():
            denominator = counts["true_positive"] + counts["false_negative"]
            counts["recall"] = (
                counts["true_positive"] / denominator if denominator else None
            )
        metrics[arm_id] = {
            "admissible_agreement": agreement,
            "primary_tag_agreement": primary_agreement,
            "process_bearing_false_negatives": process_false_negative,
            "process_family_recall": (
                (human_process_count - process_false_negative) / human_process_count
                if human_process_count
                else None
            ),
            "stable_high_confidence_errors": stable_high_error,
            "human_primary_tag_support": {
                tag: sum(item["primary_label"] == tag for item in human.values())
                for tag in COARSE_TAGS
            },
            "primary_tag_confusion": confusion,
            "stability_by_human_ambiguity": {
                "clear": {
                    "stable": clear_stable,
                    "uncertain_majority": clear_uncertain,
                    "disputed": clear_disputed,
                    "total": clear_total,
                },
                "ambiguous": {
                    "stable": ambiguous_stable,
                    "uncertain_majority": ambiguous_uncertain,
                    "disputed": ambiguous_disputed,
                    "total": ambiguous_total,
                },
            },
            "boundary_concerns": {
                "true_positive": boundary_tp,
                "false_positive": boundary_fp,
                "false_negative": boundary_fn,
                "recall": (
                    boundary_tp / (boundary_tp + boundary_fn)
                    if boundary_tp + boundary_fn
                    else None
                ),
                "by_concern": boundary_by_concern,
                "any_concern": {
                    "true_positive": any_boundary_tp,
                    "false_positive": any_boundary_fp,
                    "false_negative": any_boundary_fn,
                    "recall": (
                        any_boundary_tp / (any_boundary_tp + any_boundary_fn)
                        if any_boundary_tp + any_boundary_fn
                        else None
                    ),
                },
            },
        }
    paired_wins = paired_losses = 0
    paired_outcomes = Counter()
    block_differences: dict[str, list[int]] = {}
    for unit_id, item in human.items():
        admissible = {item["primary_label"], *item.get("defensible_alternatives", [])}
        zero = rows[(ARM_ZERO_SHOT, unit_id)]["majority_label"] in admissible
        few = rows[(ARM_FEW_SHOT, unit_id)]["majority_label"] in admissible
        response_id = rows[(ARM_ZERO_SHOT, unit_id)].get("response_id")
        if not isinstance(response_id, str):
            raise ValueError("human gate vote row lacks response block identity")
        block_differences.setdefault(response_id, []).append(int(few) - int(zero))
        paired_wins += int(few and not zero)
        paired_losses += int(zero and not few)
        paired_outcomes[
            (
                "both_correct"
                if zero and few
                else ("zero_only" if zero else "few_shot_only" if few else "neither")
            )
        ] += 1
    net_wins = paired_wins - paired_losses
    discordant = paired_wins + paired_losses
    mcnemar_p = 1.0
    if discordant:
        lower = min(paired_wins, paired_losses)
        mcnemar_p = min(
            1.0,
            2
            * sum(
                math.comb(discordant, index) * (0.5**discordant)
                for index in range(lower + 1)
            ),
        )
    if len(block_differences) != 24 or any(
        len(values) != 6 for values in block_differences.values()
    ):
        raise ValueError("human gate does not cover 24 complete response blocks")
    block_ids = sorted(block_differences)
    rng = random.Random(2026081703)
    bootstrap = []
    for _ in range(10_000):
        sampled = [rng.choice(block_ids) for _ in block_ids]
        differences = [
            value for block_id in sampled for value in block_differences[block_id]
        ]
        bootstrap.append(sum(differences) / len(differences))
    bootstrap.sort()
    zero_fn = metrics[ARM_ZERO_SHOT]["process_bearing_false_negatives"]
    few_fn = metrics[ARM_FEW_SHOT]["process_bearing_false_negatives"]
    rejected = {
        ARM_ZERO_SHOT: False,
        ARM_FEW_SHOT: few_fn > zero_fn + 2,
    }
    no_stable_error_increase = (
        metrics[ARM_FEW_SHOT]["stable_high_confidence_errors"]
        <= metrics[ARM_ZERO_SHOT]["stable_high_confidence_errors"]
    )
    few_improved = (
        not rejected[ARM_FEW_SHOT] and net_wins >= 5 and no_stable_error_increase
    )
    if few_improved:
        selected, rationale = ARM_FEW_SHOT, "few-shot met every improvement gate"
    else:
        selected, rationale = ARM_ZERO_SHOT, "tie or failed improvement gate; parsimony"
    gate = {
        "status": "human_gate_complete",
        "arm_metrics": metrics,
        "paired_few_shot_wins": paired_wins,
        "paired_few_shot_losses": paired_losses,
        "net_paired_few_shot_wins": net_wins,
        "paired_outcomes": dict(paired_outcomes),
        "exact_mcnemar_two_sided_p": mcnemar_p,
        "prompt_blocked_bootstrap_accuracy_difference": {
            "seed": 2026081703,
            "replicates": 10_000,
            "point_estimate": net_wins / 144,
            "percentile_95_interval": [bootstrap[249], bootstrap[9749]],
        },
        "arm_rejected_for_additional_process_false_negatives": rejected,
        "few_shot_no_increase_in_stable_high_confidence_errors": no_stable_error_increase,
        "few_shot_improved": few_improved,
        "selected_arm": selected,
        "selection_rationale": rationale,
    }
    result = dict(report)
    result["status"] = "comparison_and_human_gate_complete"
    result["human_gate"] = gate
    return result
