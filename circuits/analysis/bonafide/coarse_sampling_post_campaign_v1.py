"""Frozen post-campaign analysis for coarse trace-target sampling metadata."""

from __future__ import annotations

import ctypes
import errno
import gc
import hashlib
import json
import math
import os
import shutil
import subprocess
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from itertools import pairwise, zip_longest
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
from circuits.labeling.io import atomic_write_bytes, atomic_write_json, read_jsonl

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
FRONTIER_SHARES = {
    "balanced": {
        "provider_process": 0.20,
        "provider_contextual": 0.20,
        "provider_uncertain_or_incomplete": 0.20,
        "deterministic_terminal": 0.20,
        "deterministic_control": 0.20,
    },
    "process_weighted": {
        "provider_process": 0.55,
        "provider_contextual": 0.15,
        "provider_uncertain_or_incomplete": 0.10,
        "deterministic_terminal": 0.10,
        "deterministic_control": 0.10,
    },
    "uncertainty_weighted": {
        "provider_process": 0.25,
        "provider_contextual": 0.10,
        "provider_uncertain_or_incomplete": 0.45,
        "deterministic_terminal": 0.10,
        "deterministic_control": 0.10,
    },
}
UNIFORM_RESERVE_FRACTION = 0.10
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


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


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
    events: Iterable[Mapping[str, Any]],
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
        "proposal_status": "complete" if complete else "insufficient_exact_votes",
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


def _psu_route(
    members: Sequence[Mapping[str, Any]],
    proposal_by_id: Mapping[str, Mapping[str, Any]],
) -> str:
    routes = {
        _proposal_route(unit, proposal_by_id[str(unit["unit_id"])]) for unit in members
    }
    if len(routes) == 1:
        return next(iter(routes))
    return "provider_uncertain_or_incomplete"


def _sampling_psus(
    units: Sequence[Mapping[str, Any]], proposals: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Construct mandatory fragment PSUs; adjacency is metadata, never a PSU."""

    proposal_by_id = {str(row["unit_id"]): row for row in proposals}
    by_response: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for unit in units:
        by_response[str(unit["response_id"])].append(unit)
    output: list[dict[str, Any]] = []
    for response_id, response_units in by_response.items():
        ordered = sorted(response_units, key=lambda row: int(row["sequence_index"]))
        components: list[list[Mapping[str, Any]]] = []
        cursor = 0
        while cursor < len(ordered):
            fragment = ordered[cursor].get("fragment_of")
            if fragment is None:
                components.append([ordered[cursor]])
                cursor += 1
                continue
            members = []
            while (
                cursor < len(ordered) and ordered[cursor].get("fragment_of") == fragment
            ):
                members.append(ordered[cursor])
                cursor += 1
            components.append(members)
        prior_route: str | None = None
        correlation_run = -1
        for response_psu_index, members in enumerate(components):
            route = _psu_route(members, proposal_by_id)
            incomplete = any(
                proposal_by_id[str(member["unit_id"])]["proposal_status"] != "complete"
                for member in members
            )
            if incomplete or route != prior_route:
                correlation_run += 1
            identity = {
                "schema_version": "adag.process-witness.coarse-sampling-psu.v1",
                "response_id": response_id,
                "response_psu_index": response_psu_index,
                "member_unit_ids": [member["unit_id"] for member in members],
            }
            psu_id = f"pwcoarsepsuv1-{canonical_sha256(identity)[:32]}"
            atoms = []
            for member in members:
                width = int(member["token_span"][1]) - int(member["token_span"][0])
                if width <= 0:
                    raise ValueError("sampling atom has no token positions")
                atoms.append(
                    {
                        "unit_id": member["unit_id"],
                        "proposal_id": proposal_by_id[str(member["unit_id"])][
                            "proposal_id"
                        ],
                        "token_span": member["token_span"],
                        "position_count": width,
                        "atom_priority_sha256": canonical_sha256(
                            {
                                "namespace": "coarse-wave1-frontier-atom-v1",
                                "psu_id": psu_id,
                                "unit_id": member["unit_id"],
                            }
                        ),
                        "position_priority_namespace": "sha256(coarse-wave1-frontier-position-v1,psu_id,unit_id,token_index)",
                    }
                )
            output.append(
                {
                    **identity,
                    "psu_id": psu_id,
                    "route": route,
                    "mandatory_fragment_component": members[0].get("fragment_of")
                    is not None,
                    "fragment_of": members[0].get("fragment_of"),
                    "incomplete_hard_barrier": incomplete,
                    "correlation_run_id": f"{response_id}:run-{correlation_run}",
                    "correlation_run_is_weighting_psu": False,
                    "atoms": atoms,
                    "atom_count": len(atoms),
                    "position_count": sum(
                        int(atom["position_count"]) for atom in atoms
                    ),
                    "group_priority_sha256": canonical_sha256(
                        {
                            "namespace": "coarse-wave1-frontier-group-v1",
                            "psu_id": psu_id,
                        }
                    ),
                }
            )
            prior_route = None if incomplete else route
    if any(row["route"] not in ROUTES for row in output):
        raise ValueError("sampling PSU route drift")
    member_ids = [unit_id for row in output for unit_id in row["member_unit_ids"]]
    if member_ids != [str(unit["unit_id"]) for unit in units]:
        raise ValueError("sampling PSU ordered unit coverage drift")
    return output


def _capped_expected_allocations(
    *, capacities: Mapping[str, int], budget: int, desired_shares: Mapping[str, float]
) -> dict[str, float]:
    total_capacity = sum(capacities.values())
    if budget > total_capacity:
        raise ValueError("frontier budget exceeds PSU capacity")
    allocation = {
        route: budget
        * (
            UNIFORM_RESERVE_FRACTION * capacities[route] / total_capacity
            + (1.0 - UNIFORM_RESERVE_FRACTION) * desired_shares[route]
        )
        for route in ROUTES
    }
    free = set(ROUTES)
    while True:
        capped = {route for route in free if allocation[route] > capacities[route]}
        if not capped:
            break
        excess = sum(allocation[route] - capacities[route] for route in capped)
        for route in capped:
            allocation[route] = float(capacities[route])
        free -= capped
        if not free:
            break
        free_mass = sum(allocation[route] for route in free)
        if free_mass <= 0:
            per_route = excess / len(free)
            for route in free:
                allocation[route] += per_route
        else:
            for route in free:
                allocation[route] += excess * allocation[route] / free_mass
    if not math.isclose(sum(allocation.values()), budget, rel_tol=0, abs_tol=1e-7):
        raise ValueError("frontier expected allocation drift")
    return allocation


def _feasibility_frontiers(psus: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    capacities = Counter(str(row["route"]) for row in psus)
    if set(capacities) != set(ROUTES):
        raise ValueError("frontier route capacity is incomplete")
    rows = []
    for policy, shares in FRONTIER_SHARES.items():
        prior_pi = dict.fromkeys(ROUTES, 0.0)
        for budget in (30_000, 35_000, 40_000):
            allocation = _capped_expected_allocations(
                capacities=capacities, budget=budget, desired_shares=shares
            )
            for route in ROUTES:
                group_pi = allocation[route] / capacities[route]
                if group_pi + 1e-15 < prior_pi[route] or not 0 < group_pi <= 1:
                    raise ValueError("frontier nesting or positivity drift")
                prior_pi[route] = group_pi
                rows.append(
                    {
                        "schema_version": "adag.process-witness.coarse-yield-frontier.v1",
                        "policy": policy,
                        "nominal_expected_budget": budget,
                        "route": route,
                        "route_psu_capacity": capacities[route],
                        "expected_psus": allocation[route],
                        "group_inclusion_probability": group_pi,
                        "atom_conditional_probability": "1 / psu.atom_count",
                        "position_conditional_probability": "1 / atom.position_count",
                        "position_marginal_probability": "group_pi / psu.atom_count / atom.position_count",
                        "inverse_probability_weight": "1 / position_marginal_probability",
                        "uniform_reserve_fraction": UNIFORM_RESERVE_FRACTION,
                        "nested_priority_namespaces": {
                            "group": "coarse-wave1-frontier-group-v1",
                            "atom": "coarse-wave1-frontier-atom-v1",
                            "position": "coarse-wave1-frontier-position-v1",
                        },
                        "selected_or_frozen_trace_policy": False,
                        "interpretation": "yield_feasibility_only",
                        "exact_integer_sample_selected": False,
                        "allocation_kind": "fractional_expected_yield_design",
                        "t5_context_qualification": "pending; current measured qualification reaches only 1268 tokens",
                    }
                )
    return rows


def _candidate_probabilities(
    psus: Sequence[Mapping[str, Any]], frontiers: Sequence[Mapping[str, Any]]
) -> Iterable[dict[str, Any]]:
    pi_by_key = {
        (
            str(row["policy"]),
            int(row["nominal_expected_budget"]),
            str(row["route"]),
        ): float(row["group_inclusion_probability"])
        for row in frontiers
    }
    for psu in psus:
        atom_pi = 1.0 / int(psu["atom_count"])
        for atom in psu["atoms"]:
            position_pi = 1.0 / int(atom["position_count"])
            designs = []
            for policy in FRONTIER_SHARES:
                for budget in (30_000, 35_000, 40_000):
                    group_pi = pi_by_key[(policy, budget, str(psu["route"]))]
                    marginal = group_pi * atom_pi * position_pi
                    if marginal <= 0:
                        raise ValueError(
                            "candidate marginal inclusion probability is not positive"
                        )
                    designs.append(
                        {
                            "policy": policy,
                            "nominal_expected_budget": budget,
                            "group_inclusion_probability": group_pi,
                            "atom_conditional_probability": atom_pi,
                            "each_position_conditional_probability": position_pi,
                            "each_position_marginal_inclusion_probability": marginal,
                            "each_position_inverse_probability_weight": 1.0 / marginal,
                            "selected_or_frozen_trace_policy": False,
                            "exact_integer_sample_selected": False,
                        }
                    )
            yield {
                "schema_version": "adag.process-witness.coarse-candidate-inclusion.v1",
                "psu_id": psu["psu_id"],
                "unit_id": atom["unit_id"],
                "route": psu["route"],
                "token_span": atom["token_span"],
                "position_count": atom["position_count"],
                "group_priority_sha256": psu["group_priority_sha256"],
                "atom_priority_sha256": atom["atom_priority_sha256"],
                "position_priority_namespace": atom["position_priority_namespace"],
                "nested_budget_order": [30_000, 35_000, 40_000],
                "designs": designs,
            }


def _headline(proposals: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    coverage_counts: Counter[int] = Counter()
    broad_counts: Counter[str] = Counter()
    fine_agreement: Counter[str] = Counter()
    broad_agreement: Counter[str] = Counter()
    for row in proposals:
        broad_counts[str(row["broad_majority"])] += 1
        if row["assignment_route"] != "openai_pending":
            continue
        coverage_counts[int(row["replica_coverage"])] += 1
        if row["proposal_status"] == "complete":
            fine_agreement[str(row["fine_agreement_pattern"])] += 1
            broad_agreement[str(row["broad_agreement_pattern"])] += 1
    return {
        "provider_vote_coverage": {
            str(coverage): coverage_counts[coverage] for coverage in range(4)
        },
        "broad_counts": dict(sorted(broad_counts.items())),
        "fine_agreement": dict(sorted(fine_agreement.items())),
        "broad_agreement": dict(sorted(broad_agreement.items())),
    }


def _response_contexts(bundle: Mapping[str, Any]) -> list[dict[str, Any]]:
    manifest = bundle["manifest"]
    source_path = Path(str(manifest["source_workstation_bundle"]))
    if file_sha256(source_path) != manifest["source_workstation_bundle_sha256"]:
        raise ValueError("source workstation bundle hash drift")
    workstation = _load_object(source_path)
    documents = workstation.get("documents")
    if not isinstance(documents, list):
        raise ValueError("source workstation documents absent")
    output = []
    for document in documents:
        prompt = str(document["task_context"]["prompt"])
        response = str(document["text"])
        if (
            hashlib.sha256(prompt.encode()).hexdigest() != document["prompt_sha256"]
            or hashlib.sha256(response.encode()).hexdigest() != document["text_sha256"]
        ):
            raise ValueError("response context hash drift")
        output.append(
            {
                "schema_version": "adag.process-witness.coarse-response-context.v1",
                "response_id": document["response_id"],
                "response_source": document["response_source"],
                "task_title": next(
                    (line.strip() for line in prompt.splitlines() if line.strip()),
                    "untitled",
                ),
                "prompt_sha256": document["prompt_sha256"],
                "full_response_sha256": document["text_sha256"],
                "task_prompt": prompt,
                "full_response": response,
            }
        )
    if len(output) != EXPECTED_CENSUS["responses"]:
        raise ValueError("response context census drift")
    return output


def _length_bin(width: int) -> str:
    if width <= 8:
        return "01_1_to_8"
    if width <= 24:
        return "02_9_to_24"
    if width <= 48:
        return "03_25_to_48"
    return "04_49_to_96"


def _aggregate_table(
    units: Sequence[Mapping[str, Any]],
    proposals: Sequence[Mapping[str, Any]],
    contexts: Mapping[str, Mapping[str, Any]],
    dimension: str,
) -> list[dict[str, Any]]:
    proposal_by_id = {str(row["unit_id"]): row for row in proposals}
    response_sizes = Counter(str(row["response_id"]) for row in units)
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for unit in units:
        response_id = str(unit["response_id"])
        if dimension == "response":
            key = response_id
        elif dimension == "task":
            key = str(contexts[response_id]["task_title"])
        elif dimension == "source":
            key = json.dumps(contexts[response_id]["response_source"], sort_keys=True)
        elif dimension == "position":
            denominator = max(1, response_sizes[response_id] - 1)
            key = (
                f"decile_{min(9, int(int(unit['sequence_index']) * 10 / denominator))}"
            )
        elif dimension == "length":
            key = _length_bin(int(unit["token_span"][1]) - int(unit["token_span"][0]))
        elif dimension == "fragment":
            key = "fragment" if unit.get("fragment_of") else "not_fragment"
        else:
            raise ValueError(f"unknown aggregate dimension: {dimension}")
        proposal = proposal_by_id[str(unit["unit_id"])]
        counts[key]["units"] += 1
        counts[key][f"broad:{proposal['broad_majority']}"] += 1
        counts[key][f"coverage:{proposal['replica_coverage']}"] += 1
        counts[key][f"route:{_proposal_route(unit, proposal)}"] += 1
    return [
        {"dimension": dimension, "key": key, "counts": dict(sorted(value.items()))}
        for key, value in sorted(counts.items())
    ]


def _metrics(proposals: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    provider_votes = [
        vote
        for proposal in proposals
        if proposal["assignment_route"] == "openai_pending"
        for vote in proposal["physical_votes"]
    ]
    return {
        "schema_version": "adag.process-witness.coarse-confidence-boundary-metrics.v1",
        "provider_vote_confidence": dict(
            sorted(Counter(str(vote["confidence"]) for vote in provider_votes).items())
        ),
        "provider_vote_boundary_concerns": dict(
            sorted(
                Counter(
                    str(concern)
                    for vote in provider_votes
                    for concern in vote["boundary_concerns"]
                ).items()
            )
        ),
        "provider_votes_with_any_boundary_concern": sum(
            bool(vote["boundary_concerns"]) for vote in provider_votes
        ),
        "interpretation": "model-proposal stability and boundary metadata, not truth accuracy",
    }


def _missing_ledgers(
    *,
    proposals: Sequence[Mapping[str, Any]],
    request_by_id: Mapping[str, Mapping[str, Any]],
    residual_events: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    request_rows = [
        {
            "request_id": event["request_id"],
            "window_id": event["window_id"],
            "response_id": event["response_id"],
            "replica_index": event["replica_index"],
            "validation_status": event["validation_status"],
            "error_type": event["error_type"],
        }
        for event in residual_events
    ]
    windows: dict[str, dict[str, Any]] = {}
    for event in residual_events:
        request = request_by_id[str(event["request_id"])]
        row = windows.setdefault(
            str(event["window_id"]),
            {
                "window_id": event["window_id"],
                "response_id": event["response_id"],
                "focal_unit_ids": request["focal_unit_ids"],
                "residual_request_ids": [],
            },
        )
        row["residual_request_ids"].append(event["request_id"])
    unit_rows = [
        {
            "unit_id": proposal["unit_id"],
            "response_id": proposal["response_id"],
            "replica_coverage": proposal["replica_coverage"],
            "missing_replica_indices": proposal["missing_replica_indices"],
        }
        for proposal in proposals
        if proposal["proposal_status"] != "complete"
    ]
    return request_rows, list(windows.values()), unit_rows


def _audit_payloads(
    *,
    units: Sequence[Mapping[str, Any]],
    proposals: Sequence[Mapping[str, Any]],
    psus: Sequence[Mapping[str, Any]],
    windows: Sequence[Mapping[str, Any]],
    residual_events: Sequence[Mapping[str, Any]],
    contexts: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    unit_by_id = {str(row["unit_id"]): row for row in units}
    proposal_by_id = {str(row["unit_id"]): row for row in proposals}
    context_by_id = {str(row["response_id"]): row for row in contexts}
    psus_by_cell: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    psu_by_unit = {}
    for psu in psus:
        psus_by_cell[(str(psu["response_id"]), str(psu["route"]))].append(psu)
        for unit_id in psu["member_unit_ids"]:
            psu_by_unit[str(unit_id)] = psu
    selected: dict[str, dict[str, Any]] = {}

    def add(
        psu: Mapping[str, Any], stratum: str, probability: float | None = None
    ) -> None:
        audit_id = f"pwcoarseauditv1-{canonical_sha256({'psu_id': psu['psu_id']})[:32]}"
        row = selected.setdefault(
            audit_id,
            {
                "audit_id": audit_id,
                "psu": psu,
                "strata": [],
                "probability_base_inclusion_probability": None,
            },
        )
        if stratum not in row["strata"]:
            row["strata"].append(stratum)
        if probability is not None:
            row["probability_base_inclusion_probability"] = probability

    unavailable = []
    for response_id in sorted(context_by_id):
        for route in ("provider_process", "provider_contextual"):
            candidates = psus_by_cell.get((response_id, route), [])
            if not candidates:
                unavailable.append({"response_id": response_id, "route": route})
                continue
            psu = min(candidates, key=lambda row: str(row["group_priority_sha256"]))
            add(psu, f"probability_base:{route}", 1.0 / len(candidates))
    residual_window_ids = {str(row["window_id"]) for row in residual_events}
    window_by_id = {str(row["window_id"]): row for row in windows}
    for window_id in sorted(residual_window_ids):
        for unit_id in window_by_id[window_id]["focal_unit_ids"]:
            add(psu_by_unit[str(unit_id)], "diagnostic:incomplete_window")
    predicates = {
        "diagnostic:vote_disagreement": lambda proposal: (
            proposal["fine_agreement_pattern"] in {"two_one", "one_one_one"}
        ),
        "diagnostic:boundary_concern": lambda proposal: any(
            vote["boundary_concerns"] for vote in proposal["physical_votes"]
        ),
        "diagnostic:fragment": lambda proposal: proposal.get("fragment_of") is not None,
    }
    for stratum, predicate in predicates.items():
        candidates = {
            str(psu_by_unit[str(proposal["unit_id"])]["psu_id"]): psu_by_unit[
                str(proposal["unit_id"])
            ]
            for proposal in proposals
            if predicate(proposal)
        }
        for psu in sorted(
            candidates.values(), key=lambda row: str(row["group_priority_sha256"])
        )[:60]:
            add(psu, stratum)
    blind = []
    reveal = []
    for audit_id, item in sorted(selected.items()):
        psu = item["psu"]
        context = context_by_id[str(psu["response_id"])]
        targets = [unit_by_id[str(unit_id)] for unit_id in psu["member_unit_ids"]]
        blind.append(
            {
                "schema_version": "adag.process-witness.coarse-blind-audit-item.v1",
                "audit_id": audit_id,
                "response_id": psu["response_id"],
                "task_prompt": context["task_prompt"],
                "full_response": context["full_response"],
                "targets": [
                    {
                        "unit_id": unit["unit_id"],
                        "text": unit["text"],
                        "token_span": unit["token_span"],
                        "core_character_span": unit["core_character_span"],
                        "covering_character_span": unit["covering_character_span"],
                    }
                    for unit in targets
                ],
            }
        )
        reveal.append(
            {
                "schema_version": "adag.process-witness.coarse-audit-reveal-item.v1",
                "audit_id": audit_id,
                "psu_id": psu["psu_id"],
                "route": psu["route"],
                "strata": sorted(item["strata"]),
                "probability_base_inclusion_probability": item[
                    "probability_base_inclusion_probability"
                ],
                "proposals": [proposal_by_id[str(unit["unit_id"])] for unit in targets],
            }
        )
    plan = _hashed(
        {
            "schema_version": "adag.process-witness.coarse-audit-plan.v1",
            "probability_base": "one group-first draw per response for each available provider_process and provider_contextual cell",
            "probability_base_unavailable_cells": unavailable,
            "diagnostic_oversamples": [
                "all residual incomplete windows",
                "vote disagreement",
                "boundary concern",
                "fragment components",
            ],
            "residual_window_count": len(residual_window_ids),
            "prompt_block_bootstrap": "resample the 47 prompt_sha256 blocks; estimator and intervals pending predeclaration",
            "prompt_block_count": len(
                {str(context["prompt_sha256"]) for context in contexts}
            ),
            "reviewer_visible_blind_payload_excludes_model_strata": True,
            "acceptance_bounds_status": "pending_predeclaration",
            "blind_before_reveal_required": True,
            "claim_boundary": CLAIM_BOUNDARY,
        },
        "audit_plan_sha256",
    )
    return blind, reveal, plan


def _analysis_source_revision(temporary: Path) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[3]
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=no"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise ValueError("analysis build requires a clean tracked worktree")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tracked_paths = subprocess.run(
        ["git", "ls-files"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    relative_paths = [
        path
        for path in tracked_paths
        if path in {"pyproject.toml", "uv.lock"}
        or path.startswith(("circuits/analysis/bonafide/", "circuits/labeling/"))
        or path == "circuits/analysis/__init__.py"
        or path == "circuits/__init__.py"
        or path == "scripts/bonafide/build_process_witness_coarse_post_campaign_v1.py"
    ]
    root = temporary / "execution-source"
    files = []
    for relative in relative_paths:
        source = repo_root / relative
        committed = subprocess.run(
            ["git", "show", f"HEAD:{relative}"],
            cwd=repo_root,
            check=True,
            capture_output=True,
        ).stdout
        if source.read_bytes() != committed:
            raise ValueError(f"analysis source differs from HEAD: {relative}")
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        blob = subprocess.run(
            ["git", "rev-parse", f"HEAD:{relative}"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        files.append(
            {
                "path": relative,
                "git_blob": blob,
                "sha256": file_sha256(source),
                "bytes": source.stat().st_size,
                "copied_path": str(destination.relative_to(temporary)),
            }
        )
    return {
        "git_commit": commit,
        "git_tree": tree,
        "tracked_worktree_clean": True,
        "files": files,
        "lockfile_path": "uv.lock",
        "lockfile_sha256": file_sha256(repo_root / "uv.lock"),
        "repository_archive_copied": False,
        "transitive_source_scope": "complete tracked circuits.analysis.bonafide and circuits.labeling packages plus entrypoint and lock files",
    }


def _write_inventory(root: Path) -> dict[str, Any]:
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path in {
            root / "manifest.json",
            root / "evidence-inventory.json",
        }:
            continue
        if path.is_symlink():
            raise ValueError("analysis artifact cannot contain symlinks")
        files.append(
            {
                "path": str(path.relative_to(root)),
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    inventory = _hashed(
        {"schema_version": INVENTORY_SCHEMA, "files": files}, "inventory_sha256"
    )
    atomic_write_json(root / "evidence-inventory.json", inventory)
    return inventory


def _readonly_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def _writable_tree(root: Path) -> None:
    if not root.exists():
        return
    root.chmod(0o755)
    for path in root.rglob("*"):
        path.chmod(0o755 if path.is_dir() else 0o644)


def _publish_no_replace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("renameat2 is required for collision-safe publication")
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        raise FileExistsError(f"analysis destination exists: {destination}")
    raise OSError(error, os.strerror(error), str(destination))


def _iter_exact_rows(path: Path) -> Iterable[tuple[int, bytes, dict[str, Any]]]:
    with path.open("rb") as handle:
        for ordinal, line in enumerate(handle):
            if not line.endswith(b"\n"):
                raise ValueError(f"JSONL source lacks terminal newline: {path}")
            yield ordinal, line, json.loads(line)


def _exact_rows(path: Path) -> list[tuple[int, bytes, dict[str, Any]]]:
    return list(_iter_exact_rows(path))


def _copy_campaign_evidence(
    *,
    temporary: Path,
    bundle_root: Path,
    run_root: Path,
    continuation: Mapping[str, Any],
    bundle: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    root = temporary / "source-evidence"
    bundle_copy = root / "bundle"
    bundle_copy.mkdir(parents=True)
    for name in (
        "manifest.json",
        "protocol-config.json",
        "launch-gates.json",
        "cost-plan.json",
        "price-snapshot.json",
        "shards.json",
        "units.jsonl",
        "windows.jsonl",
        "request-index.jsonl",
    ):
        shutil.copyfile(bundle_root / name, bundle_copy / name)
    workstation_path = Path(str(bundle["manifest"]["source_workstation_bundle"]))
    shutil.copyfile(workstation_path, bundle_copy / "workstation-bundle.json")
    campaign = root / "campaign"
    campaign.mkdir()
    for name in (
        "continuation-manifest.json",
        "inherited-cost-reconciliation.json",
    ):
        shutil.copyfile(run_root / name, campaign / name)
    shutil.copytree(
        run_root / "failed-only-recovery", campaign / "failed-only-recovery"
    )
    latest_cost = sorted((run_root / "cost-status").glob("receipt-*.json"))[-1]
    (campaign / "cost-status").mkdir()
    shutil.copyfile(latest_cost, campaign / "cost-status" / latest_cost.name)
    sources: list[tuple[str, Path, Path]] = []
    inherited_source = run_root / "inherited-calibration-run/shards/shard-005"
    inherited_dest = campaign / "inherited-calibration/shard-005"
    inherited_dest.mkdir(parents=True)
    for name in ("collection.json", "events.jsonl"):
        shutil.copyfile(inherited_source / name, inherited_dest / name)
    shutil.copytree(inherited_source / "raw", inherited_dest / "raw")
    sources.append(
        ("inherited-calibration-shard-005", inherited_source, inherited_dest)
    )
    for binding in _all_attempts(run_root, continuation):
        attempt_id = str(binding["attempt_id"])
        source = run_root / "attempts" / attempt_id
        dest = campaign / "attempts" / attempt_id
        dest.mkdir(parents=True)
        for name in ("binding.json", "collection.json", "events.jsonl"):
            shutil.copyfile(source / name, dest / name)
        shutil.copytree(source / "raw", dest / "raw")
        sources.append((attempt_id, source, dest))
    precedence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    non_success_sources: list[tuple[dict[str, Any], Path, Path, str]] = []
    for source_id, source, dest in sources:
        event_path = source / "events.jsonl"
        copied_event_path = dest / "events.jsonl"
        for ordinal, line, event in _exact_rows(event_path):
            request_id = str(event["request_id"])
            binding = {
                "source_id": source_id,
                "source_path": str(copied_event_path.relative_to(temporary)),
                "source_ordinal": ordinal,
                "source_line_sha256": hashlib.sha256(line).hexdigest(),
                "validation_status": event["validation_status"],
            }
            precedence[request_id].append(binding)
            if event.get("validation_status") != "success":
                non_success_sources.append((event, source, dest, source_id))
    request_ids = [str(row["request_id"]) for row in bundle["request_index"]]
    event_ledger = []
    for request_id in request_ids:
        candidates = precedence.get(request_id, [])
        if not candidates:
            raise ValueError("event precedence source is absent")
        event_ledger.append(
            {
                "request_id": request_id,
                "effective_source": candidates[-1],
                "superseded_sources": candidates[:-1],
                "replacement_count": len(candidates) - 1,
            }
        )
    raw_cache: dict[Path, dict[str, tuple[int, bytes]]] = {}
    raw_ledger = []
    for event, source, dest, source_id in non_success_sources:
        collection = _load_object(source / "collection.json")
        matches = []
        for binding in collection["raw_file_bindings"]:
            if binding["source"] not in {"output", "error"}:
                continue
            original = (
                run_root / "inherited-calibration-run" / binding["path"]
                if source_id == "inherited-calibration-shard-005"
                else run_root / binding["path"]
            )
            if original not in raw_cache:
                raw_cache[original] = {
                    str(row["custom_id"]): (ordinal, line)
                    for ordinal, line, row in _exact_rows(original)
                }
            if str(event["request_id"]) in raw_cache[original]:
                matches.append(
                    (binding, original, raw_cache[original][str(event["request_id"])])
                )
        if len(matches) != 1:
            raise ValueError("non-success raw provider row binding is not unique")
        binding, original, (ordinal, line) = matches[0]
        copied = dest / "raw" / original.name
        if file_sha256(copied) != binding["sha256"]:
            raise ValueError("copied raw provider file binding drift")
        raw_ledger.append(
            {
                "request_id": event["request_id"],
                "source_id": source_id,
                "raw_file_path": str(copied.relative_to(temporary)),
                "raw_file_sha256": binding["sha256"],
                "raw_file_bytes": binding["bytes"],
                "raw_line_ordinal": ordinal,
                "raw_line_sha256": hashlib.sha256(line).hexdigest(),
            }
        )
    omitted = [
        {
            "path": shard["path"],
            "sha256": shard["sha256"],
            "bytes": shard["bytes"],
            "request_count": shard["request_count"],
            "reason": "deterministic large Batch transport input; exact metadata and generation sources retained",
        }
        for shard in bundle["shards"]
    ]
    omission = {
        "schema_version": "adag.process-witness.coarse-source-omission-ledger.v1",
        "omitted_batch_shards": omitted,
        "omitted_repository_archive": True,
        "retained_full_raw_provider_files": True,
        "retained_exact_collection_event_streams": True,
        "event_precedence_row_count": len(event_ledger),
        "non_success_raw_line_count": len(raw_ledger),
    }
    return event_ledger, raw_ledger, omission


def build_post_campaign_analysis(
    *, bundle_root: Path, run_root: Path, destination: Path
) -> dict[str, Any]:
    """Validate the completed campaign and atomically freeze its coarse analysis."""

    if destination.exists():
        raise FileExistsError(f"analysis destination exists: {destination}")
    bundle, continuation, effective, source_validation = _validated_sources(
        bundle_root=bundle_root, run_root=run_root
    )
    units = bundle["units"]
    if not isinstance(units, list):
        raise ValueError("source units were not loaded")
    request_by_id = {str(row["request_id"]): row for row in bundle["request_index"]}
    residual_events = [
        event for event in effective if event.get("validation_status") != "success"
    ]
    recovery_events = read_jsonl(
        run_root / "attempts/failed-only-recovery-000/events.jsonl"
    )
    recovery_ids = {str(row["request_id"]) for row in recovery_events}
    pre_recovery_events = read_jsonl(
        run_root / "inherited-calibration-run/shards/shard-005/events.jsonl"
    )
    for binding in continuation["attempts"]:
        pre_recovery_events.extend(
            read_jsonl(run_root / "attempts" / binding["attempt_id"] / "events.jsonl")
        )
    superseded = [
        event
        for event in pre_recovery_events
        if str(event["request_id"]) in recovery_ids
    ]
    non_success = [
        event
        for event in pre_recovery_events
        if event.get("validation_status") != "success"
    ] + [
        event
        for event in recovery_events
        if event.get("validation_status") != "success"
    ]
    if (
        len(effective) != 37_671
        or len(residual_events) != 15
        or {str(row["validation_status"]) for row in residual_events}
        != {"invalid_output"}
        or len(superseded) != 74
        or len(non_success) != 89
        or len({str(row["window_id"]) for row in residual_events}) != 12
        or len({str(row["response_id"]) for row in residual_events}) != 12
    ):
        raise ValueError("post-recovery event census drift")
    strict_votes = _provider_votes(effective)
    salvage_additions, exact_cover = _salvage_exact_ids(residual_events)
    if len(exact_cover) != 15 or not all(
        row["unique_one_extra_one_missing_predicate"] for row in exact_cover
    ):
        raise ValueError("residual exact-cover diagnostic drift")
    salvage_votes = {unit_id: list(votes) for unit_id, votes in strict_votes.items()}
    for unit_id, votes in salvage_additions.items():
        salvage_votes.setdefault(unit_id, []).extend(votes)
    strict_proposals = [
        _proposal(unit, strict_votes.get(str(unit["unit_id"]), [])) for unit in units
    ]
    proposals = [
        _proposal(unit, salvage_votes.get(str(unit["unit_id"]), [])) for unit in units
    ]
    strict_headline = _headline(strict_proposals)
    salvage_full_headline = _headline(proposals)
    salvage_headline = {
        key: salvage_full_headline[key]
        for key in ("provider_vote_coverage", "broad_counts", "fine_agreement")
    }
    if (
        strict_headline != EXPECTED_HEADLINES["strict_proposals"]
        or salvage_headline != EXPECTED_HEADLINES["conservative_exact_id_salvage"]
    ):
        raise ValueError("proposal headline exact census drift")
    contexts = _response_contexts(bundle)
    context_by_id = {str(row["response_id"]): row for row in contexts}
    psus = _sampling_psus(units, proposals)
    frontiers = _feasibility_frontiers(psus)
    strict_missing = _missing_ledgers(
        proposals=strict_proposals,
        request_by_id=request_by_id,
        residual_events=residual_events,
    )
    salvage_missing = _missing_ledgers(
        proposals=proposals,
        request_by_id=request_by_id,
        residual_events=residual_events,
    )
    blind, reveal, audit_plan = _audit_payloads(
        units=units,
        proposals=proposals,
        psus=psus,
        windows=bundle["windows"],
        residual_events=residual_events,
        contexts=contexts,
    )
    if audit_plan["prompt_block_count"] != 47:
        raise ValueError("audit prompt-block census drift")
    census = {
        "physical_requests": len(effective),
        "effective_success": sum(
            row.get("validation_status") == "success" for row in effective
        ),
        "residual_invalid_output": len(residual_events),
        "responses": len(contexts),
        "units": len(units),
        "openai_pending_units": sum(
            row["assignment_route"] == "openai_pending" for row in units
        ),
        "deterministic_surface_units": sum(
            row["assignment_route"] == "deterministic_surface" for row in units
        ),
        "deterministic_terminal_units": sum(
            row["assignment_route"] == "deterministic_terminal_serialization"
            for row in units
        ),
    }
    if census != EXPECTED_CENSUS:
        raise ValueError("analysis source literal census drift")
    tables = {
        dimension: _aggregate_table(
            units, proposals, context_by_id, dimension=dimension
        )
        for dimension in (
            "response",
            "task",
            "source",
            "position",
            "length",
            "fragment",
        )
    }
    completion_report = {
        "schema_version": REPORT_SCHEMA,
        "claim_boundary": CLAIM_BOUNDARY,
        "census": census,
        "strict_proposals": strict_headline,
        "conservative_exact_id_salvage": salvage_headline,
        "primary_analysis_lane": "conservative_exact_id_salvage.v1",
        "primary_lane_identity_basis": "schema-valid decisions whose unit_id exactly equals an expected focal ID; unknown IDs are rejected and never mapped",
        "strict_lane_role": "completion and missingness report",
        "unique_complement_predicate_role": "diagnostic only; no inferred substitution is promoted",
        "unique_complement_diagnostic": {
            "request_count": len(exact_cover),
            "predicate_true_count": sum(
                row["unique_one_extra_one_missing_predicate"] for row in exact_cover
            ),
            "unit_id_substitutions_performed": 0,
            "promoted_to_primary_lane": False,
        },
        "actual_total_cost_usd": source_validation["actual_total_cost_usd"],
        "sampling_policy_status": "yield_feasibility_only_not_selected_or_frozen",
        "audit_acceptance_bounds_status": "pending_predeclaration",
    }
    temporary = destination.parent / f".{destination.name}.building-{os.getpid()}"
    if temporary.exists():
        raise FileExistsError(f"analysis staging destination exists: {temporary}")
    temporary.mkdir(parents=True)
    try:
        event_precedence, raw_line_ledger, omission = _copy_campaign_evidence(
            temporary=temporary,
            bundle_root=bundle_root,
            run_root=run_root,
            continuation=continuation,
            bundle=bundle,
        )
        source_root = temporary / "source-evidence"
        atomic_write_bytes(
            source_root / "effective-events.jsonl", _jsonl_bytes(effective)
        )
        atomic_write_bytes(
            source_root / "superseded-events.jsonl", _jsonl_bytes(superseded)
        )
        atomic_write_bytes(
            source_root / "non-success-events.jsonl", _jsonl_bytes(non_success)
        )
        atomic_write_bytes(
            source_root / "response-contexts.jsonl", _jsonl_bytes(contexts)
        )
        atomic_write_bytes(
            source_root / "event-precedence-ledger.jsonl",
            _jsonl_bytes(event_precedence),
        )
        atomic_write_bytes(
            source_root / "non-success-raw-line-ledger.jsonl",
            _jsonl_bytes(raw_line_ledger),
        )
        atomic_write_json(temporary / "source-validation.json", source_validation)
        atomic_write_json(temporary / "source-omission-ledger.json", omission)
        atomic_write_bytes(
            temporary / "strict-proposals.jsonl", _jsonl_bytes(strict_proposals)
        )
        atomic_write_bytes(temporary / "proposals.jsonl", _jsonl_bytes(proposals))
        atomic_write_bytes(
            temporary / "exact-cover-diagnostics.jsonl", _jsonl_bytes(exact_cover)
        )
        atomic_write_bytes(temporary / "sampling-psus.jsonl", _jsonl_bytes(psus))
        atomic_write_bytes(
            temporary / "feasibility-frontiers.jsonl", _jsonl_bytes(frontiers)
        )
        atomic_write_bytes(
            temporary / "candidate-inclusion-probabilities.jsonl",
            _jsonl_bytes(_candidate_probabilities(psus, frontiers)),
        )
        atomic_write_json(
            temporary / "confidence-boundary-metrics.json", _metrics(proposals)
        )
        atomic_write_json(temporary / "analysis-tables.json", tables)
        for prefix, ledgers in (
            ("strict", strict_missing),
            ("primary", salvage_missing),
        ):
            for name, rows in zip(
                ("requests", "windows", "units"), ledgers, strict=True
            ):
                atomic_write_bytes(
                    temporary / f"{prefix}-missing-{name}.jsonl", _jsonl_bytes(rows)
                )
        atomic_write_bytes(temporary / "blind-audit.jsonl", _jsonl_bytes(blind))
        blind_sha = file_sha256(temporary / "blind-audit.jsonl")
        audit_plan["blind_payload_sha256"] = blind_sha
        audit_plan.pop("audit_plan_sha256")
        audit_plan = _hashed(audit_plan, "audit_plan_sha256")
        atomic_write_json(temporary / "audit-plan.json", audit_plan)
        for row in reveal:
            row["blind_payload_sha256"] = blind_sha
            row["audit_plan_sha256"] = audit_plan["audit_plan_sha256"]
        atomic_write_bytes(temporary / "audit-reveal.jsonl", _jsonl_bytes(reveal))
        atomic_write_json(temporary / "completion-report.json", completion_report)
        analysis_source = _analysis_source_revision(temporary)
        inventory = _write_inventory(temporary)
        manifest = _hashed(
            {
                "schema_version": ANALYSIS_SCHEMA,
                "status": ANALYSIS_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
                "bundle_manifest_sha256": bundle["manifest"]["manifest_sha256"],
                "continuation_manifest_sha256": continuation[
                    "continuation_manifest_sha256"
                ],
                "analysis_source_revision": analysis_source,
                "inventory_sha256": inventory["inventory_sha256"],
                "completion_report_sha256": file_sha256(
                    temporary / "completion-report.json"
                ),
                "primary_proposals_sha256": file_sha256(temporary / "proposals.jsonl"),
                "strict_proposals_sha256": file_sha256(
                    temporary / "strict-proposals.jsonl"
                ),
                "blind_audit_sha256": blind_sha,
                "audit_reveal_sha256": file_sha256(temporary / "audit-reveal.jsonl"),
                "network_calls_made": 0,
                "source_roots_mutated": False,
            },
            "manifest_sha256",
        )
        atomic_write_json(temporary / "manifest.json", manifest)
        del bundle, continuation, effective, source_validation, units
        del request_by_id, residual_events, recovery_events, recovery_ids
        del pre_recovery_events, superseded, non_success
        del strict_votes, salvage_additions, exact_cover, salvage_votes
        del strict_proposals, proposals, contexts, context_by_id, psus, frontiers
        del strict_missing, salvage_missing, blind, reveal, audit_plan
        del tables, event_precedence, raw_line_ledger, omission
        gc.collect()
        _readonly_tree(temporary)
        load_frozen_post_campaign_analysis(temporary)
        _publish_no_replace(temporary, destination)
        load_frozen_post_campaign_analysis(destination)
        return manifest
    except BaseException:
        if temporary.exists():
            _writable_tree(temporary)
            shutil.rmtree(temporary)
        raise


def _validate_readonly_modes(root: Path) -> None:
    if root.stat().st_mode & 0o777 != 0o555:
        raise ValueError("analysis root mode drift")
    for path in root.rglob("*"):
        expected = 0o555 if path.is_dir() else 0o444
        if path.stat().st_mode & 0o777 != expected:
            raise ValueError(f"analysis mode drift: {path.relative_to(root)}")


def _validate_sampling_design(root: Path) -> None:
    units = read_jsonl(root / "source-evidence/bundle/units.jsonl")
    unit_ids = [str(row["unit_id"]) for row in units]
    psu_id_by_unit: dict[str, str] = {}
    by_response: dict[str, list[tuple[int, bool, str]]] = defaultdict(list)
    unit_cursor = 0
    for psu in _iter_jsonl(root / "sampling-psus.jsonl"):
        psu_id = str(psu["psu_id"])
        for member in psu["member_unit_ids"]:
            unit_id = str(member)
            if (
                unit_cursor >= len(unit_ids)
                or unit_ids[unit_cursor] != unit_id
                or unit_id in psu_id_by_unit
            ):
                raise ValueError("sampling PSU ordered unit coverage drift")
            psu_id_by_unit[unit_id] = psu_id
            unit_cursor += 1
        by_response[str(psu["response_id"])].append(
            (
                int(psu["response_psu_index"]),
                bool(psu.get("incomplete_hard_barrier")),
                str(psu.get("correlation_run_id")),
            )
        )
    if unit_cursor != len(unit_ids):
        raise ValueError("sampling PSU ordered unit coverage drift")
    fragments: dict[str, set[str]] = defaultdict(set)
    for unit in units:
        fragment = unit.get("fragment_of")
        if fragment is not None:
            fragments[str(fragment)].add(psu_id_by_unit[str(unit["unit_id"])])
    if any(len(psu_ids) != 1 for psu_ids in fragments.values()):
        raise ValueError("fragment PSU partition drift")
    for response_psus in by_response.values():
        ordered = sorted(response_psus)
        if [row[0] for row in ordered] != list(range(len(ordered))):
            raise ValueError("sampling PSU response index drift")
        for index, (_psu_index, incomplete, correlation_run_id) in enumerate(ordered):
            if not incomplete:
                continue
            for neighbor_index in (index - 1, index + 1):
                if (
                    0 <= neighbor_index < len(ordered)
                    and ordered[neighbor_index][2] == correlation_run_id
                ):
                    raise ValueError("incomplete PSU hard-barrier drift")
    seen_candidate_units: set[str] = set()
    for candidate in _iter_jsonl(root / "candidate-inclusion-probabilities.jsonl"):
        unit_id = str(candidate["unit_id"])
        if (
            unit_id not in psu_id_by_unit
            or psu_id_by_unit[unit_id] != candidate["psu_id"]
            or unit_id in seen_candidate_units
        ):
            raise ValueError("candidate ownership or policy-status drift")
        seen_candidate_units.add(unit_id)
        by_policy: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in candidate["designs"]:
            if (
                row.get("selected_or_frozen_trace_policy") is not False
                or row.get("exact_integer_sample_selected") is not False
            ):
                raise ValueError("candidate ownership or policy-status drift")
            group_pi = float(row["group_inclusion_probability"])
            atom_pi = float(row["atom_conditional_probability"])
            position_pi = float(row["each_position_conditional_probability"])
            marginal = float(row["each_position_marginal_inclusion_probability"])
            weight = float(row["each_position_inverse_probability_weight"])
            if (
                not 0 < group_pi <= 1
                or not 0 < atom_pi <= 1
                or not 0 < position_pi <= 1
                or not math.isclose(
                    marginal,
                    group_pi * atom_pi * position_pi,
                    rel_tol=0,
                    abs_tol=1e-15,
                )
                or not math.isclose(weight, 1.0 / marginal, rel_tol=1e-12, abs_tol=0)
            ):
                raise ValueError("candidate numeric inclusion-probability drift")
            by_policy[str(row["policy"])].append(row)
        for rows in by_policy.values():
            ordered = sorted(rows, key=lambda row: int(row["nominal_expected_budget"]))
            if [int(row["nominal_expected_budget"]) for row in ordered] != [
                30_000,
                35_000,
                40_000,
            ] or any(
                float(later["group_inclusion_probability"])
                < float(earlier["group_inclusion_probability"])
                for earlier, later in pairwise(ordered)
            ):
                raise ValueError("candidate nesting drift")
    if seen_candidate_units != set(psu_id_by_unit):
        raise ValueError("candidate ownership or policy-status drift")


def _compare_jsonl_rows(
    path: Path, rows: Iterable[Mapping[str, Any]], label: str
) -> None:
    with path.open(encoding="utf-8") as handle:
        observed = (json.loads(line) for line in handle if line.strip())
        for index, (actual, expected) in enumerate(
            zip_longest(observed, rows), start=1
        ):
            if (
                actual is None
                or expected is None
                or canonical_sha256(actual) != canonical_sha256(expected)
            ):
                raise ValueError(f"{label} recomputation drift at row {index}")


def _validate_embedded_event_sources(root: Path) -> None:
    effective_hash_by_id: dict[str, str] = {}
    for event in _iter_jsonl(root / "source-evidence/effective-events.jsonl"):
        request_id = str(event["request_id"])
        if request_id in effective_hash_by_id:
            raise ValueError("embedded effective event identity collision")
        effective_hash_by_id[request_id] = canonical_sha256(event)
    sources: dict[Path, dict[int, tuple[str, str, bool]]] = defaultdict(dict)
    ledger_request_ids: set[str] = set()
    precedence_count = 0
    replacement_count = 0
    for row in _iter_jsonl(root / "source-evidence/event-precedence-ledger.jsonl"):
        precedence_count += 1
        request_id = str(row["request_id"])
        if request_id in ledger_request_ids:
            raise ValueError("embedded event precedence identity collision")
        ledger_request_ids.add(request_id)
        candidates = [*row["superseded_sources"], row["effective_source"]]
        replacement_count += int(row["replacement_count"])
        if row["replacement_count"] != len(row["superseded_sources"]):
            raise ValueError("embedded event replacement ledger drift")
        for binding_index, binding in enumerate(candidates):
            path = root / binding["source_path"]
            ordinal = int(binding["source_ordinal"])
            if ordinal in sources[path]:
                raise ValueError("embedded event source ordinal collision")
            sources[path][ordinal] = (
                request_id,
                str(binding["source_line_sha256"]),
                binding_index == len(candidates) - 1,
            )
    if (
        precedence_count != 37_671
        or replacement_count != 74
        or ledger_request_ids != set(effective_hash_by_id)
    ):
        raise ValueError("embedded event precedence census drift")
    for path, expected_by_ordinal in sources.items():
        seen_ordinals: set[int] = set()
        for ordinal, line, event in _iter_exact_rows(path):
            expected = expected_by_ordinal.get(ordinal)
            if expected is None:
                continue
            request_id, line_sha256, is_effective = expected
            if (
                str(event["request_id"]) != request_id
                or hashlib.sha256(line).hexdigest() != line_sha256
                or (
                    is_effective
                    and canonical_sha256(event) != effective_hash_by_id[request_id]
                )
            ):
                raise ValueError("embedded event source line drift")
            seen_ordinals.add(ordinal)
        if seen_ordinals != set(expected_by_ordinal):
            raise ValueError("embedded event source ordinal drift")

    raw_sources: dict[Path, dict[int, tuple[str, str, str, int]]] = defaultdict(dict)
    raw_count = 0
    for binding in _iter_jsonl(
        root / "source-evidence/non-success-raw-line-ledger.jsonl"
    ):
        raw_count += 1
        path = root / binding["raw_file_path"]
        ordinal = int(binding["raw_line_ordinal"])
        if ordinal in raw_sources[path]:
            raise ValueError("embedded non-success raw ordinal collision")
        raw_sources[path][ordinal] = (
            str(binding["request_id"]),
            str(binding["raw_line_sha256"]),
            str(binding["raw_file_sha256"]),
            int(binding["raw_file_bytes"]),
        )
    if raw_count != 89:
        raise ValueError("embedded non-success raw row census drift")
    for path, expected_by_ordinal in raw_sources.items():
        file_bindings = {
            (raw_sha256, raw_bytes)
            for _request_id, _line_sha256, raw_sha256, raw_bytes in expected_by_ordinal.values()
        }
        if len(file_bindings) != 1 or (file_sha256(path), path.stat().st_size) != next(
            iter(file_bindings)
        ):
            raise ValueError("embedded non-success raw provider file drift")
        seen_ordinals: set[int] = set()
        for ordinal, line, provider_row in _iter_exact_rows(path):
            expected = expected_by_ordinal.get(ordinal)
            if expected is None:
                continue
            request_id, line_sha256, _raw_sha256, _raw_bytes = expected
            if (
                str(provider_row["custom_id"]) != request_id
                or hashlib.sha256(line).hexdigest() != line_sha256
            ):
                raise ValueError("embedded non-success raw provider line drift")
            seen_ordinals.add(ordinal)
        if seen_ordinals != set(expected_by_ordinal):
            raise ValueError("embedded non-success raw provider ordinal drift")


def _validate_independent_recomputation(
    root: Path, manifest: Mapping[str, Any], report: Mapping[str, Any]
) -> None:
    units = read_jsonl(root / "source-evidence/bundle/units.jsonl")
    events_path = root / "source-evidence/effective-events.jsonl"
    strict_votes = _provider_votes(_iter_jsonl(events_path))
    residual = [
        row
        for row in _iter_jsonl(events_path)
        if row.get("validation_status") != "success"
    ]
    additions, diagnostics = _salvage_exact_ids(residual)
    salvage_votes = {unit_id: list(votes) for unit_id, votes in strict_votes.items()}
    for unit_id, votes in additions.items():
        salvage_votes.setdefault(unit_id, []).extend(votes)
    primary = [
        _proposal(unit, salvage_votes.get(str(unit["unit_id"]), [])) for unit in units
    ]
    _compare_jsonl_rows(
        root / "strict-proposals.jsonl",
        (_proposal(unit, strict_votes.get(str(unit["unit_id"]), [])) for unit in units),
        "strict proposals",
    )
    _compare_jsonl_rows(root / "proposals.jsonl", primary, "primary proposals")
    _compare_jsonl_rows(
        root / "exact-cover-diagnostics.jsonl", diagnostics, "exact-cover diagnostics"
    )
    strict_headline = _headline(
        _proposal(unit, strict_votes.get(str(unit["unit_id"]), [])) for unit in units
    )
    primary_headline = _headline(primary)
    if (
        strict_headline != report["strict_proposals"]
        or {
            key: primary_headline[key]
            for key in ("provider_vote_coverage", "broad_counts", "fine_agreement")
        }
        != report["conservative_exact_id_salvage"]
    ):
        raise ValueError("proposal summary independent recomputation drift")
    del strict_votes, additions, salvage_votes, residual
    psus = _sampling_psus(units, primary)
    frontiers = _feasibility_frontiers(psus)
    _compare_jsonl_rows(root / "sampling-psus.jsonl", psus, "sampling PSUs")
    _compare_jsonl_rows(
        root / "feasibility-frontiers.jsonl", frontiers, "feasibility frontiers"
    )
    _compare_jsonl_rows(
        root / "candidate-inclusion-probabilities.jsonl",
        _candidate_probabilities(psus, frontiers),
        "candidate inclusion probabilities",
    )
    del units, primary, psus, frontiers
    gc.collect()
    contexts = read_jsonl(root / "source-evidence/response-contexts.jsonl")
    if len(contexts) != 188 or len({row["prompt_sha256"] for row in contexts}) != 47:
        raise ValueError("embedded response context census drift")
    for context in contexts:
        if (
            hashlib.sha256(str(context["task_prompt"]).encode()).hexdigest()
            != context["prompt_sha256"]
            or hashlib.sha256(str(context["full_response"]).encode()).hexdigest()
            != context["full_response_sha256"]
        ):
            raise ValueError("embedded response context hash drift")
    del contexts
    audit_plan = _load_object(root / "audit-plan.json")
    _verify_self_hash(audit_plan, "audit_plan_sha256", "audit plan")
    blind_path = root / "blind-audit.jsonl"
    if (
        audit_plan.get("prompt_block_count") != 47
        or audit_plan.get("residual_window_count") != 12
        or audit_plan.get("acceptance_bounds_status") != "pending_predeclaration"
        or audit_plan.get("blind_payload_sha256") != file_sha256(blind_path)
        or manifest.get("blind_audit_sha256") != file_sha256(blind_path)
    ):
        raise ValueError("blind audit plan binding drift")
    blind = read_jsonl(blind_path)
    forbidden = {"route", "strata", "proposals", "fine_votes", "broad_majority"}
    if any(forbidden.intersection(row) for row in blind):
        raise ValueError("blind audit leaks model-derived strata")
    blind_ids = {str(row["audit_id"]) for row in blind}
    reveal = read_jsonl(root / "audit-reveal.jsonl")
    if {str(row["audit_id"]) for row in reveal} != blind_ids or any(
        row.get("blind_payload_sha256") != file_sha256(blind_path)
        or row.get("audit_plan_sha256") != audit_plan["audit_plan_sha256"]
        for row in reveal
    ):
        raise ValueError("audit reveal binding drift")
    del blind, reveal
    gc.collect()
    _validate_embedded_event_sources(root)
    source_manifest = _load_object(root / "source-evidence/bundle/manifest.json")
    _verify_self_hash(source_manifest, "manifest_sha256", "copied source bundle")
    if source_manifest["manifest_sha256"] != manifest["bundle_manifest_sha256"]:
        raise ValueError("copied source bundle manifest binding drift")
    workstation = root / "source-evidence/bundle/workstation-bundle.json"
    if file_sha256(workstation) != source_manifest["source_workstation_bundle_sha256"]:
        raise ValueError("copied workstation bundle binding drift")
    revision = manifest.get("analysis_source_revision")
    if (
        not isinstance(revision, Mapping)
        or revision.get("repository_archive_copied") is not False
    ):
        raise ValueError("analysis source revision drift")
    for binding in revision["files"]:
        path = root / binding["copied_path"]
        if (
            file_sha256(path) != binding["sha256"]
            or path.stat().st_size != binding["bytes"]
        ):
            raise ValueError("analysis copied source file drift")


def load_frozen_post_campaign_analysis(root: Path) -> dict[str, Any]:
    """Validate a frozen analysis using only evidence beneath ``root``."""

    if (
        root.is_symlink()
        or (root / "manifest.json").is_symlink()
        or (root / "evidence-inventory.json").is_symlink()
    ):
        raise ValueError("analysis root manifest or inventory cannot be a symlink")
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
    _validate_sampling_design(root)
    if "analysis_source_revision" in manifest:
        _validate_independent_recomputation(root, manifest, report)
    return {"manifest": manifest, "completion_report": report}
