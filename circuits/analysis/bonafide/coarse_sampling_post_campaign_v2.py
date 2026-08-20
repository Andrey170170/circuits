"""Additive sampling-design sensitivity artifact derived from immutable v1.

The v2 artifact deliberately depends on the frozen v1 post-campaign artifact.  It
does not copy or mutate the original proposal bank or continuation run, select a
trace policy, or claim that any target is resource qualified.
"""

from __future__ import annotations

import bisect
import gzip
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from itertools import pairwise, zip_longest
from pathlib import Path
from typing import Any

import numpy as np

from circuits.analysis.bonafide.canonical import canonical_sha256, file_sha256
from circuits.analysis.bonafide.coarse_sampling_openai_batch_production_v1 import (
    _parse_row as parse_provider_row,
)
from circuits.analysis.bonafide.coarse_sampling_post_campaign_v1 import (
    _publish_no_replace,
    load_frozen_post_campaign_analysis,
)
from circuits.labeling.io import atomic_write_bytes, atomic_write_json

ANALYSIS_SCHEMA = "adag.process-witness.coarse-post-campaign-sampling.v2"
INVENTORY_SCHEMA = "adag.process-witness.coarse-post-campaign-sampling-inventory.v2"
PARENT_BINDING_SCHEMA = "adag.process-witness.coarse-post-campaign-parent-binding.v2"
ELIGIBILITY_SCHEMA = "adag.process-witness.coarse-mechanism-eligibility.v2"
FRONTIER_SCHEMA = "adag.process-witness.coarse-overlap-frontier.v2"
CANDIDATE_SCHEMA = "adag.process-witness.coarse-position-inclusion.v2"
REALIZED_SCHEMA = "adag.process-witness.coarse-realized-tier.v2"
AUDIT_PLAN_SCHEMA = "adag.process-witness.coarse-audit-supplement-plan.v2"
CLAIM_BOUNDARY = (
    "This is an additive graph-blind sampling-design sensitivity artifact. It "
    "freezes candidate designs and nested candidate tiers, but does not select "
    "a policy or target set for tracing, establish label truth, test ADAG "
    "adequacy, identify motifs or witnesses, or make a faithfulness claim."
)

MECHANISMS = (
    "process_enrichment",
    "evaluation_commitment",
    "diversity",
    "uncertainty_missing",
    "uniform_reserve",
)
OWNERSHIP_ORDER = (
    "uniform_reserve",
    "uncertainty_missing",
    "evaluation_commitment",
    "process_enrichment",
    "diversity",
)
SHARES = {
    "balanced": {
        "process_enrichment": 0.20,
        "evaluation_commitment": 0.20,
        "diversity": 0.20,
        "uncertainty_missing": 0.20,
        "uniform_reserve": 0.20,
    },
    "process_weighted": {
        "process_enrichment": 0.40,
        "evaluation_commitment": 0.20,
        "diversity": 0.15,
        "uncertainty_missing": 0.15,
        "uniform_reserve": 0.10,
    },
    "uncertainty_weighted": {
        "process_enrichment": 0.25,
        "evaluation_commitment": 0.20,
        "diversity": 0.15,
        "uncertainty_missing": 0.30,
        "uniform_reserve": 0.10,
    },
}
BUDGETS = (30_000, 35_000, 40_000)
EXPECTED_FRAME = {"psus": 94_479, "atoms": 94_546, "positions": 842_007}
EXPECTED_ELIGIBILITY = {
    "process_enrichment": {"psus": 20_330, "atoms": 20_373, "positions": 304_951},
    "evaluation_commitment": {
        "psus": 17_441,
        "atoms": 17_459,
        "positions": 171_041,
    },
    "uncertainty_missing": {
        "psus": 30_842,
        "atoms": 30_908,
        "positions": 284_695,
    },
    "diversity": {"psus": 74_979, "atoms": 75_046, "positions": 820_236},
    "uniform_reserve": EXPECTED_FRAME,
}
EVALUATION_LABELS = {
    "evaluation_or_revision",
    "intermediate_commitment",
    "final_answer",
}
RARE_FINE_LABELS = (*sorted(EVALUATION_LABELS), "uncertain")
AUDIT_SUPPLEMENT_POOLS = (
    "low_confidence",
    "unresolved_or_incomplete",
    "long_unit_at_96_token_segmentation_cap",
    "long_computation_syntax_ge_49_tokens",
    *(f"rare_fine:{label}" for label in RARE_FINE_LABELS),
)
ANCHOR_RE = re.compile(
    r"(?<![\w.])[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?![\w.])|"
    r"(?:->|=>|→|←|=|\+|-|\*|/|%|\^|<=|>=|<|>)"
)
DEFAULT_COHORT_ROOT = Path(
    "/scratch/rai/vast1/u1653998/bonafide/campaigns/"
    "qwen3-4b-thinking-2507-process-witness-broad-v1/cohorts/"
    "atlas-responses-backfilled-v2"
)
MEASURED_CONTEXT_ENVELOPE = 1268
STEP0_MANIFEST_RELATIVE = Path(
    "scripts/bonafide/manifests/qwen3_4b_thinking_process_witness_step0_v1.json"
)
EXPECTED_CONTEXT_CENSUS = {
    "responses": 188,
    "assistant_prefix_token_count_min": 175,
    "assistant_prefix_token_count_max": 2631,
    "frame_positions": 842_007,
    "rendered_total_context_token_count_min": 176,
    "rendered_total_context_token_count_max": 10_767,
    "within_measured_1268_envelope": 102_019,
    "above_measured_1268_envelope": 739_988,
}


def _expected_design_contract(
    *, kernel_stream_sha256: str, group_base_stream_sha256: str
) -> dict[str, Any]:
    return {
        "schema_version": "adag.process-witness.coarse-sampling-design-contract.v2",
        "mechanisms_plan_order": list(MECHANISMS),
        "first_owner_precedence": list(OWNERSHIP_ORDER),
        "shares": SHARES,
        "budgets": list(BUDGETS),
        "kernel_stream_sha256": kernel_stream_sha256,
        "group_base_stream_sha256": group_base_stream_sha256,
        "group_to_atom_to_position": True,
        "uniform_each_position_probability_equal": True,
        "observable_process_anchors_only": True,
        "halo_status": "deferred",
        "candidate_design_status": "frozen",
        "selected_for_tracing": False,
        "trace_ready": False,
    }


def _validate_candidate_only_manifest(manifest: Mapping[str, Any]) -> None:
    expected = {
        "status": "frozen_candidate_designs_not_selected_for_tracing",
        "candidate_design_status": "frozen",
        "candidate_tier_status": "frozen_candidate_only",
        "selected_for_tracing": False,
        "trace_ready": False,
        "trace_policy_selection_status": "pending_audit_and_resource_gate",
        "network_calls_made": 0,
        "parent_v1_mutated": False,
    }
    if any(manifest.get(field) != value for field, value in expected.items()):
        raise ValueError("v2 sampling candidate-only manifest drift")


def _validate_static_artifact_contract(root: Path, manifest: Mapping[str, Any]) -> None:
    contract_path = root / "design-contract.json"
    if file_sha256(contract_path) != manifest.get("design_contract_sha256"):
        raise ValueError("v2 sampling design-contract manifest binding drift")
    contract = _load_object(contract_path)
    expected_contract = _expected_design_contract(
        kernel_stream_sha256=str(contract.get("kernel_stream_sha256", "")),
        group_base_stream_sha256=str(contract.get("group_base_stream_sha256", "")),
    )
    if contract != expected_contract:
        raise ValueError("v2 sampling design contract drift")
    for relative, field in (
        ("expected-frontiers.jsonl", "expected_frontiers_sha256"),
        ("realized-candidate-tiers.jsonl.gz", "realized_candidate_tiers_sha256"),
    ):
        if file_sha256(root / relative) != manifest.get(field):
            raise ValueError(f"v2 sampling {relative} manifest binding drift")


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"expected JSONL object: {path}")
                yield value


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (
            json.dumps(
                dict(row),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        for row in rows
    )


def _verify_self_hash(value: Mapping[str, Any], field: str, label: str) -> None:
    payload = dict(value)
    observed = payload.pop(field, None)
    if observed != canonical_sha256(payload):
        raise ValueError(f"{label} self-hash drift")


def _hashed(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(value)
    result[field] = canonical_sha256(result)
    return result


def _reject_descendant_symlinks(root: Path) -> None:
    if root.is_symlink():
        raise ValueError("artifact root may not be a symlink")
    for directory, names, files in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in [*names, *files]:
            if (base / name).is_symlink():
                raise ValueError(
                    f"artifact descendant symlink is forbidden: {(base / name).relative_to(root)}"
                )


def _exact_file_rows(path: Path) -> list[tuple[int, bytes, dict[str, Any]]]:
    rows = []
    with path.open("rb") as handle:
        for ordinal, line in enumerate(handle):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"expected provider row object: {path}")
                rows.append((ordinal, line, value))
    return rows


def _validate_parent_bundle(parent: Path) -> dict[str, Any]:
    bundle_root = parent / "source-evidence/bundle"
    manifest = _load_object(bundle_root / "manifest.json")
    _verify_self_hash(manifest, "manifest_sha256", "copied bundle manifest")
    omitted = {f"batch-shards/shard-{index:03d}.jsonl" for index in range(6)}
    declared = {str(row["path"]): row for row in manifest["files"]}
    retained = {
        path.relative_to(bundle_root).as_posix(): path
        for path in bundle_root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    workstation = retained.pop("workstation-bundle.json", None)
    if (
        workstation is None
        or file_sha256(workstation) != manifest["source_workstation_bundle_sha256"]
        or set(declared) - omitted != set(retained)
    ):
        raise ValueError("copied bundle retained-file membership drift")
    for relative, path in retained.items():
        binding = declared[relative]
        if (path.stat().st_size, file_sha256(path)) != (
            int(binding["bytes"]),
            str(binding["sha256"]),
        ):
            raise ValueError(f"copied bundle file drift: {relative}")
    return manifest


def _validate_parent_collections(parent: Path) -> None:
    campaign = parent / "source-evidence/campaign"
    continuation = _load_object(campaign / "continuation-manifest.json")
    _verify_self_hash(
        continuation, "continuation_manifest_sha256", "copied continuation manifest"
    )
    recovery = _load_object(campaign / "failed-only-recovery/manifest.json")
    _verify_self_hash(recovery, "recovery_manifest_sha256", "copied recovery manifest")
    recovery_intent = _load_object(
        campaign / "failed-only-recovery/preparation-intent.json"
    )
    _verify_self_hash(
        recovery_intent, "recovery_intent_sha256", "copied recovery intent"
    )
    if (
        recovery["continuation_manifest_sha256"]
        != continuation["continuation_manifest_sha256"]
        or recovery["successful_requests_rerun"] != 0
        or recovery["attempt"]["generation"] != "failed-only-recovery"
        or recovery["recovery_intent_sha256"]
        != recovery_intent["recovery_intent_sha256"]
        or recovery["attempt"]["request_ids_in_order"]
        != recovery_intent["request_ids_in_order"]
    ):
        raise ValueError("copied recovery semantics drift")

    event_request_ids: set[str] = set()
    attempt_roots = sorted((campaign / "attempts").iterdir())
    for attempt_root in attempt_roots:
        binding = _load_object(attempt_root / "binding.json")
        _verify_self_hash(binding, "binding_sha256", "copied attempt binding")
        collection = _load_object(attempt_root / "collection.json")
        _verify_self_hash(collection, "collection_sha256", "copied collection")
        events_path = attempt_root / "events.jsonl"
        events = list(_iter_jsonl(events_path))
        expected_ids = [str(value) for value in binding["request_ids_in_order"]]
        ids = [str(row["request_id"]) for row in events]
        generation = str(binding["generation"])
        unexpected_overlap = (
            not event_request_ids.isdisjoint(ids)
            if generation != "failed-only-recovery"
            else False
        )
        if (
            file_sha256(events_path) != collection["events_sha256"]
            or len(events) != collection["request_count"]
            or Counter(row["validation_status"] == "success" for row in events)[True]
            != collection["success_count"]
            or Counter(row["validation_status"] == "success" for row in events)[False]
            != collection["failure_count"]
            or ids != expected_ids
            or unexpected_overlap
        ):
            raise ValueError(
                f"copied collection event semantics drift: {attempt_root.name}"
            )
        if generation != "failed-only-recovery":
            event_request_ids.update(ids)
        for raw_binding in collection["raw_file_bindings"]:
            relative = Path(str(raw_binding["path"]))
            path = campaign / relative
            if (path.stat().st_size, file_sha256(path)) != (
                int(raw_binding["bytes"]),
                str(raw_binding["sha256"]),
            ):
                raise ValueError(f"copied collection raw binding drift: {relative}")

    inherited = _load_object(
        campaign / "inherited-calibration/shard-005/collection.json"
    )
    _verify_self_hash(inherited, "collection_sha256", "copied inherited collection")
    inherited_events_path = campaign / "inherited-calibration/shard-005/events.jsonl"
    inherited_events = list(_iter_jsonl(inherited_events_path))
    if (
        file_sha256(inherited_events_path) != inherited["events_sha256"]
        or len(inherited_events) != inherited["request_count"]
        or sum(row["validation_status"] == "success" for row in inherited_events)
        != inherited["success_count"]
        or sum(row["validation_status"] != "success" for row in inherited_events)
        != inherited["failure_count"]
    ):
        raise ValueError("copied inherited collection semantics drift")

    recovery_ids = set(map(str, recovery["attempt"]["request_ids_in_order"]))
    failed_before_recovery = {
        str(row["request_id"])
        for row in [*inherited_events]
        if row["validation_status"] != "success"
    }
    for attempt_root in attempt_roots:
        binding = _load_object(attempt_root / "binding.json")
        if binding["generation"] == "continuation-primary":
            failed_before_recovery.update(
                str(row["request_id"])
                for row in _iter_jsonl(attempt_root / "events.jsonl")
                if row["validation_status"] != "success"
            )
    if recovery_ids != failed_before_recovery:
        raise ValueError("failed-only recovery membership drift")
    recovery_binding = _load_object(
        campaign / "attempts/failed-only-recovery-000/binding.json"
    )
    if (
        recovery_binding["request_ids_in_order"]
        != recovery_intent["request_ids_in_order"]
    ):
        raise ValueError("failed-only recovery order binding drift")


def _validate_parent_raw_reparse(parent: Path) -> None:
    source = parent / "source-evidence"
    requests = {
        str(row["request_id"]): row
        for row in _iter_jsonl(source / "bundle/request-index.jsonl")
    }
    events_by_id = defaultdict(list)
    for row in _iter_jsonl(source / "non-success-events.jsonl"):
        events_by_id[str(row["request_id"])].append(row)
    raw_cache: dict[Path, dict[int, tuple[bytes, dict[str, Any]]]] = {}
    count = 0
    for binding in _iter_jsonl(source / "non-success-raw-line-ledger.jsonl"):
        count += 1
        path = parent / str(binding["raw_file_path"])
        if path not in raw_cache:
            raw_cache[path] = {
                ordinal: (line, row) for ordinal, line, row in _exact_file_rows(path)
            }
        ordinal = int(binding["raw_line_ordinal"])
        try:
            line, provider_row = raw_cache[path][ordinal]
        except KeyError as error:
            raise ValueError("bound provider raw ordinal absent") from error
        request_id = str(binding["request_id"])
        if (
            hashlib.sha256(line).hexdigest() != binding["raw_line_sha256"]
            or str(provider_row.get("custom_id")) != request_id
        ):
            raise ValueError("bound provider raw line drift")
        reparsed = parse_provider_row(provider_row, requests[request_id])
        matching_events = [
            row
            for row in events_by_id[request_id]
            if all(row.get(key) == value for key, value in reparsed.items())
        ]
        if not matching_events:
            raise ValueError("provider raw row does not reparse to bound event")
        for event in matching_events:
            _verify_self_hash(event, "event_sha256", "reparsed non-success event")
    if count != 89:
        raise ValueError("provider raw reparse census drift")


def _parent_binding(parent: Path) -> dict[str, Any]:
    _reject_descendant_symlinks(parent)
    loaded = load_frozen_post_campaign_analysis(parent)
    manifest = loaded["manifest"]
    inventory = _load_object(parent / "evidence-inventory.json")
    _verify_self_hash(inventory, "inventory_sha256", "parent inventory")
    _validate_parent_bundle(parent)
    _validate_parent_collections(parent)
    _validate_parent_raw_reparse(parent)
    relevant = {}
    for relative in (
        "proposals.jsonl",
        "strict-proposals.jsonl",
        "sampling-psus.jsonl",
        "blind-audit.jsonl",
        "audit-reveal.jsonl",
        "audit-plan.json",
        "source-evidence/bundle/units.jsonl",
        "source-evidence/bundle/workstation-bundle.json",
        "source-evidence/bundle/request-index.jsonl",
        "source-evidence/effective-events.jsonl",
    ):
        relevant[relative] = file_sha256(parent / relative)
    return _hashed(
        {
            "schema_version": PARENT_BINDING_SCHEMA,
            "parent_v1_root": str(parent.resolve()),
            "parent_v1_manifest_sha256": manifest["manifest_sha256"],
            "parent_v1_inventory_sha256": inventory["inventory_sha256"],
            "relevant_file_sha256": relevant,
            "parent_validation": {
                "v1_loader": True,
                "all_descendant_symlinks_rejected": True,
                "bundle_retained_files_checked_against_manifest": True,
                "continuation_recovery_collection_semantics_checked": True,
                "extra_event_rows_rejected": True,
                "recovery_membership_and_zero_success_reruns_checked": True,
                "all_89_non_success_raw_rows_reparsed": True,
            },
        },
        "parent_binding_sha256",
    )


def _unique_fine_majority(proposal: Mapping[str, Any]) -> str | None:
    if proposal["proposal_status"] != "complete":
        return None
    counts = Counter(map(str, proposal["fine_votes"]))
    ordered = counts.most_common()
    if not ordered or (len(ordered) > 1 and ordered[0][1] == ordered[1][1]):
        return None
    return ordered[0][0]


def _atom_eligibility(
    unit: Mapping[str, Any], proposal: Mapping[str, Any]
) -> dict[str, bool]:
    fine = _unique_fine_majority(proposal)
    physical = proposal["physical_votes"]
    uncertainty = (
        proposal["proposal_status"] != "complete"
        or proposal["broad_majority"] == "unresolved"
        or proposal["fine_agreement_pattern"] != "unanimous"
        or any(vote["confidence"] != "high" for vote in physical)
        or any(vote["boundary_concerns"] for vote in physical)
    )
    deterministic_terminal = (
        unit["assignment_route"] == "deterministic_terminal_serialization"
    )
    deterministic_control = unit["assignment_route"] == "deterministic_surface"
    return {
        "process_enrichment": (
            proposal["proposal_status"] == "complete"
            and fine == "active_task_work"
            and proposal["broad_majority"] == "process_bearing"
        ),
        "evaluation_commitment": deterministic_terminal
        or (proposal["proposal_status"] == "complete" and fine in EVALUATION_LABELS),
        "diversity": not deterministic_control,
        "uncertainty_missing": uncertainty,
        "uniform_reserve": True,
    }


def _mechanism_frame(parent: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    units = list(_iter_jsonl(parent / "source-evidence/bundle/units.jsonl"))
    proposals = {
        str(row["unit_id"]): row for row in _iter_jsonl(parent / "proposals.jsonl")
    }
    psus = list(_iter_jsonl(parent / "sampling-psus.jsonl"))
    workstation = _load_object(
        parent / "source-evidence/bundle/workstation-bundle.json"
    )
    documents = {str(row["response_id"]): row for row in workstation["documents"]}
    unit_by_id = {str(row["unit_id"]): row for row in units}
    rows = []
    census = {mechanism: Counter() for mechanism in MECHANISMS}
    frame = Counter()
    for psu in psus:
        staged_atoms: list[dict[str, Any]] = []
        psu_mechanisms = dict.fromkeys(MECHANISMS, False)
        for unit_id in map(str, psu["member_unit_ids"]):
            unit = unit_by_id[unit_id]
            proposal = proposals[unit_id]
            eligible = _atom_eligibility(unit, proposal)
            positions = int(unit["token_span"][1]) - int(unit["token_span"][0])
            staged_atoms.append(
                {
                    "unit_id": unit_id,
                    "token_span": unit["token_span"],
                    "position_count": positions,
                    "local_qualifying_mechanisms": [
                        mechanism for mechanism in MECHANISMS if eligible[mechanism]
                    ],
                }
            )
            frame["atoms"] += 1
            frame["positions"] += positions
            for mechanism, is_eligible in eligible.items():
                psu_mechanisms[mechanism] = psu_mechanisms[mechanism] or is_eligible
        frame["psus"] += 1
        for mechanism, is_eligible in psu_mechanisms.items():
            census[mechanism]["psus"] += int(is_eligible)
        atom_rows: list[dict[str, Any]] = []
        for atom in staged_atoms:
            expanded = [
                mechanism for mechanism in MECHANISMS if psu_mechanisms[mechanism]
            ]
            atom_rows.append({**atom, "eligible_mechanisms": expanded})
            for mechanism in expanded:
                census[mechanism]["atoms"] += 1
                census[mechanism]["positions"] += int(atom["position_count"])
        rows.append(
            {
                "schema_version": ELIGIBILITY_SCHEMA,
                "psu_id": psu["psu_id"],
                "response_id": psu["response_id"],
                "source_key": json.dumps(
                    documents[str(psu["response_id"])]["response_source"],
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "prompt_sha256": documents[str(psu["response_id"])]["prompt_sha256"],
                "response_relative_third": min(
                    2,
                    (
                        min(int(atom["token_span"][0]) for atom in atom_rows)
                        + max(int(atom["token_span"][1]) for atom in atom_rows)
                    )
                    * 3
                    // (
                        2
                        * int(
                            documents[str(psu["response_id"])]["tokenization"][
                                "token_count"
                            ]
                        )
                    ),
                ),
                "response_relative_third_definition": (
                    "PSU token-support midpoint divided by frozen response token_count"
                ),
                "fragment_only_weighting_psu": True,
                "adjacency_is_correlation_metadata_only": True,
                "correlation_run_id": psu["correlation_run_id"],
                "eligible_mechanisms": [
                    mechanism for mechanism in MECHANISMS if psu_mechanisms[mechanism]
                ],
                "atoms": atom_rows,
                "atom_count": len(atom_rows),
                "position_count": sum(row["position_count"] for row in atom_rows),
            }
        )
    observed_frame = dict(frame)
    observed = {mechanism: dict(values) for mechanism, values in census.items()}
    if observed_frame != EXPECTED_FRAME or observed != EXPECTED_ELIGIBILITY:
        raise ValueError(
            f"mechanism eligibility census drift: frame={observed_frame}, mechanisms={observed}"
        )
    return rows, {"frame": observed_frame, "mechanisms": observed}


def _hash_uniform(*parts: object) -> float:
    digest = canonical_sha256([str(part) for part in parts])
    return int(digest, 16) / float(1 << 256)


def _anchor_positions(unit: Mapping[str, Any], document: Mapping[str, Any]) -> set[int]:
    core_start, core_end = map(int, unit["core_character_span"])
    response = str(document["text"])
    text = response[core_start:core_end]
    character_spans = [
        (core_start + match.start(), core_start + match.end())
        for match in ANCHOR_RE.finditer(text)
    ]
    result = set()
    for token_index, token in enumerate(document["tokenization"]["tokens"]):
        _token_id, start, end = map(int, token)
        if any(
            start < anchor_end and end > anchor_start
            for anchor_start, anchor_end in character_spans
        ):
            result.add(token_index)
    first, last = map(int, unit["token_span"])
    if any(not first <= position < last for position in result):
        raise ValueError("observable syntax anchor escaped its frozen unit")
    return result


def _kernel_for_mechanism(
    row: Mapping[str, Any],
    mechanism: str,
    *,
    unit_by_id: Mapping[str, Mapping[str, Any]],
    document_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[int, float]:
    atom_kernels: list[dict[int, float]] = []
    for atom in row["atoms"]:
        if mechanism not in atom["eligible_mechanisms"]:
            continue
        unit = unit_by_id[str(atom["unit_id"])]
        first, last = map(int, atom["token_span"])
        positions = list(range(first, last))
        width = len(positions)
        if width <= 0:
            raise ValueError("mechanism atom has no positions")
        within: Counter[int] = Counter()
        if mechanism == "process_enrichment":
            anchors = _anchor_positions(unit, document_by_id[str(row["response_id"])])
            if anchors:
                for position in anchors:
                    within[position] += 0.50 / len(anchors)
                all_mass = 0.15
            else:
                all_mass = 0.65
            within[first] += 0.20
            within[last - 1] += 0.15
            for position in positions:
                within[position] += all_mass / width
        elif mechanism == "evaluation_commitment":
            within[first] += 0.30
            within[last - 1] += 0.30
            for position in positions:
                within[position] += 0.40 / width
        elif mechanism == "uncertainty_missing":
            within[first] += 0.25
            within[last - 1] += 0.25
            for position in positions:
                within[position] += 0.50 / width
        else:
            for position in positions:
                within[position] += 1.0 / width
        atom_kernels.append(dict(within))
    if not atom_kernels:
        return {}
    if mechanism == "uniform_reserve":
        widths = [len(kernel) for kernel in atom_kernels]
        total_width = sum(widths)
        atom_probabilities = [width / total_width for width in widths]
    else:
        atom_probabilities = [1.0 / len(atom_kernels)] * len(atom_kernels)
    combined: Counter[int] = Counter()
    for alpha, atom_kernel in zip(atom_probabilities, atom_kernels, strict=True):
        for position, beta in atom_kernel.items():
            combined[position] += alpha * beta
    return dict(combined)


def build_position_kernels_v2(
    frame_rows: Sequence[Mapping[str, Any]],
    unit_by_id: Mapping[str, Mapping[str, Any]],
    document_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, dict[int, float]]]:
    """Build exact group-to-atom-to-position kernels for every mechanism."""

    result: dict[str, dict[str, dict[int, float]]] = {}
    for row in frame_rows:
        psu_id = str(row["psu_id"])
        result[psu_id] = {
            mechanism: _kernel_for_mechanism(
                row,
                mechanism,
                unit_by_id=unit_by_id,
                document_by_id=document_by_id,
            )
            for mechanism in row["eligible_mechanisms"]
        }
        for mechanism, kernel in result[psu_id].items():
            if not kernel or not math.isclose(
                sum(kernel.values()), 1.0, rel_tol=0, abs_tol=1e-12
            ):
                raise ValueError(
                    f"mechanism position kernel drift: {psu_id}/{mechanism}"
                )
    return result


def build_hierarchical_group_bases_v2(
    eligibility: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, float]]:
    """Balance non-uniform mechanisms through their declared hierarchy."""

    bases: dict[str, dict[str, float]] = {mechanism: {} for mechanism in MECHANISMS}
    for mechanism in MECHANISMS:
        eligible = [
            row for row in eligibility if mechanism in row["eligible_mechanisms"]
        ]
        if mechanism == "uniform_reserve":
            for row in eligible:
                bases[mechanism][str(row["psu_id"])] = float(row["position_count"])
            continue
        dimensions = ["source_key", "prompt_sha256", "response_id"]
        if mechanism == "diversity":
            dimensions.append("response_relative_third")
        branch_values: dict[tuple[object, ...], set[object]] = defaultdict(set)
        leaf_counts: Counter[tuple[object, ...]] = Counter()
        for row in eligible:
            path = tuple(row[dimension] for dimension in dimensions)
            for depth in range(len(path)):
                branch_values[path[:depth]].add(path[depth])
            leaf_counts[path] += 1
        for row in eligible:
            path = tuple(row[dimension] for dimension in dimensions)
            base = 1.0
            for depth in range(len(path)):
                base /= len(branch_values[path[:depth]])
            base /= leaf_counts[path]
            bases[mechanism][str(row["psu_id"])] = base
        if not math.isclose(
            sum(bases[mechanism].values()), 1.0, rel_tol=0, abs_tol=1e-12
        ):
            raise ValueError(f"hierarchical base normalization drift: {mechanism}")
    return bases


def _route_position_probability(rate: float, conditional_mass: float) -> float:
    return -math.expm1(-rate * conditional_mass)


def _solve_rate(
    *,
    target: float,
    mechanism: str,
    kernels: Mapping[str, Mapping[str, Mapping[int, float]]],
    group_bases: Mapping[str, Mapping[str, float]],
    prior_survival: Mapping[tuple[str, int], float],
) -> float:
    pairs = [
        (
            mass * group_bases[mechanism][psu_id],
            prior_survival[(psu_id, position)],
        )
        for psu_id, by_mechanism in kernels.items()
        for position, mass in by_mechanism.get(mechanism, {}).items()
    ]
    masses = np.fromiter((pair[0] for pair in pairs), dtype=np.float64)
    priors = np.fromiter((pair[1] for pair in pairs), dtype=np.float64)

    def owner_mass(rate: float) -> float:
        return float(np.sum(-np.expm1(-rate * masses) * priors))

    low, high = 0.0, 1.0
    while owner_mass(high) < target:
        high *= 2.0
        if high > 1e8:
            raise ValueError(f"mechanism capacity cannot satisfy target: {mechanism}")
    for _ in range(100):
        midpoint = (low + high) / 2.0
        if owner_mass(midpoint) < target:
            low = midpoint
        else:
            high = midpoint
    rate = (low + high) / 2.0
    if not math.isclose(owner_mass(rate), target, rel_tol=0, abs_tol=1e-8):
        raise ValueError("Poisson mechanism rate solve drift")
    return rate


def _solve_frontier(
    kernels: Mapping[str, Mapping[str, Mapping[int, float]]],
    group_bases: Mapping[str, Mapping[str, float]],
    *,
    policy: str,
    budget: int,
) -> tuple[
    dict[str, float],
    dict[tuple[str, int], dict[str, float]],
    dict[str, dict[str, float]],
]:
    shares = SHARES[policy]
    positions = {
        (psu_id, position)
        for psu_id, by_mechanism in kernels.items()
        for kernel in by_mechanism.values()
        for position in kernel
    }
    prior_survival = dict.fromkeys(positions, 1.0)
    rates: dict[str, float] = {}
    owner: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)
    diagnostics: dict[str, dict[str, float]] = {}
    for mechanism in OWNERSHIP_ORDER:
        target = budget * shares[mechanism]
        rate = _solve_rate(
            target=target,
            mechanism=mechanism,
            kernels=kernels,
            group_bases=group_bases,
            prior_survival=prior_survival,
        )
        rates[mechanism] = rate
        raw_arrivals = 0.0
        route_unique = 0.0
        owner_unique = 0.0
        for psu_id, by_mechanism in kernels.items():
            kernel = by_mechanism.get(mechanism, {})
            if kernel:
                raw_arrivals += rate * group_bases[mechanism][psu_id]
            for position, mass in kernel.items():
                key = (psu_id, position)
                route_pi = _route_position_probability(
                    rate, mass * group_bases[mechanism][psu_id]
                )
                owned = route_pi * prior_survival[key]
                owner[key][mechanism] = owned
                prior_survival[key] *= 1.0 - route_pi
                route_unique += route_pi
                owner_unique += owned
        diagnostics[mechanism] = {
            "expected_raw_arrivals": raw_arrivals,
            "expected_route_unique_positions": route_unique,
            "expected_within_route_collisions": raw_arrivals - route_unique,
            "expected_first_owner_unique_positions": owner_unique,
            "expected_cross_route_collisions": route_unique - owner_unique,
        }
    expected_unique = sum(sum(values.values()) for values in owner.values())
    if not math.isclose(expected_unique, budget, rel_tol=0, abs_tol=1e-8):
        raise ValueError("expected unique-position budget drift")
    return rates, owner, diagnostics


def build_overlap_frontiers_v2(
    kernels: Mapping[str, Mapping[str, Mapping[int, float]]],
    group_bases: Mapping[str, Mapping[str, float]] | None = None,
    *,
    include_candidates: bool = True,
) -> tuple[
    list[dict[str, Any]],
    dict[tuple[str, int], list[dict[str, Any]]],
    dict[tuple[str, int], dict[str, dict[str, Any]]],
]:
    """Build all candidate-only overlap frontiers from fixed position kernels.

    This public pure seam is intentionally independent of filesystem artifacts.
    It returns frontier summaries, exact per-position design probabilities, and
    the solved mechanism rates used by deterministic realized-tier construction.
    """

    bases = group_bases or {
        mechanism: {
            psu_id: 1.0
            for psu_id, by_mechanism in kernels.items()
            if mechanism in by_mechanism
        }
        for mechanism in MECHANISMS
    }
    frontiers: list[dict[str, Any]] = []
    candidates: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    solutions: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    prior_rates: dict[str, dict[str, float]] = defaultdict(dict)
    for policy, shares in SHARES.items():
        for budget in BUDGETS:
            rates, owner, diagnostics = _solve_frontier(
                kernels, bases, policy=policy, budget=budget
            )
            for mechanism in MECHANISMS:
                if rates[mechanism] + 1e-12 < prior_rates[policy].get(mechanism, 0.0):
                    raise ValueError("mechanism Poisson rate is not budget-nested")
                prior_rates[policy][mechanism] = rates[mechanism]
            expected_unique = sum(sum(values.values()) for values in owner.values())
            probability_hasher = hashlib.sha256()
            minimum_marginal = 1.0
            maximum_marginal = 0.0
            for (psu_id, position), owner_values in sorted(owner.items()):
                route_pi = {
                    mechanism: _route_position_probability(
                        rates[mechanism],
                        kernels[psu_id].get(mechanism, {}).get(position, 0.0)
                        * bases[mechanism].get(psu_id, 0.0),
                    )
                    for mechanism in MECHANISMS
                }
                prior = 1.0
                full_owner = {}
                for mechanism in OWNERSHIP_ORDER:
                    full_owner[mechanism] = route_pi[mechanism] * prior
                    prior *= 1.0 - route_pi[mechanism]
                marginal = 1.0 - prior
                if not math.isclose(
                    marginal,
                    sum(owner_values.values()),
                    rel_tol=0,
                    abs_tol=1e-14,
                ):
                    raise ValueError("all-frame owner mass reconstruction drift")
                minimum_marginal = min(minimum_marginal, marginal)
                maximum_marginal = max(maximum_marginal, marginal)
                probability_hasher.update(
                    (
                        canonical_sha256(
                            {
                                "psu_id": psu_id,
                                "token_index": position,
                                "mechanism_position_inclusion_probabilities": (
                                    route_pi
                                ),
                                "first_owner_position_masses": full_owner,
                                "marginal_inclusion_probability": marginal,
                                "inverse_probability_weight": 1.0 / marginal,
                            }
                        )
                        + "\n"
                    ).encode()
                )
            frontiers.append(
                {
                    "schema_version": FRONTIER_SCHEMA,
                    "policy": policy,
                    "nominal_expected_unique_target_budget": budget,
                    "mechanism_plan_order": list(MECHANISMS),
                    "first_owner_precedence": list(OWNERSHIP_ORDER),
                    "shares": shares,
                    "poisson_rates": rates,
                    "mechanism_diagnostics": diagnostics,
                    "expected_unique_target_positions": expected_unique,
                    "all_frame_probability_stream_sha256": (
                        probability_hasher.hexdigest()
                    ),
                    "all_frame_positive_support_verified": minimum_marginal > 0,
                    "all_frame_minimum_marginal_inclusion_probability": (
                        minimum_marginal
                    ),
                    "all_frame_maximum_marginal_inclusion_probability": (
                        maximum_marginal
                    ),
                    "expected_raw_arrivals": sum(
                        row["expected_raw_arrivals"] for row in diagnostics.values()
                    ),
                    "expected_within_route_collisions": sum(
                        row["expected_within_route_collisions"]
                        for row in diagnostics.values()
                    ),
                    "expected_cross_route_collisions": sum(
                        row["expected_cross_route_collisions"]
                        for row in diagnostics.values()
                    ),
                    "candidate_design_status": "frozen",
                    "selected_for_tracing": False,
                    "trace_ready": False,
                    "trace_policy_selection_status": (
                        "pending_audit_and_resource_gate"
                    ),
                }
            )
            for key in owner if include_candidates else ():
                route_pi = {
                    mechanism: _route_position_probability(
                        rates[mechanism],
                        kernels[key[0]].get(mechanism, {}).get(key[1], 0.0)
                        * bases[mechanism].get(key[0], 0.0),
                    )
                    for mechanism in MECHANISMS
                }
                prior = 1.0
                full_owner = {}
                for mechanism in OWNERSHIP_ORDER:
                    full_owner[mechanism] = route_pi[mechanism] * prior
                    prior *= 1.0 - route_pi[mechanism]
                marginal = 1.0 - prior
                if not 0 < marginal <= 1:
                    raise ValueError("candidate marginal probability drift")
                candidates[key].append(
                    {
                        "policy": policy,
                        "nominal_expected_unique_target_budget": budget,
                        "mechanism_position_inclusion_probabilities": route_pi,
                        "first_owner_position_masses": full_owner,
                        "marginal_inclusion_probability": marginal,
                        "inverse_probability_weight": 1.0 / marginal,
                        "candidate_design_status": "frozen",
                        "candidate_tier_status": "frozen_candidate_only",
                        "selected_for_tracing": False,
                        "trace_ready": False,
                        "trace_policy_selection_status": (
                            "pending_audit_and_resource_gate"
                        ),
                    }
                )
            solutions[(policy, budget)] = {
                mechanism: {
                    "poisson_rate": rate,
                    "group_base_intensities_sha256": canonical_sha256(bases[mechanism]),
                }
                for mechanism, rate in rates.items()
            }
    return frontiers, candidates, solutions


def _poisson_psu_arrivals(
    *,
    policy: str,
    mechanism: str,
    psu_id: str,
    position_kernel: Mapping[int, float],
    maximum_mean: float,
) -> list[tuple[float, int]]:
    """Draw one fixed nested route/PSU stream, then choose a position per arrival."""

    positions = sorted(position_kernel)
    cumulative = []
    running = 0.0
    for position in positions:
        running += float(position_kernel[position])
        cumulative.append(running)
    if not positions or not math.isclose(running, 1.0, rel_tol=0, abs_tol=1e-12):
        raise ValueError("Poisson PSU arrival kernel is not normalized")
    cumulative[-1] = 1.0
    arrivals = []
    elapsed = 0.0
    arrival_index = 0
    while True:
        uniform = _hash_uniform(
            "coarse-sampling-v2-poisson-arrival",
            policy,
            mechanism,
            psu_id,
            arrival_index,
        )
        # SHA-256 divided by 2**256 is in [0, 1); the half-open adjustment
        # avoids log(0) without changing any nonzero digest.
        uniform = max(uniform, 1.0 / float(1 << 256))
        elapsed += -math.log(uniform)
        if elapsed > maximum_mean:
            return arrivals
        position_uniform = _hash_uniform(
            "coarse-sampling-v2-poisson-position",
            policy,
            mechanism,
            psu_id,
            arrival_index,
        )
        position_index = bisect.bisect_right(cumulative, position_uniform)
        position_index = min(position_index, len(positions) - 1)
        arrivals.append((elapsed, positions[position_index]))
        arrival_index += 1


def _iter_candidate_rows(
    kernels: Mapping[str, Mapping[str, Mapping[int, float]]],
    group_bases: Mapping[str, Mapping[str, float]],
    solutions: Mapping[tuple[str, int], Mapping[str, Mapping[str, Any]]],
) -> Iterable[dict[str, Any]]:
    all_positions = sorted(
        {
            (psu_id, position)
            for psu_id, by_mechanism in kernels.items()
            for kernel in by_mechanism.values()
            for position in kernel
        }
    )
    for psu_id, position in all_positions:
        designs = []
        for policy in SHARES:
            for budget in BUDGETS:
                route_pi = {
                    mechanism: _route_position_probability(
                        float(solutions[(policy, budget)][mechanism]["poisson_rate"]),
                        kernels[psu_id].get(mechanism, {}).get(position, 0.0)
                        * group_bases[mechanism].get(psu_id, 0.0),
                    )
                    for mechanism in MECHANISMS
                }
                prior = 1.0
                owner = {}
                for mechanism in OWNERSHIP_ORDER:
                    owner[mechanism] = route_pi[mechanism] * prior
                    prior *= 1.0 - route_pi[mechanism]
                marginal = 1.0 - prior
                if marginal <= 0:
                    raise ValueError("streamed candidate marginal is not positive")
                designs.append(
                    {
                        "policy": policy,
                        "nominal_expected_unique_target_budget": budget,
                        "mechanism_position_inclusion_probabilities": route_pi,
                        "first_owner_position_masses": owner,
                        "marginal_inclusion_probability": marginal,
                        "inverse_probability_weight": 1.0 / marginal,
                        "candidate_design_status": "frozen",
                        "candidate_tier_status": "frozen_candidate_only",
                        "selected_for_tracing": False,
                        "trace_ready": False,
                        "trace_policy_selection_status": (
                            "pending_audit_and_resource_gate"
                        ),
                    }
                )
        yield {
            "schema_version": CANDIDATE_SCHEMA,
            "psu_id": psu_id,
            "token_index": position,
            "designs": designs,
        }


def build_realized_tiers_v2(
    kernels: Mapping[str, Mapping[str, Mapping[int, float]]],
    group_bases: Mapping[str, Mapping[str, float]],
    solutions: Mapping[tuple[str, int], Mapping[str, Mapping[str, float]]],
    *,
    context_metadata: Mapping[tuple[str, int], Mapping[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Realize fixed nested per-mechanism Poisson streams and union positions.

    Multiple mechanisms may select the same position and each PSU may contribute
    multiple positions.  Target identity is mechanism-independent, so union and
    deduplication are exact and target sets are asserted nested by identity.
    """

    metadata = context_metadata or {}
    rows: list[dict[str, Any]] = []
    summary_rows = []
    targets_by_policy_budget: dict[tuple[str, int], set[str]] = {}
    for policy in SHARES:
        policy_budgets = sorted(
            budget
            for candidate_policy, budget in solutions
            if candidate_policy == policy
        )
        if policy_budgets != list(BUDGETS):
            raise ValueError(f"realized-tier solution budgets drift: {policy}")
        arrivals_by_budget: dict[int, dict[tuple[str, int], dict[str, int]]] = {
            budget: defaultdict(dict) for budget in policy_budgets
        }
        for psu_id, by_mechanism in kernels.items():
            for mechanism, kernel in by_mechanism.items():
                rates = {
                    budget: float(
                        solutions[(policy, budget)][mechanism]["poisson_rate"]
                    )
                    for budget in policy_budgets
                }
                maximum_rate = rates[policy_budgets[-1]]
                base = float(group_bases[mechanism][psu_id])
                arrivals = _poisson_psu_arrivals(
                    policy=policy,
                    mechanism=mechanism,
                    psu_id=psu_id,
                    position_kernel=kernel,
                    maximum_mean=maximum_rate * base,
                )
                arrival_times = [arrival[0] for arrival in arrivals]
                for budget in policy_budgets:
                    count = bisect.bisect_right(arrival_times, rates[budget] * base)
                    for _time, position in arrivals[:count]:
                        route_counts = arrivals_by_budget[budget][(psu_id, position)]
                        route_counts[mechanism] = route_counts.get(mechanism, 0) + 1
        for budget in policy_budgets:
            tier_arrivals = arrivals_by_budget[budget]
            target_ids = {
                f"pwcoarsetargetv2-{canonical_sha256({'psu_id': psu_id, 'token_index': position})[:32]}"
                for psu_id, position in tier_arrivals
            }
            targets_by_policy_budget[(policy, budget)] = target_ids
            raw_count = sum(
                sum(by_mechanism.values()) for by_mechanism in tier_arrivals.values()
            )
            route_unique_count = sum(
                sum(mechanism in values for values in tier_arrivals.values())
                for mechanism in MECHANISMS
            )
            per_mechanism = {}
            for mechanism in MECHANISMS:
                mechanism_raw = sum(
                    values.get(mechanism, 0) for values in tier_arrivals.values()
                )
                mechanism_route_unique = sum(
                    mechanism in values for values in tier_arrivals.values()
                )
                mechanism_first_owner = sum(
                    next(
                        (
                            candidate
                            for candidate in OWNERSHIP_ORDER
                            if candidate in values
                        ),
                        None,
                    )
                    == mechanism
                    for values in tier_arrivals.values()
                )
                expected_owner = budget * SHARES[policy][mechanism]
                per_mechanism[mechanism] = {
                    "realized_raw_arrivals": mechanism_raw,
                    "realized_route_unique_positions": mechanism_route_unique,
                    "realized_first_owner_unique_positions": mechanism_first_owner,
                    "expected_first_owner_unique_positions": expected_owner,
                    "first_owner_absolute_deviation": mechanism_first_owner
                    - expected_owner,
                    "first_owner_share_deviation": (
                        mechanism_first_owner / len(target_ids)
                        - SHARES[policy][mechanism]
                    ),
                }
            tier_context_counts = sorted(
                int(metadata[key]["rendered_total_context_token_count"])
                for key in tier_arrivals
                if key in metadata
            )
            if metadata and len(tier_context_counts) != len(target_ids):
                raise ValueError("realized target context metadata coverage drift")
            for (psu_id, position), by_mechanism in sorted(tier_arrivals.items()):
                selected_mechanisms = [
                    mechanism
                    for mechanism in OWNERSHIP_ORDER
                    if mechanism in by_mechanism
                ]
                target_id = f"pwcoarsetargetv2-{canonical_sha256({'psu_id': psu_id, 'token_index': position})[:32]}"
                context = dict(metadata.get((psu_id, position), {}))
                route_pi = {
                    mechanism: _route_position_probability(
                        float(solutions[(policy, budget)][mechanism]["poisson_rate"]),
                        kernels[psu_id].get(mechanism, {}).get(position, 0.0)
                        * float(group_bases[mechanism].get(psu_id, 0.0)),
                    )
                    for mechanism in MECHANISMS
                }
                prior = 1.0
                owner_masses = {}
                for mechanism in OWNERSHIP_ORDER:
                    owner_masses[mechanism] = route_pi[mechanism] * prior
                    prior *= 1.0 - route_pi[mechanism]
                marginal = 1.0 - prior
                rows.append(
                    {
                        "schema_version": REALIZED_SCHEMA,
                        "policy": policy,
                        "nominal_expected_unique_target_budget": budget,
                        "target_id": target_id,
                        "psu_id": psu_id,
                        "token_index": position,
                        "arrival_mechanisms": selected_mechanisms,
                        "first_owner_mechanism": selected_mechanisms[0],
                        "raw_arrival_count_by_mechanism": {
                            mechanism: by_mechanism[mechanism]
                            for mechanism in selected_mechanisms
                        },
                        "mechanism_position_inclusion_probabilities": route_pi,
                        "first_owner_position_masses": owner_masses,
                        "marginal_inclusion_probability": marginal,
                        "inverse_probability_weight": 1.0 / marginal,
                        **context,
                        "candidate_design_status": "frozen",
                        "candidate_tier_status": "frozen_candidate_only",
                        "selected_for_tracing": False,
                        "trace_ready": False,
                        "trace_policy_selection_status": (
                            "pending_audit_and_resource_gate"
                        ),
                    }
                )
            summary_rows.append(
                {
                    "policy": policy,
                    "nominal_expected_unique_target_budget": budget,
                    "realized_raw_arrivals": raw_count,
                    "realized_route_unique_positions": route_unique_count,
                    "realized_within_route_collisions": raw_count - route_unique_count,
                    "realized_cross_route_collisions": route_unique_count
                    - len(target_ids),
                    "realized_unique_target_positions": len(target_ids),
                    "per_mechanism": per_mechanism,
                    "resource_envelope": (
                        {
                            "within_measured_1268_envelope": sum(
                                value <= MEASURED_CONTEXT_ENVELOPE
                                for value in tier_context_counts
                            ),
                            "above_measured_1268_envelope": sum(
                                value > MEASURED_CONTEXT_ENVELOPE
                                for value in tier_context_counts
                            ),
                            "rendered_context_min": tier_context_counts[0],
                            "rendered_context_median": (
                                tier_context_counts[len(tier_context_counts) // 2]
                                if len(tier_context_counts) % 2
                                else (
                                    tier_context_counts[
                                        len(tier_context_counts) // 2 - 1
                                    ]
                                    + tier_context_counts[len(tier_context_counts) // 2]
                                )
                                / 2
                            ),
                            "rendered_context_p95_nearest_rank": tier_context_counts[
                                math.ceil(0.95 * len(tier_context_counts)) - 1
                            ],
                            "rendered_context_max": tier_context_counts[-1],
                            "resource_qualified": None,
                            "resource_qualification_status": (
                                "pending_strict_t5_receipt_binding"
                            ),
                        }
                        if tier_context_counts
                        else None
                    ),
                }
            )
        for earlier, later in pairwise(policy_budgets):
            if not targets_by_policy_budget[(policy, earlier)].issubset(
                targets_by_policy_budget[(policy, later)]
            ):
                raise ValueError(
                    f"realized target identity is not nested: {policy}/{earlier}/{later}"
                )
    return rows, {
        "schema_version": "adag.process-witness.coarse-realized-tier-summary.v2",
        "tiers": summary_rows,
        "target_identity_nesting_verified": True,
        "nested_budget_order": list(BUDGETS),
        "candidate_design_status": "frozen",
        "candidate_tier_status": "frozen_candidate_only",
        "selected_for_tracing": False,
        "trace_ready": False,
        "trace_policy_selection_status": "pending_audit_and_resource_gate",
    }


def _context_count_evidence(
    *, parent: Path, cohort_root: Path
) -> tuple[list[dict[str, Any]], dict[tuple[str, int], dict[str, Any]], dict[str, Any]]:
    cohort_manifest_path = cohort_root / "manifest.json"
    cohort_index_path = cohort_root / "index.jsonl"
    cohort_manifest = _load_object(cohort_manifest_path)
    if cohort_manifest.get("records") != 188 or file_sha256(
        cohort_index_path
    ) != cohort_manifest.get("index_sha256"):
        raise ValueError("context-count cohort manifest/index drift")
    index_rows = list(_iter_jsonl(cohort_index_path))
    if len(index_rows) != 188:
        raise ValueError("context-count cohort census drift")
    prefix_by_response: dict[str, int] = {}
    evidence = []
    for index_ordinal, index_row in enumerate(index_rows):
        record_path = cohort_root / str(index_row["record_path"])
        if file_sha256(record_path) != index_row["record_sha256"]:
            raise ValueError("context-count source record hash drift")
        record = _load_object(record_path)
        response_id = str(index_row["response_id"])
        if record.get("response_id") != response_id:
            raise ValueError("context-count source response identity drift")
        if index_row["trace_scope"] == "full_assistant_serialization":
            reproducibility_raw = str(record["generation_row"]["reproducibility_info"])
            reproducibility = json.loads(reproducibility_raw)
            prompt_ids = reproducibility["prompt_token_ids"]
            prefix_count = len(prompt_ids)
            source_kind = "generation_reproducibility_prompt_token_ids"
            prefix_ids_sha256 = canonical_sha256(prompt_ids)
            model = reproducibility["model"]
            model_revision = reproducibility["model_revision"]
            embedded_prompt_ids: list[int] | None = list(map(int, prompt_ids))
        else:
            token_identity = record["historical_dense_record"]["token_identity"]
            prefix_count = int(token_identity["assistant_prefix_token_count"])
            source_kind = "historical_dense_token_identity"
            prefix_ids_sha256 = token_identity["assistant_prefix_ids_sha256"]
            model = "Qwen/Qwen3-4B-Thinking-2507"
            model_revision = "768f209d9ea81521153ed38c47d515654e938aea"
            embedded_prompt_ids = None
        if response_id in prefix_by_response or prefix_count <= 0:
            raise ValueError("context-count response coverage drift")
        prefix_by_response[response_id] = prefix_count
        evidence.append(
            {
                "schema_version": "adag.process-witness.context-count-evidence.v2",
                "response_id": response_id,
                "source_kind": source_kind,
                "cohort_index_ordinal": index_ordinal,
                "cohort_record_path": index_row["record_path"],
                "cohort_record_sha256": index_row["record_sha256"],
                "assistant_prefix_token_count": prefix_count,
                "assistant_prefix_ids_sha256": prefix_ids_sha256,
                "assistant_prefix_token_ids": embedded_prompt_ids,
                "model": model,
                "model_revision": model_revision,
                "rendered_context_formula": (
                    "assistant_prefix_token_count + response token_index + 1"
                ),
            }
        )
    workstation = _load_object(
        parent / "source-evidence/bundle/workstation-bundle.json"
    )
    response_ids = {str(row["response_id"]) for row in workstation["documents"]}
    if set(prefix_by_response) != response_ids:
        raise ValueError("context-count evidence/workstation response coverage drift")
    metadata: dict[tuple[str, int], dict[str, Any]] = {}
    for psu in _iter_jsonl(parent / "sampling-psus.jsonl"):
        response_id = str(psu["response_id"])
        for atom in psu["atoms"]:
            first, last = map(int, atom["token_span"])
            for position in range(first, last):
                total = prefix_by_response[response_id] + position + 1
                metadata[(str(psu["psu_id"]), position)] = {
                    "response_id": response_id,
                    "unit_id": atom["unit_id"],
                    "rendered_total_context_token_count": total,
                    "within_measured_1268_envelope": (
                        total <= MEASURED_CONTEXT_ENVELOPE
                    ),
                    "resource_tier": (
                        "within_measured_1268_envelope"
                        if total <= MEASURED_CONTEXT_ENVELOPE
                        else "above_measured_1268_envelope"
                    ),
                    "resource_qualified": None,
                    "resource_qualification_status": (
                        "pending_strict_t5_receipt_binding"
                    ),
                }
    step0_path = Path(__file__).resolve().parents[3] / STEP0_MANIFEST_RELATIVE
    step0 = _load_object(step0_path)
    if (
        step0["model"]["model_id"] != "Qwen/Qwen3-4B-Thinking-2507"
        or step0["model"]["revision"] != "768f209d9ea81521153ed38c47d515654e938aea"
        or step0["t5"]["trace_family_id"] != "bonafide.t5-upstream-summed-top5.v1"
    ):
        raise ValueError("Step-0 model/T5 context-comparability evidence drift")
    source_binding = {
        "cohort_manifest_sha256": file_sha256(cohort_manifest_path),
        "cohort_index_sha256": file_sha256(cohort_index_path),
        "cohort_id": cohort_manifest["cohort_id"],
        "records": len(evidence),
        "measured_context_envelope": MEASURED_CONTEXT_ENVELOPE,
        "strict_t5_receipt_binding_status": "pending",
        "step0_manifest_sha256": file_sha256(step0_path),
        "step0_model_id": step0["model"]["model_id"],
        "step0_model_revision": step0["model"]["revision"],
        "step0_t5_trace_family_id": step0["t5"]["trace_family_id"],
        "step0_evidence_is_insufficient_for_resource_qualification": True,
        "long_candidates_excluded": False,
    }
    context_counts = [
        row["rendered_total_context_token_count"] for row in metadata.values()
    ]
    within_count = sum(
        row["within_measured_1268_envelope"] for row in metadata.values()
    )
    if (
        len(metadata) != 842_007
        or min(prefix_by_response.values()) != 175
        or max(prefix_by_response.values()) != 2631
        or min(context_counts) != 176
        or max(context_counts) != 10_767
        or within_count != 102_019
    ):
        raise ValueError("context-count literal census drift")
    source_binding["literal_census"] = EXPECTED_CONTEXT_CENSUS
    return evidence, metadata, source_binding


def _audit_supplement_pools(
    *, parent: Path, eligibility: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Enumerate diagnostic pools without selecting an audit supplement."""

    units = {
        str(row["unit_id"]): row
        for row in _iter_jsonl(parent / "source-evidence/bundle/units.jsonl")
    }
    proposals = {
        str(row["unit_id"]): row for row in _iter_jsonl(parent / "proposals.jsonl")
    }
    workstation = _load_object(
        parent / "source-evidence/bundle/workstation-bundle.json"
    )
    documents = {str(row["response_id"]): row for row in workstation["documents"]}
    rows = []
    census: Counter[str] = Counter(dict.fromkeys(AUDIT_SUPPLEMENT_POOLS, 0))
    for psu in eligibility:
        atom_units = [units[str(atom["unit_id"])] for atom in psu["atoms"]]
        atom_proposals = [proposals[str(atom["unit_id"])] for atom in psu["atoms"]]
        pool_ids = []
        if any(
            any(vote["confidence"] == "low" for vote in proposal["physical_votes"])
            for proposal in atom_proposals
        ):
            pool_ids.append("low_confidence")
        if any(
            proposal["proposal_status"] != "complete"
            or proposal["broad_majority"] in {"unresolved", "missing_proposal"}
            for proposal in atom_proposals
        ):
            pool_ids.append("unresolved_or_incomplete")
        if any(
            int(unit["token_span"][1]) - int(unit["token_span"][0]) >= 96
            for unit in atom_units
        ):
            pool_ids.append("long_unit_at_96_token_segmentation_cap")
        if any(
            int(unit["token_span"][1]) - int(unit["token_span"][0]) >= 49
            and bool(_anchor_positions(unit, documents[str(psu["response_id"])]))
            for unit in atom_units
        ):
            pool_ids.append("long_computation_syntax_ge_49_tokens")
        fine_labels = {
            fine
            for proposal in atom_proposals
            if (fine := _unique_fine_majority(proposal)) is not None
        }
        pool_ids.extend(
            f"rare_fine:{fine_label}"
            for fine_label in RARE_FINE_LABELS
            if fine_label in fine_labels
        )
        if pool_ids:
            for pool_id in pool_ids:
                census[pool_id] += 1
            rows.append(
                {
                    "schema_version": (
                        "adag.process-witness.coarse-audit-supplement-pool.v2"
                    ),
                    "psu_id": psu["psu_id"],
                    "response_id": psu["response_id"],
                    "pool_ids": sorted(pool_ids),
                }
            )
    plan = {
        "schema_version": AUDIT_PLAN_SCHEMA,
        "claim_boundary": CLAIM_BOUNDARY,
        "parent_v1_audit_preserved_unchanged": True,
        "parent_v1_probability_core_sha256": file_sha256(parent / "blind-audit.jsonl"),
        "parent_v1_reveal_sha256": file_sha256(parent / "audit-reveal.jsonl"),
        "parent_v1_plan_sha256": file_sha256(parent / "audit-plan.json"),
        "parent_diagnostic_pools_preserved": [
            "residual incomplete windows",
            "vote disagreement",
            "boundary concern",
            "fragment components",
        ],
        "supplement_pool_census": dict(sorted(census.items())),
        "supplement_union_psus": len(rows),
        "pool_rows_are_model_derived_and_not_reviewer_visible": True,
        "audit_sample_selected": False,
        "reveal_free_review_packet_status": "not_built_pending_predeclaration",
        "acceptance_thresholds_status": "pending_predeclaration",
        "estimator_status": "pending_predeclaration",
        "selected_for_tracing": False,
        "trace_ready": False,
    }
    return rows, plan


def _stream_hash_nested(
    values: Mapping[str, Mapping[str, Mapping[int, float]]]
    | Mapping[str, Mapping[str, float]],
) -> str:
    hasher = hashlib.sha256()
    for outer_key, inner in sorted(values.items()):
        for middle_key, leaf in sorted(inner.items()):
            if isinstance(leaf, Mapping):
                for final_key, value in sorted(leaf.items()):
                    payload = [outer_key, middle_key, final_key, value]
                    hasher.update((canonical_sha256(payload) + "\n").encode())
            else:
                payload = [outer_key, middle_key, leaf]
                hasher.update((canonical_sha256(payload) + "\n").encode())
    return hasher.hexdigest()


def _write_gzip_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with (
        path.open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as handle,
    ):
        for row in rows:
            handle.write(
                (
                    json.dumps(
                        dict(row),
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    + "\n"
                ).encode()
            )


def _iter_gzip_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"expected gzip JSONL object: {path}")
                yield value


def _compare_rows(
    observed: Iterable[Mapping[str, Any]],
    expected: Iterable[Mapping[str, Any]],
    label: str,
) -> None:
    for index, (actual, wanted) in enumerate(zip_longest(observed, expected), start=1):
        if (
            actual is None
            or wanted is None
            or canonical_sha256(actual) != canonical_sha256(wanted)
        ):
            raise ValueError(f"{label} recomputation drift at row {index}")


def _write_inventory(root: Path) -> dict[str, Any]:
    _reject_descendant_symlinks(root)
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in {"manifest.json", "evidence-inventory.json"}:
            continue
        files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    inventory = _hashed(
        {"schema_version": INVENTORY_SCHEMA, "files": files}, "inventory_sha256"
    )
    atomic_write_json(root / "evidence-inventory.json", inventory)
    return inventory


def _execution_source_revision(temporary: Path) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[3]
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
    tracked = subprocess.run(
        ["git", "ls-files"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    paths = [
        path
        for path in tracked
        if path in {"pyproject.toml", "uv.lock"}
        or path.startswith(("circuits/analysis/bonafide/", "circuits/labeling/"))
        or path in {"circuits/__init__.py", "circuits/analysis/__init__.py"}
        or path == "scripts/bonafide/build_process_witness_coarse_post_campaign_v2.py"
    ]
    required = {
        "circuits/analysis/bonafide/coarse_sampling_post_campaign_v2.py",
        "scripts/bonafide/build_process_witness_coarse_post_campaign_v2.py",
        "circuits/analysis/bonafide/coarse_sampling_post_campaign_v1.py",
        "circuits/analysis/bonafide/coarse_sampling_openai_batch_production_v1.py",
        "circuits/analysis/bonafide/canonical.py",
        "circuits/labeling/io.py",
        "pyproject.toml",
        "uv.lock",
    }
    if not required.issubset(paths):
        raise ValueError("v2 execution source is untracked or incomplete")
    files = []
    execution_root = temporary / "execution-source"
    for relative in paths:
        source = repo_root / relative
        committed = subprocess.run(
            ["git", "show", f"HEAD:{relative}"],
            cwd=repo_root,
            check=True,
            capture_output=True,
        ).stdout
        if source.read_bytes() != committed:
            raise ValueError(f"v2 execution source differs from HEAD: {relative}")
        destination = execution_root / relative
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
                "copied_path": destination.relative_to(temporary).as_posix(),
            }
        )
    return {
        "git_commit": commit,
        "git_tree": tree,
        "execution_source_subset_matches_head": True,
        "files": files,
        "transitive_source_scope": (
            "complete tracked circuits.analysis.bonafide and circuits.labeling "
            "packages plus v2 CLI, package initializers, pyproject, and uv.lock"
        ),
    }


def _validate_execution_source(root: Path, revision: Mapping[str, Any]) -> None:
    lowercase_hex_40 = re.compile(r"[0-9a-f]{40}").fullmatch
    if (
        not revision.get("execution_source_subset_matches_head")
        or lowercase_hex_40(str(revision.get("git_commit", ""))) is None
        or lowercase_hex_40(str(revision.get("git_tree", ""))) is None
    ):
        raise ValueError("v2 exact execution-source subset gate absent")
    seen = set()
    for binding in revision["files"]:
        relative = str(binding["copied_path"])
        source_relative = str(binding["path"])
        if (
            relative in seen
            or relative != f"execution-source/{source_relative}"
            or lowercase_hex_40(str(binding.get("git_blob", ""))) is None
        ):
            raise ValueError("v2 execution source copied-path collision")
        seen.add(relative)
        path = root / relative
        data = path.read_bytes()
        git_blob = hashlib.sha1(
            b"blob " + str(len(data)).encode() + b"\0" + data
        ).hexdigest()
        if (len(data), hashlib.sha256(data).hexdigest(), git_blob) != (
            int(binding["bytes"]),
            str(binding["sha256"]),
            str(binding["git_blob"]),
        ):
            raise ValueError(f"v2 copied execution source drift: {relative}")


def _readonly_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def _writable_tree(root: Path) -> None:
    root.chmod(0o755)
    for path in root.rglob("*"):
        if not path.is_symlink():
            path.chmod(0o755 if path.is_dir() else 0o644)


def _context_metadata_from_embedded(
    *, parent: Path, evidence: Sequence[Mapping[str, Any]]
) -> dict[tuple[str, int], dict[str, Any]]:
    prefix_by_response = {}
    for row in evidence:
        response_id = str(row["response_id"])
        prompt_ids = row.get("assistant_prefix_token_ids")
        if prompt_ids is not None and (
            len(prompt_ids) != row["assistant_prefix_token_count"]
            or canonical_sha256(prompt_ids) != row["assistant_prefix_ids_sha256"]
        ):
            raise ValueError("embedded context prompt-token evidence drift")
        prefix_by_response[response_id] = int(row["assistant_prefix_token_count"])
    if len(prefix_by_response) != 188:
        raise ValueError("embedded context response census drift")
    metadata = {}
    for psu in _iter_jsonl(parent / "sampling-psus.jsonl"):
        psu_id = str(psu["psu_id"])
        response_id = str(psu["response_id"])
        for atom in psu["atoms"]:
            for position in range(*map(int, atom["token_span"])):
                total = prefix_by_response[response_id] + position + 1
                metadata[(psu_id, position)] = {
                    "response_id": response_id,
                    "unit_id": atom["unit_id"],
                    "rendered_total_context_token_count": total,
                    "within_measured_1268_envelope": total <= MEASURED_CONTEXT_ENVELOPE,
                    "resource_tier": (
                        "within_measured_1268_envelope"
                        if total <= MEASURED_CONTEXT_ENVELOPE
                        else "above_measured_1268_envelope"
                    ),
                    "resource_qualified": None,
                    "resource_qualification_status": (
                        "pending_strict_t5_receipt_binding"
                    ),
                }
    if len(metadata) != 842_007:
        raise ValueError("embedded context position census drift")
    prefix_counts = [int(row["assistant_prefix_token_count"]) for row in evidence]
    context_counts = []
    for row in metadata.values():
        value = row["rendered_total_context_token_count"]
        if not isinstance(value, int):
            raise ValueError("embedded context token count type drift")
        context_counts.append(value)
    observed_census = {
        "responses": len(prefix_counts),
        "assistant_prefix_token_count_min": min(prefix_counts),
        "assistant_prefix_token_count_max": max(prefix_counts),
        "frame_positions": len(context_counts),
        "rendered_total_context_token_count_min": min(context_counts),
        "rendered_total_context_token_count_max": max(context_counts),
        "within_measured_1268_envelope": sum(
            value <= MEASURED_CONTEXT_ENVELOPE for value in context_counts
        ),
        "above_measured_1268_envelope": sum(
            value > MEASURED_CONTEXT_ENVELOPE for value in context_counts
        ),
    }
    if observed_census != EXPECTED_CONTEXT_CENSUS:
        raise ValueError("embedded context literal census drift")
    return metadata


def _validate_embedded_context_sources(
    *,
    root: Path,
    evidence: Sequence[Mapping[str, Any]],
    binding: Mapping[str, Any],
) -> None:
    manifest_path = root / "source-evidence/cohort-manifest.json"
    index_path = root / "source-evidence/cohort-index.jsonl"
    step0_path = root / "source-evidence/step0-manifest.json"
    manifest = _load_object(manifest_path)
    step0 = _load_object(step0_path)
    index = list(_iter_jsonl(index_path))
    if (
        manifest.get("records") != 188
        or len(index) != 188
        or file_sha256(index_path) != manifest.get("index_sha256")
        or file_sha256(manifest_path) != binding["cohort_manifest_sha256"]
        or file_sha256(index_path) != binding["cohort_index_sha256"]
        or file_sha256(step0_path) != binding["step0_manifest_sha256"]
        or step0["model"]["model_id"] != binding["step0_model_id"]
        or step0["model"]["revision"] != binding["step0_model_revision"]
        or step0["t5"]["trace_family_id"] != binding["step0_t5_trace_family_id"]
        or binding.get("step0_evidence_is_insufficient_for_resource_qualification")
        is not True
        or binding.get("strict_t5_receipt_binding_status") != "pending"
        or binding.get("literal_census") != EXPECTED_CONTEXT_CENSUS
        or len(evidence) != 188
    ):
        raise ValueError("embedded cohort manifest/index context binding drift")
    seen = set()
    for row in evidence:
        ordinal = int(row["cohort_index_ordinal"])
        if not 0 <= ordinal < len(index):
            raise ValueError("embedded context cohort ordinal drift")
        index_row = index[ordinal]
        response_id = str(row["response_id"])
        expected_kind = (
            "generation_reproducibility_prompt_token_ids"
            if index_row["trace_scope"] == "full_assistant_serialization"
            else "historical_dense_token_identity"
        )
        if (
            response_id in seen
            or response_id != index_row["response_id"]
            or row["cohort_record_path"] != index_row["record_path"]
            or row["cohort_record_sha256"] != index_row["record_sha256"]
            or row["source_kind"] != expected_kind
            or row["model"] != "Qwen/Qwen3-4B-Thinking-2507"
            or row["model_revision"] != "768f209d9ea81521153ed38c47d515654e938aea"
        ):
            raise ValueError("embedded context evidence/index identity drift")
        seen.add(response_id)
        prompt_ids = row.get("assistant_prefix_token_ids")
        if expected_kind.startswith("generation_"):
            if (
                not isinstance(prompt_ids, list)
                or len(prompt_ids) != row["assistant_prefix_token_count"]
                or canonical_sha256(prompt_ids) != row["assistant_prefix_ids_sha256"]
            ):
                raise ValueError("embedded generation prompt-ID evidence drift")
        elif (
            prompt_ids is not None
            or not isinstance(row["assistant_prefix_token_count"], int)
            or row["assistant_prefix_token_count"] <= 0
            or not isinstance(row["assistant_prefix_ids_sha256"], str)
            or len(row["assistant_prefix_ids_sha256"]) != 64
        ):
            raise ValueError("embedded historical prefix identity evidence drift")


def _solutions_from_frontiers(
    frontiers: Sequence[Mapping[str, Any]],
    bases: Mapping[str, Mapping[str, float]],
) -> dict[tuple[str, int], dict[str, dict[str, Any]]]:
    return {
        (str(row["policy"]), int(row["nominal_expected_unique_target_budget"])): {
            mechanism: {
                "poisson_rate": float(row["poisson_rates"][mechanism]),
                "group_base_intensities_sha256": canonical_sha256(bases[mechanism]),
            }
            for mechanism in MECHANISMS
        }
        for row in frontiers
    }


def build_post_campaign_sampling_v2(
    *,
    parent_v1_root: Path,
    destination: Path,
    cohort_root: Path = DEFAULT_COHORT_ROOT,
) -> dict[str, Any]:
    """Build and strictly reload an immutable additive v2 candidate artifact."""

    if parent_v1_root.is_symlink():
        raise ValueError("parent v1 root may not be a symlink")
    if destination.is_symlink():
        raise ValueError("v2 sampling destination may not be a symlink")
    if cohort_root.is_symlink():
        raise ValueError("cohort root may not be a symlink")
    parent = parent_v1_root.resolve()
    destination = destination.resolve()
    cohort_root = cohort_root.resolve()
    if destination.exists():
        raise FileExistsError(f"v2 sampling destination exists: {destination}")
    parent_binding = _parent_binding(parent)
    eligibility, eligibility_census = _mechanism_frame(parent)
    units = {
        str(row["unit_id"]): row
        for row in _iter_jsonl(parent / "source-evidence/bundle/units.jsonl")
    }
    workstation = _load_object(
        parent / "source-evidence/bundle/workstation-bundle.json"
    )
    documents = {str(row["response_id"]): row for row in workstation["documents"]}
    kernels = build_position_kernels_v2(eligibility, units, documents)
    bases = build_hierarchical_group_bases_v2(eligibility)
    frontiers, _discarded_candidates, solutions = build_overlap_frontiers_v2(
        kernels, bases, include_candidates=False
    )
    context_evidence, context_metadata, context_binding = _context_count_evidence(
        parent=parent, cohort_root=cohort_root
    )
    realized, realized_summary = build_realized_tiers_v2(
        kernels, bases, solutions, context_metadata=context_metadata
    )
    audit_pools, audit_plan = _audit_supplement_pools(
        parent=parent, eligibility=eligibility
    )
    temporary = destination.parent / f".{destination.name}.building-{os.getpid()}"
    if temporary.exists():
        raise FileExistsError(f"v2 sampling staging root exists: {temporary}")
    temporary.mkdir(parents=True)
    try:
        atomic_write_json(temporary / "parent-v1-binding.json", parent_binding)
        atomic_write_bytes(
            temporary / "mechanism-eligibility.jsonl", _jsonl_bytes(eligibility)
        )
        atomic_write_json(temporary / "eligibility-census.json", eligibility_census)
        design_contract = _expected_design_contract(
            kernel_stream_sha256=_stream_hash_nested(kernels),
            group_base_stream_sha256=_stream_hash_nested(bases),
        )
        atomic_write_json(temporary / "design-contract.json", design_contract)
        atomic_write_bytes(
            temporary / "expected-frontiers.jsonl", _jsonl_bytes(frontiers)
        )
        _write_gzip_jsonl(temporary / "realized-candidate-tiers.jsonl.gz", realized)
        atomic_write_json(temporary / "realized-tier-summary.json", realized_summary)
        _write_gzip_jsonl(
            temporary / "context-count-evidence.jsonl.gz", context_evidence
        )
        atomic_write_json(temporary / "context-source-binding.json", context_binding)
        source_evidence = temporary / "source-evidence"
        source_evidence.mkdir()
        shutil.copy2(
            cohort_root / "manifest.json", source_evidence / "cohort-manifest.json"
        )
        shutil.copy2(
            cohort_root / "index.jsonl", source_evidence / "cohort-index.jsonl"
        )
        shutil.copy2(
            Path(__file__).resolve().parents[3] / STEP0_MANIFEST_RELATIVE,
            source_evidence / "step0-manifest.json",
        )
        atomic_write_bytes(
            temporary / "audit-supplement-pools.jsonl", _jsonl_bytes(audit_pools)
        )
        atomic_write_json(temporary / "audit-supplement-plan.json", audit_plan)
        atomic_write_json(
            temporary / "conservative-exact-id-salvage-sensitivity-binding.json",
            {
                "lane_name": "conservative_exact_id_salvage_sensitivity",
                "parent_file": "proposals.jsonl",
                "parent_file_sha256": file_sha256(parent / "proposals.jsonl"),
                "unknown_id_mapping_performed": False,
                "unique_complement_promoted": False,
                "semantic_enrichment_requires_complete_three_vote_proposal": True,
            },
        )
        execution_source_revision = _execution_source_revision(temporary)
        inventory = _write_inventory(temporary)
        manifest = _hashed(
            {
                "schema_version": ANALYSIS_SCHEMA,
                "claim_boundary": CLAIM_BOUNDARY,
                "status": "frozen_candidate_designs_not_selected_for_tracing",
                "parent_v1_root": str(parent),
                "parent_v1_binding_sha256": parent_binding["parent_binding_sha256"],
                "inventory_sha256": inventory["inventory_sha256"],
                "execution_source_revision": execution_source_revision,
                "design_contract_sha256": file_sha256(
                    temporary / "design-contract.json"
                ),
                "expected_frontiers_sha256": file_sha256(
                    temporary / "expected-frontiers.jsonl"
                ),
                "realized_candidate_tiers_sha256": file_sha256(
                    temporary / "realized-candidate-tiers.jsonl.gz"
                ),
                "candidate_design_status": "frozen",
                "candidate_tier_status": "frozen_candidate_only",
                "selected_for_tracing": False,
                "trace_ready": False,
                "trace_policy_selection_status": ("pending_audit_and_resource_gate"),
                "publication_semantics": (
                    "renameat2(RENAME_NOREPLACE) when available; otherwise "
                    "manifest-last crash-detectable no-overwrite fallback"
                ),
                "network_calls_made": 0,
                "parent_v1_mutated": False,
            },
            "manifest_sha256",
        )
        atomic_write_json(temporary / "manifest.json", manifest)
        _readonly_tree(temporary)
        load_frozen_post_campaign_sampling_v2(temporary, parent_v1_root=parent)
        _publish_no_replace(temporary, destination)
        load_frozen_post_campaign_sampling_v2(destination, parent_v1_root=parent)
        return manifest
    except BaseException:
        if temporary.exists():
            _writable_tree(temporary)
            shutil.rmtree(temporary)
        raise


def load_frozen_post_campaign_sampling_v2(
    root: Path, *, parent_v1_root: Path | None = None
) -> dict[str, Any]:
    """Strict-load v2 and independently recompute every derived artifact."""

    if root.is_symlink():
        raise ValueError("v2 sampling root may not be a symlink")
    root = root.resolve()
    _reject_descendant_symlinks(root)
    manifest = _load_object(root / "manifest.json")
    _verify_self_hash(manifest, "manifest_sha256", "v2 sampling manifest")
    if manifest.get("schema_version") != ANALYSIS_SCHEMA:
        raise ValueError("v2 sampling schema drift")
    _validate_candidate_only_manifest(manifest)
    inventory = _load_object(root / "evidence-inventory.json")
    _verify_self_hash(inventory, "inventory_sha256", "v2 sampling inventory")
    if inventory.get("schema_version") != INVENTORY_SCHEMA:
        raise ValueError("v2 sampling inventory schema drift")
    inventory_files = inventory.get("files")
    if not isinstance(inventory_files, list):
        raise ValueError("v2 sampling inventory files drift")
    inventory_paths = [str(row.get("path")) for row in inventory_files]
    if len(inventory_paths) != len(set(inventory_paths)):
        raise ValueError("v2 sampling duplicate inventory path")
    if inventory["inventory_sha256"] != manifest["inventory_sha256"]:
        raise ValueError("v2 sampling manifest/inventory binding drift")
    _validate_execution_source(root, manifest["execution_source_revision"])
    declared = {str(row["path"]): row for row in inventory_files}
    observed = {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file()
        and path.relative_to(root).as_posix()
        not in {"manifest.json", "evidence-inventory.json"}
    }
    if set(declared) != set(observed):
        raise ValueError("v2 sampling inventory file membership drift")
    allowed_directories = {Path(".")}
    for relative in declared:
        path = Path(relative).parent
        while path != Path("."):
            allowed_directories.add(path)
            path = path.parent
    actual_directories = {
        path.relative_to(root) for path in root.rglob("*") if path.is_dir()
    }
    if actual_directories != allowed_directories - {Path(".")}:
        raise ValueError("v2 sampling uninventoryable directory drift")
    for relative, path in observed.items():
        binding = declared[relative]
        if (path.stat().st_size, file_sha256(path)) != (
            int(binding["bytes"]),
            str(binding["sha256"]),
        ):
            raise ValueError(f"v2 sampling inventoried file drift: {relative}")
    _validate_static_artifact_contract(root, manifest)
    if root.stat().st_mode & 0o777 != 0o555:
        raise ValueError("v2 sampling root mode drift")
    for path in root.rglob("*"):
        expected_mode = 0o555 if path.is_dir() else 0o444
        if path.stat().st_mode & 0o777 != expected_mode:
            raise ValueError(f"v2 sampling mode drift: {path.relative_to(root)}")

    requested_parent = (
        parent_v1_root
        if parent_v1_root is not None
        else Path(str(manifest["parent_v1_root"]))
    )
    if requested_parent.is_symlink():
        raise ValueError("parent v1 root may not be a symlink")
    parent = requested_parent.resolve()
    parent_binding = _parent_binding(parent)
    recorded_binding = _load_object(root / "parent-v1-binding.json")
    if (
        canonical_sha256(parent_binding) != canonical_sha256(recorded_binding)
        or parent_binding["parent_binding_sha256"]
        != manifest["parent_v1_binding_sha256"]
    ):
        raise ValueError("v2 sampling parent binding drift")
    eligibility, eligibility_census = _mechanism_frame(parent)
    _compare_rows(
        _iter_jsonl(root / "mechanism-eligibility.jsonl"),
        eligibility,
        "mechanism eligibility",
    )
    if _load_object(root / "eligibility-census.json") != eligibility_census:
        raise ValueError("v2 sampling eligibility census drift")
    units = {
        str(row["unit_id"]): row
        for row in _iter_jsonl(parent / "source-evidence/bundle/units.jsonl")
    }
    workstation = _load_object(
        parent / "source-evidence/bundle/workstation-bundle.json"
    )
    documents = {str(row["response_id"]): row for row in workstation["documents"]}
    kernels = build_position_kernels_v2(eligibility, units, documents)
    bases = build_hierarchical_group_bases_v2(eligibility)
    contract = _load_object(root / "design-contract.json")
    expected_contract = _expected_design_contract(
        kernel_stream_sha256=_stream_hash_nested(kernels),
        group_base_stream_sha256=_stream_hash_nested(bases),
    )
    if contract != expected_contract:
        raise ValueError("v2 sampling kernel/base contract drift")
    frontiers, _discarded, solutions = build_overlap_frontiers_v2(
        kernels, bases, include_candidates=False
    )
    _compare_rows(
        _iter_jsonl(root / "expected-frontiers.jsonl"),
        frontiers,
        "expected frontiers",
    )
    context_evidence = list(_iter_gzip_jsonl(root / "context-count-evidence.jsonl.gz"))
    context_metadata = _context_metadata_from_embedded(
        parent=parent, evidence=context_evidence
    )
    context_binding = _load_object(root / "context-source-binding.json")
    _validate_embedded_context_sources(
        root=root, evidence=context_evidence, binding=context_binding
    )
    realized, realized_summary = build_realized_tiers_v2(
        kernels, bases, solutions, context_metadata=context_metadata
    )
    _compare_rows(
        _iter_gzip_jsonl(root / "realized-candidate-tiers.jsonl.gz"),
        realized,
        "realized candidate tiers",
    )
    if _load_object(root / "realized-tier-summary.json") != realized_summary:
        raise ValueError("v2 sampling realized tier summary drift")
    audit_rows, audit_plan = _audit_supplement_pools(
        parent=parent, eligibility=eligibility
    )
    _compare_rows(
        _iter_jsonl(root / "audit-supplement-pools.jsonl"),
        audit_rows,
        "audit supplement pools",
    )
    if _load_object(root / "audit-supplement-plan.json") != audit_plan:
        raise ValueError("v2 sampling audit supplement plan drift")
    if set(audit_plan["supplement_pool_census"]) != set(AUDIT_SUPPLEMENT_POOLS):
        raise ValueError("v2 sampling required audit pool coverage drift")
    return {
        "manifest": manifest,
        "parent_binding": parent_binding,
        "eligibility_census": eligibility_census,
        "frontiers": frontiers,
        "realized_summary": realized_summary,
        "audit_plan": audit_plan,
        "context_binding": context_binding,
    }
