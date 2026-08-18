"""Frozen full-corpus coarse proposal-bank production protocol.

This module is deliberately network free.  It reuses the qualified v4
segmentation and prompt presentation, freezes every request before launch, and
packs immutable response-affinity Batch shards.  Its labels are sampling
proposals, never semantic truth.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from circuits.analysis.bonafide.canonical import canonical_sha256, file_sha256
from circuits.analysis.bonafide.coarse_sampling_annotation import (
    BOUNDARY_CONCERNS,
    COARSE_TAGS,
)
from circuits.analysis.bonafide.coarse_sampling_annotation_v3 import (
    ARM_ZERO_SHOT,
    _base_system_prompt,
)
from circuits.analysis.bonafide.coarse_sampling_annotation_v4 import (
    DECISION_SCHEMA_NAME,
    OPENAI_BATCH_ENDPOINT,
    SEGMENTATION_POLICY_ID,
    decision_json_schema_v4,
    render_v4_user_prompt,
    segment_document_v4,
)
from circuits.labeling.io import read_jsonl

CONFIG_SCHEMA = "adag.process-witness.coarse-production-config.v1"
UNIT_SCHEMA = "adag.process-witness.coarse-production-unit.v1"
WINDOW_SCHEMA = "adag.process-witness.coarse-production-window.v1"
REQUEST_SCHEMA = "adag.process-witness.coarse-production-request.v1"
BUNDLE_SCHEMA = "adag.process-witness.coarse-production-bundle.v1"
PROPOSAL_SCHEMA = "adag.process-witness.coarse-proposal.v1"
GROUP_SCHEMA = "adag.process-witness.coarse-sampling-group.v1"
MAXIMUM_FOCAL_UNITS = 6
REPLICAS = 3
REQUEST_IDENTITY_NAMESPACE_SHA256 = (
    "b673793a4cf2e9db254c04dac4772d6c8b9cc50de26d0f6f9a0ac36df29ba3a3"
)

BROAD_PROJECTION = {
    "active_task_work": "process_bearing",
    "evaluation_or_revision": "process_bearing",
    "intermediate_commitment": "process_bearing",
    "final_answer": "process_bearing",
    "other_semantic_text": "contextual",
    "surface_or_control": "contextual",
    "uncertain": "unresolved",
}


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


def load_production_config(path: Path) -> dict[str, Any]:
    value = _load_object(path)
    if value.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("unsupported coarse production config schema")
    if (
        value.get("request_identity_namespace_sha256")
        != REQUEST_IDENTITY_NAMESPACE_SHA256
    ):
        raise ValueError("coarse production request identity namespace drift")
    if tuple(value.get("tags", {})) != COARSE_TAGS:
        raise ValueError("coarse production ontology drift")
    if value.get("boundary_concerns") != list(BOUNDARY_CONCERNS):
        raise ValueError("coarse production boundary vocabulary drift")
    if value.get("segmentation") != {
        "policy_id": SEGMENTATION_POLICY_ID,
        "maximum_semantic_unit_tokens": 96,
    }:
        raise ValueError("coarse production segmentation drift")
    if value.get("request_protocol") != {
        "maximum_focal_units_per_window": MAXIMUM_FOCAL_UNITS,
        "replicas_per_window": REPLICAS,
        "arm": ARM_ZERO_SHOT,
        "target_markup": "target_only",
        "complete_task_prompt": True,
        "complete_response": True,
    }:
        raise ValueError("coarse production request shape drift")
    provider = value.get("provider", {})
    if provider != {
        "name": "openai",
        "model": "gpt-5.6-luna",
        "api_surface": "responses",
        "transport": "native_batch",
        "batch_endpoint": OPENAI_BATCH_ENDPOINT,
        "api_key_env": "OPENAI_API_KEY",
        "reasoning": {"effort": "medium"},
        "max_output_tokens": 16384,
        "store": False,
        "price_snapshot": "labeling/prices-2026-08-16-coarse-v2.json",
        "input_token_overhead_per_request": 4096,
    }:
        raise ValueError("coarse production provider contract drift")
    projection = value.get("broad_projection", {})
    flattened = {tag: broad for broad, tags in projection.items() for tag in tags}
    if flattened != BROAD_PROJECTION:
        raise ValueError("coarse production broad projection drift")
    sharding = value.get("sharding", {})
    if (
        sharding.get("policy_id") != "response-affinity-first-fit-decreasing-v1"
        or sharding.get("maximum_batch_input_bytes") != 180_000_000
        or sharding.get("official_batch_input_limit_bytes") != 200_000_000
        or sharding.get("official_batch_request_limit") != 50_000
    ):
        raise ValueError("coarse production sharding contract drift")
    if value.get("launch_gates") != {
        "fresh_run_specific_spend_authorization_required": True,
        "provider_batch_queued_input_token_limit_must_be_recorded": True,
        "maximum_failed_only_recovery_waves": 1,
        "fresh_recovery_authorization_required": True,
    }:
        raise ValueError("coarse production launch-gate contract drift")
    return value


def load_workstation_bundle(path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    if file_sha256(path) != config["source"]["workstation_bundle_sha256"]:
        raise ValueError("coarse production workstation hash drift")
    value = _load_object(path)
    if (
        value.get("annotation_set_id") != config["source"]["annotation_set_id"]
        or value.get("cohort_id") != config["source"]["cohort_id"]
        or len(value.get("documents", [])) != 188
    ):
        raise ValueError("coarse production workstation identity drift")
    return value


def production_units(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    units = segment_document_v4(document)
    return [
        {
            **unit,
            "schema_version": UNIT_SCHEMA,
            "proposal_status": (
                "provider_pending"
                if unit["assignment_route"] == "openai_pending"
                else "deterministic"
            ),
        }
        for unit in units
    ]


def response_windows(
    document: Mapping[str, Any],
    units: Sequence[Mapping[str, Any]],
    *,
    window_start: int,
) -> list[dict[str, Any]]:
    pending = [u for u in units if u["assignment_route"] == "openai_pending"]
    output = []
    for local_index, start in enumerate(range(0, len(pending), MAXIMUM_FOCAL_UNITS)):
        focal = pending[start : start + MAXIMUM_FOCAL_UNITS]
        identity = {
            "schema_version": WINDOW_SCHEMA,
            "window_index": window_start + local_index,
            "response_id": document["response_id"],
            "response_window_index": local_index,
            "focal_unit_ids": [u["unit_id"] for u in focal],
        }
        output.append(
            {
                **identity,
                "window_id": f"pwcoarseprodwinv1-{canonical_sha256(identity)[:32]}",
                "prompt_sha256": document["prompt_sha256"],
                "full_response_sha256": document["text_sha256"],
                "focal_sequence_indices": [u["sequence_index"] for u in focal],
            }
        )
    return output


def production_request(
    *,
    physical_index: int,
    replica_index: int,
    window: Mapping[str, Any],
    document: Mapping[str, Any],
    focal: Sequence[Mapping[str, Any]],
    all_units: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    primary_request_id: str | None,
) -> dict[str, Any]:
    user_prompt, markup_audit = render_v4_user_prompt(document, focal, all_units)
    system_prompt = _base_system_prompt(config) + (
        "\n\nNo labeled demonstrations are provided in this arm."
    )
    provider = config["provider"]
    body = {
        "model": provider["model"],
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_output_tokens": provider["max_output_tokens"],
        "reasoning": provider["reasoning"],
        "prompt_cache_key": (
            f"pwcp1-{canonical_sha256(system_prompt)[:16]}-"
            f"{str(document['text_sha256'])[:32]}"
        ),
        "store": False,
        "text": {
            "format": {
                "type": "json_schema",
                "name": DECISION_SCHEMA_NAME,
                "schema": decision_json_schema_v4(len(focal)),
                "strict": True,
            }
        },
    }
    body_sha = canonical_sha256(body)
    identity = {
        "schema_version": REQUEST_SCHEMA,
        "physical_index": physical_index,
        "window_id": window["window_id"],
        "window_index": window["window_index"],
        "response_id": window["response_id"],
        "replica_index": replica_index,
        "body_sha256": body_sha,
        "config_sha256": config["request_identity_namespace_sha256"],
    }
    request_id = f"pwcoarseprodv1-{canonical_sha256(identity)[:32]}"
    return {
        **identity,
        "request_id": request_id,
        "arm_id": ARM_ZERO_SHOT,
        "repeat_of_request_id": primary_request_id,
        "focal_unit_ids": list(window["focal_unit_ids"]),
        "prompt_sha256": window["prompt_sha256"],
        "full_response_sha256": window["full_response_sha256"],
        "markup_audit": markup_audit,
        "provider_body": body,
    }


def openai_batch_line(request: Mapping[str, Any]) -> dict[str, Any]:
    if canonical_sha256(request["provider_body"]) != request.get("body_sha256"):
        raise ValueError("coarse production provider body hash drift")
    return {
        "custom_id": request["request_id"],
        "method": "POST",
        "url": OPENAI_BATCH_ENDPOINT,
        "body": request["provider_body"],
    }


def compact_jsonl_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def assign_response_shards(
    blocks: Sequence[Mapping[str, Any]], maximum_bytes: int
) -> list[list[dict[str, Any]]]:
    """Deterministic response-affinity first-fit decreasing packing."""

    bins: list[dict[str, Any]] = []
    ordered = sorted(
        blocks, key=lambda item: (-int(item["bytes"]), int(item["response_index"]))
    )
    for raw in ordered:
        block = dict(raw)
        if int(block["bytes"]) >= maximum_bytes:
            raise ValueError("one response block exceeds the shard byte guard")
        destination = next(
            (
                value
                for value in bins
                if value["bytes"] + int(block["bytes"]) < maximum_bytes
            ),
            None,
        )
        if destination is None:
            destination = {"bytes": 0, "blocks": []}
            bins.append(destination)
        destination_blocks = destination["blocks"]
        destination_bytes = destination["bytes"]
        if not isinstance(destination_blocks, list) or not isinstance(
            destination_bytes, int
        ):
            raise ValueError("coarse production shard bin state drift")
        destination_blocks.append(block)
        destination["bytes"] = destination_bytes + int(block["bytes"])
    return [
        sorted(value["blocks"], key=lambda item: int(item["response_index"]))
        for value in bins
    ]


def broad_family(tag: str) -> str:
    try:
        return BROAD_PROJECTION[tag]
    except KeyError as error:
        raise ValueError(f"unknown coarse tag: {tag}") from error


def proposal_from_votes(
    unit: Mapping[str, Any], votes: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if unit["assignment_route"] != "openai_pending":
        tag = str(unit["deterministic_tag"])
        fine_votes = [tag, tag, tag]
        source = "deterministic_rule"
        physical = []
    else:
        if len(votes) != REPLICAS:
            raise ValueError("provider-pending atom requires exactly three votes")
        replica_indices = [int(v["replica_index"]) for v in votes]
        if sorted(replica_indices) != list(range(REPLICAS)):
            raise ValueError("provider-pending atom replica coverage drift")
        fine_votes = [
            str(v["tag"]) for v in sorted(votes, key=lambda v: v["replica_index"])
        ]
        if any(tag not in COARSE_TAGS for tag in fine_votes):
            raise ValueError("provider-pending atom contains unknown fine tag")
        source = "openai_replica_votes"
        physical = [dict(v) for v in sorted(votes, key=lambda v: v["replica_index"])]
    broad_votes = [broad_family(tag) for tag in fine_votes]
    counts = Counter(broad_votes)
    best = counts.most_common()
    broad_majority = (
        best[0][0] if len(best) == 1 or best[0][1] > best[1][1] else "unresolved"
    )
    identity = {
        "schema_version": PROPOSAL_SCHEMA,
        "unit_id": unit["unit_id"],
        "source": source,
        "fine_votes": fine_votes,
        "broad_votes": broad_votes,
    }
    return {
        **identity,
        "proposal_id": f"pwcoarseproposalv1-{canonical_sha256(identity)[:32]}",
        "response_id": unit["response_id"],
        "sequence_index": unit["sequence_index"],
        "assignment_route": unit["assignment_route"],
        "fine_vote_histogram": dict(sorted(Counter(fine_votes).items())),
        "broad_vote_histogram": dict(sorted(counts.items())),
        "broad_majority": broad_majority,
        "fine_unanimous": len(set(fine_votes)) == 1,
        "fine_one_one_one": len(set(fine_votes)) == 3,
        "physical_votes": physical,
        "fragment_of": unit.get("fragment_of"),
        "token_span": unit["token_span"],
        "core_character_span": unit["core_character_span"],
        "covering_character_span": unit["covering_character_span"],
    }


def sampling_groups(
    units: Sequence[Mapping[str, Any]], proposals: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Group fragments first; then adjacent equal-family atoms across surface gaps."""

    by_unit = {str(p["unit_id"]): p for p in proposals}
    by_response: dict[str, list[Mapping[str, Any]]] = {}
    for unit in units:
        by_response.setdefault(str(unit["response_id"]), []).append(unit)
    result = []
    for response_id, response_units in by_response.items():
        response_units = sorted(response_units, key=lambda u: int(u["sequence_index"]))
        atomic: list[list[Mapping[str, Any]]] = []
        cursor = 0
        while cursor < len(response_units):
            unit = response_units[cursor]
            fragment = unit.get("fragment_of")
            if fragment is None:
                atomic.append([unit])
                cursor += 1
                continue
            group = []
            while (
                cursor < len(response_units)
                and response_units[cursor].get("fragment_of") == fragment
            ):
                group.append(response_units[cursor])
                cursor += 1
            atomic.append(group)

        runs: list[list[Mapping[str, Any]]] = []
        current: list[Mapping[str, Any]] = []
        current_family: str | None = None
        pending_surface: list[Mapping[str, Any]] = []
        for group in atomic:
            semantic = [u for u in group if u["assignment_route"] == "openai_pending"]
            families = {by_unit[str(u["unit_id"])]["broad_majority"] for u in semantic}
            family = next(iter(families)) if len(families) == 1 else None
            if not semantic:
                pending_surface.extend(group)
                continue
            if current and family is not None and family == current_family:
                current.extend(pending_surface)
                current.extend(group)
            else:
                if current:
                    runs.append(current)
                if pending_surface:
                    runs.append(pending_surface)
                current = list(group)
                current_family = family
            pending_surface = []
        if current:
            current.extend(pending_surface)
            runs.append(current)
        elif pending_surface:
            runs.append(pending_surface)
        for run_index, members in enumerate(runs):
            member_proposals = [by_unit[str(u["unit_id"])] for u in members]
            selection_proposals = [
                by_unit[str(u["unit_id"])]
                for u in members
                if u["assignment_route"] != "deterministic_surface"
            ]
            families = {p["broad_majority"] for p in selection_proposals}
            family = (
                next(iter(families)) if len(families) == 1 else "mixed_or_unresolved"
            )
            identity = {
                "schema_version": GROUP_SCHEMA,
                "response_id": response_id,
                "run_index": run_index,
                "member_unit_ids": [u["unit_id"] for u in members],
            }
            result.append(
                {
                    **identity,
                    "group_id": f"pwcoarsegroupv1-{canonical_sha256(identity)[:32]}",
                    "broad_family": family,
                    "member_proposal_ids": [p["proposal_id"] for p in member_proposals],
                    "fragment_of_values": sorted(
                        {str(u["fragment_of"]) for u in members if u.get("fragment_of")}
                    ),
                    "token_span": [
                        members[0]["token_span"][0],
                        members[-1]["token_span"][1],
                    ],
                    "core_character_spans": [u["core_character_span"] for u in members],
                    "selection_note": "sample group first, then atom/position with recorded inclusion probabilities",
                }
            )
    return result


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(
                        f"JSONL row is not an object: {path}:{line_number}"
                    )
                yield value
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unreadable JSONL: {path}") from error


def _validate_bundle_topology(
    *,
    root: Path,
    manifest: Mapping[str, Any],
    config: Mapping[str, Any],
    shards: Sequence[Mapping[str, Any]],
    units: Sequence[Mapping[str, Any]],
    windows: Sequence[Mapping[str, Any]],
    requests: Sequence[Mapping[str, Any]],
) -> None:
    counts = manifest["counts"]
    expected = {
        "responses": 188,
        "units": 94_546,
        "provider_pending_units": 74_860,
        "deterministic_surface_units": 19_500,
        "deterministic_terminal_units": 186,
        "fragment_groups_over_96_tokens": 51,
        "windows": 12_557,
        "physical_requests": 37_671,
        "replica_requests": 25_114,
        "shards": len(shards),
    }
    if counts != expected:
        raise ValueError("coarse production exact census drift")
    if len(units) != counts["units"] or len({u["unit_id"] for u in units}) != len(
        units
    ):
        raise ValueError("coarse production unit coverage drift")
    by_response: dict[str, list[Mapping[str, Any]]] = {}
    for unit in units:
        by_response.setdefault(str(unit["response_id"]), []).append(unit)
    if len(by_response) != counts["responses"]:
        raise ValueError("coarse production response coverage drift")
    for response_units in by_response.values():
        ordered = sorted(response_units, key=lambda u: int(u["sequence_index"]))
        if [u["sequence_index"] for u in ordered] != list(range(len(ordered))):
            raise ValueError("coarse production response unit order drift")
        if [u["token_span"][0] for u in ordered[1:]] != [
            u["token_span"][1] for u in ordered[:-1]
        ]:
            raise ValueError("coarse production response token partition drift")
    pending_ids = [
        u["unit_id"] for u in units if u["assignment_route"] == "openai_pending"
    ]
    window_ids = [unit_id for window in windows for unit_id in window["focal_unit_ids"]]
    if (
        len(windows) != counts["windows"]
        or window_ids != pending_ids
        or any(
            not 1 <= len(window["focal_unit_ids"]) <= MAXIMUM_FOCAL_UNITS
            for window in windows
        )
    ):
        raise ValueError("coarse production window atom partition drift")
    window_by_id = {window["window_id"]: window for window in windows}
    if len(window_by_id) != len(windows):
        raise ValueError("coarse production window identity collision")
    if (
        len(requests) != counts["physical_requests"]
        or len({r["request_id"] for r in requests}) != len(requests)
        or [r["physical_index"] for r in requests] != list(range(len(requests)))
    ):
        raise ValueError("coarse production physical request coverage drift")
    by_window: dict[str, list[Mapping[str, Any]]] = {}
    for request in requests:
        by_window.setdefault(str(request["window_id"]), []).append(request)
    if set(by_window) != set(window_by_id):
        raise ValueError("coarse production request/window coverage drift")
    for window_id, group in by_window.items():
        group = sorted(group, key=lambda r: int(r["replica_index"]))
        primary = group[0]
        if (
            len(group) != REPLICAS
            or [r["replica_index"] for r in group] != list(range(REPLICAS))
            or [r["physical_index"] for r in group]
            != list(
                range(
                    int(primary["physical_index"]),
                    int(primary["physical_index"]) + REPLICAS,
                )
            )
            or any(r["body_sha256"] != primary["body_sha256"] for r in group)
            or primary["repeat_of_request_id"] is not None
            or any(
                r["repeat_of_request_id"] != primary["request_id"] for r in group[1:]
            )
            or primary["focal_unit_ids"] != window_by_id[window_id]["focal_unit_ids"]
        ):
            raise ValueError("coarse production replica topology drift")
    request_by_id = {request["request_id"]: request for request in requests}
    shard_ids = []
    response_shards: dict[str, str] = {}
    for shard in shards:
        path = root / shard["path"]
        observed = []
        previous_window = None
        previous_replica = None
        for line in _iter_jsonl(path):
            request_id = line.get("custom_id")
            request = request_by_id.get(request_id)
            if request is None or request["shard_id"] != shard["shard_id"]:
                raise ValueError("coarse production shard contains unknown request")
            if line.get("method") != "POST" or line.get("url") != OPENAI_BATCH_ENDPOINT:
                raise ValueError("coarse production Batch line method/endpoint drift")
            if canonical_sha256(line.get("body")) != request["body_sha256"]:
                raise ValueError("coarse production Batch line body drift")
            response = request["response_id"]
            prior = response_shards.setdefault(response, shard["shard_id"])
            if prior != shard["shard_id"]:
                raise ValueError("coarse production response affinity drift")
            if (
                previous_window == request["window_id"]
                and previous_replica is not None
                and request["replica_index"] != previous_replica + 1
            ):
                raise ValueError("coarse production replicas are not consecutive")
            previous_window = request["window_id"]
            previous_replica = request["replica_index"]
            observed.append(request_id)
        if (
            observed != shard["request_ids_in_order"]
            or len(observed) != shard["request_count"]
        ):
            raise ValueError("coarse production shard ordered coverage drift")
        shard_ids.extend(observed)
    if set(shard_ids) != set(request_by_id) or len(shard_ids) != len(request_by_id):
        raise ValueError("coarse production shard union drift")
    if any(
        request["config_sha256"] != config["request_identity_namespace_sha256"]
        for request in requests
    ):
        raise ValueError("coarse production request config binding drift")


def load_production_bundle(
    root: Path, *, load_units: bool = True, strict_topology: bool = True
) -> dict[str, Any]:
    manifest = _load_object(root / "manifest.json")
    _verify_self_hash(manifest, "manifest_sha256", "coarse production manifest")
    if (
        manifest.get("schema_version") != BUNDLE_SCHEMA
        or manifest.get("status") != "prepared_offline_no_provider_calls"
        or manifest.get("network_calls_made") != 0
    ):
        raise ValueError("coarse production bundle is not a frozen offline artifact")
    for binding in manifest.get("files", []):
        path = root / binding["path"]
        if (
            not path.is_file()
            or path.stat().st_size != binding["bytes"]
            or file_sha256(path) != binding["sha256"]
        ):
            raise ValueError(f"coarse production payload drift: {path}")
    config = load_production_config(root / "protocol-config.json")
    if file_sha256(root / "protocol-config.json") != manifest["config_sha256"]:
        raise ValueError("coarse production config binding drift")
    shards = _load_object(root / "shards.json")["shards"]
    if len(shards) != manifest["counts"]["shards"]:
        raise ValueError("coarse production shard census drift")
    for shard in shards:
        path = root / shard["path"]
        if path.stat().st_size >= config["sharding"]["maximum_batch_input_bytes"]:
            raise ValueError("coarse production shard violates internal byte guard")
        if shard["request_count"] >= config["sharding"]["official_batch_request_limit"]:
            raise ValueError("coarse production shard violates request limit")
    windows = read_jsonl(root / "windows.jsonl")
    request_index = read_jsonl(root / "request-index.jsonl")
    units = read_jsonl(root / "units.jsonl") if load_units or strict_topology else None
    if strict_topology:
        assert units is not None
        _validate_bundle_topology(
            root=root,
            manifest=manifest,
            config=config,
            shards=shards,
            units=units,
            windows=windows,
            requests=request_index,
        )
    return {
        "manifest": manifest,
        "config": config,
        "shards": shards,
        "windows": windows,
        "request_index": request_index,
        "units": units if load_units else None,
        "cost_plan": _load_object(root / "cost-plan.json"),
    }


def iter_shard_requests(root: Path, shard_id: str) -> Iterable[dict[str, Any]]:
    loaded = load_production_bundle(root, load_units=False, strict_topology=False)
    shard = next((s for s in loaded["shards"] if s["shard_id"] == shard_id), None)
    if shard is None:
        raise ValueError(f"unknown coarse production shard: {shard_id}")
    metadata = {
        row["request_id"]: row
        for row in loaded["request_index"]
        if row["shard_id"] == shard_id
    }
    seen = []
    for line in read_jsonl(root / shard["path"]):
        request_id = line.get("custom_id")
        if request_id not in metadata:
            raise ValueError("coarse production shard contains unknown custom_id")
        row = {**metadata[request_id], "provider_body": line["body"]}
        if canonical_sha256(row["provider_body"]) != row["body_sha256"]:
            raise ValueError("coarse production shard body hash drift")
        seen.append(request_id)
        yield row
    if seen != shard["request_ids_in_order"]:
        raise ValueError("coarse production shard request order drift")
