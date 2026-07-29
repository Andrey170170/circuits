"""Freeze the discovery-only rank-screen pool for the C2 utility pilot."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from scripts.bonafide.build_topk_manifest import save_manifest
from scripts.bonafide.execution_plan import sha256_file
from scripts.bonafide.runner import validate_target_selection

C2_SCREEN_POOL_SCHEMA = "bonafide-topk-c2-screen-pool/v1"
C2_RESPONSE_COUNT = 35
C2_PHASE_BIN_COUNT = 7
C2_ITEMS_PER_BIN = 2
C2_SCREEN_ITEM_COUNT = (
    C2_RESPONSE_COUNT * C2_PHASE_BIN_COUNT * C2_ITEMS_PER_BIN
)
DISCOVERY_ROLES = {"dense_discovery", "broad_discovery"}


def _stored_probability(item: Mapping[str, Any]) -> float:
    value = (
        item["target_selection"]
        .get("final_selection", {})
        .get("refinement_diagnostics", {})
        .get("probability")
    )
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"source item {item.get('artifact_id')} lacks probability")
    return float(value)


def _position(item: Mapping[str, Any]) -> int:
    positions = item["target_selection"]["response_token_positions"]
    if len(positions) != 1:
        raise ValueError("C2 screen items must select exactly one response position")
    return int(positions[0])


def _partition(items: list[Mapping[str, Any]]) -> list[list[Mapping[str, Any]]]:
    ordered = sorted(items, key=lambda item: (_position(item), item["artifact_id"]))
    bins: list[list[Mapping[str, Any]]] = [[] for _ in range(C2_PHASE_BIN_COUNT)]
    for index, item in enumerate(ordered):
        phase_bin = min(
            C2_PHASE_BIN_COUNT - 1,
            index * C2_PHASE_BIN_COUNT // len(ordered),
        )
        bins[phase_bin].append(item)
    if any(len(items_in_bin) < C2_ITEMS_PER_BIN for items_in_bin in bins):
        raise ValueError("one C2 response lacks two regular targets in every phase bin")
    return bins


def build_c2_screen_pool(
    source_manifest: Mapping[str, Any],
    *,
    source_manifest_path: Path,
    source_manifest_sha256: str,
) -> dict[str, Any]:
    """Choose two deterministic rank-screen candidates in seven response bins."""

    tokenizer = source_manifest.get("tokenizer")
    if not isinstance(tokenizer, Mapping):
        raise ValueError("C2 source manifest lacks tokenizer provenance")

    by_response: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    response_metadata: dict[str, tuple[str, str, str]] = {}
    seen_artifacts: set[str] = set()
    for wave in source_manifest.get("waves", []):
        role = wave.get("corpus_role")
        if role not in DISCOVERY_ROLES or wave.get("extreme_workload_isolation", False):
            continue
        for item in wave.get("items", []):
            validate_target_selection(item)
            artifact_id = item.get("artifact_id")
            if (
                not isinstance(artifact_id, str)
                or not artifact_id
                or artifact_id in seen_artifacts
            ):
                raise ValueError(
                    f"invalid or duplicate C2 source artifact: {artifact_id}"
                )
            seen_artifacts.add(artifact_id)
            example = item.get("example")
            if not isinstance(example, Mapping):
                raise ValueError(f"C2 item {artifact_id} lacks example metadata")
            response_id = example.get("example_id")
            family_id = example.get("base_question_id")
            if (
                not isinstance(response_id, str)
                or not response_id
                or not isinstance(family_id, str)
                or not family_id
            ):
                raise ValueError(
                    f"C2 item {artifact_id} lacks response/family identity"
                )
            prior = response_metadata.setdefault(
                response_id, (family_id, str(role), str(example.get("response", "")))
            )
            if prior != (family_id, role, example.get("response", "")):
                raise ValueError(f"C2 response metadata drift: {response_id}")
            _stored_probability(item)
            _position(item)
            by_response[response_id].append(item)

    if len(by_response) != C2_RESPONSE_COUNT:
        raise ValueError(
            f"C2 requires {C2_RESPONSE_COUNT} discovery responses, "
            f"got {len(by_response)}"
        )

    cases: list[dict[str, Any]] = []
    for response_index, response_id in enumerate(sorted(by_response)):
        family_id, role, _ = response_metadata[response_id]
        for phase_bin, items_in_bin in enumerate(_partition(by_response[response_id])):
            low_probability = min(
                items_in_bin,
                key=lambda item: (_stored_probability(item), item["artifact_id"]),
            )
            bin_center = (
                min(_position(item) for item in items_in_bin)
                + max(_position(item) for item in items_in_bin)
            ) / 2.0
            ordered_by_center = sorted(
                items_in_bin,
                key=lambda item: (
                    abs(_position(item) - bin_center),
                    item["artifact_id"],
                ),
            )
            temporal_center = next(
                item
                for item in ordered_by_center
                if item["artifact_id"] != low_probability["artifact_id"]
            )
            for slot, item in (
                ("minimum_probability", low_probability),
                ("temporal_center", temporal_center),
            ):
                cases.append(
                    {
                        "screen_case_id": (
                            f"c2-screen-r{response_index:02d}-p{phase_bin}-{slot}"
                        ),
                        "source_width1_artifact_id": item["artifact_id"],
                        "example_id": response_id,
                        "base_question_id": family_id,
                        "corpus_role": role,
                        "phase_bin": phase_bin,
                        "screen_slot": slot,
                        "target_response_position": _position(item),
                        "stored_observed_probability": _stored_probability(item),
                    }
                )

    if len(cases) != C2_SCREEN_ITEM_COUNT:
        raise AssertionError("C2 screen pool size drift")
    if len({case["source_width1_artifact_id"] for case in cases}) != len(cases):
        raise ValueError("C2 screen pool contains duplicate source artifacts")

    response_counts = Counter(case["example_id"] for case in cases)
    family_ids = {case["base_question_id"] for case in cases}
    role_counts = Counter(case["corpus_role"] for case in cases)
    return {
        "schema_version": C2_SCREEN_POOL_SCHEMA,
        "pool_id": "qwen3-4b-instruct-topk-c2-screen-pool-v1",
        "selection_evidence_only": True,
        "source_manifest_path": str(source_manifest_path.resolve()),
        "source_manifest_sha256": source_manifest_sha256,
        "selection_contract": {
            "discovery_only": True,
            "exclude_extreme_workload_isolation": True,
            "response_count": C2_RESPONSE_COUNT,
            "base_question_family_count": len(family_ids),
            "phase_bin_count": C2_PHASE_BIN_COUNT,
            "items_per_response_phase_bin": C2_ITEMS_PER_BIN,
            "screen_item_count": len(cases),
            "final_trace_count": C2_RESPONSE_COUNT * C2_PHASE_BIN_COUNT,
            "final_selection_rule": (
                "one_per_response_phase_bin_prefer_alternating_realized_width_"
                "then_temporal_center"
            ),
            "role_counts": dict(sorted(role_counts.items())),
            "response_counts_are_uniform": set(response_counts.values())
            == {C2_PHASE_BIN_COUNT * C2_ITEMS_PER_BIN},
        },
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.source_manifest.open(encoding="utf-8") as handle:
        source_manifest = json.load(handle)
    payload = build_c2_screen_pool(
        source_manifest,
        source_manifest_path=args.source_manifest,
        source_manifest_sha256=sha256_file(args.source_manifest),
    )
    save_manifest(args.output, payload)
    print(json.dumps(payload, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
