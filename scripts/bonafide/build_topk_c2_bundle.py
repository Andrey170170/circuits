"""Freeze the C2 cohort and independent-candidate tracing manifests."""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts.bonafide.build_topk_c0_bundle import (
    RANK_SCREEN_SCHEMA,
    _load_json,
    _source_contract,
    _trace_family,
)
from scripts.bonafide.build_topk_c2_screen_pool import (
    C2_PHASE_BIN_COUNT,
    C2_RESPONSE_COUNT,
    C2_SCREEN_POOL_SCHEMA,
)
from scripts.bonafide.build_topk_manifest import save_manifest
from scripts.bonafide.execution_plan import sha256_file
from scripts.bonafide.topk_manifest import SCHEMA_VERSION, validate_topk_manifest

C2_SELECTION_SCHEMA = "bonafide-topk-c2-cohort-selection/v1"
C2_BUNDLE_SCHEMA = "bonafide-topk-c2-launch-bundle/v1"
C2_CASE_COUNT = C2_RESPONSE_COUNT * C2_PHASE_BIN_COUNT
C2_MAX_ITEMS_PER_WAVE = 28


def _source_item_index(
    source_manifest: Mapping[str, Any],
) -> dict[str, tuple[str, Mapping[str, Any]]]:
    result: dict[str, tuple[str, Mapping[str, Any]]] = {}
    for wave in source_manifest.get("waves", []):
        role = wave.get("corpus_role")
        for item in wave.get("items", []):
            artifact_id = item.get("artifact_id")
            if not isinstance(artifact_id, str) or artifact_id in result:
                raise ValueError(f"invalid or duplicate C2 source: {artifact_id}")
            result[artifact_id] = (str(role), item)
    return result


def _rank_index(rank_screen: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rank_screen.get("results", []):
        if not isinstance(row, Mapping):
            raise ValueError("C2 rank results must be objects")
        artifact_id = row.get("source_width1_artifact_id")
        if not isinstance(artifact_id, str) or artifact_id in result:
            raise ValueError(f"invalid or duplicate C2 rank result: {artifact_id}")
        result[artifact_id] = row
    return result


def select_c2_cases(
    screen_pool: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    rank_screen: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Apply the frozen one-per-response/bin C2 selection rule."""

    if screen_pool.get("schema_version") != C2_SCREEN_POOL_SCHEMA:
        raise ValueError("unsupported C2 screen-pool schema")
    if rank_screen.get("schema_version") != RANK_SCREEN_SCHEMA:
        raise ValueError("unsupported C2 rank-screen schema")
    source_items = _source_item_index(source_manifest)
    rank_results = _rank_index(rank_screen)
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for case in screen_pool.get("cases", []):
        if not isinstance(case, Mapping):
            raise ValueError("C2 screen cases must be objects")
        response_id = case.get("example_id")
        phase_bin = case.get("phase_bin")
        if (
            not isinstance(response_id, str)
            or isinstance(phase_bin, bool)
            or not isinstance(phase_bin, int)
        ):
            raise ValueError("C2 screen case response/bin identity is invalid")
        grouped[(response_id, phase_bin)].append(case)

    responses = sorted({response_id for response_id, _ in grouped})
    expected_cells = {
        (response_id, phase_bin)
        for response_id in responses
        for phase_bin in range(C2_PHASE_BIN_COUNT)
    }
    if len(responses) != C2_RESPONSE_COUNT or set(grouped) != expected_cells:
        raise ValueError("C2 screen pool response/bin coverage drift")

    selected: list[dict[str, Any]] = []
    for response_index, response_id in enumerate(responses):
        for phase_bin in range(C2_PHASE_BIN_COUNT):
            pool_cases = grouped[(response_id, phase_bin)]
            if len(pool_cases) != 2:
                raise ValueError("C2 response/bin cells must have two screen cases")
            desired_width = 6 if (response_index + phase_bin) % 2 == 0 else 5
            eligible: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
            for pool_case in pool_cases:
                artifact_id = pool_case["source_width1_artifact_id"]
                rank = rank_results.get(artifact_id)
                if rank is None:
                    raise ValueError(f"C2 case lacks rank evidence: {artifact_id}")
                candidate_selection = rank.get("candidate_selection")
                candidates = (
                    candidate_selection.get("candidates")
                    if isinstance(candidate_selection, Mapping)
                    else None
                )
                if (
                    not isinstance(candidates, list)
                    or len(candidates) not in {5, 6}
                    or rank.get("candidate_count") != len(candidates)
                ):
                    raise ValueError(f"C2 candidate evidence is invalid: {artifact_id}")
                eligible.append((pool_case, rank))
            preferred = [
                pair
                for pair in eligible
                if pair[1]["candidate_count"] == desired_width
            ]
            candidates_to_rank = preferred or eligible
            pool_case, rank = min(
                candidates_to_rank,
                key=lambda pair: (
                    pair[0].get("screen_slot") != "temporal_center",
                    pair[0]["source_width1_artifact_id"],
                ),
            )
            artifact_id = pool_case["source_width1_artifact_id"]
            role, source_item = source_items[artifact_id]
            candidate_selection = rank["candidate_selection"]
            candidate_records = candidate_selection["candidates"]
            token_ids = [record.get("token_id") for record in candidate_records]
            if (
                role != pool_case.get("corpus_role")
                or source_item["example"]["example_id"] != response_id
                or source_item["example"]["base_question_id"]
                != pool_case.get("base_question_id")
                or source_item["target_selection"]["response_token_positions"][0]
                != pool_case.get("target_response_position")
                or rank.get("corpus_role") != role
                or token_ids[0]
                != source_item["target_selection"]["final_target_token_id"]
                or any(
                    isinstance(token_id, bool) or not isinstance(token_id, int)
                    for token_id in token_ids
                )
                or len(set(token_ids)) != len(token_ids)
            ):
                raise ValueError(f"C2 selected case provenance drift: {artifact_id}")
            selected.append(
                {
                    "case_id": f"c2-r{response_index:02d}-p{phase_bin}",
                    "source_width1_artifact_id": artifact_id,
                    "example_id": response_id,
                    "base_question_id": pool_case["base_question_id"],
                    "corpus_role": role,
                    "phase_bin": phase_bin,
                    "target_response_position": pool_case[
                        "target_response_position"
                    ],
                    "screen_slot": pool_case["screen_slot"],
                    "desired_candidate_count": desired_width,
                    "candidate_count": len(token_ids),
                    "observed_token_rank": candidate_selection[
                        "observed_token_rank"
                    ],
                    "candidate_token_ids": token_ids,
                    "input_token_count": rank["input_token_count"],
                    "_source_item": source_item,
                }
            )
    if len(selected) != C2_CASE_COUNT:
        raise AssertionError("C2 final case-count drift")
    return selected


def _chunks(values: Sequence[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    return [
        list(values[index : index + C2_MAX_ITEMS_PER_WAVE])
        for index in range(0, len(values), C2_MAX_ITEMS_PER_WAVE)
    ]


def build_c2_manifests(
    cases: Sequence[Mapping[str, Any]],
    source_manifest: Mapping[str, Any],
    *,
    selection_path: Path,
    selection_sha256: str,
    source_manifest_path: Path,
    source_manifest_sha256: str,
    rank_screen_path: Path,
    rank_screen_sha256: str,
) -> dict[str, dict[str, Any]]:
    """Build six independently executable specified-token C2 manifests."""

    if len(cases) != C2_CASE_COUNT:
        raise ValueError(f"C2 requires exactly {C2_CASE_COUNT} final cases")
    source_contract = _source_contract(
        source_manifest,
        source_manifest_path=source_manifest_path,
        source_manifest_sha256=source_manifest_sha256,
    )
    cohort_contract = {
        "cohort_id": "qwen3-4b-instruct-topk-c2-v1",
        "selection_path": str(selection_path.resolve()),
        "selection_sha256": selection_sha256,
        "rank_screen_path": str(rank_screen_path.resolve()),
        "rank_screen_sha256": rank_screen_sha256,
        "case_count": len(cases),
    }
    manifests: dict[str, dict[str, Any]] = {}
    for candidate_index in range(6):
        eligible = [
            case for case in cases if case["candidate_count"] > candidate_index
        ]
        waves: list[dict[str, Any]] = []
        for role in ("dense_discovery", "broad_discovery"):
            role_cases = sorted(
                (case for case in eligible if case["corpus_role"] == role),
                key=lambda case: (
                    case["example_id"],
                    case["phase_bin"],
                    case["source_width1_artifact_id"],
                ),
            )
            for shard_index, shard in enumerate(_chunks(role_cases)):
                items = []
                for case in shard:
                    item = copy.deepcopy(case["_source_item"])
                    item["specified_candidate_token_id"] = case[
                        "candidate_token_ids"
                    ][candidate_index]
                    items.append(item)
                waves.append(
                    {
                        "wave_id": (
                            f"c2-independent-candidate-{candidate_index}-"
                            f"{role.replace('_discovery', '')}-{shard_index:02d}"
                        ),
                        "corpus_role": role,
                        "items": items,
                    }
                )
        label = f"independent-candidate-{candidate_index}"
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "phase": "c2_scientific_utility",
            "trace_family": _trace_family(
                trace_family_id=f"bonafide.c2.{label}.v1",
                policy_id="specified_token",
                objective_id="raw_logit_sum",
            ),
            "source": source_contract,
            "cohort": cohort_contract,
            "waves": waves,
        }
        validate_topk_manifest(manifest)
        manifests[label] = manifest
    return manifests


def _public_cases(cases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in case.items() if key != "_source_item"}
        for case in cases
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen-pool", type=Path, required=True)
    parser.add_argument("--rank-screen", type=Path, required=True)
    parser.add_argument("--selection-output", type=Path, required=True)
    parser.add_argument("--manifest-output-dir", type=Path, required=True)
    parser.add_argument("--bundle-output", type=Path, required=True)
    args = parser.parse_args()

    screen_pool = _load_json(args.screen_pool)
    rank_screen = _load_json(args.rank_screen)
    source_manifest_path = Path(screen_pool["source_manifest_path"])
    source_manifest_sha256 = sha256_file(source_manifest_path)
    if (
        source_manifest_sha256 != screen_pool.get("source_manifest_sha256")
        or source_manifest_sha256 != rank_screen.get("source_manifest_sha256")
        or sha256_file(args.screen_pool)
        != rank_screen.get("selection_pool_sha256")
    ):
        raise ValueError("C2 source, pool, or rank-screen hash drift")
    source_manifest = _load_json(source_manifest_path)
    cases = select_c2_cases(screen_pool, source_manifest, rank_screen)
    public_cases = _public_cases(cases)
    selection = {
        "schema_version": C2_SELECTION_SCHEMA,
        "cohort_id": "qwen3-4b-instruct-topk-c2-v1",
        "source_manifest_path": str(source_manifest_path.resolve()),
        "source_manifest_sha256": source_manifest_sha256,
        "screen_pool_path": str(args.screen_pool.resolve()),
        "screen_pool_sha256": sha256_file(args.screen_pool),
        "rank_screen_path": str(args.rank_screen.resolve()),
        "rank_screen_sha256": sha256_file(args.rank_screen),
        "selection_contract": dict(screen_pool["selection_contract"]),
        "balance": {
            "case_count": len(cases),
            "response_count": len({case["example_id"] for case in cases}),
            "family_count": len({case["base_question_id"] for case in cases}),
            "role_counts": dict(
                sorted(Counter(case["corpus_role"] for case in cases).items())
            ),
            "candidate_count_counts": dict(
                sorted(
                    Counter(
                        str(case["candidate_count"]) for case in cases
                    ).items()
                )
            ),
        },
        "cases": public_cases,
    }
    save_manifest(args.selection_output, selection)
    selection_sha256 = sha256_file(args.selection_output)
    manifests = build_c2_manifests(
        cases,
        source_manifest,
        selection_path=args.selection_output,
        selection_sha256=selection_sha256,
        source_manifest_path=source_manifest_path,
        source_manifest_sha256=source_manifest_sha256,
        rank_screen_path=args.rank_screen,
        rank_screen_sha256=sha256_file(args.rank_screen),
    )
    manifest_records = []
    for label, manifest in manifests.items():
        path = args.manifest_output_dir / (
            f"qwen3_4b_instruct_topk_c2_{label.replace('-', '_')}_v1.json"
        )
        save_manifest(path, manifest)
        manifest_records.append(
            {
                "label": label,
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "trace_family_id": manifest["trace_family"]["trace_family_id"],
                "waves": [
                    {
                        "wave_id": wave["wave_id"],
                        "work_item_count": len(wave["items"]),
                    }
                    for wave in manifest["waves"]
                ],
                "work_item_count": sum(
                    len(wave["items"]) for wave in manifest["waves"]
                ),
            }
        )
    bundle = {
        "schema_version": C2_BUNDLE_SCHEMA,
        "cohort_id": selection["cohort_id"],
        "selection_path": str(args.selection_output.resolve()),
        "selection_sha256": selection_sha256,
        "rank_screen_path": str(args.rank_screen.resolve()),
        "rank_screen_sha256": sha256_file(args.rank_screen),
        "case_count": len(cases),
        "expected_trace_count": sum(
            record["work_item_count"] for record in manifest_records
        ),
        "balance": selection["balance"],
        "manifests": manifest_records,
    }
    save_manifest(args.bundle_output, bundle)
    print(json.dumps(bundle, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
