"""Build all independent-candidate pass-one manifests for a T5 corpus."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts.bonafide.build_topk_c0_bundle import _source_contract, _trace_family
from scripts.bonafide.execution_plan import sha256_file
from scripts.bonafide.runner import _sha256, validate_target_selection
from scripts.bonafide.topk_manifest import SCHEMA_VERSION, validate_topk_manifest
from scripts.bonafide.topk_rank_screen import SCHEMA_VERSION as RANK_SCREEN_SCHEMA

SELECTION_SCHEMA = "bonafide-t5-corpus-selection/v1"
BUNDLE_SCHEMA = "bonafide-t5-corpus-pass1-bundle/v1"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _source_index(
    source_manifest: Mapping[str, Any],
) -> tuple[dict[str, tuple[str, Mapping[str, Any]]], list[str]]:
    result: dict[str, tuple[str, Mapping[str, Any]]] = {}
    order: list[str] = []
    for wave in source_manifest.get("waves", []):
        role = wave.get("corpus_role")
        if not isinstance(role, str) or not role:
            raise ValueError("T5 source wave lacks corpus role")
        for item in wave.get("items", []):
            validate_target_selection(item)
            artifact_id = item.get("artifact_id")
            if (
                not isinstance(artifact_id, str)
                or not artifact_id
                or artifact_id in result
            ):
                raise ValueError(f"invalid or duplicate T5 source: {artifact_id}")
            result[artifact_id] = (role, item)
            order.append(artifact_id)
    if not result:
        raise ValueError("T5 source manifest contains no targets")
    return result, order


def resolve_cases(
    source_manifest: Mapping[str, Any],
    rank_screen: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Resolve every source target to exact top-five-plus-observed candidates."""

    if rank_screen.get("schema_version") != RANK_SCREEN_SCHEMA:
        raise ValueError("unsupported T5 rank-screen schema")
    source_items, source_order = _source_index(source_manifest)
    rank_index: dict[str, Mapping[str, Any]] = {}
    for result in rank_screen.get("results", []):
        if not isinstance(result, Mapping):
            raise TypeError("T5 rank-screen results must be objects")
        source_id = result.get("source_width1_artifact_id")
        if not isinstance(source_id, str) or not source_id or source_id in rank_index:
            raise ValueError(f"invalid or duplicate T5 rank result: {source_id}")
        rank_index[source_id] = result
    if set(rank_index) != set(source_items):
        missing = len(set(source_items) - set(rank_index))
        extra = len(set(rank_index) - set(source_items))
        raise ValueError(
            f"T5 rank screen is not complete: missing={missing}, extra={extra}"
        )

    cases: list[dict[str, Any]] = []
    for case_index, source_id in enumerate(source_order):
        role, source_item = source_items[source_id]
        rank = rank_index[source_id]
        selection = rank.get("candidate_selection")
        if not isinstance(selection, Mapping):
            raise TypeError(f"candidate selection must be an object: {source_id}")
        candidates = selection.get("candidates")
        if (
            not isinstance(candidates, list)
            or len(candidates) not in {5, 6}
            or rank.get("candidate_count") != len(candidates)
            or selection.get("policy_id") != "model_top5_plus_observed"
        ):
            raise ValueError(f"invalid candidate selection for T5 source {source_id}")
        token_ids = [candidate.get("token_id") for candidate in candidates]
        observed = source_item["target_selection"]["final_target_token_id"]
        position = source_item["target_selection"]["response_token_positions"][0]
        if (
            any(
                isinstance(token_id, bool) or not isinstance(token_id, int)
                for token_id in token_ids
            )
            or len(set(token_ids)) != len(token_ids)
            or token_ids[0] != observed
            or rank.get("example_id") != source_item["example"]["example_id"]
            or rank.get("corpus_role") != role
            or rank.get("target_response_position") != position
        ):
            raise ValueError(f"T5 rank/source provenance drift: {source_id}")
        cases.append(
            {
                "case_id": f"t5-case-{case_index:05d}-{source_id[-12:]}",
                "source_width1_artifact_id": source_id,
                "example_id": source_item["example"]["example_id"],
                "base_question_id": source_item["example"]["base_question_id"],
                "corpus_role": role,
                "target_response_position": position,
                "input_token_count": int(rank["input_token_count"]),
                "candidate_count": len(token_ids),
                "observed_token_rank": selection["observed_token_rank"],
                "candidate_token_ids": token_ids,
                "candidate_selection": copy.deepcopy(selection),
                "_source_item": source_item,
            }
        )
    return cases


def _balanced_shards(
    cases: Sequence[Mapping[str, Any]], *, max_items: int
) -> list[list[Mapping[str, Any]]]:
    if isinstance(max_items, bool) or max_items < 1:
        raise ValueError("pass-one max_items_per_wave must be positive")
    if not cases:
        return []
    shard_count = math.ceil(len(cases) / max_items)
    shards: list[list[Mapping[str, Any]]] = [[] for _ in range(shard_count)]
    weights = [0 for _ in range(shard_count)]
    for case in sorted(
        cases,
        key=lambda value: (
            -int(value["input_token_count"]),
            str(value["case_id"]),
        ),
    ):
        available = [
            index for index, shard in enumerate(shards) if len(shard) < max_items
        ]
        target = min(
            available,
            key=lambda index: (weights[index], len(shards[index]), index),
        )
        shards[target].append(case)
        weights[target] += int(case["input_token_count"])
    for shard in shards:
        shard.sort(key=lambda value: str(value["case_id"]))
    return shards


def build_pass1_manifests(
    cases: Sequence[Mapping[str, Any]],
    source_manifest: Mapping[str, Any],
    *,
    source_manifest_path: Path,
    source_manifest_sha256: str,
    rank_screen_path: Path,
    rank_screen_sha256: str,
    selection_path: Path,
    selection_sha256: str,
    cohort_id: str,
    max_items_per_wave: int,
) -> dict[str, dict[str, Any]]:
    source = _source_contract(
        source_manifest,
        source_manifest_path=source_manifest_path,
        source_manifest_sha256=source_manifest_sha256,
    )
    cohort = {
        "cohort_id": cohort_id,
        "selection_path": str(selection_path.resolve()),
        "selection_sha256": selection_sha256,
        "rank_screen_path": str(rank_screen_path.resolve()),
        "rank_screen_sha256": rank_screen_sha256,
        "case_count": len(cases),
    }
    roles = sorted({str(case["corpus_role"]) for case in cases})
    manifests: dict[str, dict[str, Any]] = {}
    for candidate_index in range(6):
        eligible = [case for case in cases if case["candidate_count"] > candidate_index]
        waves: list[dict[str, Any]] = []
        for role in roles:
            role_cases = [case for case in eligible if case["corpus_role"] == role]
            for shard_index, shard in enumerate(
                _balanced_shards(role_cases, max_items=max_items_per_wave)
            ):
                items = []
                for case in shard:
                    item = copy.deepcopy(case["_source_item"])
                    item["specified_candidate_token_id"] = case["candidate_token_ids"][
                        candidate_index
                    ]
                    items.append(item)
                waves.append(
                    {
                        "wave_id": (
                            f"t5-pass1-c{candidate_index}-{role}-{shard_index:03d}"
                        ),
                        "corpus_role": role,
                        "items": items,
                    }
                )
        label = f"independent-candidate-{candidate_index}"
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "phase": "matched_corpus",
            "trace_family": _trace_family(
                trace_family_id=f"bonafide.t5-corpus-v1.{label}",
                policy_id="specified_token",
                objective_id="raw_logit_sum",
            ),
            "source": source,
            "cohort": cohort,
            "waves": waves,
        }
        validate_topk_manifest(manifest)
        manifests[label] = manifest
    return manifests


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def build_bundle(
    *,
    profile_path: Path,
    source_manifest_path: Path,
    rank_screen_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    profile_path = profile_path.resolve()
    source_manifest_path = source_manifest_path.resolve()
    rank_screen_path = rank_screen_path.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"T5 pass-one bundle output exists: {output_dir}")
    profile = _load_json(profile_path)
    source = _load_json(source_manifest_path)
    rank = _load_json(rank_screen_path)
    source_sha256 = sha256_file(source_manifest_path)
    rank_sha256 = sha256_file(rank_screen_path)
    if rank.get("source_manifest_sha256") != source_sha256:
        raise ValueError("T5 rank-screen source hash drift")
    cases = resolve_cases(source, rank)
    cohort_id = str(profile["profile_id"])
    max_items = int(profile["execution"]["pass1_max_items_per_wave"])

    public_cases = [
        {key: value for key, value in case.items() if key != "_source_item"}
        for case in cases
    ]
    width_counts = dict(
        sorted(Counter(str(case["candidate_count"]) for case in cases).items())
    )
    role_counts = dict(
        sorted(Counter(str(case["corpus_role"]) for case in cases).items())
    )
    selection_name = "t5-corpus-selection.json"
    selection_final = output_dir / selection_name
    selection = {
        "schema_version": SELECTION_SCHEMA,
        "cohort_id": cohort_id,
        "source_manifest_path": str(source_manifest_path),
        "source_manifest_sha256": source_sha256,
        "rank_screen_path": str(rank_screen_path),
        "rank_screen_sha256": rank_sha256,
        "profile_path": str(profile_path),
        "profile_sha256": sha256_file(profile_path),
        "selection_contract": {
            "all_source_targets_included": True,
            "candidate_policy_id": "model_top5_plus_observed",
            "candidate_union_contract": "adag.bonafide.candidate-union.v1",
        },
        "counts": {
            "cases": len(cases),
            "roles": role_counts,
            "candidate_widths": width_counts,
            "expected_pass1_traces": sum(
                int(case["candidate_count"]) for case in cases
            ),
        },
        "cases": public_cases,
    }
    # JSON formatting is part of the immutable file hash used by manifests.
    selection_bytes = (
        json.dumps(selection, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    selection_sha256 = __import__("hashlib").sha256(selection_bytes).hexdigest()
    manifests = build_pass1_manifests(
        cases,
        source,
        source_manifest_path=source_manifest_path,
        source_manifest_sha256=source_sha256,
        rank_screen_path=rank_screen_path,
        rank_screen_sha256=rank_sha256,
        selection_path=selection_final,
        selection_sha256=selection_sha256,
        cohort_id=cohort_id,
        max_items_per_wave=max_items,
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    try:
        (staging / selection_name).write_bytes(selection_bytes)
        manifest_records: list[dict[str, Any]] = []
        tasks: list[dict[str, Any]] = []
        for candidate_index in range(6):
            label = f"independent-candidate-{candidate_index}"
            filename = f"t5-pass1-candidate-{candidate_index}.json"
            path = staging / filename
            _write_json(path, manifests[label])
            final_path = output_dir / filename
            record = {
                "candidate_index": candidate_index,
                "label": label,
                "path": str(final_path),
                "sha256": sha256_file(path),
                "canonical_sha256": _sha256(manifests[label]),
                "trace_family_id": manifests[label]["trace_family"]["trace_family_id"],
                "wave_count": len(manifests[label]["waves"]),
                "work_item_count": sum(
                    len(wave["items"]) for wave in manifests[label]["waves"]
                ),
            }
            manifest_records.append(record)
            for wave in manifests[label]["waves"]:
                tasks.append(
                    {
                        "task_index": len(tasks),
                        "candidate_index": candidate_index,
                        "manifest_path": str(final_path),
                        "manifest_sha256": record["sha256"],
                        "wave_id": wave["wave_id"],
                        "corpus_role": wave["corpus_role"],
                        "work_item_count": len(wave["items"]),
                    }
                )
        max_array_tasks = int(profile["execution"]["max_array_tasks"])
        if len(tasks) > max_array_tasks:
            raise ValueError(
                "T5 pass-one task count exceeds the frozen scheduler limit: "
                f"tasks={len(tasks)}, limit={max_array_tasks}"
            )
        bundle = {
            "schema_version": BUNDLE_SCHEMA,
            "cohort_id": cohort_id,
            "selection_path": str(selection_final),
            "selection_sha256": selection_sha256,
            "source_manifest_path": str(source_manifest_path),
            "source_manifest_sha256": source_sha256,
            "rank_screen_path": str(rank_screen_path),
            "rank_screen_sha256": rank_sha256,
            "counts": selection["counts"],
            "execution_profile": {
                "max_items_per_wave": max_items,
                "task_count": len(tasks),
                "max_array_tasks": max_array_tasks,
                "recommended_array_concurrency": int(
                    profile["execution"]["recommended_array_concurrency"]
                ),
            },
            "manifests": manifest_records,
            "tasks": tasks,
        }
        _write_json(staging / "t5-pass1-bundle.json", bundle)
        os.replace(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--rank-screen", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    bundle = build_bundle(
        profile_path=args.profile,
        source_manifest_path=args.source_manifest,
        rank_screen_path=args.rank_screen,
        output_dir=args.output_dir,
    )
    print(json.dumps(bundle["counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
