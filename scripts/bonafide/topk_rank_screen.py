"""Discovery-only observed-rank screening for top-five-plus-observed probes."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from circuits.tracing.candidates import select_candidate_logits
from circuits.tracing.trace import (
    _teacher_forced_position_logits,
    prepare_teacher_forced_input,
)

from scripts.bonafide.execution_plan import sha256_file
from scripts.bonafide.runner import (
    _load_model_and_tokenizer,
    collect_code_revision,
    collect_runtime_environment,
    load_json,
    validate_run_config,
    validate_target_selection,
)

SCHEMA_VERSION = "bonafide-topk-rank-screen/v1"
SELECTION_RULE = "lowest_stored_observed_probability_then_artifact_id"
EXACT_POOL_SELECTION_RULE = "exact_frozen_selection_pool_order"


def select_rank_screen_items(
    source_manifest: Mapping[str, Any], *, max_items: int
) -> list[dict[str, Any]]:
    """Select low-probability discovery targets without using holdout data."""

    if isinstance(max_items, bool) or max_items < 1:
        raise ValueError("max_items must be a positive integer")
    candidates: list[tuple[float, str, dict[str, Any]]] = []
    for wave in source_manifest.get("waves", []):
        corpus_role = wave.get("corpus_role")
        if not isinstance(corpus_role, str) or corpus_role.endswith(
            "confirmatory_holdout"
        ):
            continue
        for raw_item in wave.get("items", []):
            item = dict(raw_item)
            validate_target_selection(item)
            positions = item["target_selection"]["response_token_positions"]
            if len(positions) != 1:
                continue
            probability = (
                item["target_selection"]
                .get("final_selection", {})
                .get("refinement_diagnostics", {})
                .get("probability")
            )
            artifact_id = item.get("artifact_id")
            if (
                isinstance(probability, bool)
                or not isinstance(probability, (int, float))
                or not isinstance(artifact_id, str)
                or not artifact_id
            ):
                continue
            candidates.append((float(probability), artifact_id, item))
    candidates.sort(key=lambda row: (row[0], row[1]))
    selected = [item for _, _, item in candidates[:max_items]]
    if not selected:
        raise ValueError("rank screen selection produced no discovery targets")
    return selected


def select_exact_rank_screen_items(
    source_manifest: Mapping[str, Any],
    selection_pool: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Resolve a frozen discovery-only screen pool against the source manifest."""

    from scripts.bonafide.build_topk_c2_screen_pool import C2_SCREEN_POOL_SCHEMA

    if selection_pool.get("schema_version") != C2_SCREEN_POOL_SCHEMA:
        raise ValueError("unsupported exact rank-screen selection pool")
    raw_cases = selection_pool.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("exact rank-screen selection pool has no cases")
    source_items: dict[str, tuple[str, dict[str, Any]]] = {}
    for wave in source_manifest.get("waves", []):
        role = wave.get("corpus_role")
        if not isinstance(role, str) or role.endswith("confirmatory_holdout"):
            continue
        if wave.get("extreme_workload_isolation", False):
            continue
        for raw_item in wave.get("items", []):
            item = dict(raw_item)
            artifact_id = item.get("artifact_id")
            if (
                not isinstance(artifact_id, str)
                or not artifact_id
                or artifact_id in source_items
            ):
                raise ValueError(
                    f"invalid or duplicate rank-screen source: {artifact_id}"
                )
            source_items[artifact_id] = (role, item)

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for case in raw_cases:
        if not isinstance(case, Mapping):
            raise ValueError("exact rank-screen cases must be objects")
        artifact_id = case.get("source_width1_artifact_id")
        if not isinstance(artifact_id, str) or not artifact_id or artifact_id in seen:
            raise ValueError(f"invalid or duplicate exact screen case: {artifact_id}")
        seen.add(artifact_id)
        pair = source_items.get(artifact_id)
        if pair is None:
            raise ValueError(
                f"exact screen source is absent or ineligible: {artifact_id}"
            )
        role, item = pair
        if (
            case.get("corpus_role") != role
            or case.get("example_id") != item["example"]["example_id"]
            or case.get("base_question_id") != item["example"]["base_question_id"]
            or case.get("target_response_position")
            != item["target_selection"]["response_token_positions"][0]
        ):
            raise ValueError(f"exact screen case provenance drift: {artifact_id}")
        validate_target_selection(item)
        selected.append(item)
    return selected


def screen_candidate_ranks(
    model,
    tokenizer,
    items: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Measure observed rank and the realized union width without tracing graphs."""

    results: list[dict[str, Any]] = []
    for item in items:
        positions = item["target_selection"]["response_token_positions"]
        if len(positions) != 1:
            raise ValueError("rank screening requires one response target position")
        position = int(positions[0])
        example = item["example"]
        prepared = prepare_teacher_forced_input(
            tokenizer,
            example["prompt"],
            example["response"],
            [position],
        )
        expected_observed_token_id = int(
            item["target_selection"]["final_target_token_id"]
        )
        if prepared.target_token_ids != [expected_observed_token_id]:
            raise ValueError(
                "rank-screen tokenizer target disagrees with frozen source item"
            )
        position_logits = _teacher_forced_position_logits(model, prepared)
        selection = select_candidate_logits(
            position_logits,
            observed_token_id=expected_observed_token_id,
            policy_id="model_top5_plus_observed",
            candidate_count=6,
            decode_token=lambda token_id: tokenizer.decode([token_id]),
        )
        result = {
            "source_width1_artifact_id": item["artifact_id"],
            "example_id": example["example_id"],
            "corpus_role": item["target_selection"]["final_selection"]["corpus_role"],
            "target_response_position": position,
            "input_token_count": len(prepared.input_ids),
            "candidate_count": len(selection.candidates),
            "candidate_selection": selection.to_dict(),
        }
        results.append(result)
        print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return results


def save_rank_screen(path: Path, payload: Mapping[str, Any]) -> Path:
    """Create rank-screen selection evidence atomically without overwriting."""

    if path.exists():
        raise FileExistsError(f"rank-screen output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    selection_group = parser.add_mutually_exclusive_group()
    selection_group.add_argument("--max-items", type=int)
    selection_group.add_argument("--selection-pool", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = load_json(args.config)
    validate_run_config(config)
    source_manifest = load_json(args.source_manifest)
    tokenizer_provenance = source_manifest.get("tokenizer")
    if not isinstance(tokenizer_provenance, Mapping):
        raise ValueError("rank-screen source manifest lacks tokenizer provenance")
    if (
        tokenizer_provenance.get("model_id") != config["model"]["model_id"]
        or tokenizer_provenance.get("revision") != config["model"]["revision"]
    ):
        raise ValueError("rank-screen source model provenance disagrees with config")

    repo_root = Path(__file__).resolve().parents[2]
    code_revision = collect_code_revision(repo_root)
    if code_revision["git_dirty"]:
        raise ValueError("rank-screen execution requires a clean frozen worktree")
    selection_pool = None
    if args.selection_pool is not None:
        selection_pool = load_json(args.selection_pool)
        if selection_pool.get("source_manifest_sha256") != sha256_file(
            args.source_manifest
        ):
            raise ValueError("rank-screen selection pool source hash drift")
        items = select_exact_rank_screen_items(source_manifest, selection_pool)
        selection_rule = EXACT_POOL_SELECTION_RULE
    else:
        max_items = args.max_items if args.max_items is not None else 32
        items = select_rank_screen_items(source_manifest, max_items=max_items)
        selection_rule = SELECTION_RULE
    model, tokenizer = _load_model_and_tokenizer(config)
    results = screen_candidate_ranks(model, tokenizer, items)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "selection_evidence_only": True,
        "selection_rule": selection_rule,
        "max_items": args.max_items,
        "source_manifest_path": str(args.source_manifest.resolve()),
        "source_manifest_sha256": sha256_file(args.source_manifest),
        "config_path": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config),
        "model": dict(config["model"]),
        "code_revision": code_revision,
        "runtime_environment": collect_runtime_environment(),
        "results": results,
    }
    if args.selection_pool is not None:
        payload["selection_pool_path"] = str(args.selection_pool.resolve())
        payload["selection_pool_sha256"] = sha256_file(args.selection_pool)
    save_rank_screen(args.output, payload)
    counts: dict[int, int] = {}
    for result in results:
        count = int(result["candidate_count"])
        counts[count] = counts.get(count, 0) + 1
    print(
        json.dumps(
            {
                "output": str(args.output),
                "result_count": len(results),
                "candidate_count_histogram": counts,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
