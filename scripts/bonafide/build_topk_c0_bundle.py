"""Build the immutable joint and independent manifests for a C0 cohort."""

from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from scripts.bonafide.build_topk_manifest import save_manifest
from scripts.bonafide.execution_plan import sha256_file
from scripts.bonafide.topk_manifest import (
    MODEL_TOP5_PLUS_OBSERVED_COUNT_RULE,
    SCHEMA_VERSION,
    validate_topk_manifest,
)

C0_SELECTION_SCHEMA = "bonafide-topk-c0-cohort-selection/v1"
C0_BUNDLE_SCHEMA = "bonafide-topk-c0-launch-bundle/v1"
RANK_SCREEN_SCHEMA = "bonafide-topk-rank-screen/v1"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _source_contract(
    source_manifest: Mapping[str, Any],
    *,
    source_manifest_path: Path,
    source_manifest_sha256: str,
) -> dict[str, str]:
    tokenizer = source_manifest.get("tokenizer")
    if not isinstance(tokenizer, Mapping):
        raise ValueError("C0 source manifest lacks tokenizer provenance")
    for field in ("model_id", "revision", "chat_template_sha256"):
        if not isinstance(tokenizer.get(field), str) or not tokenizer[field]:
            raise ValueError(f"C0 source tokenizer.{field} is required")
    return {
        "width1_manifest_path": str(source_manifest_path.resolve()),
        "width1_manifest_sha256": source_manifest_sha256,
        "model_id": tokenizer["model_id"],
        "model_revision": tokenizer["revision"],
        "tokenizer_revision": tokenizer["revision"],
        "chat_template_sha256": tokenizer["chat_template_sha256"],
    }


def _trace_family(
    *,
    trace_family_id: str,
    policy_id: str,
    objective_id: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "trace_family_id": trace_family_id,
        "candidate_policy_id": policy_id,
        "candidate_policy_version": "1",
        "joint_objective_id": objective_id,
        "joint_objective_version": "1",
    }
    if policy_id == "model_top5_plus_observed":
        result.update(
            {
                "candidate_count_min": 5,
                "candidate_count_max": 6,
                "candidate_count_rule": MODEL_TOP5_PLUS_OBSERVED_COUNT_RULE,
            }
        )
    elif policy_id == "specified_token":
        result["candidate_count"] = 1
    else:
        raise ValueError(f"unsupported C0 policy: {policy_id}")
    return result


def build_c0_manifests(
    selection: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    rank_screen: Mapping[str, Any],
    *,
    selection_path: Path,
    selection_sha256: str,
    source_manifest_path: Path,
    source_manifest_sha256: str,
    rank_screen_path: Path,
    rank_screen_sha256: str,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Create two joint and six candidate-index reference manifests."""

    if selection.get("schema_version") != C0_SELECTION_SCHEMA:
        raise ValueError("unsupported C0 cohort selection schema")
    if rank_screen.get("schema_version") != RANK_SCREEN_SCHEMA:
        raise ValueError("unsupported C0 rank-screen schema")
    if rank_screen.get("source_manifest_sha256") != source_manifest_sha256:
        raise ValueError("C0 rank screen and source manifest hashes disagree")
    cohort_id = selection.get("cohort_id")
    if not isinstance(cohort_id, str) or not cohort_id:
        raise ValueError("C0 cohort_id must be a non-empty string")
    raw_cases = selection.get("cases")
    if not isinstance(raw_cases, list) or not 8 <= len(raw_cases) <= 12:
        raise ValueError("C0 cohort must contain 8--12 cases")

    source_items: dict[str, tuple[str, Mapping[str, Any]]] = {}
    for wave in source_manifest.get("waves", []):
        role = wave.get("corpus_role")
        for item in wave.get("items", []):
            artifact_id = item.get("artifact_id")
            if artifact_id in source_items:
                raise ValueError(f"duplicate C0 source artifact: {artifact_id}")
            source_items[artifact_id] = (role, item)
    rank_results = {
        result.get("source_width1_artifact_id"): result
        for result in rank_screen.get("results", [])
        if isinstance(result, Mapping)
    }

    seen: set[str] = set()
    cases: list[dict[str, Any]] = []
    for raw_case in raw_cases:
        if not isinstance(raw_case, Mapping):
            raise ValueError("C0 selection cases must be objects")
        artifact_id = raw_case.get("source_width1_artifact_id")
        if not isinstance(artifact_id, str) or not artifact_id or artifact_id in seen:
            raise ValueError(f"invalid or duplicate C0 source artifact: {artifact_id}")
        seen.add(artifact_id)
        source_pair = source_items.get(artifact_id)
        rank_result = rank_results.get(artifact_id)
        if source_pair is None or not isinstance(rank_result, Mapping):
            raise ValueError(f"C0 case lacks source or rank evidence: {artifact_id}")
        role, source_item = source_pair
        if role not in {"dense_discovery", "broad_discovery"}:
            raise ValueError(f"C0 case has invalid discovery role: {artifact_id}")
        if rank_result.get("corpus_role") != role:
            raise ValueError(f"C0 rank evidence role mismatch: {artifact_id}")
        candidate_selection = rank_result.get("candidate_selection")
        if (
            not isinstance(candidate_selection, Mapping)
            or candidate_selection.get("policy_id") != "model_top5_plus_observed"
        ):
            raise ValueError(f"C0 case lacks union-policy rank evidence: {artifact_id}")
        candidates = candidate_selection.get("candidates")
        if not isinstance(candidates, list) or len(candidates) not in {5, 6}:
            raise ValueError(f"C0 case candidate width is invalid: {artifact_id}")
        token_ids = [candidate.get("token_id") for candidate in candidates]
        if any(
            isinstance(token_id, bool) or not isinstance(token_id, int)
            for token_id in token_ids
        ) or len(set(token_ids)) != len(token_ids):
            raise ValueError(f"C0 case candidate IDs are invalid: {artifact_id}")
        observed_token_id = source_item["target_selection"]["final_target_token_id"]
        if token_ids[0] != observed_token_id:
            raise ValueError(f"C0 observed candidate mismatch: {artifact_id}")
        expected_role = raw_case.get("corpus_role")
        expected_width = raw_case.get("candidate_count")
        expected_rank = raw_case.get("observed_token_rank")
        if (
            expected_role != role
            or expected_width != len(candidates)
            or expected_rank != candidate_selection.get("observed_token_rank")
        ):
            raise ValueError(f"C0 frozen selection expectation drift: {artifact_id}")
        reasons = raw_case.get("selection_reasons")
        if (
            not isinstance(reasons, list)
            or not reasons
            or any(not isinstance(reason, str) or not reason for reason in reasons)
        ):
            raise ValueError(f"C0 selection reasons are required: {artifact_id}")
        cases.append(
            {
                "case_id": raw_case["case_id"],
                "source_width1_artifact_id": artifact_id,
                "example_id": source_item["example"]["example_id"],
                "corpus_role": role,
                "target_response_position": source_item["target_selection"][
                    "response_token_positions"
                ][0],
                "candidate_count": len(candidates),
                "observed_token_rank": candidate_selection["observed_token_rank"],
                "candidate_token_ids": token_ids,
                "selection_reasons": list(reasons),
                "_source_item": source_item,
            }
        )

    roles = {case["corpus_role"] for case in cases}
    widths = {case["candidate_count"] for case in cases}
    if roles != {"dense_discovery", "broad_discovery"} or widths != {5, 6}:
        raise ValueError("C0 cohort must cover both discovery roles and widths")

    source_contract = _source_contract(
        source_manifest,
        source_manifest_path=source_manifest_path,
        source_manifest_sha256=source_manifest_sha256,
    )
    cohort_contract = {
        "cohort_id": cohort_id,
        "selection_path": str(selection_path.resolve()),
        "selection_sha256": selection_sha256,
        "rank_screen_path": str(rank_screen_path.resolve()),
        "rank_screen_sha256": rank_screen_sha256,
        "case_count": len(cases),
    }

    def waves_for(
        family_label: str,
        selected_cases: list[dict[str, Any]],
        *,
        candidate_index: int | None = None,
    ) -> list[dict[str, Any]]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for case in selected_cases:
            item = copy.deepcopy(case["_source_item"])
            if candidate_index is not None:
                item["specified_candidate_token_id"] = case["candidate_token_ids"][
                    candidate_index
                ]
            grouped[case["corpus_role"]].append(item)
        return [
            {
                "wave_id": f"c0-{family_label}-{role.replace('_discovery', '')}",
                "corpus_role": role,
                "items": grouped[role],
            }
            for role in ("dense_discovery", "broad_discovery")
            if grouped[role]
        ]

    manifests: dict[str, dict[str, Any]] = {}
    for objective_id, label in (
        ("raw_logit_sum", "joint-raw"),
        ("observed_vs_alternatives", "joint-contrastive"),
    ):
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "phase": "c0_candidate_reference",
            "trace_family": _trace_family(
                trace_family_id=f"bonafide.c0.{label}.v1",
                policy_id="model_top5_plus_observed",
                objective_id=objective_id,
            ),
            "source": source_contract,
            "cohort": cohort_contract,
            "waves": waves_for(label, cases),
        }
        validate_topk_manifest(manifest)
        manifests[label] = manifest

    for candidate_index in range(6):
        eligible = [case for case in cases if case["candidate_count"] > candidate_index]
        label = f"independent-candidate-{candidate_index}"
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "phase": "c0_candidate_reference",
            "trace_family": _trace_family(
                trace_family_id=f"bonafide.c0.{label}.v1",
                policy_id="specified_token",
                objective_id="raw_logit_sum",
            ),
            "source": source_contract,
            "cohort": cohort_contract,
            "waves": waves_for(label, eligible, candidate_index=candidate_index),
        }
        validate_topk_manifest(manifest)
        manifests[label] = manifest

    public_cases = [
        {key: value for key, value in case.items() if key != "_source_item"}
        for case in cases
    ]
    return manifests, public_cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bundle-output", type=Path, required=True)
    args = parser.parse_args()

    selection = _load_json(args.selection)
    source_manifest_path = Path(selection["source_manifest_path"])
    rank_screen_path = Path(selection["rank_screen_path"])
    source_manifest_sha256 = sha256_file(source_manifest_path)
    rank_screen_sha256 = sha256_file(rank_screen_path)
    if source_manifest_sha256 != selection.get("source_manifest_sha256"):
        raise ValueError("C0 selection source manifest hash drift")
    if rank_screen_sha256 != selection.get("rank_screen_sha256"):
        raise ValueError("C0 selection rank-screen hash drift")

    manifests, cases = build_c0_manifests(
        selection,
        _load_json(source_manifest_path),
        _load_json(rank_screen_path),
        selection_path=args.selection,
        selection_sha256=sha256_file(args.selection),
        source_manifest_path=source_manifest_path,
        source_manifest_sha256=source_manifest_sha256,
        rank_screen_path=rank_screen_path,
        rank_screen_sha256=rank_screen_sha256,
    )
    filenames = {
        label: f"qwen3_4b_instruct_topk_c0_{label.replace('-', '_')}_v1.json"
        for label in manifests
    }
    targets = [args.output_dir / filename for filename in filenames.values()]
    if args.bundle_output.exists() or any(path.exists() for path in targets):
        raise FileExistsError("one or more C0 bundle destinations already exist")
    manifest_records: list[dict[str, Any]] = []
    for label, manifest in manifests.items():
        path = args.output_dir / filenames[label]
        save_manifest(path, manifest)
        manifest_records.append(
            {
                "label": label,
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "trace_family_id": manifest["trace_family"]["trace_family_id"],
                "waves": [wave["wave_id"] for wave in manifest["waves"]],
                "work_item_count": sum(
                    len(wave["items"]) for wave in manifest["waves"]
                ),
            }
        )
    bundle = {
        "schema_version": C0_BUNDLE_SCHEMA,
        "cohort_id": selection["cohort_id"],
        "selection_path": str(args.selection.resolve()),
        "selection_sha256": sha256_file(args.selection),
        "source_manifest_path": str(source_manifest_path.resolve()),
        "source_manifest_sha256": source_manifest_sha256,
        "rank_screen_path": str(rank_screen_path.resolve()),
        "rank_screen_sha256": rank_screen_sha256,
        "case_count": len(cases),
        "expected_trace_count": sum(
            record["work_item_count"] for record in manifest_records
        ),
        "cases": cases,
        "manifests": manifest_records,
    }
    save_manifest(args.bundle_output, bundle)
    print(json.dumps(bundle, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
