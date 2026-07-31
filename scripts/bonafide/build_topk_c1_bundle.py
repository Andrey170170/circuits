"""Build the immutable independent-candidate manifests for the C1 cohort."""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from scripts.bonafide.build_topk_c0_bundle import (
    RANK_SCREEN_SCHEMA,
    _load_json,
    _source_contract,
    _trace_family,
)
from scripts.bonafide.build_topk_manifest import save_manifest
from scripts.bonafide.execution_plan import sha256_file
from scripts.bonafide.topk_manifest import SCHEMA_VERSION, validate_topk_manifest

C1_SELECTION_SCHEMA = "bonafide-topk-c1-cohort-selection/v1"
C1_BUNDLE_SCHEMA = "bonafide-topk-c1-launch-bundle/v1"


def build_c1_manifests(
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
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Create six specified-token manifests for a balanced C1 resource cohort."""

    if selection.get("schema_version") != C1_SELECTION_SCHEMA:
        raise ValueError("unsupported C1 cohort selection schema")
    if rank_screen.get("schema_version") != RANK_SCREEN_SCHEMA:
        raise ValueError("unsupported C1 rank-screen schema")
    if rank_screen.get("source_manifest_sha256") != source_manifest_sha256:
        raise ValueError("C1 rank screen and source manifest hashes disagree")
    cohort_id = selection.get("cohort_id")
    if not isinstance(cohort_id, str) or not cohort_id:
        raise ValueError("C1 cohort_id must be a non-empty string")
    raw_cases = selection.get("cases")
    if not isinstance(raw_cases, list) or not 24 <= len(raw_cases) <= 48:
        raise ValueError("C1 cohort must contain 24--48 cases")

    source_items: dict[str, tuple[str, Mapping[str, Any]]] = {}
    for wave in source_manifest.get("waves", []):
        role = wave.get("corpus_role")
        for item in wave.get("items", []):
            artifact_id = item.get("artifact_id")
            if artifact_id in source_items:
                raise ValueError(f"duplicate C1 source artifact: {artifact_id}")
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
            raise ValueError("C1 selection cases must be objects")
        artifact_id = raw_case.get("source_width1_artifact_id")
        if not isinstance(artifact_id, str) or not artifact_id or artifact_id in seen:
            raise ValueError(f"invalid or duplicate C1 source artifact: {artifact_id}")
        seen.add(artifact_id)
        source_pair = source_items.get(artifact_id)
        rank_result = rank_results.get(artifact_id)
        if source_pair is None or not isinstance(rank_result, Mapping):
            raise ValueError(f"C1 case lacks source or rank evidence: {artifact_id}")
        role, source_item = source_pair
        if role not in {"dense_discovery", "broad_discovery"}:
            raise ValueError(f"C1 case has invalid discovery role: {artifact_id}")
        if rank_result.get("corpus_role") != role:
            raise ValueError(f"C1 rank evidence role mismatch: {artifact_id}")
        candidate_selection = rank_result.get("candidate_selection")
        if (
            not isinstance(candidate_selection, Mapping)
            or candidate_selection.get("policy_id") != "model_top5_plus_observed"
        ):
            raise ValueError(f"C1 case lacks union-policy rank evidence: {artifact_id}")
        candidates = candidate_selection.get("candidates")
        if not isinstance(candidates, list) or len(candidates) not in {5, 6}:
            raise ValueError(f"C1 case candidate width is invalid: {artifact_id}")
        token_ids = [candidate.get("token_id") for candidate in candidates]
        if any(
            isinstance(token_id, bool) or not isinstance(token_id, int)
            for token_id in token_ids
        ) or len(set(token_ids)) != len(token_ids):
            raise ValueError(f"C1 case candidate IDs are invalid: {artifact_id}")
        observed_token_id = source_item["target_selection"]["final_target_token_id"]
        if token_ids[0] != observed_token_id:
            raise ValueError(f"C1 observed candidate mismatch: {artifact_id}")
        if (
            raw_case.get("corpus_role") != role
            or raw_case.get("candidate_count") != len(candidates)
            or raw_case.get("observed_token_rank")
            != candidate_selection.get("observed_token_rank")
        ):
            raise ValueError(f"C1 frozen selection expectation drift: {artifact_id}")
        reasons = raw_case.get("selection_reasons")
        if (
            not isinstance(reasons, list)
            or not reasons
            or any(not isinstance(reason, str) or not reason for reason in reasons)
        ):
            raise ValueError(f"C1 selection reasons are required: {artifact_id}")
        example = source_item["example"]
        diversity = example.get("diversity", {})
        cases.append(
            {
                "case_id": raw_case["case_id"],
                "source_width1_artifact_id": artifact_id,
                "example_id": example["example_id"],
                "base_question_id": example["base_question_id"],
                "corpus_role": role,
                "target_response_position": source_item["target_selection"][
                    "response_token_positions"
                ][0],
                "input_token_count": rank_result["input_token_count"],
                "candidate_count": len(candidates),
                "observed_token_rank": candidate_selection["observed_token_rank"],
                "candidate_token_ids": token_ids,
                "cot_phenotype": diversity.get("cot_phenotype"),
                "hint_types": list(example.get("hint_types", [])),
                "selection_reasons": list(reasons),
                "_source_item": source_item,
            }
        )

    role_counts = Counter(case["corpus_role"] for case in cases)
    width_counts = Counter(str(case["candidate_count"]) for case in cases)
    role_width_counts = Counter(
        f"{case['corpus_role']}:{case['candidate_count']}" for case in cases
    )
    response_counts = Counter(case["example_id"] for case in cases)
    family_counts = Counter(case["base_question_id"] for case in cases)
    phenotype_counts = Counter(case["cot_phenotype"] for case in cases)
    if role_counts != {"dense_discovery": 16, "broad_discovery": 16}:
        raise ValueError("C1 cohort must retain the frozen 16/16 role balance")
    if width_counts != {"5": 17, "6": 15}:
        raise ValueError("C1 cohort must retain the frozen 17/15 width balance")
    if min(role_width_counts.values()) < 7 or len(role_width_counts) != 4:
        raise ValueError("C1 cohort must cover every role/width cell")
    if len(response_counts) < 24 or max(response_counts.values()) > 3:
        raise ValueError("C1 response balance contract failed")
    if len(family_counts) < 24 or max(family_counts.values()) > 4:
        raise ValueError("C1 family balance contract failed")
    if set(phenotype_counts) != {"faithful", "omission", "commission", "both"}:
        raise ValueError("C1 cohort must cover all four CoT phenotypes")

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
        candidate_index: int, selected_cases: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for case in selected_cases:
            item = copy.deepcopy(case["_source_item"])
            item["specified_candidate_token_id"] = case["candidate_token_ids"][
                candidate_index
            ]
            grouped[case["corpus_role"]].append(item)
        return [
            {
                "wave_id": (
                    f"c1-independent-candidate-{candidate_index}-"
                    f"{role.replace('_discovery', '')}"
                ),
                "corpus_role": role,
                "items": grouped[role],
            }
            for role in ("dense_discovery", "broad_discovery")
            if grouped[role]
        ]

    manifests: dict[str, dict[str, Any]] = {}
    for candidate_index in range(6):
        eligible = [case for case in cases if case["candidate_count"] > candidate_index]
        label = f"independent-candidate-{candidate_index}"
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "phase": "c1_policy_resource",
            "trace_family": _trace_family(
                trace_family_id=f"bonafide.c1.{label}.v1",
                policy_id="specified_token",
                objective_id="raw_logit_sum",
            ),
            "source": source_contract,
            "cohort": cohort_contract,
            "waves": waves_for(candidate_index, eligible),
        }
        validate_topk_manifest(manifest)
        manifests[label] = manifest

    balance = {
        "role_counts": dict(sorted(role_counts.items())),
        "width_counts": dict(sorted(width_counts.items())),
        "role_width_counts": dict(sorted(role_width_counts.items())),
        "response_count": len(response_counts),
        "max_targets_per_response": max(response_counts.values()),
        "family_count": len(family_counts),
        "max_targets_per_family": max(family_counts.values()),
        "phenotype_counts": dict(sorted(phenotype_counts.items())),
        "input_token_count_min": min(case["input_token_count"] for case in cases),
        "input_token_count_max": max(case["input_token_count"] for case in cases),
    }
    public_cases = [
        {key: value for key, value in case.items() if key != "_source_item"}
        for case in cases
    ]
    return manifests, public_cases, balance


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
        raise ValueError("C1 selection source manifest hash drift")
    if rank_screen_sha256 != selection.get("rank_screen_sha256"):
        raise ValueError("C1 selection rank-screen hash drift")

    manifests, cases, balance = build_c1_manifests(
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
        label: f"qwen3_4b_instruct_topk_c1_{label.replace('-', '_')}_v1.json"
        for label in manifests
    }
    targets = [args.output_dir / filename for filename in filenames.values()]
    if args.bundle_output.exists() or any(path.exists() for path in targets):
        raise FileExistsError("one or more C1 bundle destinations already exist")
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
        "schema_version": C1_BUNDLE_SCHEMA,
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
        "balance": balance,
        "cases": cases,
        "manifests": manifest_records,
    }
    save_manifest(args.bundle_output, bundle)
    print(json.dumps(bundle, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
