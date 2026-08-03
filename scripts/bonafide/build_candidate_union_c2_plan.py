"""Freeze the C2 fixed-union refinement plan from completed pass-one traces."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from circuits.tracing.artifact import (
    load_topk_compact_trace,
    validate_topk_compact_trace_integrity,
)
from circuits.tracing.candidate_union import frozen_union_topologies

from scripts.bonafide.build_candidate_union_c0_plan import (
    PLAN_SCHEMA_VERSION,
    _load_json,
    _source_items,
)
from scripts.bonafide.build_topk_c2_bundle import (
    C2_BUNDLE_SCHEMA,
    C2_CASE_COUNT,
    C2_SELECTION_SCHEMA,
)
from scripts.bonafide.build_topk_manifest import save_manifest
from scripts.bonafide.candidate_union_runner import validate_candidate_union_plan
from scripts.bonafide.execution_plan import sha256_file
from scripts.bonafide.runner import _sha256, validate_run_config
from scripts.bonafide.topk_manifest import validate_topk_manifest

REFERENCE_FAMILY_PREFIX = "bonafide.c2.independent-candidate-"
ROLE_SHARD_COUNTS = {"dense_discovery": 5, "broad_discovery": 11}
MAX_CASES_PER_WAVE = 16
PASS1_GIT_COMMIT = "a6a9557bfcfe5e099a164ef686220dfc5be0abe6"


def _bundle_manifest_contracts(
    bundle: Mapping[str, Any],
) -> dict[int, dict[str, Any]]:
    records = bundle.get("manifests")
    if not isinstance(records, list) or len(records) != 6:
        raise ValueError("C2 bundle must bind exactly six candidate manifests")
    result: dict[int, dict[str, Any]] = {}
    for candidate_index in range(6):
        label = f"independent-candidate-{candidate_index}"
        matches = [record for record in records if record.get("label") == label]
        if len(matches) != 1:
            raise ValueError(f"C2 bundle does not bind {label} exactly once")
        record = matches[0]
        path = Path(str(record.get("path", "")))
        expected_file_sha256 = record.get("sha256")
        if (
            not path.is_absolute()
            or not path.is_file()
            or not isinstance(expected_file_sha256, str)
            or sha256_file(path) != expected_file_sha256
        ):
            raise ValueError(f"C2 candidate manifest file/hash drift: {label}")
        manifest = _load_json(path)
        validate_topk_manifest(manifest)
        expected_family = f"bonafide.c2.{label}.v1"
        family = manifest["trace_family"]
        if (
            manifest.get("phase") != "c2_scientific_utility"
            or family.get("trace_family_id") != expected_family
            or family.get("candidate_policy_id") != "specified_token"
            or family.get("joint_objective_id") != "raw_logit_sum"
            or record.get("trace_family_id") != expected_family
        ):
            raise ValueError(f"C2 candidate manifest contract drift: {label}")
        items: dict[str, tuple[str, Mapping[str, Any]]] = {}
        for wave in manifest["waves"]:
            for item in wave["items"]:
                source_id = item.get("artifact_id")
                if not isinstance(source_id, str) or source_id in items:
                    raise ValueError(
                        f"C2 candidate manifest source drift: {label}, {source_id}"
                    )
                items[source_id] = (wave["wave_id"], item)
        result[candidate_index] = {
            "path": path,
            "file_sha256": expected_file_sha256,
            "canonical_sha256": _sha256(manifest),
            "manifest": manifest,
            "items": items,
        }
    source_contracts = {
        _sha256(contract["manifest"]["source"]) for contract in result.values()
    }
    cohort_contracts = {
        _sha256(contract["manifest"]["cohort"]) for contract in result.values()
    }
    if len(source_contracts) != 1 or len(cohort_contracts) != 1:
        raise ValueError("C2 candidate manifests disagree on source or cohort")
    return result


def _reference_index(reference_root: Path) -> dict[tuple[int, str], Path]:
    result: dict[tuple[int, str], Path] = {}
    for candidate_index in range(6):
        family_root = reference_root / (
            f"{REFERENCE_FAMILY_PREFIX}{candidate_index}.v1"
        )
        if not family_root.is_dir():
            raise FileNotFoundError(f"missing C2 reference family: {family_root}")
        for manifest_path in family_root.glob("*/*/manifest.json"):
            manifest = _load_json(manifest_path)
            source_id = manifest.get("source_width1_artifact_id")
            if not isinstance(source_id, str) or not source_id:
                raise ValueError(f"invalid C2 reference source ID: {source_id!r}")
            key = (candidate_index, source_id)
            if key in result:
                raise ValueError(f"duplicate C2 reference: {key}")
            result[key] = manifest_path.parent.resolve()
    return result


def _balanced_shards(
    cases: Sequence[dict[str, Any]],
    *,
    shard_count: int,
) -> list[list[dict[str, Any]]]:
    if (
        isinstance(shard_count, bool)
        or shard_count < 1
        or len(cases) > shard_count * MAX_CASES_PER_WAVE
    ):
        raise ValueError("C2 shard count cannot hold the supplied cases")
    shards: list[list[dict[str, Any]]] = [[] for _ in range(shard_count)]
    weights = [0 for _ in range(shard_count)]

    def rescore_weight(case: Mapping[str, Any]) -> int:
        counts = case.get("frozen_union_candidate_edge_counts")
        if not isinstance(counts, list) or not counts:
            raise ValueError("C2 cases require exact frozen-union edge counts")
        return sum(int(count) for count in counts)

    for case in sorted(
        cases,
        key=lambda value: (
            -rescore_weight(value),
            value["case_id"],
        ),
    ):
        available = [
            index
            for index, shard in enumerate(shards)
            if len(shard) < MAX_CASES_PER_WAVE
        ]
        target = min(
            available,
            key=lambda index: (weights[index], len(shards[index]), index),
        )
        shards[target].append(case)
        weights[target] += rescore_weight(case)
    if any(not shard for shard in shards):
        raise ValueError("C2 balancing produced an empty refinement wave")
    for shard in shards:
        shard.sort(key=lambda case: case["case_id"])
    return shards


def build_candidate_union_c2_plan(
    bundle: Mapping[str, Any],
    selection: Mapping[str, Any],
    candidate_zero_manifest: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    bundle_path: Path,
    bundle_sha256: str,
    selection_path: Path,
    selection_sha256: str,
    candidate_zero_manifest_path: Path,
    candidate_zero_manifest_sha256: str,
    config_path: Path,
    config_sha256: str,
    reference_root: Path,
) -> dict[str, Any]:
    """Bind the frozen C2 cohort to exact validated pass-one artifacts."""

    if bundle.get("schema_version") != C2_BUNDLE_SCHEMA:
        raise ValueError("unsupported C2 launch-bundle schema")
    validate_run_config(config)
    if selection.get("schema_version") != C2_SELECTION_SCHEMA:
        raise ValueError("unsupported C2 selection schema")
    cases = selection.get("cases")
    if (
        not isinstance(cases, list)
        or len(cases) != C2_CASE_COUNT
        or bundle.get("case_count") != C2_CASE_COUNT
        or bundle.get("cohort_id") != selection.get("cohort_id")
        or bundle.get("rank_screen_path") != selection.get("rank_screen_path")
        or bundle.get("rank_screen_sha256") != selection.get("rank_screen_sha256")
        or Path(str(bundle.get("selection_path", ""))).resolve()
        != selection_path.resolve()
        or bundle.get("selection_sha256") != selection_sha256
    ):
        raise ValueError("candidate-union C2 cohort or selection hash drift")
    candidate_contracts = _bundle_manifest_contracts(bundle)
    expected_manifest_cohort = {
        "cohort_id": selection["cohort_id"],
        "case_count": C2_CASE_COUNT,
        "rank_screen_path": selection["rank_screen_path"],
        "rank_screen_sha256": selection["rank_screen_sha256"],
        "selection_path": str(selection_path.resolve()),
        "selection_sha256": selection_sha256,
    }
    if any(
        contract["manifest"].get("cohort") != expected_manifest_cohort
        for contract in candidate_contracts.values()
    ):
        raise ValueError("C2 candidate manifest cohort disagrees with selection")
    if (
        candidate_contracts[0]["manifest"] != candidate_zero_manifest
        or candidate_contracts[0]["path"].resolve()
        != candidate_zero_manifest_path.resolve()
        or candidate_contracts[0]["file_sha256"] != candidate_zero_manifest_sha256
    ):
        raise ValueError("candidate-zero C2 manifest disagrees with bundle")
    source = candidate_zero_manifest.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("candidate-zero manifest lacks source provenance")

    source_items = _source_items(candidate_zero_manifest)
    reference_paths = _reference_index(reference_root)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    width_counts: Counter[int] = Counter()
    reference_code_revisions: set[str] = set()
    reference_source_trees: set[str] = set()

    for case in cases:
        source_id = case.get("source_width1_artifact_id")
        case_id = case.get("case_id")
        role = case.get("corpus_role")
        expected_tokens = case.get("candidate_token_ids")
        if (
            not isinstance(source_id, str)
            or not isinstance(case_id, str)
            or role not in ROLE_SHARD_COUNTS
            or not isinstance(expected_tokens, list)
            or len(expected_tokens) not in {5, 6}
        ):
            raise ValueError(f"invalid frozen C2 case: {case_id!r}")
        source_item = source_items.get(source_id)
        if source_item is None:
            raise ValueError(f"C2 case lacks its source work item: {source_id}")
        if source_item["target_selection"]["response_token_positions"] != [
            case["target_response_position"]
        ]:
            raise ValueError(f"C2 source position drift: {case_id}")

        references = []
        estimated_edges = 0
        loaded_references = []
        for candidate_index, expected_token_id in enumerate(expected_tokens):
            candidate_contract = candidate_contracts[candidate_index]
            manifest_item_pair = candidate_contract["items"].get(source_id)
            if manifest_item_pair is None:
                raise ValueError(
                    f"C2 case is absent from candidate {candidate_index} manifest: "
                    f"{case_id}"
                )
            manifest_wave_id, candidate_manifest_item = manifest_item_pair
            if (
                candidate_manifest_item.get("specified_candidate_token_id")
                != expected_token_id
            ):
                raise ValueError(
                    f"C2 manifest candidate token drift: {case_id}, "
                    f"candidate {candidate_index}"
                )
            path = reference_paths.get((candidate_index, source_id))
            if path is None:
                raise ValueError(
                    f"C2 case lacks candidate {candidate_index} reference: {case_id}"
                )
            manifest = validate_topk_compact_trace_integrity(path)
            artifact = load_topk_compact_trace(path)
            trace = artifact.topk_trace
            candidate = trace.candidate_selection.candidates[0]
            identity = manifest.get("artifact_identity")
            trace_contract = manifest.get("candidate_trace_contract")
            if (
                not isinstance(identity, Mapping)
                or not isinstance(trace_contract, Mapping)
                or trace.candidate_count != 1
                or candidate.token_id != expected_token_id
                or trace.shared_response_position != case["target_response_position"]
                or manifest.get("source_width1_artifact_id") != source_id
                or manifest.get("code_revision", {}).get("git_commit")
                != PASS1_GIT_COMMIT
                or identity.get("topk_manifest_sha256")
                != candidate_contract["canonical_sha256"]
                or identity.get("trace_family")
                != candidate_contract["manifest"]["trace_family"]
                or identity.get("wave_id") != manifest_wave_id
                or identity.get("source_width1_work_item_sha256")
                != _sha256(candidate_manifest_item)
                or identity.get("source_width1_manifest_sha256")
                != source["width1_manifest_sha256"]
                or identity.get("model") != config["model"]
                or identity.get("adag_config") != config["adag_config"]
                or manifest.get("source_target_selection")
                != source_item["target_selection"]
                or manifest.get("bonafide_example") != source_item["example"]
                or trace_contract.get("trace_family_id")
                != candidate_contract["manifest"]["trace_family"]["trace_family_id"]
            ):
                raise ValueError(
                    f"C2 candidate reference contract drift: {case_id}, "
                    f"candidate {candidate_index}"
                )
            revision = manifest["code_revision"]
            if (
                revision.get("git_dirty") is not False
                or identity.get("code_revision") != revision
            ):
                raise ValueError(
                    f"C2 candidate reference code provenance drift: {case_id}, "
                    f"candidate {candidate_index}"
                )
            reference_code_revisions.add(_sha256(revision))
            reference_source_trees.add(str(revision.get("source_tree_sha256")))
            estimated_edges += int(manifest["edge_count"])
            loaded_references.append(artifact)
            references.append(
                {
                    "candidate_index": candidate_index,
                    "token_id": candidate.token_id,
                    "artifact_id": manifest["artifact_id"],
                    "path": str(path),
                    "payload_sha256": manifest["data_sha256"],
                }
            )
        if reference_paths.get((len(expected_tokens), source_id)) is not None:
            raise ValueError(f"C2 case has an unexpected extra reference: {case_id}")
        width_counts[len(expected_tokens)] += 1
        topology_sha256, topologies = frozen_union_topologies(loaded_references)
        union_mlp_nodes = len(topologies[0].mlp_nodes)
        candidate_edge_counts = [len(topology.edges) for topology in topologies]
        grouped[role].append(
            {
                "case_id": case_id,
                "source_width1_artifact_id": source_id,
                "source_item": source_item,
                "reference_artifacts": references,
                "estimated_reference_edge_count": estimated_edges,
                "frozen_union_topology_sha256": topology_sha256,
                "frozen_union_mlp_node_count": union_mlp_nodes,
                "frozen_union_candidate_edge_counts": candidate_edge_counts,
                "frozen_union_max_candidate_edge_count": max(candidate_edge_counts),
            }
        )

    if {role: len(values) for role, values in grouped.items()} != {
        "dense_discovery": 77,
        "broad_discovery": 168,
    }:
        raise ValueError("candidate-union C2 role counts drifted")
    if width_counts != Counter({5: 235, 6: 10}):
        raise ValueError("candidate-union C2 width counts drifted")
    if len(reference_code_revisions) != 1 or len(reference_source_trees) != 1:
        raise ValueError("candidate-union C2 pass-one code cohort drifted")

    waves = []
    for role in ("dense_discovery", "broad_discovery"):
        shards = _balanced_shards(
            grouped[role],
            shard_count=ROLE_SHARD_COUNTS[role],
        )
        for shard_index, shard in enumerate(shards):
            waves.append(
                {
                    "wave_id": (
                        f"candidate-union-c2-{role.replace('_discovery', '')}-"
                        f"{shard_index:02d}"
                    ),
                    "corpus_role": role,
                    "estimated_reference_edge_count": sum(
                        case["estimated_reference_edge_count"] for case in shard
                    ),
                    "frozen_union_rescore_edge_count": sum(
                        sum(case["frozen_union_candidate_edge_counts"])
                        for case in shard
                    ),
                    "cases": shard,
                }
            )

    plan = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "phase": "c2_candidate_union_refinement",
        "plan_id": "qwen3-4b-instruct-candidate-union-c2-v1",
        "source": {
            "model_id": source["model_id"],
            "model_revision": source["model_revision"],
            "tokenizer_revision": source["tokenizer_revision"],
            "chat_template_sha256": source["chat_template_sha256"],
            "width1_manifest_path": source["width1_manifest_path"],
            "width1_manifest_sha256": source["width1_manifest_sha256"],
        },
        "execution": {
            "config_path": str(config_path.resolve()),
            "config_sha256": config_sha256,
            "config_canonical_sha256": _sha256(config),
            "required_clean_worktree": True,
            "required_launch_git_commit_via_environment": True,
            "pass1_git_commit": PASS1_GIT_COMMIT,
            "pass1_source_tree_sha256": next(iter(reference_source_trees)),
        },
        "cohort": {
            "cohort_id": bundle["cohort_id"],
            "launch_bundle_path": str(bundle_path.resolve()),
            "launch_bundle_sha256": bundle_sha256,
            "selection_path": str(selection_path.resolve()),
            "selection_sha256": selection_sha256,
            "candidate_zero_manifest_path": str(candidate_zero_manifest_path.resolve()),
            "candidate_zero_manifest_sha256": (candidate_zero_manifest_sha256),
            "reference_root": str(reference_root.resolve()),
            "case_count": len(cases),
            "reference_trace_count": sum(
                width * count for width, count in width_counts.items()
            ),
            "candidate_width_counts": {
                str(width): count for width, count in sorted(width_counts.items())
            },
        },
        "refinement": {
            "topology_semantics": "exact_union_of_independent_candidate_k1_graphs",
            "node_measurement_semantics": (
                "candidate_specific_fixed_union_node_rescore_including_zero"
            ),
            "edge_measurement_semantics": (
                "candidate_specific_fixed_union_edge_rescore_including_zero"
            ),
            "terminal_edge_applicability": (
                "only_edges_targeting_the_current_candidate_logit"
            ),
            "wave_balancing": (
                "greedy_descending_sum_of_exact_candidate_union_edge_counts"
            ),
            "maximum_cases_per_wave": MAX_CASES_PER_WAVE,
        },
        "waves": waves,
    }
    validate_candidate_union_plan(plan)
    if len(waves) != 16 or max(len(wave["cases"]) for wave in waves) > 16:
        raise AssertionError("C2 refinement wave-shape drift")
    return plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--candidate-zero-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"candidate-union plan already exists: {args.output}")
    bundle = _load_json(args.bundle)
    selection = _load_json(args.selection)
    if Path(bundle["selection_path"]).resolve() != args.selection.resolve():
        raise ValueError("C2 bundle points to another selection path")
    candidate_zero_manifest = _load_json(args.candidate_zero_manifest)
    config = _load_json(args.config)
    plan = build_candidate_union_c2_plan(
        bundle,
        selection,
        candidate_zero_manifest,
        config,
        bundle_path=args.bundle,
        bundle_sha256=sha256_file(args.bundle),
        selection_path=args.selection,
        selection_sha256=sha256_file(args.selection),
        candidate_zero_manifest_path=args.candidate_zero_manifest,
        candidate_zero_manifest_sha256=sha256_file(args.candidate_zero_manifest),
        config_path=args.config,
        config_sha256=sha256_file(args.config),
        reference_root=args.reference_root,
    )
    save_manifest(args.output, plan)
    print(json.dumps(plan, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
