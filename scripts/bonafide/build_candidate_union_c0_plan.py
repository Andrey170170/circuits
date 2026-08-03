"""Freeze a candidate-union refinement plan from completed C0 k1 traces."""

from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from circuits.tracing.artifact import (
    load_topk_compact_trace,
    validate_topk_compact_trace_integrity,
)

from scripts.bonafide.build_topk_manifest import save_manifest
from scripts.bonafide.execution_plan import sha256_file

C0_BUNDLE_SCHEMA = "bonafide-topk-c0-launch-bundle/v1"
PLAN_SCHEMA_VERSION = "bonafide-candidate-union-plan/v1"
REFERENCE_FAMILY_PREFIX = "bonafide.c0.independent-candidate-"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _source_items(
    candidate_zero_manifest: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    result = {}
    for wave in candidate_zero_manifest.get("waves", []):
        for raw_item in wave.get("items", []):
            item = copy.deepcopy(raw_item)
            source_id = item.get("artifact_id")
            if not isinstance(source_id, str) or not source_id or source_id in result:
                raise ValueError(f"invalid candidate-zero source item: {source_id!r}")
            item.pop("specified_candidate_token_id", None)
            result[source_id] = item
    return result


def _reference_index(reference_root: Path) -> dict[tuple[int, str], Path]:
    result = {}
    for candidate_index in range(6):
        family_root = reference_root / f"{REFERENCE_FAMILY_PREFIX}{candidate_index}.v1"
        if not family_root.is_dir():
            raise FileNotFoundError(f"missing C0 reference family: {family_root}")
        for manifest_path in family_root.glob("*/*/manifest.json"):
            manifest = _load_json(manifest_path)
            source_id = manifest.get("source_width1_artifact_id")
            if not isinstance(source_id, str) or not source_id:
                raise ValueError(f"invalid C0 reference source ID: {source_id!r}")
            key = (candidate_index, source_id)
            if key in result:
                raise ValueError(f"duplicate C0 reference: {key}")
            result[key] = manifest_path.parent.resolve()
    return result


def build_candidate_union_plan(
    bundle: Mapping[str, Any],
    candidate_zero_manifest: Mapping[str, Any],
    *,
    bundle_path: Path,
    bundle_sha256: str,
    candidate_zero_manifest_path: Path,
    candidate_zero_manifest_sha256: str,
    reference_root: Path,
) -> dict[str, Any]:
    """Bind the selected C0 cases to exact completed independent artifacts."""

    if bundle.get("schema_version") != C0_BUNDLE_SCHEMA:
        raise ValueError("unsupported C0 launch-bundle schema")
    cases = bundle.get("cases")
    if not isinstance(cases, list) or len(cases) != 10:
        raise ValueError("candidate-union C0 plan requires the frozen ten cases")
    source = candidate_zero_manifest.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("candidate-zero manifest lacks source provenance")
    source_items = _source_items(candidate_zero_manifest)
    reference_paths = _reference_index(reference_root)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for case in cases:
        source_id = case.get("source_width1_artifact_id")
        case_id = case.get("case_id")
        role = case.get("corpus_role")
        expected_tokens = case.get("candidate_token_ids")
        if (
            not isinstance(source_id, str)
            or not isinstance(case_id, str)
            or role not in {"dense_discovery", "broad_discovery"}
            or not isinstance(expected_tokens, list)
            or len(expected_tokens) not in {5, 6}
        ):
            raise ValueError(f"invalid frozen C0 case: {case_id!r}")
        source_item = source_items.get(source_id)
        if source_item is None:
            raise ValueError(f"C0 case lacks its source work item: {source_id}")
        if source_item["target_selection"]["response_token_positions"] != [
            case["target_response_position"]
        ]:
            raise ValueError(f"C0 source position drift: {case_id}")

        references = []
        for candidate_index, expected_token_id in enumerate(expected_tokens):
            path = reference_paths.get((candidate_index, source_id))
            if path is None:
                raise ValueError(
                    f"C0 case lacks candidate {candidate_index} reference: {case_id}"
                )
            manifest = validate_topk_compact_trace_integrity(path)
            artifact = load_topk_compact_trace(path)
            trace = artifact.topk_trace
            candidate = trace.candidate_selection.candidates[0]
            if (
                trace.candidate_count != 1
                or candidate.token_id != expected_token_id
                or trace.shared_response_position != case["target_response_position"]
                or manifest.get("source_width1_artifact_id") != source_id
            ):
                raise ValueError(
                    f"C0 candidate reference contract drift: {case_id}, "
                    f"candidate {candidate_index}"
                )
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
            raise ValueError(f"C0 case has an unexpected extra reference: {case_id}")
        grouped[role].append(
            {
                "case_id": case_id,
                "source_width1_artifact_id": source_id,
                "source_item": source_item,
                "reference_artifacts": references,
            }
        )

    if {role: len(values) for role, values in grouped.items()} != {
        "dense_discovery": 6,
        "broad_discovery": 4,
    }:
        raise ValueError("candidate-union C0 role counts drifted")
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "phase": "c0_candidate_union_refinement",
        "plan_id": "qwen3-4b-instruct-candidate-union-c0-v1",
        "source": {
            "model_id": source["model_id"],
            "model_revision": source["model_revision"],
            "tokenizer_revision": source["tokenizer_revision"],
            "chat_template_sha256": source["chat_template_sha256"],
            "width1_manifest_path": source["width1_manifest_path"],
            "width1_manifest_sha256": source["width1_manifest_sha256"],
        },
        "cohort": {
            "cohort_id": bundle["cohort_id"],
            "launch_bundle_path": str(bundle_path.resolve()),
            "launch_bundle_sha256": bundle_sha256,
            "selection_path": bundle["selection_path"],
            "selection_sha256": bundle["selection_sha256"],
            "candidate_zero_manifest_path": str(candidate_zero_manifest_path.resolve()),
            "candidate_zero_manifest_sha256": candidate_zero_manifest_sha256,
            "reference_root": str(reference_root.resolve()),
            "case_count": len(cases),
            "reference_trace_count": sum(
                len(case["reference_artifacts"])
                for values in grouped.values()
                for case in values
            ),
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
        },
        "waves": [
            {
                "wave_id": f"candidate-union-c0-{role.replace('_discovery', '')}",
                "corpus_role": role,
                "cases": grouped[role],
            }
            for role in ("dense_discovery", "broad_discovery")
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--candidate-zero-manifest", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"candidate-union plan already exists: {args.output}")
    bundle = _load_json(args.bundle)
    candidate_zero_manifest = _load_json(args.candidate_zero_manifest)
    plan = build_candidate_union_plan(
        bundle,
        candidate_zero_manifest,
        bundle_path=args.bundle,
        bundle_sha256=sha256_file(args.bundle),
        candidate_zero_manifest_path=args.candidate_zero_manifest,
        candidate_zero_manifest_sha256=sha256_file(args.candidate_zero_manifest),
        reference_root=args.reference_root,
    )
    save_manifest(args.output, plan)
    print(json.dumps(plan, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
