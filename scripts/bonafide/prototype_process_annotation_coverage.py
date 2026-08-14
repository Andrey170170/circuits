#!/usr/bin/env python3
"""Measure versioned graph-blind suggestion coverage on a workstation bundle.

This is a detector-yield diagnostic, not an annotation-quality evaluation and not a
frozen annotation artifact.
"""

from __future__ import annotations

import argparse
import json
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

from circuits.analysis.bonafide.process_annotation import (
    event_token_position_matches,
    file_sha256,
    load_ontology,
    suggest_matches,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--ontology", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def percentage(numerator: int, denominator: int) -> float:
    return round(100 * numerator / denominator, 3) if denominator else 0.0


def main() -> None:
    args = parse_args()
    bundle = json.loads(args.bundle.read_text(encoding="utf-8"))
    ontology = load_ontology(args.ontology)
    cohort_manifest_path = args.cohort / "manifest.json"
    cohort_index_path = args.cohort / "index.jsonl"
    cohort_manifest = json.loads(cohort_manifest_path.read_text(encoding="utf-8"))
    cohort_rows = [
        json.loads(line)
        for line in cohort_index_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if cohort_manifest.get("status") != "frozen":
        raise ValueError("source cohort is not frozen")
    if len(cohort_rows) != cohort_manifest.get("records"):
        raise ValueError("source cohort record count mismatch")
    if file_sha256(cohort_index_path) != cohort_manifest.get("index_sha256"):
        raise ValueError("source cohort index hash drift")
    accepted_keys: dict[str, set[str]] = {}
    for row in cohort_rows:
        record_path = args.cohort / row["record_path"]
        if file_sha256(record_path) != row["record_sha256"]:
            raise ValueError(f"source record hash drift: {row['response_id']}")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        keys: set[str] = set()
        generation = record.get("generation_row")
        if generation is not None:
            for schema in json.loads(
                generation.get("accepted_answer_schemas_json", "[]")
            ):
                keys.update(str(key) for key in schema.get("exact_keys", []))
        accepted_keys[row["response_id"]] = keys
    if set(accepted_keys) != {
        document["response_id"] for document in bundle["documents"]
    }:
        raise ValueError("bundle/cohort response identities differ")

    axes = (
        "discourse_phase",
        "process_span",
        "event_operation",
        "operation",
        "process_role",
        "event_token_position",
    )
    covered_by_axis: dict[str, int] = Counter()
    covered_by_value: dict[str, int] = Counter()
    response_support: dict[str, int] = Counter()
    source_totals: dict[str, int] = Counter()
    source_process_covered: dict[str, int] = Counter()
    total_tokens = 0
    exact_serializations = 0
    per_response: list[dict[str, object]] = []

    for document in bundle["documents"]:
        text = document["text"]
        tokens = document["tokenization"]["tokens"]
        starts = [int(token[1]) for token in tokens]
        ends = [int(token[2]) for token in tokens]
        if starts != sorted(starts) or ends != sorted(ends):
            raise ValueError(f"non-monotonic offsets in {document['response_id']}")
        matches = suggest_matches(
            text,
            ontology,
            accepted_answer_keys=accepted_keys[document["response_id"]] or None,
        )
        matches.extend(
            event_token_position_matches(
                text,
                [[int(token[1]), int(token[2])] for token in tokens],
                matches,
            )
        )
        positions_by_axis: dict[str, set[int]] = defaultdict(set)
        positions_by_value: dict[tuple[str, str], set[int]] = defaultdict(set)
        values_seen: set[tuple[str, str]] = set()
        for match in matches:
            if match.axis not in axes:
                continue
            token_start = bisect_right(ends, match.start)
            token_end = bisect_left(starts, match.end)
            if token_start >= token_end:
                raise ValueError(
                    f"span overlaps no tokens in {document['response_id']}: {match}"
                )
            positions = range(token_start, token_end)
            positions_by_axis[match.axis].update(positions)
            positions_by_value[(match.axis, match.value)].update(positions)
            values_seen.add((match.axis, match.value))
        token_count = len(tokens)
        total_tokens += token_count
        for axis, positions in positions_by_axis.items():
            covered_by_axis[axis] += len(positions)
        for (axis, value), positions in positions_by_value.items():
            covered_by_value[f"{axis}/{value}"] += len(positions)
        for axis, value in values_seen:
            response_support[f"{axis}/{value}"] += 1
        if ("discourse_phase", "answer_serialization") in values_seen:
            exact_serializations += 1
        process_covered = len(positions_by_axis["process_span"])
        source_types = document.get("task_context", {}).get("source_types", []) or [
            "unknown"
        ]
        for source_type in source_types:
            source_totals[str(source_type)] += token_count
            source_process_covered[str(source_type)] += process_covered
        per_response.append(
            {
                "response_id": document["response_id"],
                "token_count": token_count,
                "discourse_phase_tokens": len(positions_by_axis["discourse_phase"]),
                "process_span_tokens": process_covered,
            }
        )

    report = {
        "schema_version": "adag.process-witness.annotation-coverage-prototype.v1",
        "status": "prototype_not_frozen",
        "generated_at": datetime.now(UTC).isoformat(),
        "claim_boundary": (
            "Unique-token detector yield on frozen text; this does not measure label "
            "accuracy, process execution, correctness, or faithfulness."
        ),
        "inputs": {
            "bundle": str(args.bundle),
            "bundle_sha256": file_sha256(args.bundle),
            "source_annotation_set_id": bundle["annotation_set_id"],
            "cohort": str(args.cohort),
            "cohort_manifest_sha256": file_sha256(cohort_manifest_path),
            "cohort_index_sha256": file_sha256(cohort_index_path),
            "cohort_id": cohort_manifest["cohort_id"],
            "ontology": str(args.ontology),
            "ontology_file_sha256": file_sha256(args.ontology),
            "ontology_id": ontology["ontology_id"],
            "base_ontology_sha256": ontology.get("extension_provenance", {}).get(
                "base_ontology_sha256"
            ),
        },
        "responses": len(bundle["documents"]),
        "tokens": total_tokens,
        "terminal_answer_serializations": exact_serializations,
        "builder_fidelity": {
            "accepted_answer_keys_from_hash_verified_source_records": True,
            "derived_event_token_positions_included": True,
            "token_offsets_from_source_workstation_bundle": True,
        },
        "unique_token_coverage_by_axis": {
            axis: {
                "tokens": covered_by_axis[axis],
                "percent": percentage(covered_by_axis[axis], total_tokens),
            }
            for axis in axes
        },
        "unique_token_coverage_by_value": dict(sorted(covered_by_value.items())),
        "response_support_by_value": dict(sorted(response_support.items())),
        "process_span_coverage_by_source_type": {
            source_type: {
                "tokens": source_process_covered[source_type],
                "total_tokens": source_totals[source_type],
                "percent": percentage(
                    source_process_covered[source_type], source_totals[source_type]
                ),
            }
            for source_type in sorted(source_totals)
        },
        "per_response": per_response,
    }
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
