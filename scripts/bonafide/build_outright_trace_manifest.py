"""Freeze the approved Qwen observatory targets into independent trace units.

The human export remains an unmodified review artifact. This builder resolves
its approved subset against the immutable v2 review payload and emits a
launchable benchmark manifest with historical-thinking token identity bound.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from circuits.analysis.bonafide.canonical import canonical_sha256, file_sha256
from circuits.analysis.bonafide.outright_target_review import load_tokenizer_registry

from scripts.bonafide.manifest import SCHEMA_VERSION

MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
MODEL_REVISION = "768f209d9ea81521153ed38c47d515654e938aea"
COMPLETION_ID = (
    "completion_42bc5b3c2f9a878254b646838bd019b9c337f257db33c92a5048850d230e7173"
)
APPROVED_POSITIONS = (65, 88, 120, 135, 162, 181, 184)
WAVE_ID = "raw-observatory-qwen-modular-q1-width1-v1"
TOKEN_IDENTITY_SCHEMA = "adag.teacher-forced-token-identity.v1"
TOKEN_HASH_ENCODING = "sha256_utf8_canonical_json_integer_array_v1"
FROZEN_SELECTION_SCHEMA = "adag.raw-graph-observatory.frozen-target-selection/v1"
EXPORT_SELECTION_SCHEMA = "adag.raw-graph-observatory.outright-target-selection.v2"


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _load_review_payload(path: Path) -> dict[str, Any]:
    html = path.read_text(encoding="utf-8")
    match = re.search(r'atob\("([A-Za-z0-9+/=]+)"\)', html)
    if match is None:
        raise ValueError("review page lacks its embedded payload")
    value = json.loads(gzip.decompress(base64.b64decode(match.group(1))))
    if not isinstance(value, dict):
        raise ValueError("embedded review payload must be an object")
    return value


def _one(values: list[Any], description: str) -> Any:
    if len(values) != 1:
        raise ValueError(f"expected exactly one {description}; found {len(values)}")
    return values[0]


def build_manifest(
    *,
    selection_path: Path,
    review_path: Path,
    review_manifest_path: Path,
    registry_path: Path,
) -> dict[str, Any]:
    selection = _load_object(selection_path)
    payload = _load_review_payload(review_path)
    review_manifest = _load_object(review_manifest_path)
    registry, registry_sha256 = load_tokenizer_registry(registry_path)

    selection_schema = selection.get("schema_version", selection.get("schemaVersion"))
    if selection_schema not in {FROZEN_SELECTION_SCHEMA, EXPORT_SELECTION_SCHEMA}:
        raise ValueError("unsupported target-selection export")
    if payload.get("schemaVersion") != "adag.raw-graph-observatory.outright-review.v2":
        raise ValueError("unsupported review payload")
    if file_sha256(review_path) != review_manifest.get("page_sha256"):
        raise ValueError("review page hash drift")
    if canonical_sha256(payload) != review_manifest.get("payload_sha256"):
        raise ValueError("embedded review payload hash drift")
    provenance = selection.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("selection export lacks provenance")
    expected_provenance = {
        "sourceSha256": payload["meta"]["sourceSha256"],
        "reviewPayloadSha256": review_manifest["payload_sha256"],
        "tokenizerRegistrySha256": registry_sha256,
    }
    for field, expected in expected_provenance.items():
        if provenance.get(field) != expected:
            raise ValueError(f"selection provenance {field} drift")
    if selection_schema == FROZEN_SELECTION_SCHEMA:
        policy = selection.get("selection_policy")
        if not isinstance(policy, Mapping):
            raise ValueError("frozen selection lacks selection_policy")
        if policy.get("model") != MODEL_ID:
            raise ValueError("frozen selection model drift")
        if policy.get("completion_id") != COMPLETION_ID:
            raise ValueError("frozen selection completion drift")
        if tuple(policy.get("approved_response_positions", [])) != APPROVED_POSITIONS:
            raise ValueError("frozen selection approved positions drift")
        source_export = selection.get("source_export")
        if not isinstance(source_export, Mapping):
            raise ValueError("frozen selection lacks source_export provenance")
        if source_export.get("schema_version") != EXPORT_SELECTION_SCHEMA:
            raise ValueError("frozen selection source export schema drift")

    completion = _one(
        [
            item
            for item in payload["completions"]
            if item.get("completionId") == COMPLETION_ID
        ],
        "approved completion",
    )
    if completion.get("model") != MODEL_ID:
        raise ValueError("approved completion model drift")
    tokenization = completion["tokenization"]
    if tokenization.get("tokenizerRevision") != MODEL_REVISION:
        raise ValueError("approved completion tokenizer revision drift")
    if tokenization.get("serializationMode") != ("historical_thinking_continuation"):
        raise ValueError("approved completion serialization-mode drift")

    profile = registry["profiles"][MODEL_ID]
    prompt_record = registry["prompt_provenance"][profile["system_prompt"]]
    if (
        hashlib.sha256(prompt_record["value"].encode("utf-8")).hexdigest()
        != (tokenization["systemPromptSha256"])
    ):
        raise ValueError("system prompt hash drift")

    selected = list(selection.get("targetSelections", []))
    if selection_schema == EXPORT_SELECTION_SCHEMA:
        selected = [
            item
            for item in selected
            if item.get("completionId") == COMPLETION_ID
            and item.get("responsePosition") in APPROVED_POSITIONS
        ]
    selected.sort(key=lambda item: item["responsePosition"])
    positions = tuple(item["responsePosition"] for item in selected)
    if positions != APPROVED_POSITIONS:
        raise ValueError(
            f"approved target set drift: expected {APPROVED_POSITIONS}, found {positions}"
        )
    if len({item["targetId"] for item in selected}) != len(selected):
        raise ValueError("approved target IDs are not unique")

    token_identity = {
        "schema_version": TOKEN_IDENTITY_SCHEMA,
        "hash_encoding": TOKEN_HASH_ENCODING,
        "assistant_prefix_ids_sha256": tokenization["assistantPrefixIdsSha256"],
        "response_ids_sha256": tokenization["responseIdsSha256"],
        "assistant_prefix_token_count": tokenization["assistantPrefixTokenCount"],
        "response_token_count": tokenization["responseTokenCount"],
    }
    example = {
        "example_id": COMPLETION_ID,
        "task_id": completion["taskId"],
        "annotation_row_ids": sorted(
            {annotation["sourceRowId"] for annotation in completion["annotations"]}
        ),
        "question_ids": completion["questionIds"],
        "label_types": completion["exactLabelTypes"],
        "source_label_status": completion["broadLabel"],
        "prompt": completion["prompt"],
        "response": completion["reasoning"],
        "system_prompt": prompt_record["value"],
        "teacher_forced_serialization_mode": tokenization["serializationMode"],
        "historical_replay_scope": "stored_reasoning_segment_only",
        "historical_replay_status": "reconstructed_not_byte_recovered",
        "token_identity": token_identity,
    }

    items = []
    for target in selected:
        position = target["responsePosition"]
        encoded_token = completion["tokens"][position]
        if (
            target["tokenId"] != encoded_token[0]
            or target["surfaceText"] != (encoded_token[2])
        ):
            raise ValueError(f"exported token identity drift at position {position}")
        for field, expected in (
            ("tokenizerRevision", MODEL_REVISION),
            ("responseIdsSha256", tokenization["responseIdsSha256"]),
            (
                "assistantPrefixIdsSha256",
                tokenization["assistantPrefixIdsSha256"],
            ),
            ("serializationMode", tokenization["serializationMode"]),
            ("systemPromptSha256", tokenization["systemPromptSha256"]),
            ("chatTemplateSha256", tokenization["chatTemplateSha256"]),
        ):
            if target.get(field) != expected:
                raise ValueError(
                    f"exported target {field} drift at position {position}"
                )
        items.append(
            {
                "artifact_id": target["targetId"],
                "example": example,
                "response_token_count": tokenization["responseTokenCount"],
                "target_selection": {
                    "kind": "explicit_response_positions",
                    "response_token_positions": [position],
                    "width": 1,
                    "final_target_token_id": target["tokenId"],
                    "human_selection": {
                        "target_id": target["targetId"],
                        "surface_text": target["surfaceText"],
                        "token_text": target["tokenText"],
                        "comment": target.get("comment", ""),
                        "response_tokens_before": target["responseTokensBefore"],
                        "prediction_position": target["predictionPosition"],
                    },
                },
                "objective": {
                    "name": "sum_selected_logits",
                    "benchmark_only_multi_target": False,
                },
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "phase": "raw_graph_observatory_selected_targets_v1",
        "dataset": {
            "name": payload["meta"]["sourceName"],
            "sha256": payload["meta"]["sourceSha256"],
        },
        "source_selection": {
            "selection_path": str(selection_path),
            "selection_sha256": file_sha256(selection_path),
            "selection_schema_version": selection_schema,
            "review_path": str(review_path),
            "review_page_sha256": file_sha256(review_path),
            "review_payload_sha256": canonical_sha256(payload),
            "tokenizer_registry_path": str(registry_path),
            "tokenizer_registry_sha256": registry_sha256,
            "approved_completion_id": COMPLETION_ID,
            "approved_response_positions": list(APPROVED_POSITIONS),
            "adjudication": "faithful_only; source negative-label disputes excluded",
        },
        "tokenizer": {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "chat_template_sha256": tokenization["chatTemplateSha256"],
            "snapshot_manifest_sha256": tokenization["snapshotManifestSha256"],
        },
        "teacher_forcing_contract": {
            "serialization_mode": tokenization["serializationMode"],
            "system_prompt_sha256": tokenization["systemPromptSha256"],
            "token_identity_schema_version": TOKEN_IDENTITY_SCHEMA,
            "hash_encoding": TOKEN_HASH_ENCODING,
            "identity_status": tokenization["reconstructionStatus"],
        },
        "execution_contract": {
            "trace_units_are_independent": True,
            "target_width": 1,
            "objective": "observed_token_logit",
            "merge_graphs": False,
            "claim_boundary": (
                "Exploratory pruned local attribution graphs for selected observed "
                "tokens; not causal or faithfulness verdicts."
            ),
        },
        "waves": [
            {
                "wave_id": WAVE_ID,
                "corpus_role": "human_selected_faithful_inspection",
                "purpose": "independent raw-neuron graphs around an early computation",
                "items": items,
            }
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path(
            "scripts/bonafide/selections/"
            "qwen3_4b_thinking_raw_graph_observatory_v1.json"
        ),
    )
    parser.add_argument(
        "--review",
        type=Path,
        default=Path(
            "experiments/raw_graph_observatory/outright-task-review-v2/review.html"
        ),
    )
    parser.add_argument(
        "--review-manifest",
        type=Path,
        default=Path(
            "experiments/raw_graph_observatory/outright-task-review-v2/manifest.json"
        ),
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("scripts/bonafide/outright_review_tokenizer_profiles_v2.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_manifest(
        selection_path=args.selection,
        review_path=args.review,
        review_manifest_path=args.review_manifest,
        registry_path=args.registry,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": file_sha256(args.output),
                "target_count": len(manifest["waves"][0]["items"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
