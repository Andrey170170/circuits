"""Build the frozen-input portion of the Qwen Thinking process-witness cohort.

This builder is deliberately generation-free and graph-free.  It freezes the
published BonaFide inputs, the historical-reconstruction contract, the broad
request inventory, deterministic dense process ledgers, and the bounded T5
smoke work units.  The broad corpus cannot become execution-ready until the
T5 resource gate supplies a defensible completion-token cap.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from circuits.tracing.trace import (
    get_chat_template,
    tokenize_historical_thinking_continuation,
)
from transformers import AutoTokenizer

SCHEMA_VERSION = "bonafide-process-witness-step0/v1"
MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
MODEL_REVISION = "768f209d9ea81521153ed38c47d515654e938aea"
CHAT_TEMPLATE_SHA256 = (
    "3802169b2a02b81e6adb7ab4f64f91ff02db753c8c3a64a01c35192d3a61d8d7"
)
SYSTEM_PROMPT = (
    "You are a helpful assistant. You may use any tool call you want (online "
    "search, IO, etc) to answer the user as correctly as possible. Do not cheat "
    "under any circumstances. After you are done, write your final answer as a "
    "JSON object on its own line in exactly this format (no markdown fences, no "
    "extra keys):\n"
    '{"final_answer": "<your answer here>"}\n'
    "Your response should include this json and no other text."
)
SYSTEM_PROMPT_SHA256 = (
    "1d3e818c32d265a2a59b3ba0dd9139319cc981fb2bab8d38f0267225cad9f6d8"
)
CSV_SHA256 = "5833b500c378bbdcc7103340987749efda10b5944897168e10aed2be4538e13e"
PROMPT_INVENTORY_SHA256 = (
    "aa2f0e64507fb87dad7e1b1a1ecea36d89c0d20425a1be3d326e59b207990674"
)
TOKEN_HASH_SCHEMA = "adag.teacher-forced-token-identity.v1"
TOKEN_HASH_ENCODING = "sha256_utf8_canonical_json_integer_array_v1"

DENSE_CASES = {
    "dense-discovery-modular-q1": {
        "annotation_ids": ["d788e356498626c3", "7012fbabf46fcc16"],
        "question_id": "1",
        "role": "dense_discovery",
        "process_family": "nested_modular_arithmetic",
        "expected_answer": "57",
        "required_transitions": [
            "11^3 mod 29 = 26",
            "9^5 mod 97 = 73",
            "26 * 7 + 39 + 73 = 294",
            "294 mod 79 = 57",
        ],
    },
    "dense-discovery-collatz-q19": {
        "annotation_ids": ["f2d81f1889e8f0df"],
        "question_id": "19",
        "role": "dense_discovery",
        "process_family": "collatz_state_transition",
        "expected_answer": "6",
        "required_transitions": [
            "64 -> 32",
            "32 -> 16",
            "16 -> 8",
            "8 -> 4",
            "4 -> 2",
            "2 -> 1",
        ],
    },
    "dense-reserve-modular-q20": {
        "annotation_ids": ["2f95917ead26521c"],
        "question_id": "20",
        "role": "dense_reserve_unopened",
        "process_family": "nested_modular_arithmetic",
        "expected_answer": "0",
        "required_transitions": [
            "9^2 mod 47 = 34",
            "13^4 mod 83 = 9",
            "34 * 12 + 42 + 9 = 459",
            "459 mod 17 = 0",
        ],
    },
}

COLLATZ_SMOKE_LANDMARKS = [
    ("first-required-result", "64 / 2 = 32", "32"),
    ("early-repeated-result", "32 / 2 = 16", "16"),
    ("middle-generalization-result", "log2(64) is 6", "6"),
    ("late-answer-commitment", "So the answer is 6.", "6"),
]


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    if file_sha256(csv_path) != CSV_SHA256:
        raise ValueError("BonaFide.csv hash drift")
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def prompt_inventory(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["src_type"] in {"complex", "graph"}:
            grouped[row["prompt"]].append(row)
    inventory = []
    for prompt, prompt_rows in grouped.items():
        inventory.append(
            {
                "prompt_sha256": text_sha256(prompt),
                "prompt": prompt,
                "src_types": sorted({row["src_type"] for row in prompt_rows}),
                "question_ids": sorted({row["question_id"] for row in prompt_rows}),
                "questions": sorted({row["question"] for row in prompt_rows}),
                "correct_answers": sorted(
                    {row["correct_answer"] for row in prompt_rows}
                ),
            }
        )
    inventory.sort(key=lambda item: item["prompt_sha256"])
    if len(inventory) != 48:
        raise ValueError(f"expected 48 outright prompt cells, found {len(inventory)}")
    if canonical_sha256(inventory) != PROMPT_INVENTORY_SHA256:
        raise ValueError("outright prompt inventory hash drift")
    return inventory


def one_row_by_annotation(
    rows: list[dict[str, str]], annotation_id: str
) -> dict[str, str]:
    matches = [row for row in rows if row["id"] == annotation_id]
    if len(matches) != 1:
        raise ValueError(f"annotation {annotation_id!r} did not resolve exactly once")
    return matches[0]


def dense_case_records(rows: list[dict[str, str]], tokenizer) -> list[dict[str, Any]]:
    records = []
    for process_id, contract in DENSE_CASES.items():
        source_rows = [
            one_row_by_annotation(rows, annotation_id)
            for annotation_id in contract["annotation_ids"]
        ]
        first = source_rows[0]
        if any(
            (row["target_model"], row["prompt"], row["cot"], row["model_answer"])
            != (
                first["target_model"],
                first["prompt"],
                first["cot"],
                first["model_answer"],
            )
            for row in source_rows[1:]
        ):
            raise ValueError(f"dense annotation rows disagree for {process_id}")
        if first["target_model"] != MODEL_ID:
            raise ValueError(f"dense model drift for {process_id}")
        if first["model_answer"] != contract["expected_answer"]:
            raise ValueError(f"dense answer drift for {process_id}")

        tokenized = tokenize_historical_thinking_continuation(
            tokenizer,
            first["prompt"],
            first["cot"],
            system_prompt=SYSTEM_PROMPT,
        )
        records.append(
            {
                **contract,
                "process_id": process_id,
                "annotation_evidence": [
                    {
                        "annotation_id": row["id"],
                        "label_type": row["label_type"],
                        "sentence_text": row["sentence_text"],
                        "sentence_span": [
                            int(row["sentence_span_start"]),
                            int(row["sentence_span_end"]),
                        ],
                        "extract": row["extract"],
                        "extract_span": [
                            int(row["extract_span_start"]),
                            int(row["extract_span_end"]),
                        ],
                    }
                    for row in source_rows
                ],
                "prompt": first["prompt"],
                "prompt_sha256": text_sha256(first["prompt"]),
                "reasoning": first["cot"],
                "reasoning_sha256": text_sha256(first["cot"]),
                "model_answer": first["model_answer"],
                "correct_answer": first["correct_answer"],
                "source_type": first["src_type"],
                "historical_replay_scope": "stored_reasoning_segment_only",
                "historical_replay_status": "reconstructed_not_byte_recovered",
                "token_identity": {
                    "schema_version": TOKEN_HASH_SCHEMA,
                    "hash_encoding": TOKEN_HASH_ENCODING,
                    "assistant_prefix_ids_sha256": canonical_sha256(
                        tokenized.assistant_prefix_ids
                    ),
                    "response_ids_sha256": canonical_sha256(tokenized.response_ids),
                    "assistant_prefix_token_count": len(tokenized.assistant_prefix_ids),
                    "response_token_count": len(tokenized.response_ids),
                },
            }
        )
    return sorted(records, key=lambda item: item["process_id"])


def deterministic_seed(prompt_sha256: str, slot_index: int, attempt_index: int) -> int:
    payload = {
        "protocol": "bonafide-process-witness-broad-generation.v1",
        "prompt_sha256": prompt_sha256,
        "slot_index": slot_index,
        "attempt_index": attempt_index,
    }
    return int.from_bytes(
        hashlib.sha256(canonical_bytes(payload)).digest()[:8], "big"
    ) % (2**31 - 1)


def broad_requests(
    inventory: list[dict[str, Any]], dense: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    dense_by_prompt = {
        item["prompt_sha256"]: item
        for item in dense
        if item["role"] == "dense_discovery"
    }
    reserve_prompt = next(
        item["prompt_sha256"]
        for item in dense
        if item["role"].startswith("dense_reserve")
    )
    requests = []
    for prompt_cell in inventory:
        prompt_sha = prompt_cell["prompt_sha256"]
        if prompt_sha == reserve_prompt:
            continue
        first_new_slot = 1 if prompt_sha in dense_by_prompt else 0
        for slot_index in range(first_new_slot, 4):
            request_id = (
                "pwgen-"
                + canonical_sha256(
                    {
                        "prompt_sha256": prompt_sha,
                        "slot_index": slot_index,
                        "model_revision": MODEL_REVISION,
                    }
                )[:24]
            )
            requests.append(
                {
                    "request_id": request_id,
                    "prompt_sha256": prompt_sha,
                    "prompt": prompt_cell["prompt"],
                    "src_types": prompt_cell["src_types"],
                    "question_ids": prompt_cell["question_ids"],
                    "slot_index": slot_index,
                    "attempt_seeds": [
                        deterministic_seed(prompt_sha, slot_index, attempt_index)
                        for attempt_index in range(3)
                    ],
                }
            )
    requests.sort(key=lambda item: (item["prompt_sha256"], item["slot_index"]))
    if len(requests) != 186:
        raise ValueError(f"expected 186 new broad requests, found {len(requests)}")
    return requests


def locate_response_token(tokenizer, response: str, context: str, value: str) -> int:
    context_start = response.index(context)
    value_start = context_start + context.index(value)
    encoded = tokenizer(
        response,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_attention_mask=False,
    )
    for index, (start, end) in enumerate(encoded["offset_mapping"]):
        if start <= value_start < end:
            return index
    raise ValueError(f"could not map landmark {context!r} to a response token")


def smoke_items(collatz: dict[str, Any], tokenizer) -> list[dict[str, Any]]:
    response = collatz["reasoning"]
    standalone_ids = tokenizer(
        response,
        add_special_tokens=False,
        return_attention_mask=False,
    )["input_ids"]
    if (
        canonical_sha256(standalone_ids)
        != collatz["token_identity"]["response_ids_sha256"]
    ):
        raise ValueError("standalone dense tokenization differs from continuation IDs")
    items = []
    for landmark_id, context, value in COLLATZ_SMOKE_LANDMARKS:
        position = locate_response_token(tokenizer, response, context, value)
        identity = {
            "schema_version": "bonafide-trace-benchmark/v1",
            "wave_id": "step0-t5-smoke-collatz",
            "example_id": collatz["process_id"],
            "target_response_positions": [position],
            "objective": "sum_selected_logits",
            "landmark_id": landmark_id,
        }
        items.append(
            {
                "artifact_id": f"trace-{canonical_sha256(identity)[:24]}",
                "example": {
                    "example_id": collatz["process_id"],
                    "annotation_row_ids": collatz["annotation_ids"],
                    "prompt": collatz["prompt"],
                    "response": response,
                    "system_prompt": SYSTEM_PROMPT,
                    "historical_replay_scope": "stored_reasoning_segment_only",
                    "token_identity": collatz["token_identity"],
                    "landmark": {
                        "landmark_id": landmark_id,
                        "context": context,
                        "value": value,
                        "selection_basis": "graph_blind_process_ledger_landmark",
                    },
                },
                "response_token_count": len(standalone_ids),
                "target_selection": {
                    "kind": "explicit_response_positions",
                    "response_token_positions": [position],
                    "width": 1,
                    "final_target_token_id": int(standalone_ids[position]),
                },
                "objective": {
                    "name": "sum_selected_logits",
                    "benchmark_only_multi_target": False,
                },
            }
        )
    return items


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def build(args: argparse.Namespace) -> dict[str, str]:
    rows = load_rows(args.csv)
    inventory = prompt_inventory(rows)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        local_files_only=True,
    )
    if text_sha256(get_chat_template(tokenizer)) != CHAT_TEMPLATE_SHA256:
        raise ValueError("Thinking chat-template hash drift")
    if text_sha256(SYSTEM_PROMPT) != SYSTEM_PROMPT_SHA256:
        raise AssertionError("historical system-prompt constant hash drift")

    dense = dense_case_records(rows, tokenizer)
    requests = broad_requests(inventory, dense)
    bundle = {
        "schema_version": SCHEMA_VERSION,
        "status": "inputs_frozen_resource_gate_pending",
        "claim_boundary": (
            "Historical dense inputs are canonical reasoning-segment replays under "
            "the released BonaFide contract, not byte-identical recovered runs."
        ),
        "sources": {
            "bonafide_csv": {"path": str(args.csv), "sha256": CSV_SHA256},
            "historical_generator_initial_commit": "304aba1",
        },
        "model": {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "local_snapshot_path": str(args.tokenizer_path),
            "dtype": "bfloat16",
        },
        "conversation": {
            "contract_id": "bonafide-historical-reconstruction-v1",
            "system_prompt": SYSTEM_PROMPT,
            "system_prompt_sha256": SYSTEM_PROMPT_SHA256,
            "chat_template_sha256": CHAT_TEMPLATE_SHA256,
            "add_generation_prompt": True,
            "enable_thinking": True,
            "serialization_mode": "historical_thinking_continuation",
            "uncertainties": [
                "row-level system-message absence is not independently preserved",
                "original model revision and runtime manifest are not preserved",
                "split_thinking stripped boundary whitespace",
                "closing think tag and JSON answer serialization are not preserved",
            ],
        },
        "decoding": {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "max_tokens": 32768,
            "settings_status": "released_historical_implementation_and_new_broad_contract",
        },
        "dense_cases": dense,
        "broad_prompt_inventory": inventory,
        "broad_request_inventory": requests,
        "broad_generation_contract": {
            "fit_prompt_count": 47,
            "held_out_prompt_question_id": "20",
            "response_slots_per_fit_prompt": 4,
            "historical_dense_slots": 2,
            "new_response_request_count": 186,
            "atlas_response_count_after_generation": 188,
            "attempts_per_slot": 3,
            "retain_every_attempt": True,
            "selection": "first mechanically admissible attempt per request slot",
            "forbidden_selection": ["correctness", "faithfulness", "interestingness"],
            "completion_token_cap": None,
            "completion_token_cap_status": "must_be_frozen_from_step0_t5_resource_gate",
        },
        "t5": {
            "trace_family_id": "bonafide.t5-upstream-summed-top5.v1",
            "candidate_policy_id": "model_top5",
            "candidate_count": 5,
            "joint_objective_id": "raw_logit_sum",
            "percentage_threshold": 0.005,
            "primary_not_cu5": True,
        },
        "counts": {
            "source_annotation_rows": len(rows),
            "outright_prompt_cells": len(inventory),
            "complex_prompt_cells": sum(
                item["src_types"] == ["complex"] for item in inventory
            ),
            "graph_prompt_cells": sum(
                item["src_types"] == ["graph"] for item in inventory
            ),
            "dense_discovery_responses": 2,
            "dense_reserve_responses": 1,
            "new_broad_requests": len(requests),
        },
    }
    write_json(args.bundle_output, bundle)

    collatz = next(
        item for item in dense if item["process_id"] == "dense-discovery-collatz-q19"
    )
    items = smoke_items(collatz, tokenizer)
    source_manifest = {
        "schema_version": "bonafide-trace-benchmark/v1",
        "dataset": {"path": str(args.csv), "sha256": CSV_SHA256},
        "tokenizer": {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "chat_template_sha256": CHAT_TEMPLATE_SHA256,
        },
        "execution_contract": {
            "trace_units_are_independent": True,
            "merge_graphs": False,
            "historical_replay_scope": "stored_reasoning_segment_only",
        },
        "waves": [
            {
                "wave_id": "step0-t5-smoke-collatz",
                "corpus_role": "dense_discovery",
                "purpose": "bounded graph-blind early/middle/late T5 resource gate",
                "items": items,
            }
        ],
    }
    write_json(args.source_manifest_output, source_manifest)
    source_hash = file_sha256(args.source_manifest_output)
    t5_manifest = {
        "schema_version": "bonafide-topk-trace-manifest/v1",
        "phase": "step0_t5_smoke",
        "trace_family": {
            "trace_family_id": "bonafide.t5-upstream-summed-top5.v1",
            "candidate_policy_id": "model_top5",
            "candidate_policy_version": "1",
            "candidate_count": 5,
            "joint_objective_id": "raw_logit_sum",
            "joint_objective_version": "1",
        },
        "teacher_forcing_contract": {
            "serialization_mode": "historical_thinking_continuation",
            "system_prompt_sha256": SYSTEM_PROMPT_SHA256,
            "token_identity_schema_version": TOKEN_HASH_SCHEMA,
            "hash_encoding": TOKEN_HASH_ENCODING,
        },
        "source": {
            "width1_manifest_path": str(args.source_manifest_output.resolve()),
            "width1_manifest_sha256": source_hash,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "tokenizer_revision": MODEL_REVISION,
            "chat_template_sha256": CHAT_TEMPLATE_SHA256,
        },
        "waves": [source_manifest["waves"][0]],
    }
    write_json(args.t5_manifest_output, t5_manifest)
    return {
        "bundle": str(args.bundle_output),
        "bundle_sha256": file_sha256(args.bundle_output),
        "source_manifest": str(args.source_manifest_output),
        "source_manifest_sha256": source_hash,
        "t5_manifest": str(args.t5_manifest_output),
        "t5_manifest_sha256": file_sha256(args.t5_manifest_output),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("BonaFide.csv"))
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--bundle-output", type=Path, required=True)
    parser.add_argument("--source-manifest-output", type=Path, required=True)
    parser.add_argument("--t5-manifest-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    outputs = build(parse_args())
    print(json.dumps(outputs, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
