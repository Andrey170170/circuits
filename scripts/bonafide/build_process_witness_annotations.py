#!/usr/bin/env python3
"""Build a graph-blind, automatically suggested annotation draft."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from circuits.analysis.bonafide.process_annotation import (
    annotate_response,
    audit_documents,
    build_inventory,
    build_workstation_bundle,
    canonical_sha256,
    captured_byte_level_token_offsets,
    continuation_token_offsets,
    file_sha256,
    inspection_examples,
    load_ontology,
    text_sha256,
)
from circuits.tracing.trace import get_chat_template
from transformers import AutoTokenizer

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
CHAT_TEMPLATE_SHA256 = (
    "3802169b2a02b81e6adb7ab4f64f91ff02db753c8c3a64a01c35192d3a61d8d7"
)
MODEL_REVISION = "768f209d9ea81521153ed38c47d515654e938aea"
MANIFEST_SCHEMA_VERSION = "adag.process-witness.annotation-set-manifest.v1"
REVIEW_UI_VERSION = "process-witness-token-painter.v7"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--ontology", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--annotation-set-id",
        required=True,
        help="New immutable annotation-set identity; no implicit version is allowed.",
    )
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(
                json.dumps(
                    value,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )


def write_compact_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(
            value,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def load_cohort(cohort: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = cohort / "manifest.json"
    index_path = cohort / "index.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in index_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if manifest["status"] != "frozen":
        raise ValueError("source cohort is not frozen")
    if len(rows) != manifest["records"]:
        raise ValueError("cohort index count differs from manifest")
    if file_sha256(index_path) != manifest["index_sha256"]:
        raise ValueError("cohort index hash drift")
    return manifest, rows


def _tokenizer_file_identity(tokenizer_path: Path) -> dict[str, str]:
    identities = {}
    for name in ("tokenizer.json", "tokenizer_config.json", "config.json"):
        path = tokenizer_path / name
        if not path.is_file():
            raise FileNotFoundError(path)
        identities[name] = file_sha256(path)
    return identities


def _generated_tokenization(
    tokenizer: Any, record: dict[str, Any], text: str
) -> tuple[list[int], list[list[int]], dict[str, Any]]:
    row = record["generation_row"]
    repro = json.loads(row["reproducibility_info"])
    if repro["raw_output"] != text or text_sha256(text) != repro["raw_output_sha256"]:
        raise ValueError(f"raw generation output drift for {record['response_id']}")
    prefix_text = repro["formatted_prompt"]
    if text_sha256(prefix_text) != repro["formatted_prompt_sha256"]:
        raise ValueError(f"formatted prompt hash drift for {record['response_id']}")
    prefix_ids = [int(value) for value in repro["prompt_token_ids"]]
    completion_ids = [int(value) for value in repro["completion_token_ids"]]
    if int(repro["num_tokens"]) != len(completion_ids):
        raise ValueError(f"captured completion count drift for {record['response_id']}")

    content_ids = list(completion_ids)
    suffix_ids: list[int] = []
    stop_ids = [int(value) for value in repro["effective_default_stop_token_ids"]]
    if content_ids and content_ids[-1] in stop_ids:
        suffix_ids = [content_ids.pop()]
    encoded_prefix = tokenizer(
        prefix_text,
        add_special_tokens=False,
        return_attention_mask=False,
    )
    if [int(value) for value in encoded_prefix["input_ids"]] != prefix_ids:
        raise ValueError(f"formatted prompt token drift for {record['response_id']}")
    ids = content_ids
    offsets = captured_byte_level_token_offsets(
        tokenizer,
        text=text,
        token_ids=ids,
    )
    return (
        ids,
        offsets,
        {
            "kind": "captured_generation_continuation",
            "model_revision": repro["model_revision"],
            "formatted_prompt_sha256": repro["formatted_prompt_sha256"],
            "assistant_prefix_token_count": len(prefix_ids),
            "assistant_prefix_ids_sha256": canonical_sha256(prefix_ids),
            "captured_completion_token_count": len(completion_ids),
            "response_token_count": len(content_ids),
            "response_ids_sha256": canonical_sha256(content_ids),
            "offset_alignment": "captured_byte_level_vocabulary_pieces_to_utf8",
            "excluded_terminal_suffix_token_ids": suffix_ids,
            "excluded_terminal_suffix_decoded": [
                tokenizer.decode(suffix_ids, skip_special_tokens=False)
            ]
            if suffix_ids
            else [],
            "exclusion_reason": (
                "one captured terminal default-stop token is an assistant suffix, not "
                "decoded response text"
                if suffix_ids
                else None
            ),
        },
    )


def _historical_tokenization(
    tokenizer: Any, record: dict[str, Any], text: str
) -> tuple[list[int], list[list[int]], dict[str, Any]]:
    historical = record["historical_dense_record"]
    frozen = historical["token_identity"]
    if (
        historical["reasoning"] != text
        or text_sha256(text) != historical["reasoning_sha256"]
    ):
        raise ValueError(f"historical reasoning drift for {record['response_id']}")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": historical["prompt"]},
    ]
    prefix_text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=True,
        chat_template=get_chat_template(tokenizer),
    )
    if not isinstance(prefix_text, str):
        raise TypeError("historical chat template did not return text")
    prefix_encoded = tokenizer(
        prefix_text, add_special_tokens=False, return_attention_mask=False
    )
    prefix_ids = [int(value) for value in prefix_encoded["input_ids"]]
    if len(prefix_ids) != frozen["assistant_prefix_token_count"]:
        raise ValueError(f"historical prefix count drift for {record['response_id']}")
    if canonical_sha256(prefix_ids) != frozen["assistant_prefix_ids_sha256"]:
        raise ValueError(f"historical prefix hash drift for {record['response_id']}")
    ids, offsets = continuation_token_offsets(
        tokenizer,
        prefix_text=prefix_text,
        response_text=text,
        expected_prefix_ids=prefix_ids,
        expected_response_ids_sha256=frozen["response_ids_sha256"],
    )
    if len(ids) != frozen["response_token_count"]:
        raise ValueError(f"historical response count drift for {record['response_id']}")
    return (
        ids,
        offsets,
        {
            "kind": "frozen_historical_reconstruction",
            "model_revision": MODEL_REVISION,
            "historical_replay_status": historical["historical_replay_status"],
            "assistant_prefix_token_count": len(prefix_ids),
            "assistant_prefix_ids_sha256": canonical_sha256(prefix_ids),
            "response_token_count": len(ids),
            "response_ids_sha256": canonical_sha256(ids),
            "offset_alignment": "reconstructed_combined_continuation_offsets",
            "excluded_terminal_suffix_token_ids": [],
        },
    )


def _record_tokenization(
    tokenizer: Any, record: dict[str, Any], text: str
) -> tuple[list[int], list[list[int]], dict[str, Any]]:
    if record["source"] == "historical_dense_reconstruction":
        return _historical_tokenization(tokenizer, record, text)
    return _generated_tokenization(tokenizer, record, text)


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite annotation set: {args.output}")
    if text_sha256(SYSTEM_PROMPT) != SYSTEM_PROMPT_SHA256:
        raise AssertionError("historical system-prompt constant hash drift")
    cohort_manifest, cohort_rows = load_cohort(args.cohort)
    ontology = load_ontology(args.ontology)
    ontology_sha256 = file_sha256(args.ontology)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    if text_sha256(get_chat_template(tokenizer)) != CHAT_TEMPLATE_SHA256:
        raise ValueError("Thinking chat-template hash drift")
    tokenizer_files = _tokenizer_file_identity(args.tokenizer)
    review_ui_path = (
        Path(__file__).resolve().parent / "process_witness_annotation_review.html"
    )
    review_ui_sha256 = file_sha256(review_ui_path)

    build_path = args.output.with_name(args.output.name + f".building-{os.getpid()}")
    if build_path.exists():
        raise FileExistsError(f"stale build path exists: {build_path}")
    records_path = build_path / "records"
    records_path.mkdir(parents=True)
    documents: list[dict[str, Any]] = []
    document_sha256s: list[str] = []
    index_rows: list[dict[str, Any]] = []
    for cohort_row in cohort_rows:
        source_record_path = args.cohort / cohort_row["record_path"]
        if file_sha256(source_record_path) != cohort_row["record_sha256"]:
            raise ValueError(f"source record hash drift: {source_record_path}")
        record = json.loads(source_record_path.read_text(encoding="utf-8"))
        raw_text_path = args.cohort / cohort_row["raw_text_path"]
        if file_sha256(raw_text_path) != cohort_row["raw_text_sha256"]:
            raise ValueError(f"raw text hash drift: {raw_text_path}")
        text = raw_text_path.read_text(encoding="utf-8")
        if text != record["raw_response"]:
            raise ValueError(f"record/raw text mismatch for {record['response_id']}")
        ids, offsets, token_identity = _record_tokenization(tokenizer, record, text)
        document = annotate_response(
            response=record,
            text=text,
            ids=ids,
            offsets=offsets,
            token_identity=token_identity,
            ontology=ontology,
            ontology_sha256=ontology_sha256,
            cohort_id=cohort_manifest["cohort_id"],
            annotation_set_id=args.annotation_set_id,
        )
        document["review_ui"] = {
            "version": REVIEW_UI_VERSION,
            "sha256": review_ui_sha256,
        }
        output_name = f"{record['response_id']}.json"
        output_path = records_path / output_name
        write_json(output_path, document)
        document_sha256 = file_sha256(output_path)
        index_rows.append(
            {
                "response_id": record["response_id"],
                "prompt_sha256": record["prompt_sha256"],
                "task_family": document["task_context"]["task_family"],
                "source_types": document["task_context"]["source_types"],
                "question_ids": document["task_context"]["question_ids"],
                "source": record["source"],
                "trace_scope": record["trace_scope"],
                "annotation_status": document["annotation_status"],
                "token_count": document["tokenization"]["token_count"],
                "suggestion_count": len(document["suggestions"]),
                "record_path": f"records/{output_name}",
                "record_sha256": document_sha256,
            }
        )
        documents.append(document)
        document_sha256s.append(document_sha256)

    audit = audit_documents(documents)
    if audit["status"] != "passed":
        raise ValueError(f"annotation audit failed with {len(audit['errors'])} errors")
    inventory = build_inventory(documents)
    inspection = {
        "schema_version": "adag.process-witness.annotation-rule-inspection.v1",
        "status": "awaiting_human_review",
        "selection": "deterministic response-stratified matches across frozen cohort support, at most seven per rule",
        "rules": inspection_examples(documents),
    }
    write_jsonl(build_path / "index.jsonl", index_rows)
    write_json(build_path / "inventory.json", inventory)
    write_json(build_path / "audit.json", audit)
    write_json(build_path / "rule-inspection.json", inspection)
    write_compact_json(
        build_path / "workstation-bundle.json",
        build_workstation_bundle(
            documents,
            source_record_sha256s=document_sha256s,
            review_ui_version=REVIEW_UI_VERSION,
            review_ui_sha256=review_ui_sha256,
        ),
    )

    payload_files = sorted(path for path in build_path.rglob("*") if path.is_file())
    payload = [
        {
            "path": str(path.relative_to(build_path)),
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in payload_files
    ]
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "annotation_set_id": args.annotation_set_id,
        "status": "automatic_draft_frozen_for_review",
        "claim_boundary": (
            "This artifact contains graph-blind automatic suggestions, not reviewed "
            "annotations, process truth, trace targets, or ADAG evidence."
        ),
        "source_cohort": {
            "path": str(args.cohort),
            "cohort_id": cohort_manifest["cohort_id"],
            "manifest_sha256": file_sha256(args.cohort / "manifest.json"),
            "index_sha256": file_sha256(args.cohort / "index.jsonl"),
            "responses": len(cohort_rows),
        },
        "ontology": {
            "path": str(args.ontology),
            "ontology_id": ontology["ontology_id"],
            "sha256": ontology_sha256,
        },
        "tokenizer": {
            "model_revision": MODEL_REVISION,
            "local_snapshot_path": str(args.tokenizer),
            "files": tokenizer_files,
            "chat_template_sha256": CHAT_TEMPLATE_SHA256,
            "system_prompt_sha256_for_historical_reconstruction": (
                SYSTEM_PROMPT_SHA256
            ),
            "identity_contract": (
                "generated continuations exactly match captured generation IDs after "
                "removing at most one captured terminal default-stop suffix; historical "
                "continuations exactly match frozen reconstructed ID hashes"
            ),
        },
        "implementation": {
            "builder_path": str(Path(__file__).resolve()),
            "builder_sha256": file_sha256(Path(__file__).resolve()),
            "module_path": str(
                Path(__file__).resolve().parents[2]
                / "circuits/analysis/bonafide/process_annotation.py"
            ),
            "module_sha256": file_sha256(
                Path(__file__).resolve().parents[2]
                / "circuits/analysis/bonafide/process_annotation.py"
            ),
            "review_ui_path": str(review_ui_path),
            "review_ui_sha256": review_ui_sha256,
            "review_ui_version": REVIEW_UI_VERSION,
        },
        "counts": {
            "responses": len(documents),
            "tokens": inventory["tokens"],
            "suggestions": inventory["suggestions"],
        },
        "audit_status": audit["status"],
        "payload": payload,
        "payload_tree_sha256": canonical_sha256(payload),
    }
    write_json(build_path / "manifest.json", manifest)
    build_path.rename(args.output)
    for path in args.output.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
        elif path.is_dir():
            path.chmod(0o555)
    args.output.chmod(0o555)
    return manifest


def main() -> None:
    manifest = build(parse_args())
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
