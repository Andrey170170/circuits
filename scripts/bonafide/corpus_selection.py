"""Build deterministic prompt-candidate inventories for BonaFide tracing.

This stage selects *examples*, not response-token targets.  In particular, the
artifact deliberately contains no response positions and cannot be consumed as
a tracing manifest until a later span/target-selection stage has run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from circuits.tracing.trace import get_chat_template, tokenize_teacher_forced_response
from scripts.bonafide.manifest import resolve_pretrained_source
from transformers import AutoTokenizer, PreTrainedTokenizerBase


SCHEMA_VERSION = "bonafide-prompt-candidates/v1"
SELECTION_POLICY_VERSION = "bonafide-prompt-coverage-v1"
DEFAULT_TARGET_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_RECOMMENDED_DENSE_IDS = (
    "bf-2ed391444282be41b715",
    "bf-5f186d2224cd8a515ac9",
    "bf-89f277b79caf27f7f6ad",
    "bf-2981baca0442c8e8021f",
    "bf-6145690c43f611af97cb",
    "bf-c5acd500a6bcb288be61",
    "bf-662aa74003bb97f2ea07",
    "bf-d2b6d6de52232d107a08",
    "bf-a430b14be4b2c3a58ac5",
    "bf-3b3dc26f6e91f4bc543a",
)

TOKENIZER_FILE_NAMES = frozenset(
    {
        "added_tokens.json",
        "config.json",
        "merges.txt",
        "sentencepiece.bpe.model",
        "special_tokens_map.json",
        "spiece.model",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "vocab.json",
        "vocab.txt",
    }
)

DENSE_MAX_RESPONSE_TOKENS = 224
DENSE_MAX_TOTAL_TOKENS = 512
BROAD_MAX_RESPONSE_TOKENS = 768
BROAD_MAX_TOTAL_TOKENS = 1024
BROAD_PRIMARY_COUNT = 48
BROAD_ALTERNATE_COUNT = 24

REQUIRED_COLUMNS = {
    "id",
    "question_id",
    "label_type",
    "sentence_text",
    "sentence_span_start",
    "sentence_span_end",
    "extract",
    "extract_span_start",
    "extract_span_end",
    "labeling_reason",
    "target_model",
    "question",
    "prompt",
    "cot",
    "model_answer",
    "correct_answer",
    "hinted_answer",
    "src_type",
    "hint_dataset",
    "hint_type",
    "prompted_hint",
}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_tokenizer_file(path: Path, *, root: Path) -> bool:
    name = path.name.casefold()
    template_suffix = path.suffix.casefold() in {".jinja", ".json", ".txt"}
    relative_parts = tuple(part.casefold() for part in path.relative_to(root).parts)
    return (
        name in TOKENIZER_FILE_NAMES
        or template_suffix
        and ("chat_template" in name or "chat_templates" in relative_parts[:-1])
    )


def _tokenizer_file_manifest(tokenizer_path: Path | None) -> dict[str, Any]:
    """Return location-independent provenance for tokenizer loader inputs."""

    if tokenizer_path is None:
        return {
            "state": "unavailable_not_file_backed",
            "files": [],
            "aggregate_sha256": None,
        }
    if not tokenizer_path.is_dir():
        raise ValueError(f"tokenizer path is not a directory: {tokenizer_path}")

    records: list[dict[str, Any]] = []
    normalized_names: set[str] = set()
    for path in tokenizer_path.rglob("*"):
        if not path.is_file() or not _is_tokenizer_file(path, root=tokenizer_path):
            continue
        relative_name = unicodedata.normalize(
            "NFC", path.relative_to(tokenizer_path).as_posix()
        )
        if relative_name in normalized_names:
            raise ValueError(
                "tokenizer path contains duplicate normalized relative file name: "
                f"{relative_name}"
            )
        normalized_names.add(relative_name)
        records.append(
            {
                "path": relative_name,
                "size_bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        )
    records.sort(key=lambda record: record["path"])
    if not records:
        raise ValueError(
            "tokenizer path contains no recognized tokenizer/config/template files: "
            f"{tokenizer_path}"
        )
    return {
        "state": "file_backed",
        "files": records,
        "aggregate_sha256": _sha256_bytes(_canonical_json(records)),
    }


def _unique(values: Iterable[str]) -> list[str]:
    return sorted({value for value in values if value != ""})


def _optional_int(value: str) -> int | None:
    value = value.strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _normalized_answer(value: str) -> str:
    return " ".join(value.casefold().split())


def _answer_relation(
    rows: Sequence[Mapping[str, str]],
) -> tuple[str, list[dict[str, str]]]:
    answer_records = sorted(
        {
            (
                row["model_answer"],
                row["correct_answer"],
                row["hinted_answer"],
            )
            for row in rows
        }
    )
    records: list[dict[str, str]] = []
    relations: set[str] = set()
    for model_answer, correct_answer, hinted_answer in answer_records:
        model = _normalized_answer(model_answer)
        correct = _normalized_answer(correct_answer)
        hinted = _normalized_answer(hinted_answer)
        matches_correct = bool(model) and model == correct
        matches_hint = bool(model) and model == hinted
        if matches_correct and matches_hint:
            relation = "model_matches_correct_and_hint"
        elif matches_correct:
            relation = "model_matches_correct_only"
        elif matches_hint:
            relation = "model_matches_hint_only"
        else:
            relation = "model_matches_neither"
        relations.add(relation)
        records.append(
            {
                "model_answer": model_answer,
                "correct_answer": correct_answer,
                "hinted_answer": hinted_answer,
                "relation": relation,
            }
        )
    return (next(iter(relations)) if len(relations) == 1 else "mixed"), records


def _cot_phenotype(rows: Sequence[Mapping[str, str]]) -> str:
    labels = {row["label_type"] for row in rows}
    reasons = " | ".join(row["labeling_reason"].casefold() for row in rows)
    omission = "no acknowledgements of hint and no faithful steps" in reasons
    commission = "UNFAITHFUL_STEP" in labels or bool(
        re.search(r"contains \d+ unfaithful step", reasons)
    )
    if omission and commission:
        return "both"
    if omission:
        return "omission"
    if commission:
        return "commission"
    return "faithful"


def _valid_annotation_span(
    row: Mapping[str, str], *, kind: str, response: str
) -> tuple[int, int] | None:
    text = row[kind].strip()
    coordinate_prefix = "sentence" if kind == "sentence_text" else kind
    start = _optional_int(row[f"{coordinate_prefix}_span_start"])
    end = _optional_int(row[f"{coordinate_prefix}_span_end"])
    if (
        not text
        or start is None
        or end is None
        or not (0 <= start < end <= len(response))
    ):
        return None
    return start, end


def _annotation_position_bin(rows: Sequence[Mapping[str, str]], response: str) -> str:
    starts: list[int] = []
    for row in rows:
        for kind in ("sentence_text", "extract"):
            span = _valid_annotation_span(row, kind=kind, response=response)
            if span is not None:
                starts.append(span[0])
    if not starts or not response:
        return "no_valid_annotation_span"
    fraction = min(starts) / len(response)
    if fraction < 0.25:
        return "opening_quarter"
    if fraction < 0.50:
        return "second_quarter"
    if fraction < 0.75:
        return "third_quarter"
    return "closing_quarter"


def _response_length_bin(count: int) -> str:
    if count <= 64:
        return "001-064"
    if count <= 128:
        return "065-128"
    if count <= 224:
        return "129-224"
    if count <= 384:
        return "225-384"
    if count <= 512:
        return "385-512"
    if count <= 768:
        return "513-768"
    return "769-plus"


def _total_length_bin(count: int) -> str:
    if count <= 256:
        return "001-256"
    if count <= 512:
        return "257-512"
    if count <= 768:
        return "513-768"
    if count <= 1024:
        return "769-1024"
    return "1025-plus"


def _annotation_spans(
    rows: Sequence[Mapping[str, str]], response: str
) -> list[dict[str, Any]]:
    return [
        {
            "annotation_row_id": row["id"],
            "label_type": row["label_type"],
            "labeling_reason": row["labeling_reason"],
            "sentence_text": row["sentence_text"],
            "sentence_span_start": _optional_int(row["sentence_span_start"]),
            "sentence_span_end": _optional_int(row["sentence_span_end"]),
            "sentence_span_valid": _valid_annotation_span(
                row, kind="sentence_text", response=response
            )
            is not None,
            "extract": row["extract"],
            "extract_span_start": _optional_int(row["extract_span_start"]),
            "extract_span_end": _optional_int(row["extract_span_end"]),
            "extract_span_valid": _valid_annotation_span(
                row, kind="extract", response=response
            )
            is not None,
        }
        for row in rows
    ]


def _coverage_features(example: Mapping[str, Any]) -> list[str]:
    diversity = example["diversity"]
    features: list[str] = []
    for axis in ("label_types", "hint_types", "hint_datasets", "src_types"):
        features.extend(f"{axis}={value}" for value in diversity[axis])
    for axis in (
        "cot_phenotype",
        "answer_relation",
        "annotation_position_bin",
        "response_length_bin",
        "total_length_bin",
        "question_novelty_control_family_marker",
    ):
        features.append(f"{axis}={diversity[axis]}")
    return features


def _coverage_order(
    examples: Sequence[Mapping[str, Any]],
    count: int,
    *,
    initial_feature_counts: Counter[str] | None = None,
    unique_questions_first: bool,
) -> tuple[list[Mapping[str, Any]], Counter[str]]:
    """Greedily balance categorical coverage with stable example-ID ties."""

    available = {str(example["example_id"]): example for example in examples}
    selected: list[Mapping[str, Any]] = []
    feature_counts = Counter(initial_feature_counts or {})
    global_frequency = Counter(
        feature for example in examples for feature in _coverage_features(example)
    )
    selected_questions: set[str] = set()

    while available and len(selected) < count:
        candidates = list(available.values())
        if unique_questions_first:
            unused = [
                example
                for example in candidates
                if example["base_question_id"] not in selected_questions
            ]
            if unused:
                candidates = unused

        def rank(example: Mapping[str, Any]) -> tuple[Any, ...]:
            features = _coverage_features(example)
            new_values = sum(feature_counts[feature] == 0 for feature in features)
            existing_load = sum(feature_counts[feature] for feature in features)
            rarity = sum(1_000_000 // global_frequency[feature] for feature in features)
            return (-new_values, existing_load, -rarity, example["example_id"])

        chosen = min(candidates, key=rank)
        selected.append(chosen)
        selected_questions.add(str(chosen["base_question_id"]))
        feature_counts.update(_coverage_features(chosen))
        del available[str(chosen["example_id"])]

    return selected, feature_counts


def _read_target_rows(
    csv_path: Path, target_model: str
) -> tuple[list[str], list[dict[str, str]], int]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        headers = list(reader.fieldnames or ())
        missing = REQUIRED_COLUMNS.difference(headers)
        if missing:
            raise ValueError(f"BonaFide CSV is missing columns: {sorted(missing)}")
        all_rows = [dict(row) for row in reader]
    rows = [
        row
        for row in all_rows
        if row["target_model"] == target_model and row["cot"].strip()
    ]
    if not rows:
        raise ValueError(
            f"No non-empty BonaFide responses found for model {target_model!r}"
        )
    return headers, rows, len(all_rows)


def load_prompt_candidates(
    *,
    csv_path: Path,
    tokenizer: PreTrainedTokenizerBase,
    target_model: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Aggregate rich source metadata and exact tracing-tokenizer lengths."""

    headers, rows, source_row_count = _read_target_rows(csv_path, target_model)
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["target_model"], row["prompt"], row["cot"])].append(row)

    preliminary: list[dict[str, Any]] = []
    for (model, prompt, response), source_rows in grouped.items():
        source_rows = sorted(
            source_rows, key=lambda row: (row["id"], _canonical_json(row))
        )
        identity = {"target_model": model, "prompt": prompt, "response": response}
        example_id = f"bf-{_sha256_bytes(_canonical_json(identity))[:20]}"
        question_values = _unique(row["question"] for row in source_rows)
        canonical_question = question_values[0] if question_values else ""
        base_question_id = f"bfq-{_sha256_text(canonical_question)[:20]}"
        tokenized = tokenize_teacher_forced_response(tokenizer, prompt, response)
        prefix_count = len(tokenized.assistant_prefix_ids)
        response_count = len(tokenized.response_ids)
        suffix_count = len(tokenized.assistant_suffix_ids)
        maximum_input_count = prefix_count + response_count
        full_conversation_count = maximum_input_count + suffix_count
        answer_relation, answer_records = _answer_relation(source_rows)
        preliminary.append(
            {
                "example_id": example_id,
                "target_model": model,
                "question": canonical_question,
                "questions": question_values,
                "base_question_id": base_question_id,
                "prompt": prompt,
                "response": response,
                "prompted_hints": _unique(row["prompted_hint"] for row in source_rows),
                "annotation_row_ids": _unique(row["id"] for row in source_rows),
                "question_ids": _unique(row["question_id"] for row in source_rows),
                "label_types": _unique(row["label_type"] for row in source_rows),
                "labeling_reasons": _unique(
                    row["labeling_reason"] for row in source_rows
                ),
                "hint_types": _unique(row["hint_type"] for row in source_rows),
                "hint_datasets": _unique(row["hint_dataset"] for row in source_rows),
                "src_types": _unique(row["src_type"] for row in source_rows),
                "answer_records": answer_records,
                "annotation_spans": _annotation_spans(source_rows, response),
                "source_annotations": [
                    {header: row[header] for header in headers} for row in source_rows
                ],
                "token_counts": {
                    "assistant_prefix": prefix_count,
                    "response": response_count,
                    "assistant_suffix": suffix_count,
                    "maximum_teacher_forced_input": maximum_input_count,
                    "full_conversation_with_assistant_suffix": full_conversation_count,
                },
                "diversity": {
                    "label_types": _unique(row["label_type"] for row in source_rows),
                    "hint_types": _unique(row["hint_type"] for row in source_rows),
                    "hint_datasets": _unique(
                        row["hint_dataset"] for row in source_rows
                    ),
                    "src_types": _unique(row["src_type"] for row in source_rows),
                    "cot_phenotype": _cot_phenotype(source_rows),
                    "answer_relation": answer_relation,
                    "annotation_position_bin": _annotation_position_bin(
                        source_rows, response
                    ),
                    "response_length_bin": _response_length_bin(response_count),
                    "total_length_bin": _total_length_bin(maximum_input_count),
                    # Filled after all responses have been grouped by base question.
                    "question_novelty_control_family_marker": "",
                },
                "provenance": {
                    "identity_sha256": _sha256_bytes(_canonical_json(identity)),
                    "prompt_sha256": _sha256_text(prompt),
                    "response_sha256": _sha256_text(response),
                    "source_annotations_sha256": _sha256_bytes(
                        _canonical_json(source_rows)
                    ),
                },
            }
        )

    question_family_counts = Counter(
        example["base_question_id"] for example in preliminary
    )
    for example in preliminary:
        family_size = question_family_counts[example["base_question_id"]]
        example["question_family_size"] = family_size
        example["diversity"]["question_novelty_control_family_marker"] = (
            "control_family" if family_size > 1 else "novel_singleton"
        )
        response_count = example["token_counts"]["response"]
        total_count = example["token_counts"]["maximum_teacher_forced_input"]
        dense_eligible = (
            response_count <= DENSE_MAX_RESPONSE_TOKENS
            and total_count <= DENSE_MAX_TOTAL_TOKENS
        )
        broad_eligible = (
            not dense_eligible
            and response_count <= BROAD_MAX_RESPONSE_TOKENS
            and total_count <= BROAD_MAX_TOTAL_TOKENS
        )
        example["eligibility"] = {
            "dense_inventory": dense_eligible,
            "broad_inventory": broad_eligible,
            "dense_reasons": {
                "response_within_cap": response_count <= DENSE_MAX_RESPONSE_TOKENS,
                "total_context_within_cap": total_count <= DENSE_MAX_TOTAL_TOKENS,
            },
            "broad_reasons": {
                "excluded_from_dense_inventory": not dense_eligible,
                "response_within_cap": response_count <= BROAD_MAX_RESPONSE_TOKENS,
                "total_context_within_cap": total_count <= BROAD_MAX_TOTAL_TOKENS,
            },
        }

    return sorted(preliminary, key=lambda value: value["example_id"]), {
        "source_row_count": source_row_count,
        "target_model_annotation_row_count": len(rows),
        "target_model_deduplicated_example_count": len(preliminary),
    }


def _validate_recommended_dense_ids(
    examples: Sequence[Mapping[str, Any]], recommended_ids: Sequence[str]
) -> None:
    if len(set(recommended_ids)) != len(recommended_ids):
        raise ValueError("recommended dense example IDs must be unique")
    by_id = {example["example_id"]: example for example in examples}
    missing = [example_id for example_id in recommended_ids if example_id not in by_id]
    ineligible = [
        example_id
        for example_id in recommended_ids
        if example_id in by_id
        and not by_id[example_id]["eligibility"]["dense_inventory"]
    ]
    if missing or ineligible:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if ineligible:
            details.append(f"not_dense_eligible={ineligible}")
        raise ValueError("invalid recommended dense core: " + "; ".join(details))


def build_corpus_selection(
    *,
    csv_path: Path,
    tokenizer: PreTrainedTokenizerBase,
    target_model: str,
    model_revision: str,
    tokenizer_path: Path | None = None,
    recommended_dense_ids: Sequence[str] = DEFAULT_RECOMMENDED_DENSE_IDS,
    broad_primary_count: int = BROAD_PRIMARY_COUNT,
    broad_alternate_count: int = BROAD_ALTERNATE_COUNT,
) -> dict[str, Any]:
    if broad_primary_count < 0 or broad_alternate_count < 0:
        raise ValueError("broad selection counts cannot be negative")
    examples, source_counts = load_prompt_candidates(
        csv_path=csv_path, tokenizer=tokenizer, target_model=target_model
    )
    _validate_recommended_dense_ids(examples, recommended_dense_ids)

    dense = [
        example for example in examples if example["eligibility"]["dense_inventory"]
    ]
    broad = [
        example for example in examples if example["eligibility"]["broad_inventory"]
    ]
    requested_broad_count = broad_primary_count + broad_alternate_count
    if requested_broad_count > len(broad):
        raise ValueError(
            "requested broad primary + alternate counts exceed the eligible pool: "
            f"requested={requested_broad_count}, eligible={len(broad)}"
        )
    primary, feature_counts = _coverage_order(
        broad,
        broad_primary_count,
        unique_questions_first=True,
    )
    primary_ids = {example["example_id"] for example in primary}
    alternate_candidates = [
        example for example in broad if example["example_id"] not in primary_ids
    ]
    alternates, _ = _coverage_order(
        alternate_candidates,
        broad_alternate_count,
        initial_feature_counts=feature_counts,
        unique_questions_first=False,
    )
    alternate_ids = {example["example_id"] for example in alternates}
    remaining = [
        example
        for example in broad
        if example["example_id"] not in primary_ids | alternate_ids
    ]

    recommended_ids = set(recommended_dense_ids)
    for example in examples:
        example_id = example["example_id"]
        if example_id in primary_ids:
            broad_role = "primary"
        elif example_id in alternate_ids:
            broad_role = "alternate"
        elif example["eligibility"]["broad_inventory"]:
            broad_role = "remaining_eligible"
        else:
            broad_role = None
        example["coverage_features"] = _coverage_features(example)
        example["selection_membership"] = {
            "dense_inventory": example["eligibility"]["dense_inventory"],
            "recommended_dense_core": example_id in recommended_ids,
            "broad_eligible_inventory": example["eligibility"]["broad_inventory"],
            "broad_role": broad_role,
        }

    chat_template = get_chat_template(tokenizer)
    tokenizer_files = _tokenizer_file_manifest(tokenizer_path)
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "bonafide_prompt_candidates",
        "candidate_contract": {
            "selection_unit": "deduplicated_prompt_response_example",
            "prompt_candidates_selected": True,
            "target_spans_selected": False,
            "target_spans_frozen": False,
            "trace_work_items_created": False,
        },
        "dataset": {
            "path": str(csv_path),
            "sha256": _file_sha256(csv_path),
            "dedupe_key": ["target_model", "prompt", "cot"],
            **source_counts,
        },
        "tokenizer": {
            "model_id": target_model,
            "revision": model_revision,
            "class": type(tokenizer).__name__,
            "chat_template_sha256": _sha256_text(chat_template),
            "file_manifest": tokenizer_files,
            "length_semantics": {
                "helper": "circuits.tracing.trace.tokenize_teacher_forced_response",
                "eligibility_total": "assistant_prefix + complete_response",
                "assistant_suffix_excluded_from_trace_input_cap": True,
            },
        },
        "selection_policy": {
            "version": SELECTION_POLICY_VERSION,
            "dense": {
                "response_token_cap": DENSE_MAX_RESPONSE_TOKENS,
                "total_context_token_cap": DENSE_MAX_TOTAL_TOKENS,
                "inventory_rule": "all examples satisfying both caps",
                "recommended_core_is_explicit": True,
            },
            "broad": {
                "disjoint_from": "dense_inventory",
                "response_token_cap": BROAD_MAX_RESPONSE_TOKENS,
                "total_context_token_cap": BROAD_MAX_TOTAL_TOKENS,
                "primary_requested_count": broad_primary_count,
                "alternate_requested_count": broad_alternate_count,
                "ranking": "greedy categorical coverage; stable example_id tie-break",
                "primary_question_rule": (
                    "at most one response per base question until unique questions are exhausted"
                ),
            },
            "diversity_axes": [
                "nonexclusive_label_types",
                "hint_types",
                "hint_datasets",
                "src_types",
                "cot_phenotype",
                "answer_relation",
                "annotation_position_bin",
                "response_length_bin",
                "total_length_bin",
                "question_novelty_control_family_marker",
            ],
            "cot_phenotype_semantics": {
                "omission": "annotation reason reports no hint acknowledgement and no faithful steps",
                "commission": "UNFAITHFUL_STEP or an unfaithful-step count is present",
                "both": "both omission and commission signals are present",
                "faithful": "neither omission nor commission signal is present",
            },
        },
        "selections": {
            "dense_inventory": [example["example_id"] for example in dense],
            "recommended_dense_core": list(recommended_dense_ids),
            "broad_eligible_inventory": [example["example_id"] for example in broad],
            "broad_primary": [example["example_id"] for example in primary],
            "broad_alternates": [example["example_id"] for example in alternates],
            "broad_remaining_eligible": [
                example["example_id"] for example in remaining
            ],
        },
        "examples": examples,
    }


def write_corpus_selection(selection: Mapping[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(selection, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output_path)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("BonaFide.csv"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model", "--model-id", dest="model_id", default=DEFAULT_TARGET_MODEL
    )
    parser.add_argument(
        "--revision", required=True, help="Exact Hugging Face model revision"
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        help="Optional local snapshot path; model/revision remain the provenance identity",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow tokenizer downloads (default: local cache only)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pretrained_source = resolve_pretrained_source(
        model_id=args.model_id,
        revision=args.revision,
        local_files_only=not args.allow_download,
        explicit_path=args.tokenizer_path,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_source,
        revision=None if pretrained_source != args.model_id else args.revision,
        local_files_only=not args.allow_download,
    )
    selection = build_corpus_selection(
        csv_path=args.csv,
        tokenizer=tokenizer,
        target_model=args.model_id,
        model_revision=args.revision,
        tokenizer_path=(
            None if pretrained_source == args.model_id else Path(pretrained_source)
        ),
    )
    write_corpus_selection(selection, args.output)
    selected = selection["selections"]
    print(
        json.dumps(
            {
                "output": str(args.output),
                "examples": len(selection["examples"]),
                "dense_inventory": len(selected["dense_inventory"]),
                "recommended_dense_core": len(selected["recommended_dense_core"]),
                "broad_eligible": len(selected["broad_eligible_inventory"]),
                "broad_primary": len(selected["broad_primary"]),
                "broad_alternates": len(selected["broad_alternates"]),
                "broad_remaining": len(selected["broad_remaining_eligible"]),
                "target_spans_frozen": selection["candidate_contract"][
                    "target_spans_frozen"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
