"""Build the all-model v2 outright review and trace-target selection packet.

Unlike the immutable v1 browsing packet, v2 reconstructs exact response-token
identities at pinned tokenizer revisions.  It still performs no model forward
pass, ranking, tracing, or interpretation.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
import os
import re
import shutil
import uuid
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from circuits.analysis.bonafide.canonical import (
    canonical_json,
    canonical_sha256,
    file_sha256,
)
from circuits.analysis.bonafide.outright_review import (
    EXPECTED_COLUMNS,
    EXPECTED_SOURCE_SHA256,
    OUTRIGHT_SOURCE_TYPES,
    _label_flags,
    _parse_span,
    read_source_rows,
)
from circuits.tracing.trace import (
    get_chat_template,
    tokenize_historical_thinking_continuation,
    tokenize_teacher_forced_response,
)

SCHEMA_VERSION = "adag.raw-graph-observatory.outright-review.v2"
EXPORT_SCHEMA_VERSION = (
    "adag.raw-graph-observatory.outright-target-selection.v2"
)
TOKENIZER_REGISTRY_SCHEMA = (
    "adag.raw-graph-observatory.tokenizer-profiles.v2"
)
DEFAULT_REGISTRY_PATH = Path(
    "scripts/bonafide/outright_review_tokenizer_profiles_v2.json"
)
SNAPSHOT_MANIFEST_WHITELIST = frozenset(
    {
        "chat_template.jinja",
        "config.json",
        "generation_config.json",
        "merges.txt",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "vocab.json",
    }
)


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _atomic_write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"destination already exists: {path}")
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        with temporary.open("xb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_write_json(path: Path, value: object) -> None:
    _atomic_write_bytes(
        path,
        (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
            "utf-8"
        ),
    )


def _content_id(kind: str, value: object) -> str:
    return f"{kind}_{canonical_sha256(value)}"


def load_tokenizer_registry(path: Path) -> tuple[dict[str, Any], str]:
    """Load and structurally validate the frozen tokenizer profile registry."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"unreadable tokenizer registry: {path}") from error
    if not isinstance(value, dict):
        raise ValueError("tokenizer registry must be an object")
    if value.get("schema_version") != TOKENIZER_REGISTRY_SCHEMA:
        raise ValueError("tokenizer registry schema drift")
    prompts = value.get("prompt_provenance")
    profiles = value.get("profiles")
    if not isinstance(prompts, dict) or not isinstance(profiles, dict) or not profiles:
        raise ValueError("tokenizer registry lacks prompts or profiles")
    for prompt_name, prompt in prompts.items():
        if not isinstance(prompt, dict) or not isinstance(prompt.get("value"), str):
            raise ValueError(f"invalid system prompt profile: {prompt_name}")
        prompt["sha256"] = _text_sha256(prompt["value"])
    required = {
        "tokenizer_model_id",
        "tokenizer_revision",
        "local_snapshot",
        "snapshot_manifest_sha256",
        "serialization_mode",
        "system_prompt",
        "reconstruction_status",
    }
    for model, profile in profiles.items():
        if not isinstance(profile, dict) or not required <= set(profile):
            raise ValueError(f"incomplete tokenizer profile: {model}")
        if profile["serialization_mode"] not in {
            "assistant_turn",
            "historical_thinking_continuation",
        }:
            raise ValueError(f"invalid serialization mode: {model}")
        if profile["system_prompt"] not in prompts:
            raise ValueError(f"unknown system prompt profile: {model}")
        if profile["reconstruction_status"] != "reconstructed_at_pinned_revision":
            raise ValueError(f"unsafe reconstruction status: {model}")
        revision = profile["tokenizer_revision"]
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError(f"tokenizer revision is not a commit hash: {model}")
        if Path(profile["local_snapshot"]).name != revision:
            raise ValueError(f"snapshot/revision mismatch: {model}")
        manifest_sha256 = profile["snapshot_manifest_sha256"]
        if not re.fullmatch(r"[0-9a-f]{64}", manifest_sha256):
            raise ValueError(f"invalid snapshot manifest SHA256: {model}")
    return value, file_sha256(path)


def snapshot_manifest(snapshot: Path) -> list[dict[str, Any]]:
    """Hash the normalized root-level tokenizer/config whitelist."""

    files = []
    for relative_path in sorted(SNAPSHOT_MANIFEST_WHITELIST):
        path = snapshot / relative_path
        if path.is_file():
            files.append(
                {
                    "path": relative_path,
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
    if not files or not any(
        item["path"] in {"tokenizer.json", "tokenizer.model"} for item in files
    ):
        raise ValueError(f"snapshot lacks a whitelisted tokenizer artifact: {snapshot}")
    return files


def _default_tokenizer_loader(profile: Mapping[str, Any]) -> Any:
    from transformers import AutoTokenizer

    snapshot = Path(str(profile["local_snapshot"]))
    if not snapshot.is_dir():
        raise FileNotFoundError(f"pinned tokenizer snapshot is missing: {snapshot}")
    actual_manifest_sha256 = canonical_sha256(snapshot_manifest(snapshot))
    if actual_manifest_sha256 != profile["snapshot_manifest_sha256"]:
        raise ValueError(
            "pinned tokenizer snapshot manifest drift: "
            f"expected {profile['snapshot_manifest_sha256']}, "
            f"found {actual_manifest_sha256} at {snapshot}"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError(f"fast tokenizer with offsets required: {snapshot}")
    return tokenizer


def _flat_ints(value: Any, *, field: str) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError(f"{field} unexpectedly contains a batch")
        value = value[0]
    return [int(item) for item in value]


def _offsets(value: Any) -> list[tuple[int, int]]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list) and value[0] and isinstance(value[0][0], list):
        if len(value) != 1:
            raise ValueError("offset mapping unexpectedly contains a batch")
        value = value[0]
    return [(int(start), int(end)) for start, end in value]


def _profile_identity(
    model: str,
    profile: Mapping[str, Any],
    prompt_record: Mapping[str, Any],
    tokenizer: Any,
) -> dict[str, Any]:
    template = get_chat_template(tokenizer)
    if not isinstance(template, str) or not template:
        raise ValueError(f"tokenizer has no usable chat template: {model}")
    return {
        "model": model,
        "tokenizerModelId": profile["tokenizer_model_id"],
        "tokenizerRevision": profile["tokenizer_revision"],
        "snapshotManifestSha256": profile["snapshot_manifest_sha256"],
        "serializationMode": profile["serialization_mode"],
        "systemPromptName": profile["system_prompt"],
        "systemPromptSha256": prompt_record["sha256"],
        "systemPromptSource": prompt_record["source"],
        "systemPromptSourceRevision": prompt_record["source_revision"],
        "chatTemplateSha256": _text_sha256(template),
        "reconstructionStatus": profile["reconstruction_status"],
    }


def _tokenize_completion(
    *,
    tokenizer: Any,
    profile_identity: Mapping[str, Any],
    system_prompt: str,
    prompt: str,
    response: str,
    completion_id: str,
    annotations: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[list[Any]]]:
    mode = profile_identity["serializationMode"]
    if mode == "historical_thinking_continuation":
        tokenized = tokenize_historical_thinking_continuation(
            tokenizer, prompt, response, system_prompt=system_prompt
        )
    else:
        tokenized = tokenize_teacher_forced_response(
            tokenizer, prompt, response, system_prompt=system_prompt
        )
    response_ids = [int(value) for value in tokenized.response_ids]
    encoded = tokenizer(
        response,
        add_special_tokens=False,
        return_attention_mask=False,
        return_offsets_mapping=True,
    )
    standalone_ids = _flat_ints(encoded["input_ids"], field="input_ids")
    offsets = _offsets(encoded["offset_mapping"])
    if standalone_ids != response_ids:
        raise ValueError(
            f"{completion_id}: standalone response IDs do not equal exact "
            f"teacher-forced response IDs ({mode})"
        )
    if len(offsets) != len(response_ids):
        raise ValueError(f"{completion_id}: token/offset length mismatch")

    prefix_count = len(tokenized.assistant_prefix_ids)
    suffix_count = len(tokenized.assistant_suffix_ids)
    tokens: list[list[Any]] = []
    for token_id, (start, end) in zip(response_ids, offsets, strict=True):
        if not 0 <= start <= end <= len(response):
            raise ValueError(f"{completion_id}: invalid token character span")
        overlapping = [
            annotation
            for annotation in annotations
            if annotation["extractSpan"][1] > annotation["extractSpan"][0]
            and start < annotation["extractSpan"][1]
            and end > annotation["extractSpan"][0]
        ]
        overlap_polarities = sorted(
            {
                "unfaithful"
                if annotation["labelType"].upper().startswith("UNFAITHFUL")
                else "faithful"
                for annotation in overlapping
            }
        )
        annotation_polarity = (
            "mixed"
            if len(overlap_polarities) > 1
            else overlap_polarities[0]
            if overlap_polarities
            else "unlabeled"
        )
        tokens.append(
            [
                token_id,
                tokenizer.decode(
                    [token_id],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ),
                response[start:end],
                start,
                end,
                {
                    "unlabeled": "",
                    "faithful": "f",
                    "unfaithful": "u",
                    "mixed": "m",
                }[annotation_polarity],
                [
                    annotation["sourceRowId"] for annotation in overlapping
                ],
                sorted(
                    {annotation["labelType"] for annotation in overlapping}
                ),
            ]
        )
    response_hash = canonical_sha256(response_ids)
    tokenization = {
        **dict(profile_identity),
        "assistantPrefixTokenCount": prefix_count,
        "assistantPrefixIdsSha256": canonical_sha256(
            [int(value) for value in tokenized.assistant_prefix_ids]
        ),
        "responseTokenCount": len(response_ids),
        "responseIdsSha256": response_hash,
        "assistantSuffixTokenCount": suffix_count,
        "assistantSuffixIdsSha256": canonical_sha256(
            [int(value) for value in tokenized.assistant_suffix_ids]
        ),
        "serializedConversationTokenCount": (
            prefix_count + len(response_ids) + suffix_count
        ),
        "causalContextAtResponseEndTokenCount": prefix_count + len(response_ids),
        "offsetContract": (
            "fast standalone response offsets; standalone IDs exactly equal "
            "teacher-forced response IDs"
        ),
    }
    return tokenization, tokens


def assemble_target_review_payload(
    rows: Sequence[Mapping[str, str]],
    *,
    source_sha256: str,
    registry: Mapping[str, Any],
    registry_sha256: str,
    source_name: str = "BonaFide.csv",
    tokenizer_loader: Callable[[Mapping[str, Any]], Any] = _default_tokenizer_loader,
) -> dict[str, Any]:
    """Assemble all outright completions with exact trace-ready token identities."""

    grouped: dict[tuple[str, str, str], list[tuple[int, Mapping[str, str]]]] = (
        defaultdict(list)
    )
    source_row_count = 0
    for row_number, row in enumerate(rows, start=2):
        if tuple(row.keys()) != EXPECTED_COLUMNS:
            raise ValueError(f"row {row_number}: source schema drift")
        if row["src_type"].strip().lower() not in OUTRIGHT_SOURCE_TYPES:
            continue
        _parse_span(row, row_number=row_number, prefix="sentence", text_field="sentence_text")
        _parse_span(row, row_number=row_number, prefix="extract", text_field="extract")
        source_row_count += 1
        grouped[(row["target_model"], row["prompt"], row["cot"])].append(
            (row_number, row)
        )

    registry_profiles = registry["profiles"]
    models_in_source = {key[0] for key in grouped}
    missing = sorted(models_in_source - set(registry_profiles))
    extra = sorted(set(registry_profiles) - models_in_source)
    if missing:
        raise ValueError(f"models missing tokenizer profiles: {missing!r}")
    if extra:
        raise ValueError(f"tokenizer profiles not represented in source: {extra!r}")

    tokenizers: dict[str, Any] = {}
    resolved_profiles: dict[str, dict[str, Any]] = {}
    for model in sorted(models_in_source, key=str.casefold):
        profile = registry_profiles[model]
        tokenizer = tokenizer_loader(profile)
        tokenizers[model] = tokenizer
        prompt_record = registry["prompt_provenance"][profile["system_prompt"]]
        resolved_profiles[model] = _profile_identity(
            model, profile, prompt_record, tokenizer
        )

    completions: list[dict[str, Any]] = []
    tasks_by_id: dict[str, dict[str, Any]] = {}
    for (model, prompt, cot), source_rows in grouped.items():
        first = source_rows[0][1]
        invariant_fields = (
            "question",
            "model_answer",
            "correct_answer",
            "src_type",
            "hint_dataset",
            "hint_type",
            "prompted_hint",
            "hinted_answer",
        )
        for field in invariant_fields:
            values = {row[field] for _, row in source_rows}
            if len(values) != 1:
                raise ValueError(f"completion metadata drift for {field}: {values!r}")
        task_id = _content_id(
            "task",
            {
                "source_type": first["src_type"].strip().lower(),
                "question": first["question"],
                "correct_answer": first["correct_answer"],
            },
        )
        completion_id = _content_id(
            "completion", {"target_model": model, "prompt": prompt, "cot": cot}
        )
        annotations = []
        for row_number, row in source_rows:
            annotations.append(
                {
                    "sourceRowIndex": row_number,
                    "sourceRowId": row["id"],
                    "questionId": row["question_id"],
                    "labelType": row["label_type"],
                    "sentenceText": row["sentence_text"],
                    "sentenceSpan": _parse_span(
                        row,
                        row_number=row_number,
                        prefix="sentence",
                        text_field="sentence_text",
                    ),
                    "extract": row["extract"],
                    "extractSpan": _parse_span(
                        row,
                        row_number=row_number,
                        prefix="extract",
                        text_field="extract",
                    ),
                    "labelingReason": row["labeling_reason"],
                }
            )
        annotations.sort(key=lambda value: (value["sourceRowIndex"], value["sourceRowId"]))
        exact_labels = sorted({item["labelType"] for item in annotations})
        has_faithful, has_unfaithful = _label_flags(exact_labels)
        broad_label = (
            "mixed"
            if has_faithful and has_unfaithful
            else "contains-unfaithful"
            if has_unfaithful
            else "faithful-only"
        )
        profile = registry_profiles[model]
        prompt_record = registry["prompt_provenance"][profile["system_prompt"]]
        tokenization, tokens = _tokenize_completion(
            tokenizer=tokenizers[model],
            profile_identity=resolved_profiles[model],
            system_prompt=prompt_record["value"],
            prompt=prompt,
            response=cot,
            completion_id=completion_id,
            annotations=annotations,
        )
        completion = {
            "completionId": completion_id,
            "taskId": task_id,
            "model": model,
            "questionIds": sorted(
                {item["questionId"] for item in annotations if item["questionId"]}
            ),
            "question": first["question"],
            "prompt": prompt,
            "reasoning": cot,
            "modelAnswer": first["model_answer"],
            "correctAnswer": first["correct_answer"],
            "sourceType": first["src_type"].strip().lower(),
            "hintedAnswer": first["hinted_answer"],
            "hintDataset": first["hint_dataset"],
            "hintType": first["hint_type"],
            "promptedHint": first["prompted_hint"],
            "broadLabel": broad_label,
            "hasUnfaithful": has_unfaithful,
            "exactLabelTypes": exact_labels,
            "annotations": annotations,
            "statistics": {
                "responseTokens": len(tokens),
                "assistantPrefixTokens": tokenization["assistantPrefixTokenCount"],
                "causalContextAtResponseEndTokens": tokenization[
                    "causalContextAtResponseEndTokenCount"
                ],
                "serializedConversationTokens": tokenization[
                    "serializedConversationTokenCount"
                ],
                "characters": len(cot),
                "words": len(re.findall(r"\S+", cot)),
                "lines": cot.count("\n") + 1,
            },
            "tokenization": tokenization,
            "tokens": tokens,
        }
        completions.append(completion)
        tasks_by_id.setdefault(
            task_id,
            {
                "taskId": task_id,
                "sourceType": completion["sourceType"],
                "question": completion["question"],
                "correctAnswer": completion["correctAnswer"],
            },
        )

    completions.sort(
        key=lambda item: (
            item["question"].casefold(),
            item["sourceType"],
            item["model"].casefold(),
            item["completionId"],
        )
    )
    completion_counts_by_task = Counter(item["taskId"] for item in completions)
    tasks = [
        {**task, "completionCount": completion_counts_by_task[task["taskId"]]}
        for task in tasks_by_id.values()
    ]
    tasks.sort(key=lambda item: (item["question"].casefold(), item["taskId"]))
    model_names = sorted(models_in_source, key=str.casefold)
    models = [
        {
            "model": model,
            "completionCount": sum(item["model"] == model for item in completions),
        }
        for model in model_names
    ]
    label_types = sorted(
        {label for item in completions for label in item["exactLabelTypes"]}
    )

    def dimension_counts(field: str) -> dict[str, dict[str, int]]:
        values = sorted({str(item[field]) for item in completions}, key=str.casefold)
        return {
            value: {
                "completions": len(selected := [
                    item for item in completions if str(item[field]) == value
                ]),
                "source_rows": sum(len(item["annotations"]) for item in selected),
                "tasks": len({item["taskId"] for item in selected}),
            }
            for value in values
        }

    counts_by_label = {
        label: {
            "source_rows": sum(
                annotation["labelType"] == label
                for item in completions
                for annotation in item["annotations"]
            ),
            "completions": sum(label in item["exactLabelTypes"] for item in completions),
        }
        for label in label_types
    }
    response_lengths = [item["statistics"]["responseTokens"] for item in completions]
    counts = {
        "source_rows": source_row_count,
        "completions": len(completions),
        "tasks": len(tasks),
        "models": len(models),
        "by_model": dimension_counts("model"),
        "by_source_type": dimension_counts("sourceType"),
        "by_label_type": counts_by_label,
        "by_broad_label": {
            "faithful-only": sum(item["broadLabel"] == "faithful-only" for item in completions),
            "contains-unfaithful": sum(item["hasUnfaithful"] for item in completions),
            "mixed": sum(item["broadLabel"] == "mixed" for item in completions),
        },
        "response_token_range": {
            "minimum": min(response_lengths),
            "maximum": max(response_lengths),
            "total": sum(response_lengths),
        },
    }
    public_profiles = [resolved_profiles[model] for model in model_names]
    return {
        "schemaVersion": SCHEMA_VERSION,
        "exportSchemaVersion": EXPORT_SCHEMA_VERSION,
        "meta": {
            "title": "Outright Trace Target Review",
            "sourceName": source_name,
            "sourceSha256": source_sha256,
            "tokenizerRegistrySha256": registry_sha256,
            "tokenEncoding": [
                "tokenId",
                "tokenText",
                "surfaceText",
                "charStart",
                "charEnd",
                "annotationPolarityCode: empty|f|u|m",
                "overlappingAnnotationIds",
                "overlappingLabelTypes",
            ],
            "targetIdContract": {
                "algorithm": "target_ + sha256(utf8(canonical-json(identity)))",
                "canonicalJson": (
                    "UTF-8, sorted object keys, no whitespace, JSON string escaping"
                ),
                "identityFields": [
                    "completionId",
                    "responsePosition",
                    "serializationMode",
                    "systemPromptSha256",
                    "tokenId",
                    "tokenizerRevision",
                ],
            },
            "outrightDefinition": "src_type is exactly complex or graph",
            "modelPolicy": "all models, including Qwen",
            "deduplicationKey": ["target_model", "prompt", "cot"],
            "claimBoundary": (
                "Exploratory human candidate and token-target selection only. "
                "Token identities are reconstructed at pinned tokenizer revisions, "
                "not original generation-run identities. No tracing, scoring, neuron "
                "interpretation, or faithfulness validation occurs here."
            ),
        },
        "counts": counts,
        "models": models,
        "tasks": tasks,
        "labelTypes": label_types,
        "tokenizerProfiles": public_profiles,
        "completions": completions,
    }


_HTML_HEAD = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="light"><title>Outright Trace Target Review</title>
<style>
:root{--ink:#12182c;--muted:#687187;--rule:#d9dde6;--soft:#f6f7fb;--indigo:#405de6;--indigo-soft:#eef1ff;--green:#26723b;--green-soft:#e5f4e8;--red:#ad422d;--red-soft:#fbe8e2;--yellow:#e3aa18;--yellow-soft:#fff2bd;--blue:#2866cf;--blue-soft:#dce9ff;--mixed:#8e4c7a;--ui:Inter,ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;--prose:Georgia,"Times New Roman",serif}
*{box-sizing:border-box}html,body{height:100%;margin:0;color:var(--ink);background:#fff}body{font:14px var(--ui);overflow:hidden}button,input,select,textarea{font:inherit;color:inherit}button,select,input,textarea{background:#fff;border:1px solid #cbd0dc;border-radius:5px}button{cursor:pointer;min-height:34px;padding:0 11px}button:hover{border-color:#8994ac}button:focus-visible,input:focus-visible,select:focus-visible,textarea:focus-visible{outline:3px solid #b8c3ff;outline-offset:1px}.topbar{height:64px;border-bottom:1px solid var(--rule);display:flex;align-items:center;gap:18px;padding:0 22px}.title{font-size:19px;margin:0;letter-spacing:-.02em}.subtitle{font-size:11px;color:var(--muted)}.actions{margin-left:auto;display:flex;gap:8px;align-items:center}.check{display:flex;align-items:center;gap:7px;white-space:nowrap}.primary{color:#284bb8;border-color:#9eb0ff}.shell{height:calc(100vh - 64px);display:grid;grid-template-columns:220px 380px minmax(0,1fr)}.rail,.index{border-right:1px solid var(--rule);overflow:auto;min-width:0}.rail{padding:18px 0;display:flex;flex-direction:column}.section-title{font-size:10px;text-transform:uppercase;letter-spacing:.1em;color:var(--muted);margin:0 17px 9px}.model-list{display:flex;flex-direction:column}.model{border:0;border-left:3px solid transparent;border-radius:0;min-height:47px;padding:7px 14px;text-align:left;display:grid;grid-template-columns:minmax(0,1fr) auto;gap:7px;align-items:center}.model:hover{background:var(--soft)}.model.active{background:var(--indigo-soft);border-left-color:var(--indigo);color:#243fae}.model-name{overflow-wrap:anywhere;font-size:11px;font-weight:700}.count{font-size:10px;color:var(--muted);font-variant-numeric:tabular-nums}.rail-note{margin:auto 17px 0;padding-top:20px;color:var(--muted);font-size:10px;line-height:1.5}.index{display:flex;flex-direction:column}.filters{padding:15px;border-bottom:1px solid var(--rule);display:grid;grid-template-columns:1fr 1fr;gap:8px;position:sticky;top:0;background:#fff;z-index:2}.filters .wide{grid-column:1/-1}.filters select,.filters input{width:100%;min-width:0;min-height:34px;padding:0 8px}.result-summary{grid-column:1/-1;display:flex;align-items:center;justify-content:space-between;color:var(--muted);font-size:10px}.completion-list{padding-bottom:30px}.task-group{border-bottom:1px solid var(--rule)}.task-heading{font:13px/1.35 var(--prose);margin:0;padding:12px 15px 7px}.task-meta{font:10px var(--ui);display:block;color:var(--muted);margin-top:4px;text-transform:uppercase;letter-spacing:.06em}.completion-row{width:100%;border:0;border-radius:0;border-left:3px solid transparent;min-height:65px;text-align:left;padding:9px 13px;display:grid;grid-template-columns:18px minmax(0,1fr);gap:8px}.completion-row:hover{background:var(--soft)}.completion-row.active{background:var(--indigo-soft);border-left-color:var(--indigo)}.completion-row input{pointer-events:none;margin:3px 0}.row-model{font-size:11px;font-weight:700;overflow-wrap:anywhere}.row-meta{color:var(--muted);font-size:10px;margin-top:5px;display:flex;gap:6px;align-items:center;flex-wrap:wrap}.dot{width:6px;height:6px;border-radius:50%;display:inline-block}.dot.faithful{background:var(--green)}.dot.unfaithful,.dot.mixed{background:var(--red)}.target-count{color:var(--blue);font-weight:700}.reader{overflow:auto;min-width:0}.reader-inner{max-width:1000px;margin:0 auto;padding:22px 30px 80px}.reader-header{display:flex;gap:14px;align-items:start;border-bottom:1px solid var(--rule);padding-bottom:15px}.reader-task{font:18px/1.35 var(--prose);margin:0 0 6px}.reader-heading{min-width:0}.reader-byline{font-size:10px;color:var(--muted);overflow-wrap:anywhere}.status{margin-left:auto;font-size:10px;font-weight:800;text-transform:uppercase;letter-spacing:.07em}.status.faithful{color:var(--green)}.status.mixed,.status.unfaithful{color:var(--red)}.doc-section{border-bottom:1px solid var(--rule);padding:18px 0}.doc-section h2{font:750 10px var(--ui);text-transform:uppercase;letter-spacing:.1em;color:var(--muted);margin:0 0 10px}.prose{font:15px/1.62 var(--prose);white-space:pre-wrap;overflow-wrap:anywhere;margin:0}.stats{display:flex;gap:7px;flex-wrap:wrap;margin:12px 0 0}.stat{border:1px solid var(--rule);border-radius:4px;padding:5px 8px;font-size:10px;color:var(--muted)}.stat strong{color:var(--ink);font-variant-numeric:tabular-nums}.legend{display:flex;gap:13px;flex-wrap:wrap;font-size:10px;color:var(--muted);margin:0 0 12px}.swatch{display:inline-block;width:12px;height:12px;border-radius:3px;vertical-align:-2px;margin-right:4px}.swatch.faithful{background:var(--green-soft);border-bottom:3px solid var(--green)}.swatch.unfaithful{background:var(--red-soft);border-bottom:3px solid var(--red)}.swatch.draft{background:var(--yellow-soft);border:1px solid var(--yellow)}.swatch.saved{background:var(--blue-soft);border:1px solid var(--blue)}.tokens{font:15px/1.8 var(--prose);white-space:pre-wrap;overflow-wrap:anywhere}.token{font:inherit;line-height:1.35;min-height:25px;padding:1px 2px;margin:1px 0;border:0;border-radius:3px;background:transparent;white-space:pre-wrap}.token.faithful{background:var(--green-soft);box-shadow:inset 0 -3px var(--green)}.token.unfaithful{background:var(--red-soft);box-shadow:inset 0 -3px var(--red)}.token.mixed{background:#f3e3ee;box-shadow:inset 0 -3px var(--mixed)}.token.draft{background:var(--yellow-soft);outline:2px solid var(--yellow);box-shadow:inset 0 -3px var(--token-polarity,transparent)}.token.saved{background:var(--blue-soft);outline:2px solid var(--blue);box-shadow:inset 0 -3px var(--token-polarity,transparent)}.token[data-polarity=faithful]{--token-polarity:var(--green)}.token[data-polarity=unfaithful]{--token-polarity:var(--red)}.token[data-polarity=mixed]{--token-polarity:var(--mixed)}.target-editor{margin-top:15px;border:1px solid var(--rule);border-radius:6px;padding:13px;background:#fafbfe}.target-line{display:flex;gap:8px;align-items:center;flex-wrap:wrap}.target-summary{font-size:11px;font-variant-numeric:tabular-nums}.target-editor textarea{width:100%;min-height:68px;margin:9px 0;padding:8px;resize:vertical}.saved-list{margin-top:11px;display:flex;flex-direction:column;gap:7px}.saved-item{border-left:3px solid var(--blue);background:#f4f7ff;padding:8px 10px;display:grid;grid-template-columns:minmax(0,1fr) auto;gap:8px}.saved-title{font-size:11px;font-weight:750}.saved-comment{font-size:11px;color:#4e5870;white-space:pre-wrap;overflow-wrap:anywhere;margin-top:4px}.answer-grid{display:grid;grid-template-columns:1fr 1fr;gap:25px}.answer{font:15px/1.5 var(--prose);white-space:pre-wrap}.annotation{display:grid;grid-template-columns:125px minmax(0,1fr);gap:14px;padding:12px 0;border-top:1px solid #eceef3}.annotation:first-child{border-top:0}.annotation-key{font-size:9px;color:var(--muted);line-height:1.5}.annotation-type{font-weight:800}.annotation-type.faithful{color:var(--green)}.annotation-type.unfaithful{color:var(--red)}.annotation-extract{font:14px/1.5 var(--prose);margin:0 0 6px}.annotation-reason{font-size:11px;line-height:1.5;color:#545d70;margin:0}.empty{padding:30px 16px;color:var(--muted)}.warning{color:#8a5b00;font-size:11px}.hidden{display:none!important}@media(max-width:1080px){.shell{grid-template-columns:175px 315px minmax(0,1fr)}.reader-inner{padding-left:20px;padding-right:20px}.subtitle{display:none}}@media(max-width:780px){body{overflow:auto}.topbar{height:auto;min-height:60px;flex-wrap:wrap;padding:12px 14px}.actions{width:100%;margin:0;overflow:auto}.shell{height:auto;display:block}.rail,.index,.reader{overflow:visible;border-right:0}.rail{border-bottom:1px solid var(--rule);padding:10px 0}.model-list{flex-direction:row;overflow:auto;padding:0 10px}.model{flex:0 0 175px;border-left:0;border-bottom:3px solid transparent}.model.active{border-bottom-color:var(--indigo)}.rail-note{display:none}.filters{position:static}.completion-list{max-height:360px;overflow:auto}.index{border-bottom:1px solid var(--rule)}.reader-inner{padding:18px 14px 65px}.reader-header{flex-wrap:wrap}.status{margin-left:0;width:100%}.answer-grid,.annotation{grid-template-columns:1fr}.token{font-size:14px}.saved-item{grid-template-columns:1fr}}
.target-editor{margin:12px 0;position:sticky;top:0;z-index:4;box-shadow:0 4px 14px rgba(25,35,70,.08)}
.tokens{font:15px/1.62 var(--prose)}.token{appearance:none;display:inline;font:inherit;line-height:inherit;min-height:0;padding:0;margin:0;border:0;border-radius:2px;background:transparent;color:inherit;vertical-align:baseline;white-space:pre-wrap;overflow-wrap:inherit;cursor:pointer;box-shadow:inset 0 0 0 1px rgba(115,123,140,.28),inset 0 -2px var(--token-polarity,transparent);box-decoration-break:clone;-webkit-box-decoration-break:clone;transition:background-color 90ms ease,box-shadow 90ms ease}.token.faithful,.token.unfaithful,.token.mixed{background:transparent}.token:hover:not(.draft):not(.saved){background:#fff8d9;box-shadow:inset 0 0 0 1px #d7bd69,inset 0 -2px var(--token-polarity,transparent)}.token.draft{background:var(--yellow-soft);outline:0;box-shadow:inset 0 0 0 2px var(--yellow),inset 0 -2px var(--token-polarity,transparent)}.token.saved{background:var(--blue-soft);outline:0;box-shadow:inset 0 0 0 2px var(--blue),inset 0 -2px var(--token-polarity,transparent)}.token:focus-visible{outline:2px solid #b8c3ff;outline-offset:1px}.token.structural{display:inline;font:inherit;line-height:inherit;min-height:0;padding:0;margin:0;border:0;background:transparent}.token.component-overlap{font-family:var(--ui);font-size:10px;color:#6e5c8f;background:#f5f0fa;box-shadow:inset 0 0 0 1px #b8a9cc}.token.token-whitespace{color:inherit;background:transparent}.token.token-linebreak,.token.token-linebreak:hover:not(.draft):not(.saved){box-shadow:none}.token.token-linebreak.draft{box-shadow:inset 0 -2px var(--yellow)}.token.token-linebreak.saved{box-shadow:inset 0 -2px var(--blue)}
@media(max-width:780px){.target-editor{position:static}}
</style></head><body>
<header class="topbar"><div><h1 class="title">Outright Trace Target Review</h1><div class="subtitle" id="source-note"></div></div><div class="actions"><label class="check"><input id="selection-only" type="checkbox">Candidates only</label><label class="check"><input id="saved-only" type="checkbox">Saved targets only</label><button id="export" class="primary" type="button">Export targets + comments</button><button id="previous" type="button">Previous</button><button id="next" type="button">Next</button></div></header>
<main class="shell"><aside class="rail"><h2 class="section-title">Model</h2><nav id="model-list" class="model-list"></nav><p id="rail-note" class="rail-note"></p></aside><section class="index"><div class="filters"><select id="task-filter" class="wide" data-filter="task"><option value="">All exact tasks</option></select><select id="source-filter" data-filter="source-type"><option value="">All source types</option><option value="complex">Complex</option><option value="graph">Graph</option></select><select id="broad-filter" data-filter="broad-label"><option value="">All broad labels</option><option value="faithful-only">Faithful only</option><option value="contains-unfaithful">Contains unfaithful</option><option value="mixed">Mixed</option></select><select id="label-filter" class="wide" data-filter="exact-label-type"><option value="">All exact labels</option></select><input id="search" class="wide" data-filter="search" type="search" placeholder="Search prompt, reasoning, answer…"><input id="max-tokens" type="number" min="1" data-filter="max-response-tokens" placeholder="Max tokens"><select id="sort" data-filter="sort"><option value="source">Source order</option><option value="tokens-asc">Tokens: shortest</option><option value="tokens-desc">Tokens: longest</option></select><div class="result-summary"><span id="result-count"></span><button id="reset" type="button">Reset filters</button></div></div><div id="completion-list" class="completion-list"></div></section><article class="reader"><div id="reader-inner" class="reader-inner"></div></article></main>
'''


_HTML_SCRIPT = r'''
<script>
"use strict";
async function decodePayload(){const bytes=Uint8Array.from(atob("__PAYLOAD__"),c=>c.charCodeAt(0));if(typeof DecompressionStream!=="function")throw new Error("This review packet requires a browser with gzip DecompressionStream support.");const stream=new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));return JSON.parse(await new Response(stream).text())}
async function main(){
const payload=await decodePayload();
const payloadSha="__PAYLOAD_SHA__";
const storageKey="outright-target-review:"+payloadSha;
const completionById=new Map(payload.completions.map(item=>[item.completionId,item]));
const state={model:"",task:"",source:"",broad:"",label:"",search:"",maxTokens:"",sort:"source",selectionOnly:false,savedOnly:false,active:"",selected:new Set(),draft:null,saved:new Map()};
function validPosition(completionId,position){const item=completionById.get(completionId);return !!item&&Number.isInteger(position)&&position>=0&&position<item.tokens.length}
function targetKey(completionId,position,tokenId){return completionId+":"+position+":"+tokenId}
function tokenAt(item,position){const raw=item.tokens[position];const polarities={f:"faithful",u:"unfaithful",m:"mixed","":"unlabeled"};return{position,tokenId:raw[0],tokenText:raw[1],surfaceText:raw[2],charSpan:[raw[3],raw[4]],annotationPolarity:polarities[raw[5]],overlappingAnnotationIds:raw[6],overlappingLabelTypes:raw[7],tokensBefore:position,causalPrefixTokenCount:item.tokenization.assistantPrefixTokenCount+position,predictionPosition:item.tokenization.assistantPrefixTokenCount+position-1,key:targetKey(item.completionId,position,raw[0])}}
try{const raw=JSON.parse(localStorage.getItem(storageKey)||"null");if(raw&&raw.schemaVersion===2){if(Array.isArray(raw.selected))raw.selected.filter(id=>completionById.has(id)).forEach(id=>state.selected.add(id));if(raw.draft&&validPosition(raw.draft.completionId,raw.draft.position))state.draft={completionId:raw.draft.completionId,position:raw.draft.position,comment:String(raw.draft.comment||"").slice(0,4000)};if(Array.isArray(raw.saved))raw.saved.forEach(entry=>{if(entry&&validPosition(entry.completionId,entry.position)){const item=completionById.get(entry.completionId);const token=tokenAt(item,entry.position);if(entry.key===token.key)state.saved.set(token.key,{completionId:entry.completionId,position:entry.position,comment:String(entry.comment||"").slice(0,4000)})}})}}catch(_error){}
const $=id=>document.getElementById(id);const nodes={models:$("model-list"),list:$("completion-list"),reader:$("reader-inner"),task:$("task-filter"),source:$("source-filter"),broad:$("broad-filter"),label:$("label-filter"),search:$("search"),max:$("max-tokens"),sort:$("sort"),only:$("selection-only"),savedOnly:$("saved-only"),count:$("result-count")};
function el(tag,className,text){const node=document.createElement(tag);if(className)node.className=className;if(text!==undefined)node.textContent=String(text);return node}
function shortModel(value){const parts=value.split("/");return parts[parts.length-1]}
function polarity(label){return label.toUpperCase().startsWith("UNFAITHFUL")?"unfaithful":"faithful"}
function savedFor(completionId){return [...state.saved.values()].filter(value=>value.completionId===completionId).sort((a,b)=>a.position-b.position)}
function persist(){const value={schemaVersion:2,selected:[...state.selected].sort(),draft:state.draft,saved:[...state.saved.entries()].map(([key,value])=>({key,...value})).sort((a,b)=>a.completionId.localeCompare(b.completionId)||a.position-b.position)};localStorage.setItem(storageKey,JSON.stringify(value))}
function addOption(select,value,text){const option=el("option","",text);option.value=value;select.append(option)}
payload.tasks.forEach(task=>addOption(nodes.task,task.taskId,task.question+" · "+task.sourceType));payload.labelTypes.forEach(label=>addOption(nodes.label,label,label));
function matches(item){if(state.model&&item.model!==state.model)return false;if(state.task&&item.taskId!==state.task)return false;if(state.source&&item.sourceType!==state.source)return false;if(state.broad==="faithful-only"&&item.broadLabel!=="faithful-only")return false;if(state.broad==="contains-unfaithful"&&!item.hasUnfaithful)return false;if(state.broad==="mixed"&&item.broadLabel!=="mixed")return false;if(state.label&&!item.exactLabelTypes.includes(state.label))return false;if(state.selectionOnly&&!state.selected.has(item.completionId))return false;if(state.savedOnly&&!savedFor(item.completionId).length)return false;if(state.maxTokens&&item.statistics.responseTokens>Number(state.maxTokens))return false;if(state.search){const hay=[item.model,item.question,item.prompt,item.reasoning,item.modelAnswer,item.correctAnswer,...item.annotations.flatMap(a=>[a.labelType,a.extract,a.sentenceText,a.labelingReason])].join("\n").toLocaleLowerCase();if(!hay.includes(state.search))return false}return true}
function visible(){const items=payload.completions.filter(matches);if(state.sort==="tokens-asc")items.sort((a,b)=>a.statistics.responseTokens-b.statistics.responseTokens||a.completionId.localeCompare(b.completionId));if(state.sort==="tokens-desc")items.sort((a,b)=>b.statistics.responseTokens-a.statistics.responseTokens||a.completionId.localeCompare(b.completionId));return items}
function setActive(id){state.active=id;renderList();renderReader();requestAnimationFrame(()=>{const row=document.querySelector('[data-completion-id="'+CSS.escape(id)+'"]');if(row)row.scrollIntoView({block:"nearest"})})}
function toggleCandidate(id){state.selected.has(id)?state.selected.delete(id):state.selected.add(id);persist();renderAll()}
function renderModels(){nodes.models.replaceChildren();[{model:"",completionCount:payload.counts.completions},...payload.models].forEach(value=>{const button=el("button","model"+(state.model===value.model?" active":""));button.type="button";button.setAttribute("aria-pressed",String(state.model===value.model));button.append(el("span","model-name",value.model?shortModel(value.model):"All models"),el("span","count",value.completionCount));button.addEventListener("click",()=>{state.model=value.model;renderAll()});nodes.models.append(button)})}
function renderList(){const items=visible();nodes.count.textContent=items.length+" of "+payload.counts.completions+" completions";nodes.list.replaceChildren();if(!items.length){nodes.list.append(el("p","empty","No completions match these filters."));return}if(!items.some(item=>item.completionId===state.active))state.active=items[0].completionId;const groups=new Map();items.forEach(item=>{if(!groups.has(item.taskId))groups.set(item.taskId,[]);groups.get(item.taskId).push(item)});groups.forEach(group=>{const wrap=el("section","task-group");const heading=el("h3","task-heading",group[0].question);heading.append(el("span","task-meta",group[0].sourceType+" · "+group.length+" completion"+(group.length===1?"":"s")));wrap.append(heading);group.forEach(item=>{const button=el("button","completion-row"+(state.active===item.completionId?" active":""));button.type="button";button.dataset.completionId=item.completionId;const check=document.createElement("input");check.type="checkbox";check.checked=state.selected.has(item.completionId);check.tabIndex=-1;check.setAttribute("aria-hidden","true");const body=el("span");body.append(el("span","row-model",shortModel(item.model)));const meta=el("span","row-meta");meta.append(el("span","dot "+(item.hasUnfaithful?"unfaithful":"faithful")),document.createTextNode(item.broadLabel+" · "+item.statistics.responseTokens+" tokens"));const n=savedFor(item.completionId).length;if(n)meta.append(el("span","target-count",n+" saved"));body.append(meta);button.append(check,body);button.addEventListener("click",()=>setActive(item.completionId));wrap.append(button)});nodes.list.append(wrap)})}
function section(title){const wrap=el("section","doc-section");wrap.append(el("h2","",title));return wrap}
function stat(label,value){const node=el("span","stat");node.append(document.createTextNode(label+" "),el("strong","",value));return node}
function tokenTitle(item,token,structuralNote=""){return ["Raw surface: "+JSON.stringify(token.surfaceText),structuralNote,"Token ID: "+token.tokenId,"Response position (0-based): "+token.position,"Response tokens before: "+token.tokensBefore,"Assistant prefix tokens: "+item.tokenization.assistantPrefixTokenCount,"Total causal prefix: "+token.causalPrefixTokenCount,"Prediction position: "+token.predictionPosition].filter(Boolean).join("\n")}
async function contentTargetId(item,token){const identity={completionId:item.completionId,responsePosition:token.position,serializationMode:item.tokenization.serializationMode,systemPromptSha256:item.tokenization.systemPromptSha256,tokenId:token.tokenId,tokenizerRevision:item.tokenization.tokenizerRevision};const bytes=new TextEncoder().encode(JSON.stringify(identity));const digest=new Uint8Array(await crypto.subtle.digest("SHA-256",bytes));return "target_"+[...digest].map(value=>value.toString(16).padStart(2,"0")).join("")}
function selectDraft(item,token){const isSameDraft=state.draft&&state.draft.completionId===item.completionId&&state.draft.position===token.position;if(isSameDraft){state.draft=null;persist();renderReader();return}const existing=state.saved.get(token.key);state.draft={completionId:item.completionId,position:token.position,comment:existing?existing.comment:""};persist();renderReader()}
function saveDraft(){if(!state.draft)return;const item=completionById.get(state.draft.completionId);const token=tokenAt(item,state.draft.position);const textarea=$("target-comment");const comment=String(textarea?textarea.value:state.draft.comment||"").slice(0,4000);state.saved.set(token.key,{completionId:item.completionId,position:token.position,comment});state.selected.add(item.completionId);state.draft=null;persist();renderAll()}
function clearDraft(){state.draft=null;persist();renderReader()}
function removeSaved(key){state.saved.delete(key);persist();renderAll()}
function renderReader(){nodes.reader.replaceChildren();const item=completionById.get(state.active);if(!item||!matches(item)){nodes.reader.append(el("p","empty","Choose a completion to inspect."));return}const header=el("header","reader-header");const check=el("label","check");const input=document.createElement("input");input.type="checkbox";input.checked=state.selected.has(item.completionId);input.setAttribute("aria-label","Select completion as candidate");input.addEventListener("change",()=>toggleCandidate(item.completionId));check.append(input,document.createTextNode(" Candidate"));const heading=el("div","reader-heading");heading.append(el("h2","reader-task",item.question),el("div","reader-byline",item.model+" · "+item.sourceType+" · "+item.completionId));header.append(check,heading,el("div","status "+(item.hasUnfaithful?"mixed":"faithful"),item.broadLabel));nodes.reader.append(header);
const prompt=section("Full prompt");prompt.append(el("p","prose",item.prompt));nodes.reader.append(prompt);
const reasoning=section("Reasoning · choose an exact response token");const stats=el("div","stats");stats.append(stat("Response tokens",item.statistics.responseTokens),stat("Assistant prefix",item.statistics.assistantPrefixTokens),stat("Causal context at response end",item.statistics.causalContextAtResponseEndTokens),stat("Serialized conversation",item.statistics.serializedConversationTokens),stat("Characters",item.statistics.characters),stat("Words",item.statistics.words),stat("Lines",item.statistics.lines));reasoning.append(stats);const legend=el("div","legend");const hasLocalizedUnfaithful=item.tokens.some(raw=>raw[5]==="u"||raw[5]==="m");[["faithful","Localized faithful overlap"],["unfaithful","Localized unfaithful overlap"+(hasLocalizedUnfaithful?"":" (none in this completion)")],["draft","Draft target"],["saved","Saved target"]].forEach(([kind,label])=>{const part=el("span");part.append(el("i","swatch "+kind),document.createTextNode(label));legend.append(part)});reasoning.append(legend);const tokens=el("div","tokens");let renderedEnd=0;item.tokens.forEach((_raw,position)=>{const token=tokenAt(item,position);const visibleStart=Math.max(renderedEnd,token.charSpan[0]);const visibleSurface=token.charSpan[1]>visibleStart?item.reasoning.slice(visibleStart,token.charSpan[1]):"";const fullyOverlapped=!visibleSurface;const whitespaceOnly=!fullyOverlapped&&/^\s+$/.test(visibleSurface);renderedEnd=Math.max(renderedEnd,token.charSpan[1]);const isDraft=state.draft&&state.draft.completionId===item.completionId&&state.draft.position===token.position;const isSaved=state.saved.has(token.key);const classes=["token",token.annotationPolarity];if(fullyOverlapped)classes.push("structural","component-overlap");else{if(whitespaceOnly)classes.push("token-whitespace");if(/[\r\n]/.test(visibleSurface))classes.push("token-linebreak")}if(isDraft)classes.push("draft");else if(isSaved)classes.push("saved");const label=fullyOverlapped?"◌":visibleSurface;const structuralNote=fullyOverlapped?"Component token: its character span is fully covered by earlier token offsets; ◌ prevents duplicate text.":whitespaceOnly?"Whitespace-only token; its original spacing and line breaks are preserved.":"";const button=el("span",classes.join(" "),label||"∅");button.setAttribute("role","button");button.tabIndex=0;button.dataset.position=String(token.position);button.dataset.polarity=token.annotationPolarity;button.dataset.surfaceKind=fullyOverlapped?"component-overlap":whitespaceOnly?"whitespace":"text";button.title=tokenTitle(item,token,structuralNote);button.setAttribute("aria-label","Select response token "+token.position+", token ID "+token.tokenId+(structuralNote?", "+structuralNote:""));button.setAttribute("aria-pressed",String(!!isDraft||isSaved));button.addEventListener("click",()=>selectDraft(item,token));button.addEventListener("keydown",event=>{if(event.key==="Enter"||event.key===" "){event.preventDefault();selectDraft(item,token)}});tokens.append(button)});reasoning.append(tokens);
const draft=state.draft&&state.draft.completionId===item.completionId?tokenAt(item,state.draft.position):null;const editor=el("div","target-editor");if(draft){const summary=el("div","target-line");summary.append(el("strong","target-summary","Draft · position "+draft.position+" · ID "+draft.tokenId+" · "+draft.tokensBefore+" response tokens before · "+draft.causalPrefixTokenCount+" total causal prefix"));editor.append(summary);const textarea=document.createElement("textarea");textarea.id="target-comment";textarea.maxLength=4000;textarea.placeholder="Optional comment about why this is a useful tracing target…";textarea.value=state.draft.comment||"";textarea.addEventListener("input",()=>{if(state.draft){state.draft.comment=textarea.value.slice(0,4000);persist()}});editor.append(textarea);const actions=el("div","target-line");const save=el("button","primary",state.saved.has(draft.key)?"Update saved target":"Save target");save.type="button";save.addEventListener("click",saveDraft);const clear=el("button","","Clear draft");clear.type="button";clear.addEventListener("click",clearDraft);actions.append(save,clear);editor.append(actions)}else editor.append(el("span","target-summary","Select a token below to prepare a target. Click the yellow draft again to deselect it; saved targets are blue."));const saved=savedFor(item.completionId);if(saved.length){const list=el("div","saved-list");saved.forEach(entry=>{const token=tokenAt(item,entry.position);const row=el("div","saved-item");const body=el("div");body.append(el("div","saved-title","Position "+token.position+" · token ID "+token.tokenId+" · "+JSON.stringify(token.surfaceText)));if(entry.comment)body.append(el("div","saved-comment",entry.comment));const remove=el("button","","Remove");remove.type="button";remove.setAttribute("aria-label","Remove saved target at position "+token.position);remove.addEventListener("click",()=>removeSaved(token.key));row.append(body,remove);list.append(row)});editor.append(list)}reasoning.append(editor);nodes.reader.append(reasoning);
reasoning.insertBefore(editor,tokens);
const answers=section("Answers");const grid=el("div","answer-grid");const model=el("div");model.append(el("h2","","Model answer"),el("div","answer",item.modelAnswer||"—"));const correct=el("div");correct.append(el("h2","","Correct answer"),el("div","answer",item.correctAnswer||"—"));grid.append(model,correct);answers.append(grid);nodes.reader.append(answers);
const annotationSection=section("Annotations · "+item.annotations.length);item.annotations.forEach(a=>{const row=el("article","annotation");const key=el("div","annotation-key");key.append(el("div","annotation-type "+polarity(a.labelType),a.labelType),el("div","",a.sourceRowId),el("div","","row "+a.sourceRowIndex),el("div","","extract ["+a.extractSpan.join(", ")+"]"));const body=el("div");body.append(el("p","annotation-extract",a.extract||"Whole-completion annotation"),el("p","annotation-reason",a.labelingReason||"No reason supplied."));row.append(key,body);annotationSection.append(row)});nodes.reader.append(annotationSection)}
function renderAll(){renderModels();renderList();renderReader()}
function move(delta){const items=visible();if(!items.length)return;let index=items.findIndex(item=>item.completionId===state.active);index=index<0?0:(index+delta+items.length)%items.length;setActive(items[index].completionId)}
nodes.task.addEventListener("change",e=>{state.task=e.target.value;renderAll()});nodes.source.addEventListener("change",e=>{state.source=e.target.value;renderAll()});nodes.broad.addEventListener("change",e=>{state.broad=e.target.value;renderAll()});nodes.label.addEventListener("change",e=>{state.label=e.target.value;renderAll()});nodes.search.addEventListener("input",e=>{state.search=e.target.value.toLocaleLowerCase().trim();renderAll()});nodes.max.addEventListener("input",e=>{state.maxTokens=e.target.value;renderAll()});nodes.sort.addEventListener("change",e=>{state.sort=e.target.value;renderAll()});nodes.only.addEventListener("change",e=>{state.selectionOnly=e.target.checked;renderAll()});nodes.savedOnly.addEventListener("change",e=>{state.savedOnly=e.target.checked;renderAll()});$("previous").addEventListener("click",()=>move(-1));$("next").addEventListener("click",()=>move(1));
$("reset").addEventListener("click",()=>{Object.assign(state,{model:"",task:"",source:"",broad:"",label:"",search:"",maxTokens:"",sort:"source",selectionOnly:false,savedOnly:false});[nodes.task,nodes.source,nodes.broad,nodes.label,nodes.search,nodes.max].forEach(node=>node.value="");nodes.sort.value="source";nodes.only.checked=false;nodes.savedOnly.checked=false;renderAll()});
$("export").addEventListener("click",async()=>{if(state.draft&&!confirm("A draft target is not saved and will be omitted. Export saved targets anyway?"))return;const selected=[...state.selected].sort();const targets=await Promise.all([...state.saved.values()].map(async entry=>{const item=completionById.get(entry.completionId);const token=tokenAt(item,entry.position);return{targetId:await contentTargetId(item,token),completionId:item.completionId,taskId:item.taskId,model:item.model,question:item.question,sourceType:item.sourceType,broadLabel:item.broadLabel,exactLabelTypes:item.exactLabelTypes,sourceRowIds:item.annotations.map(annotation=>annotation.sourceRowId),sourceAnnotations:item.annotations,responsePosition:token.position,tokenId:token.tokenId,tokenText:token.tokenText,surfaceText:token.surfaceText,charSpan:token.charSpan,responseTokensBefore:token.tokensBefore,responseTokenCount:item.statistics.responseTokens,assistantPrefixTokenCount:item.tokenization.assistantPrefixTokenCount,causalPrefixTokenCount:token.causalPrefixTokenCount,predictionPosition:token.predictionPosition,causalContextAtResponseEndTokenCount:item.tokenization.causalContextAtResponseEndTokenCount,serializedConversationTokenCount:item.tokenization.serializedConversationTokenCount,responseIdsSha256:item.tokenization.responseIdsSha256,assistantPrefixIdsSha256:item.tokenization.assistantPrefixIdsSha256,tokenizerModelId:item.tokenization.tokenizerModelId,tokenizerRevision:item.tokenization.tokenizerRevision,snapshotManifestSha256:item.tokenization.snapshotManifestSha256,serializationMode:item.tokenization.serializationMode,systemPromptSha256:item.tokenization.systemPromptSha256,chatTemplateSha256:item.tokenization.chatTemplateSha256,reconstructionStatus:item.tokenization.reconstructionStatus,overlappingAnnotationIds:token.overlappingAnnotationIds,overlappingLabelTypes:token.overlappingLabelTypes,localContext:item.tokens.slice(Math.max(0,token.position-8),Math.min(item.tokens.length,token.position+9)).map((_raw,index)=>{const contextPosition=Math.max(0,token.position-8)+index;const value=tokenAt(item,contextPosition);return{position:value.position,tokenId:value.tokenId,surfaceText:value.surfaceText}}),comment:entry.comment}}));targets.sort((a,b)=>a.completionId.localeCompare(b.completionId)||a.responsePosition-b.responsePosition);const exported={schemaVersion:payload.exportSchemaVersion,provenance:{sourceName:payload.meta.sourceName,sourceSha256:payload.meta.sourceSha256,reviewPayloadSha256:payloadSha,tokenizerRegistrySha256:payload.meta.tokenizerRegistrySha256,targetIdContract:payload.meta.targetIdContract,modelPolicy:payload.meta.modelPolicy,claimBoundary:payload.meta.claimBoundary},selectedCompletionIds:selected,targetSelections:targets};const blob=new Blob([JSON.stringify(exported,null,2)+"\n"],{type:"application/json"});const link=document.createElement("a");link.href=URL.createObjectURL(blob);link.download="outright-trace-target-selection.json";link.click();setTimeout(()=>URL.revokeObjectURL(link.href),0)});
document.addEventListener("keydown",event=>{if(event.ctrlKey||event.metaKey||event.altKey||/^(INPUT|SELECT|TEXTAREA|BUTTON)$/.test(event.target.tagName))return;if(event.key.toLowerCase()==="j"){event.preventDefault();move(1)}else if(event.key.toLowerCase()==="k"){event.preventDefault();move(-1)}else if(event.key.toLowerCase()==="s"&&state.active){event.preventDefault();toggleCandidate(state.active)}});
$("source-note").textContent=payload.counts.completions+" completions · "+payload.counts.models+" models · "+payload.counts.response_token_range.total.toLocaleString()+" response tokens · source "+payload.meta.sourceSha256.slice(0,12);$("rail-note").textContent=payload.meta.claimBoundary;renderAll();}
main().catch(error=>{const reader=document.getElementById("reader-inner");reader.replaceChildren();const message=document.createElement("p");message.className="empty";message.textContent="Could not load review data: "+error.message;reader.append(message);console.error(error)});
</script></body></html>
'''


def render_target_review_html(payload: Mapping[str, Any]) -> str:
    """Render a self-contained page with inert base64-encoded corpus data."""

    payload_bytes = canonical_json(payload)
    encoded = base64.b64encode(gzip.compress(payload_bytes, mtime=0)).decode("ascii")
    payload_sha256 = hashlib.sha256(payload_bytes).hexdigest()
    return (
        _HTML_HEAD
        + _HTML_SCRIPT.replace("__PAYLOAD__", encoded).replace(
            "__PAYLOAD_SHA__", payload_sha256
        )
    )


def build_target_review_packet(
    *,
    source_path: Path,
    destination: Path,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    expected_source_sha256: str | None = EXPECTED_SOURCE_SHA256,
    tokenizer_loader: Callable[[Mapping[str, Any]], Any] = _default_tokenizer_loader,
) -> dict[str, Any]:
    """Build v2 review.html and its binding manifest without partial output."""

    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    rows, source_sha256 = read_source_rows(
        source_path, expected_source_sha256=expected_source_sha256
    )
    registry, registry_sha256 = load_tokenizer_registry(registry_path)
    payload = assemble_target_review_payload(
        rows,
        source_sha256=source_sha256,
        source_name=source_path.name,
        registry=registry,
        registry_sha256=registry_sha256,
        tokenizer_loader=tokenizer_loader,
    )
    payload_bytes = canonical_json(payload)
    html = render_target_review_html(payload).encode("utf-8")
    temporary = destination.parent / f".{destination.name}.building-{uuid.uuid4().hex}"
    temporary.mkdir(parents=True)
    try:
        _atomic_write_bytes(temporary / "review.html", html)
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "export_schema_version": EXPORT_SCHEMA_VERSION,
            "status": "exploratory_all_model_trace_target_review",
            "source": {
                "name": source_path.name,
                "sha256": source_sha256,
                "authoritative_sha256": expected_source_sha256,
            },
            "scope": {
                "outright_source_types": sorted(OUTRIGHT_SOURCE_TYPES),
                "deduplication_key": ["target_model", "prompt", "cot"],
                "model_policy": "all models, including Qwen",
            },
            "tokenization": {
                "registry_name": registry_path.name,
                "registry_sha256": registry_sha256,
                "profiles": payload["tokenizerProfiles"],
                "identity_status": "reconstructed_at_pinned_revision",
            },
            "counts": payload["counts"],
            "payload_sha256": hashlib.sha256(payload_bytes).hexdigest(),
            "embedded_payload": {
                "encoding": "base64(gzip(canonical-json))",
                "uncompressed_bytes": len(payload_bytes),
                "compressed_bytes": len(gzip.compress(payload_bytes, mtime=0)),
            },
            "page_sha256": hashlib.sha256(html).hexdigest(),
            "files": {
                "review.html": {
                    "sha256": hashlib.sha256(html).hexdigest(),
                    "bytes": len(html),
                }
            },
            "claim_boundaries": [
                "This is exploratory human candidate and token-target selection, not evaluation.",
                "Exact token identities are reconstructed at pinned revisions, not recovered original generation IDs.",
                "Source labels are displayed verbatim and are not independently validated here.",
                "No tracing, scoring, clustering, neuron labeling, or faithfulness validation occurs.",
            ],
        }
        manifest["manifest_sha256"] = canonical_sha256(manifest)
        _atomic_write_json(temporary / "manifest.json", manifest)
        os.replace(temporary, destination)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
