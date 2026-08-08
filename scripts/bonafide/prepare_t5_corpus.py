"""Prepare tokenizer-exact sparse targets from a generated Qwen corpus.

The generation campaign retained Qwen's terminal ``<|im_end|>`` token in the
completion token IDs even though its frozen screening policy described that
token as excluded.  A decoded response can also have a different, equivalent
BPE segmentation when encoded again by the tracing tokenizer.  This intake
validates the original completion provenance, derives the exact tracing token
sequence from the frozen response text and tokenizer, reapplies the frozen
mechanical predicates, and emits independent single-position target items.  It
never generates, labels, scores, or traces a response.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

from scripts.bonafide.manifest import (
    SCHEMA_VERSION as TRACE_MANIFEST_SCHEMA,
)
from scripts.bonafide.manifest import (
    select_stratified_random_positions,
)
from scripts.bonafide.runner import _sha256, validate_target_selection

PROFILE_SCHEMA = "bonafide-t5-corpus-preparation-profile/v1"
RECEIPT_SCHEMA = "bonafide-t5-corpus-preparation-receipt/v1"
SELECTED_RESPONSE_SCHEMA = "bonafide-t5-selected-response/v1"
TRACE_COMPATIBLE_SCREEN_SCHEMA = "qwen3-circuits-mechanical-screen.trace-compatible-v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _checked_path(record: Mapping[str, Any], *, label: str) -> Path:
    raw_path = record.get("path")
    expected_hash = record.get("sha256")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"profile {label}.path is required")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError(f"profile {label}.sha256 is invalid")
    path = Path(os.path.expandvars(raw_path)).expanduser().resolve()
    if not path.is_file() or file_sha256(path) != expected_hash:
        raise ValueError(f"profile {label} file/hash drift: {path}")
    return path


def _load_frozen_screen(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("frozen_qwen_corpus_screen", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import frozen screening implementation: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in (
        "PREDICATE_FIELDS",
        "RULE_CONFIG",
        "RULE_CONFIG_SHA256",
        "_ALLOWED_EOS_KINDS",
        "degeneration_metrics",
        "screen_row",
        "validate_campaign_shape",
    ):
        if not hasattr(module, name):
            raise ValueError(f"frozen screening implementation lacks {name}")
    return module


def _load_trace_tokenizer(
    config_path: Path, tokenizer_provenance: Mapping[str, Any]
) -> tuple[Any, Path]:
    """Load and authenticate the exact tokenizer used by the trace runner."""

    from transformers import AutoTokenizer

    from scripts.bonafide.corpus_selection import _tokenizer_file_manifest
    from scripts.bonafide.runner import validate_run_config

    config = _load_json(config_path)
    validate_run_config(config)
    model = config["model"]
    if model.get("model_id") != tokenizer_provenance.get("model_id") or model.get(
        "revision"
    ) != tokenizer_provenance.get("revision"):
        raise ValueError("trace config and tokenizer provenance disagree")
    raw_snapshot = model.get("local_snapshot_path")
    if not isinstance(raw_snapshot, str) or not raw_snapshot:
        raise ValueError("trace config lacks model.local_snapshot_path")
    snapshot = Path(os.path.expandvars(raw_snapshot)).expanduser().resolve()
    if not snapshot.is_dir():
        raise FileNotFoundError(f"trace tokenizer snapshot is unavailable: {snapshot}")
    if _tokenizer_file_manifest(snapshot) != tokenizer_provenance.get("file_manifest"):
        raise ValueError("trace tokenizer snapshot file-manifest drift")
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    return tokenizer, snapshot


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or len(set(reader.fieldnames)) != len(
            reader.fieldnames
        ):
            raise ValueError(f"invalid CSV header: {path}")
        return list(reader.fieldnames), list(reader)


def _validate_generation_run(
    rows: Sequence[Mapping[str, str]],
    *,
    generation_path: Path,
    request_grid_path: Path,
    run_metadata_path: Path,
    screen_module: ModuleType,
) -> dict[str, Any]:
    run = _load_json(run_metadata_path)
    output = run.get("output")
    if (
        run.get("status") != "complete"
        or run.get("exit_code") != 0
        or not isinstance(output, Mapping)
        or output.get("path") != str(generation_path)
        or output.get("sha256") != file_sha256(generation_path)
        or output.get("rows") != len(rows)
    ):
        raise ValueError("generation CSV disagrees with completed run metadata")

    attempt_manifest = output.get("attempt_store_manifest")
    attempt_store_raw = output.get("attempt_store")
    if not isinstance(attempt_manifest, Mapping) or not isinstance(
        attempt_store_raw, str
    ):
        raise TypeError("generation run metadata lacks its atomic attempt store")
    attempt_store = Path(attempt_store_raw).resolve()
    declared_files = attempt_manifest.get("files")
    if not attempt_store.is_dir() or not isinstance(declared_files, list):
        raise ValueError("generation atomic attempt store is unavailable")
    declared_by_name = {
        record.get("path"): record
        for record in declared_files
        if isinstance(record, Mapping)
    }
    actual_attempts = sorted(attempt_store.glob("attemptv1_*.json"))
    if (
        len(declared_by_name) != len(declared_files)
        or {path.name for path in actual_attempts} != set(declared_by_name)
        or len(actual_attempts) != len(rows)
    ):
        raise ValueError("generation atomic attempt inventory drift")
    atomic_rows: list[tuple[int, Mapping[str, Any]]] = []
    for path in actual_attempts:
        content = path.read_bytes()
        declared = declared_by_name[path.name]
        if len(content) != declared.get("bytes") or hashlib.sha256(
            content
        ).hexdigest() != declared.get("sha256"):
            raise ValueError(f"generation atomic attempt hash drift: {path}")
        attempt = json.loads(content)
        result_row = attempt.get("result_row")
        if attempt.get("attempt_index") != 0 or not isinstance(result_row, Mapping):
            raise ValueError(f"invalid canonical atomic attempt: {path}")
        repro = json.loads(result_row["reproducibility_info"])
        atomic_rows.append((int(repro["source_row_index"]), result_row))
    atomic_rows.sort(key=lambda pair: pair[0])
    if [index for index, _ in atomic_rows] != list(range(len(rows))):
        raise ValueError("atomic attempt source row indices are not contiguous")
    for row_index, ((_, atomic), consolidated) in enumerate(zip(atomic_rows, rows)):
        if set(atomic) != set(consolidated):
            raise ValueError(f"atomic attempt schema drift at source row {row_index}")
        for field, value in atomic.items():
            normalized = "" if value is None else str(value)
            if normalized != consolidated[field]:
                raise ValueError(
                    f"atomic/consolidated row drift at source row {row_index}: {field}"
                )

    _, request_rows = _read_csv(request_grid_path)
    if len(request_rows) != len(rows):
        raise ValueError("generation and request-grid row counts disagree")
    request_by_id = {row.get("id"): row for row in request_rows}
    if None in request_by_id or len(request_by_id) != len(request_rows):
        raise ValueError("request grid contains empty or duplicate request IDs")
    generated_columns = {
        "cot",
        "model_answer",
        "model_raw_response",
        "completion_id",
        "reproducibility_info",
    }
    for row in rows:
        request_id = row.get("id")
        source = request_by_id.get(request_id)
        if source is None:
            raise ValueError(
                f"generation row is absent from request grid: {request_id}"
            )
        for field, expected in source.items():
            if field not in generated_columns and row.get(field) != expected:
                raise ValueError(
                    f"generation input field drift for {request_id}: {field}"
                )
    screen_module.validate_campaign_shape(rows)
    return {
        "status": run["status"],
        "exit_code": run["exit_code"],
        "rows": len(rows),
        "generation_csv_sha256": output["sha256"],
        "attempt_store_tree_sha256": output.get("attempt_store_manifest", {}).get(
            "tree_sha256"
        ),
        "atomic_attempt_files": len(actual_attempts),
        "atomic_attempt_hashes_match": True,
        "canonical_atomic_rows_match_csv": True,
        "source_snapshot": run.get("source_snapshot"),
    }


def trace_compatible_screen_row(
    row: Mapping[str, str], screen_module: ModuleType, *, tokenized: Any | None = None
) -> dict[str, Any]:
    """Apply the frozen screen to the exact trace-tokenizer response IDs."""

    frozen = dict(screen_module.screen_row(row))
    repro = json.loads(row["reproducibility_info"])
    prompt_ids = [int(value) for value in repro["prompt_token_ids"]]
    generated_ids = [int(value) for value in repro["completion_token_ids"]]
    recorded_prefix_count = int(row["assistant_prefix_token_count"])
    if recorded_prefix_count != len(prompt_ids):
        raise ValueError(
            f"assistant prefix count mismatch for request {row.get('id')!r}"
        )

    stop_kind = repro["stop_reason_kind"]
    natural_eos = stop_kind in screen_module._ALLOWED_EOS_KINDS
    effective_stop_ids = [
        int(value) for value in repro["effective_default_stop_token_ids"]
    ]
    suffix_id: int | None = None
    generation_content_ids = list(generated_ids)
    if natural_eos:
        if (
            not generation_content_ids
            or generation_content_ids[-1] not in effective_stop_ids
        ):
            raise ValueError(
                "natural-EOS completion does not end in its declared default "
                f"assistant suffix token: {row.get('id')!r}"
            )
        suffix_id = generation_content_ids.pop()

    if tokenized is None:
        trace_prefix_ids = prompt_ids
        trace_suffix_ids = [] if suffix_id is None else [suffix_id]
        trace_ids = generation_content_ids
    else:
        trace_prefix_ids = [int(value) for value in tokenized.assistant_prefix_ids]
        trace_ids = [int(value) for value in tokenized.response_ids]
        trace_suffix_ids = [int(value) for value in tokenized.assistant_suffix_ids]
        if trace_prefix_ids != prompt_ids:
            raise ValueError(
                f"trace/generation assistant prefix drift for request {row.get('id')!r}"
            )
        if natural_eos and (not trace_suffix_ids or trace_suffix_ids[0] != suffix_id):
            raise ValueError(
                f"trace/generation assistant suffix drift for request {row.get('id')!r}"
            )

    degeneration = screen_module.degeneration_metrics(trace_ids)
    trace_count = len(trace_ids)
    predicates = {
        "natural_eos": natural_eos,
        "response_length_224_768": 224 <= trace_count <= 768,
        "total_length_at_most_1024": len(prompt_ids) + trace_count <= 1_024,
        "nonempty_raw_response": bool(row["model_raw_response"].strip()),
        "nonempty_reasoning_and_final_answer": bool(
            frozen["screening_reasoning"] and frozen["screening_final_answer"]
        ),
        "no_obvious_degeneration": degeneration["no_obvious_degeneration"],
    }
    failures = [name for name in screen_module.PREDICATE_FIELDS if not predicates[name]]
    return {
        **frozen,
        "screening_schema": TRACE_COMPATIBLE_SCREEN_SCHEMA,
        "base_screening_schema": screen_module.RULE_CONFIG["schema"],
        "base_screening_rule_config_sha256": screen_module.RULE_CONFIG_SHA256,
        "assistant_prefix_token_count": len(prompt_ids),
        "generation_completion_token_count": len(generated_ids),
        "generation_content_token_count": len(generation_content_ids),
        "terminal_assistant_suffix_token_count": int(suffix_id is not None),
        "terminal_assistant_suffix_token_id": suffix_id,
        "trace_assistant_suffix_token_ids": trace_suffix_ids,
        "response_token_count": trace_count,
        "assistant_prefix_plus_response_token_count": len(prompt_ids) + trace_count,
        "trace_tokenization_matches_generation": trace_ids == generation_content_ids,
        "trace_response_token_ids_sha256": _sha256(trace_ids),
        **predicates,
        **degeneration,
        "screening_eligible": not failures,
        "screening_failure_reasons": failures,
        "selected_for_primary": False,
        "selected_for_tracing": False,
        "selection_status": "pending",
        "trace_response_token_ids": trace_ids,
        "trace_response_token_logprobs": (
            list(repro.get("sampled_token_logprobs", [])[:trace_count])
            if trace_ids == generation_content_ids
            else None
        ),
    }


def select_response_draws(
    records: Sequence[dict[str, Any]],
    *,
    role_map: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Select draw zero when eligible, otherwise draw one, for each role/cell."""

    by_cell: dict[tuple[str, str], dict[int, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        role = str(record["campaign_role"])
        if role not in role_map:
            continue
        draw = int(record["draw_index"])
        key = (role, str(record["source_prompt_id"]))
        if draw in by_cell[key]:
            raise ValueError(f"duplicate draw for response cell {key!r}: {draw}")
        by_cell[key][draw] = record

    selected: list[dict[str, Any]] = []
    for key, draws in sorted(by_cell.items()):
        if set(draws) != {0, 1}:
            raise ValueError(f"response cell does not contain draws 0 and 1: {key!r}")
        draw0, draw1 = draws[0], draws[1]
        chosen = draw0 if draw0["screening_eligible"] else draw1
        if chosen["screening_eligible"]:
            chosen["selected_for_tracing"] = True
            chosen["selection_status"] = (
                "selected_draw0_default"
                if chosen is draw0
                else "selected_draw1_fallback"
            )
            chosen["trace_corpus_role"] = role_map[key[0]]
            selected.append(chosen)
        for record in (draw0, draw1):
            if record is not chosen or not chosen["screening_eligible"]:
                record["selection_status"] = (
                    "eligible_replicate_unselected"
                    if record["screening_eligible"]
                    else "ineligible_unselected"
                )
    return selected


def _example(record: Mapping[str, Any], profile: Mapping[str, Any]) -> dict[str, Any]:
    completion_id = str(record["completion_id"])
    return {
        "example_id": completion_id,
        "base_question_id": str(record["question_family_id"]),
        "source_prompt_id": str(record["source_prompt_id"]),
        "request_id": str(record["id"]),
        "completion_id": completion_id,
        "campaign_id": str(record["campaign_id"]),
        "campaign_role": str(record["campaign_role"]),
        "draw_index": int(record["draw_index"]),
        "generation_seed": int(record["generation_seed"]),
        "question": str(record["question"]),
        "prompt": str(record["prompt"]),
        "response": str(record["model_raw_response"]),
        "source_question_ids": json.loads(record["source_question_ids_json"]),
        "src_type": str(record["src_type"]),
        "hint_dataset": str(record["hint_dataset"]),
        "hint_type": str(record["hint_type"]),
        "prompted_hint": str(record["prompted_hint"]),
        "correct_answer": str(record["correct_answer"]),
        "hinted_answer": str(record["hinted_answer"]),
        "target_model": str(record["target_model"]),
        "model_revision": str(record["model_revision"]),
        "token_counts": {
            "assistant_prefix": int(record["assistant_prefix_token_count"]),
            "response": int(record["response_token_count"]),
            "assistant_prefix_plus_response": int(
                record["assistant_prefix_plus_response_token_count"]
            ),
            "generation_completion_including_suffix": int(
                record["generation_completion_token_count"]
            ),
            "terminal_assistant_suffix": int(
                record["terminal_assistant_suffix_token_count"]
            ),
        },
        "screening": {
            "schema": TRACE_COMPATIBLE_SCREEN_SCHEMA,
            "base_rule_config_sha256": record["base_screening_rule_config_sha256"],
            "selection_status": record["selection_status"],
        },
        "corpus_profile_id": profile["profile_id"],
    }


def build_source_manifest(
    selected: Sequence[Mapping[str, Any]],
    *,
    profile: Mapping[str, Any],
    tokenizer: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    target = profile["target_selection"]
    target_count = int(target["targets_per_response"])
    seed = str(target["seed"])
    waves: list[dict[str, Any]] = []
    selected_records: list[dict[str, Any]] = []
    seen_artifacts: set[str] = set()

    for record in sorted(
        selected,
        key=lambda value: (
            str(value["trace_corpus_role"]),
            str(value["question_family_id"]),
            str(value["source_prompt_id"]),
        ),
    ):
        response_ids = [int(value) for value in record["trace_response_token_ids"]]
        example = _example(record, profile)
        selections = select_stratified_random_positions(
            len(response_ids),
            target_count,
            seed=seed,
            example_id=example["example_id"],
        )
        items: list[dict[str, Any]] = []
        for sampling in selections:
            position = int(sampling["response_token_position"])
            identity = {
                "profile_id": profile["profile_id"],
                "completion_id": example["completion_id"],
                "response_token_position": position,
                "target_token_id": response_ids[position],
                "target_selection_policy": target["policy_id"],
            }
            artifact_id = f"trace-source-{_sha256(identity)[:24]}"
            if artifact_id in seen_artifacts:
                raise ValueError(f"duplicate target artifact ID: {artifact_id}")
            seen_artifacts.add(artifact_id)
            selection = {
                "kind": "explicit_response_positions",
                "width": 1,
                "response_token_positions": [position],
                "final_target_token_id": response_ids[position],
                "final_selection": {
                    "corpus_role": record["trace_corpus_role"],
                    "selection_reasons": [
                        {
                            "bucket": "stratified_random_response_coverage",
                            **sampling,
                        }
                    ],
                },
            }
            item = {
                "artifact_id": artifact_id,
                "example": example,
                "response_token_count": len(response_ids),
                "objective": {
                    "name": "single_selected_logit",
                    "benchmark_only_multi_target": False,
                },
                "target_selection": selection,
            }
            validate_target_selection(item)
            items.append(item)
        wave_id = (
            f"t5-corpus-{record['trace_corpus_role']}-{example['completion_id'][-16:]}"
        )
        waves.append(
            {
                "wave_id": wave_id,
                "corpus_role": record["trace_corpus_role"],
                "response_identity": {
                    "completion_id": example["completion_id"],
                    "request_id": example["request_id"],
                    "source_prompt_id": example["source_prompt_id"],
                    "question_family_id": example["base_question_id"],
                },
                "items": items,
            }
        )
        selected_records.append(
            {
                "schema_version": SELECTED_RESPONSE_SCHEMA,
                "corpus_role": record["trace_corpus_role"],
                "example": example,
                "trace_response_token_ids": response_ids,
                "trace_response_token_logprobs": record[
                    "trace_response_token_logprobs"
                ],
                "target_artifact_ids": [item["artifact_id"] for item in items],
                "target_response_positions": [
                    item["target_selection"]["response_token_positions"][0]
                    for item in items
                ],
            }
        )

    manifest = {
        "schema_version": TRACE_MANIFEST_SCHEMA,
        "artifact_kind": "bonafide_t5_corpus_source_targets",
        "source_artifacts": {
            name: dict(record)
            for name, record in sorted(profile.get("inputs", {}).items())
        },
        "corpus_profile": {
            "profile_id": profile["profile_id"],
            "campaign_id": profile["campaign_id"],
            "candidate_policy_id": "model_top5_plus_observed",
            "candidate_union_contract": "adag.bonafide.candidate-union.v1",
            "target_selection": dict(target),
            "response_count": len(selected_records),
            "target_count": len(seen_artifacts),
        },
        "tokenizer": dict(tokenizer),
        "waves": waves,
    }
    return manifest, selected_records


def _jsonable_csv(value: Any) -> Any:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False))
            handle.write("\n")


def _write_screening_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    source_fields: Sequence[str],
) -> None:
    extra_fields = [
        "screening_schema",
        "base_screening_schema",
        "base_screening_rule_config_sha256",
        "generation_completion_token_count",
        "generation_content_token_count",
        "terminal_assistant_suffix_token_count",
        "terminal_assistant_suffix_token_id",
        "trace_assistant_suffix_token_ids",
        "response_token_count",
        "assistant_prefix_plus_response_token_count",
        "trace_tokenization_matches_generation",
        "trace_response_token_ids_sha256",
        "stop_reason_kind",
        "natural_eos",
        "response_length_224_768",
        "total_length_at_most_1024",
        "nonempty_raw_response",
        "nonempty_reasoning_and_final_answer",
        "immediate_repeat_detected",
        "immediate_repeat_block_length",
        "immediate_repeat_start_token_index",
        "unique_4gram_count",
        "total_4gram_count",
        "unique_4gram_ratio",
        "unique_4gram_threshold_applied",
        "no_obvious_degeneration",
        "screening_eligible",
        "screening_failure_reasons",
        "selected_for_tracing",
        "selection_status",
        "trace_corpus_role",
    ]
    fields = [
        *source_fields,
        *[field for field in extra_fields if field not in source_fields],
    ]
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {field: _jsonable_csv(row.get(field, "")) for field in fields}
            )


def prepare(profile_path: Path, output_dir: Path) -> dict[str, Any]:
    profile_path = profile_path.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"T5 corpus output already exists: {output_dir}")
    profile = _load_json(profile_path)
    if profile.get("schema_version") != PROFILE_SCHEMA:
        raise ValueError("unsupported T5 corpus preparation profile")

    inputs = profile.get("inputs")
    if not isinstance(inputs, Mapping):
        raise TypeError("T5 corpus profile requires inputs")
    generation_path = _checked_path(inputs["generation_csv"], label="generation_csv")
    request_grid_path = _checked_path(inputs["request_grid"], label="request_grid")
    run_metadata_path = _checked_path(inputs["run_metadata"], label="run_metadata")
    screen_path = _checked_path(
        inputs["frozen_screening_implementation"],
        label="frozen_screening_implementation",
    )
    source_manifest_path = _checked_path(
        inputs["tokenizer_source_manifest"], label="tokenizer_source_manifest"
    )
    trace_config_path = _checked_path(
        inputs["trace_run_config"], label="trace_run_config"
    )
    tokenizer_source = _load_json(source_manifest_path)
    tokenizer_provenance = tokenizer_source.get("tokenizer")
    if not isinstance(tokenizer_provenance, Mapping):
        raise TypeError("tokenizer source manifest lacks tokenizer provenance")
    trace_tokenizer, tokenizer_snapshot = _load_trace_tokenizer(
        trace_config_path, tokenizer_provenance
    )

    from circuits.tracing.trace import tokenize_teacher_forced_response

    source_fields, generation_rows = _read_csv(generation_path)
    screen_module = _load_frozen_screen(screen_path)
    generation_validation = _validate_generation_run(
        generation_rows,
        generation_path=generation_path,
        request_grid_path=request_grid_path,
        run_metadata_path=run_metadata_path,
        screen_module=screen_module,
    )
    records = []
    for row in generation_rows:
        tokenized = tokenize_teacher_forced_response(
            trace_tokenizer,
            row["prompt"],
            row["model_raw_response"],
        )
        records.append(
            {
                **row,
                **trace_compatible_screen_row(row, screen_module, tokenized=tokenized),
            }
        )
    role_map = profile.get("role_map")
    if not isinstance(role_map, Mapping) or not role_map:
        raise ValueError("T5 corpus profile requires a role_map")
    selected = select_response_draws(records, role_map=role_map)
    source_manifest, selected_records = build_source_manifest(
        selected,
        profile=profile,
        tokenizer=tokenizer_provenance,
    )

    response_counts = Counter(row["corpus_role"] for row in selected_records)
    target_counts = Counter()
    for wave in source_manifest["waves"]:
        target_counts[wave["corpus_role"]] += len(wave["items"])
    expected = profile.get("expected_counts", {})
    actual_counts = {
        "selected_responses": len(selected_records),
        "targets": sum(target_counts.values()),
        "response_roles": dict(sorted(response_counts.items())),
        "target_roles": dict(sorted(target_counts.items())),
    }
    if actual_counts != expected:
        raise ValueError(
            f"T5 corpus expected-count drift: expected={expected}, actual={actual_counts}"
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    try:
        screening_path = staging / "screening-records.csv"
        selected_path = staging / "selected-responses.jsonl"
        trace_manifest_path = staging / "t5-source-targets.json"
        receipt_path = staging / "preparation-receipt.json"
        _write_screening_csv(screening_path, records, source_fields=source_fields)
        _write_jsonl(selected_path, selected_records)
        _write_json(trace_manifest_path, source_manifest)
        receipt = {
            "schema_version": RECEIPT_SCHEMA,
            "created_at": datetime.now(UTC).isoformat(),
            "preparation_implementation": {
                "path": str(Path(__file__).resolve()),
                "sha256": file_sha256(Path(__file__).resolve()),
            },
            "profile": {
                "path": str(profile_path),
                "sha256": file_sha256(profile_path),
                "profile_id": profile["profile_id"],
            },
            "generation_validation": generation_validation,
            "compatibility_correction": {
                "reason": (
                    "generation retained the terminal assistant suffix token while "
                    "the frozen screen declared it excluded; exact trace-tokenizer "
                    "re-encoding also replaces equivalent non-canonical BPE segmentations"
                ),
                "natural_eos_rows_with_suffix_removed": sum(
                    int(row["terminal_assistant_suffix_token_count"]) for row in records
                ),
                "generation_rows": len(records),
                "screening_schema": TRACE_COMPATIBLE_SCREEN_SCHEMA,
                "base_screening_rule_config_sha256": screen_module.RULE_CONFIG_SHA256,
                "frozen_screening_implementation_sha256": file_sha256(screen_path),
                "trace_run_config_path": str(trace_config_path),
                "trace_run_config_sha256": file_sha256(trace_config_path),
                "trace_tokenizer_snapshot": str(tokenizer_snapshot),
                "trace_tokenization_mismatch_rows": sum(
                    not bool(row["trace_tokenization_matches_generation"])
                    for row in records
                ),
                "selected_trace_tokenization_mismatch_rows": sum(
                    not bool(row["trace_tokenization_matches_generation"])
                    for row in selected
                ),
            },
            "counts": actual_counts,
            "eligible_completion_counts": dict(
                sorted(
                    Counter(
                        str(row["campaign_role"])
                        for row in records
                        if row["screening_eligible"]
                    ).items()
                )
            ),
            "selected_draw_counts": dict(
                sorted(Counter(str(row["draw_index"]) for row in selected).items())
            ),
            "outputs": {
                "screening_records": {
                    "path": screening_path.name,
                    "sha256": file_sha256(screening_path),
                    "rows": len(records),
                },
                "selected_responses": {
                    "path": selected_path.name,
                    "sha256": file_sha256(selected_path),
                    "rows": len(selected_records),
                },
                "source_targets": {
                    "path": trace_manifest_path.name,
                    "sha256": file_sha256(trace_manifest_path),
                    "canonical_sha256": _sha256(source_manifest),
                    "responses": len(source_manifest["waves"]),
                    "targets": sum(
                        len(wave["items"]) for wave in source_manifest["waves"]
                    ),
                },
            },
        }
        _write_json(receipt_path, receipt)
        os.replace(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    receipt = prepare(args.profile, args.output_dir)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
