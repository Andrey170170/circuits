"""Build one resident-model probe wave for BonaFide prompt screening.

The output uses the existing BonaFide trace-manifest schema so it can be fed
directly to ``probe_runner``.  Its targets are cheap screening measurements;
they do not freeze either final tracing targets or final corpus membership.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

from circuits.tracing.trace import get_chat_template, tokenize_teacher_forced_response
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerBase

from scripts.bonafide.corpus_selection import (
    SCHEMA_VERSION as CANDIDATE_SCHEMA_VERSION,
)
from scripts.bonafide.corpus_selection import (
    _tokenizer_file_manifest,
)
from scripts.bonafide.manifest import (
    SCHEMA_VERSION,
    resolve_pretrained_source,
    select_stratified_random_positions,
)

DEFAULT_SELECTION_PATH = Path(
    "scripts/bonafide/selections/qwen3_4b_instruct_candidates.json"
)
DEFAULT_SELECTION_SHA256 = (
    "46f7de043810bd7ab8cf86b1b4657fec2bfd50aee36f3d0aa5702f4b39166e0c"
)
DEFAULT_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_TARGETS_PER_EXAMPLE = 16
DEFAULT_SCREENING_SEED = "bonafide-prompt-screening-v1"
EXPECTED_DENSE_EXAMPLES = 25
EXPECTED_BROAD_EXAMPLES = 108
WAVE_ID = "prompt-screening-estimation"


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_selection(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("prompt candidate selection must be a JSON object")
    return value


def _require_exact_selection_source(
    selection_path: Path, expected_selection_sha256: str
) -> str:
    if len(expected_selection_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_selection_sha256
    ):
        raise ValueError(
            "expected selection SHA-256 must be 64 lowercase hex characters"
        )
    actual = _file_sha256(selection_path)
    if actual != expected_selection_sha256:
        raise ValueError(
            "prompt candidate selection SHA-256 changed: "
            f"expected {expected_selection_sha256}, found {actual}"
        )
    return actual


def _validate_tokenizer_provenance(
    *,
    selection: Mapping[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    tokenizer_path: Path,
    model_id: str,
    model_revision: str,
) -> None:
    provenance = selection.get("tokenizer")
    if not isinstance(provenance, Mapping):
        raise ValueError("prompt candidate selection requires tokenizer provenance")
    if provenance.get("model_id") != model_id:
        raise ValueError("selection tokenizer model_id does not match requested model")
    if provenance.get("revision") != model_revision:
        raise ValueError(
            "selection tokenizer revision does not match requested revision"
        )
    if provenance.get("class") != type(tokenizer).__name__:
        raise ValueError("runtime tokenizer class does not match candidate selection")
    chat_template_hash = _sha256_bytes(
        get_chat_template(cast(PreTrainedTokenizer, tokenizer)).encode("utf-8")
    )
    if provenance.get("chat_template_sha256") != chat_template_hash:
        raise ValueError(
            "runtime tokenizer chat template does not match candidate selection"
        )
    if provenance.get("file_manifest") != _tokenizer_file_manifest(tokenizer_path):
        raise ValueError("runtime tokenizer files do not match candidate selection")


def _validate_candidate_contract(selection: Mapping[str, Any]) -> None:
    if selection.get("schema_version") != CANDIDATE_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported prompt candidate schema: {selection.get('schema_version')!r}"
        )
    if selection.get("artifact_kind") != "bonafide_prompt_candidates":
        raise ValueError("input is not a BonaFide prompt candidate artifact")
    contract = selection.get("candidate_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("candidate selection requires candidate_contract")
    expected = {
        "prompt_candidates_selected": True,
        "target_spans_selected": False,
        "target_spans_frozen": False,
        "trace_work_items_created": False,
    }
    if any(contract.get(field) is not value for field, value in expected.items()):
        raise ValueError(
            "candidate contract no longer describes an unfrozen prompt inventory"
        )


def _selected_examples(
    selection: Mapping[str, Any],
) -> list[tuple[str, Mapping[str, Any]]]:
    selections = selection.get("selections")
    examples = selection.get("examples")
    if not isinstance(selections, Mapping) or not isinstance(examples, list):
        raise ValueError("candidate selection requires selections and examples")

    dense_ids = selections.get("dense_inventory")
    broad_ids = selections.get("broad_eligible_inventory")
    if not isinstance(dense_ids, list) or not isinstance(broad_ids, list):
        raise ValueError("candidate selection inventories must be lists")
    if len(dense_ids) != EXPECTED_DENSE_EXAMPLES:
        raise ValueError(
            f"dense inventory changed: expected {EXPECTED_DENSE_EXAMPLES}, "
            f"found {len(dense_ids)}"
        )
    if len(broad_ids) != EXPECTED_BROAD_EXAMPLES:
        raise ValueError(
            f"broad inventory changed: expected {EXPECTED_BROAD_EXAMPLES}, "
            f"found {len(broad_ids)}"
        )
    if len(set(dense_ids)) != len(dense_ids) or len(set(broad_ids)) != len(broad_ids):
        raise ValueError("candidate inventories contain duplicate example IDs")
    if set(dense_ids).intersection(broad_ids):
        raise ValueError("dense and broad candidate inventories must be disjoint")

    by_id: dict[str, Mapping[str, Any]] = {}
    for example in examples:
        if not isinstance(example, Mapping) or not isinstance(
            example.get("example_id"), str
        ):
            raise ValueError("candidate examples require string example_id values")
        example_id = str(example["example_id"])
        if example_id in by_id:
            raise ValueError(f"duplicate candidate example: {example_id}")
        by_id[example_id] = example

    policy = selection.get("selection_policy")
    if not isinstance(policy, Mapping):
        raise ValueError("candidate selection requires selection_policy")
    dense_policy = policy.get("dense")
    broad_policy = policy.get("broad")
    if not isinstance(dense_policy, Mapping) or not isinstance(broad_policy, Mapping):
        raise ValueError("candidate selection requires dense and broad policies")
    dense_response_cap = int(dense_policy["response_token_cap"])
    dense_total_cap = int(dense_policy["total_context_token_cap"])
    broad_response_cap = int(broad_policy["response_token_cap"])
    broad_total_cap = int(broad_policy["total_context_token_cap"])

    selected: list[tuple[str, Mapping[str, Any]]] = []
    for inventory, ids, response_cap, total_cap in (
        ("dense_inventory", dense_ids, dense_response_cap, dense_total_cap),
        ("broad_eligible_inventory", broad_ids, broad_response_cap, broad_total_cap),
    ):
        for example_id in ids:
            example = by_id.get(example_id)
            if example is None:
                raise ValueError(f"selected example is absent: {example_id}")
            counts = example.get("token_counts")
            membership = example.get("selection_membership")
            if not isinstance(counts, Mapping) or not isinstance(membership, Mapping):
                raise ValueError(
                    f"selected example lacks selection metadata: {example_id}"
                )
            if membership.get(inventory) is not True:
                raise ValueError(
                    f"selected example membership disagrees with {inventory}: {example_id}"
                )
            response_count = int(counts["response"])
            total_count = int(counts["maximum_teacher_forced_input"])
            if response_count > response_cap or total_count > total_cap:
                raise ValueError(
                    f"selected example exceeds {inventory} caps: {example_id}"
                )
            if (
                inventory == "broad_eligible_inventory"
                and membership.get("dense_inventory") is not False
            ):
                raise ValueError(
                    f"broad example is not disjoint from dense: {example_id}"
                )
            selected.append((inventory, example))
    return selected


def _runtime_response_ids(
    tokenizer: PreTrainedTokenizerBase, example: Mapping[str, Any]
) -> list[int]:
    prompt = example.get("prompt")
    response = example.get("response")
    if not isinstance(prompt, str) or not isinstance(response, str):
        raise ValueError(
            "selected examples require complete prompt and response strings"
        )
    tokenized = tokenize_teacher_forced_response(
        cast(PreTrainedTokenizer, tokenizer), prompt, response
    )
    expected = example.get("token_counts")
    if not isinstance(expected, Mapping):
        raise ValueError("selected example requires token_counts")
    actual_counts = {
        "assistant_prefix": len(tokenized.assistant_prefix_ids),
        "response": len(tokenized.response_ids),
        "assistant_suffix": len(tokenized.assistant_suffix_ids),
        "maximum_teacher_forced_input": len(tokenized.assistant_prefix_ids)
        + len(tokenized.response_ids),
        "full_conversation_with_assistant_suffix": len(tokenized.assistant_prefix_ids)
        + len(tokenized.response_ids)
        + len(tokenized.assistant_suffix_ids),
    }
    if dict(expected) != actual_counts:
        raise ValueError(
            f"runtime tokenization changed for candidate {example.get('example_id')}: "
            f"expected {dict(expected)}, found {actual_counts}"
        )
    return [int(token_id) for token_id in tokenized.response_ids]


def _runner_example(example: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "example_id",
        "target_model",
        "question",
        "base_question_id",
        "prompt",
        "response",
        "annotation_row_ids",
        "question_ids",
        "label_types",
        "labeling_reasons",
        "hint_types",
        "hint_datasets",
        "src_types",
        "diversity",
        "selection_membership",
        "token_counts",
    )
    return {field: deepcopy(example[field]) for field in fields}


def build_screening_manifest(
    *,
    selection: Mapping[str, Any],
    selection_path: Path,
    expected_selection_sha256: str,
    tokenizer: PreTrainedTokenizerBase,
    tokenizer_path: Path,
    model_id: str,
    model_revision: str,
    seed: str = DEFAULT_SCREENING_SEED,
    targets_per_example: int = DEFAULT_TARGETS_PER_EXAMPLE,
) -> dict[str, Any]:
    """Create one deterministic multi-prompt wave of independent probes."""

    selection_sha256 = _require_exact_selection_source(
        selection_path, expected_selection_sha256
    )
    if selection != load_selection(selection_path):
        raise ValueError(
            "in-memory selection does not match the hash-bound selection file"
        )
    _validate_candidate_contract(selection)
    _validate_tokenizer_provenance(
        selection=selection,
        tokenizer=tokenizer,
        tokenizer_path=tokenizer_path,
        model_id=model_id,
        model_revision=model_revision,
    )
    if not seed:
        raise ValueError("screening seed must be non-empty")
    if targets_per_example != DEFAULT_TARGETS_PER_EXAMPLE:
        raise ValueError(
            f"prompt screening requires exactly {DEFAULT_TARGETS_PER_EXAMPLE} targets"
        )

    items: list[dict[str, Any]] = []
    selected = _selected_examples(selection)
    for inventory, example in selected:
        response_ids = _runtime_response_ids(tokenizer, example)
        if len(response_ids) < targets_per_example:
            raise ValueError(
                f"candidate {example['example_id']} has fewer than "
                f"{targets_per_example} response tokens"
            )
        draws = select_stratified_random_positions(
            len(response_ids),
            targets_per_example,
            seed=seed,
            example_id=str(example["example_id"]),
        )
        for draw in draws:
            position = int(draw["response_token_position"])
            target_selection = {
                "kind": "explicit_response_positions",
                "response_token_positions": [position],
                "width": 1,
                "final_target_token_id": response_ids[position],
                # This intentionally is not the runner's frozen statistical
                # ``sampling`` field: a screening wave can contain many prompts,
                # and none of these positions are final trace targets.
                "screening_selection": {
                    "purpose": "prompt_screening_estimation",
                    "selection_reason": (
                        "one stable uniform response position from each of 16 "
                        "contiguous strata"
                    ),
                    "candidate_inventory": inventory,
                    "source_selection_sha256": selection_sha256,
                    "final_trace_prompt_membership_frozen": False,
                    "final_trace_target_membership_frozen": False,
                    "position_selection": {
                        "design": (
                            "one_uniform_position_per_contiguous_response_stratum"
                        ),
                        "seed": seed,
                        **{
                            field: value
                            for field, value in draw.items()
                            if field
                            not in {"selection_probability", "projection_weight"}
                        },
                    },
                },
            }
            identity = {
                "schema_version": SCHEMA_VERSION,
                "wave_id": WAVE_ID,
                "example_id": example["example_id"],
                "target_response_position": position,
                "screening_seed": seed,
                "source_selection_sha256": selection_sha256,
                "objective": "single_selected_logit",
            }
            items.append(
                {
                    "artifact_id": (
                        "probe-source-" + _sha256_bytes(_canonical_json(identity))[:24]
                    ),
                    "example": _runner_example(example),
                    "response_token_count": len(response_ids),
                    "target_selection": target_selection,
                    "objective": {
                        "name": "single_selected_logit",
                        "benchmark_only_multi_target": False,
                    },
                }
            )

    expected_items = (
        EXPECTED_DENSE_EXAMPLES + EXPECTED_BROAD_EXAMPLES
    ) * DEFAULT_TARGETS_PER_EXAMPLE
    if len(items) != expected_items:
        raise ValueError(
            f"screening work-item count changed: expected {expected_items}, found {len(items)}"
        )
    candidate_tokenizer = selection["tokenizer"]
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "bonafide_prompt_screening_manifest",
        "screening_contract": {
            "purpose": "prompt_screening_estimation",
            "probe_targets_selected": True,
            "probe_targets_frozen_for_this_estimation": True,
            "final_trace_prompt_membership_frozen": False,
            "final_trace_target_membership_frozen": False,
            "may_not_be_interpreted_as_final_trace_selection": True,
        },
        "source_selection": {
            "path": str(selection_path),
            "sha256": selection_sha256,
            "schema_version": selection["schema_version"],
            "selection_policy": deepcopy(selection["selection_policy"]),
            "dense_example_count": EXPECTED_DENSE_EXAMPLES,
            "broad_example_count": EXPECTED_BROAD_EXAMPLES,
        },
        "dataset": deepcopy(selection["dataset"]),
        "tokenizer": {
            "model_id": model_id,
            "revision": model_revision,
            "class": candidate_tokenizer["class"],
            "chat_template_sha256": candidate_tokenizer["chat_template_sha256"],
            "file_manifest": deepcopy(candidate_tokenizer["file_manifest"]),
            "length_semantics": deepcopy(candidate_tokenizer["length_semantics"]),
        },
        "execution_contract": {
            "batch_size": 1,
            "trace_units_are_independent": True,
            "merge_graphs": False,
            "model_load_scope": "selected_wave",
            "resident_model_across_wave": True,
        },
        "waves": [
            {
                "wave_id": WAVE_ID,
                "purpose": "prompt-screening estimation; final trace membership is not frozen",
                "screening_design": {
                    "design": "16_stratified_independent_targets_per_response",
                    "seed": seed,
                    "sampler": "sha256-rejection-v1",
                    "targets_per_example": DEFAULT_TARGETS_PER_EXAMPLE,
                    "example_count": EXPECTED_DENSE_EXAMPLES + EXPECTED_BROAD_EXAMPLES,
                    "final_trace_membership_frozen": False,
                },
                "items": items,
            }
        ],
    }


def write_screening_manifest(manifest: Mapping[str, Any], output_path: Path) -> None:
    """Atomically persist a screening manifest."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output_path)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION_PATH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--revision", required=True, help="Exact model/tokenizer revision"
    )
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--expected-selection-sha256", default=DEFAULT_SELECTION_SHA256)
    parser.add_argument("--seed", default=DEFAULT_SCREENING_SEED)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow tokenizer dependencies outside the supplied local snapshot",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    selection = load_selection(args.selection)
    source = resolve_pretrained_source(
        model_id=args.model_id,
        revision=args.revision,
        local_files_only=not args.allow_download,
        explicit_path=args.tokenizer_path,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        source,
        revision=None if source != args.model_id else args.revision,
        local_files_only=not args.allow_download,
    )
    manifest = build_screening_manifest(
        selection=selection,
        selection_path=args.selection,
        expected_selection_sha256=args.expected_selection_sha256,
        tokenizer=tokenizer,
        tokenizer_path=args.tokenizer_path,
        model_id=args.model_id,
        model_revision=args.revision,
        seed=args.seed,
    )
    write_screening_manifest(manifest, args.output)
    payload = args.output.read_bytes()
    wave = manifest["waves"][0]
    print(
        json.dumps(
            {
                "output": str(args.output),
                "manifest_sha256": _sha256_bytes(payload),
                "wave_id": wave["wave_id"],
                "example_count": wave["screening_design"]["example_count"],
                "item_count": len(wave["items"]),
                "targets_per_example": wave["screening_design"]["targets_per_example"],
                "final_trace_membership_frozen": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
