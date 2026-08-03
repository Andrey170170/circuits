"""Build deterministic, provenance-preserving BonaFide benchmark manifests.

The manifest is deliberately a list of independent trace units.  It does not
merge traces or prescribe downstream clustering.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from circuits.tracing.trace import tokenize_teacher_forced_response
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerBase

SCHEMA_VERSION = "bonafide-trace-benchmark/v1"
DEFAULT_WAVE2_WIDTHS = (1, 2, 4, 8, 16, 32)
DEFAULT_WAVE2B_TARGET_COUNT = 16
DEFAULT_WAVE2C_STRATUM_COUNT = 8
DEFAULT_WAVE2C_SEED = "bonafide-wave2c-v1"


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class BonaFideExample:
    """One unique model input/response with every source annotation attached."""

    example_id: str
    target_model: str
    prompt: str
    response: str
    annotation_row_ids: tuple[str, ...]
    question_ids: tuple[str, ...]
    label_types: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "target_model": self.target_model,
            "prompt": self.prompt,
            "response": self.response,
            "annotation_row_ids": list(self.annotation_row_ids),
            "question_ids": list(self.question_ids),
            "label_types": list(self.label_types),
        }


def load_deduplicated_examples(csv_path: Path) -> list[BonaFideExample]:
    """Deduplicate on ``(target_model, prompt, cot)`` without losing annotations."""

    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"id", "question_id", "label_type", "target_model", "prompt", "cot"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"BonaFide CSV is missing columns: {sorted(missing)}")
        for row in reader:
            key = (row["target_model"], row["prompt"], row["cot"])
            grouped[key].append(row)

    examples: list[BonaFideExample] = []
    for (target_model, prompt, response), rows in grouped.items():
        identity = {
            "target_model": target_model,
            "prompt": prompt,
            "response": response,
        }
        examples.append(
            BonaFideExample(
                example_id=f"bf-{_sha256(_canonical_json(identity))[:20]}",
                target_model=target_model,
                prompt=prompt,
                response=response,
                annotation_row_ids=tuple(sorted({row["id"] for row in rows})),
                question_ids=tuple(sorted({row["question_id"] for row in rows})),
                label_types=tuple(sorted({row["label_type"] for row in rows})),
            )
        )
    return sorted(examples, key=lambda example: example.example_id)


def response_trace_tokens(
    tokenizer: PreTrainedTokenizerBase, prompt: str, response: str
) -> list[int]:
    """Return exact response IDs under the tracing chat-template boundary."""

    tokenized = tokenize_teacher_forced_response(
        cast(PreTrainedTokenizer, tokenizer), prompt, response
    )
    return tokenized.response_ids


def _find_example_by_annotation_id(
    examples: Sequence[BonaFideExample], annotation_id: str
) -> BonaFideExample:
    matches = [
        example for example in examples if annotation_id in example.annotation_row_ids
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one deduplicated example for annotation {annotation_id!r}; "
            f"found {len(matches)}"
        )
    return matches[0]


def select_varied_lengths(
    tokenized: Sequence[tuple[BonaFideExample, list[int]]],
    count: int,
    anchors: Iterable[BonaFideExample] = (),
) -> list[tuple[BonaFideExample, list[int]]]:
    """Select deterministic length-quantile representatives plus explicit anchors."""

    if count < 1:
        raise ValueError("count must be positive")
    ordered = sorted(tokenized, key=lambda item: (len(item[1]), item[0].example_id))
    if not ordered:
        raise ValueError("No non-empty candidate responses remain")

    by_id = {example.example_id: (example, ids) for example, ids in ordered}
    selected: list[tuple[BonaFideExample, list[int]]] = []
    selected_ids: set[str] = set()
    for anchor in anchors:
        item = by_id.get(anchor.example_id)
        if item is None:
            raise ValueError(
                f"Anchor {anchor.annotation_row_ids[0]!r} is outside the candidates"
            )
        if anchor.example_id not in selected_ids:
            selected.append(item)
            selected_ids.add(anchor.example_id)
    if len(selected) > count:
        raise ValueError("More unique anchors were requested than the sample count")

    slots = count - len(selected)
    if slots:
        desired = (
            [0]
            if slots == 1
            else [round(i * (len(ordered) - 1) / (slots - 1)) for i in range(slots)]
        )
        for index in desired:
            candidates = sorted(
                range(len(ordered)),
                key=lambda candidate: (abs(candidate - index), candidate),
            )
            for candidate in candidates:
                item = ordered[candidate]
                if item[0].example_id not in selected_ids:
                    selected.append(item)
                    selected_ids.add(item[0].example_id)
                    break
        for item in ordered:
            if len(selected) == count:
                break
            if item[0].example_id not in selected_ids:
                selected.append(item)
                selected_ids.add(item[0].example_id)

    return sorted(selected, key=lambda item: (len(item[1]), item[0].example_id))


def select_evenly_spaced_positions(response_length: int, count: int) -> list[int]:
    """Select deterministic response positions, including both span endpoints."""

    if response_length < 1:
        raise ValueError("response_length must be positive")
    if count < 1:
        raise ValueError("count must be positive")
    if count > response_length:
        raise ValueError(
            f"Cannot select {count} distinct positions from {response_length} response tokens"
        )
    if count == 1:
        return [response_length - 1]
    return [
        round(index * (response_length - 1) / (count - 1)) for index in range(count)
    ]


def select_stratified_random_positions(
    response_length: int,
    stratum_count: int,
    *,
    seed: str,
    example_id: str,
) -> list[dict[str, Any]]:
    """Draw one stable uniform position from each contiguous response stratum.

    The SHA-256 rejection sampler avoids modulo bias while remaining independent
    of Python's process hash seed and random-number-generator implementation.
    Returned records contain the sampling quantities needed to project sampled
    measurements back to all response positions.
    """

    if response_length < 1:
        raise ValueError("response_length must be positive")
    if stratum_count < 1:
        raise ValueError("stratum_count must be positive")
    if stratum_count > response_length:
        raise ValueError(
            f"Cannot create {stratum_count} non-empty strata from "
            f"{response_length} response tokens"
        )
    if not seed:
        raise ValueError("seed must be non-empty")
    if not example_id:
        raise ValueError("example_id must be non-empty")

    selections: list[dict[str, Any]] = []
    digest_range = 1 << 256
    for stratum_index in range(stratum_count):
        start = (stratum_index * response_length) // stratum_count
        end_exclusive = ((stratum_index + 1) * response_length) // stratum_count
        size = end_exclusive - start
        rejection_limit = digest_range - (digest_range % size)
        counter = 0
        while True:
            draw_key = {
                "algorithm": "sha256-rejection-v1",
                "seed": seed,
                "example_id": example_id,
                "response_length": response_length,
                "stratum_count": stratum_count,
                "stratum_index": stratum_index,
                "counter": counter,
            }
            draw = int.from_bytes(
                hashlib.sha256(_canonical_json(draw_key)).digest(), "big"
            )
            if draw < rejection_limit:
                break
            counter += 1
        selections.append(
            {
                "response_token_position": start + (draw % size),
                "stratum_index": stratum_index,
                "stratum_count": stratum_count,
                "stratum_start": start,
                "stratum_end_exclusive": end_exclusive,
                "stratum_size": size,
                "selection_probability": 1 / size,
                "projection_weight": size,
                "sampler": "sha256-rejection-v1",
            }
        )
    return selections


def _trace_item(
    *,
    wave_id: str,
    example: BonaFideExample,
    token_ids: list[int],
    positions: list[int],
    benchmark_only_multi_target: bool,
    sampling: dict[str, Any] | None = None,
) -> dict[str, Any]:
    target_selection = {
        "kind": "explicit_response_positions",
        "response_token_positions": positions,
        "width": len(positions),
        "final_target_token_id": token_ids[positions[-1]],
    }
    if sampling is not None:
        target_selection["sampling"] = sampling
    identity = {
        "schema_version": SCHEMA_VERSION,
        "wave_id": wave_id,
        "example_id": example.example_id,
        "target_response_positions": positions,
        "objective": "sum_selected_logits",
    }
    # Preserve historical IDs for unsampled waves while ensuring two
    # probability samples of the same target position remain distinct source
    # work units.
    if sampling is not None:
        identity["target_selection"] = target_selection
    return {
        "artifact_id": f"trace-{_sha256(_canonical_json(identity))[:24]}",
        "example": example.to_dict(),
        "response_token_count": len(token_ids),
        "target_selection": target_selection,
        "objective": {
            "name": "sum_selected_logits",
            "benchmark_only_multi_target": benchmark_only_multi_target,
        },
    }


def build_manifest(
    *,
    csv_path: Path,
    tokenizer: PreTrainedTokenizerBase,
    target_model: str,
    model_revision: str,
    sample_count: int = 10,
    max_response_tokens: int | None = 2048,
    anchor_annotation_ids: Sequence[str] = (),
    wave2_annotation_id: str | None = None,
    wave2_widths: Sequence[int] = DEFAULT_WAVE2_WIDTHS,
    wave2b_target_count: int = DEFAULT_WAVE2B_TARGET_COUNT,
    wave2c_stratum_count: int = DEFAULT_WAVE2C_STRATUM_COUNT,
    wave2c_seed: str = DEFAULT_WAVE2C_SEED,
) -> dict[str, Any]:
    examples = [
        example
        for example in load_deduplicated_examples(csv_path)
        if example.target_model == target_model and example.response.strip()
    ]
    tokenized = [
        (
            example,
            response_trace_tokens(tokenizer, example.prompt, example.response),
        )
        for example in examples
    ]
    if max_response_tokens is not None:
        if max_response_tokens < 1:
            raise ValueError("max_response_tokens must be positive or None")
        tokenized = [item for item in tokenized if len(item[1]) <= max_response_tokens]
    anchors = [
        _find_example_by_annotation_id(examples, annotation_id)
        for annotation_id in anchor_annotation_ids
    ]
    wave1_examples = select_varied_lengths(tokenized, sample_count, anchors)

    wave1_id = "wave1-mixed-final-token"
    wave1_items = [
        _trace_item(
            wave_id=wave1_id,
            example=example,
            token_ids=ids,
            positions=[len(ids) - 1],
            benchmark_only_multi_target=False,
        )
        for example, ids in wave1_examples
    ]

    if wave2_annotation_id:
        wave2_example = _find_example_by_annotation_id(examples, wave2_annotation_id)
        wave2_ids = response_trace_tokens(
            tokenizer, wave2_example.prompt, wave2_example.response
        )
    else:
        eligible = [
            item for item in wave1_examples if len(item[1]) >= max(wave2_widths)
        ]
        if not eligible:
            eligible = [item for item in tokenized if len(item[1]) >= max(wave2_widths)]
        if not eligible:
            raise ValueError(
                f"No response is long enough for wave 2 width {max(wave2_widths)}"
            )
        # A median-length selected sample avoids making sequence length itself an extreme.
        eligible = sorted(eligible, key=lambda item: (len(item[1]), item[0].example_id))
        wave2_example, wave2_ids = eligible[len(eligible) // 2]

    wave1_example_ids = {example.example_id for example, _ in wave1_examples}
    if wave2_example.example_id not in wave1_example_ids:
        raise ValueError(
            "Wave 2 reference must belong to Wave 1 so Wave 2c is exactly the "
            "remaining Wave 1 prompts"
        )

    widths = sorted(set(wave2_widths))
    if not widths or widths[0] < 1:
        raise ValueError("wave2 widths must be positive")
    if widths[-1] > len(wave2_ids):
        raise ValueError(
            f"Wave 2 width {widths[-1]} exceeds response length {len(wave2_ids)}"
        )
    wave2_id = "wave2-progressive-target-window"
    wave2_items = [
        _trace_item(
            wave_id=wave2_id,
            example=wave2_example,
            token_ids=wave2_ids,
            positions=list(range(len(wave2_ids) - width, len(wave2_ids))),
            benchmark_only_multi_target=width > 1,
        )
        for width in widths
    ]

    wave2b_id = "wave2b-independent-target-positions"
    wave2b_positions = select_evenly_spaced_positions(
        len(wave2_ids), wave2b_target_count
    )
    wave2b_items = [
        _trace_item(
            wave_id=wave2b_id,
            example=wave2_example,
            token_ids=wave2_ids,
            positions=[position],
            benchmark_only_multi_target=False,
        )
        for position in wave2b_positions
    ]

    wave2c_waves: list[dict[str, Any]] = []
    wave2c_examples = [
        item
        for item in wave1_examples
        if item[0].example_id != wave2_example.example_id
    ]
    for prompt_index, (example, ids) in enumerate(wave2c_examples, start=1):
        stratum_count = min(wave2c_stratum_count, len(ids))
        selections = select_stratified_random_positions(
            len(ids),
            stratum_count,
            seed=wave2c_seed,
            example_id=example.example_id,
        )
        wave2c_id = (
            f"wave2c-stratified-independent-{prompt_index:02d}-{example.example_id}"
        )
        wave2c_items = [
            _trace_item(
                wave_id=wave2c_id,
                example=example,
                token_ids=ids,
                positions=[selection["response_token_position"]],
                benchmark_only_multi_target=False,
                sampling={
                    "design": "one_uniform_position_per_contiguous_response_stratum",
                    "seed": wave2c_seed,
                    **selection,
                },
            )
            for selection in selections
        ]
        wave2c_waves.append(
            {
                "wave_id": wave2c_id,
                "purpose": (
                    "stratified-random independent targets for one Wave 1 response"
                ),
                "sampling_design": {
                    "design": "one_uniform_position_per_contiguous_response_stratum",
                    "seed": wave2c_seed,
                    "sampler": "sha256-rejection-v1",
                    "response_token_population_size": len(ids),
                    "stratum_count": stratum_count,
                    "excluded_reference_example_id": wave2_example.example_id,
                },
                "items": wave2c_items,
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": {
            "path": str(csv_path),
            "sha256": _file_sha256(csv_path),
            "dedupe_key": ["target_model", "prompt", "cot"],
        },
        "tokenizer": {
            "model_id": target_model,
            "revision": model_revision,
            "length_semantics": "exact response span derived from the tracing chat template",
        },
        "execution_contract": {
            "batch_size": 1,
            "trace_units_are_independent": True,
            "merge_graphs": False,
            "model_load_scope": "selected_wave",
            "wave1_max_response_tokens": max_response_tokens,
        },
        "waves": [
            {
                "wave_id": wave1_id,
                "purpose": "mixed response lengths with one final non-EOS target each",
                "items": wave1_items,
            },
            {
                "wave_id": wave2_id,
                "purpose": "Jacobian scaling with progressive target-window width",
                "items": wave2_items,
            },
            {
                "wave_id": wave2b_id,
                "purpose": "independent single-target traces across response positions",
                "items": wave2b_items,
            },
            *wave2c_waves,
        ],
    }


def write_manifest(manifest: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(output_path)


def resolve_pretrained_source(
    *,
    model_id: str,
    revision: str,
    local_files_only: bool,
    explicit_path: Path | None = None,
) -> str:
    """Resolve a pinned cached snapshot before calling Transformers offline."""

    if explicit_path is not None:
        if not explicit_path.is_dir():
            raise FileNotFoundError(f"local model snapshot is absent: {explicit_path}")
        return str(explicit_path)
    if local_files_only:
        return snapshot_download(
            repo_id=model_id,
            revision=revision,
            local_files_only=True,
        )
    return model_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("BonaFide.csv"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument(
        "--revision", required=True, help="Exact Hugging Face model revision"
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        help="Optional local snapshot path; provenance still uses --model-id/--revision",
    )
    parser.add_argument("--sample-count", type=int, default=10)
    parser.add_argument(
        "--max-response-tokens",
        type=int,
        default=2048,
        help="Safety cap for Wave 1 candidates (default: 2048 exact chat-template tokens)",
    )
    parser.add_argument(
        "--anchor-id", action="append", default=[], help="BonaFide annotation row ID"
    )
    parser.add_argument(
        "--wave2-id", help="Annotation row ID to use for progressive windows"
    )
    parser.add_argument(
        "--wave2-widths", nargs="+", type=int, default=list(DEFAULT_WAVE2_WIDTHS)
    )
    parser.add_argument(
        "--wave2b-target-count",
        type=int,
        default=DEFAULT_WAVE2B_TARGET_COUNT,
        help="Number of evenly spaced independent target positions (default: 16)",
    )
    parser.add_argument(
        "--wave2c-stratum-count",
        type=int,
        default=DEFAULT_WAVE2C_STRATUM_COUNT,
        help="Number of response-position strata per non-reference Wave 1 prompt (default: 8)",
    )
    parser.add_argument(
        "--wave2c-seed",
        default=DEFAULT_WAVE2C_SEED,
        help="Stable seed recorded with deterministic Wave 2c samples",
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
    manifest = build_manifest(
        csv_path=args.csv,
        tokenizer=tokenizer,
        target_model=args.model_id,
        model_revision=args.revision,
        sample_count=args.sample_count,
        max_response_tokens=args.max_response_tokens,
        anchor_annotation_ids=args.anchor_id,
        wave2_annotation_id=args.wave2_id,
        wave2_widths=args.wave2_widths,
        wave2b_target_count=args.wave2b_target_count,
        wave2c_stratum_count=args.wave2c_stratum_count,
        wave2c_seed=args.wave2c_seed,
    )
    write_manifest(manifest, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "wave_counts": {
                    wave["wave_id"]: len(wave["items"]) for wave in manifest["waves"]
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
