"""Bounded 96-token segmentation compatibility qualification.

V4 is additive to the sealed v3 qualification.  It introduces a quote-aware
sentence splitter under a new policy identity, selects a deterministic repair
packet from the v3 boundary findings, and applies only the selected refined
zero-shot prompt in three body-identical replicas.  The results remain coarse
trace-sampling metadata, not semantic ground truth.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from circuits.analysis.bonafide.canonical import canonical_sha256, file_sha256
from circuits.analysis.bonafide.coarse_sampling_annotation import (
    BOUNDARY_CONCERNS,
    COARSE_TAGS,
    CONFIDENCE_VALUES,
    _covering_span,
    segment_document,
)
from circuits.analysis.bonafide.coarse_sampling_annotation_v2 import (
    ARM_TARGET_ONLY,
    _inline_markup,
    cost_plan_v2,
)
from circuits.analysis.bonafide.coarse_sampling_annotation_v3 import (
    ARM_ZERO_SHOT,
    _base_system_prompt,
)
from circuits.analysis.bonafide.process_annotation import _terminal_serialization_span
from circuits.labeling.io import read_jsonl

CONFIG_SCHEMA = "adag.process-witness.coarse-annotation-config.v4"
UNIT_SCHEMA = "adag.process-witness.coarse-unit.v2"
REQUEST_SCHEMA = "adag.process-witness.coarse-request.v4"
BUNDLE_SCHEMA = "adag.process-witness.coarse-qualification-bundle.v4"
COST_PLAN_SCHEMA = "adag.process-witness.coarse-cost-plan.v4"
DECISION_SCHEMA_NAME = "process_witness_coarse_decisions_v4"
OPENAI_BATCH_ENDPOINT = "/v1/responses"
SEGMENTATION_POLICY_ID = "token-exclusive-sentence-line-quote-aware-v2"
SEGMENTATION_CONCERNS = frozenset(("split_needed", "merge_previous", "merge_next"))
FOCAL_UNITS_PER_WINDOW = 2
WINDOW_COUNT = 12
REPLICAS_PER_WINDOW = 3


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unreadable JSON object: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _verify_self_hash(value: Mapping[str, Any], field: str, label: str) -> None:
    payload = dict(value)
    observed = payload.pop(field, None)
    if not isinstance(observed, str) or observed != canonical_sha256(payload):
        raise ValueError(f"{label} self-hash drift")


def _text_units_quote_aware(text: str) -> list[tuple[int, int]]:
    """Split lines/sentences while retaining terminal closing quotes/brackets.

    This deliberately has a new identity instead of changing the historical
    `_text_units` implementation used by v1-v3. Decimal points remain internal.
    """

    output: list[tuple[int, int]] = []
    # Do not treat every punctuation-plus-quote as a boundary: quoted examples
    # such as `The "Decision: ..." value` are common.  The added branch is
    # deliberately conservative and requires a capitalized following sentence.
    terminal = re.compile(
        r"(?P<bare>[.!?]+)(?=\s+|$)|"
        r"(?P<quoted>[.!?]+)(?P<closers>[\"\u201d]+)(?=[ \t]+[A-Z])"
    )
    for line in re.finditer(r"[^\r\n]+", text):
        raw = line.group(0)
        left = len(raw) - len(raw.lstrip())
        right = len(raw.rstrip())
        if left >= right:
            continue
        content = raw[left:right]
        absolute = line.start() + left
        if content.startswith("{") and content.endswith("}"):
            output.append((absolute, absolute + len(content)))
            continue
        if re.fullmatch(r"</?think>", content, re.IGNORECASE):
            continue
        unit_start = 0
        for found in terminal.finditer(content):
            group = "bare" if found.group("bare") is not None else "quoted"
            punct_start = found.start(group)
            punct_end = found.end(group)
            if (
                content[punct_start] == "."
                and punct_start > 0
                and punct_end < len(content)
                and content[punct_start - 1].isdigit()
                and content[punct_end].isdigit()
            ):
                continue
            end = found.end()
            piece = content[unit_start:end]
            trim_left = len(piece) - len(piece.lstrip())
            if piece.strip():
                output.append((absolute + unit_start + trim_left, absolute + end))
            unit_start = end
        tail = content[unit_start:]
        trim_left = len(tail) - len(tail.lstrip())
        if tail.strip():
            output.append((absolute + unit_start + trim_left, absolute + len(content)))
    return output


def _semantic_intervals_v4(
    text: str, tokens: Sequence[Sequence[int]]
) -> list[dict[str, int]]:
    intervals: list[dict[str, int]] = []
    token_cursor = 0
    for core_start, core_end in _text_units_quote_aware(text):
        while token_cursor < len(tokens) and int(tokens[token_cursor][2]) <= core_start:
            token_cursor += 1
        token_end = token_cursor
        while token_end < len(tokens) and int(tokens[token_end][1]) < core_end:
            token_end += 1
        if token_cursor == token_end:
            continue
        candidate = {
            "token_start": token_cursor,
            "token_end": token_end,
            "core_start": core_start,
            "core_end": core_end,
        }
        if intervals and candidate["token_start"] < intervals[-1]["token_end"]:
            intervals[-1]["token_end"] = max(
                intervals[-1]["token_end"], candidate["token_end"]
            )
            intervals[-1]["core_start"] = min(
                intervals[-1]["core_start"], candidate["core_start"]
            )
            intervals[-1]["core_end"] = max(
                intervals[-1]["core_end"], candidate["core_end"]
            )
        else:
            intervals.append(candidate)
    return intervals


def segment_document_v4(
    document: Mapping[str, Any], *, maximum_semantic_unit_tokens: int = 96
) -> list[dict[str, Any]]:
    """Partition every authoritative response token under the new v4 policy."""

    if maximum_semantic_unit_tokens != 96:
        raise ValueError("coarse v4 segmentation maximum must remain 96 tokens")
    response_id = str(document["response_id"])
    text = str(document["text"])
    tokenization = document["tokenization"]
    tokens = tokenization["tokens"]
    if len(tokens) != tokenization["token_count"] or not tokens:
        raise ValueError(f"token count drift for {response_id}")
    previous_end = -1
    for token in tokens:
        if (
            not isinstance(token, list)
            or len(token) != 3
            or not all(isinstance(value, int) for value in token)
            or token[1] < 0
            or token[2] < token[1]
            or token[2] > len(text)
            or token[1] < previous_end
        ):
            raise ValueError(f"invalid authoritative token offsets for {response_id}")
        previous_end = token[1]

    pieces: list[dict[str, Any]] = []
    cursor = 0
    for interval in _semantic_intervals_v4(text, tokens):
        if interval["token_start"] > cursor:
            pieces.append(
                {
                    "kind": "surface_or_control",
                    "token_start": cursor,
                    "token_end": interval["token_start"],
                    "core_start": None,
                    "core_end": None,
                    "fragment_of": None,
                }
            )
        width = interval["token_end"] - interval["token_start"]
        fragment_identity = None
        if width > maximum_semantic_unit_tokens:
            fragment_identity = canonical_sha256(
                {
                    "segmentation_policy_id": SEGMENTATION_POLICY_ID,
                    "response_id": response_id,
                    "token_span": [interval["token_start"], interval["token_end"]],
                    "core_span": [interval["core_start"], interval["core_end"]],
                }
            )
        for start in range(
            interval["token_start"], interval["token_end"], maximum_semantic_unit_tokens
        ):
            end = min(start + maximum_semantic_unit_tokens, interval["token_end"])
            covering = _covering_span(tokens, start, end)
            pieces.append(
                {
                    "kind": "semantic_text",
                    "token_start": start,
                    "token_end": end,
                    "core_start": max(interval["core_start"], covering[0]),
                    "core_end": min(interval["core_end"], covering[1]),
                    "fragment_of": fragment_identity,
                }
            )
        cursor = interval["token_end"]
    if cursor < len(tokens):
        pieces.append(
            {
                "kind": "surface_or_control",
                "token_start": cursor,
                "token_end": len(tokens),
                "core_start": None,
                "core_end": None,
                "fragment_of": None,
            }
        )

    terminal_span = _terminal_serialization_span(text)
    policy = {
        "policy_id": SEGMENTATION_POLICY_ID,
        "maximum_semantic_unit_tokens": maximum_semantic_unit_tokens,
    }
    policy_hash = canonical_sha256(policy)
    units: list[dict[str, Any]] = []
    for sequence_index, piece in enumerate(pieces):
        token_start, token_end = piece["token_start"], piece["token_end"]
        covering = _covering_span(tokens, token_start, token_end)
        core = (
            covering
            if piece["core_start"] is None
            else [piece["core_start"], piece["core_end"]]
        )
        deterministic_tag = None
        assignment_route = "openai_pending"
        if piece["kind"] == "surface_or_control":
            deterministic_tag = "surface_or_control"
            assignment_route = "deterministic_surface"
        elif (
            terminal_span is not None
            and core[0] >= terminal_span[0]
            and core[1] <= terminal_span[1]
        ):
            deterministic_tag = "final_answer"
            assignment_route = "deterministic_terminal_serialization"
        identity = {
            "schema_version": UNIT_SCHEMA,
            "segmentation_policy_sha256": policy_hash,
            "response_id": response_id,
            "sequence_index": sequence_index,
            "unit_kind": piece["kind"],
            "token_span": [token_start, token_end],
            "core_character_span": core,
            "covering_character_span": covering,
        }
        unit_id = f"pwcoarseunitv4-{canonical_sha256(identity)[:32]}"
        units.append(
            {
                **identity,
                "unit_id": unit_id,
                "text": text[core[0] : core[1]],
                "covering_text": text[covering[0] : covering[1]],
                "fragment_of": piece["fragment_of"],
                "deterministic_tag": deterministic_tag,
                "assignment_route": assignment_route,
                "segmentation_policy": policy,
            }
        )
    if [u["token_span"][0] for u in units] != [0] + [
        u["token_span"][1] for u in units[:-1]
    ] or units[-1]["token_span"][1] != len(tokens):
        raise ValueError(f"coarse v4 units do not exactly partition {response_id}")
    return units


def decision_json_schema_v4() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["decisions"],
        "properties": {
            "decisions": {
                "type": "array",
                "minItems": FOCAL_UNITS_PER_WINDOW,
                "maxItems": FOCAL_UNITS_PER_WINDOW,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "unit_id",
                        "tag",
                        "confidence",
                        "boundary_concerns",
                        "boundary_note",
                    ],
                    "properties": {
                        "unit_id": {"type": "string"},
                        "tag": {"type": "string", "enum": list(COARSE_TAGS)},
                        "confidence": {
                            "type": "string",
                            "enum": list(CONFIDENCE_VALUES),
                        },
                        "boundary_concerns": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": list(BOUNDARY_CONCERNS),
                            },
                        },
                        "boundary_note": {"type": "string"},
                    },
                },
            }
        },
    }


def load_coarse_v4_config(path: Path) -> dict[str, Any]:
    value = _load_object(path)
    if value.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("unsupported coarse v4 config schema")
    if tuple(value.get("tags", {})) != COARSE_TAGS or value.get(
        "boundary_concerns"
    ) != list(BOUNDARY_CONCERNS):
        raise ValueError("coarse v4 ontology drift")
    q = value.get("qualification", {})
    if q != {
        "unique_window_count": WINDOW_COUNT,
        "focal_units_per_window": FOCAL_UNITS_PER_WINDOW,
        "replicas_per_window": REPLICAS_PER_WINDOW,
        "arm": ARM_ZERO_SHOT,
        "maximum_semantic_unit_tokens": 96,
        "segmentation_policy_id": SEGMENTATION_POLICY_ID,
    }:
        raise ValueError("coarse v4 qualification shape drift")
    provider = value.get("provider", {})
    if (
        provider.get("name") != "openai"
        or provider.get("model") != "gpt-5.6-luna"
        or provider.get("api_surface") != "responses"
        or provider.get("transport") != "native_batch"
        or provider.get("batch_endpoint") != OPENAI_BATCH_ENDPOINT
        or provider.get("reasoning") != {"effort": "medium"}
        or provider.get("max_output_tokens") != 16384
        or provider.get("store") is not False
    ):
        raise ValueError("coarse v4 provider contract drift")
    source = value.get("source", {})
    required_hashes = (
        "workstation_bundle_sha256",
        "v3_qualification_manifest_sha256",
        "v3_review_packet_manifest_sha256",
        "v3_human_ledger_sha256",
        "v3_post_seal_corrections_sha256",
    )
    if any(
        not isinstance(source.get(key), str) or len(source[key]) != 64
        for key in required_hashes
    ):
        raise ValueError("coarse v4 source binding drift")
    return value


def render_v4_user_prompt(
    document: Mapping[str, Any],
    focal_units: Sequence[Mapping[str, Any]],
    all_response_units: Sequence[Mapping[str, Any]],
) -> tuple[str, dict[str, Any]]:
    if len(focal_units) != FOCAL_UNITS_PER_WINDOW:
        raise ValueError("coarse v4 requests require exactly two focal units")
    response = str(document["text"])
    prompt = str(document["task_context"]["prompt"])
    response_sha = hashlib.sha256(response.encode()).hexdigest()
    prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()
    if (
        response_sha != document["text_sha256"]
        or prompt_sha != document["prompt_sha256"]
    ):
        raise ValueError("coarse v4 authoritative document hash drift")
    target_ids = [str(unit["unit_id"]) for unit in focal_units]
    marked, tags = _inline_markup(
        response=response,
        response_id=str(document["response_id"]),
        units=focal_units,
        focal_ids=set(target_ids),
        mark_all_units=False,
    )
    prompt_begin = f"<<<BEGIN_TASK_PROMPT_SHA256:{prompt_sha}>>>"
    prompt_end = f"<<<END_TASK_PROMPT_SHA256:{prompt_sha}>>>"
    response_begin = f"<<<BEGIN_FULL_RESPONSE_SHA256:{response_sha}>>>"
    response_end = f"<<<END_FULL_RESPONSE_SHA256:{response_sha}>>>"
    content = (
        "COMPLETE TASK PROMPT (quoted context only; exact raw text follows):\n"
        f"{prompt_begin}\n{prompt}\n{prompt_end}\n\n"
        "COMPLETE MODEL RESPONSE WITH LOSSLESS target_only MARKUP "
        "(quoted context only; removing all {{...}} unit tags reconstructs the exact authoritative raw response):\n"
        f"{response_begin}\n{marked}\n{response_end}\n\n"
        "TARGET UNIT IDS IN RESPONSE ORDER:\n- "
        + "\n- ".join(target_ids)
        + "\n\nReturn decisions for these two TARGET unit_ids exactly once and no others."
    )
    return content, {
        "arm_id": ARM_TARGET_ONLY,
        "full_response_sha256": response_sha,
        "raw_response_utf8_bytes": len(response.encode()),
        "response_unit_count": len(all_response_units),
        "target_markup_count": 2,
        "context_markup_count": 0,
        "inserted_tag_count": len(tags),
        "lossless_reconstruction_verified": True,
    }


def _corrected_segmentation_rows(
    ledger_rows: Sequence[Mapping[str, Any]],
    correction_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    corrections = {str(row["unit_id"]): row for row in correction_rows}
    output = []
    for source in ledger_rows:
        row = dict(source)
        correction = corrections.get(str(row.get("unit_id")))
        if correction is not None:
            if row.get(correction["field"]) != correction["original_value"]:
                raise ValueError("coarse v4 correction original value drift")
            row[correction["field"]] = correction["corrected_value"]
        if SEGMENTATION_CONCERNS.intersection(row.get("boundary_concerns", [])):
            output.append(row)
    return output


def _select_windows(
    *,
    workstation_bundle: Mapping[str, Any],
    review_root: Path,
    ledger_rows: Sequence[Mapping[str, Any]],
    correction_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    review_items = {
        row["unit_id"]: row for row in read_jsonl(review_root / "items.jsonl")
    }
    documents = {
        str(row["response_id"]): row for row in workstation_bundle["documents"]
    }
    defect_rows = _corrected_segmentation_rows(ledger_rows, correction_rows)
    by_response: dict[str, list[dict[str, Any]]] = {}
    defect_groups_by_response: dict[str, set[tuple[int, int]]] = {}
    for row in defect_rows:
        item = review_items.get(row["unit_id"])
        if item is None:
            raise ValueError("coarse v4 defect unit absent from bound review packet")
        response_id = str(item["response_id"])
        by_response.setdefault(response_id, []).append(item)
        concern = next(
            value
            for value in row["boundary_concerns"]
            if value in SEGMENTATION_CONCERNS
        )
        span = tuple(map(int, item["core_character_span"]))
        if concern == "split_needed":
            group_span = span
        else:
            group = segment_document(
                documents[response_id], maximum_semantic_unit_tokens=24
            )
            index = next(
                index
                for index, value in enumerate(group)
                if value["unit_id"] == item["unit_id"]
            )
            adjacent_index = index - 1 if concern == "merge_previous" else index + 1
            if not 0 <= adjacent_index < len(group):
                raise ValueError("coarse v4 merge concern lacks its adjacent unit")
            adjacent = tuple(map(int, group[adjacent_index]["core_character_span"]))
            group_span = (min(span[0], adjacent[0]), max(span[1], adjacent[1]))
        defect_groups_by_response.setdefault(response_id, set()).add(group_span)
    if len(defect_rows) != 24 or len(by_response) != WINDOW_COUNT:
        raise ValueError("coarse v4 defect source census drift")

    prepared: dict[
        str,
        tuple[
            Mapping[str, Any],
            list[dict[str, Any]],
            list[tuple[int, int]],
            list[dict[str, Any]],
        ],
    ] = {}
    for response_id in sorted(by_response):
        document = documents[response_id]
        units = segment_document_v4(document)
        semantic = [u for u in units if u["assignment_route"] == "openai_pending"]
        old_spans = sorted(defect_groups_by_response[response_id])
        repaired_by_id: dict[str, dict[str, Any]] = {}
        for defect_start, defect_end in old_spans:
            group_repaired = [
                u
                for u in semantic
                if u["core_character_span"][0] < defect_end
                and u["core_character_span"][1] > defect_start
            ]
            if (
                not group_repaired
                or min(u["core_character_span"][0] for u in group_repaired)
                > defect_start
                or max(u["core_character_span"][1] for u in group_repaired) < defect_end
            ):
                raise ValueError(
                    "coarse v4 repaired units do not cover old defect group"
                )
            repaired_by_id.update((u["unit_id"], u) for u in group_repaired)
        repaired = sorted(repaired_by_id.values(), key=lambda u: u["sequence_index"])
        if len(repaired) not in (1, 2):
            raise ValueError(
                "coarse v4 expected one or two repair targets per response: "
                f"{response_id} has {len(repaired)}"
            )
        prepared[response_id] = (document, semantic, old_spans, repaired)
    single_repair_responses = sorted(
        response_id
        for response_id, (_document, _semantic, _spans, repaired) in prepared.items()
        if len(repaired) == 1
    )
    if (
        sum(len(value[3]) for value in prepared.values()) != 14
        or len(single_repair_responses) != 10
    ):
        raise ValueError("coarse v4 expected 14 repair targets and ten control slots")
    long_control_responses = set(single_repair_responses[:4])

    windows: list[dict[str, Any]] = []
    focal_units: list[dict[str, Any]] = []
    audits = []
    for window_index, response_id in enumerate(sorted(by_response)):
        document, semantic, old_spans, repaired = prepared[response_id]
        defect_start = min(start for start, _ in old_spans)
        defect_end = max(end for _, end in old_spans)

        selected = list(repaired)
        candidates = [u for u in semantic if u not in selected]
        legacy_spans = {
            (tuple(u["token_span"]), tuple(u["core_character_span"]))
            for u in segment_document(document, maximum_semantic_unit_tokens=24)
            if u["assignment_route"] == "openai_pending" and u["fragment_of"] is None
        }
        short = [
            u
            for u in candidates
            if u["token_span"][1] - u["token_span"][0] <= 24
            and u["fragment_of"] is None
            and (tuple(u["token_span"]), tuple(u["core_character_span"]))
            in legacy_spans
        ]
        long = [u for u in candidates if u["token_span"][1] - u["token_span"][0] > 24]
        control_kind = None
        if len(selected) == 1:
            control_kind = (
                "long_diagnostic"
                if response_id in long_control_responses
                else "unchanged_short"
            )
            pool = long if control_kind == "long_diagnostic" else short
            if not pool:
                raise ValueError(f"coarse v4 lacks required {control_kind} control")
            selected.append(
                sorted(pool, key=lambda u: (u["sequence_index"], u["unit_id"]))[0]
            )
        if len(selected) != FOCAL_UNITS_PER_WINDOW:
            raise ValueError("coarse v4 cannot fill deterministic compatibility window")
        selected.sort(key=lambda u: u["sequence_index"])
        windows.append(
            {
                "window_index": window_index,
                "response_id": response_id,
                "prompt_sha256": document["prompt_sha256"],
                "focal_unit_ids": [u["unit_id"] for u in selected],
                "old_defect_unit_ids": sorted(
                    item["unit_id"] for item in by_response[response_id]
                ),
                "old_defect_character_hull": [defect_start, defect_end],
                "repaired_unit_ids": [u["unit_id"] for u in repaired],
                "control_unit_id": None
                if control_kind is None
                else next(u["unit_id"] for u in selected if u not in repaired),
                "control_kind": control_kind,
            }
        )
        focal_units.extend(selected)
        audits.append(
            {
                "response_id": response_id,
                "old_defect_group_spans": [list(span) for span in old_spans],
                "old_defect_character_hull": [defect_start, defect_end],
                "repaired_spans": [u["core_character_span"] for u in repaired],
                "repair_unit_count": len(repaired),
                "selected_unit_ids": [u["unit_id"] for u in selected],
                "selected_width_tokens": [
                    u["token_span"][1] - u["token_span"][0] for u in selected
                ],
                "short_control_candidates": len(short),
                "long_control_candidates": len(long),
                "control_kind": control_kind,
            }
        )
    if (
        len(focal_units) != 24
        or len({u["unit_id"] for u in focal_units}) != 24
        or sum(window["control_kind"] == "unchanged_short" for window in windows) != 6
        or sum(window["control_kind"] == "long_diagnostic" for window in windows) != 4
    ):
        raise ValueError("coarse v4 focal unit cardinality drift")
    return (
        windows,
        focal_units,
        {"defect_rows": len(defect_rows), "response_audits": audits},
    )


def _request(
    *,
    physical_index: int,
    replica_index: int,
    repeat_of_request_id: str | None,
    window: Mapping[str, Any],
    document: Mapping[str, Any],
    focal: Sequence[Mapping[str, Any]],
    all_units: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    user_prompt, markup_audit = render_v4_user_prompt(document, focal, all_units)
    system_prompt = (
        _base_system_prompt(config)
        + "\n\nNo labeled demonstrations are provided in this arm."
    )
    provider = config["provider"]
    body = {
        "model": provider["model"],
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_output_tokens": provider["max_output_tokens"],
        "reasoning": provider["reasoning"],
        "prompt_cache_key": f"pwcv4-{canonical_sha256(system_prompt)[:16]}-{str(document['text_sha256'])[:32]}",
        "store": False,
        "text": {
            "format": {
                "type": "json_schema",
                "name": DECISION_SCHEMA_NAME,
                "schema": decision_json_schema_v4(),
                "strict": True,
            }
        },
    }
    body_sha = canonical_sha256(body)
    identity = {
        "schema_version": REQUEST_SCHEMA,
        "physical_index": physical_index,
        "arm_id": ARM_ZERO_SHOT,
        "replica_index": replica_index,
        "window_index": int(window["window_index"]),
        "repeat_of_request_id": repeat_of_request_id,
        "body_sha256": body_sha,
        "config_sha256": canonical_sha256(config),
        "focal_unit_ids": [u["unit_id"] for u in focal],
        "full_response_sha256": document["text_sha256"],
    }
    return {
        **identity,
        "request_id": f"pwcoarsequalv4-{canonical_sha256(identity)[:32]}",
        "response_id": window["response_id"],
        "prompt_sha256": window["prompt_sha256"],
        "markup_audit": markup_audit,
        "provider_body": body,
    }


def build_v4_qualification(
    *,
    workstation_bundle: Mapping[str, Any],
    review_root: Path,
    human_ledger_path: Path,
    correction_path: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    ledger_rows = read_jsonl(human_ledger_path)
    correction_rows = read_jsonl(correction_path)
    if len(ledger_rows) != 144 or any(
        row.get("globally_sealed") is not True for row in ledger_rows
    ):
        raise ValueError("coarse v4 requires the exact globally sealed v3 ledger")
    if (
        len(correction_rows) != 1
        or correction_rows[0].get("original_sealed_ledger_mutated") is not False
    ):
        raise ValueError("coarse v4 post-seal correction drift")
    windows, focal_units, repair_audit = _select_windows(
        workstation_bundle=workstation_bundle,
        review_root=review_root,
        ledger_rows=ledger_rows,
        correction_rows=correction_rows,
    )
    documents = {
        str(row["response_id"]): row for row in workstation_bundle["documents"]
    }
    focal_by_id = {u["unit_id"]: u for u in focal_units}
    all_by_response = {
        response_id: segment_document_v4(documents[response_id])
        for response_id in {w["response_id"] for w in windows}
    }
    requests: list[dict[str, Any]] = []
    bindings = []
    for window in windows:
        primary = None
        focal = [focal_by_id[unit_id] for unit_id in window["focal_unit_ids"]]
        for replica_index in range(REPLICAS_PER_WINDOW):
            request = _request(
                physical_index=len(requests),
                replica_index=replica_index,
                repeat_of_request_id=None if primary is None else primary["request_id"],
                window=window,
                document=documents[window["response_id"]],
                focal=focal,
                all_units=all_by_response[window["response_id"]],
                config=config,
            )
            if primary is None:
                primary = request
            elif request["provider_body"] != primary["provider_body"]:
                raise ValueError("coarse v4 replica body drift")
            requests.append(request)
            bindings.append(
                {
                    key: request[key]
                    for key in (
                        "request_id",
                        "arm_id",
                        "replica_index",
                        "window_index",
                        "repeat_of_request_id",
                        "focal_unit_ids",
                    )
                }
            )
    if (
        len(requests) != 36
        or sum(r["repeat_of_request_id"] is not None for r in requests) != 24
    ):
        raise ValueError("coarse v4 request cardinality drift")
    return {
        "windows": windows,
        "focal_units": focal_units,
        "requests": requests,
        "batch_lines": [openai_batch_line(r) for r in requests],
        "replica_bindings": bindings,
        "repair_audit": repair_audit,
    }


def openai_batch_line(request: Mapping[str, Any]) -> dict[str, Any]:
    if canonical_sha256(request["provider_body"]) != request.get("body_sha256"):
        raise ValueError("coarse v4 provider body hash drift")
    return {
        "custom_id": request["request_id"],
        "method": "POST",
        "url": OPENAI_BATCH_ENDPOINT,
        "body": request["provider_body"],
    }


def cost_plan_v4(
    requests: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    price_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    plan = cost_plan_v2(requests, config, price_snapshot)
    plan["schema_version"] = COST_PLAN_SCHEMA
    plan["campaign_shape"] = {
        "unique_windows": 12,
        "unique_units": 24,
        "arms": [ARM_ZERO_SHOT],
        "replicas_per_window": 3,
        "physical_requests": 36,
    }
    return plan


def forbidden_provider_input_leaks_v4(request: Mapping[str, Any]) -> list[str]:
    rendered = json.dumps(request["provider_body"]["input"], ensure_ascii=False)
    forbidden = (
        "machine_layers",
        "process_span",
        "discourse_phase",
        "cluster_id",
        "neuron_id",
        "merge_previous",
        "merge_next",
        "split_needed",
        "human_review",
    )
    return [value for value in forbidden if value in rendered]


def load_v4_qualification(root: Path) -> dict[str, Any]:
    """Validate the immutable offline v4 bundle before provider operations."""

    manifest = _load_object(root / "manifest.json")
    _verify_self_hash(manifest, "manifest_sha256", "coarse v4 manifest")
    expected_counts = {
        "arms": 1,
        "unique_windows": 12,
        "unique_focal_units": 24,
        "physical_requests": 36,
        "replica_requests": 24,
    }
    if (
        manifest.get("schema_version") != BUNDLE_SCHEMA
        or manifest.get("status") != "prepared_offline_no_provider_calls"
        or manifest.get("network_calls_made") != 0
        or manifest.get("v3_artifacts_mutated") is not False
        or manifest.get("counts") != expected_counts
    ):
        raise ValueError("coarse v4 bundle is not an exact offline artifact")
    required = {
        "batch-input.jsonl",
        "cost-plan.json",
        "focal-units.jsonl",
        "price-snapshot.json",
        "protocol-config.json",
        "replica-bindings.json",
        "requests.jsonl",
        "segmentation-repair-audit.json",
        "v3-human-ledger.jsonl",
        "v3-post-seal-corrections.jsonl",
        "windows.json",
    }
    if {binding["path"] for binding in manifest.get("files", [])} != required:
        raise ValueError("coarse v4 payload membership drift")
    for binding in manifest["files"]:
        path = root / binding["path"]
        if (
            not path.is_file()
            or path.stat().st_size != binding["bytes"]
            or file_sha256(path) != binding["sha256"]
        ):
            raise ValueError(f"coarse v4 payload drift: {path}")
    config = load_coarse_v4_config(root / "protocol-config.json")
    if (
        file_sha256(root / "protocol-config.json") != manifest.get("config_sha256")
        or manifest.get("config_id") != config["config_id"]
        or manifest.get("source_workstation_bundle_sha256")
        != config["source"]["workstation_bundle_sha256"]
        or manifest.get("source_v3_qualification_manifest_sha256")
        != config["source"]["v3_qualification_manifest_sha256"]
        or manifest.get("source_v3_review_packet_manifest_sha256")
        != config["source"]["v3_review_packet_manifest_sha256"]
        or file_sha256(root / "v3-human-ledger.jsonl")
        != config["source"]["v3_human_ledger_sha256"]
        or file_sha256(root / "v3-post-seal-corrections.jsonl")
        != config["source"]["v3_post_seal_corrections_sha256"]
    ):
        raise ValueError("coarse v4 source/config binding drift")
    requests = read_jsonl(root / "requests.jsonl")
    lines = read_jsonl(root / "batch-input.jsonl")
    units = read_jsonl(root / "focal-units.jsonl")
    windows = json.loads((root / "windows.json").read_text(encoding="utf-8"))
    bindings = json.loads((root / "replica-bindings.json").read_text(encoding="utf-8"))
    repair_audit = _load_object(root / "segmentation-repair-audit.json")
    cost_plan = _load_object(root / "cost-plan.json")
    _verify_self_hash(cost_plan, "cost_plan_sha256", "coarse v4 cost plan")
    if not (
        len(requests) == len(lines) == len(bindings) == 36
        and len(units) == 24
        and len(windows) == 12
        and len({r["request_id"] for r in requests}) == 36
        and [r["physical_index"] for r in requests] == list(range(36))
        and len({u["unit_id"] for u in units}) == 24
        and repair_audit.get("defect_rows") == 24
    ):
        raise ValueError("coarse v4 payload cardinality drift")
    units_by_id = {unit["unit_id"]: unit for unit in units}
    windows_by_index = {window["window_index"]: window for window in windows}
    if set(windows_by_index) != set(range(12)):
        raise ValueError("coarse v4 window index drift")
    for index, window in enumerate(windows):
        if (
            window["window_index"] != index
            or len(window["focal_unit_ids"]) != 2
            or any(
                units_by_id[unit_id]["response_id"] != window["response_id"]
                for unit_id in window["focal_unit_ids"]
            )
        ):
            raise ValueError("coarse v4 window binding drift")
        group = sorted(
            (r for r in requests if r["window_index"] == index),
            key=lambda r: r["replica_index"],
        )
        if (
            len(group) != 3
            or [r["replica_index"] for r in group] != [0, 1, 2]
            or any(r["arm_id"] != ARM_ZERO_SHOT for r in group)
            or group[0]["repeat_of_request_id"] is not None
            or any(
                r["repeat_of_request_id"] != group[0]["request_id"] for r in group[1:]
            )
            or any(r["provider_body"] != group[0]["provider_body"] for r in group[1:])
        ):
            raise ValueError("coarse v4 exact replica topology drift")
    if bindings != [
        {
            key: r[key]
            for key in (
                "request_id",
                "arm_id",
                "replica_index",
                "window_index",
                "repeat_of_request_id",
                "focal_unit_ids",
            )
        }
        for r in requests
    ]:
        raise ValueError("coarse v4 replica binding drift")
    for request, line in zip(requests, lines, strict=True):
        if line != openai_batch_line(request) or forbidden_provider_input_leaks_v4(
            request
        ):
            raise ValueError("coarse v4 provider request drift")
    if cost_plan.get("request_count") != 36 or cost_plan.get("campaign_shape") != {
        "unique_windows": 12,
        "unique_units": 24,
        "arms": [ARM_ZERO_SHOT],
        "replicas_per_window": 3,
        "physical_requests": 36,
    }:
        raise ValueError("coarse v4 cost plan drift")
    return {
        "manifest": manifest,
        "config": config,
        "requests": requests,
        "batch_lines": lines,
        "focal_units": units,
        "windows": windows,
        "replica_bindings": bindings,
        "repair_audit": repair_audit,
        "cost_plan": cost_plan,
    }
