"""Graph-blind annotation suggestions for process-witness responses.

This module deliberately knows nothing about attribution graphs, neurons, clusters, or
labels.  It turns frozen response text into observable-form annotations and reviewable
semantic suggestions.  Suggestions are hypotheses, never accepted ground truth.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "adag.process-witness.annotation-draft.v1"
INVENTORY_SCHEMA_VERSION = "adag.process-witness.annotation-inventory.v1"
AUDIT_SCHEMA_VERSION = "adag.process-witness.annotation-audit.v1"
WORKSTATION_BUNDLE_SCHEMA_VERSION = (
    "adag.process-witness.annotation-workstation-bundle.v1"
)


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


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Match:
    start: int
    end: int
    axis: str
    value: str
    rule_id: str
    interpretation: str
    confidence: str
    ambiguity: tuple[str, ...] = ()


def load_ontology(path: Path) -> dict[str, Any]:
    ontology = json.loads(path.read_text(encoding="utf-8"))
    if ontology.get("schema_version") not in {
        "adag.process-witness.annotation-ontology.v1",
        "adag.process-witness.annotation-ontology.v2",
    }:
        raise ValueError("unsupported annotation ontology schema")
    axes = ontology.get("axes")
    if not isinstance(axes, dict) or not axes:
        raise ValueError("ontology axes must be a non-empty object")
    if ontology["schema_version"].endswith(".v2"):
        contract = ontology.get("token_assignment_contract", {})
        if contract.get("within_axis") != (
            "zero_or_one_effective_value_per_authoritative_response_token"
        ):
            raise ValueError("v2 ontology lacks the within-axis token contract")
        if "compound_surface" not in axes.get("surface_form", {}).get("values", []):
            raise ValueError("v2 ontology lacks compound_surface")
    rule_ids: list[str] = []
    for collection in ("regex_rules", "lexical_rules", "detectors"):
        for rule in ontology.get(collection, []):
            rule_ids.append(rule["rule_id"])
            if rule["axis"] not in axes:
                raise ValueError(f"rule {rule['rule_id']} uses unknown axis")
            value = rule.get("value")
            if value is not None and value not in axes[rule["axis"]]["values"]:
                raise ValueError(f"rule {rule['rule_id']} uses unknown value {value!r}")
    if len(rule_ids) != len(set(rule_ids)):
        raise ValueError("ontology rule IDs must be unique")
    return ontology


def _regex_matches(text: str, rule: Mapping[str, Any]) -> Iterator[Match]:
    flags = re.IGNORECASE if rule.get("ignore_case", False) else 0
    pattern = re.compile(rule["pattern"], flags)
    for found in pattern.finditer(text):
        group = rule.get("capture_group", 0)
        start, end = found.span(group)
        if start == end:
            continue
        yield Match(
            start=start,
            end=end,
            axis=rule["axis"],
            value=rule["value"],
            rule_id=rule["rule_id"],
            interpretation=rule["interpretation"],
            confidence=rule["confidence"],
            ambiguity=tuple(rule.get("ambiguity", [])),
        )


def _lexical_matches(text: str, rule: Mapping[str, Any]) -> Iterator[Match]:
    terms = sorted(rule["terms"], key=lambda term: (-len(term), term.casefold()))
    pattern = re.compile(
        r"(?<![\w])(?:" + "|".join(re.escape(term) for term in terms) + r")(?![\w])",
        re.IGNORECASE,
    )
    for found in pattern.finditer(text):
        yield Match(
            start=found.start(),
            end=found.end(),
            axis=rule["axis"],
            value=rule["value"],
            rule_id=rule["rule_id"],
            interpretation=rule["interpretation"],
            confidence=rule["confidence"],
            ambiguity=tuple(rule.get("ambiguity", [])),
        )


_ARITHMETIC_ATOM_LEFT = re.compile(r"(?:\d+(?:\.\d+)?|[A-Za-z_][\w]*|\))[ \t]*$")
_ARITHMETIC_ATOM_RIGHT = re.compile(r"^[ \t]*(?:-?\d+(?:\.\d+)?|[A-Za-z_][\w]*|\()")


def _arithmetic_symbol_matches(text: str, rule: Mapping[str, Any]) -> Iterator[Match]:
    """Match operation symbols only when both sides look like arithmetic atoms."""

    for symbol, value in rule["symbols"].items():
        for found in re.finditer(re.escape(symbol), text):
            if (
                symbol == "%"
                and found.start() > 0
                and text[found.start() - 1].isdigit()
                and (
                    found.end() == len(text)
                    or text[found.end()].isspace()
                    or text[found.end()] in ".,;:!?"
                )
            ):
                # A tight numeric percentage such as "100% sure" is not modulo.
                # Explicit `100 % n` and tight `100%2` remain eligible.
                continue
            if symbol == "*" and (
                (found.start() > 0 and text[found.start() - 1] == "*")
                or (found.end() < len(text) and text[found.end()] == "*")
            ):
                # Markdown emphasis is not multiplication.
                continue
            if symbol == "-" and (
                found.start() == 0 or text[found.start() - 1] in "\r\n"
            ):
                # Markdown/plain-text list markers are not subtraction.
                continue
            if (
                symbol == "-"
                and found.start() > 0
                and found.end() < len(text)
                and text[found.start() - 1].isalpha()
                and text[found.end()].isalpha()
            ):
                # A tight alphabetic-alpha hyphen ("code-block") is much more
                # likely morphology than subtraction.  Spaced variables and any
                # numeric operand remain eligible.
                continue
            left = text[max(0, found.start() - 48) : found.start()]
            right = text[found.end() : min(len(text), found.end() + 48)]
            if not _ARITHMETIC_ATOM_LEFT.search(left):
                continue
            if not _ARITHMETIC_ATOM_RIGHT.search(right):
                continue
            yield Match(
                start=found.start(),
                end=found.end(),
                axis=rule["axis"],
                value=value,
                rule_id=f"{rule['rule_id']}.{value}",
                interpretation=rule["interpretation"],
                confidence=rule["confidence"],
                ambiguity=tuple(rule.get("ambiguity", [])),
            )


def _json_answer_matches(
    text: str,
    rule: Mapping[str, Any],
    accepted_answer_keys: set[str] | None,
) -> Iterator[Match]:
    """Mark scalar fields in the last valid one-line JSON object in the response."""

    candidates: list[tuple[int, str, dict[str, Any]]] = []
    for line_match in re.finditer(r"(?m)^[ \t]*(\{[^\r\n]*\})[ \t]*$", text):
        try:
            parsed = json.loads(line_match.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            candidates.append((line_match.start(1), line_match.group(1), parsed))
    candidates = [
        candidate
        for candidate in candidates
        if re.fullmatch(
            r"\s*(?:</think>\s*)?",
            text[candidate[0] + len(candidate[1]) :],
            re.IGNORECASE,
        )
    ]
    if not candidates:
        return
    object_start, object_text, parsed = candidates[-1]
    allowed_keys = accepted_answer_keys or set(parsed)
    pattern = re.compile(
        r'"(?P<key>(?:\\.|[^"\\])+)"\s*:\s*'
        r'(?P<value>"(?:\\.|[^"\\])*"|-?\d+(?:\.\d+)?|true|false|null)'
    )
    for found in pattern.finditer(object_text):
        try:
            key = json.loads('"' + found.group("key") + '"')
        except json.JSONDecodeError:
            continue
        if key not in allowed_keys or key not in parsed:
            continue
        for group, value in (("key", "answer_key"), ("value", "answer_value")):
            local_start, local_end = found.span(group)
            start, end = object_start + local_start, object_start + local_end
            yield Match(
                start=start,
                end=end,
                axis=rule["axis"],
                value=value,
                rule_id=f"{rule['rule_id']}.{value}",
                interpretation=rule["interpretation"],
                confidence=rule["confidence"],
            )
        local_start, local_end = found.span("value")
        start, end = object_start + local_start, object_start + local_end
        yield Match(
            start=start,
            end=end,
            axis="process_role",
            value="answer_commitment",
            rule_id=f"{rule['rule_id']}.answer_commitment",
            interpretation="semantic_hypothesis",
            confidence="high",
            ambiguity=("syntax identifies a committed answer, not its correctness",),
        )


def _numeric_relation_matches(text: str, rule: Mapping[str, Any]) -> Iterator[Match]:
    # The capture is intentionally restricted to a numeric RHS.  It may still be an
    # assignment or quoted premise, hence hypothesis/medium rather than truth/high.
    pattern = re.compile(r"(?:=|→|->)\s*(?P<result>-?\d+(?:\.\d+)?)")
    for found in pattern.finditer(text):
        start, end = found.span("result")
        yield Match(
            start=start,
            end=end,
            axis=rule["axis"],
            value=rule["value"],
            rule_id=rule["rule_id"],
            interpretation=rule["interpretation"],
            confidence=rule["confidence"],
            ambiguity=tuple(rule.get("ambiguity", [])),
        )


DETECTORS = {
    "arithmetic_symbols": _arithmetic_symbol_matches,
    "numeric_relation_results": _numeric_relation_matches,
}


def suggest_matches(
    text: str,
    ontology: Mapping[str, Any],
    *,
    accepted_answer_keys: set[str] | None = None,
) -> list[Match]:
    matches: list[Match] = []
    for rule in ontology.get("regex_rules", []):
        matches.extend(_regex_matches(text, rule))
    for rule in ontology.get("lexical_rules", []):
        matches.extend(_lexical_matches(text, rule))
    for rule in ontology.get("detectors", []):
        if rule["detector"] == "json_answers":
            matches.extend(_json_answer_matches(text, rule, accepted_answer_keys))
            continue
        try:
            detector = DETECTORS[rule["detector"]]
        except KeyError as error:
            raise ValueError(f"unknown detector {rule['detector']}") from error
        matches.extend(detector(text, rule))
    unique = {
        (
            match.start,
            match.end,
            match.axis,
            match.value,
            match.rule_id,
        ): match
        for match in matches
    }
    return sorted(
        unique.values(),
        key=lambda match: (
            match.start,
            match.end,
            match.axis,
            match.value,
            match.rule_id,
        ),
    )


def _encoded_ids_and_offsets(
    tokenizer: Any, text: str
) -> tuple[list[int], list[list[int]]]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_attention_mask=False,
    )
    ids = [int(value) for value in encoded["input_ids"]]
    offsets = [[int(start), int(end)] for start, end in encoded["offset_mapping"]]
    if len(ids) != len(offsets):
        raise ValueError("token IDs and offsets differ in length")
    previous_start = 0
    for index, (start, end) in enumerate(offsets):
        if not 0 <= start <= end <= len(text):
            raise ValueError(f"invalid offset at token {index}: {(start, end)}")
        if start < previous_start:
            raise ValueError(f"non-monotonic offset at token {index}")
        previous_start = start
    return ids, offsets


def continuation_token_offsets(
    tokenizer: Any,
    *,
    prefix_text: str,
    response_text: str,
    expected_prefix_ids: Sequence[int],
    expected_response_ids: Sequence[int] | None = None,
    expected_response_ids_sha256: str | None = None,
) -> tuple[list[int], list[list[int]]]:
    """Tokenize an exact generation continuation and fail closed on identity drift.

    Standalone response tokenization is not authoritative because a tokenizer may
    merge at the assistant-prefix boundary.  The combined rendered prefix and response
    is tokenized once, then checked against captured generation IDs (or the frozen hash
    for reconstructed historical responses).
    """

    prefix_ids, _ = _encoded_ids_and_offsets(tokenizer, prefix_text)
    expected_prefix = [int(value) for value in expected_prefix_ids]
    if prefix_ids != expected_prefix:
        raise ValueError("rendered assistant prefix IDs differ from authoritative IDs")
    combined_ids, combined_offsets = _encoded_ids_and_offsets(
        tokenizer, prefix_text + response_text
    )
    if combined_ids[: len(prefix_ids)] != prefix_ids:
        raise ValueError("response merges across the assistant-prefix token boundary")
    response_ids = combined_ids[len(prefix_ids) :]
    if expected_response_ids is not None:
        expected = [int(value) for value in expected_response_ids]
        if response_ids != expected:
            raise ValueError(
                "continuation response IDs differ from captured generation IDs"
            )
    if (
        expected_response_ids_sha256 is not None
        and canonical_sha256(response_ids) != expected_response_ids_sha256
    ):
        raise ValueError("continuation response IDs differ from frozen response hash")
    prefix_char_count = len(prefix_text)
    response_offsets = []
    for index, (start, end) in enumerate(combined_offsets[len(prefix_ids) :]):
        if start < prefix_char_count:
            raise ValueError(f"response token {index} overlaps the rendered prefix")
        response_offsets.append([start - prefix_char_count, end - prefix_char_count])
    if not response_ids or len(response_ids) != len(response_offsets):
        raise ValueError("response IDs and continuation offsets differ or are empty")
    if response_offsets[-1][1] != len(response_text):
        raise ValueError("continuation offsets do not reach the end of the response")
    return response_ids, response_offsets


def _bytes_to_unicode() -> dict[int, str]:
    """Return the reversible GPT-2 byte-level alphabet used by Qwen's tokenizer."""

    byte_values = list(range(ord("!"), ord("~") + 1))
    byte_values += list(range(ord("¡"), ord("¬") + 1))
    byte_values += list(range(ord("®"), ord("ÿ") + 1))
    codepoints = list(byte_values)
    extra = 0
    for value in range(256):
        if value not in byte_values:
            byte_values.append(value)
            codepoints.append(256 + extra)
            extra += 1
    return dict(zip(byte_values, (chr(value) for value in codepoints), strict=True))


def captured_byte_level_token_offsets(
    tokenizer: Any, *, text: str, token_ids: Sequence[int]
) -> list[list[int]]:
    """Align an authoritative byte-level token sequence to its decoded text.

    Generated BPE segmentations are not necessarily reproduced by encoding the decoded
    string.  This routine therefore maps the captured vocabulary pieces back to bytes,
    verifies those bytes are exactly the frozen UTF-8 text, and records a covering
    character span for each captured token.  Two byte-fragment tokens may cover the
    same Unicode character when a UTF-8 codepoint is split across token boundaries.
    """

    ids = [int(value) for value in token_ids]
    decoded = tokenizer.decode(
        ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if decoded != text:
        raise ValueError("captured token IDs do not decode to the frozen response text")
    byte_decoder = {value: key for key, value in _bytes_to_unicode().items()}
    pieces = tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=False)
    if len(pieces) != len(ids):
        raise ValueError(
            "tokenizer did not return one vocabulary piece per captured ID"
        )
    token_bytes: list[bytes] = []
    for index, piece in enumerate(pieces):
        try:
            token_bytes.append(bytes(byte_decoder[character] for character in piece))
        except KeyError as error:
            raise ValueError(
                f"token {index} is not representable in the byte-level alphabet"
            ) from error
    raw_bytes = text.encode("utf-8")
    if b"".join(token_bytes) != raw_bytes:
        raise ValueError("captured token pieces do not reconstruct frozen UTF-8 bytes")

    byte_owner: list[int] = []
    for char_index, character in enumerate(text):
        byte_owner.extend([char_index] * len(character.encode("utf-8")))
    offsets: list[list[int]] = []
    byte_position = 0
    for piece in token_bytes:
        start_byte = byte_position
        end_byte = start_byte + len(piece)
        if not piece:
            raise ValueError("captured response contains a zero-byte token")
        offsets.append([byte_owner[start_byte], byte_owner[end_byte - 1] + 1])
        byte_position = end_byte
    return offsets


def align_span(
    offsets: Sequence[Sequence[int]], start: int, end: int
) -> dict[str, Any]:
    overlapping = [
        index
        for index, (token_start, token_end) in enumerate(offsets)
        if token_end > start and token_start < end
    ]
    if not overlapping:
        raise ValueError(f"character span {(start, end)} overlaps no tokens")
    token_start = overlapping[0]
    token_end = overlapping[-1] + 1
    covering_start = int(offsets[token_start][0])
    covering_end = int(offsets[token_end - 1][1])
    return {
        "start": token_start,
        "end": token_end,
        "boundary_alignment": (
            "exact" if (covering_start, covering_end) == (start, end) else "covering"
        ),
        "covering_character_span": [covering_start, covering_end],
    }


def _token_surface(text: str) -> str:
    if not text:
        return "empty"
    if text.isspace():
        if "\n" in text or "\r" in text:
            return "newline_whitespace"
        return "horizontal_whitespace"
    if text.isdecimal():
        return "integer_digits"
    if all(character.isalpha() for character in text):
        return "alphabetic"
    if all(not character.isalnum() and not character.isspace() for character in text):
        return "symbol_or_punctuation"
    if any(character.isdigit() for character in text):
        return "mixed_with_digit"
    return "mixed"


def task_context(response: Mapping[str, Any]) -> dict[str, Any]:
    """Recover prompt and task metadata without consulting model internals."""

    generation = response.get("generation_row")
    if generation is not None:
        prompt = generation["prompt"]
        context = {
            "prompt": prompt,
            "prompt_sha256": text_sha256(prompt),
            "source_types": sorted(
                str(value)
                for value in json.loads(generation.get("src_types_json", "[]"))
            ),
            "question_ids": sorted(
                str(value)
                for value in json.loads(generation.get("question_ids_json", "[]"))
            ),
            "task_family": "unknown_unreviewed",
            "historical_process_role": None,
        }
    else:
        historical = response["historical_dense_record"]
        prompt = historical["prompt"]
        context = {
            "prompt": prompt,
            "prompt_sha256": text_sha256(prompt),
            "source_types": [str(historical["source_type"])],
            "question_ids": [str(historical["question_id"])],
            "task_family": str(historical["process_family"]),
            "historical_process_role": str(historical["role"]),
            "historical_annotation_ids": sorted(
                str(value) for value in historical["annotation_ids"]
            ),
        }
    if context["prompt_sha256"] != response["prompt_sha256"]:
        raise ValueError(f"prompt hash drift for {response['response_id']}")
    return context


def annotate_response(
    *,
    response: Mapping[str, Any],
    text: str,
    ids: Sequence[int],
    offsets: Sequence[Sequence[int]],
    token_identity: Mapping[str, Any],
    ontology: Mapping[str, Any],
    ontology_sha256: str,
    cohort_id: str,
    annotation_set_id: str,
) -> dict[str, Any]:
    ids = [int(value) for value in ids]
    offsets = [[int(start), int(end)] for start, end in offsets]
    tokens = [
        {
            "position": position,
            "token_id": token_id,
            "character_span": offset,
            "text": text[offset[0] : offset[1]],
            "surface": _token_surface(text[offset[0] : offset[1]]),
        }
        for position, (token_id, offset) in enumerate(zip(ids, offsets, strict=True))
    ]
    accepted_answer_keys: set[str] = set()
    generation_row = response.get("generation_row")
    if generation_row is not None:
        schemas_json = generation_row.get("accepted_answer_schemas_json", "[]")
        for schema in json.loads(schemas_json):
            accepted_answer_keys.update(
                str(key) for key in schema.get("exact_keys", [])
            )
    suggestions = []
    for match in suggest_matches(
        text,
        ontology,
        accepted_answer_keys=accepted_answer_keys or None,
    ):
        if match.axis not in ontology["axes"]:
            raise ValueError(f"suggestion uses unknown axis {match.axis!r}")
        if match.value not in ontology["axes"][match.axis]["values"]:
            raise ValueError(
                f"suggestion uses unknown {match.axis} value {match.value!r}"
            )
        identity = {
            "response_id": response["response_id"],
            "character_span": [match.start, match.end],
            "axis": match.axis,
            "value": match.value,
            "rule_id": match.rule_id,
            "ontology_sha256": ontology_sha256,
        }
        suggestions.append(
            {
                "suggestion_id": "pwsuggestion-" + canonical_sha256(identity)[:24],
                "character_span": [match.start, match.end],
                "token_span": align_span(offsets, match.start, match.end),
                "text": text[match.start : match.end],
                "axis": match.axis,
                "value": match.value,
                "interpretation": match.interpretation,
                "confidence": match.confidence,
                "ambiguity": list(match.ambiguity),
                "status": "suggested_unreviewed",
                "provenance": {
                    "method": "deterministic_rule",
                    "rule_id": match.rule_id,
                    "ontology_sha256": ontology_sha256,
                },
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "annotation_status": "automatic_suggestions_unreviewed",
        "claim_boundary": (
            "Observable-form annotations describe text; semantic annotations are "
            "reviewable lexical or local-pattern hypotheses, not process truth."
        ),
        "cohort_id": cohort_id,
        "annotation_set_id": annotation_set_id,
        "coordinate_system": {
            "character_span_unit": "Unicode_code_point",
            "character_span_interval": "zero_based_half_open",
            "token_span_interval": "zero_based_half_open",
        },
        "ontology": {
            "ontology_id": ontology["ontology_id"],
            "sha256": ontology_sha256,
            "axes": ontology["axes"],
            "review_vocabularies": ontology["review_vocabularies"],
        },
        "response_id": response["response_id"],
        "response_source": response["source"],
        "trace_scope": response["trace_scope"],
        "prompt_sha256": response["prompt_sha256"],
        "task_context": task_context(response),
        "text_sha256": text_sha256(text),
        "text": text,
        "tokenization": {
            "identity_status": "verified_against_authoritative_continuation",
            "identity": dict(token_identity),
            "token_count": len(ids),
            "input_ids_sha256": canonical_sha256(ids),
            "offset_mapping_sha256": canonical_sha256(offsets),
            "tokens": tokens,
        },
        "suggestions": suggestions,
        "process_events": {
            "schema_version": "adag.process-witness.process-event.v1",
            "status": "empty_pending_human_annotation",
            "events": [],
        },
    }


def compact_workstation_document(
    document: Mapping[str, Any], *, source_record_sha256: str
) -> dict[str, Any]:
    """Project a verbose draft record into the graph-blind painting workstation.

    The frozen draft remains authoritative.  This projection shares ontology metadata
    at bundle level, stores exact token identities compactly, and run-length encodes the
    effective machine layer.  Under ontology v2, overlapping surface-form values map to
    ``compound_surface``; semantic conflicts remain sorted lists rather than receiving a
    silent winner.
    """

    tokens = document["tokenization"]["tokens"]
    axis_values: dict[str, list[set[str]]] = {
        axis: [set() for _ in tokens] for axis in document["ontology"]["axes"]
    }
    for suggestion in document["suggestions"]:
        start = int(suggestion["token_span"]["start"])
        end = int(suggestion["token_span"]["end"])
        for position in range(start, end):
            axis_values[suggestion["axis"]][position].add(suggestion["value"])

    machine_layers: dict[str, list[list[Any]]] = {}
    for axis, values_by_token in axis_values.items():
        runs: list[list[Any]] = []
        run_start: int | None = None
        run_value: str | list[str] | None = None
        for position in range(len(values_by_token) + 1):
            values = (
                values_by_token[position] if position < len(values_by_token) else set()
            )
            value: str | list[str] | None
            if len(values) == 1:
                value = next(iter(values))
            elif values:
                if (
                    axis == "surface_form"
                    and "compound_surface"
                    in document["ontology"]["axes"][axis]["values"]
                ):
                    value = "compound_surface"
                else:
                    value = sorted(values)
            else:
                value = None
            if value == run_value:
                continue
            if run_value is not None and run_start is not None:
                runs.append([run_start, position, run_value])
            run_start = position if value is not None else None
            run_value = value
        if runs:
            machine_layers[axis] = runs

    return {
        "response_id": document["response_id"],
        "response_source": document["response_source"],
        "trace_scope": document["trace_scope"],
        "prompt_sha256": document["prompt_sha256"],
        "task_context": document["task_context"],
        "text_sha256": document["text_sha256"],
        "text": document["text"],
        "source_annotation_record_sha256": source_record_sha256,
        "tokenization": {
            "identity_status": document["tokenization"]["identity_status"],
            "token_count": document["tokenization"]["token_count"],
            "input_ids_sha256": document["tokenization"]["input_ids_sha256"],
            "offset_mapping_sha256": document["tokenization"]["offset_mapping_sha256"],
            "tokens": [
                [
                    int(token["token_id"]),
                    int(token["character_span"][0]),
                    int(token["character_span"][1]),
                ]
                for token in tokens
            ],
        },
        "machine_layers": machine_layers,
    }


def build_workstation_bundle(
    documents: Sequence[Mapping[str, Any]],
    *,
    source_record_sha256s: Sequence[str],
    review_ui_version: str,
    review_ui_sha256: str,
) -> dict[str, Any]:
    """Create one compact, self-identifying import for the painting workstation."""

    if not documents:
        raise ValueError("workstation bundle requires at least one document")
    if len(documents) != len(source_record_sha256s):
        raise ValueError("document/source-hash count mismatch")
    if not review_ui_version:
        raise ValueError("workstation bundle requires a review UI version")
    if not re.fullmatch(r"[0-9a-f]{64}", review_ui_sha256):
        raise ValueError("workstation bundle requires an exact review UI SHA-256")
    first = documents[0]
    identity = (
        first["annotation_set_id"],
        first["cohort_id"],
        first["ontology"]["sha256"],
    )
    for document in documents[1:]:
        if (
            document["annotation_set_id"],
            document["cohort_id"],
            document["ontology"]["sha256"],
        ) != identity:
            raise ValueError("mixed annotation identities in workstation bundle")
    return {
        "schema_version": WORKSTATION_BUNDLE_SCHEMA_VERSION,
        "status": "automatic_suggestions_unreviewed",
        "claim_boundary": first["claim_boundary"],
        "annotation_set_id": first["annotation_set_id"],
        "cohort_id": first["cohort_id"],
        "coordinate_system": first["coordinate_system"],
        "review_ui": {
            "version": review_ui_version,
            "sha256": review_ui_sha256,
        },
        "ontology": first["ontology"],
        "responses": len(documents),
        "documents": [
            compact_workstation_document(document, source_record_sha256=source_sha256)
            for document, source_sha256 in zip(
                documents, source_record_sha256s, strict=True
            )
        ],
    }


def validate_annotation(document: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    text = document["text"]
    tokens = document["tokenization"]["tokens"]
    if document["text_sha256"] != text_sha256(text):
        errors.append("text hash mismatch")
    task = document.get("task_context", {})
    if task.get("prompt_sha256") != text_sha256(task.get("prompt", "")):
        errors.append("task prompt hash mismatch")
    if task.get("prompt_sha256") != document.get("prompt_sha256"):
        errors.append("task/document prompt hash mismatch")
    process_events = document.get("process_events", {})
    if process_events.get(
        "schema_version"
    ) != "adag.process-witness.process-event.v1" or not isinstance(
        process_events.get("events"), list
    ):
        errors.append("invalid process-event placeholder")
    seen: set[str] = set()
    for suggestion in document["suggestions"]:
        suggestion_id = suggestion["suggestion_id"]
        if suggestion_id in seen:
            errors.append(f"duplicate suggestion ID {suggestion_id}")
        seen.add(suggestion_id)
        axes = document["ontology"]["axes"]
        if suggestion["axis"] not in axes:
            errors.append(f"unknown suggestion axis {suggestion_id}")
        elif suggestion["value"] not in axes[suggestion["axis"]]["values"]:
            errors.append(f"unknown suggestion value {suggestion_id}")
        start, end = suggestion["character_span"]
        if not 0 <= start < end <= len(text):
            errors.append(f"invalid span {suggestion_id}")
            continue
        if suggestion["text"] != text[start:end]:
            errors.append(f"span text mismatch {suggestion_id}")
        if (
            suggestion["value"] == "sentence_terminal"
            and suggestion["text"] == "."
            and start > 0
            and end < len(text)
            and text[start - 1].isdigit()
            and text[end].isdigit()
        ):
            errors.append(f"decimal dot marked as sentence terminal {suggestion_id}")
        if suggestion["value"] == "quote" and suggestion["text"] == "`":
            errors.append(f"backtick marked as quote {suggestion_id}")
        token_start = suggestion["token_span"]["start"]
        token_end = suggestion["token_span"]["end"]
        if not 0 <= token_start < token_end <= len(tokens):
            errors.append(f"invalid token span {suggestion_id}")
            continue
        covered_start = tokens[token_start]["character_span"][0]
        covered_end = tokens[token_end - 1]["character_span"][1]
        if covered_start > start or covered_end < end:
            errors.append(f"token span does not cover characters {suggestion_id}")
    return errors


def build_inventory(documents: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    response_count = 0
    token_count = 0
    suggestion_count = 0
    exact_alignment_count = 0
    by_axis: Counter[str] = Counter()
    by_value: Counter[tuple[str, str]] = Counter()
    by_rule: Counter[str] = Counter()
    response_support: dict[tuple[str, str], set[str]] = defaultdict(set)
    prompt_support: dict[tuple[str, str], set[str]] = defaultdict(set)
    for document in documents:
        response_count += 1
        token_count += document["tokenization"]["token_count"]
        for suggestion in document["suggestions"]:
            suggestion_count += 1
            exact_alignment_count += (
                suggestion["token_span"]["boundary_alignment"] == "exact"
            )
            axis_value = (suggestion["axis"], suggestion["value"])
            by_axis[suggestion["axis"]] += 1
            by_value[axis_value] += 1
            by_rule[suggestion["provenance"]["rule_id"]] += 1
            response_support[axis_value].add(document["response_id"])
            prompt_support[axis_value].add(document["prompt_sha256"])
    values = [
        {
            "axis": axis,
            "value": value,
            "suggestions": count,
            "responses": len(response_support[(axis, value)]),
            "prompts": len(prompt_support[(axis, value)]),
        }
        for (axis, value), count in sorted(by_value.items())
    ]
    return {
        "schema_version": INVENTORY_SCHEMA_VERSION,
        "annotation_status": "automatic_suggestions_unreviewed",
        "responses": response_count,
        "tokens": token_count,
        "suggestions": suggestion_count,
        "exact_token_boundary_suggestions": exact_alignment_count,
        "covering_token_boundary_suggestions": suggestion_count - exact_alignment_count,
        "by_axis": dict(sorted(by_axis.items())),
        "values": values,
        "by_rule": dict(sorted(by_rule.items())),
    }


def inspection_examples(
    documents: Iterable[Mapping[str, Any]], *, context_characters: int = 72
) -> list[dict[str, Any]]:
    """Select deterministic boundary and lexical examples for every rule."""

    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_text: dict[str, set[str]] = defaultdict(set)
    for document in documents:
        text = document["text"]
        for suggestion in document["suggestions"]:
            rule_id = suggestion["provenance"]["rule_id"]
            if len(examples[rule_id]) >= 7:
                continue
            matched = suggestion["text"].casefold()
            if matched in seen_text[rule_id] and len(examples[rule_id]) >= 3:
                continue
            seen_text[rule_id].add(matched)
            start, end = suggestion["character_span"]
            examples[rule_id].append(
                {
                    "response_id": document["response_id"],
                    "suggestion_id": suggestion["suggestion_id"],
                    "axis": suggestion["axis"],
                    "value": suggestion["value"],
                    "match": suggestion["text"],
                    "context": text[
                        max(0, start - context_characters) : min(
                            len(text), end + context_characters
                        )
                    ],
                }
            )
    return [
        {"rule_id": rule_id, "examples": examples[rule_id]}
        for rule_id in sorted(examples)
    ]


def audit_documents(documents: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    for document in documents:
        errors.extend(
            {"response_id": document["response_id"], "error": error}
            for error in validate_annotation(document)
        )
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "status": "passed" if not errors else "failed",
        "responses_checked": len(documents),
        "errors": errors,
        "checks": [
            "text hash matches embedded response text",
            "suggestion IDs are unique within each response",
            "character spans are non-empty and in bounds",
            "stored match text equals the exact response slice",
            "token spans are in bounds and cover each character span",
        ],
    }
