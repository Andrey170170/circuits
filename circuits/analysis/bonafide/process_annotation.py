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
from copy import deepcopy
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
    """Load an ontology, resolving a hash-pinned local extension when requested."""

    ontology = json.loads(path.read_text(encoding="utf-8"))
    if ontology.get("schema_version") in {
        "adag.process-witness.annotation-ontology.v3",
        "adag.process-witness.annotation-ontology.v4",
        "adag.process-witness.annotation-ontology.v5",
        "adag.process-witness.annotation-ontology.v6",
    }:
        base_name = ontology.get("extends")
        base_sha256 = ontology.get("base_ontology_sha256")
        if not isinstance(base_name, str) or Path(base_name).name != base_name:
            raise ValueError("extended ontology base must be a sibling file name")
        base_path = path.parent / base_name
        if file_sha256(base_path) != base_sha256:
            raise ValueError("base ontology hash drift")
        base = load_ontology(base_path)
        resolved = deepcopy(base)
        for key in (
            "schema_version",
            "ontology_id",
            "status",
            "claim_boundary",
            "multi_label",
            "unknown_and_ambiguous_allowed",
        ):
            if key in ontology:
                resolved[key] = ontology[key]
        for axis in ontology.get("remove_axes", []):
            resolved["axes"].pop(axis, None)
        for axis, definition in ontology.get("axes", {}).items():
            resolved["axes"][axis] = deepcopy(definition)
        for axis, values in ontology.get("axis_value_extensions", {}).items():
            if axis not in resolved["axes"]:
                raise ValueError(f"ontology extends unknown axis {axis!r}")
            existing = resolved["axes"][axis]["values"]
            existing.extend(value for value in values if value not in existing)
        disabled = set(ontology.get("disabled_rule_ids", []))
        for collection in ("regex_rules", "lexical_rules", "detectors"):
            resolved[collection] = [
                rule
                for rule in resolved.get(collection, [])
                if rule["rule_id"] not in disabled
            ] + deepcopy(ontology.get(collection, []))
        resolved["token_assignment_contract"] = {
            **resolved.get("token_assignment_contract", {}),
            **ontology.get("token_assignment_contract", {}),
        }
        resolved["machine_projection_precedence"] = {
            **resolved.get("machine_projection_precedence", {}),
            **deepcopy(ontology.get("machine_projection_precedence", {})),
        }
        resolved["extension_provenance"] = {
            "extends": base_name,
            "base_ontology_sha256": base_sha256,
        }
        ontology = resolved
    if ontology.get("schema_version") not in {
        "adag.process-witness.annotation-ontology.v1",
        "adag.process-witness.annotation-ontology.v2",
        "adag.process-witness.annotation-ontology.v3",
        "adag.process-witness.annotation-ontology.v4",
        "adag.process-witness.annotation-ontology.v5",
        "adag.process-witness.annotation-ontology.v6",
    }:
        raise ValueError("unsupported annotation ontology schema")
    axes = ontology.get("axes")
    if not isinstance(axes, dict) or not axes:
        raise ValueError("ontology axes must be a non-empty object")
    if ontology["schema_version"].endswith((".v2", ".v3", ".v4", ".v5", ".v6")):
        contract = ontology.get("token_assignment_contract", {})
        if contract.get("within_axis") != (
            "zero_or_one_effective_value_per_authoritative_response_token"
        ):
            raise ValueError("v2 ontology lacks the within-axis token contract")
        if "compound_surface" not in axes.get("surface_form", {}).get("values", []):
            raise ValueError("v2 ontology lacks compound_surface")
    for axis, precedence in ontology.get("machine_projection_precedence", {}).items():
        if axis not in axes:
            raise ValueError(f"projection precedence uses unknown axis {axis!r}")
        unknown = set(precedence) - set(axes[axis]["values"])
        if unknown:
            raise ValueError(
                f"projection precedence for {axis!r} has unknown values {sorted(unknown)}"
            )
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
    for start, end in _immediate_relation_result_spans(text):
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


_ARITHMETIC_WORDS: dict[str, tuple[str, ...]] = {
    "addition": ("add", "added", "adding", "addition", "plus", "sum", "total"),
    "subtraction": (
        "subtract",
        "subtracted",
        "subtracting",
        "subtraction",
        "minus",
        "difference",
    ),
    "multiplication": (
        "multiply",
        "multiplied",
        "multiplying",
        "multiplication",
        "product",
        "times",
    ),
    "division": ("divide", "divided", "dividing", "division", "quotient"),
    "modulo": ("mod", "modulo", "remainder"),
    "exponentiation": (
        "power",
        "squared",
        "cubed",
        "exponent",
        "exponentiation",
    ),
}
_ARITHMETIC_SYMBOL_VALUES = {
    "+": "addition",
    "-": "subtraction",
    "*": "multiplication",
    "×": "multiplication",
    "/": "division",
    "÷": "division",
    "%": "modulo",
    "^": "exponentiation",
    "²": "exponentiation",
}
_CORRECTION_RE = re.compile(
    r"\b(?:mistake|wrong|correction|reconsider|hold on|"
    r"correct(?:ing|ed)?\s+(?:myself|this|that|the\s+(?:step|calculation|result)))\b",
    re.IGNORECASE,
)
_VERIFICATION_STRONG_RE = re.compile(
    r"\b(?:verify|confirm|double-check|validate|makes? sense|consistent)\w*\b",
    re.IGNORECASE,
)
_VERIFICATION_CHECK_RE = re.compile(
    r"\bcheck(?:ing|ed)?\b(?=[^.?!\n]*(?:calculation|result|work|steps?|again|"
    r"\d+\s*(?:=|→|->)))",
    re.IGNORECASE,
)
_NEGATED_SENSE_RE = re.compile(
    r"\b(?:does(?:n't| not)|do(?:n't| not)|did(?:n't| not)|not)\s+make\s+sense\b",
    re.IGNORECASE,
)
_UNCERTAINTY_RE = re.compile(
    r"\b(?:wait|maybe|perhaps|likely|probably|not sure|hmm|I think|seems|"
    r"(?:does(?:n't| not)|do(?:n't| not)|did(?:n't| not)|not)\s+make\s+sense)\b",
    re.IGNORECASE,
)
_CONCLUSION_RE = re.compile(
    r"\b(?:therefore|thus|hence|final answer|the answer is|answer should be|"
    r"final node|final result|I'll go with|we conclude)\b",
    re.IGNORECASE,
)
_PLANNING_RE = re.compile(
    r"(?:\b(?:let(?:'s| us)|need to|plan|start|begin|break it down|tackle|"
    r"first(?:ly)?|next|I['’]ll\s+proceed)\b|^\s*(?:step|stage)\s+\d+\s*:)",
    re.IGNORECASE,
)
_RESTATING_RE = re.compile(
    r"\b(?:(?:the\s+)?(?:user|problem|question|task)\s+(?:asks|wants|says|gives|"
    r"provides|requires)|we\s+(?:are|have)\s+given)\b",
    re.IGNORECASE,
)
_DERIVED_RESULT_RE = re.compile(
    r"^\s*Given that,\s+(?:the\s+)?(?:decoded\s+plaintext|plaintext|winner|"
    r"result|answer|value|node)\s+(?:is|=)\b",
    re.IGNORECASE,
)
_INSTRUCTION_RE = re.compile(
    r"(?:\b(?:problem|rules?|instructions?|task|user|question)\s+"
    r"(?:says?|asks?|wants?|gives?|provides?|requires?)\b|\blisted\s+as\b|"
    r"\bthe\s+(?:task|goal)\s+is\b|\b(?:objective|aim)(?:\s+is|\s*:)|"
    r"\bthe\s+task\s+involves\b|"
    r"\byou\s+(?:need|must|should)\b|^\s*[\"'`]+(?:move|update|follow|choose|"
    r"return|start|stop|reverse|decode|encode|swap|shift|substitute)\b)",
    re.IGNORECASE,
)
_LOOKUP_RE = re.compile(
    r"\b(?:look(?:ing|ed)\s+(?:at|up|through)|consult(?:ing|ed)|"
    r"(?:I|we)\s+(?:(?:am|are)\s+)?check(?:ing|ed)?\s+"
    r"(?:all\s+)?(?:the\s+)?(?:references?|outgoing|edges?|list|table|mapping)|"
    r"from\s+the\s+"
    r"(?:references?|list|table)|(?:I|we)\s+(?:find|found)\s+the\s+"
    r"(?:entry|reference|edge))\b",
    re.IGNORECASE,
)
_LOOKUP_IMPERATIVE_RE = re.compile(r"^\s*(?:then\s+)?(?:check|find)\b", re.IGNORECASE)
_ACTIVE_SCAN_RE = re.compile(
    r"\b(?:looking|scanning|reading|checking)\s+(?:through|at)|"
    r"\bfrom\s+the\s+(?:list|table|references?)\b",
    re.IGNORECASE,
)
_QUOTED_LISTED_RELATION_RE = re.compile(
    r"[\"“][^\"”\n]*(?:→|->)[^\"”\n]*[\"”]\s+is\s+listed\s+as\s*:",
    re.IGNORECASE,
)
_LISTED_LANE_RELATION_RE = re.compile(
    r"[^.?!\n]+(?:→|->)[^.?!\n]+\s+is\s+listed\s+as\s+(?:a|an)\s+"
    r"(?:lane|edge|path|reference)\b",
    re.IGNORECASE,
)
_COMPACT_STEP_TRANSITION_RE = re.compile(
    r"^\s*Step\s+\d+\s*:\s*[^.?!\n]+(?:→|->)[^.?!\n]+[.]?\s*$",
    re.IGNORECASE,
)
_SO_TRANSITION_RE = re.compile(
    r"^\s*So\s+[^.?!\n]+(?:→|->)[^.?!\n]+[.]?\s*$",
    re.IGNORECASE,
)
_ARROW_OUTCOME_RE = re.compile(
    r"(?:→|->)\s*[^.?!\n]+\s+wins?\b|"
    r"^\s*(?:Match\s*\d+|Final)\s*:\s*[^.?!\n]+\bvs\b[^.?!\n]+"
    r"(?:→|->)[^.?!\n]+[.]?\s*$",
    re.IGNORECASE,
)
_NUMBERED_IMPERATIVE_RE = re.compile(
    r"^\s*(?:\d+[.)]\s*)?(?:reverse|decode|encode|swap|shift|substitute|"
    r"move|update|follow|choose|return|check|find)\b",
    re.IGNORECASE,
)
_ACTIVE_LOOKUP_IMPERATIVE_RE = re.compile(
    r"^\s*(?:Then\s+)?(?:check|find)\b[^.?!\n]*\b(?:outgoing|edges?|lanes?)\b",
    re.IGNORECASE,
)
_ACTIVE_NEXT_RE = re.compile(r"^\s*Next\s*:\s*\d+\s*[.]?\s*$", re.IGNORECASE)
_TASK_POLICY_RE = re.compile(
    r"^\s*At\s+each\s+step,\s+(?:we|you)\s+(?:choose|select|take|follow)\b",
    re.IGNORECASE,
)
_SEQUENCE_RECAP_RE = re.compile(
    r"\b(?:let(?:'s| us)|I(?:'ll| will))\s+(?:recount|list)\s+the\s+steps\b",
    re.IGNORECASE,
)
_COMPACT_NUMERIC_STEP_RE = re.compile(
    r"^\s*Step\s*\d+\s*:\s*-?\d+(?:\.\d+)?\s*[.]?\s*$",
    re.IGNORECASE,
)
_RECENT_EXECUTION_WINDOW_UNITS = 6
_SEQUENCE_RECAP_WINDOW_UNITS = 16
_STATE_EXECUTION_RE = re.compile(
    r"\b(?:move(?:d)?\s+to|transition(?:ed)?\s+to|"
    r"(?:now|current|next)\s+(?:state|node)\s+(?:is|=)|"
    r"follow(?:ed|ing)?\s+(?:the\s+)?edge\b[^.?!\n]*\bto)\b",
    re.IGNORECASE,
)
_STATE_EXECUTION_VERB_RE = re.compile(
    r"\b(?:move(?:d)?|transition(?:ed)?|follow(?:ed|ing)?)\b",
    re.IGNORECASE,
)
_COMPARISON_RE = re.compile(
    r"\b(?:smallest|largest|minimum|maximum|less than|greater than|"
    r"compare(?:d)?|choose|select)\b",
    re.IGNORECASE,
)
_ENCODING_ACTIVE_RE = re.compile(
    r"\b(?:decode|decodes|decoded|decoding|encode|encodes|encoding|reverse|"
    r"reverses|reversed|reversing|swap|swaps|swapped|swapping|substitute|"
    r"substitutes|substituted|substituting|shift|shifts|shifted|shifting)\b",
    re.IGNORECASE,
)
_ENCODING_RESULT_RE = re.compile(
    r"\b(?:decode|encode|reverse|swap|substitute|shift)\w*\b"
    r"[^.?!\n]*(?:\b(?:give|gives|get|gets|yield\w*|result\w*|"
    r"become|becomes|produce|produces|turns? into|now)\b|→|->)",
    re.IGNORECASE,
)
_ENCODING_OUTPUT_RE = re.compile(
    r"\b(?:decode|encode|reverse|swap|substitute|shift)\w*\b"
    r"[^.?!\n]*\b(?:to|as)\s+(?:[\"'`]|[A-Z0-9])"
)
_ENCODING_NONEXECUTED_RE = re.compile(
    r"\b(?:need|plan|intend|aim|want|have)\s+to\b|\b(?:will|would|should|could|"
    r"might|may)\b|\bif\b|\b(?:task|goal|objective|instruction)\b|"
    r"\b(?:for example|sometimes|background)\b|\b(?:is|are|was|were)\s+done\b",
    re.IGNORECASE,
)
_RESULT_CUE_RE = re.compile(
    r"\b(?:get|gets|got|give|gives|gave|yield|yields|yielded|result(?:s|ed)?|"
    r"becomes?|compute[ds]?|calculat(?:e|es|ed|ing)|therefore|thus|hence)\b",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?!\w|\.\d)")
_RELATION_RE = re.compile(r"(?<![<>=!])=(?!=)|→|->")


def _immediate_relation_result_spans(text: str) -> list[tuple[int, int]]:
    """Return conservative numeric RHS spans, never a later coefficient/value."""

    output: list[tuple[int, int]] = []
    for relation in _RELATION_RE.finditer(text):
        result = re.match(
            r"\s*(?P<result>-?\d+(?:\.\d+)?)(?!\w|\.\d)", text[relation.end() :]
        )
        if result is None:
            continue
        start = relation.end() + result.start("result")
        end = relation.end() + result.end("result")
        if re.match(r"\s*(?:[+*/%^×÷]|-(?=\s*\d))", text[end:]):
            continue
        clause_start = (
            max(
                text.rfind("\n", 0, relation.start()),
                text.rfind(";", 0, relation.start()),
                text.rfind(":", 0, relation.start()),
                text.rfind(",", 0, relation.start()),
                text.rfind("→", 0, relation.start()),
                text.rfind("->", 0, relation.start()),
            )
            + 1
        )
        lhs = text[clause_start : relation.start()].strip()
        context = text[max(clause_start, relation.start() - 40) : relation.start()]
        if re.search(r"\binitial\b", context, re.IGNORECASE):
            continue
        if (
            relation.group() == "="
            and re.fullmatch(r"[A-Za-z_][\w.]*", lhs)
            and not _RESULT_CUE_RE.search(context)
        ):
            continue
        if relation.group() != "=" and not (
            _STATE_EXECUTION_RE.search(context)
            or re.search(r"[+*/%^×÷]", context)
            or _RESULT_CUE_RE.search(context)
        ):
            continue
        output.append((start, end))
    return output


_TITLE_ENTITY_RE = re.compile(r"(?:[A-Z][A-Za-z0-9_-]*)(?:\s+[A-Z][A-Za-z0-9_-]*){0,5}")
_SCALAR_ENTITY_RE = re.compile(r"(?:[A-Za-z_][\w-]*|-?\d+(?:\.\d+)?)")


def _arrow_endpoint_span(
    unit: str, arrow: re.Match[str], *, side: str
) -> tuple[int, int] | None:
    """Capture a full title-case arrow entity, scalar, or identifier; else abstain."""

    if side == "source":
        segment = unit[: arrow.start()]
        title = re.search(rf"(?P<value>{_TITLE_ENTITY_RE.pattern})\s*$", segment)
        scalar = re.search(rf"(?P<value>{_SCALAR_ENTITY_RE.pattern})\s*$", segment)
        found = title or scalar
        return found.span("value") if found else None
    segment = unit[arrow.end() :]
    title = re.match(rf"\s*(?P<value>{_TITLE_ENTITY_RE.pattern})(?![\w-])", segment)
    scalar = re.match(rf"\s*(?P<value>{_SCALAR_ENTITY_RE.pattern})(?![\w-])", segment)
    found = title or scalar
    if found is None:
        return None
    start, end = found.span("value")
    return arrow.end() + start, arrow.end() + end


def _is_state_entity(value: str) -> bool:
    """Accept title-case entities or explicit identifier-shaped names only."""

    return bool(
        re.fullmatch(_TITLE_ENTITY_RE, value)
        or re.fullmatch(r"[A-Za-z][A-Za-z0-9-]*_[A-Za-z0-9_-]+", value)
    )


def _validated_state_arrows(
    unit: str,
) -> list[tuple[re.Match[str], tuple[int, int], tuple[int, int]]]:
    """Return arrows with two nonnumeric entity endpoints and no formula syntax."""

    without_arrows = re.sub(r"→|->", "", unit)
    if "=" in without_arrows or re.search(r"[+*/%^×÷]", without_arrows):
        return []
    output: list[tuple[re.Match[str], tuple[int, int], tuple[int, int]]] = []
    for arrow in re.finditer(r"→|->", unit):
        source = _arrow_endpoint_span(unit, arrow, side="source")
        destination = _arrow_endpoint_span(unit, arrow, side="destination")
        if source is None or destination is None:
            continue
        if not (
            _is_state_entity(unit[source[0] : source[1]])
            and _is_state_entity(unit[destination[0] : destination[1]])
        ):
            continue
        output.append((arrow, source, destination))
    return output


def _terminal_serialization_span(text: str) -> tuple[int, int] | None:
    """Return only a valid JSON object after ``</think>`` at response end."""

    boundary = list(re.finditer(r"</think>", text, re.IGNORECASE))
    if not boundary:
        return None
    start = boundary[-1].end()
    suffix = text[start:]
    found = re.fullmatch(r"\s*(\{[^\r\n]*\})\s*", suffix)
    if not found:
        return None
    try:
        parsed = json.loads(found.group(1))
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return start + found.start(1), start + found.end(1)


def _text_units(text: str) -> list[tuple[int, int]]:
    """Split nonblank prose into deterministic line/sentence units.

    Decimal points and terminal JSON stay inside their source unit.  Spans exclude
    surrounding whitespace so structural coverage is not mistaken for punctuation
    coverage.
    """

    output: list[tuple[int, int]] = []
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
        for found in re.finditer(r"[.!?]+(?=\s+|$)", content):
            if (
                content[found.start()] == "."
                and found.start() > 0
                and found.end() < len(content)
                and content[found.start() - 1].isdigit()
                and content[found.end()].isdigit()
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


def _arithmetic_evidence(unit: str) -> tuple[set[str], list[tuple[int, int, str]]]:
    operations: set[str] = set()
    cues: list[tuple[int, int, str]] = []
    for symbol, value in _ARITHMETIC_SYMBOL_VALUES.items():
        probe = {
            "symbols": {symbol: value},
            "axis": "operation",
            "rule_id": "probe",
            "interpretation": "semantic_hypothesis",
            "confidence": "medium",
        }
        for match in _arithmetic_symbol_matches(unit, probe):
            operations.add(value)
            cues.append((match.start, match.end, value))
    for value, terms in _ARITHMETIC_WORDS.items():
        pattern = re.compile(
            r"(?<!\w)(?:" + "|".join(re.escape(term) for term in terms) + r")(?!\w)",
            re.IGNORECASE,
        )
        for found in pattern.finditer(unit):
            if (
                value == "modulo"
                and found.group().casefold() == "mod"
                and re.match(r"\s*=", unit[found.end() :])
            ):
                continue
            operations.add(value)
            cues.append((found.start(), found.end(), value))
    return operations, cues


def _unit_phase(
    unit: str,
    *,
    is_serialization: bool,
    instruction: bool,
    lookup: bool,
    verification: bool,
    has_executed_event: bool,
    derived_result: bool = False,
) -> str:
    if is_serialization:
        return "answer_serialization"
    if derived_result:
        return "conclusion"
    if instruction:
        return "instruction_or_task_description"
    if _CORRECTION_RE.search(unit):
        return "correction_or_reconsideration"
    if _CONCLUSION_RE.search(unit):
        return "conclusion"
    if _UNCERTAINTY_RE.search(unit):
        return "uncertainty_or_deliberation"
    if _PLANNING_RE.search(unit) and not has_executed_event:
        return "planning"
    if lookup:
        return "reference_lookup_or_reading"
    if verification:
        return "verification"
    if _RESTATING_RE.search(unit) and not has_executed_event:
        return "orientation_or_restating"
    if has_executed_event:
        return "working_or_derivation"
    return "unclassified_or_other"


def _quoted_spans(text: str) -> list[tuple[int, int]]:
    """Return paired double-quote spans for conservative instruction gating."""

    output: list[tuple[int, int]] = []
    open_start: int | None = None
    for found in re.finditer(r'["“”]', text):
        marker = found.group()
        if marker == "“":
            open_start = found.start()
        elif marker == "”" and open_start is not None:
            output.append((open_start, found.end()))
            open_start = None
        elif marker == '"':
            if open_start is None:
                open_start = found.start()
            else:
                output.append((open_start, found.end()))
                open_start = None
    return output


def _process_structure_matches(text: str, rule: Mapping[str, Any]) -> Iterator[Match]:
    """Suggest broad phases and locally evidenced process-event structure."""

    serialization = _terminal_serialization_span(text)
    process_rule = rule["rule_id"]
    event_modality = rule.get("event_modality")
    conservative = event_modality in {
        "conservative_v7",
        "conservative_v8",
        "conservative_v9",
    }
    conservative_v8 = event_modality in {"conservative_v8", "conservative_v9"}
    conservative_v9 = event_modality == "conservative_v9"
    quoted_spans = _quoted_spans(text) if conservative else []
    units = _text_units(text)
    recent_execution_units = 0
    sequence_recap_units = 0
    for unit_index, (start, end) in enumerate(units):
        unit = text[start:end]
        previous_unit = text[slice(*units[unit_index - 1])] if unit_index else ""
        recent_execution = recent_execution_units > 0
        if conservative_v9 and _SEQUENCE_RECAP_RE.search(unit):
            sequence_recap_units = _SEQUENCE_RECAP_WINDOW_UNITS
        active_sequence_step = bool(
            conservative_v9
            and sequence_recap_units > 0
            and _COMPACT_NUMERIC_STEP_RE.fullmatch(unit)
        )
        is_serialization = serialization == (start, end)
        numbers = list(_NUMBER_RE.finditer(unit))
        operations, operator_cues = _arithmetic_evidence(unit)
        relations = list(_RELATION_RE.finditer(unit))
        valid_state_arrows = _validated_state_arrows(unit) if conservative else []
        arrow_evidence = (
            bool(valid_state_arrows) if conservative else bool(re.search(r"→|->", unit))
        )
        quoted_imperative = any(
            quote_start < start < quote_end
            and re.match(
                r"\s*(?:\d+\.\s*)?(?:move|update|follow|choose|return|start|stop)\b",
                unit,
                re.IGNORECASE,
            )
            for quote_start, quote_end in quoted_spans
        )
        active_scan_context = bool(
            _ACTIVE_SCAN_RE.search(text[max(0, start - 400) : end])
        )
        listed_relation_lookup = bool(
            conservative_v8
            and valid_state_arrows
            and active_scan_context
            and (
                _QUOTED_LISTED_RELATION_RE.search(unit)
                or (conservative_v9 and _LISTED_LANE_RELATION_RE.search(unit))
            )
        )
        lookup_imperative = conservative and bool(_LOOKUP_IMPERATIVE_RE.search(unit))
        mid_execution_lookup = bool(
            conservative_v9
            and recent_execution
            and _ACTIVE_LOOKUP_IMPERATIVE_RE.search(unit)
        )
        numbered_instruction = bool(
            conservative_v9
            and _NUMBERED_IMPERATIVE_RE.search(unit)
            and re.fullmatch(r"\s*\d+[.)]\s*", previous_unit)
        )
        policy_instruction = bool(conservative_v9 and _TASK_POLICY_RE.search(unit))
        instruction = conservative and bool(
            (_INSTRUCTION_RE.search(unit) and not listed_relation_lookup)
            or quoted_imperative
            or (lookup_imperative and not mid_execution_lookup)
            or numbered_instruction
            or policy_instruction
        )
        lookup = (
            conservative
            and not instruction
            and bool(
                listed_relation_lookup
                or mid_execution_lookup
                or _LOOKUP_RE.search(unit)
            )
        )
        verification = bool(
            _VERIFICATION_STRONG_RE.search(unit) or _VERIFICATION_CHECK_RE.search(unit)
        ) and not (instruction or lookup or _NEGATED_SENSE_RE.search(unit))
        result_spans = set(_immediate_relation_result_spans(unit))
        arithmetic_syntax = bool(operations) and (len(numbers) >= 2 or bool(relations))
        arithmetic_evidence = arithmetic_syntax and (
            not conservative
            or (
                not instruction
                and (bool(result_spans) or bool(_RESULT_CUE_RE.search(unit)))
            )
        )
        compact_step_execution = bool(
            conservative_v9
            and valid_state_arrows
            and _COMPACT_STEP_TRANSITION_RE.fullmatch(unit)
        )
        so_transition = bool(
            conservative_v9 and valid_state_arrows and _SO_TRANSITION_RE.fullmatch(unit)
        )
        comparison_outcome = bool(
            conservative_v9 and valid_state_arrows and _ARROW_OUTCOME_RE.search(unit)
        )
        state_execution = not instruction and bool(
            _STATE_EXECUTION_RE.search(unit)
            or compact_step_execution
            or (so_transition and not comparison_outcome)
        )
        state_schema = (
            conservative
            and arrow_evidence
            and not state_execution
            and not listed_relation_lookup
            and not comparison_outcome
        )
        comparison_evidence = bool(_COMPARISON_RE.search(unit)) and (
            not conservative
            or (
                not instruction
                and bool(
                    re.search(
                        r"\b(?:is|are|choose|select|selected)\b", unit, re.IGNORECASE
                    )
                )
            )
        )
        comparison_evidence = comparison_evidence or comparison_outcome
        encoding_evidence = bool(_ENCODING_ACTIVE_RE.search(unit)) and not instruction
        if conservative:
            encoding_evidence = (
                encoding_evidence
                and bool(
                    _ENCODING_RESULT_RE.search(unit) or _ENCODING_OUTPUT_RE.search(unit)
                )
                and not bool(_ENCODING_NONEXECUTED_RE.search(unit))
            )
        correction = bool(_CORRECTION_RE.search(unit)) and not instruction
        derived_result = conservative_v8 and bool(_DERIVED_RESULT_RE.search(unit))
        active_next = bool(
            conservative_v9 and recent_execution and _ACTIVE_NEXT_RE.fullmatch(unit)
        )
        has_executed_event = any(
            (
                is_serialization,
                arithmetic_evidence,
                state_execution,
                comparison_evidence,
                encoding_evidence,
                lookup,
                verification,
                correction,
                active_next,
                active_sequence_step,
            )
        )
        phase = _unit_phase(
            unit,
            is_serialization=is_serialization,
            instruction=instruction,
            lookup=lookup,
            verification=verification,
            has_executed_event=has_executed_event,
            derived_result=derived_result,
        )
        yield Match(
            start,
            end,
            "discourse_phase",
            phase,
            f"{process_rule}.discourse.{phase}",
            "observable_structure" if is_serialization else "semantic_hypothesis",
            "high" if is_serialization else "medium",
            ()
            if is_serialization
            else ("discourse phase is a deterministic structural hypothesis",),
        )

        event_value: str | None = None
        if is_serialization:
            event_value = "answer_event_candidate"
        elif state_schema:
            event_value = "state_relation_or_schema_candidate"
        elif instruction:
            event_value = None
        elif correction:
            event_value = "correction_event_candidate"
        elif lookup:
            event_value = "reference_lookup_event_candidate"
        elif verification:
            event_value = "verification_event_candidate"
        elif phase == "conclusion":
            event_value = "answer_event_candidate"
        elif arithmetic_evidence and state_execution:
            event_value = "state_update_with_arithmetic"
        elif arithmetic_evidence:
            event_value = "arithmetic_event_candidate"
        elif state_execution:
            event_value = "state_transition_event_candidate"
        elif comparison_evidence:
            event_value = "comparison_or_selection_event_candidate"
        elif encoding_evidence:
            event_value = "encoding_or_decoding_event_candidate"
        current_active = bool(
            has_executed_event
            and not instruction
            and phase in {"working_or_derivation", "reference_lookup_or_reading"}
        )
        recent_execution_units = (
            _RECENT_EXECUTION_WINDOW_UNITS
            if current_active
            else max(0, recent_execution_units - 1)
        )
        sequence_recap_units = max(0, sequence_recap_units - 1)
        if event_value is None:
            continue
        yield Match(
            start,
            end,
            "process_span",
            event_value,
            f"{process_rule}.span.{event_value}",
            "semantic_hypothesis",
            "high" if is_serialization else "medium",
            ("candidate event span does not establish execution or correctness",),
        )

        if event_value in {
            "arithmetic_event_candidate",
            "state_update_with_arithmetic",
        }:
            operation = (
                next(iter(operations)) if len(operations) == 1 else "mixed_arithmetic"
            )
            yield Match(
                start,
                end,
                "event_operation",
                operation,
                f"{process_rule}.operation.{operation}",
                "semantic_hypothesis",
                "medium",
                ("operation is propagated across a locally evidenced candidate event",),
            )
            yield Match(
                start,
                end,
                "domain",
                "arithmetic",
                f"{process_rule}.domain.arithmetic",
                "semantic_hypothesis",
                "medium",
            )
        elif event_value == "state_transition_event_candidate":
            yield Match(
                start,
                end,
                "event_operation",
                "state_transition",
                f"{process_rule}.operation.state-transition",
                "semantic_hypothesis",
                "medium",
            )
            for evidence in _STATE_EXECUTION_RE.finditer(unit):
                for cue in _STATE_EXECUTION_VERB_RE.finditer(
                    unit, evidence.start(), evidence.end()
                ):
                    yield Match(
                        start + cue.start(),
                        start + cue.end(),
                        "operation",
                        "state_transition",
                        f"{process_rule}.cue.state-transition",
                        "semantic_hypothesis",
                        "medium",
                    )
        elif event_value == "state_relation_or_schema_candidate":
            yield Match(
                start,
                end,
                "event_operation",
                "state_relation_or_schema",
                f"{process_rule}.operation.state-relation-or-schema",
                "semantic_hypothesis",
                "medium",
                ("arrow relation may be task data or schema rather than execution",),
            )
        elif event_value == "reference_lookup_event_candidate":
            yield Match(
                start,
                end,
                "event_operation",
                "lookup",
                f"{process_rule}.operation.lookup",
                "semantic_hypothesis",
                "medium",
            )
        elif event_value == "verification_event_candidate":
            yield Match(
                start,
                end,
                "event_operation",
                "verification",
                f"{process_rule}.operation.verification",
                "semantic_hypothesis",
                "medium",
            )
            for cue in (
                *_VERIFICATION_STRONG_RE.finditer(unit),
                *_VERIFICATION_CHECK_RE.finditer(unit),
            ):
                yield Match(
                    start + cue.start(),
                    start + cue.end(),
                    "operation",
                    "verification",
                    f"{process_rule}.cue.verification",
                    "semantic_hypothesis",
                    "medium",
                )
                yield Match(
                    start + cue.start(),
                    start + cue.end(),
                    "process_role",
                    "verification_cue",
                    f"{process_rule}.role.verification-cue",
                    "semantic_hypothesis",
                    "medium",
                )
        elif event_value == "correction_event_candidate":
            yield Match(
                start,
                end,
                "event_operation",
                "correction",
                f"{process_rule}.operation.correction",
                "semantic_hypothesis",
                "medium",
            )
        elif event_value == "comparison_or_selection_event_candidate":
            yield Match(
                start,
                end,
                "event_operation",
                "order_comparison",
                f"{process_rule}.operation.comparison",
                "semantic_hypothesis",
                "medium",
            )
        elif event_value == "encoding_or_decoding_event_candidate":
            yield Match(
                start,
                end,
                "event_operation",
                "encoding_or_decoding",
                f"{process_rule}.operation.encoding",
                "semantic_hypothesis",
                "medium",
            )
            for cue in _ENCODING_ACTIVE_RE.finditer(unit):
                yield Match(
                    start + cue.start(),
                    start + cue.end(),
                    "operation",
                    "encoding_or_decoding",
                    f"{process_rule}.cue.encoding",
                    "semantic_hypothesis",
                    "medium",
                )

        if event_value in {
            "arithmetic_event_candidate",
            "state_update_with_arithmetic",
        }:
            for cue_start, cue_end, _ in operator_cues:
                yield Match(
                    start + cue_start,
                    start + cue_end,
                    "process_role",
                    "operator_cue",
                    f"{process_rule}.role.operator-cue",
                    "semantic_hypothesis",
                    "medium",
                )

        result_starts = {result_start for result_start, _ in result_spans}
        if not result_starts and arithmetic_evidence:
            for number in reversed(numbers):
                prefix = unit[max(0, number.start() - 32) : number.start()]
                if re.search(
                    r"\b(?:get|gets|got|give|gives|gave|yield|yields|yielded|"
                    r"result(?:s)?(?:\s+is)?|become|becomes|became)\s*$",
                    prefix,
                    re.IGNORECASE,
                ):
                    result_starts.add(number.start())
                    break
        if event_value in {
            "arithmetic_event_candidate",
            "state_update_with_arithmetic",
        }:
            for number in numbers:
                prefix = unit[max(0, number.start() - 12) : number.start()]
                if re.search(
                    r"(?:step|stage|match|sentence)\s*$", prefix, re.IGNORECASE
                ):
                    continue
                values = ["operand_candidate"]
                if number.start() in result_starts:
                    values.append("intermediate_result_candidate")
                for value in values:
                    yield Match(
                        start + number.start(),
                        start + number.end(),
                        "process_role",
                        value,
                        f"{process_rule}.role.{value}",
                        "semantic_hypothesis",
                        "medium",
                        ("role is inferred from local event syntax, not correctness",),
                    )

        if arrow_evidence and event_value in {
            "state_transition_event_candidate",
            "state_update_with_arithmetic",
            "state_relation_or_schema_candidate",
        }:
            arrow_spans = (
                valid_state_arrows
                if conservative
                else [
                    (
                        arrow,
                        _arrow_endpoint_span(unit, arrow, side="source"),
                        _arrow_endpoint_span(unit, arrow, side="destination"),
                    )
                    for arrow in re.finditer(r"→|->", unit)
                ]
            )
            for _, source, destination in arrow_spans:
                if source is None or destination is None:
                    continue
                yield Match(
                    start + source[0],
                    start + source[1],
                    "process_role",
                    "state_value_candidate",
                    f"{process_rule}.role.state-value",
                    "semantic_hypothesis",
                    "medium",
                )
                destination_role = (
                    "state_value_candidate"
                    if event_value == "state_relation_or_schema_candidate"
                    else "state_update"
                )
                yield Match(
                    start + destination[0],
                    start + destination[1],
                    "process_role",
                    destination_role,
                    f"{process_rule}.role.{destination_role}",
                    "semantic_hypothesis",
                    "medium",
                )


def _serialization_segment_matches(
    text: str, rule: Mapping[str, Any]
) -> Iterator[Match]:
    boundary = list(re.finditer(r"</?think>", text, re.IGNORECASE))
    close = next(
        (found for found in reversed(boundary) if found.group().lower() == "</think>"),
        None,
    )
    opening = next(
        (found for found in boundary if found.group().lower() == "<think>"), None
    )
    thinking_start = opening.end() if opening else 0
    thinking_end = close.start() if close else len(text)
    if thinking_end > thinking_start:
        yield Match(
            thinking_start,
            thinking_end,
            "serialization_segment",
            "thinking_segment",
            f"{rule['rule_id']}.thinking",
            "observable_structure",
            "high",
        )
    for found in boundary:
        yield Match(
            found.start(),
            found.end(),
            "serialization_segment",
            "boundary_or_control",
            f"{rule['rule_id']}.boundary",
            "observable_structure",
            "high",
        )
    final = _terminal_serialization_span(text)
    if final:
        yield Match(
            final[0],
            final[1],
            "serialization_segment",
            "final_answer_segment",
            f"{rule['rule_id']}.final",
            "observable_structure",
            "high",
        )


DETECTORS = {
    "arithmetic_symbols": _arithmetic_symbol_matches,
    "numeric_relation_results": _numeric_relation_matches,
    "process_structure": _process_structure_matches,
    "serialization_segments": _serialization_segment_matches,
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
    if accepted_answer_keys:
        terminal = _terminal_serialization_span(text)
        accepted_terminal_key = any(
            match.axis == "representation"
            and match.value == "answer_key"
            and terminal is not None
            and terminal[0] <= match.start < match.end <= terminal[1]
            for match in matches
        )
        if terminal is not None and not accepted_terminal_key:
            matches = [
                match
                for match in matches
                if not (
                    (match.start, match.end) == terminal
                    and match.value
                    in {
                        "answer_serialization",
                        "answer_event_candidate",
                        "final_answer_segment",
                    }
                )
            ]
    if "discourse_phase" in ontology["axes"]:
        matches.extend(
            [
                Match(
                    match.start,
                    match.end,
                    "process_role",
                    "final_result",
                    "process.structure.v3.role.final-result-from-answer-commitment",
                    "semantic_hypothesis",
                    "high",
                    ("terminal answer syntax identifies commitment, not correctness",),
                )
                for match in matches
                if match.axis == "process_role" and match.value == "answer_commitment"
            ]
        )
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


def event_token_position_matches(
    text: str, offsets: Sequence[Sequence[int]], matches: Sequence[Match]
) -> list[Match]:
    """Derive exclusive event-relative token positions from process spans."""

    output: list[Match] = []
    occupied: set[int] = set()
    process_spans = sorted(
        (match for match in matches if match.axis == "process_span"),
        key=lambda match: (match.start, match.end),
    )
    for process in process_spans:
        aligned = align_span(offsets, process.start, process.end)
        positions = list(range(aligned["start"], aligned["end"]))
        available = [position for position in positions if position not in occupied]
        if not available:
            continue
        position_groups: list[tuple[int, int, str]] = []
        if len(available) == 1:
            position_groups.append((available[0], available[0] + 1, "span_terminal"))
        else:
            position_groups.append((available[0], available[0] + 1, "span_onset"))
            if len(available) > 2:
                position_groups.append((available[1], available[-1], "span_interior"))
            position_groups.append((available[-1], available[-1] + 1, "span_terminal"))
        for token_start_position, token_end_position, value in position_groups:
            token_start = int(offsets[token_start_position][0])
            token_end = int(offsets[token_end_position - 1][1])
            if token_start == token_end:
                continue
            output.append(
                Match(
                    token_start,
                    token_end,
                    "event_token_position",
                    value,
                    f"process.structure.v3.event-position.{value}",
                    "semantic_hypothesis",
                    "medium",
                )
            )
        occupied.update(available)
        following = aligned["end"]
        if following < len(offsets) and following not in occupied:
            token_start, token_end = offsets[following]
            separator = text[token_start:token_end]
            if separator and all(
                character.isspace() or character in ".,;:!?" for character in separator
            ):
                output.append(
                    Match(
                        int(token_start),
                        int(token_end),
                        "event_token_position",
                        "following_separator",
                        "process.structure.v3.event-position.following-separator",
                        "semantic_hypothesis",
                        "medium",
                    )
                )
                occupied.add(following)
    return output


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
    matches = suggest_matches(
        text,
        ontology,
        accepted_answer_keys=accepted_answer_keys or None,
    )
    if "event_token_position" in ontology["axes"]:
        matches.extend(event_token_position_matches(text, offsets, matches))
        matches.sort(
            key=lambda match: (
                match.start,
                match.end,
                match.axis,
                match.value,
                match.rule_id,
            )
        )
    for match in matches:
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
            "machine_projection_precedence": ontology.get(
                "machine_projection_precedence", {}
            ),
            "extension_provenance": ontology.get("extension_provenance"),
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
    effective machine layer. Overlapping surface-form values map to
    ``compound_surface``. Ontology-declared precedence resolves only explicitly listed
    candidate conflicts; all other semantic conflicts remain sorted lists.
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
        precedence = (
            document["ontology"].get("machine_projection_precedence", {}).get(axis, [])
        )
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
                elif precedence and any(
                    candidate in values for candidate in precedence
                ):
                    value = next(
                        candidate for candidate in precedence if candidate in values
                    )
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
    """Select deterministic examples stratified across each rule's response support."""

    candidates: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for document in documents:
        text = document["text"]
        for suggestion in document["suggestions"]:
            rule_id = suggestion["provenance"]["rule_id"]
            start, end = suggestion["character_span"]
            candidate = {
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
            response_candidates = candidates[rule_id]
            response_id = document["response_id"]
            retained = response_candidates.get(response_id)
            if retained is None or (
                len(candidate["match"]),
                candidate["suggestion_id"],
            ) < (len(retained["match"]), retained["suggestion_id"]):
                response_candidates[response_id] = candidate

    examples: dict[str, list[dict[str, Any]]] = {}
    for rule_id, by_response in candidates.items():
        response_ids = list(by_response)
        limit = 7
        if len(response_ids) <= limit:
            selected_response_ids = response_ids
        else:
            selected_response_ids = [
                response_ids[round(index * (len(response_ids) - 1) / (limit - 1))]
                for index in range(limit)
            ]
        selected: list[dict[str, Any]] = []
        for response_id in selected_response_ids:
            candidate = by_response[response_id]
            selected.append(candidate)
        examples[rule_id] = selected
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
