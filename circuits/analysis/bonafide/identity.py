"""Versioned signed-basis and trace-local occurrence identities."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

BASIS_KEY_SCHEMA = "adag.bonafide.signed-basis-key.v1"
OCCURRENCE_KEY_SCHEMA = "adag.bonafide.occurrence-key.v1"
POLARITY_DERIVATION = "activation-sign-nonnegative-positive.v1"

Polarity = Literal["+", "-"]


def _required_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _integer(value: object, field: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _polarity(value: object) -> Polarity:
    if value == "+":
        return "+"
    if value == "-":
        return "-"
    raise ValueError("polarity must be '+' or '-'")


def polarity_from_activation(value: object) -> Polarity:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("activation must be numeric")
    activation = float(value)
    if not math.isfinite(activation):
        raise ValueError("activation must be finite")
    return "+" if activation >= 0 else "-"


def polarity_from_raw_node(row: Mapping[str, Any]) -> Polarity:
    """Resolve polarity once at raw-row ingestion and reject conflicts."""

    derived = polarity_from_activation(row.get("activation"))
    explicit = row.get("polarity")
    if explicit is None:
        return derived
    resolved = _polarity(explicit)
    if resolved != derived:
        raise ValueError(
            "explicit raw-node polarity conflicts with activation-sign polarity"
        )
    return resolved


@dataclass(frozen=True, order=True)
class SignedBasisKey:
    model_id: str
    model_revision: str
    layer: int
    neuron_index: int
    polarity: Polarity

    def __post_init__(self) -> None:
        _required_string(self.model_id, "model_id")
        _required_string(self.model_revision, "model_revision")
        _integer(self.layer, "layer", minimum=-1)
        _integer(self.neuron_index, "neuron_index", minimum=0)
        _polarity(self.polarity)

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": BASIS_KEY_SCHEMA,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "layer": self.layer,
            "neuron_index": self.neuron_index,
            "polarity": self.polarity,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> SignedBasisKey:
        if record.get("schema_version") != BASIS_KEY_SCHEMA:
            raise ValueError("unsupported signed-basis key schema")
        return cls(
            model_id=_required_string(record.get("model_id"), "model_id"),
            model_revision=_required_string(
                record.get("model_revision"), "model_revision"
            ),
            layer=_integer(record.get("layer"), "layer", minimum=-1),
            neuron_index=_integer(
                record.get("neuron_index"), "neuron_index", minimum=0
            ),
            polarity=_polarity(record.get("polarity")),
        )


@dataclass(frozen=True, order=True)
class OccurrenceKey:
    trace_unit_id: str
    token_position: int
    layer: int
    neuron_index: int
    polarity: Polarity

    def __post_init__(self) -> None:
        _required_string(self.trace_unit_id, "trace_unit_id")
        _integer(self.token_position, "token_position", minimum=0)
        _integer(self.layer, "layer", minimum=-1)
        _integer(self.neuron_index, "neuron_index", minimum=0)
        _polarity(self.polarity)

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": OCCURRENCE_KEY_SCHEMA,
            "trace_unit_id": self.trace_unit_id,
            "token_position": self.token_position,
            "layer": self.layer,
            "neuron_index": self.neuron_index,
            "polarity": self.polarity,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> OccurrenceKey:
        if record.get("schema_version") != OCCURRENCE_KEY_SCHEMA:
            raise ValueError("unsupported occurrence-key schema")
        return cls(
            trace_unit_id=_required_string(
                record.get("trace_unit_id"), "trace_unit_id"
            ),
            token_position=_integer(
                record.get("token_position"), "token_position", minimum=0
            ),
            layer=_integer(record.get("layer"), "layer", minimum=-1),
            neuron_index=_integer(
                record.get("neuron_index"), "neuron_index", minimum=0
            ),
            polarity=_polarity(record.get("polarity")),
        )


def basis_key_from_raw_node(
    row: Mapping[str, Any],
    *,
    model_id: str,
    model_revision: str,
) -> SignedBasisKey:
    return SignedBasisKey(
        model_id=model_id,
        model_revision=model_revision,
        layer=_integer(row.get("layer"), "layer", minimum=-1),
        neuron_index=_integer(row.get("neuron"), "neuron", minimum=0),
        polarity=polarity_from_raw_node(row),
    )


def occurrence_key_from_raw_node(
    row: Mapping[str, Any],
    *,
    trace_unit_id: str,
) -> OccurrenceKey:
    return OccurrenceKey(
        trace_unit_id=trace_unit_id,
        token_position=_integer(row.get("token"), "token", minimum=0),
        layer=_integer(row.get("layer"), "layer", minimum=-1),
        neuron_index=_integer(row.get("neuron"), "neuron", minimum=0),
        polarity=polarity_from_raw_node(row),
    )


def basis_from_occurrence(
    occurrence: OccurrenceKey,
    *,
    model_id: str,
    model_revision: str,
) -> SignedBasisKey:
    return SignedBasisKey(
        model_id=model_id,
        model_revision=model_revision,
        layer=occurrence.layer,
        neuron_index=occurrence.neuron_index,
        polarity=occurrence.polarity,
    )


@dataclass(frozen=True, order=True)
class CircuitInputRef:
    trace_unit_id: str
    local_ci_index: int
    local_label: str
    global_atlas_ci_index: int

    def __post_init__(self) -> None:
        _required_string(self.trace_unit_id, "trace_unit_id")
        _integer(self.local_ci_index, "local_ci_index", minimum=0)
        _required_string(self.local_label, "local_label")
        _integer(
            self.global_atlas_ci_index,
            "global_atlas_ci_index",
            minimum=0,
        )


def build_circuit_input_refs(
    local_inputs: list[tuple[str, int, str]],
) -> tuple[CircuitInputRef, ...]:
    """Assign reconstructable global indices in deterministic local-key order."""

    if len(set(local_inputs)) != len(local_inputs):
        raise ValueError("duplicate local circuit-input identity")
    ordered = sorted(local_inputs)
    return tuple(
        CircuitInputRef(
            trace_unit_id=trace_unit_id,
            local_ci_index=local_index,
            local_label=local_label,
            global_atlas_ci_index=global_index,
        )
        for global_index, (trace_unit_id, local_index, local_label) in enumerate(
            ordered
        )
    )
