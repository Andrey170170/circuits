"""Reconstruct cluster-level attribution profiles from frozen compact traces."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from circuits.analysis.bonafide.canonical import file_sha256
from circuits.descriptions.types import ActivationRecord
from circuits.tracing.artifact import load_compact_trace


@dataclass(frozen=True)
class ClusterProfile:
    trace_unit_id: str
    family_partition: str
    source_manifest_sha256: str
    matched_signed_basis_count: int
    record: ActivationRecord


def load_cluster_members(
    assignments_path: Path,
) -> dict[int, dict[tuple[int, int], list[int]]]:
    frame = pd.read_parquet(
        assignments_path,
        columns=["layer", "neuron_index", "polarity", "assigned", "cluster_id"],
    )
    members: dict[int, dict[tuple[int, int], list[int]]] = {}
    for layer, neuron_index, polarity, assigned, cluster_id_value in frame.itertuples(
        index=False, name=None
    ):
        if not bool(assigned):
            continue
        cluster_id = int(cluster_id_value)
        key = (int(layer), int(neuron_index))
        sign = 1 if polarity == "+" else -1
        members.setdefault(cluster_id, {}).setdefault(key, []).append(sign)
    return members


def _source_tokens(tokenizer: Any, token_ids: list[int]) -> list[str]:
    return [
        tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        for token_id in token_ids
    ]


def build_cluster_profile(
    exemplar: dict[str, Any],
    *,
    cluster_members: dict[tuple[int, int], list[int]],
    source_tokenizer: Any,
) -> ClusterProfile:
    manifest_path = Path(str(exemplar["artifact_manifest_path"]))
    if file_sha256(manifest_path) != exemplar["artifact_manifest_sha256"]:
        raise ValueError(f"exemplar manifest hash mismatch: {manifest_path}")
    artifact = load_compact_trace(manifest_path.parent)
    if artifact.manifest["data_sha256"] != exemplar["artifact_payload_sha256"]:
        raise ValueError(f"exemplar payload identity mismatch: {manifest_path.parent}")
    if artifact.manifest["artifact_id"] != exemplar["trace_unit_id"]:
        raise ValueError(f"exemplar trace identity mismatch: {manifest_path.parent}")

    data = artifact.circuit_data
    token_ids = [int(value) for value in data.cis[0]]
    maps: list[list[float]] = []
    matched_signed = 0
    for row in data.df_node.itertuples(index=False):
        layer = getattr(row, "layer", None)
        neuron = getattr(row, "neuron", None)
        attr_map = getattr(row, "attr_map", None)
        if layer is None or neuron is None or attr_map is None:
            continue
        try:
            key = (int(layer), int(neuron))
        except (TypeError, ValueError):
            continue
        signs = cluster_members.get(key, [])
        if not signs:
            continue
        values = [float(value) for value in attr_map]
        if len(values) < len(token_ids):
            values = [0.0] * (len(token_ids) - len(values)) + values
        elif len(values) > len(token_ids):
            values = values[-len(token_ids) :]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(
                f"invalid attr_map for trace {exemplar['trace_unit_id']} and basis {key}"
            )
        for sign in signs:
            maps.append([sign * value for value in values])
            matched_signed += 1
    if not maps:
        raise ValueError(
            f"cluster has no matched signed bases in trace {exemplar['trace_unit_id']}"
        )
    activations = [sum(values) / len(maps) for values in zip(*maps, strict=True)]
    return ClusterProfile(
        trace_unit_id=str(exemplar["trace_unit_id"]),
        family_partition=str(exemplar["family_partition"]),
        source_manifest_sha256=str(exemplar["artifact_manifest_sha256"]),
        matched_signed_basis_count=matched_signed,
        record=ActivationRecord(
            tokens=_source_tokens(source_tokenizer, token_ids),
            token_ids=token_ids,
            activations=activations,
        ),
    )


def build_partition_profiles(
    row: dict[str, Any],
    *,
    partition: str,
    members: dict[int, dict[tuple[int, int], list[int]]],
    source_tokenizer: Any,
) -> list[ClusterProfile]:
    cluster_id = int(row["cluster_id"])
    return [
        build_cluster_profile(
            exemplar,
            cluster_members=members[cluster_id],
            source_tokenizer=source_tokenizer,
        )
        for exemplar in row["balanced_target_exemplars"]
        if exemplar["family_partition"] == partition
    ]


def render_highlighted_record(
    record: ActivationRecord, *, max_highlights: int = 16
) -> str:
    ranked = sorted(
        range(len(record.activations)),
        key=lambda index: (-abs(record.activations[index]), index),
    )
    highlighted = set(ranked[:max_highlights])
    parts: list[str] = []
    for index, (token, activation) in enumerate(
        zip(record.tokens, record.activations, strict=True)
    ):
        if index in highlighted:
            escaped = token.replace("</mark>", "&lt;/mark&gt;")
            parts.append(
                f'<mark token_index="{index}" score="{activation:+.6g}">{escaped}</mark>'
            )
        else:
            parts.append(token)
    return "".join(parts)


def render_highlighted_profile(
    profile: ClusterProfile, *, max_highlights: int = 16
) -> str:
    return render_highlighted_record(profile.record, max_highlights=max_highlights)


def retokenize_for_simulator(
    record: ActivationRecord, simulator_tokenizer: Any
) -> tuple[ActivationRecord, dict[str, float | int]]:
    """Map source-token activations to simulator tokens by character overlap."""

    text = "".join(record.tokens)
    source_spans: list[tuple[int, int, float]] = []
    cursor = 0
    for token, activation in zip(record.tokens, record.activations, strict=True):
        end = cursor + len(token)
        source_spans.append((cursor, end, float(activation)))
        cursor = end
    encoded = simulator_tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    token_ids = [int(value) for value in encoded["input_ids"]]
    offsets = [(int(start), int(end)) for start, end in encoded["offset_mapping"]]
    mapped: list[float] = []
    covered_chars = 0
    source_index = 0
    for start, end in offsets:
        if end <= start:
            mapped.append(0.0)
            continue
        while (
            source_index < len(source_spans) and source_spans[source_index][1] <= start
        ):
            source_index += 1
        numerator = 0.0
        denominator = 0
        cursor_index = source_index
        while cursor_index < len(source_spans):
            source_start, source_end, activation = source_spans[cursor_index]
            if source_start >= end:
                break
            overlap = max(0, min(end, source_end) - max(start, source_start))
            if overlap:
                numerator += overlap * activation
                denominator += overlap
            cursor_index += 1
        mapped.append(numerator / denominator if denominator else 0.0)
        covered_chars += denominator
    tokens = simulator_tokenizer.convert_ids_to_tokens(token_ids)
    return (
        ActivationRecord(tokens=tokens, token_ids=token_ids, activations=mapped),
        {
            "source_token_count": len(record.tokens),
            "simulator_token_count": len(token_ids),
            "source_character_count": len(text),
            "covered_character_instances": covered_chars,
            "coverage_fraction": covered_chars / len(text) if text else 1.0,
        },
    )
