"""Build cluster profiles from frozen evidence and authenticated source-token axes."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
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
    for row in frame.itertuples(index=False):
        if not bool(row.assigned):
            continue
        cluster_id = int(row.cluster_id)
        key = (int(row.layer), int(row.neuron_index))
        sign = 1 if row.polarity == "+" else -1
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


def _align_source_vector(values: Sequence[float], token_count: int) -> list[float]:
    aligned = list(values)
    if len(aligned) < token_count:
        aligned = [0.0] * (token_count - len(aligned)) + aligned
    elif len(aligned) > token_count:
        aligned = aligned[-token_count:]
    return aligned


def _frozen_fixed_union_profile(
    exemplar: Mapping[str, Any], *, token_count: int
) -> tuple[list[float], int] | None:
    summary = exemplar.get("fixed_union_input_summary")
    if summary is None:
        return None
    if token_count < 1:
        raise ValueError("authenticated source-token axis is empty")
    if not isinstance(summary, Mapping) or set(summary) != {
        "schema_version",
        "source",
        "representation",
        "member_basis_count",
        "member_occurrence_count",
        "signed_sum_by_source_token",
        "mean_by_member_occurrence",
        "support_occurrence_count_by_source_token",
    }:
        raise ValueError("malformed frozen fixed-union input summary")
    if (
        summary["schema_version"]
        != "adag.bonafide.fixed-union-input-summary.v1"
        or summary["source"] != "observed_candidate_fixed_union_refinement"
        or summary["representation"]
        not in {"raw_input_attribution", "paper_normalized_input_attribution"}
    ):
        raise ValueError("frozen fixed-union input summary identity drift")
    means = summary["mean_by_member_occurrence"]
    sums = summary["signed_sum_by_source_token"]
    counts = summary["support_occurrence_count_by_source_token"]
    if (
        not isinstance(means, list)
        or not isinstance(sums, list)
        or not isinstance(counts, list)
        or not means
        or len(sums) != len(means)
        or len(counts) != len(means)
    ):
        raise ValueError("malformed frozen fixed-union input vectors")
    if (
        isinstance(summary["member_basis_count"], bool)
        or not isinstance(summary["member_basis_count"], int)
        or isinstance(summary["member_occurrence_count"], bool)
        or not isinstance(summary["member_occurrence_count"], int)
    ):
        raise TypeError("malformed frozen fixed-union support totals")
    member_count = summary["member_basis_count"]
    occurrence_count = summary["member_occurrence_count"]
    if member_count <= 0 or occurrence_count <= 0:
        raise ValueError("frozen fixed-union input summary has no member support")
    values: list[float] = []
    any_support = False
    for mean, signed_sum, count_value in zip(means, sums, counts, strict=True):
        if isinstance(count_value, bool) or not isinstance(count_value, int):
            raise TypeError("malformed frozen fixed-union support count")
        if count_value < 0:
            raise ValueError("negative frozen fixed-union support count")
        if count_value == 0:
            if mean is not None or signed_sum is not None:
                raise ValueError("unsupported frozen fixed-union coordinate is not missing")
            values.append(0.0)
            continue
        any_support = True
        if mean is None or signed_sum is None:
            raise ValueError("supported frozen fixed-union coordinate is missing")
        if (
            isinstance(mean, bool)
            or not isinstance(mean, (int, float))
            or isinstance(signed_sum, bool)
            or not isinstance(signed_sum, (int, float))
        ):
            raise TypeError("invalid frozen fixed-union input value")
        value = float(mean)
        total = float(signed_sum)
        if not math.isfinite(value) or not math.isfinite(total):
            raise ValueError("nonfinite frozen fixed-union input value")
        if not math.isclose(value, total / count_value, rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError("frozen fixed-union input mean is inconsistent")
        values.append(value)
    if not any_support:
        raise ValueError("frozen fixed-union input summary has no coordinate support")
    if not any(value != 0.0 for value in values):
        raise ValueError("frozen fixed-union input summary is all zero")
    aligned = _align_source_vector(values, token_count)
    if not any(value != 0.0 for value in aligned):
        raise ValueError("aligned frozen fixed-union input summary is all zero")
    return aligned, member_count


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
    frozen = _frozen_fixed_union_profile(exemplar, token_count=len(token_ids))
    if frozen is not None:
        activations, matched_signed = frozen
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
        values = _align_source_vector(values, len(token_ids))
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
