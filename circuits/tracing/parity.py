"""Observed-token k=1 compatibility checks for candidate-axis tracing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from circuits.tracing.artifact import validate_topk_trace_data
from circuits.tracing.trace import CircuitData, TopKPositionTrace


@dataclass(frozen=True)
class K1ParityReport:
    """Canonical comparison result for one legacy/candidate trace pair."""

    passed: bool
    mismatches: tuple[str, ...]
    node_count: int
    edge_count: int
    absolute_tolerance: float
    relative_tolerance: float

    def require_pass(self) -> None:
        if not self.passed:
            raise AssertionError(
                "observed-token k=1 parity failed: " + "; ".join(self.mismatches)
            )


def _canonical_frame(
    frame: pd.DataFrame, *, identity_columns: tuple[str, ...]
) -> pd.DataFrame:
    missing = [column for column in identity_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"parity frame is missing identity columns: {missing}")
    ordered_columns = [*identity_columns]
    ordered_columns.extend(
        sorted(column for column in frame.columns if column not in identity_columns)
    )
    return (
        frame.loc[:, ordered_columns]
        .sort_values(list(identity_columns), kind="stable")
        .reset_index(drop=True)
    )


def _without_timing(value: Any) -> Any:
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, item in sorted(value.items()):
            if not isinstance(key, str):
                raise TypeError("instrumentation keys must be strings")
            if key.endswith("_seconds") or key in {"stages", "candidate_count"}:
                continue
            normalized[key] = _without_timing(item)
        return normalized
    if isinstance(value, list):
        return [_without_timing(item) for item in value]
    return value


def _append_exact_mismatch(
    mismatches: list[str],
    field_name: str,
    legacy_value: object,
    candidate_value: object,
) -> None:
    if legacy_value != candidate_value:
        mismatches.append(f"{field_name} mismatch")


def _structural_target_provenance(
    value: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Exclude floating scores already compared with numeric tolerances."""

    return [
        {
            key: item
            for key, item in provenance.items()
            if key not in {"logit", "probability"}
        }
        for provenance in value
    ]


def _compare_frame(
    mismatches: list[str],
    name: str,
    legacy: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    identity_columns: tuple[str, ...],
    absolute_tolerance: float,
    relative_tolerance: float,
) -> None:
    try:
        pd.testing.assert_frame_equal(
            _canonical_frame(legacy, identity_columns=identity_columns),
            _canonical_frame(candidate, identity_columns=identity_columns),
            check_exact=False,
            atol=absolute_tolerance,
            rtol=relative_tolerance,
            check_like=False,
        )
    except AssertionError as error:
        first_line = str(error).splitlines()[0] if str(error) else "frame differs"
        mismatches.append(f"{name} mismatch: {first_line}")


def compare_observed_token_k1(
    legacy: CircuitData,
    candidate: TopKPositionTrace,
    *,
    absolute_tolerance: float = 1e-6,
    relative_tolerance: float = 1e-5,
) -> K1ParityReport:
    """Compare the new observed-token k=1 path with a width-one trace."""

    if candidate.candidate_count != 1:
        raise ValueError("k=1 parity requires exactly one candidate")
    if candidate.candidate_selection.policy_id != "observed_token":
        raise ValueError("k=1 parity requires the observed_token candidate policy")
    if candidate.joint_objective.objective_id != "raw_logit_sum":
        raise ValueError("k=1 parity requires the raw_logit_sum objective")
    if candidate.joint_objective.candidate_weights != (1.0,):
        raise ValueError("k=1 parity requires a unit candidate objective weight")
    validate_topk_trace_data(candidate)

    current = candidate.circuit_data
    mismatches: list[str] = []
    _append_exact_mismatch(mismatches, "model_id", legacy.model_id, current.model_id)
    _append_exact_mismatch(mismatches, "inputs", legacy.cis, current.cis)
    _append_exact_mismatch(
        mismatches,
        "attention_masks",
        legacy.attention_masks,
        current.attention_masks,
    )
    _append_exact_mismatch(mismatches, "labels", legacy.labels, current.labels)
    _append_exact_mismatch(
        mismatches, "target_logits", legacy.target_logits, current.target_logits
    )
    _append_exact_mismatch(
        mismatches,
        "target_provenance",
        _structural_target_provenance(legacy.target_provenance),
        _structural_target_provenance(current.target_provenance),
    )
    _append_exact_mismatch(mismatches, "config", legacy.config, current.config)

    for field_name in ("target_logit_values", "target_logit_probs"):
        legacy_values = getattr(legacy, field_name)
        candidate_values = getattr(current, field_name)
        try:
            pd.testing.assert_series_equal(
                pd.Series(legacy_values[0], dtype=float),
                pd.Series(candidate_values[0], dtype=float),
                check_exact=False,
                atol=absolute_tolerance,
                rtol=relative_tolerance,
            )
        except (AssertionError, IndexError):
            mismatches.append(f"{field_name} mismatch")

    _compare_frame(
        mismatches,
        "node table",
        legacy.df_node,
        current.df_node,
        identity_columns=("layer", "token", "neuron", "label"),
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )
    _compare_frame(
        mismatches,
        "edge table",
        legacy.df_edge,
        current.df_edge,
        identity_columns=("layer", "token", "neuron", "label"),
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )

    legacy_instrumentation = legacy.trace_metadata.get("instrumentation")
    candidate_instrumentation = current.trace_metadata.get("instrumentation")
    if (legacy_instrumentation is None) != (candidate_instrumentation is None):
        mismatches.append("instrumentation presence mismatch")
    elif legacy_instrumentation is not None and _without_timing(
        legacy_instrumentation
    ) != _without_timing(candidate_instrumentation):
        mismatches.append("non-timing instrumentation mismatch")

    return K1ParityReport(
        passed=not mismatches,
        mismatches=tuple(mismatches),
        node_count=len(legacy.df_node),
        edge_count=len(legacy.df_edge),
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )
