"""Execution adapters for ordinary selected target logits.

The interface owns the choice between the historical causal-LM forward, which
materializes logits for every sequence position, and a selected-position
causal-LM forward that applies the unchanged LM head only to requested rows.
Callers receive the same ordered target-logit matrix and do not coordinate
hidden-state selection, vocabulary gathering, centering, or execution receipts
themselves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, cast

import torch
from torch import nn

from circuits.tracing.instrumentation import TraceInstrumentation

SelectedTargetLogitExecution = Literal[
    "full_logits_v1",
    "selected_position_logits_v1",
]
DEFAULT_SELECTED_TARGET_LOGIT_EXECUTION: SelectedTargetLogitExecution = "full_logits_v1"


@dataclass(frozen=True)
class SelectedTargetLogitResult:
    """Ordered target logits and fail-closed materialization receipts."""

    target_logits: torch.Tensor
    execution: SelectedTargetLogitExecution
    batch_size: int
    sequence_position_count: int
    selected_position_count: int
    unique_selected_position_count: int
    vocab_size: int
    lm_head_input_shape: tuple[int, int, int]
    lm_head_output_shape: tuple[int, int, int]
    selected_position_logit_shape: tuple[int, int, int]
    target_logit_shape: tuple[int, int]
    causal_lm_forward_completed: bool
    selected_position_request_forwarded: bool
    full_sequence_logits_materialized: bool
    selected_position_logits_materialized: bool
    center_logits: bool
    # Retain the same forward-output lifetime as the historical caller. The
    # candidate differs only in the LM-head position dimension held here.
    _retained_forward_output: object = field(repr=False, compare=False)
    _retained_lm_head_logits: torch.Tensor = field(repr=False, compare=False)


def resolve_selected_target_logit_execution(
    execution: str,
) -> SelectedTargetLogitExecution:
    """Validate and return a provenance-bearing target-logit strategy."""

    if execution not in {"full_logits_v1", "selected_position_logits_v1"}:
        raise ValueError(
            "invalid selected target-logit execution "
            f"{execution!r}; expected one of "
            "['full_logits_v1', 'selected_position_logits_v1']"
        )
    return cast(SelectedTargetLogitExecution, execution)


def validate_selected_target_logit_configuration(
    execution: str,
    *,
    center_logits: bool,
) -> SelectedTargetLogitExecution:
    """Validate cross-field constraints for the ordinary target-logit seam."""

    resolved = resolve_selected_target_logit_execution(execution)
    if resolved == "selected_position_logits_v1" and center_logits:
        raise ValueError(
            "selected-position target logits do not support center_logits=true"
        )
    return resolved


def _validated_coordinates(
    embeddings: torch.Tensor,
    focus_positions: list[int],
    focus_logits: list[list[int]],
) -> tuple[int, int, int]:
    if embeddings.ndim != 3:
        raise ValueError("selected target-logit embeddings must have shape (b, s, d)")
    batch_size, sequence_count, _hidden_size = map(int, embeddings.shape)
    if batch_size <= 0 or sequence_count <= 0:
        raise ValueError("selected target-logit embeddings must be nonempty")
    if not focus_positions:
        raise ValueError("selected target-logit positions must be nonempty")
    if any(
        type(position) is not int or not 0 <= position < sequence_count
        for position in focus_positions
    ):
        raise ValueError("selected target-logit position is outside the sequence")
    if len(focus_logits) != batch_size or any(
        not isinstance(row, list) or len(row) != len(focus_positions)
        for row in focus_logits
    ):
        raise ValueError(
            "selected target-logit token IDs must have shape (batch, positions)"
        )
    if any(
        type(token_id) is not int or token_id < 0
        for row in focus_logits
        for token_id in row
    ):
        raise ValueError("selected target-logit token IDs must be nonnegative integers")
    return batch_size, sequence_count, len(focus_positions)


def _record(
    instrumentation: TraceInstrumentation | None,
    result: SelectedTargetLogitResult,
    *,
    execution_index: int | None,
) -> None:
    if instrumentation is None:
        return
    instrumentation.increment_counter("selected_target_logit_execution_count")
    instrumentation.increment_counter(
        f"selected_target_logit_{result.execution}_execution_count"
    )
    instrumentation.increment_counter(
        "selected_target_logit_full_sequence_logits_materialized_count",
        int(result.full_sequence_logits_materialized),
    )
    instrumentation.increment_counter(
        "selected_target_logit_selected_position_logits_materialized_count",
        int(result.selected_position_logits_materialized),
    )
    instrumentation.increment_counter(
        "selected_target_logit_lm_head_position_rows",
        result.batch_size * result.lm_head_input_shape[1],
    )
    instrumentation.append_execution_record(
        "selected_target_logit_execution",
        execution=result.execution,
        execution_index=execution_index,
        batch_size=result.batch_size,
        sequence_position_count=result.sequence_position_count,
        selected_position_count=result.selected_position_count,
        unique_selected_position_count=result.unique_selected_position_count,
        vocab_size=result.vocab_size,
        lm_head_input_shape=result.lm_head_input_shape,
        lm_head_output_shape=result.lm_head_output_shape,
        selected_position_logit_shape=result.selected_position_logit_shape,
        target_logit_shape=result.target_logit_shape,
        causal_lm_forward_completed=result.causal_lm_forward_completed,
        selected_position_request_forwarded=(
            result.selected_position_request_forwarded
        ),
        full_sequence_logits_materialized=result.full_sequence_logits_materialized,
        selected_position_logits_materialized=(
            result.selected_position_logits_materialized
        ),
        center_logits=result.center_logits,
    )


def run_selected_target_logits(
    model: nn.Module,
    embeddings: torch.Tensor,
    attention_mask: torch.Tensor | list[list[int]] | None,
    focus_positions: list[int],
    focus_logits: list[list[int]],
    *,
    execution: SelectedTargetLogitExecution = DEFAULT_SELECTED_TARGET_LOGIT_EXECUTION,
    center_logits: bool = False,
    instrumentation: TraceInstrumentation | None = None,
    execution_index: int | None = None,
) -> SelectedTargetLogitResult:
    """Execute the requested logit path and return ordered ``(target, batch)`` values.

    Both adapters apply the model's own final decoder normalization and
    ``lm_head``.  The candidate changes only how many position rows enter the
    head and still materializes the full vocabulary for each requested row.
    Candidate centering is refused because the historical full-logit centering
    expression is not valid for ordinary ``(batch, sequence, vocabulary)`` shapes.
    """

    resolved = validate_selected_target_logit_configuration(
        execution,
        center_logits=center_logits,
    )
    batch_size, sequence_count, selected_count = _validated_coordinates(
        embeddings, focus_positions, focus_logits
    )
    lm_head = getattr(model, "lm_head", None)
    if not isinstance(lm_head, nn.Module):
        raise RuntimeError("selected target-logit execution requires model.lm_head")

    observed_lm_head_inputs: list[tuple[int, ...]] = []
    observed_lm_head_outputs: list[tuple[int, ...]] = []

    def observe_lm_head_input(_module, inputs) -> None:
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            raise RuntimeError("LM head did not receive a tensor input")
        observed_lm_head_inputs.append(tuple(map(int, inputs[0].shape)))

    def observe_lm_head_output(_module, _inputs, output) -> None:
        if not isinstance(output, torch.Tensor):
            raise RuntimeError("LM head did not return a tensor")
        observed_lm_head_outputs.append(tuple(map(int, output.shape)))

    input_handle = lm_head.register_forward_pre_hook(observe_lm_head_input)
    output_handle = lm_head.register_forward_hook(observe_lm_head_output)
    try:
        if resolved == "full_logits_v1":
            output = model(inputs_embeds=embeddings, attention_mask=attention_mask)
            full_logits = getattr(output, "logits", None)
            if not isinstance(full_logits, torch.Tensor) or full_logits.ndim != 3:
                raise RuntimeError(
                    "causal LM forward did not return logits with shape (b, s, v)"
                )
            if tuple(full_logits.shape[:2]) != (batch_size, sequence_count):
                raise RuntimeError(
                    "causal LM logits disagree with embedding batch/sequence shape"
                )
            # Preserve the historical control operation-for-operation. In
            # particular, the in-place centering expression is intentionally
            # retained even though it raises for ordinary (B, S, V) shapes.
            if center_logits:
                full_logits -= full_logits.mean(dim=-1)
            target_nodes = []
            for target_index, position in enumerate(focus_positions):
                token_ids = [row[target_index] for row in focus_logits]
                target_nodes.append(
                    full_logits[
                        torch.arange(full_logits.shape[0]),
                        position,
                        token_ids,
                    ]
                )
            target_logits = torch.stack(target_nodes)
            position_logits = full_logits
            full_sequence_logits_materialized = True
            selected_position_request_forwarded = False
        else:
            position_indices = torch.tensor(
                focus_positions,
                device=embeddings.device,
                dtype=torch.long,
            )
            try:
                output = model(
                    inputs_embeds=embeddings,
                    attention_mask=attention_mask,
                    logits_to_keep=position_indices,
                )
            except TypeError as error:
                raise RuntimeError(
                    "selected-position logits require a causal LM forward that "
                    "supports tensor logits_to_keep"
                ) from error
            position_logits = getattr(output, "logits", None)
            if not isinstance(position_logits, torch.Tensor):
                raise RuntimeError("causal LM forward did not return selected logits")
            target_nodes = []
            for target_index in range(selected_count):
                token_ids = [row[target_index] for row in focus_logits]
                target_nodes.append(
                    position_logits[
                        torch.arange(position_logits.shape[0]),
                        target_index,
                        token_ids,
                    ]
                )
            target_logits = torch.stack(target_nodes)
            full_sequence_logits_materialized = False
            selected_position_request_forwarded = True
    finally:
        output_handle.remove()
        input_handle.remove()

    if len(observed_lm_head_inputs) != 1 or len(observed_lm_head_outputs) != 1:
        raise RuntimeError("causal LM forward must execute its LM head exactly once")
    if len(observed_lm_head_inputs[0]) != 3 or len(observed_lm_head_outputs[0]) != 3:
        raise RuntimeError("LM head receipts must have shape (batch, positions, width)")
    lm_head_input_shape = cast(tuple[int, int, int], observed_lm_head_inputs[0])
    observed_lm_head_output_shape = cast(
        tuple[int, int, int], observed_lm_head_outputs[0]
    )
    expected_head_positions = (
        sequence_count if resolved == "full_logits_v1" else selected_count
    )
    if lm_head_input_shape[:2] != (batch_size, expected_head_positions):
        raise RuntimeError("LM head input receipt disagrees with requested strategy")
    if observed_lm_head_output_shape[:2] != (
        batch_size,
        expected_head_positions,
    ):
        raise RuntimeError("LM head output receipt disagrees with requested strategy")

    if position_logits.ndim != 3 or tuple(position_logits.shape[:2]) != (
        batch_size,
        expected_head_positions,
    ):
        raise RuntimeError("LM head output disagrees with requested execution shape")
    vocab_size = int(position_logits.shape[-1])
    if vocab_size <= 0 or any(
        token_id >= vocab_size for row in focus_logits for token_id in row
    ):
        raise ValueError("selected target-logit token ID is outside the vocabulary")
    result = SelectedTargetLogitResult(
        target_logits=target_logits,
        execution=resolved,
        batch_size=batch_size,
        sequence_position_count=sequence_count,
        selected_position_count=selected_count,
        unique_selected_position_count=len(set(focus_positions)),
        vocab_size=vocab_size,
        lm_head_input_shape=lm_head_input_shape,
        lm_head_output_shape=observed_lm_head_output_shape,
        selected_position_logit_shape=(batch_size, selected_count, vocab_size),
        target_logit_shape=tuple(map(int, target_logits.shape)),
        causal_lm_forward_completed=True,
        selected_position_request_forwarded=selected_position_request_forwarded,
        full_sequence_logits_materialized=full_sequence_logits_materialized,
        selected_position_logits_materialized=True,
        center_logits=bool(center_logits),
        _retained_forward_output=output,
        _retained_lm_head_logits=position_logits,
    )
    _record(instrumentation, result, execution_index=execution_index)
    return result
