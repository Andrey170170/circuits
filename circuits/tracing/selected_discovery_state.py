"""Post-selection storage adapters for discovery tensors.

Selection math ends before this seam.  Callers consume the same ordered
occurrence interface regardless of whether the historical dense discovery
buffers remain live or only exact selected values are retained on the CPU.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, cast

import torch

from circuits.tracing.instrumentation import TraceInstrumentation
from circuits.tracing.tensor_receipts import raw_tensor_sha256

PostSelectionStateStorage = Literal["dense_v1", "compact_cpu_v1"]
DEFAULT_POST_SELECTION_STATE_STORAGE: PostSelectionStateStorage = "dense_v1"


@dataclass(frozen=True)
class SelectedDiscoveryOccurrence:
    """One selected raw-MLP occurrence in canonical ``neuron_cfg`` order."""

    layer: int
    token: int
    neuron: int
    final_attribution: torch.Tensor


@dataclass(frozen=True)
class SelectedDiscoveryState:
    """Strategy-independent selected state consumed by graph expansion."""

    ordered_occurrences: tuple[SelectedDiscoveryOccurrence, ...]
    active_layers: tuple[int, ...]
    logical_input_bytes: int
    logical_retained_bytes: int
    logical_released_bytes: int
    _occurrences_by_layer: Mapping[int, tuple[SelectedDiscoveryOccurrence, ...]] = (
        field(default_factory=lambda: MappingProxyType({}))
    )
    _dense_lifetime_buffers: tuple[torch.Tensor, ...] = ()
    _compact_values: torch.Tensor | None = None

    def occurrences_for_layer(
        self, layer: int
    ) -> tuple[SelectedDiscoveryOccurrence, ...]:
        """Return canonical selected occurrences for one layer."""

        return self._occurrences_by_layer.get(layer, ())


def resolve_post_selection_state_storage(
    strategy: str,
) -> PostSelectionStateStorage:
    """Validate and return a provenance-bearing discovery-state strategy."""

    if strategy not in {"dense_v1", "compact_cpu_v1"}:
        raise ValueError(
            "invalid post-selection state storage "
            f"{strategy!r}; expected one of ['dense_v1', 'compact_cpu_v1']"
        )
    return cast(PostSelectionStateStorage, strategy)


def _logical_tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _ordered_coordinates(
    neuron_cfg: dict[int, list[list[int]]],
    *,
    layer_count: int,
    token_count: int,
    neuron_count: int,
) -> tuple[tuple[int, int, int], ...]:
    coordinates: list[tuple[int, int, int]] = []
    for layer in neuron_cfg:
        if type(layer) is not int:
            raise ValueError("selected discovery layer coordinate must be an integer")
    for layer in sorted(neuron_cfg):
        if not 0 <= layer < layer_count:
            raise ValueError("selected discovery layer coordinate is out of bounds")
        positions = neuron_cfg[layer]
        if not isinstance(positions, (list, tuple)):
            raise ValueError("selected discovery layer positions must be a sequence")
        for position in positions:
            if not isinstance(position, (list, tuple)) or len(position) != 2:
                raise ValueError(
                    "selected discovery positions must contain token and neuron"
                )
            token, neuron = position
            if type(token) is not int:
                raise ValueError(
                    "selected discovery token coordinate must be an integer"
                )
            if type(neuron) is not int:
                raise ValueError(
                    "selected discovery neuron coordinate must be an integer"
                )
            if not 0 <= token < token_count:
                raise ValueError("selected discovery token coordinate is out of bounds")
            if not 0 <= neuron < neuron_count:
                raise ValueError(
                    "selected discovery neuron coordinate is out of bounds"
                )
            coordinates.append((layer, token, neuron))
    if len(coordinates) != len(set(coordinates)):
        raise ValueError("selected discovery coordinates must be unique")
    return tuple(coordinates)


def _coordinate_tensor(
    coordinates: tuple[tuple[int, int, int], ...],
) -> torch.Tensor:
    if not coordinates:
        return torch.empty((0, 3), dtype=torch.int64)
    return torch.tensor(coordinates, dtype=torch.int64)


def _gather_selected_values(
    mlp_final_attributions: torch.Tensor,
    coordinates: tuple[tuple[int, int, int], ...],
) -> torch.Tensor:
    batch_size = mlp_final_attributions.shape[1]
    target_width = mlp_final_attributions.shape[4]
    if not coordinates:
        return torch.empty(
            (0, batch_size, target_width),
            dtype=mlp_final_attributions.dtype,
            device="cpu",
        )
    device = mlp_final_attributions.device
    layer_indices = torch.tensor(
        [coordinate[0] for coordinate in coordinates],
        dtype=torch.long,
        device=device,
    )
    token_indices = torch.tensor(
        [coordinate[1] for coordinate in coordinates],
        dtype=torch.long,
        device=device,
    )
    neuron_indices = torch.tensor(
        [coordinate[2] for coordinate in coordinates],
        dtype=torch.long,
        device=device,
    )
    layer_token_neuron_batch_target = mlp_final_attributions.permute(0, 2, 3, 1, 4)
    return (
        layer_token_neuron_batch_target[layer_indices, token_indices, neuron_indices]
        .detach()
        .to(device="cpu")
        .contiguous()
    )


def store_selected_discovery_state(
    mlp_final_attributions: torch.Tensor,
    global_important_neurons_mask: torch.Tensor,
    mlp_final_acts: torch.Tensor,
    embed_final_acts: torch.Tensor,
    *,
    neuron_cfg: dict[int, list[list[int]]],
    strategy: PostSelectionStateStorage = DEFAULT_POST_SELECTION_STATE_STORAGE,
    instrumentation: TraceInstrumentation | None = None,
) -> SelectedDiscoveryState:
    """Store selected state after probe and return-only exits have completed."""

    resolved = resolve_post_selection_state_storage(strategy)
    if mlp_final_attributions.ndim != 5:
        raise ValueError(
            "MLP final attributions must have shape (layer, batch, token, neuron, target)"
        )
    if global_important_neurons_mask.ndim != 3:
        raise ValueError("important-neuron mask must have shape (layer, token, neuron)")
    if tuple(global_important_neurons_mask.shape) != (
        mlp_final_attributions.shape[0],
        mlp_final_attributions.shape[2],
        mlp_final_attributions.shape[3],
    ):
        raise ValueError("important-neuron mask shape disagrees with MLP attributions")

    coordinates = _ordered_coordinates(
        neuron_cfg,
        layer_count=mlp_final_attributions.shape[0],
        token_count=mlp_final_attributions.shape[2],
        neuron_count=mlp_final_attributions.shape[3],
    )
    configured_coordinates_by_layer = {
        layer: {
            (token, neuron)
            for item_layer, token, neuron in coordinates
            if item_layer == layer
        }
        for layer in neuron_cfg
    }
    for layer, configured_coordinates in configured_coordinates_by_layer.items():
        mask_coordinates = {
            (int(token), int(neuron))
            for token, neuron in global_important_neurons_mask[layer]
            .nonzero(as_tuple=False)
            .tolist()
        }
        if mask_coordinates != configured_coordinates:
            raise ValueError(
                "neuron_cfg coordinates must exactly match the important-neuron mask "
                "within configured layers"
            )

    gather_receipts = instrumentation is not None
    selected_values = (
        _gather_selected_values(mlp_final_attributions, coordinates)
        if resolved == "compact_cpu_v1" or gather_receipts
        else None
    )
    coordinates_tensor = (
        _coordinate_tensor(coordinates)
        if resolved == "compact_cpu_v1" or gather_receipts
        else None
    )
    active_layers = tuple(
        layer for layer in sorted(neuron_cfg) if bool(neuron_cfg[layer])
    )

    dense_inputs = (
        mlp_final_attributions,
        global_important_neurons_mask,
        mlp_final_acts,
        embed_final_acts,
    )
    logical_input_bytes = sum(_logical_tensor_bytes(tensor) for tensor in dense_inputs)
    if resolved == "dense_v1":
        occurrences = tuple(
            SelectedDiscoveryOccurrence(
                layer=layer,
                token=token,
                neuron=neuron,
                final_attribution=mlp_final_attributions[layer, :, token, neuron, :],
            )
            for layer, token, neuron in coordinates
        )
        logical_retained_bytes = logical_input_bytes
        retains_dense = True
    else:
        if selected_values is None or coordinates_tensor is None:
            raise RuntimeError("compact selected discovery state lacks compact tensors")
        occurrences = tuple(
            SelectedDiscoveryOccurrence(
                layer=layer,
                token=token,
                neuron=neuron,
                final_attribution=selected_values[index],
            )
            for index, (layer, token, neuron) in enumerate(coordinates)
        )
        # Coordinates are retained by the occurrence interface itself. The
        # temporary int64 tensor exists only to produce an exact receipt, so
        # it is not part of the state's retained tensor-byte count.
        logical_retained_bytes = _logical_tensor_bytes(selected_values)
        retains_dense = False

    occurrences_by_layer = MappingProxyType(
        {
            layer: tuple(
                occurrence for occurrence in occurrences if occurrence.layer == layer
            )
            for layer in active_layers
        }
    )
    state = SelectedDiscoveryState(
        ordered_occurrences=occurrences,
        active_layers=active_layers,
        logical_input_bytes=logical_input_bytes,
        logical_retained_bytes=logical_retained_bytes,
        logical_released_bytes=logical_input_bytes - logical_retained_bytes,
        _occurrences_by_layer=occurrences_by_layer,
        _dense_lifetime_buffers=dense_inputs if retains_dense else (),
        _compact_values=selected_values if not retains_dense else None,
    )

    if instrumentation is not None:
        if selected_values is None or coordinates_tensor is None:
            raise RuntimeError("selected discovery instrumentation lacks exact tensors")
        selected_values_bytes = _logical_tensor_bytes(selected_values)
        selected_coordinates_bytes = _logical_tensor_bytes(coordinates_tensor)
        instrumentation.set_counter("post_selection_state_storage", resolved)
        instrumentation.increment_counter(
            "post_selection_state_storage_execution_count"
        )
        instrumentation.set_counter(
            "post_selection_state_selected_occurrence_count", len(coordinates)
        )
        instrumentation.append_execution_record(
            "post_selection_state_storage",
            strategy=resolved,
            selected_occurrence_count=len(coordinates),
            active_layers=list(active_layers),
            selected_values_shape=list(selected_values.shape),
            selected_values_dtype=str(selected_values.dtype),
            selected_values_bytes=selected_values_bytes,
            selected_values_raw_sha256=raw_tensor_sha256(selected_values),
            selected_coordinates_shape=list(coordinates_tensor.shape),
            selected_coordinates_dtype=str(coordinates_tensor.dtype),
            selected_coordinates_bytes=selected_coordinates_bytes,
            selected_coordinates_raw_sha256=raw_tensor_sha256(coordinates_tensor),
            logical_input_bytes=logical_input_bytes,
            logical_retained_bytes=logical_retained_bytes,
            logical_released_bytes=state.logical_released_bytes,
            retains_dense_mlp_final_attributions=retains_dense,
            retains_dense_important_neuron_mask=retains_dense,
            retains_unused_mlp_final_acts=retains_dense,
            retains_unused_embed_final_acts=retains_dense,
            state_values_device=str(
                mlp_final_attributions.device
                if retains_dense
                else selected_values.device
            ),
        )
    return state


__all__ = [
    "DEFAULT_POST_SELECTION_STATE_STORAGE",
    "PostSelectionStateStorage",
    "SelectedDiscoveryOccurrence",
    "SelectedDiscoveryState",
    "resolve_post_selection_state_storage",
    "store_selected_discovery_state",
]
