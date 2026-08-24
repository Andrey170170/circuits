"""Execution strategies for materializing embedding-to-MLP graph edges.

The public seam accepts already ordered embedding sources and MLP targets.  The
scalar adapter preserves the historical nested-loop implementation.  The
vectorized adapter prepares each target once, evaluates every ordered source in
one tensor operation, then buckets retained edges back into historical
source-major, target-major order.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, cast

import torch

from circuits.tracing.candidates import reduce_candidate_contributions
from circuits.tracing.utils import Edge, NeuronIdx

EmbeddingEdgeMaterialization = Literal["scalar_v1", "vectorized_v1"]
DEFAULT_EMBEDDING_EDGE_MATERIALIZATION: EmbeddingEdgeMaterialization = "scalar_v1"


@dataclass(frozen=True)
class EmbeddingSource:
    """One embedding node in exact graph-output order."""

    key: NeuronIdx
    batch_mask: torch.Tensor


@dataclass(frozen=True)
class EmbeddingTarget:
    """One MLP target in exact attribution-map insertion order."""

    key: NeuronIdx
    attribution_by_source: torch.Tensor
    activation: torch.Tensor
    final_attribution: torch.Tensor | None


@dataclass(frozen=True)
class EmbeddingEdgeMaterializationRequest:
    """All facts needed to materialize embedding edges without model access."""

    sources: Sequence[EmbeddingSource]
    targets: Sequence[EmbeddingTarget]
    device: str | torch.device
    edge_threshold: float | None = None
    parent_threshold: float | None = None
    objective_weights: Sequence[float] | torch.Tensor | None = None
    frozen_edges: frozenset[tuple[NeuronIdx, NeuronIdx]] | None = None
    return_nodes_only: bool = False


@dataclass(frozen=True)
class _EmbeddingEdgeMaterializationAdapter:
    execution: EmbeddingEdgeMaterialization
    materialize: Callable[[EmbeddingEdgeMaterializationRequest], list[Edge]]


def resolve_embedding_edge_materialization(
    execution: str,
) -> EmbeddingEdgeMaterialization:
    """Validate and return a named embedding-edge execution strategy."""

    if execution not in _ADAPTERS:
        choices = ", ".join(sorted(_ADAPTERS))
        raise ValueError(
            f"invalid embedding edge materialization {execution!r}; "
            f"expected one of: {choices}"
        )
    return cast(EmbeddingEdgeMaterialization, execution)


def _target_final_attribution(
    target: EmbeddingTarget,
    request: EmbeddingEdgeMaterializationRequest,
) -> torch.Tensor | None:
    final_attribution = target.final_attribution
    if final_attribution is None:
        return None
    if final_attribution.device != torch.device(request.device):
        final_attribution = final_attribution.to(request.device)
    if request.objective_weights is not None:
        final_attribution = reduce_candidate_contributions(
            final_attribution,
            request.objective_weights,
        )
    return final_attribution


def _retain_edge(
    request: EmbeddingEdgeMaterializationRequest,
    source: NeuronIdx,
    target: NeuronIdx,
    weight: torch.Tensor,
) -> bool:
    if request.frozen_edges is not None:
        return (source, target) in request.frozen_edges
    if (
        request.edge_threshold is not None
        and weight.abs().max() < request.edge_threshold
    ):
        return False
    return not (
        request.parent_threshold is not None
        and weight.abs().max() < request.parent_threshold
    )


def _materialize_scalar_v1(
    request: EmbeddingEdgeMaterializationRequest,
) -> list[Edge]:
    if request.return_nodes_only:
        return []

    edges: list[Edge] = []
    device = torch.device(request.device)
    for source in request.sources:
        mask = source.batch_mask.to(device=device, dtype=torch.bool)
        for target in request.targets:
            target_attr = target.attribution_by_source
            if target_attr.device != device:
                target_attr = target_attr.to(device)
            target_activation = target.activation
            if target_activation.device != device:
                target_activation = target_activation.to(device)
            final_attribution = _target_final_attribution(target, request)

            eps = target_activation.abs().mean() * 1e-6
            edge_weight = target_attr[:, source.key.token] / (target_activation + eps)
            edge_weight = torch.where(mask, edge_weight, 0)
            if not _retain_edge(request, source.key, target.key, edge_weight):
                continue

            edges.append(
                Edge(
                    src=source.key,
                    tgt=target.key,
                    weight=edge_weight.detach().float().cpu(),
                    final_attribution=(
                        (edge_weight[:, None] * final_attribution)
                        .detach()
                        .float()
                        .cpu()
                        if final_attribution is not None
                        else None
                    ),
                )
            )
    return edges


def _retained_source_indices(
    request: EmbeddingEdgeMaterializationRequest,
    target: EmbeddingTarget,
    edge_weights: torch.Tensor,
) -> list[int]:
    if request.frozen_edges is not None:
        return [
            index
            for index, source in enumerate(request.sources)
            if (source.key, target.key) in request.frozen_edges
        ]
    if request.edge_threshold is None and request.parent_threshold is None:
        return list(range(len(request.sources)))

    maxima = edge_weights.abs().amax(dim=1)
    retained = torch.ones_like(maxima, dtype=torch.bool)
    # Keep comparisons on the original tensor dtype so Python thresholds are
    # cast exactly as they were in the scalar implementation. Equality and NaN
    # therefore survive both strict-less-than tests.
    if request.edge_threshold is not None:
        retained &= ~(maxima < request.edge_threshold)
    if request.parent_threshold is not None:
        retained &= ~(maxima < request.parent_threshold)
    return retained.nonzero(as_tuple=False).flatten().cpu().tolist()


def _materialize_vectorized_v1(
    request: EmbeddingEdgeMaterializationRequest,
) -> list[Edge]:
    if request.return_nodes_only or not request.sources or not request.targets:
        return []

    device = torch.device(request.device)
    source_masks = torch.stack(
        [
            source.batch_mask.to(device=device, dtype=torch.bool)
            for source in request.sources
        ]
    )
    source_tokens = torch.tensor(
        [source.key.token for source in request.sources],
        device=device,
        dtype=torch.long,
    )
    edges_by_source: list[list[Edge]] = [[] for _ in request.sources]

    for target in request.targets:
        target_attr = target.attribution_by_source
        if target_attr.device != device:
            target_attr = target_attr.to(device)
        target_activation = target.activation
        if target_activation.device != device:
            target_activation = target_activation.to(device)
        final_attribution = _target_final_attribution(target, request)

        eps = target_activation.abs().mean() * 1e-6
        edge_weights = target_attr.index_select(1, source_tokens).movedim(0, 1)
        edge_weights = edge_weights / (target_activation + eps)
        edge_weights = torch.where(source_masks, edge_weights, 0)
        retained_indices = _retained_source_indices(request, target, edge_weights)
        if not retained_indices:
            continue

        retained_index_tensor = torch.tensor(
            retained_indices, device=device, dtype=torch.long
        )
        retained_weights = (
            edge_weights.index_select(0, retained_index_tensor).detach().float().cpu()
        )
        retained_final_attributions = (
            (
                edge_weights.index_select(0, retained_index_tensor)[:, :, None]
                * final_attribution[None, :, :]
            )
            .detach()
            .float()
            .cpu()
            if final_attribution is not None
            else None
        )

        for retained_offset, source_index in enumerate(retained_indices):
            edges_by_source[source_index].append(
                Edge(
                    src=request.sources[source_index].key,
                    tgt=target.key,
                    weight=retained_weights[retained_offset].clone(),
                    final_attribution=(
                        retained_final_attributions[retained_offset].clone()
                        if retained_final_attributions is not None
                        else None
                    ),
                )
            )

    return [edge for source_edges in edges_by_source for edge in source_edges]


_ADAPTERS: dict[EmbeddingEdgeMaterialization, _EmbeddingEdgeMaterializationAdapter] = {
    "scalar_v1": _EmbeddingEdgeMaterializationAdapter(
        execution="scalar_v1",
        materialize=_materialize_scalar_v1,
    ),
    "vectorized_v1": _EmbeddingEdgeMaterializationAdapter(
        execution="vectorized_v1",
        materialize=_materialize_vectorized_v1,
    ),
}


def materialize_embedding_edges(
    execution: EmbeddingEdgeMaterialization,
    request: EmbeddingEdgeMaterializationRequest,
) -> list[Edge]:
    """Materialize embedding edges through the selected execution adapter."""

    resolved = resolve_embedding_edge_materialization(execution)
    return _ADAPTERS[resolved].materialize(request)
