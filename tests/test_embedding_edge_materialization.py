"""Focused CPU tests for embedding-edge materialization strategies."""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import cast

import pytest
import torch
from circuits.tracing.clja import ADAGConfig
from circuits.tracing.embedding_edge_materialization import (
    EmbeddingEdgeMaterialization,
    EmbeddingEdgeMaterializationRequest,
    EmbeddingSource,
    EmbeddingTarget,
    materialize_embedding_edges,
    resolve_embedding_edge_materialization,
)
from circuits.tracing.utils import Edge, NeuronIdx


def _request() -> EmbeddingEdgeMaterializationRequest:
    sources = [
        EmbeddingSource(
            NeuronIdx(-1, 0, 11),
            torch.tensor([True, False, True]),
        ),
        EmbeddingSource(
            NeuronIdx(-1, 0, 17),
            torch.tensor([False, True, False]),
        ),
        EmbeddingSource(
            NeuronIdx(-1, 2, 23),
            torch.tensor([True, True, False]),
        ),
    ]
    targets = [
        EmbeddingTarget(
            key=NeuronIdx(0, 1, 5),
            attribution_by_source=torch.tensor(
                [
                    [2.0, 9.0, 0.25],
                    [1.0, 8.0, -2.0],
                    [-3.0, 7.0, 4.0],
                ]
            ),
            activation=torch.tensor([2.0, -4.0, 1.0]),
            final_attribution=torch.tensor(
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                    [5.0, 6.0],
                ]
            ),
        ),
        EmbeddingTarget(
            key=NeuronIdx(1, 2, 7),
            attribution_by_source=torch.tensor(
                [
                    [-0.5, 4.0, 2.0],
                    [3.0, 5.0, 1.0],
                    [0.25, 6.0, -8.0],
                ]
            ),
            activation=torch.tensor([1.0, 2.0, -4.0]),
            final_attribution=None,
        ),
    ]
    return EmbeddingEdgeMaterializationRequest(
        sources=sources,
        targets=targets,
        device="cpu",
    )


def _materialize_both(
    request: EmbeddingEdgeMaterializationRequest,
) -> tuple[list[Edge], list[Edge]]:
    return (
        materialize_embedding_edges("scalar_v1", request),
        materialize_embedding_edges("vectorized_v1", request),
    )


def _assert_edges_exact(actual: list[Edge], expected: list[Edge]) -> None:
    assert [(edge.src, edge.tgt) for edge in actual] == [
        (edge.src, edge.tgt) for edge in expected
    ]
    for actual_edge, expected_edge in zip(actual, expected, strict=True):
        torch.testing.assert_close(
            actual_edge.weight,
            expected_edge.weight,
            atol=0,
            rtol=0,
            equal_nan=True,
        )
        if expected_edge.final_attribution is None:
            assert actual_edge.final_attribution is None
        else:
            torch.testing.assert_close(
                actual_edge.final_attribution,
                expected_edge.final_attribution,
                atol=0,
                rtol=0,
                equal_nan=True,
            )


@pytest.mark.parametrize(
    ("edge_threshold", "parent_threshold"),
    [
        (None, None),
        (0.4, None),
        (None, 0.4),
        (0.2, 0.5),
    ],
)
def test_vectorized_matches_scalar_thresholds_masks_order_and_missing_contributions(
    edge_threshold: float | None,
    parent_threshold: float | None,
) -> None:
    request = replace(
        _request(),
        edge_threshold=edge_threshold,
        parent_threshold=parent_threshold,
    )

    scalar, vectorized = _materialize_both(request)

    _assert_edges_exact(vectorized, scalar)
    order = [(edge.src, edge.tgt) for edge in vectorized]
    assert order == sorted(
        order,
        key=lambda pair: (
            list(request.sources).index(
                next(source for source in request.sources if source.key == pair[0])
            ),
            list(request.targets).index(
                next(target for target in request.targets if target.key == pair[1])
            ),
        ),
    )
    for edge in vectorized:
        source = next(source for source in request.sources if source.key == edge.src)
        masked_weight = edge.weight[~source.batch_mask]
        assert torch.equal(masked_weight, torch.zeros_like(masked_weight))
        if edge.tgt == request.targets[1].key:
            assert edge.final_attribution is None


def test_vectorized_matches_scalar_objective_weight_reduction() -> None:
    request = replace(_request(), objective_weights=(1.0, -0.25))

    scalar, vectorized = _materialize_both(request)

    _assert_edges_exact(vectorized, scalar)
    first = vectorized[0]
    reduced = torch.tensor([[0.5], [2.0], [3.5]])
    torch.testing.assert_close(
        first.final_attribution,
        first.weight[:, None] * reduced,
        atol=0,
        rtol=0,
    )


def test_materialization_preserves_precast_objective_weight_values() -> None:
    base = _request()
    request = replace(
        base,
        targets=[
            replace(
                base.targets[0],
                final_attribution=base.targets[0].final_attribution.float(),
            )
        ],
        objective_weights=torch.tensor([1.001, 0.0], dtype=torch.bfloat16),
    )

    scalar, vectorized = _materialize_both(request)

    _assert_edges_exact(vectorized, scalar)
    expected_reduction = base.targets[0].final_attribution[:, :1]
    torch.testing.assert_close(
        vectorized[0].final_attribution,
        vectorized[0].weight[:, None] * expected_reduction,
        atol=0,
        rtol=0,
    )


def test_frozen_edges_override_thresholds_and_preserve_order() -> None:
    base = _request()
    frozen = frozenset(
        {
            (base.sources[0].key, base.targets[1].key),
            (base.sources[2].key, base.targets[0].key),
        }
    )
    request = replace(
        base,
        edge_threshold=1e9,
        parent_threshold=1e9,
        frozen_edges=frozen,
    )

    scalar, vectorized = _materialize_both(request)

    _assert_edges_exact(vectorized, scalar)
    assert [(edge.src, edge.tgt) for edge in vectorized] == [
        (base.sources[0].key, base.targets[1].key),
        (base.sources[2].key, base.targets[0].key),
    ]


def test_strict_threshold_keeps_equality_and_nan() -> None:
    target = EmbeddingTarget(
        key=NeuronIdx(0, 0, 3),
        attribution_by_source=torch.tensor([[1.0, float("nan")]]),
        activation=torch.tensor([1.0]),
        final_attribution=None,
    )
    request = EmbeddingEdgeMaterializationRequest(
        sources=[
            EmbeddingSource(NeuronIdx(-1, 0, 1), torch.tensor([True])),
            EmbeddingSource(NeuronIdx(-1, 1, 2), torch.tensor([True])),
        ],
        targets=[target],
        device="cpu",
        edge_threshold=1.0 / 1.000001,
    )

    scalar, vectorized = _materialize_both(request)

    _assert_edges_exact(vectorized, scalar)
    assert len(vectorized) == 2


def test_vectorized_threshold_comparison_preserves_bfloat16_scalar_cast() -> None:
    request = EmbeddingEdgeMaterializationRequest(
        sources=[
            EmbeddingSource(NeuronIdx(-1, 0, 1), torch.tensor([True])),
        ],
        targets=[
            EmbeddingTarget(
                key=NeuronIdx(0, 0, 3),
                attribution_by_source=torch.tensor([[1.0]], dtype=torch.bfloat16),
                activation=torch.tensor([1.0], dtype=torch.bfloat16),
                final_attribution=None,
            )
        ],
        device="cpu",
        # This rounds to 1.0 in bfloat16, so the historical strict comparison
        # sees equality and retains the edge.
        edge_threshold=1.001,
    )

    scalar, vectorized = _materialize_both(request)

    _assert_edges_exact(vectorized, scalar)
    assert len(vectorized) == 1


def test_empty_targets_and_return_nodes_only_have_no_edges() -> None:
    request = _request()
    empty = replace(request, targets=[])
    nodes_only = replace(request, return_nodes_only=True)

    for execution in ("scalar_v1", "vectorized_v1"):
        assert materialize_embedding_edges(execution, empty) == []
        assert materialize_embedding_edges(execution, nodes_only) == []


def test_config_validates_strategy_and_restores_legacy_state() -> None:
    config = ADAGConfig(device="cpu", embedding_edge_materialization="vectorized_v1")
    assert asdict(config)["embedding_edge_materialization"] == "vectorized_v1"
    assert resolve_embedding_edge_materialization("scalar_v1") == "scalar_v1"

    with pytest.raises(ValueError, match="invalid embedding edge materialization"):
        resolve_embedding_edge_materialization("auto")
    with pytest.raises(ValueError, match="invalid embedding edge materialization"):
        ADAGConfig(
            embedding_edge_materialization=cast(
                EmbeddingEdgeMaterialization,
                "auto",
            )
        )

    restored = ADAGConfig.__new__(ADAGConfig)
    restored.__setstate__({"device": "cpu"})
    assert restored.embedding_edge_materialization == "scalar_v1"


def test_vectorized_preserves_exact_duplicate_sources() -> None:
    base = _request()
    duplicate = base.sources[0]
    request = replace(
        base,
        sources=[duplicate, duplicate],
        targets=[base.targets[0]],
    )

    scalar, vectorized = _materialize_both(request)

    _assert_edges_exact(vectorized, scalar)
    assert len(vectorized) == 2
    assert vectorized[0].weight.data_ptr() != vectorized[1].weight.data_ptr()
