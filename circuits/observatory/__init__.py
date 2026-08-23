"""Lossless, local-first projection and serving for compact ADAG traces.

The package deliberately keeps its import surface light.  In particular, importing
``circuits.observatory`` or running the ``serve`` command does not import the tracing
stack, pickle, a tokenizer, or a model.  Trusted compact artifacts are opened only by
the explicit ``sync`` command.
"""

from __future__ import annotations

CATALOG_SCHEMA = "adag.observatory.catalog.v1"
LABEL_SET_SCHEMA = "adag.observatory.label-set.v1"
MANIFEST_SCHEMA = "adag.observatory.bundle.v1"
TRACE_GRAPH_SCHEMA = "adag.observatory.trace-graph.v1"
WORKSPACE_SCHEMA = "adag.observatory.workspace.v1"

CLAIM_BOUNDARY = (
    "Each graph is a pruned, locally approximate attribution subgraph for one "
    "selected logit, not a complete transcript of model computation. Labels and "
    "clusters are exploratory overlays, not causal or faithfulness evidence."
)

__all__ = [
    "CATALOG_SCHEMA",
    "CLAIM_BOUNDARY",
    "LABEL_SET_SCHEMA",
    "MANIFEST_SCHEMA",
    "TRACE_GRAPH_SCHEMA",
    "WORKSPACE_SCHEMA",
]
