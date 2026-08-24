# tracing/

Circuit tracing via Cross-Layer Jacobian Attribution (CLJA). Identifies important MLP neurons and computes edge weights between them to build a circuit graph.

## Files

- **`trace.py`** — High-level tracing pipeline: prepares tokenized inputs, runs CLJA, and returns a `CircuitData` artifact containing node/edge DataFrames, tokenized inputs, target logits, and config metadata.
- **`clja.py`** — Core CLJA algorithm (`get_all_pairs_cl_ja_effects_with_attributions`). Orchestrates node selection, attribution/contribution computation, and edge tracing via jacobians.
- **`attribution.py`** — Gradient-based attribution helpers for scoring neuron importance (used by `clja.py` to filter neurons before edge computation).
- **`grad/`** — Custom backward-pass wrappers and selectable stop-gradient
  attention adapters (straight-through layernorm, stop-grad on attention/MLP
  gate, Shapley gradient approximations).
- **`utils.py`** — Data classes (`NeuronIdx`, `Node`, `Edge`) and activation collection utilities.

## Stop-gradient attention backends

`ADAGConfig.stop_gradient_attention_backend` is a provenance-bearing choice:

- `legacy_eager_unmasked_v1` is the default solely for reproduction of frozen
  traces. With Transformers 4.57, the historical custom attention key did not
  prepare a causal or padding mask.
- `eager_causal_v1` is the materialized, corrected causal reference.
- `flash_sdpa_causal_v1` detaches Q/K, preserves V/O gradients, returns no
  attention weights, and permits only PyTorch Flash SDPA. It fails instead of
  falling back to a math or memory-efficient kernel.

Qualify corrected eager against Flash SDPA on one GPU model before promotion.
Do not compare `legacy_eager_unmasked_v1` and either causal backend as if they
differed only by floating-point implementation drift.

## Stop-gradient contribution execution

`ADAGConfig.stop_gradient_contribution_execution` selects the autograd lifetime
for each selected-neuron contribution VJP and is included in artifact identity:

- `full_graph_v1` is the compatibility default. It begins the graph at the
  embeddings and retains it after the VJP, matching historical execution.
- `source_leaf_v1` computes the prefix without a graph, reinjects the selected
  MLP `down_proj` input as an equal-valued autograd leaf, and releases the
  downstream graph after its sole batched VJP.
- `sparse_source_leaf_v1` is the bounded-memory Level 1a adapter. It computes
  the same graph-free prefix but differentiates only the ordered selected
  `(token, neuron)` coordinates through an equal-valued zero correction at
  `down_proj`; it never materializes the dense source VJP result. Its current
  scope is only the stop-gradient per-layer contribution loop. Ordinary
  combined contribution and cross-layer Jacobian execution remain unchanged.

Treat `source_leaf_v1` as unqualified until an exact same-target, same-GPU
comparison confirms target values, topology, node values, edge values, and
candidate profiles against a trusted `full_graph_v1` trace.

Treat `sparse_source_leaf_v1` as unqualified until both focused CPU algebra and
lifecycle tests and an accepted same-target, same-GPU comparison against
`source_leaf_v1` pass the declared forward, Jacobian, topology, value, memory,
and runtime gates. CPU tests alone do not qualify scientific use.
