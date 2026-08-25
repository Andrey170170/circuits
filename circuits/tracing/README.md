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

`ADAGConfig.stop_gradient_contribution_target_lane_chunk_size` independently
limits how many contiguous target lanes are differentiated by each
stop-gradient contribution VJP. `None` is the compatibility default and uses
all targets at once; a positive integer slices only the target axis and keeps
every batch lane for each target together. Each chunk is projected into
canonical `(coordinate, batch, target)` order before the next backward pass.
This option currently applies only to the stop-gradient per-layer contribution
loop and composes with every contribution execution adapter. Commit
`d022d5dc13b0d7e4f93e5bdc1afc507a1e98be75` qualified width one as an exact
capacity primitive against the all-target `source_leaf_v1` control on one
frozen 2,951-token A100 target: all 20 projected raw-dtype receipts and the
zero-tolerance execution report were exact, while the contribution-VJP
incremental workspace fell 80 percent from 13,451,133,440 to 2,692,746,240
bytes. It remains opt-in, and `None` remains the compatibility default.

This qualification did not lower the 46,990,911,488-byte run-wide allocated
peak. The remaining owner is localized only to the enclosing
`stop_grad_mlp_attribution_contribution` stage, not yet to a specific operation.
Add finer forward/post-VJP telemetry before choosing offload or recomputation
work from this result.

`ADAGConfig.selected_neuron_contribution_target_lane_chunk_size` is the
independent counterpart for the ordinary selected-neuron contribution loop.
It preserves the shared forward graph across every layer and target chunk,
projects each dense raw VJP into canonical selected-coordinate storage before
the next traversal, and leaves the embedding contribution VJP unchunked.
`None` preserves the historical all-target execution and identity-matrix reuse;
a positive integer opts into bounded target-axis chunks. Treat this option as
an engineering candidate until a frozen same-code `None` versus width-one A100
qualification passes exact projected-receipt and compact-artifact gates.
