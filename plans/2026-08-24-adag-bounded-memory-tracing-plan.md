# ADAG bounded-memory tracing execution plan

Status: proposed and recorded; no implementation or experiment launch is authorized by this
document.

This plan follows the completed exact-trace optimization checkpoints recorded in
`plans/2026-08-23-adag-tracing-optimization-plan.md`. Those checkpoints reduced the 2,951-token
A100 reference trace from 276.49 to 138.20 seconds and from 60.20 to 46.99 GB peak allocated CUDA
memory. The latest cached-range cross-layer adapter is exact on the qualification target and lowers
that stage to 26.90 GB, but the run-wide peak is still owned by attribution/contribution work.

## Objective and claim boundary

Make CUDA memory a bounded working set while permitting sequence-scaled immutable state to live in
host RAM. The near-term capability target is a scientifically valid 20,000-token trace on one H200,
provided empirical scaling retains a safe device- and host-memory margin. A completed 20k run is a
target, not a prediction or current capability claim.

The execution method, target selection, attribution objective, graph topology rules, thresholding,
and saved values must remain unchanged. Each optimization is a named execution adapter with
artifact provenance and a reference adapter. Ordinary floating-point drift from a different
projection kernel may be acceptable only under a tolerance declared before GPU qualification;
identity, ordering, forward-value, topology, and lifecycle gates remain exact.

This plan deliberately stops at three levels:

1. sparse selected-coordinate source differentiation;
2. host-backed boundary streaming with source-group reuse and overlapped transfer;
3. host checkpoints plus bounded layer-window recomputation.

Fully token-blocked attention/backward, model-weight offload, multi-GPU sequence parallelism, and
integrated-gradients support for the new adapters are out of scope. They require separate designs
only if the three levels below cannot approach the 20k H200 target.

## Current measured resource model

The frozen 2,951-token A100 qualification target has 139 selected neurons across 20 active layers,
190 active layer pairs, 3,094 graph nodes, and 2,366 graph edges. Its current optimized profile uses
`source_leaf_v1`, `vectorized_v1`, `cached_range_v1`, `flash_sdpa_causal_v1`, and the default CUDA
allocator.

Current top-level runtime:

| Stage | Seconds | Share of 138.20 seconds |
| --- | ---: | ---: |
| Graph expansion | 87.04 | 63 percent |
| Cross-layer expansion, nested above | 81.58 | 59 percent |
| Selected attribution/contribution | 26.27 | 19 percent |
| Stop-gradient attribution/contribution | 21.65 | 16 percent |

Current stage allocation peaks:

| Stage | Peak allocated CUDA memory |
| --- | ---: |
| Stop-gradient attribution/contribution | 46.99 GB |
| Selected attribution/contribution | 42.72 GB |
| Important-neuron mask selection | 36.36 GB |
| Cached-range cross-layer expansion | 26.90 GB |
| Model and input baseline | about 8.04 GB |

The largest stop-gradient contribution VJP differentiates five candidate lanes with respect to a
`[1, 2951, 9728]` source. It begins at 30.84 GB and peaks at 44.29 GB. Reducing constants in that
path helps but retains a sequence-scaled dense source tangent. The new plan first removes that
tangent, then moves genuinely reusable sequence state to host RAM.

## Intended memory model

Let `T` be sequence length, `L` transformer depth, `D` residual width, `D_ff` MLP width, `C`
candidate lanes, `K` selected source coordinates, and `W` the resident replay-window width.

The current expensive device terms include saved state across depth and dense source VJPs:

```text
M_device = weights + O(L * T * D) + O(C * T * D_ff) + other workspaces
```

The three-level target is:

```text
M_device = weights + O(W * T * D) + O(C * K) + fixed staging buffers
M_host   = O(checkpoints * T * D) + compact selection and provenance state
```

This does not make total memory independent of `T`; it removes the depth and dense-source
multipliers from GPU residency and transfers the remaining linear state to a larger resource pool.
The memory model must be fitted from measured tensor bytes and peaks, not token count alone, because
selected-neuron count, active layers, pair topology, kernels, and allocator behavior also matter.

## Execution architecture

Keep one small external interface: prepare a deterministic residency plan, then compute the ordered
trace work through it. The resolved plan must include the adapter identity, source and target
groups, checkpoint layers, replay windows, VJP chunk widths, host and device budgets, buffer count,
and transfer policy. It is hashed into artifact identity before execution.

The implementation may use these internal modules without exposing their mechanics to callers:

- `SparseSourceInjection`: constructs an equal-valued selected-coordinate differentiation point;
- `HostBoundaryStore`: owns immutable raw-dtype CPU checkpoints and their receipts;
- `DeviceStagingPool`: owns a fixed number of GPU buffers, CUDA streams, and readiness events;
- `ReplayScheduler`: orders source groups, checkpoint loads, layer windows, VJPs, and eviction;
- `ResidencyTelemetry`: records device, host, transfer, wait, recomputation, and reuse evidence.

The existing device-resident adapters remain exact references. A streamed adapter is justified at
the established contribution and cross-layer execution seams; residency logic must not be spread as
`.cpu()` and `.to(device)` branches throughout attribution and graph code.

## Level 1: sparse selected-coordinate source differentiation

### Method

For a source MLP activation `x` and linear down projection `W`, retain the ordinary detached forward
value while differentiating only selected coordinates `x_S`:

```text
y_base = down_proj(x.detach())
y      = y_base + selected_projection(z - x_S.detach(), W_S)
z      = x_S.detach().requires_grad_(True)
```

At `z = x_S`, the correction is zero and `y` equals the reference forward value. The derivative
with respect to `z` is the selected derivative of the original linear down projection. The VJP
result becomes candidate-by-selected-coordinate rather than candidate-by-sequence-by-MLP-width.

Apply the adapter to:

1. ordinary selected-neuron contribution execution;
2. stop-gradient selected-neuron contribution execution;
3. pairwise and later grouped cross-layer Jacobian execution.

Do not change initial neuron discovery: the selected coordinates must already be frozen before
sparse source injection is used.

### Qualification gates

- Prove the selected-coordinate identity on CPU toy models across dtype, bias, duplicate-token,
  multi-batch, and multi-chunk cases.
- Preserve exact selected source and target forward receipts and canonical coordinate ordering.
- Attempt zero-tolerance raw-Jacobian parity first. If kernel shape changes cause only ordinary
  floating-point drift, declare absolute, relative, and ULP limits before the A100 job; never widen
  them after inspecting scientific outputs.
- Require exact graph topology, node/edge identities, target values where operation order is
  unchanged, and artifact provenance.
- Telemetry must demonstrate that no dense source leaf or dense source-gradient result remains
  live after projection.
- Accept the adapter only if the affected stage peak falls without an unexplained runtime or
  allocator regression. Record the new global owner rather than inferring it.

## Level 2: host-backed boundary streaming and source-group reuse

### Method

Run one no-grad preparation pass, store immutable layer-boundary residual state in host RAM, and
keep model weights resident on the GPU. Order work by source so one loaded boundary serves all
applicable later targets. Replay from the source through the furthest target in a bounded target
group, collect selected target activations, compute projected VJPs, and release the graph before
evicting the source.

Implement correctness before overlap:

1. `host_streamed_sync_v1` copies one source boundary synchronously and establishes exact execution
   and memory evidence.
2. `host_streamed_double_buffer_v1` uses two fixed staging slots, pinned host staging memory, a
   dedicated CUDA transfer stream, and explicit producer/consumer events.
3. Prefetch the next source while the compute stream processes the current source; do not reuse a
   slot until both computation and any outbound copy are complete.

Bulk host state may be pageable if pinning the full cache would create excessive locked memory. In
that case, copy through a bounded pinned staging pool. Never offload model weights in this plan.

Source grouping is part of the movement algorithm, not merely a runtime optimization. On the
2,951-token reference, pairwise cached replay enters 2,915 decoder layers. A full source-group
schedule would enter approximately 316 before accounting for the one preparation forward. Actual
time savings will be smaller because 190 target VJPs remain, but the schedule prevents reloading and
replaying the same source state for every target pair.

### Qualification gates

- Compare synchronous host streaming against the accepted device-resident adapter on the 2,951-token
  A100 target before enabling asynchronous overlap.
- Host-to-device and device-to-host copies must preserve raw tensor bytes and receipts exactly.
- Async and sync adapters must produce identical resolved work order, receipts, topology, and saved
  values under the declared numerical policy.
- Inject copy, replay, VJP, and cancellation failures; all hooks, buffers, streams, model state, and
  allocator ownership must be restored on exit.
- Record total D2H/H2D bytes, raw copy time, compute time, unhidden copy-wait time, overlap fraction,
  cache hits, source reuse, staging occupancy, pinned/pageable host peaks, and CUDA peaks.
- Judge async performance by unhidden wait, not theoretical PCIe bandwidth. Retain the synchronous
  adapter as the capacity/correctness reference even if it is slow.
- A target group must be bounded and explicit. Qualify widths such as one, two, four, and all targets
  per source; do not silently grow a retained graph to use available memory.

Generic saved-tensor CPU hooks may be used only as an isolated feasibility baseline. They are not
the intended production adapter because they obscure scheduling, can move tensors cheaper to
recompute, and make transfer prefetch difficult.

## Level 3: host checkpoints with bounded replay windows

### Method

When storing every reusable boundary or retaining the full downstream graph is too expensive,
checkpoint selected layers in host RAM and reconstruct omitted state on demand. For window width
`W`, load the nearest earlier checkpoint, replay at most `W` real decoder layers, perform all useful
source/target work in that window, and discard the graph and staging state.

The resolved scheduler chooses from explicit checkpoint/window profiles rather than making an
unrecorded adaptive decision. Initial qualification profiles should include representative widths
such as one, two, four, and eight layers. A memory-budget planner may later select among qualified
profiles, but its resolved profile and schedule must be frozen before tracing begins.

Apply checkpoint/recompute first to the contribution and cross-layer paths whose source and target
interfaces are already explicit. Initial neuron discovery/selection may use a separately qualified
host-saved activation or layerwise-recomputation adapter only after Levels 1 and 2 reveal it as the
remaining ceiling.

### Qualification gates

- Each window profile must pass the same forward receipts, selected Jacobian checks, exact topology,
  and declared numerical tolerances as Levels 1 and 2.
- Verify the maximum resident layer window and staging-buffer count directly in telemetry.
- Report replayed decoder-layer entries and recompute time separately from transfer and VJP time.
- Demonstrate the intended trade: smaller windows cannot use more device state without explanation;
  larger windows cannot perform more replay without explanation.
- Fail closed before scientific output if host budget, device budget, pinned-buffer allocation,
  checkpoint receipt, resolved schedule, or required backend differs from provenance.

## Telemetry contract

Every adapter must report four resource categories separately:

1. live CUDA allocation and stage/call peaks;
2. CUDA reservation, inactive splits, retries, and OOMs;
3. resident, pinned, and peak host memory;
4. movement and recomputation: bytes, raw copy time, blocked wait, overlap, replay entries, cache
   hits, and VJP chunks.

Also retain the existing workload predictors: input/response token counts, selected-neuron count by
layer and token, active-layer span, pair count, source/target group sizes, candidate width, and
frozen-topology identity. Scaling models must use these predictors rather than fitting token count
alone.

## Development and scaling sequence

For each level:

1. implement focused CPU algebra, interface, lifecycle, and failure tests;
2. run the broader tracing regression set and formatting/static checks;
3. commit a cohesive checkpoint;
4. create an immutable VAST execution worktree and new configuration;
5. run `sbatch --test-only`;
6. submit one A100 qualification job at a time on the 2,951-token reference;
7. compare against the accepted artifact and record a strict report;
8. only after acceptance, extend the same-GPU-family context ladder.

Use A100s for development and the first capacity ladder. Candidate ladder points should be chosen
from frozen workloads near 4k, 6k, 8k, and 10k while recording selected-neuron and pair topology;
these are target regions, not permission to modify a frozen prompt. Stop the ladder at the first
failed correctness, memory, runtime, or resource gate.

Before H200 scaling, run a same-profile H200 anchor at a context already qualified on A100. Keep all
H200 parity and timing comparisons within the H200 family. Then attempt frozen targets near 10k,
12k, 16k, and 20k sequentially, stopping at the first failed gate. Do not infer H200 correctness or
capacity from A100 headroom alone.

## Capacity decisions

Classify each ladder point separately:

- **correctness-qualified:** all identity, receipt, topology, and numerical gates pass;
- **capacity-feasible:** correctness-qualified, completes without OOM or allocator retry, and stays
  within requested host memory;
- **production-ready:** capacity-feasible with at least 10 percent physical CUDA headroom and at
  least 20 percent host-allocation headroom at peak, plus completion inside the selected partition
  walltime;
- **failed:** any correctness, provenance, memory, scheduler, or lifecycle gate fails.

The 20k H200 objective is achieved only at the production-ready level. A run that barely completes
is useful scaling evidence but does not authorize a full tracing campaign.

## Implementation order

1. Add sparse selected-coordinate source injection behind the current contribution and cross-layer
   seams.
2. Qualify Level 1 and refresh the memory-owner/runtime table.
3. Add the synchronous host boundary store and source-group scheduler.
4. Qualify synchronous Level 2, then add double-buffered prefetch and qualify it independently.
5. Run the A100 ladder until a new measured owner or failure appears.
6. Add Level 3 checkpoint/window profiles only where Level 2 telemetry shows retained state or host
   traffic requires the trade.
7. Complete the H200 anchor and sequential ladder, ending at 20k only if every earlier gate passes.

Do not begin fully token-blocked attention work merely because total memory remains linear in
sequence length. Reconsider it only if a single full-sequence layer working set becomes the measured
ceiling after Level 3, or if the 20k H200 target remains infeasible under acceptable runtime.
