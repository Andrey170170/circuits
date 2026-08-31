# ADAG bounded-memory tracing execution plan

Status: active under explicit user authorization. The Level 1a sparse-source and Level 1b bounded
target-lane stop-gradient contribution slices are implemented and measured. The Level 1e ordinary
selected-neuron and Level 1f ordinary embedding contribution slices are accepted as numerically
equivalent BF16 capacity primitives. The Level 1g stop-gradient embedding contribution slice is an
exact capacity primitive on the frozen calibration target. Level 1j bounds ordinary
selected-attribution neuron lanes and establishes a measured A100 safety bracket between 6,997 and
7,796 tokens for that optimized profile. Level 1k removes the unused decoder suffix from
stop-gradient selected-attribution forwards, and Level 1l releases the compact projection's
autograd edge immediately. The latter lowers the intermediate stop-gradient live set by 8.51 GB
at 8,266 tokens but does not move the later ordinary selected-attribution/contribution peak.
Level 1m projects only the selected LM-head positions and is exact-qualified. Allocator-aware
telemetry then showed that the 8,266-token run completed without retries or OOMs and was near the
capacity boundary rather than certain to fail. Level 1n, post-selection discovery-state
compaction, is exact-qualified on the 2,951-token A100 reference and comfortable at 8,266 tokens.
The frozen 9,397-token endpoint is capacity-qualified with the optional
`expandable_segments_v1` allocator policy: it removes the default allocator's retry, retains exact
saved-artifact equality, and restores more than 10 percent physical CUDA headroom. A new immutable
balanced/40k bundle then selected the nearest qualifying item above 10k at 10,006 tokens. Job
`15369505` completed that item on A100 with zero retries/OOMs and comfortable allocator-aware
headroom. This is an exact-workload capacity result, not permission to infer capacity from token
count alone or to authorize broader tracing runs by itself.

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

The execution method, target selection, attribution objective, graph topology rules, and
thresholding must remain unchanged. Each optimization is a named execution adapter with artifact
provenance and a reference adapter. A change in CUDA kernel shape, batching, accumulation order, or
storage schedule may introduce ordinary dtype-scale floating-point drift without changing the
scientific computation. Accept such an adapter only when the numerical mechanism is localized,
identity and ordering remain exact, forward values and selections remain unchanged, topology is
exact, and the drift magnitude and qualitative edge cases are reported. Use a labeled calibration
target to set any numerical policy, then freeze that policy before a promotion panel or scientific
holdout; do not tune it against the holdout.

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

The pre-Level-1h calibration profile additionally used width-one target-lane chunks for the
stop-gradient neuron, stop-gradient embedding, ordinary selected-neuron, and ordinary embedding
contribution VJPs. The following Level 1g tables are retained as the owner snapshot that motivated
Levels 1h-1j:

| Stage | Seconds | Share of 136.11 seconds |
| --- | ---: | ---: |
| Graph expansion | 86.10 | 63 percent |
| Cross-layer expansion, nested above | 80.80 | 59 percent |
| Selected attribution/contribution | 26.23 | 19 percent |
| Stop-gradient attribution/contribution | 21.06 | 15 percent |

Current stage allocation peaks:

| Stage | Peak allocated CUDA memory |
| --- | ---: |
| Important-neuron mask selection | 36.36 GB |
| Selected attribution/contribution | 36.08 GB |
| Stop-gradient attribution/contribution | 30.70 GB |
| Cached-range cross-layer expansion | 26.90 GB |
| Model and input baseline | about 8.04 GB |

Width-one target-lane execution reduced the measured dense contribution-VJP workspaces from about
13.45 GB to 2.69-2.75 GB without changing target selection or compact topology. The current
Level 1m 2,951-token selected-position profile peaks at 26,723,487,232 allocated and
34,613,493,760 reserved bytes and completes tracing in 132.615 seconds. At 8,266 tokens it peaks at
59,592,814,592 allocated and 81,797,316,608 reserved bytes and completes tracing in 341.097
seconds.

The allocator-aware 8,266-token diagnostic measured 54.82 GiB active allocation, 11.14 GiB
inactive-split reservation, and 0.51 GiB external pressure at the limiting boundary. The observed
joint state left 12.78 GiB of physical headroom; the deliberately conservative independent-max
estimate left 6.78 GiB. There were no allocator retries or OOMs. Its classification is `watch`
with a warning action: useful diagnostic evidence, not a prediction that the next allocation must
fail. Continue Level 1 only against measured live owners; move genuinely reusable sequence state
to host RAM under Level 2 when local lifetime reductions no longer provide enough capacity.

The 10,006-token balanced/40k target selected 91 neurons across 20 active layers and 190 active
layer pairs. Under `expandable_segments_v1`, job `15369505` peaked at 60,957,301,248 allocated and
69,640,126,464 reserved CUDA bytes, recorded zero retries/OOMs, and retained 23,584,007,680 bytes
under the conservative allocator-aware estimate. The trace took 452.881 seconds; Slurm peak host
RSS was 13,974,872 KiB against 64 GiB requested. This point meets the plan's exact-workload device,
host, and walltime margins and moves the measured A100 Level 1 bracket beyond 10k. It does not
change the need for Levels 2 and 3 if the intended H200 ladder extends materially beyond this
workload.

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

### Level 1a measured checkpoint: stop-gradient contribution only

Commit `9c696f161b037042a526d3103e9e79f0bf9bb5cb` contains the reviewed
`sparse_source_leaf_v1` implementation and its fail-closed qualification harness. The immutable
execution worktree is
`/scratch/general/vast/u1653998/circuits/run-worktrees/sparse-source-qual-9c696f1`.
The implementation is deliberately narrower than Level 1: it applies only to the per-layer
stop-gradient selected-neuron contribution VJP. Ordinary combined attribution/contribution and
cached-range cross-layer Jacobians still use their existing dense source representations.

The CPU matrix covered FP32 and BF16, bias and no bias, nested and direct MLP layouts, duplicate
and unsorted coordinates, multiple batch elements and target lanes, output-modifying hooks,
compact storage, failure cleanup, configuration restoration, and telemetry. The bounded regression
set passed 78 tests before the qualification harness was added and 79 after it was added. Forward
logits and selected source values were exact; selected CPU VJPs were tolerance-bounded rather than
bit-exact.

Jobs `15065186` and `15065517` ran sequentially on the same A100 80GB node, pinned commit, target,
driver, package environment, allocator, attention backend, embedding materialization, and
cached-range replay adapter. They differed only in `source_leaf_v1` versus
`sparse_source_leaf_v1`:

| Measurement | Dense source leaf | Sparse source leaf | Difference |
| --- | ---: | ---: | ---: |
| Trace wall seconds | 137.662 | 137.456 | -0.206 (-0.15 percent) |
| Global peak allocated bytes | 46,990,911,488 | 46,990,911,488 | 0 |
| Global peak reserved bytes | 51,661,242,368 | 51,661,242,368 | 0 |
| Contribution-VJP stage peak allocated bytes | 44,293,482,496 | 44,237,182,976 | -56,299,520 |
| Contribution-VJP incremental allocated workspace | 13,451,132,416 | 13,451,132,416 | 0 |
| Contribution-VJP wall seconds | 5.117 | 5.060 | -0.057 |

Sparse telemetry records 139 differentiated coordinates across 20 forwards and VJPs, 695 raw
VJP elements, 574,146,421 logical dense source elements avoided, and 2,870,732,105 logical dense
raw-VJP result elements avoided. Each active layer records
`source_representation=selected_coordinates` and
`dense_vjp_result_materialized=false`. This proves that the dense endpoint was removed, but the
unchanged 13.45 GB incremental workspace shows that downstream backward state owns the working
set. The run-wide owner remains the enclosing stop-gradient attribution/contribution stage at
46.99 GB; sparse endpoint projection alone is not a capacity optimization at 2,951 tokens.

The zero-tolerance report at
`/scratch/general/vast/u1653998/circuits/results/process_witness/sparse-source-qualification-v1/9c696f1/qualification-reports/source-leaf-vs-sparse-source-leaf-zero-v1.json`
passed exact target values, node values and topology, edge topology, and candidate profiles, but
failed edge-value equality. Only 10 of 2,366 edges changed, none changed sign, the maximum absolute
weight and attribution differences were `4.9801e-4` and `6.6103e-5`, and the maximum symmetric
relative difference was 0.7634 percent. Rounding the final edge metrics to BF16 produced at most a
two-representable-value distance; this is a final-metric diagnostic, not a raw-Jacobian ULP receipt.

A second report using exact gates everywhere except edge `rtol=0.008` passed:
`/scratch/general/vast/u1653998/circuits/results/process_witness/sparse-source-qualification-v1/9c696f1/qualification-reports/source-leaf-vs-sparse-source-leaf-edge-rtol-008-v1.json`.
Because that bound was calibrated after inspecting this target's zero-tolerance result, treat it as
a bounded diagnostic and candidate policy, not a predeclared holdout qualification or scientific
parity claim. `sparse_source_leaf_v1` remains opt-in and unpromoted. Before extending or promoting
it, freeze the numerical policy and validate it on a separate target, then choose the next memory
owner rather than assuming wider sparse-source integration will lower the global peak.

### Level 1b measured checkpoint: bounded target-lane VJP chunks

The next bounded slice addresses the measured 13.45 GB backward workspace without changing source
selection. `stop_gradient_contribution_target_lane_chunk_size` limits contiguous target lanes per
stop-gradient contribution VJP while keeping all batch lanes for those targets together. `None`
preserves the historical all-target execution. Positive widths project each raw dense or sparse
chunk immediately into canonical `(coordinate, batch, target)` order, retain the graph for every
non-final chunk, and apply the adapter's historical graph-lifetime contract to the final chunk.

This is an orthogonal, provenance-bearing stop-gradient contribution option. It does not chunk the
ordinary combined contribution path, selected-neuron attribution lanes, embedding contributions,
or cross-layer Jacobians.

Commit `d022d5dc13b0d7e4f93e5bdc1afc507a1e98be75` contains the reviewed Level 1b
implementation and fail-closed qualification harness. The immutable execution worktree is
`/scratch/general/vast/u1653998/circuits/run-worktrees/target-lane-chunk-qual-d022d5d`.
Jobs `15069125` (all targets) and `15069394` (width one) ran sequentially on the same `notch370`
A100. Their control artifact was `topk-trace-b5b11b680bd7a29258737f00`; their candidate artifact
was `topk-trace-bf669dbbd495e7242a93cdf9`. Both used accepted `source_leaf_v1` execution and differed
only in `stop_gradient_contribution_target_lane_chunk_size=None` versus `1`.

| Measurement | All targets | Width one | Result |
| --- | ---: | ---: | ---: |
| Trace wall seconds | 137.77288887114264 | 136.8338000040967 | -0.93908886704594 |
| Global peak allocated bytes | 46,990,911,488 | 46,990,911,488 | unchanged |
| Global peak reserved bytes | 51,661,242,368 | 51,661,242,368 | unchanged |
| CUDA headroom bytes | 33,432,535,040 | 33,432,535,040 | unchanged |
| Contribution-VJP peak allocated bytes | 44,293,483,008 | 33,535,095,808 | -10,758,387,200 |
| Maximum incremental VJP workspace bytes | 13,451,133,440 | 2,692,746,240 | -80 percent |
| Contribution-VJP wall seconds | 5.134982774732634 | 5.095142133766785 | -0.039840640965849 |
| Contribution-VJP executions | 20 | 100 | 5x |

Both runs recorded zero OOMs and allocator retries. The compact projected raw-dtype SHA-256 receipt
matched exactly for every one of the 20 layers. The zero-tolerance report
`.../qualification-reports/all-targets-vs-width1-zero-v1.json`, SHA-256
`c3f3c7c4102d6bbd8bf8aa0c6ce4c02e8e7792856d6c2966840b84dad66d20da`, recorded
`qualification_passed=true` with only
`artifact_identity.adag_config.stop_gradient_contribution_target_lane_chunk_size` allowed to
differ.

This qualifies width-one chunking as an exact capacity primitive on this frozen target. It remains
opt-in and the default remains `None`. It did not improve the run-wide allocated, reserved, or
headroom peaks. The remaining owner is localized only to the enclosing
`stop_grad_mlp_attribution_contribution` stage at 46,990,911,488 allocated bytes, not yet to a
specific operation inside that stage. Add finer forward and post-VJP telemetry before choosing an
offload or recomputation design.

### Level 1c measured checkpoint: stop-gradient allocation lifetimes

Commit `f1204016c67034273071434c2f46bc5ad62adb9d` adds an exhaustive partition of the
allocation-bearing happy path inside `stop_grad_mlp_attribution_contribution`. The new lifetime
regions checkpoint and reset allocator peak counters without adding CUDA synchronization; the
three existing VJP timing regions retain their historical synchronized timing semantics. The CPU
gate passed 97 focused tests, including exact telemetry-off versus telemetry-on toy outputs and an
explicit assertion that memory-only regions do not synchronize when the enclosing recorder does.

Immutable worktree
`/scratch/general/vast/u1653998/circuits/run-worktrees/stop-grad-telemetry-qual-f120401` ran job
`15117291` alone on the frozen 2,951-token width-one A100 target. It completed on `notch369` in
3:27 with zero OOMs or allocator retries. Its artifact is
`topk-trace-4d0c76be6126f7b52b4c5f3f`. The strict zero-tolerance report
`.../qualification-reports/width1-telemetry-zero-v1.json`, SHA-256
`ba7f454782131cb13bbc0168944b1a1b5a1a201c4435dfe2b3925924bff0ff4e`, passed exact target,
node, edge, candidate-profile, and topology gates. All 20 projected contribution-VJP raw-dtype
receipts also match the accepted width-one reference exactly.

The telemetry preserved both the 46,990,911,488-byte allocated peak and the
51,661,242,368-byte reserved peak exactly. Trace wall time changed from 136.8338000040967 to
137.63252190989442 seconds, an approximately 0.58 percent increase. The largest direct child,
`stop_grad_selected_layer_forward`, reproduced the enclosing allocated peak exactly, so the
partition closed the previous 6,400,353,792-byte localization gap.

The decisive call is the selected-attribution forward for layer 1:

| Boundary | Allocated bytes |
| --- | ---: |
| Layer-1 forward entry | 30,956,872,704 |
| Layer-1 forward peak | 46,990,911,488 |
| Layer-1 forward exit | 30,926,695,936 |

Layer 0's forward exits with 30,594,133,504 allocated bytes; the next model forward transiently
overlaps that retained graph before rebinding the previous `out`. The sharp rise and return to the
same approximately 30.9 GB baseline confirms a cross-forward graph-lifetime overlap rather than a
persistent accumulation of the compact attribution outputs. The next bounded optimization is to
release the selected-attribution autograd graph on the final neuron chunk of each layer, then rerun
the same exact qualification. If successful, the next exposed owner is expected to be the
40,590,557,696-byte embedding-contribution VJP; that expectation is a forecast, not qualification
evidence.

### Level 1d measured checkpoint: selected-forward graph release

Two one-variable lifetime changes tested the layer-forward diagnosis sequentially.

Commit `0853c7e55d6b6416e4f691480cf6d16bdcde7be8` releases the selected-attribution
autograd graph on the final neuron chunk of each layer. Immutable worktree
`/scratch/general/vast/u1653998/circuits/run-worktrees/selected-graph-release-qual-0853c7e`
ran job `15117371` on `notch369`; artifact `topk-trace-54c0f3000eb4319cc1d701a2` passed the
strict report `.../qualification-reports/telemetry-vs-final-release-zero-v1.json`, SHA-256
`65a098e44b75bedb097674e66ab46d7292ee9189127abb6d4d71ad74da3888e9`. All 20 projected
contribution-VJP receipts remained exact. This reduced global allocated peak by only 270,263,296
bytes and left the selected-layer forward as the owner. The result showed that the traversed prefix
graph was only a small part of the overlap.

Commit `ca8aa5f4d86c70a4eb256104b4dff167f22f914a` removes each selected-activation capture hook
immediately after the forward and releases the unused logits and downstream suffix graph before
the selected VJPs. The selected activation and its required prefix graph remain live. A forced
forward-failure test verifies hook cleanup, and the focused CPU gate passed 98 tests. Immutable
worktree
`/scratch/general/vast/u1653998/circuits/run-worktrees/selected-suffix-release-qual-ca8aa5f`
ran job `15117489` alone on the same `notch369` A100. Artifact
`topk-trace-1162ee18dc3ac10c7b585107` passed the strict report
`.../qualification-reports/final-release-vs-suffix-release-zero-v1.json`, SHA-256
`350d8923b424db7f4e106f080009c6659d66ac9ffef896d7ba0019d60aa9cafd`, with exact targets,
nodes, edges, candidate profiles, topology, and all 20 projected contribution-VJP receipts.

| Measurement | Telemetry baseline | Suffix release | Change |
| --- | ---: | ---: | ---: |
| Trace wall seconds | 137.63252190989442 | 136.75125901889987 | -0.88126289099455 |
| Global peak allocated bytes | 46,990,911,488 | 42,720,715,264 | -4,270,196,224 |
| Global peak reserved bytes | 51,661,242,368 | 51,661,242,368 | unchanged |
| Stop-gradient phase peak allocated bytes | 46,990,911,488 | 37,109,096,960 | -9,881,814,528 |
| Stop-gradient selected-forward peak bytes | 46,990,911,488 | 30,699,150,848 | -16,291,760,640 |

This retires selected-forward suffix overlap as the run-wide allocated owner without adding runtime.
The next stop-gradient-local owner is the 37,109,096,960-byte embedding-contribution VJP. The new
run-wide owner is outside that phase: `selected_neuron_contribution_vjp` in the ordinary selected
attribution/contribution path at 42,720,715,264 bytes. That path still materializes all target lanes
and was explicitly outside Level 1b target-lane chunking. The next bounded optimization should
generalize the already-qualified target-lane chunk/project pattern to this ordinary selected-neuron
contribution VJP, with a separate exact receipt and artifact qualification. Reserved peak remains
unchanged, so the current result establishes lower live allocation, not yet lower reservation-based
headroom.

### Level 1e measured checkpoint: ordinary contribution target-lane chunks

Commit `a73834ce257933bee2bb7a6b8b10ddb6c17989ab` adds an independent
`selected_neuron_contribution_target_lane_chunk_size` execution option for the ordinary selected
attribution/contribution path. The default remains `None`, which reuses the historical full target
identity. Positive widths preserve every batch lane, traverse contiguous target chunks against the
shared forward graph, project each dense result immediately into canonical
`(coordinate, batch, target)` order, and release that dense chunk before the next backward. This is
separate from the already-qualified stop-gradient option and does not change its lifecycle policy.

The focused CPU, provenance, launcher, and negative-test gate passed 111 tests, Ruff check and
format, shell syntax, and diff hygiene. The immutable execution worktree is
`/scratch/general/vast/u1653998/circuits/run-worktrees/selected-neuron-target-lane-chunk-qual-a73834c`.
Jobs `15118022` (`None`, `notch370`) and `15118070` (width one, `notch369`) ran sequentially on the
same frozen 2,951-token A100 target. Their artifacts are respectively
`topk-trace-2b70c50d011cd49dd27a7835` and `topk-trace-69d94f9ba37aa27401ad2a2f`.

| Measurement | All targets | Width one | Difference |
| --- | ---: | ---: | ---: |
| Trace wall seconds | 137.97593464702368 | 137.08704141899943 | -0.88889322802425 (-0.64 percent) |
| Global peak allocated bytes | 42,720,715,776 | 42,645,131,776 | -75,584,000 |
| Global peak reserved bytes | 51,661,242,368 | 51,661,242,368 | unchanged |
| Ordinary contribution-VJP peak allocated bytes | 42,720,715,776 | 31,960,002,560 | -10,760,713,216 |
| Maximum incremental ordinary VJP workspace bytes | 13,450,895,872 | 2,690,183,168 | -10,760,712,704 (-80 percent) |
| Ordinary contribution-VJP wall seconds | 8.828284760937095 | 8.989955821307376 | +0.161671060370281 (+1.83 percent) |
| Ordinary contribution-VJP executions | 20 | 100 | 5x |

Both runs completed with zero OOMs and allocator retries. The width-one run bounded the maximum
materialized target and autograd lanes at one. It retires the ordinary selected-neuron VJP as the
allocated-memory owner; the unchanged ordinary `selected_embed_contribution_vjp` becomes the new
owner at 42,645,131,776 bytes. This explains why a 10.76 GB local reduction lowers the run-wide peak
by only 75.584 MB. Reserved memory is unchanged, so this is lower live allocation rather than new
reservation-based headroom. The one-pair timing result does not demonstrate a runtime penalty at
whole-trace scale.

The first strict report is deliberately retained as a failed calibration:
`/scratch/general/vast/u1653998/circuits/results/process_witness/selected-neuron-target-lane-chunk-qualification-v1/a73834c/qualification-reports/all-targets-vs-width1-zero-v1.json`,
SHA-256 `8b31ec9c7bd46634efcf267b890739ba9427e260a8a263a10510528d344b8e1f`.
Target values, all 3,094 node values, node and edge topology, and all identities except the declared
chunk-width field are exact. The execution-shape change alters CUDA floating-point accumulation:
19 of 20 projected raw-dtype VJP hashes differ, 325 of 2,366 edge attributions differ with maximum
absolute error `2.644103648863083e-4` and cosine `0.9999969217694802`, and 181 of 15,470 candidate
profile values differ with maximum absolute error `0.125`, RMSE `0.00123554639726896`, and cosine
`0.9999977997927378`. Of the changed profile values, 116 are within one BF16 ULP, 154 within two,
and 172 within four. One near-zero value changes sign from `-0.00274658203125` to
`0.00299072265625`; graph topology remains exact. The jobs used different physical A100 nodes, so
the pair also retains a run- or hardware-level nondeterminism confound.

These diagnostics are consistent with the batched-five-lane versus repeated-single-lane backward
kernel shape. The zero-tolerance report remains failed evidence rather than being rewritten, but
the user explicitly accepted this localized dtype-scale drift on 2026-08-25. Width one is therefore
an accepted numerically equivalent BF16 capacity primitive and may be used in subsequent optimized
profiles. It remains an explicit option and the compatibility default remains `None`; acceptance
does not claim bitwise equivalence or validate other chunk widths, targets, dtypes, or GPU families.

Level 1f below completes the required same-node `None`, explicit-width-five, and width-one
calibration for the next ordinary embedding-contribution owner. The compatibility default remains
unchanged; a short/medium/long promotion panel is still required before a broader production
claim.

### Level 1f measured checkpoint: ordinary embedding contribution target-lane chunks

Execution commit `54339ff41b4f87be285acae8300743154f519fc9` adds the independent
`selected_embed_contribution_target_lane_chunk_size` adapter and its runtime receipts. Direct mode
projects each raw chunk into canonical `(source, batch, target)` storage; integrated-gradient mode
projects into `(target, batch, source, hidden)` storage. Both preserve source-token order and
duplicates and retain the shared forward graph for the later selected-neuron VJPs. A shared full
target identity is created only when this adapter or the ordinary selected-neuron adapter uses its
`None` compatibility width, and it is released as soon as the final unchunked consumer finishes.

The focused CPU, provenance, launcher, lifetime, and negative-test gate passed 147 tests before the
observed numerical scope was repaired and 150 after it, plus Ruff check and format, shell syntax,
JSON parsing, and diff hygiene. The immutable execution worktree is
`/scratch/general/vast/u1653998/circuits/run-worktrees/selected-embed-target-lane-chunk-qual-54339ff`.
Jobs `15118840` (`None`), `15118858` (explicit width five), and `15118877` (width one) ran strictly
sequentially on the same `notch370` A100 80GB node. They completed with exit zero in 3:23, 2:57,
and 3:05 respectively, with no OOM or allocator retry. Their artifacts are
`topk-trace-a8a65d1b7f8b0112b59a4b3c`, `topk-trace-e7e8750348a60f90b7a833c6`, and
`topk-trace-94ab54b7847b32afc1d7ef93`.

The `None` versus explicit-width-five adapter gate passed bit-exactly: target values, 3,094 nodes,
2,366 edges, every numerical field, topology, and the projected raw-dtype embedding-VJP receipt
were exact. Both executions proved one `[5, 1, 2951, 2560]` raw VJP. The report is
`/scratch/general/vast/u1653998/circuits/results/bonafide/process-witness-selected-embed-target-lane-chunk-qualification-v1-54339ff/qualification-reports/none-vs-5-exact-v1.json`,
SHA-256 `f804b84973aeab775c7eeeac45679649595d109a275db3125ab35e07063560db`.

| Measurement | Width five | Width one | Difference |
| --- | ---: | ---: | ---: |
| Trace wall seconds | 135.80221712309867 | 136.79571696510538 | +0.9934998420067132 (+0.73 percent) |
| Global peak allocated bytes | 42,660,260,352 | 37,109,122,560 | -5,551,137,792 (-13.01 percent) |
| Global peak reserved bytes | 51,661,242,368 | 51,661,242,368 | unchanged |
| Embedding contribution-VJP peak allocated bytes | 42,660,260,352 | 31,960,006,144 | -10,700,254,208 |
| Embedding contribution-VJP incremental workspace bytes | 13,450,896,384 | 2,750,642,176 | -10,700,254,208 (-79.55 percent) |
| Embedding contribution-VJP wall seconds | 0.9374974151141942 | 0.9710942069068551 | +0.0335967917926609 |
| Embedding contribution-VJP executions | 1 | 5 | 5x |

Width-one telemetry proves five ordered raw chunks of `[1, 1, 2951, 2560]`, maximum target and
autograd lane counts of one, immediate compact projection, and retained-graph execution. The
5.55 GB run-wide reduction retires the ordinary embedding VJP as the allocated-memory owner.
Reserved peak remains unchanged, so this is lower live allocation rather than new
reservation-based headroom. The new run-wide owner is the stop-gradient phase at
37,109,122,560 allocated bytes: `stop_grad_embed_contribution_vjp` and its enclosing
`stop_grad_mlp_attribution_contribution` stage reach the same peak. Important-mask selection is
next at 36,359,686,656 bytes. Further Level 1 work should therefore return to the measured
stop-gradient lifetime/owner seam rather than further tuning the now-bounded ordinary VJPs.

The zero-tolerance width-five versus width-one report is deliberately retained as failed
calibration evidence:
`.../qualification-reports/5-vs-1-zero-tolerance-diagnostic-v1.json`, SHA-256
`a9a63b4772cda98d97524ae33c748d4333f308d99900948fa0fbf60f38f08c36`. Topology, targets,
scalar node values, all edges, and all non-embedding contribution profiles were exact. The
execution-shape change affected the 14,750 embedding-source contribution-profile values and their
direct aggregation into the five layer-36 logit-node `attr_map` rows: maximum absolute error
`0.25`, mean absolute error `3.6672170510983205e-5`, RMSE `0.0020611133909148884`, and cosine
`0.9999872832861618`. Every one of the 9,112,550 non-logit `attr_map` values and all 720
non-embedding candidate-profile values remained exact.

The initial scoped report correctly failed because its contract had not declared the downstream
logit `attr_map` dependency. Comparator repair commit
`c575667` identifies those rows fail-closed from exact topology and the exact candidate token axis;
it permits the predeclared BF16 tolerance only for embedding-source profiles and those five logit
maps while keeping every other node/profile/edge class exact. The repaired report
`.../qualification-reports/5-vs-1-bf16-scoped-v2.json`, SHA-256
`2e336a028f72167206e2f244e27e92a1a1081d2e4fbfb1cf27b61b3be6945b34`, records
`qualification_passed=true` with no failed required gate.

Width one is therefore accepted as a numerically equivalent BF16 capacity primitive on this
frozen calibration target. It remains opt-in, `None` remains the compatibility default, and the
result does not claim bitwise equivalence, scientific parity, other widths/dtypes/GPU families, or
promotion-panel validation. No broader trace run was launched during this checkpoint.

### Level 1g measured checkpoint: stop-gradient embedding target-lane chunks

Execution commit `032e0c6bd83b0297382125e1ef74ce0c41956dc8` adds the independent
`stop_gradient_embed_contribution_target_lane_chunk_size` adapter. It shares the dense target-lane
execution module used by ordinary contribution paths, but owns a separate public interface,
configuration field, telemetry namespace, and receipt contract. Intermediate chunks retain the
embedding forward graph; the final chunk releases it because every later stop-gradient neuron
contribution starts a fresh forward. Ordinary paths preserve their prior final-retain behavior and
telemetry key.

The focused algebra, lifecycle, provenance, launcher, and negative-test gate passed 158 tests,
Ruff check and format, shell syntax, JSON parsing, and diff hygiene. The immutable execution
worktree is
`/scratch/general/vast/u1653998/circuits/run-worktrees/stop-gradient-embed-target-lane-chunk-qual-032e0c6`.
One initial job, `15119232`, failed closed during preflight because the submitted width-one source
receipt referred to an older smoke manifest; it loaded no model and produced no artifact. The
corrected jobs `15119245` (`None`), `15119292` (explicit width five), and `15119432` (width one)
then ran strictly sequentially on the same `notch370` A100 and exited zero in 3:31, 3:06, and 3:18.
Their artifacts are `topk-trace-03563a72446c0d6d283fef95`,
`topk-trace-d01b18e983c4bd3b1dd25e90`, and `topk-trace-316172902659e9da51d59ddb`.

The migration comparison from the prior accepted Level 1f artifact to the new `None` adapter is
bit-exact. Its report is `.../qualification-reports/prior-width1-vs-new-none-zero-v1.json`, SHA-256
`6476768e7ad2165233d3d8e24ead13be619ac50fb668dbfb51dd28da78fbe178`. The canonical `None`
versus explicit-width-five report also passes exact targets, nodes, edges, candidate profiles,
topology, execution shapes, and projected raw-dtype receipt:
`.../qualification-reports/none-vs-5-exact-v1.json`, SHA-256
`27984770f0cf85e344f4ed66bdc766537812df0e97499a7fdfa9cc456e7baa1c`.

| Measurement | Width five | Width one | Difference |
| --- | ---: | ---: | ---: |
| Trace wall seconds | 135.704180726083 | 136.11017425195314 | +0.40599352587014 (+0.30 percent) |
| Global peak allocated bytes | 37,124,255,232 | 36,359,686,656 | -764,568,576 (-2.06 percent) |
| Global peak reserved bytes | 51,661,242,368 | 51,661,242,368 | unchanged |
| Stop-gradient phase peak allocated bytes | 37,124,255,232 | 30,699,176,448 | -6,425,078,784 |
| Embedding contribution-VJP peak allocated bytes | 37,124,255,232 | 26,423,821,824 | -10,700,433,408 |
| Embedding contribution-VJP incremental workspace bytes | 13,451,054,080 | 2,750,620,672 | -10,700,433,408 (-79.55 percent) |
| Embedding contribution-VJP wall seconds | 0.46693867398425937 | 0.4629479080904275 | -0.00399076589383185 |
| Embedding contribution-VJP executions | 1 | 5 | 5x |

Unlike the ordinary embedding analogue, the five-lane and width-one stop-gradient executions are
bit-exact. The projected receipt is the same
`acde4279e54ca423754c9a209e7bd64ed2b50c4625d9b5a4fd0b9282e695f391` on both sides. The
canonical width-one report is `.../qualification-reports/5-vs-1-exact-v1.json`, SHA-256
`c1cf9f0cd12868e3c58feb0cea19305b0acfcfa853d2526145f4c00e709a5686`; it proves the requested
and resolved widths, five ordered raw chunks of `[1, 1, 2951, 2560]`, exact projected receipt,
same GPU model, exact topology, and zero numerical tolerance.

Width one therefore retires the stop-gradient embedding VJP as the allocated-memory owner and is
accepted as an exact opt-in capacity primitive on this frozen calibration target. It has no
measured runtime penalty. The global peak now lands exactly on `important_mask_selection` at
36,359,686,656 bytes; `selected_attribution_contribution` is close behind at 36,083,632,640 bytes.
Reserved memory remains unchanged, so this is a live-allocation improvement rather than additional
reservation-based headroom. The next Level 1 checkpoint should remove the mask-selection boolean
compaction used only for the all-zero test, then separately qualify immediate detach of terminal
attribution projections. No broader trace run was launched during this checkpoint.

### Level 1h measured checkpoint: important-mask positive-value guard

Execution commit `6e61b4f5459d383ff4b483b63ca41510046dd870` replaces the
`important_mask_selection` boolean-index compaction with an explicit comparison reduction. The
old path materialized all positive values, together with the dynamic nonzero/index workspace,
although it consumed only `len(nonzero_values) == 0`. The shared replacement is used by both the
active tracing path and the legacy core path. It retains the strict `> 0` comparison, so empty
tensors, signed zero, negative inputs, NaNs, and infinities keep the previous predicate semantics.
The reduction still materializes a one-byte comparison tensor and is therefore a Level 1
coefficient reduction rather than bounded-memory selection.

The focused semantic, operator-lifetime, teacher-forced probe, and upstream T5 gates passed 37
tests, Ruff check and format, and diff hygiene. A separate critical review found no blocking issue.
The repository-wide ty gate remains red on pre-existing coarse-sampling and tracing diagnostics;
`circuits/tracing/attribution.py` is also outside the configured typed boundary. The immutable
execution worktree is
`/scratch/general/vast/u1653998/circuits/run-worktrees/important-mask-selection-qual-6e61b4f`.
Slurm job `15120817` ran the frozen 2,951-token width-one target on a single A100 80GB PCIe at
`notch369`, completed with exit `0:0` in 3:35, and produced artifact
`topk-trace-b506c0605383a78f4d6756b5`.

| Measurement | Accepted width-one baseline | Positive-value reduction | Difference |
| --- | ---: | ---: | ---: |
| Trace wall seconds | 136.11017425195314 | 139.66994787286967 | +3.55977362091653 (+2.62 percent) |
| Global peak allocated bytes | 36,359,686,656 | 36,084,537,856 | -275,148,800 (-0.76 percent) |
| Global peak reserved bytes | 51,661,242,368 | 38,486,933,504 | -13,174,308,864 (-25.50 percent) |
| Important-mask peak allocated bytes | 36,359,686,656 | 19,313,699,328 | -17,045,987,328 (-46.88 percent) |
| Important-mask incremental workspace bytes | 23,249,362,944 | 6,203,375,616 | -17,045,987,328 (-73.32 percent) |
| Important-mask peak reserved bytes | 51,661,242,368 | 34,613,493,760 | -17,047,748,608 (-33.00 percent) |
| Important-mask wall seconds | 0.14330808888189495 | 0.10546518489718437 | -0.03784290398471 (-26.41 percent) |

The single-run total trace time was 2.62 percent slower on a different physical A100 node, while
the changed stage itself was 26.41 percent faster. This is insufficient evidence for a total
runtime regression or improvement, and no repeated timing campaign was run. The strict report
`.../qualification-reports/prior-width1-vs-mask-reduction-zero-v1.json`, SHA-256
`9a10a242cb9347e0ee0f614ee6eeaa869eda5d101218d9568044521f16b27bfb`, records
`qualification_passed=true`: both artifact identities are internally valid, the frozen scientific
identity and A100 model match, node and edge topology are exact, and targets, node values, edge
values, source-attribution profiles, and all five candidate profiles have zero maximum absolute
error.

The selection allocation is therefore retired as the global owner. The new allocated peak is
36,084,537,856 bytes in `selected_attribution_contribution`, specifically its
`selected_attribution_vjp` calls; its first measured call begins at 14,144,219,648 bytes and peaks
at 36,084,537,856 bytes. The next Level 1 checkpoint should qualify immediate detach/release of
terminal attribution projections within that path before introducing Level 2 host-backed
offload. No broader trace run was launched during this checkpoint.

### Level 1i measured checkpoint: terminal selected-attribution projection release

Execution commit `014324851be2b2e037d95ac4aab4a86699acad67` moves the ordinary
selected-attribution reshape, batch-diagonal extraction, source-token projection, and terminal
detach behind one internal projection contract. Regular attribution multiplies by a detached
embedding value, and integrated-gradient execution retains compact gradient-only values without a
backward edge. The shared dense target-lane executor now also detaches each compact first-order
projection before retaining it. Its callback contract requires independent compact storage, which
all current adapters establish through advanced indexing or a reduction.

The focused storage, contribution, teacher-forced trace, telemetry, benchmark, and launcher gates
passed 138 tests, Ruff format and check, and diff hygiene. Tests cover regular and gradient-only
exactness, batch size greater than one, reordered and duplicate source tokens, compact storage,
terminal graph state, and release of one raw VJP before the next backward. A separate critical
review found no blocking correctness issue. The repository-wide ty gate remains red on 44
pre-existing diagnostics outside this checkpoint; `circuits/tracing/attribution.py` also remains
outside the configured typed boundary. The immutable execution worktree is
`/scratch/general/vast/u1653998/circuits/run-worktrees/selected-attribution-release-qual-0143248`.
Slurm job `15128945` ran the unchanged frozen 2,951-token width-one target on one A100 80GB PCIe at
`notch369`, completed with exit `0:0` in 3:41, and produced artifact
`topk-trace-c090599f4a3be645e46f0484`.

| Measurement | Accepted important-mask baseline | Terminal projection release | Difference |
| --- | ---: | ---: | ---: |
| Trace wall seconds | 139.66994787286967 | 138.81428890000097 | -0.85565897286870 (-0.61 percent) |
| Global peak allocated bytes | 36,084,537,856 | 34,436,713,472 | -1,647,824,384 (-4.57 percent) |
| Global peak reserved bytes | 38,486,933,504 | 37,107,007,488 | -1,379,926,016 (-3.59 percent) |
| Selected-attribution phase peak allocated bytes | 36,084,537,856 | 34,436,713,472 | -1,647,824,384 |
| Selected-attribution phase end allocated bytes | 25,809,368,576 | 25,733,809,152 | -75,559,424 |
| First selected-attribution VJP start bytes | 27,093,016,576 | 27,093,016,576 | unchanged |
| Last selected-attribution VJP start bytes | 28,741,503,488 | 27,093,679,104 | -1,647,824,384 |
| Last selected-attribution VJP peak bytes | 36,084,537,856 | 34,436,713,472 | -1,647,824,384 |
| Last VJP incremental workspace bytes | 7,343,034,368 | 7,343,034,368 | unchanged |

The strict report
`.../qualification-reports/prior-mask-vs-selected-attribution-release-zero-v1.json`, SHA-256
`545f724fba71e58018bb53cca40452abbcaf8edfbcc74bf46da1a04dede4e1e7`, records
`qualification_passed=true`. Both artifact identities are internally valid; the frozen scientific
identity and exact A100 model match; node and edge topology are exact; and targets, node values,
edge values, source-attribution profiles, and all five candidate profiles have zero maximum
absolute error. This establishes exact engineering parity for this target, not general scientific
parity.

The VJP-call baseline is now effectively flat: the first call starts at 27,093,016,576 bytes and
the final call starts only 662,528 bytes higher. The removed 1.648 GB was therefore retained raw
VJP storage, not required shared-forward state. The additional 75.6 MB reduction at phase end is
consistent with terminal release in the shared embedding projection path. The global owner is
still `selected_attribution_vjp`, but it is now the final 30-neuron call's real within-call
workspace rather than accumulation from prior layers. Its incremental workspace remains
7,343,034,368 bytes.

The next Level 1 checkpoint should expose the selected-attribution neuron-lane width independently
of the other contribution widths and qualify a narrower value on this same target. The active
hard-coded width is 50 and the peak layer has 30 selected neurons, so that call is currently
unchunked. A narrower width should bound the remaining raw VJP/workspace coefficient at the cost of
additional backward traversals; it requires exact or explicitly bounded BF16 evidence before any
default change. No broader trace run was launched during this checkpoint.

### Level 1j measured checkpoint: selected-attribution neuron-lane chunks and A100 ladder

Execution commit `0d3c1c5141e3102af318d033884e69c5845f559e` adds the independent
`selected_attribution_neuron_lane_chunk_size` adapter. `None` remains the compatibility default and
resolves to the historical width 50; a positive integer chunks only ordinary
selected-attribution VJPs. It does not change integrated gradients, stop-gradient attribution,
cross-layer Jacobians, or any contribution lane. Every chunk is projected into terminal compact
source-attribution storage before the next traversal. Comparator-hardening commit
`52d67b3aa6e9da6b551471cb16331ea03062501e` adds a canonical explicit-`None` versus width-one
runtime contract and separates the historical Jacobian width into its own constant.

The implementation gate passed 145 focused tests before comparator hardening. The final
qualification/Jacobian gate passed 91 tests, plus Ruff and diff hygiene. The immutable execution
worktree is
`/scratch/general/vast/u1653998/circuits/run-worktrees/selected-attribution-neuron-lane-qual-0d3c1c5`.
All GPU runs used one A100 80GB PCIe at a time, the same frozen manifest, and an 8 GiB physical
headroom stop gate. A gate failure after an artifact was saved is capacity evidence, not an OOM or
a correctness failure.

The unmodified width-50 profile first established the local scaling regime:

| Tokens | Job | Selected / active layers / pairs / max layer count | Trace seconds | Peak allocated GB | Peak reserved GB | Headroom GB | Result |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 2,951 | `15128945` | 139 / 20 / 190 / 30 | 138.814 | 34.437 | 37.107 | 47.987 | accepted anchor |
| 4,201 | `15129303` | 100 / 22 / 231 / 16 | 164.782 | 40.688 | 45.519 | 39.575 | passed |
| 4,999 | `15129322` | 101 / 20 / 190 / 12 | 189.623 | 46.260 | 52.855 | 32.239 | passed |
| 6,202 | `15129394` | 79 / 16 / 120 / 13 | 176.337 | 55.613 | 63.552 | 21.542 | passed |
| 6,997 | `15129406` | 118 / 18 / 153 / 35 | 357.661 | 73.436 | 77.588 | 7.505 | artifact complete; headroom gate failed |

The first four points through 6,202 tokens fit a reserved-memory slope of approximately 8.19
decimal GB per 1,000 tokens with `R^2=0.994`. Including the structurally heavier 6,997-token target
raises the fitted slope to approximately 9.71 GB per 1,000 tokens with `R^2=0.969`. These fits are
descriptive only: selected-neuron count, active layers, pairs, and the maximum selected layer alter
the working set, so token count alone is not a capacity law. The ladder stopped before 7,796 as
predeclared when 6,997 crossed the safety gate.

Jobs `15129472` and `15129486` then ran an explicit-`None` versus width-one A/B on the 2,951-token
anchor:

| Measurement | Explicit `None` / resolved 50 | Width one | Difference |
| --- | ---: | ---: | ---: |
| Trace wall seconds | 138.04753871797584 | 137.4302997670602 | -0.617239 (-0.45 percent) |
| Peak allocated bytes | 34,436,713,472 | 30,616,958,976 | -3,819,754,496 (-11.09 percent) |
| Peak reserved bytes | 37,107,007,488 | 34,613,493,760 | -2,493,513,728 (-6.72 percent) |
| CUDA headroom bytes | 47,986,769,920 | 50,480,283,648 | +2,493,513,728 |
| Selected-attribution VJP executions | 20 | 139 | +119 |

The canonical strict report
`.../qualification-reports/none-vs-1-canonical-zero-v1.json`, SHA-256
`c6ee9a2e318fc40411c3de88ed3f0115602516c4fd14e265b55a5f5b7311d110`, passes artifact
identity, frozen scientific identity, the exact allowlist, same GPU model, exact topology, and the
runtime contract proving requested/resolved widths `None/50` versus `1/1` and 20 versus 139 ordered
VJP/projection calls. Its overall zero-tolerance result remains failed only on node numerics. The
changed values are confined to source-attribution profiles: 181,354 of 9,127,300 cells (1.99
percent), median changed absolute difference `7.63e-6`, 99th percentile `3.66e-4`, cosine
`0.9999999968`, and maximum `0.5`; only 36 cells differ by more than `0.01`. Targets, topology,
scalar node values, edges, and all candidate profiles are exact. The separate posthoc scoped
diagnostic passes an observed absolute bound of `0.5`; it is retained as calibration evidence, not
a predeclared scientific-parity gate. The user explicitly accepted this localized BF16 execution-
order drift. Width one is therefore accepted as an opt-in capacity primitive on this calibration
lane, while `None` remains the default.

The optimized boundary replay demonstrates the capacity effect:

| Tokens | Profile | Job | Trace seconds | Peak allocated GB | Peak reserved GB | Headroom GB | Result |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 6,997 | width 50 | `15129406` | 357.661 | 73.436 | 77.588 | 7.505 | headroom gate failed |
| 6,997 | width 1 | `15129592` | 355.066 | 61.265 | 70.460 | 14.634 | passed |
| 7,796 | width 1 | `15129662` | 361.769 | 67.545 | 77.773 | 7.321 | artifact complete; headroom gate failed |

At 6,997 tokens, width one removes 12,170,908,672 allocated bytes (16.57 percent) and
7,128,219,648 reserved bytes (9.19 percent), adds the same amount of physical headroom, and changes
trace time by -2.59 seconds (-0.73 percent). The generic exact comparison against the legacy
width-50 artifact passes every target, topology, node, edge, and candidate-profile value across
49,797,528 source-profile cells. At 7,796 tokens there is still no OOM, but the saved artifact is
outside the predeclared 8 GiB safety margin. Its allocated owner is
`stop_grad_selected_layer_forward`; its reserved peak occurs during `selected_attribution_vjp`,
where 58.886 GB allocated coexists with 77.773 GB reserved, exposing allocator/cache pressure as a
separate capacity concern.

The strongest current A100 statement is therefore a measured safety bracket: 6,997 tokens passes
and 7,796 tokens fails the 8 GiB headroom gate under width one. A three-point optimized reserved-
memory fit estimates the gate crossing near 7,664 tokens and physical exhaustion near 8,630, but
it uses only three heterogeneous workloads and is not a launch guarantee. Ten thousand tokens is
not projected to fit safely on an A100 under this profile. The next Level 1 targets are the
stop-gradient selected-layer forward live set and phase-scoped allocator/cache release; Level 2
host-backed streaming remains the planned route once those local reductions stop paying.

### Level 1k measured checkpoint: prefix-stop selected-attribution forwards

Execution commit `6e9c89ac12ef27f84fb09faf1618b14716ede611` adds an explicit
`full_model_v1` versus `prefix_stop_v1` strategy for stop-gradient selected-attribution forwards.
The candidate executes the embedding and decoder prefix only through the selected MLP input and
captures that input before the selected down projection. It does not execute the selected down
projection, decoder suffix, final norm, LM head, or logits. The historical full-model strategy
remains the default.

Sequential A100 jobs `15130871` and `15131428` ran the frozen 2,951-token target. The canonical
zero-tolerance report passes exact identity, ordered execution receipts, 3,094 nodes, 2,366 edges,
and every saved target, node, edge, and candidate-profile value. Prefix-stop reduced trace wall
time from 139.598 to 134.264 seconds and peak allocated memory from 30,616,958,976 to
29,801,943,552 bytes. Peak reserved memory remained 34,613,493,760 bytes, so this is an exact local
live-set reduction rather than additional allocator headroom.

Job `15131443` then completed the 8,266-token artifact without an OOM or allocator retry but failed
the 8 GiB headroom gate. The run peaked at 68,952,779,776 allocated and 81,797,316,608 reserved
bytes and left 3,296,460,800 bytes of physical headroom. Prefix-stop reduced its own selected
forward peak to 44,454,334,464 allocated bytes, retiring that path as the owner. The later ordinary
selected-attribution/contribution phase became the measured global owner.

### Level 1l measured checkpoint: terminal stop-gradient projection storage

Execution commit `80f5393835fbfa47c179f6f410cd12cdc57aa98c` adds a storage strategy after
the unchanged FP32 selected-attribution projection and source-token indexing. Historical
`graph_retaining_v1` remains the default; opt-in `terminal_detached_v1` shares the compact
projection's storage but removes its autograd edge before retention. This does not claim to release
every local reference or the entire selected forward graph.

Sequential A100 jobs `15133602` and `15133654` ran the frozen 2,951-token target on the same node.
The canonical report at
`.../qualification-reports/context-2501-4000-graph-retaining-vs-terminal-detached-exact-v1.json`
passes every required gate at zero tolerance: topology is exactly 3,094 nodes and 2,366 edges,
all saved values are exact, all 26 ordered workloads match, the reference retains 26 projection
graphs, and the candidate detaches all 26 and retains none. Trace time was 137.090 versus 134.465
seconds. Both runs nevertheless had the same 29,801,943,552 allocated and 34,613,493,760 reserved
global peaks.

Job `15133678` completed the candidate's 8,266-token artifact and then failed only the unchanged
headroom gate. All 20 compact projections were detached and none retained a graph. Relative to the
historical graph-retaining artifact, the candidate reduced the stop-gradient selected phase-final
live allocation by 8,508,037,120 bytes and the stop-gradient embedding contribution-VJP peak by
8,509,245,952 bytes. The effect mostly disappeared across later fresh forwards, however, and the
ordinary selected phase began at the same 25,092,750,336 allocated bytes on both runs. Global peak
allocated, peak reserved, and headroom were byte-for-byte unchanged at 68,952,779,776,
81,797,316,608, and 3,296,460,800 bytes; trace time was 348.878 versus 347.375 seconds.

### Level 1m measured checkpoint: selected-position target logits

Execution commit `907e52c` adds a named `full_logits_v1` reference and
`selected_position_logits_v1` candidate. Sequential A100 jobs `15146889` and `15150153` ran the
frozen 2,951-token target. The candidate reduced LM-head projection rows from 2,951 to 5 and peak
allocated CUDA memory from 29,801,940,992 to 26,723,487,232 bytes; peak reservation remained
34,613,493,760 bytes. The strict report passes exact 3,094-node and 2,366-edge topology plus every
saved value, with SHA-256
`2d8dd1bf775a2c987c745a696a79b61e44183a837ee022dbd61dd7621734a14b`.

Job `15164818` then completed the 8,266-token selected-position artifact with 8,362 nodes and 1,046
edges and no allocator retry or OOM. Target-logit projection is therefore no longer the next
owner.

### Level 1n qualification checkpoint: post-selection discovery state

The next target is dense discovery state retained after important-neuron selection. The selectable
`dense_v1` reference preserves the historical lifetime. The `compact_cpu_v1` candidate preserves
ordered selected coordinates and their raw-dtype initial-attribution values on CPU, drives graph
expansion through a strategy-independent state interface, and releases the unused dense
`mlp_final_acts`, `mlp_final_attributions`, `embed_final_acts`, and global mask from the candidate's
retained device state. Probe and return-only modes retain their historical behavior and bypass this
normal-trace adapter.

Receipts bind strategy, coordinate order and hash, compact value shape, dtype, raw hash and bytes,
logical input/retained/released bytes, placement, and allocator state immediately before and after
the storage transition. Implementation commit `55e7bcf` and checkpoint-contract fix `e536b6a`
passed focused CPU algebra, lifecycle, integration, adversarial receipt, and launcher tests.

Sequential A100 jobs `15348279` (`dense_v1`) and `15351183` (`compact_cpu_v1`) qualified the frozen
2,951-token reference. The zero-tolerance report passes exact 3,094-node/2,366-edge topology and
every target, node, edge, and candidate-profile value; its SHA-256 is
`1e6e16a67fe7bf94ba0f1941ff00132266b4fc7b09b80db7251c3fab04b837b2`. Compact storage released
5,182,427,882 logical bytes and reduced peak allocated memory from 26,723,485,696 to
23,726,469,632 bytes while reservation stayed 34,613,493,760 bytes. Trace time was 132.940 versus
132.640 seconds.

Job `15356053` then completed the frozen 8,266-token candidate in 343.508 seconds. It released
14,516,418,376 logical bytes, peaked at 51,873,096,704 allocated and 81,797,316,608 reserved bytes,
recorded zero retries/OOMs, and retained 22,972,558,336 bytes of conservative allocator-aware
headroom. This point is `comfortable`.

Job `15356095` completed the largest frozen item, 9,397 actual context tokens, in 298.132 seconds.
It released 16,502,635,380 logical bytes and peaked at 57,800,285,696 allocated and
78,741,766,144 reserved bytes. It did not OOM, but one allocator retry makes the receipt `critical`:
conservative headroom was 9,053,827,584 bytes and legacy physical headroom was only
6,352,011,264 bytes. Its lower runtime is workload-specific (70 selected occurrences and 597
edges) and is not evidence that longer traces are faster. Initial attribution is now the measured
57.80-GB global owner; selected attribution/contribution peaks at 50.02 GB and stop-gradient work
at 47.74 GB.

Commit `658e6a8` froze the same-commit 9,397-token `default_v1` versus
`expandable_segments_v1` allocator A/B while retaining `compact_cpu_v1`. Default job `15356157`
reproduced one retry and the `critical` receipt. Expandable job `15356591` completed with zero
retries/OOMs and a `comfortable` receipt. It reduced peak reservation from 78,741,766,144 to
65,905,098,752 bytes while leaving peak allocation nearly unchanged (57,800,285,696 versus
57,737,490,944 bytes). Conservative allocator-aware headroom rose from 9,053,827,584 to
26,803,817,984 bytes, and legacy physical headroom rose from 6,352,011,264 to 19,188,678,656
bytes. Trace time changed from 292.596 to 296.771 seconds (+1.4%).

The strict allocator report passes exact 9,471-node/597-edge topology and every target, node,
edge, source-profile, and candidate-profile value at zero tolerance. Its SHA-256 is
`8b871948873a3aa72351518cb527ffb6e357de26a23fd480eb0a1a8d0dab5487`. The optional expandable
policy is therefore capacity-qualified for this frozen 9,397-token workload. The comparator
establishes exact saved-artifact equivalence under its explicit contract; it does not by itself
make a broader scientific-parity claim. The immutable qualification-v4 manifest contains no
actual 10,000-token item, so an exact-10k proof requires a new versioned manifest rather than an
edit to the frozen one.

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

1. Completed: the same-commit compact 9,397-token default-versus-expandable allocator A/B passed
   strict saved-artifact equality; expandable segments recorded zero retries/OOMs.
2. Completed: the immutable balanced/40k nearest-above-10k bundle selected 10,006 tokens, and A100
   job `15369505` completed with zero retries/OOMs and comfortable device and host margins.
3. Record initial attribution as the current 10k global allocation owner; continue Level 1 only if
   another local lifetime reduction has a clear payoff.
4. Implement and qualify the synchronous Level 2 host boundary store and source-group scheduler,
   then add double-buffered prefetch and qualify it independently.
5. Run the A100 ladder until a new measured owner or failure appears.
6. Add Level 3 checkpoint/window profiles only where Level 2 telemetry shows retained state or host
   traffic requires the trade.
7. Complete the H200 anchor and sequential ladder, ending at 20k only if every earlier gate passes.

Do not begin fully token-blocked attention work merely because total memory remains linear in
sequence length. Reconsider it only if a single full-sequence layer working set becomes the measured
ceiling after Level 3, or if the 20k H200 target remains infeasible under acceptable runtime.
