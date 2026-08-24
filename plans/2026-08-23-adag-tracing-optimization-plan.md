# ADAG exact-trace memory and runtime optimization plan

Status: active optimization plan; source-leaf contribution execution, allocator-policy
qualification, vectorized embedding-edge materialization, and pairwise source-to-target partial
forwards are complete. A separately named source-group reuse strategy is the next runtime target.

## Objective and claim boundary

Make the frozen BonaFide teacher-forced tracing workload fit longer contexts on one A100 80 GB,
and eventually one H200, without changing its scientific objective, selected targets, graph
topology, or reported attribution/contribution values.

Every optimization is an execution strategy, not a new attribution method. It must be represented
in artifact provenance and qualified against a trusted artifact for the same target. A faster or
smaller run is not accepted merely because it completes: topology must match exactly and numerical
drift must remain within an explicit comparison report. Frozen manifests and existing profiling
configs are never edited in place.

## Measured baseline

The current trusted checkpoint is contribution compaction at code revision
`1b821e85c05fd0b5b10a0571dff084ca091da7f1`, using the A100 80 GB PCIe and
`flash_sdpa_causal_v1` stop-gradient attention.

For the 2,951-token qualification target:

- trace time: 261.44 seconds;
- peak allocated CUDA memory: 54.74 GB;
- peak reserved CUDA memory: 69.62 GB;
- physical headroom at peak: 15.47 GB;
- selected neurons: 139 across 20 active layers;
- graph: 3,094 nodes and 2,366 edges;
- allocator retries and OOMs: zero.

The 3,251-token target completed at 57.98 GB allocated and 74.57 GB reserved, with 10.52 GB of
physical headroom. It selected only 93 neurons rather than 139, so it is useful capacity evidence
but not a pure token-scaling comparison.

Fine telemetry identifies two independent resource problems:

1. The stop-gradient neuron-contribution stage owns the allocation peak. Its 20 per-layer VJPs
   take only 7.73 seconds in total, but the differentiable prefix grows from 27.49 GB at layer 0 to
   38.70 GB at layer 35 before each VJP. Each five-lane VJP then adds about 13.45 GB. The roughly
   11.2 GB depth-dependent prefix is the immediate memory target.
2. Graph expansion owns runtime. Embedding-edge materialization takes 100.49 seconds and
   cross-layer expansion takes 103.51 seconds, about 78 percent of trace time together.

## Completed checkpoints

1. Selectable OV-only causal attention for the stop-gradient wrapper, qualified with exact graph
   and value parity on the reference target.
2. Coarse stage and fine VJP CUDA telemetry, including allocator peaks and lane shapes.
3. Compact extraction of selected neuron contributions, allowing dense VJP storage to be released
   immediately. The compact artifact has exact parity with its trusted reference.
4. Source-leaf stop-gradient contribution execution, qualified on the 2,951-token A100 reference
   with exact topology and zero numerical error.
5. Default-versus-expandable CUDA allocator qualification on the same 2,951-token A100 target and
   code revision, with exact topology and zero numerical error.
6. Vectorized embedding-edge materialization, qualified on the same A100 target with exact
   topology and zero numerical error.
7. Pairwise cached-range cross-layer execution, qualified on the same A100 target with exact
   topology, zero numerical error, and exact internal receipts for all 190 layer pairs.

## Ordered optimization work

### P1. Source-leaf stop-gradient contribution execution

Implementation status: accepted at commit `d7cc45f63d070640fd71e1a447d99f60882798f8`.

Run the prefix to a selected MLP source without an autograd graph, replace the selected
`down_proj` input with a detached leaf requiring gradients, and build the graph only from that
source through the target logits. Release that sole backward graph immediately after the batched
VJP.

Engineering contract:

- expose one named execution strategy in `ADAGConfig`, preserving the legacy path as the default;
- centralize source-module resolution, hook lifecycle, and forward execution behind one small
  interface rather than branching throughout attribution code;
- record the strategy in artifact identity and telemetry;
- preserve the exact source activation values, target logits, selected contributions, and graph;
- remove hooks on both successful and exceptional exits.

Expected effect: eliminate most of the measured approximately 11.2 GB differentiable-prefix
growth. This is a hypothesis until the A100 qualification run measures it; the five-lane VJP and
other stage memory remain.

### P2. Allocator-policy A/B qualification

Compare the default CUDA allocator with a narrowly specified expandable-segment policy on the
same code, target, GPU model, and tracing config. Treat allocator settings as runtime provenance.
Accept a policy only if it reduces reserved-memory pressure or fragmentation without numerical or
material runtime regressions. This does not reduce live tensor storage and is therefore secondary
to P1.

Qualification status: complete at commit `9350b738973f70b168b29c1c34706afe439cf79f`.
The two immutable run configurations clone the accepted source-leaf profiling configuration and
differ only in the scalar policy declaration:

- `qwen3_4b_thinking_allocator_qualification_default_v1.json` explicitly unsets
  `PYTORCH_CUDA_ALLOC_CONF`;
- `qwen3_4b_thinking_allocator_qualification_expandable_segments_v1.json` sets it exactly to
  `expandable_segments:True`.

The launcher applies the policy before Python imports PyTorch and records both the intended policy
and an observed runtime receipt. Preflight, runner binding, and postflight reject policy,
environment, backend, or receipt disagreement. The dedicated `--cuda-allocator-ab` comparison
profile requires a canonical default-to-expandable lane pair, the same code and A100 model, exact
node and edge topology, and zero tolerance for target, node, edge, and candidate-profile values.
Only the scalar policy and the exact receipt leaves that necessarily change are allow-listed.

Decision: keep `expandable_segments_v1` as an opt-in capacity policy, not the unconditional
throughput default. It removes measured inactive-split fragmentation and gains 4.41 GB of physical
headroom, while this single paired run was 6.2 percent slower overall. That trade is useful when a
target is near the A100 memory boundary; ordinary targets should retain `default_v1` unless repeat
timing or longer-context evidence changes the throughput decision.

### P3. Vectorized embedding-edge materialization

Replace the scalar Python/token-neuron materialization path with tensorized filtering and batched
row construction while preserving edge ordering, thresholds, values, and provenance. The measured
target is the 94.90-second embedding materialization stage in the accepted P2 default-allocator
control; CUDA peak reduction is not the primary claim.

Implementation status: accepted at commit `cd5e17b3e35d7470407e53ad159fd365694b0be5`.
`ADAGConfig` exposes the named `scalar_v1` and `vectorized_v1` strategies, with `scalar_v1`
retaining the historical default.
The strategy seam receives ordered embedding sources and MLP targets after node creation. The
vectorized adapter prepares each target once, evaluates every ordered source together, and buckets
retained rows back into exact source-major and target-major graph order. Threshold comparisons stay
in the original tensor dtype, frozen topology overrides thresholds, duplicate sources remain
distinct, and objective reduction and missing final attributions retain the scalar contract.

The paired immutable qualification configs use the accepted source-leaf contribution path,
`flash_sdpa_causal_v1`, and `default_v1`; they differ only in
`adag_config.embedding_edge_materialization`. The strict `--embedding-edge-ab` comparator requires
the canonical scalar-to-vectorized pair, identical code and GPU model, exact node and edge topology,
and zero tolerance for target, node, edge, and candidate-profile values. The scalar artifact must
also match the trusted P2 default-allocator artifact exactly before the vectorized result is
accepted.

### P4. Source-to-target partial forwards for cross-layer tracing

Avoid replaying the whole transformer for each of the 190 active layer pairs. Establish explicit
source-leaf and target-boundary interfaces, reuse invariant prefix state where scientifically and
numerically valid, and stream projected results so pair-local graphs die promptly. This is the
largest architectural optimization and requires separate parity gates for source activation,
target activation, pair Jacobian, retained edges, and whole artifact.

Implementation status: accepted at commit `763982dcc4d6045ef08d8750556333b34e5ff660`.
The accepted P3 artifact made the performance loop explicit:
cross-layer expansion takes 103.95 seconds and fails the provisional under-50-second P4 target.
Pair Jacobians account for 101.76 seconds across 190 pairs, while pair materialization takes only
1.43 seconds. The legacy implementation executes all 36 decoder layers for every pair, retains a
full source-width VJP tensor, and copies that tensor through `torch.cat` before selecting the few
planned source coordinates.

The first P4 seam is a named cross-layer Jacobian execution strategy. `full_model_v1` remains the
default and exact historical reference; `cached_range_v1` is the candidate. Pair planning, frozen
topology, thresholds, target-contribution reduction, and edge ordering remain in `clja.py`. The
candidate prepares source-layer inputs under the exact stop-gradient attention implementation,
replays only the real decoder layers from source through target, begins autograd at an equal-valued
source leaf, terminates at the target MLP input, and projects every target VJP chunk to ordered
selected-source coordinates before accumulation. It must never silently fall back, remove
unowned hooks, or retain a pair graph after returning. Integrated-gradients execution remains on
the legacy strategy until separately designed and qualified.

Qualification proceeds in two steps: first, the lifted `full_model_v1` artifact must match the
trusted P3 vectorized artifact exactly; second, the strict full-to-cached comparison must use the
same commit and A100 model, allow only the scalar execution-strategy identity field to differ,
require exact topology, and apply zero tolerance to all saved values. Each canonical pair records
raw-dtype SHA-256 receipts over the ordered selected source activations, target activations, and
raw selected Jacobian before source multiplication or normalization. The strict comparator fails
closed on missing, malformed, reordered, or unequal receipts, so retained-edge pruning cannot hide
an internal-path difference. Focused and broader tracing regressions pass on CPU, including exact
float32/BF16 multi-chunk parity and injected preparation, replay, and VJP failures. A later
source-group sweep may reuse one graph for several targets, but it will be a separately named
adapter after pairwise partial execution establishes the target-boundary contract.

### P5. Batched-VJP lane chunking as a capacity fallback

Split the five candidate lanes only if P1--P4 do not provide sufficient headroom. Telemetry shows
that the current five-lane VJP adds about 13.45 GB, so lane chunking can bound peak allocation, but
it adds repeated backward work and may hurt runtime. Make chunk width explicit in provenance and
qualify widths on one GPU family before production.

## Qualification gates

For each checkpoint:

1. Add focused CPU tests for interface validation, lifecycle cleanup, shapes, and numerical
   equivalence on a small model.
2. Commit a cohesive code/config/test checkpoint before GPU execution.
3. Create an immutable VAST execution worktree at that commit and a new profiling config; do not
   mutate an executable tree while a job is active.
4. Run `sbatch --test-only`, then submit exactly one A100 qualification job at a time.
5. Compare the 2,951-token result with the trusted compact reference. Require exact target,
   topology, node/edge identities, and value parity unless a narrowly bounded floating-point drift
   is explained by the execution change.
6. Report allocated and reserved peaks, physical headroom, allocator events, total trace time, and
   affected stage/VJP timings. Do not infer token-only scaling from targets with different selected
   neuron counts.
7. Attempt a longer capacity target only after the 2,951-token qualification passes and retains a
   safety margin. Keep A100 and H200 result series separate because kernel/hardware drift is an
   independent variable.

## P1 acceptance decision

P1 is accepted. Slurm job `14966195` completed the frozen 2,951-token target on an NVIDIA A100
80GB PCIe. The candidate artifact is:

`/scratch/general/vast/u1653998/circuits/results/process_witness/source-leaf-contribution-v1/a100/d7cc45f63d070640fd71e1a447d99f60882798f8/flash_sdpa_causal_v1/bonafide.t5-upstream-summed-top5.v1/context-2501-4000/topk-trace-93757737c07a5c8546958174`

The execution-qualification report is:

`/scratch/general/vast/u1653998/circuits/results/process_witness/source-leaf-contribution-v1/a100/d7cc45f63d070640fd71e1a447d99f60882798f8/flash_sdpa_causal_v1/parity-reports/contribution-compaction-vs-source-leaf-2951-exact.json`

All identity and requested result gates passed. The candidate exactly matched 3,094 nodes, 2,366
edges, target values, node activation/attribution/source profiles, edge attribution/weight values,
and all 15,470 candidate-profile values at zero absolute and relative tolerance. This establishes
exact execution equivalence for the saved artifact; it does not upgrade the tracing method's
scientific claim boundary.

Resource result relative to the trusted contribution-compaction artifact:

- peak allocated: 54.74 GB -> 46.99 GB, down 7.75 GB or 14.2 percent;
- peak reserved: 69.62 GB -> 51.66 GB, down 17.96 GB or 25.8 percent;
- physical CUDA headroom: 15.47 GB -> 33.43 GB, up 17.96 GB;
- trace time: 261.44 seconds -> 258.77 seconds, down 2.67 seconds or 1.0 percent;
- stop-gradient attribution/contribution stage: 24.87 seconds -> 21.69 seconds;
- stop-gradient neuron-contribution VJPs: 7.73 seconds -> 5.13 seconds;
- allocator retries and OOMs: zero.

The per-layer contribution-VJP baseline no longer grows with source-layer depth: the trusted run
rose from 27.49 GB at layer 0 to 38.70 GB at layer 35, while the accepted run ranged from 28.18 GB
to 30.84 GB and ended at 28.18 GB. The remaining global 46.99 GB peak is therefore not the old
deep differentiable-prefix failure mode. P2 should proceed as an allocator-pressure experiment,
not as a substitute for missing live-tensor reduction.

## P2 qualification decision

P2 is qualified as an exact execution-equivalent capacity option. The sequential A100 jobs used
the same detached commit and the same physical GPU model:

- default allocator: Slurm job `14979993`, completed in 5:13;
- expandable segments: Slurm job `14980626`, completed in 5:06.

The default artifact is:

`/scratch/general/vast/u1653998/circuits/results/process_witness/allocator-policy-ab-v1/a100/9350b738973f70b168b29c1c34706afe439cf79f/default_v1/bonafide.t5-upstream-summed-top5.v1/context-2501-4000/topk-trace-aa592a715c951c45230960bc`

The expandable artifact is:

`/scratch/general/vast/u1653998/circuits/results/process_witness/allocator-policy-ab-v1/a100/9350b738973f70b168b29c1c34706afe439cf79f/expandable_segments_v1/bonafide.t5-upstream-summed-top5.v1/context-2501-4000/topk-trace-7e273ecd101d3c359cd06dae`

The strict comparison report is:

`/scratch/general/vast/u1653998/circuits/results/process_witness/allocator-policy-ab-v1/a100/9350b738973f70b168b29c1c34706afe439cf79f/parity-reports/default-vs-expandable-2951-exact.json`

Its SHA-256 is `089783db43953aa55b2e714a045b0dd48747d143f4ef5fbe352d6fd0e37d0834`.
All identity and requested result gates passed. Both artifacts contain 3,094 nodes and 2,366 edges,
and target values, node values, edge values, and all 15,470 candidate-profile values match at zero
absolute and relative tolerance. Both runs reported zero allocator retries and zero OOMs.

Resource result, expandable relative to default:

- peak allocated: 46.991 GB -> 46.956 GB, down 0.035 GB or 0.07 percent;
- peak reserved: 51.661 GB -> 47.249 GB, down 4.412 GB or 8.54 percent;
- physical CUDA headroom: 33.433 GB -> 37.845 GB, up 4.412 GB;
- peak inactive-split bytes: 18.968 GB -> 0 according to PyTorch allocator telemetry;
- trace time: 252.95 seconds -> 268.73 seconds, up 15.78 seconds or 6.24 percent.

The runtime increase is concentrated in allocation-heavy VJP stages rather than graph expansion:
selected-neuron contribution VJPs increased by 6.41 seconds and stop-gradient neuron-contribution
VJPs by 3.75 seconds, while total graph expansion changed by -0.04 seconds. This supports treating
the slowdown as part of the allocator trade rather than a graph-construction fluctuation. That
result motivated P3's independent 95-second embedding-edge target; P3 did not depend on enabling
the expandable allocator policy.

## P3 qualification decision

P3 is accepted as an exact execution-equivalent runtime optimization. The sequential A100 jobs
used the same detached commit, default allocator, and physical GPU model:

- scalar control: Slurm job `14984742`, completed in 4:50;
- vectorized candidate: Slurm job `14985350`, completed in 3:17.

The scalar artifact is:

`/scratch/general/vast/u1653998/circuits/results/process_witness/embedding-edge-materialization-v1/a100/cd5e17b3e35d7470407e53ad159fd365694b0be5/scalar_v1/bonafide.t5-upstream-summed-top5.v1/context-2501-4000/topk-trace-b33ec3f7d99af60e7c7397cd`

The vectorized artifact is:

`/scratch/general/vast/u1653998/circuits/results/process_witness/embedding-edge-materialization-v1/a100/cd5e17b3e35d7470407e53ad159fd365694b0be5/vectorized_v1/bonafide.t5-upstream-summed-top5.v1/context-2501-4000/topk-trace-1bf345a2d2612b692a81e070`

The strict scalar-versus-vectorized report is:

`/scratch/general/vast/u1653998/circuits/results/process_witness/embedding-edge-materialization-v1/a100/cd5e17b3e35d7470407e53ad159fd365694b0be5/parity-reports/scalar-vs-vectorized-2951-exact.json`

Its SHA-256 is `7ab68872bdbff2abf6d5120daff4ec62300ecf8182979ef0bdaf5ed2df035034`.
The scalar refactor also passed an exact comparison against the trusted P2 default artifact; that
report has SHA-256 `3a20cd571eef705b5fcb3d0fb05910ccd1f83a16f7cc4ecda8ae972f9328eeca`.
Both reports passed every identity and requested result gate. Scalar and vectorized artifacts each
contain 3,094 nodes and 2,366 edges, and target values, node values, edge values, and all 15,470
candidate-profile values match at zero absolute and relative tolerance.

Resource result, vectorized relative to scalar:

- embedding-edge materialization: 88.10 seconds -> 1.38 seconds, down 86.72 seconds or 98.4
  percent, a 64.0x stage speedup;
- total graph expansion: 194.91 seconds -> 109.43 seconds, down 85.48 seconds or 43.9 percent;
- trace time: 244.65 seconds -> 159.23 seconds, down 85.42 seconds or 34.9 percent;
- peak allocated: unchanged at 46.991 GB;
- peak reserved: unchanged at 51.661 GB;
- physical CUDA headroom: unchanged at 33.433 GB;
- candidate embedding edges: 410,050 in both runs, with 38 retained;
- allocator retries and OOMs: zero in both runs.

Decision: use `vectorized_v1` in future optimized trace configurations, while retaining
`scalar_v1` as the library default and exact reference strategy. The optimization removes the
embedding Python/synchronization bottleneck without changing the global CUDA peak. Cross-layer
graph expansion now takes 103.95 seconds, about 65 percent of vectorized trace time, confirming P4
as the next runtime target.

## P4 qualification decision

P4's pairwise cached-range strategy is accepted as an exact execution-equivalent incremental
runtime and stage-memory optimization. Both successful jobs used the same detached commit, default
allocator, `flash_sdpa_causal_v1`, vectorized embedding-edge materialization, and NVIDIA A100 80 GB
PCIe model:

- lifted full-model control: Slurm job `14990763`, completed in 3:27;
- cached-range candidate: Slurm job `14991104`, completed in 3:29.

The scheduler elapsed times include unequal model-loading time and are not used as the performance
comparison. The instrumented trace timings inside the artifacts are the qualification evidence.
An earlier control attempt, job `14990224`, failed before producing a scientific artifact because
the receipt serializer assumed a BF16 singleton target axis had unit stride. Commit
`763982dcc4d6045ef08d8750556333b34e5ff660` fixed the serializer by flattening before byte view and
added a focused regression test. The failed root and logs remain preserved as failure evidence.

The full-model control artifact is:

`/scratch/general/vast/u1653998/circuits/results/process_witness/cross-layer-jacobian-v1/a100/763982dcc4d6045ef08d8750556333b34e5ff660/full_model_v1/bonafide.t5-upstream-summed-top5.v1/context-2501-4000/topk-trace-342ad9256763bb93fb10e9f0`

The cached-range candidate artifact is:

`/scratch/general/vast/u1653998/circuits/results/process_witness/cross-layer-jacobian-v1/a100/763982dcc4d6045ef08d8750556333b34e5ff660/cached_range_v1/bonafide.t5-upstream-summed-top5.v1/context-2501-4000/topk-trace-47dd3e39db8fb8b74872bced`

The lifted-control comparison against the trusted P3 artifact is:

`/scratch/general/vast/u1653998/circuits/results/process_witness/cross-layer-jacobian-v1/a100/763982dcc4d6045ef08d8750556333b34e5ff660/parity-reports/p3-vs-full-model-2951-exact.json`

Its SHA-256 is `dddff04631cc428308a470fb850633d1cf84fa905578f17bdb175c1209a8dafb`.
All 12 gates passed. The strict full-to-cached report is:

`/scratch/general/vast/u1653998/circuits/results/process_witness/cross-layer-jacobian-v1/a100/763982dcc4d6045ef08d8750556333b34e5ff660/parity-reports/full-vs-cached-range-2951-exact.json`

Its SHA-256 is `2acd8150f6c0140a7a7cb38f3130277fd914106c537a40a547beab22ad950c18`.
All 13 gates passed. Both artifacts contain exactly 3,094 nodes and 2,366 edges; target values,
node values, edge values, and all candidate-profile values match at zero absolute and relative
tolerance. All 190 canonical pairs also have exactly equal raw-dtype receipts for selected source
activations, selected target activations, and selected raw Jacobians. This establishes equality
before contribution multiplication, normalization, and graph pruning, rather than only equality of
the saved retained graph.

Resource result, cached range relative to full model:

- cross-layer expansion: 104.09 seconds -> 81.58 seconds, down 22.52 seconds or 21.6 percent;
- pair-Jacobian work: 101.98 seconds -> 79.09 seconds, down 22.89 seconds or 22.4 percent;
- total trace time: 159.23 seconds -> 138.20 seconds, down 21.03 seconds or 13.2 percent;
- decoder executions or layer entries: 6,840 -> 2,951, down 3,889 or 56.9 percent;
- cross-layer peak allocated: 38.64 GB -> 26.90 GB, down 11.73 GB or 30.4 percent;
- cross-layer peak reserved: 38.88 GB -> 31.08 GB, down 7.79 GB or 20.0 percent;
- cached preparation state: one forward and 288,631,408 bytes;
- global peak allocated: unchanged at 46.991 GB;
- global peak reserved: unchanged at 51.661 GB;
- physical CUDA headroom: unchanged at 33.433 GB;
- allocator retries and OOMs: zero in both runs.

Decision: use `cached_range_v1` in future optimized trace configurations while retaining
`full_model_v1` as the library default and exact reference strategy. P4 reduces the intended
cross-layer bottleneck and its local live allocation, but it does not move the run-wide memory peak
and does not meet the provisional under-50-second cross-layer target. The next runtime experiment
should be a separately named source-group strategy that reuses one source graph across multiple
targets; it must preserve the same target-boundary and internal-receipt contract before acceptance.
