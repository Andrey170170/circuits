# ADAG exact-trace memory and runtime optimization plan

Status: active optimization plan; source-leaf contribution execution is accepted, and allocator
policy qualification is complete. Vectorized embedding-edge materialization is next.

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
target is the 100.49-second embedding materialization stage; CUDA peak reduction is not the primary
claim.

### P4. Source-to-target partial forwards for cross-layer tracing

Avoid replaying the whole transformer for each of the 190 active layer pairs. Establish explicit
source-leaf and target-boundary interfaces, reuse invariant prefix state where scientifically and
numerically valid, and stream projected results so pair-local graphs die promptly. This is the
largest architectural optimization and requires separate parity gates for source activation,
target activation, pair Jacobian, retained edges, and whole artifact.

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
the slowdown as part of the allocator trade rather than a graph-construction fluctuation. P3 now
targets the 95-second embedding-edge materialization path and does not depend on enabling the
expandable allocator policy.
