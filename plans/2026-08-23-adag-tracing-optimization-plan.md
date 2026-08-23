# ADAG exact-trace memory and runtime optimization plan

Status: active optimization plan; source-leaf contribution execution is implemented and awaiting
A100 artifact qualification.

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

## Ordered optimization work

### P1. Source-leaf stop-gradient contribution execution

Implementation status: complete in the development checkpoint; GPU acceptance is pending.

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

P1 is accepted only if the new source-leaf strategy is provenance-visible, the legacy strategy
remains available, focused tests pass, the reference artifact comparison passes, and measured A100
peak allocation decreases without a material trace-time regression. Otherwise retain the trusted
contribution-compaction checkpoint and diagnose the failed gate before proceeding.
