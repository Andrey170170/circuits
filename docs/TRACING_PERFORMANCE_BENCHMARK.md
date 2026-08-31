# Target-aware tracing performance benchmark

Status: active benchmark and qualification record through the allocator-qualified 9,397-token
Level 1n A100 endpoint.

## Purpose

Measure the wall-clock, GPU-memory, host-memory, and storage cost of teacher-forced ADAG tracing
before committing to a 50–150-context reference corpus. The benchmark must exercise the same
target selection, pruning, and serialization path intended for the later experiment; otherwise
its extrapolation is not trustworthy.

This is a systems benchmark, not an attempt to interpret or cluster a ten-prompt sample.

## Pipeline boundary

The tracing pipeline ends after saving compact, reusable trace artifacts with explicit target
provenance:

```text
frozen prompt and response
        |
selected response-token target window
        |
ADAG tracing and attribution profiles
        |
compact trace artifact + measurements
```

It does **not** construct a response-wide mega-graph, merge graph topology, cluster neurons,
generate descriptions, or summarize cluster labels. Those are separate downstream analyses.
Keeping this boundary narrow makes raw traces reusable when graph, clustering, or labeling
choices change.

## Stage-level trace instrumentation

Performance runs attach a JSON-only `adag.trace-instrumentation.v1` snapshot to
`trace.trace_metadata.instrumentation`, `metrics.json`, and the benchmark summary JSONL. A
recorder is created independently for every trace, so an OOM or other error record still retains
all counters and completed stage timings observed before the failure. Instrumentation is
observational: it does not alter scientific node/edge tables, attribution values, or tracing
return signatures.

The snapshot contains:

- `stages`: accumulated synchronized wall seconds and call counts for input preparation, target
  scoring, initial attribution, mask selection, selected-neuron attribution/contribution,
  optional stop-grad-on-MLP attribution/contribution, graph expansion, layer-pair Jacobians and
  materialization, and dataframe conversion;
- `early_predictors`: values available immediately after important-neuron mask selection,
  including selected neurons by layer and token, active-layer count/span, planned active layer
  pairs, `sum(n_src * n_tgt)` candidate MLP edges, planned Jacobian target chunks, and selected
  attribution chunks;
- `layers` and `layer_pairs`: selected counts plus, for every active source/target layer pair,
  source and target counts, candidate edges, target chunks, Jacobian seconds, materialization
  seconds, and retained edges;
- `counters`: raw and final dataframe graph sizes and node/edge deltas for logit, MLP, embedding,
  and cross-layer graph construction.

These are workload measurements, not new causal quantities. In particular,
`candidate_mlp_edge_count` is known before Jacobian computation, while `retained_edges` is known
after thresholding. `parent_threshold` is applied after the Jacobian in the current ADAG code, so
the early selected neurons must not be described as "parents." Counts from benchmark-only summed
objectives also retain those objective semantics and should not be interpreted as independent
per-token traces.

Chunk accounting distinguishes work per pass from actual executions. The main selected-neuron
pass records `selected_attribution_chunks_per_pass` and
`selected_attribution_chunk_executions`. With integrated gradients, the execution count is
multiplied by `ig_steps + 1`, matching the implementation's inclusion of the zero-alpha step.
Layer-pair Jacobians similarly record `target_chunks_per_pass`, `jacobian_pass_count`, and
`target_chunk_executions`. When stop-grad-on-MLPs is active, its separate chunk-size-10 pass is
reported explicitly and included in `total_selected_attribution_chunk_executions`. Planned layer
pairs list their concrete source and target layers and use the same exclusive `start_layer`
boundary as the runtime loop.

Stage timers are nested and therefore not additive. For example, `clja_total` contains
`initial_attribution` and `graph_expansion`, while `graph_expansion` contains the layer-pair
timers. Use the outer timer for end-to-end attribution cost and the inner timers for composition;
do not sum all stage values into another total.

CUDA performance runs synchronize the device at instrumentation timing boundaries. This makes
stage measurements meaningful despite asynchronous kernels, but slightly changes scheduling and
adds overhead. Compare instrumented runs with each other; do not treat their wall times as exactly
equivalent to earlier unsynchronized benchmarks. A stage interrupted by an exception records
unsynchronized host elapsed time and increments `failed_calls`; it never attempts a CUDA
synchronization while unwinding, because that could mask the original tracing error.

Allocator-layout telemetry is a separate, explicit profiling option. Set both
`instrumentation.cuda_memory_telemetry=true` and
`instrumentation.cuda_allocator_snapshot_telemetry=true` in a new run config to record compact
`adag.cuda-allocator-fragmentation-snapshot.v1` summaries. The policy is part of the runtime
artifact identity and defaults to disabled. It uses history-free `torch.cuda.memory_snapshot()`
calls at four allocation boundaries: after important-mask selection, immediately before the first
selected-attribution VJP, while that first raw VJP result is live, and after the complete selected
attribution/contribution phase. Integrated-gradients runs capture the two VJP boundaries once per
IG execution rather than once per neuron chunk.

Each checkpoint records its capture cost and trace-relative time, device free/total bytes,
ordinary allocator counters, segment and block totals, active/pending-free/inactive bytes, the
largest inactive block, fully inactive segments, inactive bytes in mixed segments, allocation
rounding slack, and fixed inactive-block size buckets. Raw allocator addresses, frames, and
history are not retained, and the checkpoint helper adds no explicit CUDA synchronization. Treat
inactive bytes in mixed segments as a fragmentation-risk diagnostic, not as proof that those bytes
are unusable or unreleasable: reuse and release behavior depends on request size, stream state, and
the configured allocator policy. The largest inactive block is likewise diagnostic rather than a
guarantee that a future allocation will succeed.

### Allocator snapshot telemetry qualification (2026-08-27)

Implementation commit `dcac9dd` and strict-comparator commit `c130663` qualified the opt-in
telemetry on the same A100 model and frozen trace items used by the selected-position-logit
optimization. Job `15168827` completed the 2,951-token smoke in 132.89 seconds, versus 132.61
seconds for its telemetry-off reference. Both runs had byte-identical 26,723,487,232-byte peak
allocation and 34,613,493,760-byte peak reservation. Its four captures took 11.85 milliseconds in
total. The strict report required exact node and edge topology plus zero tolerance for target,
node, edge, and candidate-profile values, and passed all gates.

Job `15169634` completed and saved the 8,266-token pressure artifact in 342.94 seconds before the
existing 8-GiB post-run headroom gate intentionally stopped the wave and produced Slurm exit 1.
Its peak allocation and reservation were byte-identical to the telemetry-off reference
(59,592,814,592 and 81,797,316,608 bytes), and all four snapshots took 8.85 milliseconds in total.
Its separate strict zero-tolerance report also passed every gate. At the four checkpoints, the
native allocator reported:

| Checkpoint | Allocated GiB | Reserved GiB | Inactive GiB | Fully inactive segment GiB | Inactive in mixed segments GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| After important-mask selection | 23.37 | 76.18 | 52.81 | 44.56 | 8.25 |
| Before first selected-attribution VJP | 54.82 | 76.18 | 21.36 | 10.22 | 11.14 |
| First raw selected-attribution VJP live | 54.86 | 76.18 | 21.32 | 10.22 | 11.10 |
| After selected attribution/contribution | 53.63 | 65.96 | 12.33 | 0.07 | 12.25 |

This establishes that the 76.18-GiB reservation after mask selection was mostly cached memory in
wholly inactive segments, not mostly inactive bytes interleaved with live allocations. By the
first selected-attribution VJP, the live allocation had genuinely risen to about 54.8 GiB and the
inactive remainder was split roughly evenly between wholly inactive and mixed segments. This is
evidence against treating the large reserved value alone as an OOM or fragmentation diagnosis;
request-size-aware failure analysis must use the block layout and allocator retry/OOM counters as
well.

### Dense allocator-aware CUDA headroom diagnostic (live-qualified 2026-08-28)

The opt-in `allocator_dense_joint_v1` policy requires
`instrumentation.cuda_dense_joint_pressure_telemetry=true` in addition to CUDA-memory and
allocator-snapshot telemetry. The separate flag keeps existing CUDA-memory artifacts and runtime
overhead stable. When enabled, it samples the following quantities together at every existing
CUDA-memory measurement boundary; it still takes full allocator snapshots only at the four
structural hot spots above:

```text
external[s] = max(0, device_total - device_free[s] - reserved[s])
joint_pressure[s] = active[s] + inactive_split[s] + external[s]
observed_joint_headroom = device_total - max(joint_pressure[s])
```

The receipt keeps the limiting sample, the 16 highest-pressure samples, the maximum sampled
external use, sample count, and cumulative sampling overhead. Samples include their trace point,
active measurement stack, elapsed time, and pressure components. They are boundary samples, not
continuous monitoring, and the result describes a completed work unit rather than predicting that
the next unit will or will not OOM. The receipt also retains the more pessimistic independent-max
composite, legacy peak-reserved estimate, allocator retry/OOM deltas, and structural-snapshot
validation. Active plus inactive-split bytes exceeding reserved bytes is rejected as malformed
telemetry.

The classification is `comfortable` when both the dense observed-joint and independent-max
headroom estimates meet the configured threshold and no allocator retry/OOM occurred, `watch`
when either estimate crosses below the boundary without allocator trouble, and `critical` when
allocator retries or OOM counters increase. `cuda_headroom_action: warn` records and preserves a
warning while continuing; `stop` retains the frozen hard-stop behavior. Missing or malformed
requested evidence fails closed regardless of action. A trace that actually raises OOM remains
governed separately by `stop_on_oom`.

Commit `332a1a8d2d6b87f3a19ecf229f53f1d17a60f5fe` was live-qualified on one NVIDIA A100
80GB PCIe in job `15324627`. Slurm reported `COMPLETED` with exit `0:0` and 6:46 elapsed; the
trace wall time was 341.43673443305306 seconds. The receipt retained 2,044 joint samples with
1.1856326239649206 seconds of cumulative sampling wall time, about 0.35% of trace wall time.
That ratio measures time spent inside the sampling calls, not the counterfactual runtime cost of
enabling telemetry, and must not be interpreted as a causal overhead estimate.

The limiting dense sample was
`allocator_snapshot:before_first_selected_attribution_vjp`: 58,862,236,160 active bytes,
11,958,586,880 inactive-split bytes, and 552,468,480 external bytes, for 71,373,291,520 bytes
of joint pressure and 13,720,485,888 bytes of observed headroom (about 12.78 GiB). The
independent-max composite reported 7,280,496,128 bytes of headroom (about 6.78 GiB), while the
legacy peak-reserved calculation reported 3,296,460,800 bytes (about 3.07 GiB). No allocator
retry or OOM counter increased. The resulting receipt was `watch` with action `warn` and
`should_stop=false`, so the completed item produced a durable warning and continued successfully.

Strict comparison to the prior accepted artifact found exact `df_node` and `df_edge` equality and
equality for every other checked scientific payload field: 8,362 nodes, 1,046 edges, and five
candidates. The prior trace wall time was 342.9395004948601 seconds, so this single pair shows no
visible slowdown; it is not a causal runtime comparison. The immutable artifact and execution
summary are rooted at
`/scratch/general/vast/u1653998/circuits/results/bonafide/process-witness-smart-headroom-qualification-v1-332a1a8`.

The 8-GiB number is a convention, not an empirically calibrated failure threshold. The July
benchmark introduced a 4-GiB floor in commit `2f86c5f`. Commit `39b4a21` (2026-08-20, “Add strict
T5 resource calibration ladder”) first raised it to exactly 8 GiB, and the August 21 Qwen Thinking
A100 qualification configs copied it without documenting a rationale. On the A100 receipt's
reported 79.25-GiB device total, 8 GiB is about 10.1%. It should therefore be read as a diagnostic
“watch this run” margin unless a frozen config explicitly chooses the stopping action. The live
8,266-token qualification now confirms that interpretation for this item: its independent-max
estimate crossed the convention while the denser joint observation retained about 12.78 GiB,
the trace completed without allocator trouble, and exact scientific parity held. This threshold
was a useful warning boundary here, not a failure boundary. Legacy configs that omit the policy
and action remain `peak_reserved_v1` plus `stop`, and retain their old artifact identity.

### Post-selection discovery-state compaction (live-qualified 2026-08-28)

Implementation commit `55e7bcf` adds explicit `dense_v1` and `compact_cpu_v1` storage adapters
behind a strategy-independent discovery-state interface. Checkpoint-contract fix `e536b6a`
admits only a complete, correctly ordered before/after storage snapshot pair into allocator-aware
headroom evidence. The compact adapter retains ordered selected coordinates and their raw-dtype
initial-attribution values on CPU, then releases dense discovery activations, attributions, and
the global mask from its retained GPU state. Probe and return-only modes bypass the adapter.

Sequential A100 jobs `15348279` and `15351183` ran the frozen 2,951-token reference. The strict
report at
`.../qualification-reports/context-2501-4000-dense-vs-compact-cpu-exact-v1.json` passes all gates
at zero tolerance: exact 3,094-node/2,366-edge topology, every saved target/node/edge value, all
15,470 candidate-profile values, selected-coordinate order and hash, and selected BF16 values and
hash. Its SHA-256 is
`1e6e16a67fe7bf94ba0f1941ff00132266b4fc7b09b80db7251c3fab04b837b2`.

At 2,951 tokens, compact storage released 5,182,427,882 logical bytes; allocator active blocks
dropped 5,184,588,800 bytes across the storage boundary. Peak allocated CUDA memory fell from
26,723,485,696 to 23,726,469,632 bytes (11.2%), while peak reservation remained
34,613,493,760 bytes. Trace wall time was 132.940 versus 132.640 seconds, so this pair shows no
material runtime cost.

The sequential compact ladder then produced:

| Actual context tokens | Job | Selected occurrences | Nodes / edges | Trace seconds | Peak allocated | Peak reserved | Logical bytes released | Retry / OOM | Allocator-aware result |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 8,266 | `15356053` | 92 | 8,362 / 1,046 | 343.508 | 51,873,096,704 | 81,797,316,608 | 14,516,418,376 | 0 / 0 | `comfortable`; 22,972,558,336-byte conservative headroom |
| 9,397 | `15356095` | 70 | 9,471 / 597 | 298.132 | 57,800,285,696 | 78,741,766,144 | 16,502,635,380 | 1 / 0 | `critical`; 9,053,827,584-byte conservative headroom |

The 8,266-token result is capacity-feasible under the frozen plan. The 9,397-token artifact is a
successful scientific trace and useful near-10k evidence, but it is not capacity-qualified: one
allocator retry occurred, and the legacy physical-headroom estimate was only 6,352,011,264 bytes.
Its shorter runtime reflects a different selected-neuron/pair topology and must not be interpreted
as favorable length scaling.

Compaction changes the measured owner. At 9,397 tokens, `initial_attribution` owns the 57.80-GB
global active peak; selected attribution/contribution peaks at 50.02 GB, stop-gradient
attribution/contribution at 47.74 GB, and cross-layer expansion at 35.17 GB. The retry is first
visible in important-mask selection, where inactive-split pressure peaks near 9.96 GB; the
run-wide inactive-split maximum is 17.69 GB.

Commit `658e6a8` then froze a same-commit compact `default_v1` versus
`expandable_segments_v1` allocator A/B on that same 9,397-token item. Default job `15356157`
reproduced one allocator retry and the `critical` receipt. Expandable job `15356591` completed
with zero retries/OOMs and a `comfortable` receipt. Its intended policy and observed
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` environment are bound into both the artifact
identity and runtime receipt.

The strict report
`.../qualification-reports/context-gt-8000-default-vs-expandable-segments-exact-v1.json` passes
every identity, hardware, topology, and zero-tolerance numerical gate. Both artifacts contain
exactly 9,471 nodes and 597 edges; target, node, edge, all 88,989,516 source-profile values, and
all 47,355 candidate-profile values are identical. The report SHA-256 is
`8b871948873a3aa72351518cb527ffb6e357de26a23fd480eb0a1a8d0dab5487`.

Expandable segments left peak allocation essentially unchanged (57,800,285,696 to
57,737,490,944 bytes) but reduced peak reservation from 78,741,766,144 to 65,905,098,752 bytes,
a 12,836,667,392-byte or 16.3% reduction. Conservative allocator-aware headroom increased from
9,053,827,584 to 26,803,817,984 bytes, while legacy physical headroom increased from
6,352,011,264 to 19,188,678,656 bytes. Trace time was 292.596 versus 296.771 seconds (+1.4%),
which is not a material regression in this single pair. This qualifies expandable segments as an
optional near-capacity A100 policy for this exact workload. No exact 10,000-token item exists in
qualification-v4, so the evidence remains a 9,397-token result rather than an exact-10k claim.

## Required trace semantics

For a frozen response with tokens `y[0], ..., y[n-1]`, target response position `i` means:

```text
context: prompt + y[:i]
target:  logit for y[i]
```

The current ADAG implementation forms one scalar objective by summing the selected logits before
its first attribution backward pass. Consequently, a multi-target result is **not** equivalent to
a lossless collection or union of independent single-target traces: signed effects may cancel,
and its aggregate node/edge attributions cannot later be separated by token.

Therefore, a reusable scientific trace has exactly one target token. Multiple targets may still
be evaluated together as a bounded **benchmark-only** window to measure Jacobian scaling. Every
artifact records its objective semantics and the mapping from each selected target to:

- response token position;
- absolute sequence prediction position;
- target token ID and decoded token text.

The runner must not generate an extra token. Later response tokens may be present in a padded
tensor only when causally masked; the initial implementation should instead truncate each input
after the final token in the requested window to reduce work and make the boundary explicit.

The primary objective is the actual teacher-forced token. A later benchmark wave may compare an
actual-token-only objective with actual-plus-top-alternative output profiles, but that comparison
must not silently change the main benchmark configuration.

## Reusable artifact contract

One trace unit is one prompt/response and either one reusable target or one benchmark-only target
window. Store it as a directory containing:

```text
trace-unit/
  manifest.json
  metrics.json
  circuit_data.pkl.gz
```

`manifest.json` records:

- artifact schema version and run/trace-unit IDs;
- BonaFide row/question IDs and condition;
- exact model ID/revision, tokenizer revision, and code commit;
- prompt and response text plus stable hashes;
- input token IDs, attention mask, and assistant-token boundary;
- target response positions, absolute prediction positions, token IDs, and token text;
- complete `ADAGConfig`, target-window configuration, and pruning settings;
- start/end time, wall time, peak CUDA allocation/reservation, peak host RSS, and device metadata;
- node/edge counts and serialized byte counts;
- completion status and error information.

`circuit_data.pkl.gz` is a compressed `CircuitData` payload, preserving the repository's native node
and edge tables, occurrence identities, activation values, input-attribution profiles, and any
output-contribution axes that the current implementation retains. `metrics.json` holds the small,
machine-readable performance record independently of Python/pickle loading.

The manifest must declare either `reusable_single_target` or `benchmark_only_summed_logits`. A
multi-target benchmark artifact must never be presented to downstream code as independently
reusable per-token traces. If later work changes ADAG to retain a target axis throughout the
objective and edge attribution path, the artifact schema must be versioned rather than silently
changing these semantics.

Artifacts are immutable after successful completion. A manifest is written atomically, and a
completed trace unit can be skipped on resume only when its input/config hashes match.

No graph merging is part of this format. A downstream loader may concatenate tables or construct
graphs while retaining `trace_unit_id` and target provenance.

## Benchmark waves

Run waves sequentially. Review each wave before expanding the matrix.

### Wave 0: correctness and cold-start smoke test

- one short Qwen Instruct example;
- one teacher-forced target token;
- `batch_size=1`;
- verify token alignment, finite attributions, artifact round-trip, and offline model loading;
- separately report model-load time and warm trace time.

### Wave 1: mixed-prompt resource sample

Use approximately ten prompts chosen by **tokenized** prompt and response lengths, not visible
character count. Include short, median, long, and one high-but-valid tail case, together with at
least one known hint anchor. If both Qwen checkpoints are candidates for a full run, benchmark
them separately or include enough Thinking examples to cover their longer response tail.

Use one target per response. For timing across varied sequence lengths, select the final non-EOS
response token so the traced prefix actually spans the full response. Keep the short annotated
hint/bottleneck anchor in Wave 0 as a semantic-alignment check. The goal here is to estimate
variation across contexts while keeping target count fixed.

### Wave 2: target-window scaling

On a smaller fixed set of representative short/median/long contexts, sweep target-window sizes
progressively, for example:

```text
1 -> 2 -> 4 -> 8 -> 16 -> 32
```

Stop expanding a context after an out-of-memory error, a configured wall-time limit, or loss of
the required GPU-memory headroom. Do not submit the whole matrix at once. Each window is an
independent resumable trace unit. Widths greater than one are marked
`benchmark_only_summed_logits`; the width-one baseline remains an ordinary single-target trace.
Multi-target results measure scaling but are not production traces to be split or merged later.

### Wave 3: objective-width scaling, only if needed

Compare the actual-token-only objective with the intended alternative-logit profile width on a
small fixed subset. This wave exists because top-`k` width can change both Jacobian cost and the
quality of output-contribution profiles.

### Wave 4: corpus projection and go/no-go

Tokenize the candidate BonaFide reference corpus on CPU and combine its length/target distribution
with measured scaling curves. Report:

- projected GPU-hours;
- projected one-worker and planned-concurrency wall time;
- p50/p90/p95 and worst-case trace-unit time and memory;
- projected artifact storage;
- the fraction of contexts outside the measured range.

Proceed to the reference-corpus run only if the projected workload fits the agreed budget (the
current working target is roughly one to two days of tracing), p95 examples retain safe memory
headroom, and target-aware artifacts pass correctness checks.

## Measurements

Record for every trace unit:

- prompt, forced-response, total-input, and target token counts;
- model-load and trace wall times;
- peak CUDA allocated and reserved bytes;
- peak host RSS;
- GPU name and total memory;
- node and edge counts;
- output bytes by file and total;
- tracing/pruning configuration;
- success, timeout, OOM, or other failure state.

The benchmark report must distinguish cold cache/model load from steady-state tracing and must
report both GPU-hours and parallel wall-clock projections. Per-artifact `metrics.json` contains
the trace measurements known before atomic serialization; the append-only benchmark summary also
records serialization time, total unit time, and completed artifact bytes.

## Initial execution constraints

- Use `batch_size=1` until variable-length target batching is fixed and checked against unbatched
  output.
- Keep one model resident while processing multiple trace units in a worker.
- Use exact cached model revisions from `ADAG_BONAFIDE_NAIVE_PILOT.md`.
- Run GPU/model work through the project benchmark launcher and ordinary Slurm submission, never
  through the interactive-development helper.
- Write temporary files to allocation-local scratch and completed artifacts to project VAST
  scratch.
- Do not mix Qwen Instruct and Qwen Thinking traces in one clustering corpus or cost model without
  reporting them separately.

## Implemented entry points

The first Instruct benchmark is frozen in
`scripts/bonafide/manifests/qwen3_4b_instruct.json`. Its mixed-length wave has ten exact
chat-template response lengths from 145 to 1,992 tokens and retains all annotations for the three
initial Instruct anchors. Its separate scaling wave uses one 170-token response with target
widths 1, 2, 4, 8, 16, and 32. Wave 2b uses that same response for 16 independent
single-target traces at evenly spaced positions spanning the response. Wave 2c excludes the
Wave 2/2b reference response and gives each of the other nine Wave 1 responses its own named
wave. Each Wave 2c wave contains eight independent single-target traces: one deterministically
sampled position from each contiguous response-position octile.

Wave 2c uses a recorded `sha256-rejection-v1` sampler and stable seed. Each item's
`target_selection.sampling` records the stratum bounds and size, its inclusion probability
`1 / stratum_size`, and projection weight `stratum_size`. These fields are also copied into the
run summary as `target_sampling` and into the compact artifact manifest under
`source_target_selection`, so timing and graph statistics can be analyzed without an implicit
join to the frozen benchmark manifest. The eight sampled observations are a probability sample
of positions within one response; the projection weights do not remove token-dependent variance
or make eight positions equivalent to tracing the full response. In particular, Wave 2c has
`n_h = 1` observation per stratum, so the design supplies no within-stratum variance estimate and
cannot by itself produce a design-based variance or confidence interval. Use repeated independent
seeds or at least two independently sampled positions per stratum before reporting such intervals.

The pinned run config enables `first_wave_item_full_trace_discard` only for wave IDs beginning
with `wave2c-`; Wave 1, Wave 2, and Wave 2b do not warm up. After loading the model, each selected
Wave 2c job performs one complete ADAG trace using the fixed first item in the full wave, discards
it, runs garbage collection, empties the CUDA allocator cache, and then measures all planned work
items including that first item again on a fresh run. The fixed source does not change on resume
or when `ONLY_ARTIFACT_ID` selects another item. This prevents model/Jacobian cold-start cost from
being assigned to one probability-weighted observation. Runtime identity includes the normalized
prefix policy plus the warm-up source artifact ID, work-item hash, and target selection.
The summary JSONL records a `discarded_trace_warmup` row with its source, status, wall time, and
instrumentation, while each measured summary record and compact artifact records the same warm-up
provenance. A failed, OOM, or cleanup-failed warm-up records its failure before re-raising the
original error, so the CLI/Slurm job exits nonzero and no measured item is written.

Regenerate the manifest deterministically after changing the dataset or selection policy:

```bash
source scripts/chpc_env.sh
"$UV_PROJECT_ENVIRONMENT/bin/python" -m scripts.bonafide.manifest \
  --csv BonaFide.csv \
  --output scripts/bonafide/manifests/qwen3_4b_instruct.json \
  --revision cdbee75f17c01a7cc42f958dc650907174af0554 \
  --sample-count 10 \
  --anchor-id f44d85b57fe09ddc \
  --anchor-id 23068c9e9e56a270 \
  --anchor-id a92b84d0920c5100 \
  --wave2-id f44d85b57fe09ddc \
  --wave2b-target-count 16 \
  --wave2c-stratum-count 8 \
  --wave2c-seed bonafide-wave2c-v1
```

Check the plan without loading the model:

```bash
"$UV_PROJECT_ENVIRONMENT/bin/python" -m scripts.bonafide.runner \
  --config scripts/bonafide/configs/qwen3_4b_instruct.json \
  --manifest scripts/bonafide/manifests/qwen3_4b_instruct.json \
  --wave wave1-mixed-final-token \
  --dry-run
```

The GPU launcher accepts exactly one named wave. First run the shortest Wave 1 item alone as the
operational Wave 0 finite-attribution/artifact check. Then run the full mixed wave; exact-identity
resume skips the already completed item. Submit Wave 2 only after reviewing Wave 1 timings and
memory:

```bash
sbatch --export=ALL,MANIFEST="$PWD/scripts/bonafide/manifests/qwen3_4b_instruct.json",WAVE=wave1-mixed-final-token,ONLY_ARTIFACT_ID=trace-851cabe5820bb7b666765f13 \
  scripts/bonafide/benchmark_tracing.sbatch

# After inspecting the single completed trace:
sbatch --export=ALL,MANIFEST="$PWD/scripts/bonafide/manifests/qwen3_4b_instruct.json",WAVE=wave1-mixed-final-token \
  scripts/bonafide/benchmark_tracing.sbatch

# Later, after the Wave 1 review:
sbatch --export=ALL,MANIFEST="$PWD/scripts/bonafide/manifests/qwen3_4b_instruct.json",WAVE=wave2-progressive-target-window \
  scripts/bonafide/benchmark_tracing.sbatch

# After reviewing the summed-window scaling results:
sbatch --export=ALL,MANIFEST="$PWD/scripts/bonafide/manifests/qwen3_4b_instruct.json",WAVE=wave2b-independent-target-positions \
  scripts/bonafide/benchmark_tracing.sbatch

# Wave 2c wave IDs are listed by the manifest generator and runner dry-run output. Submit one
# wave per job so the model loads once and remains resident for its eight sampled targets:
sbatch --export=ALL,MANIFEST="$PWD/scripts/bonafide/manifests/qwen3_4b_instruct.json",WAVE=wave2c-stratified-independent-01-bf-2ed391444282be41b715 \
  scripts/bonafide/benchmark_tracing.sbatch
```

The runner loads the pinned local snapshot once, processes trace units at batch size one, resumes
only exact identity matches, and keeps that model resident for every item in the selected wave.
Each Wave 2b item is saved as a separate scientifically reusable single-target artifact; the run
does not sum targets or merge graphs. The runner stops after an OOM, an over-budget trace, or
insufficient CUDA headroom. Completed artifacts go to the VAST-backed
`$CIRCUITS_RESULTS_DIR` by default. No command above merges graphs.

The first Wave 2c job is the 145-token response and serves as the instrumentation smoke. Inspect
its eight records before submitting the remaining eight per-prompt waves; all Wave 2c outputs are
also separate scientifically reusable single-target artifacts.

`max_trace_seconds` is a post-completion expansion gate, not an asynchronous CUDA-kernel timeout.
The four-hour Slurm limit remains the hard job cap. If the long-tail Wave 1 items approach that
limit, submit them individually with `ONLY_ARTIFACT_ID` so each has its own allocation boundary
and completed shorter artifacts remain intact.

## Recorded A100 results (2026-07-19)

These measurements used commit `0581d44`, Qwen3-4B-Instruct-2507 at the pinned revision, and one
NVIDIA A100 80GB PCIe on `notch369`. Wave 1 job `14060289` completed all ten final-token traces in
12m23s of allocation time. The traces themselves used 717.2 seconds in total; the median was
42.1 seconds. The 1,992-response-token / 2,259-total-token tail took 337.8 seconds and peaked at
65.6 GiB reserved CUDA memory, leaving 13.7 GiB headroom. The ten compact artifacts occupy 2.7 MB
and passed checksum-aware reload validation.

Wave 2 reused the 170-token Wave 1 result as the width-one baseline. Widths greater than one were
run individually as benchmark-only summed-logit objectives, with a fresh model load for each job:

| Target width | Trace seconds | Peak CUDA reserved (GiB) | Nodes | Edges |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 28.34 | 10.53 | 351 | 6,892 |
| 2 | 25.68 | 10.96 | 335 | 4,181 |
| 4 | 18.63 | 11.46 | 309 | 1,669 |
| 8 | 13.95 | 12.25 | 293 | 919 |
| 16 | 22.86 | 15.24 | 295 | 456 |
| 32 | 47.88 | 19.18 | 321 | 794 |

The five new Wave 2 artifacts occupy 440 KB and passed checksum-aware reload validation. Their
Slurm jobs were `14060720`, `14060736`, `14060824`, `14060839`, and `14060911`; all exited `0:0`.

Interpret this wave narrowly. Memory grows modestly with summed-objective width, but runtime and
graph size are non-monotonic because changing the summed objective changes attributions and which
nodes survive pruning. These results do **not** show that 32 independent, provenance-preserving
per-token traces cost the same as one width-32 trace. The production cost projection still needs
independent single-target traces or a tracing implementation that shares computation while
retaining a distinct target axis.

Wave 2b measures that independent-target cost directly. It traces response positions
`0, 11, 23, 34, 45, 56, 68, 79, 90, 101, 113, 124, 135, 146, 158, 169` from the same
170-token example in one model-resident run. The positions include both response endpoints and
have gaps of 11 or 12 tokens. Compare per-position runtime, memory, and graph size; sum the trace
times for the sampled workload, then interpolate across response position and integrate over all
170 positions to project the no-sharing cost of fully tracing this response.

The items run in ascending position order, so position 0 is also the first trace after model load
and may include one-time CUDA/kernel warm-up. Keep its graph and memory measurements, but exclude
its time from the position-cost fit unless a warm repeat shows that the first-item effect is
negligible.

Wave 2b job `14061259` used commit `44e229f` and completed all 16 items in 14m41s of allocation
time. The sampled traces used 843.8 seconds, peaked at 12.38 GiB reserved CUDA memory, produced
4.30 MB of compact artifacts, and all passed checksum-aware reload validation:

| Response position | Input tokens | Trace seconds | Peak CUDA reserved (GiB) | Nodes | Edges |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 78 | 56.45 | 8.57 | 417 | 21,532 |
| 11 | 89 | 41.85 | 8.67 | 288 | 8,186 |
| 23 | 101 | 35.71 | 8.85 | 269 | 7,132 |
| 34 | 112 | 57.31 | 9.07 | 476 | 24,356 |
| 45 | 123 | 51.24 | 9.23 | 356 | 10,141 |
| 56 | 134 | 29.89 | 9.69 | 230 | 5,035 |
| 68 | 146 | 73.58 | 9.85 | 687 | 31,782 |
| 79 | 157 | 41.77 | 9.96 | 382 | 6,948 |
| 90 | 168 | 39.10 | 10.43 | 354 | 5,540 |
| 101 | 179 | 55.08 | 10.35 | 411 | 11,206 |
| 113 | 191 | 218.03 | 12.38 | 1,438 | 131,851 |
| 124 | 202 | 19.82 | 11.31 | 295 | 8,060 |
| 135 | 213 | 29.89 | 10.93 | 293 | 5,762 |
| 146 | 224 | 34.34 | 10.29 | 349 | 6,295 |
| 158 | 236 | 31.33 | 10.39 | 340 | 5,813 |
| 169 | 247 | 28.44 | 10.55 | 351 | 6,892 |

For a sensitivity projection, position 0 was excluded from the timing fit and assigned the warm
position-11 time. Linear interpolation across the other samples projects 9,030 seconds (2.51
A100-hours) for all 170 independent targets if the position-113 bottleneck represents its local
interval. Interpolating without that outlier and then restoring its measured cost at position 113
projects 7,126 seconds (1.98 A100-hours) if the bottleneck is isolated. Treat 2.0--2.5 A100-hours
as a scenario range for this one response, not a confidence interval or a corpus-wide estimate.

Within these 16 samples, trace time tracks retained edge count much more strongly than warm input
length (descriptive Pearson correlations `r=0.988` and `r=-0.014`, respectively). Position 113
alone took 218 seconds and retained 131,851 edges, while position 124 took 19.8 seconds and retained
8,060. A length-only resource model is therefore inadequate; more responses need position samples
to estimate the prevalence of these graph-complexity bottlenecks.

### Wave 2c: instrumented stratified position sample

Wave 2c used commits `76ed934` (stage/workload instrumentation) and `cc5d63c` (sampling,
provenance, and warm-up semantics). Jobs `14062598`, `14062748`, `14062752`, `14062758`,
`14062759`, `14062760`, `14062763`, `14062764`, and `14062766` ran one response per job on the
four lab A100s. All nine discarded warm-ups and all 72 measured traces completed with exit code
zero; there were no OOM, error, or stop-gate records. The jobs consumed 81m22s (1.36 A100-hours)
of allocation time in total. After the 145-token smoke gate, the remaining eight jobs completed in
28m42s of parallel wall time.

The measured traces used 4,172.3 seconds, the discarded warm-ups used 367.1 seconds, model loads
used 41.0 seconds, and serialization used 11.3 seconds. Trace time ranged from 17.1 to 278.6
seconds (median 48.7; 90th percentile 95.2). Peak reserved CUDA memory ranged up to 57.10 GiB
(median 11.40; 90th percentile 23.03), so even the sampled position 1,823 of the 1,992-token
response retained about 23 GiB of A100 headroom. The 72 compact artifacts occupy 43.1 MiB and all
passed checksum-aware integrity validation as numerically valid, scientifically reusable,
single-target traces.

| Response tokens | Sample median (s) | Sample max (s) | Max peak CUDA (GiB) | Projected full-response A100-hours |
| ---: | ---: | ---: | ---: | ---: |
| 145 | 27.7 | 47.4 | 10.20 | 1.15 |
| 159 | 44.4 | 58.9 | 11.13 | 1.89 |
| 161 | 39.5 | 54.3 | 10.66 | 1.78 |
| 244 | 49.2 | 68.3 | 13.42 | 3.28 |
| 285 | 45.3 | 74.4 | 14.28 | 3.92 |
| 482 | 55.3 | 90.7 | 18.14 | 7.40 |
| 715 | 59.1 | 158.7 | 23.06 | 15.09 |
| 963 | 54.9 | 95.7 | 26.34 | 15.97 |
| 1,992 | 113.8 | 278.6 | 57.10 | 67.47 |

The last column is the stratified Horvitz--Thompson point estimate
`sum(stratum_size * sampled_trace_seconds)`. It totals 117.94 A100-hours for every response token
in these nine prompts and projects about 3.24 GiB of compact artifacts. It does not include the
170-token Wave 2/2b reference response. With four perfectly utilized lab A100s, the nine-prompt
compute total is about 29.5 wall-clock hours. These are point estimates, not confidence intervals:
one observation per stratum supplies no design-based variance estimate, and the 1,992-token prompt
alone contributes 67.47 hours.

The instrumentation separates the two cost drivers that were confounded in Wave 2b:

- Actual input length has overall Pearson correlation `r=0.824` with trace time.
- After linearly controlling for input length, selected-neuron count and planned Jacobian chunk
  count have partial correlations `r=0.738` and `r=0.703` with trace time.
- A descriptive in-sample linear model using input tokens plus planned chunks has `R^2=0.838`;
  adding input-length curvature and an input-by-chunk interaction raises it to `R^2=0.929`.
- Layer-pair Jacobians consume 59.5% of measured trace time and their measured stage time has
  `r=0.985` correlation with total trace time.
- The early workload predictors become available after a median 0.161 seconds (90th percentile
  0.270; maximum 0.853), only 0.38% of total trace time at the median. They are therefore early
  enough to support a future resource governor before expensive cross-layer Jacobians begin.

Treat the correlations and regressions as descriptive calibration on 72 deliberately sampled
positions, not a validated runtime predictor. In particular, planned chunks alone have weak
pooled correlation because the cost of one chunk grows with context length. The 1,992-token
response makes this concrete: position 1,561 took 278.6 seconds with 406 planned chunks, while
position 1,823 took 172.0 seconds with 210 chunks but used more memory (57.10 versus 49.03 GiB).

The saved graphs are non-empty and materially variable: 200--2,161 nodes and 1,812--29,842 edges
per sampled target. This establishes that the tracing stage is producing substantial reusable
internal structure across positions. It does not yet establish that ADAG clustering will recover
clean or stable semantic entities; that is the next downstream experiment and does not require
retracing these 72 targets.

## After the performance check

If resource use is acceptable, trace the reference corpus incrementally (for example 50, then
100, then 150 contexts). Clustering convergence, description quality, synthetic prompt expansion,
and any graph construction remain downstream decisions and can reuse the saved trace artifacts
without retracing.
