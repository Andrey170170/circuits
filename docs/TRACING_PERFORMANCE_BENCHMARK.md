# Target-aware tracing performance benchmark

Status: implementation contract for the first ADAG–BonaFide GPU run.

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
single-target traces at evenly spaced positions spanning the response.

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
  --wave2b-target-count 16
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
```

The runner loads the pinned local snapshot once, processes trace units at batch size one, resumes
only exact identity matches, and keeps that model resident for every item in the selected wave.
Each Wave 2b item is saved as a separate scientifically reusable single-target artifact; the run
does not sum targets or merge graphs. The runner stops after an OOM, an over-budget trace, or
insufficient CUDA headroom. Completed artifacts go to the VAST-backed
`$CIRCUITS_RESULTS_DIR` by default. No command above merges graphs.

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

## After the performance check

If resource use is acceptable, trace the reference corpus incrementally (for example 50, then
100, then 150 contexts). Clustering convergence, description quality, synthetic prompt expansion,
and any graph construction remain downstream decisions and can reuse the saved trace artifacts
without retracing.
