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
widths 1, 2, 4, 8, 16, and 32.

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
  --anchor-id a92b84d0920c5100
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
```

The runner loads the pinned local snapshot once, processes trace units at batch size one, resumes
only exact identity matches, and stops progressive expansion after an OOM, an over-budget trace,
or insufficient CUDA headroom. Completed artifacts go to the VAST-backed
`$CIRCUITS_RESULTS_DIR` by default. No command above merges graphs.

`max_trace_seconds` is a post-completion expansion gate, not an asynchronous CUDA-kernel timeout.
The four-hour Slurm limit remains the hard job cap. If the long-tail Wave 1 items approach that
limit, submit them individually with `ONLY_ARTIFACT_ID` so each has its own allocation boundary
and completed shorter artifacts remain intact.

## After the performance check

If resource use is acceptable, trace the reference corpus incrementally (for example 50, then
100, then 150 contexts). Clustering convergence, description quality, synthetic prompt expansion,
and any graph construction remain downstream decisions and can reuse the saved trace artifacts
without retracing.
