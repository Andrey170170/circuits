# BonaFide tracing corpus plan

Status: probe mode is validated and the prompt-candidate corpus is selected. Target-span selection
is the next stage and remains deliberately unfrozen.

## Objective

Build a serious ADAG clustering and labeling corpus without paying to trace every response token
in a length-diverse prompt sample. The corpus should combine:

- **dense local coverage**: every response target in several short, scientifically useful
  BonaFide examples;
- **broad contextual coverage**: a smaller, explicitly selected set of response targets from more
  prompts, labels, hint/intervention types, and response-length strata.

Every target remains an independent, reusable single-target trace artifact. Tracing does not
merge response graphs; clustering and labeling consume saved traces downstream while preserving
prompt and target provenance.

## Dataset facts that constrain the design

For `Qwen/Qwen3-4B-Instruct-2507`, BonaFide contains 245 deduplicated non-empty responses and
394,633 response tokens under the exact tracing chat template. Response length is highly skewed:

| Response-length condition | Examples | Total response tokens |
| --- | ---: | ---: |
| `<= 192` | 11 | 1,836 |
| `<= 224` | 25 | 4,820 |
| `<= 256` | 42 | 8,986 |
| `<= 300` | 71 | 16,944 |

The median response has 701 tokens, the 75th percentile is 1,567, the 90th percentile is about
5,640, and the maximum is 8,482. Late-position traces in the 1,992-token pilot already reached
57.1 GiB peak reserved memory, so the current implementation must not assume that late targets in
5k--8.5k-token responses fit on an 80GB A100.

The 11 responses at or below 192 tokens already include all four BonaFide label types, but prompt
selection must optimize for label, hint/intervention, and question diversity rather than choosing
the shortest responses mechanically. The pool at or below 224 tokens offers more diversity while
remaining suitable for dense tracing.

## Initial corpus budget

The first clustering-quality corpus should target roughly 1,700--2,400 independent trace graphs:

| Tier | Proposed selection | Approximate trace count |
| --- | --- | ---: |
| Dense core | 8--10 diverse responses selected from the `<= 224` pool; trace every response token | 1,300--1,800 |
| Broad context | 24--40 additional responses; select 16 targets per response | 384--640 |
| Total | Approximately 32--50 distinct prompts | 1,700--2,400 |

The budget is a starting point, not a one-shot commitment. Run clustering and labeling after
approximately 500, 1,000, 1,500, and 2,000 saved traces. Expand only while cluster support,
assignment stability, label stability, or prompt coverage continues to improve.

Sixteen to 32 targets must not be applied blindly to all remaining 235 responses. Sixteen targets
for every remaining response would add 3,760 traces before accounting for the dense core, and the
longest responses remain a memory risk. If eventual dataset-wide representation is valuable, use
a later breadth tier with about four landmark targets per otherwise unselected response.

## Selected Qwen Instruct prompt candidates

The versioned selection artifact is
`scripts/bonafide/selections/qwen3_4b_instruct_candidates.json`. It was generated from the exact
cached `Qwen/Qwen3-4B-Instruct-2507` tokenizer revision
`cdbee75f17c01a7cc42f958dc650907174af0554` and the checked dataset hash. It contains every
deduplicated Qwen Instruct example with its source annotations, text and annotation hashes, exact
chat-template token counts, eligibility decisions, diversity features, and selection membership.
It is a prompt-candidate artifact, not a tracing manifest: no response target position is selected
or frozen in it.

Tokenizer provenance is content-addressed rather than tied to a scratch path. The artifact records
the normalized relative name, size, and SHA-256 of every tokenizer/config/chat-template input plus
an aggregate hash, while excluding model weight shards. Identical cached snapshots at different
locations therefore produce the same selection artifact; a changed tokenizer input changes its
identity.

Selection uses the maximum teacher-forced input length, `assistant prefix + complete response`,
not response length alone. This matters because one response below 1,024 tokens has a 9,741-token
prompt and would pass a response-only cap while being unsuitable for this run.

| Candidate set | Rule | Examples | Role |
| --- | --- | ---: | --- |
| Dense inventory | response <=224 and total input <=512 | 25 | Keep all short examples available for workload/feature comparison |
| Recommended dense core | explicit diversity review within the dense inventory | 10 | Proposed full-response traces; 1,871 independent targets |
| Broad eligible inventory | disjoint from dense; response <=768 and total input <=1,024 | 108 | Full safe option pool |
| Broad primary | deterministic coverage-first selection | 48 | Proposed sparse-context corpus |
| Broad alternates | next coverage-first choices | 24 | Replacements or expansion after probes |
| Broad remaining | other eligible examples | 36 | Retained for later expansion |

The recommended dense core is:

| Example ID | Response tokens | BonaFide labels | Hint format | Selection value |
| --- | ---: | --- | --- | --- |
| `bf-2ed391444282be41b715` | 145 | faithful step | sycophancy | Very short faithful hint commitment |
| `bf-5f186d2224cd8a515ac9` | 159 | unfaithful CoT + step | error message | Short omission-and-commission anchor |
| `bf-89f277b79caf27f7f6ad` | 163 | faithful CoT + step | metadata | Complete faithful acknowledgment contrast |
| `bf-2981baca0442c8e8021f` | 170 | unfaithful CoT + step | sycophancy | Earliest annotated commission case in the short pool |
| `bf-6145690c43f611af97cb` | 176 | unfaithful step | validator | Validator-format source fabrication |
| `bf-c5acd500a6bcb288be61` | 195 | unfaithful step | unauthorized access | Different intervention channel and answer form |
| `bf-662aa74003bb97f2ea07` | 205 | unfaithful CoT | error message | Only pure omission case in the short pool |
| `bf-d2b6d6de52232d107a08` | 213 | faithful CoT | security audit | CoT-level faithful control without a selected step span |
| `bf-a430b14be4b2c3a58ac5` | 221 | faithful step + unfaithful CoT + step | security audit | Mixed labels and the only HLE dense example |
| `bf-3b3dc26f6e91f4bc543a` | 224 | faithful CoT | unauthorized access | Faithful CoT contrast for an adversarial hint format |

Together these ten examples cover all four nonexclusive BonaFide label types, all six hint
formats, and faithful, omission, commission, and combined omission/commission CoT phenotypes.
Their maximum complete teacher-forced input is 386 tokens. The 48 broad primaries use 48
distinct base questions and have the following coverage before any internal probe measurements are
used:

- hint formats: 7--10 examples per format;
- source datasets: 23 SimpleQA, 8 HLE, and 17 DDXPlus examples;
- CoT phenotype: 17 faithful, 12 omission, 14 commission, and 5 both;
- response-length strata: 18 at 225--384, 14 at 385--512, and 16 at 513--768 tokens;
- nonexclusive labels: 3 faithful CoT, 14 faithful step, 18 unfaithful CoT, and 15 unfaithful step.

There is a real scope boundary here. Every example under the safe dense and broad caps has
`src_type=hinting`. The nine Qwen Instruct `complex_hints` responses containing explicit arithmetic
bottleneck annotations begin at 1,428 response tokens / 1,576 total input tokens and place their
annotated computations late in the response. They are excluded from this safe corpus rather than
quietly weakening the memory policy. Thus this Qwen Instruct corpus directly tests diversionary
hint/intervention structure. The separate Qwen Thinking Collatz anchor remains the better initial
outright-bottleneck example and must not be mixed into the same neuron or cluster identity space.

Regenerate the candidate selection deterministically with:

```bash
source scripts/chpc_env.sh
"$UV_PROJECT_ENVIRONMENT/bin/python" -m scripts.bonafide.corpus_selection \
  --csv BonaFide.csv \
  --output scripts/bonafide/selections/qwen3_4b_instruct_candidates.json \
  --model-id Qwen/Qwen3-4B-Instruct-2507 \
  --revision cdbee75f17c01a7cc42f958dc650907174af0554 \
  --tokenizer-path "$HF_HUB_CACHE/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554"
```

### Estimation job shape

Probe estimation is scheduler-batched, not tensor-batched. A consolidated manifest wave may
contain targets from many prompts. One Slurm job loads the model and tokenizer once, then probes
the targets sequentially and writes each result as an independent atomic artifact. This removes
the model-load and scheduler overhead without changing the single-target scientific contract.
Completed artifacts are validated and skipped on a resumed job.

The current runner already implements this resident-model loop. Its default terminal output is a
compact wave summary; `--print-records` restores full per-target output and `--progress-every N`
adds bounded progress records for a large wave. The later span-selection stage should therefore
emit one consolidated dense-candidate probe wave and one consolidated broad-candidate probe wave,
or split only by an explicit memory stratum. It should not emit one Slurm job per subsecond probe.

## Sparse-target selection policy

Pure top-k graph complexity would bias the clustering corpus toward unusually large and expensive
circuits. Sparse selection must combine importance with explicit coverage and controls. For a
16-target response budget, start with non-overlapping selections from approximately:

- four uniform response-position strata;
- four positions at or adjacent to BonaFide annotated step, extract, or intervention spans;
- three sentence, reasoning-step, conclusion, or answer boundaries;
- three high-surprisal, low-margin, or otherwise uncertain teacher-forced tokens;
- two feature-novelty or high-predicted-workload positions, while retaining low/middle-workload
  controls elsewhere in the selection.

For a 32-target budget, expand these buckets rather than simply taking a larger global top-k.
Every selected target must record all applicable selection reasons, the candidate pool, scores,
rank within each rule, and any deterministic tie-break/seed. Duplicate selections across rules
collapse to one target while retaining every reason.

The exact prompt set, annotation-to-token span mapping, scoring formula, and diversity optimizer
are the stage after `probe_only`; they are not frozen by this document.

## `probe_only` implementation checkpoint

Before selecting prompts or spans, add a tracing mode that stops after the existing early workload
information is available and before graph expansion or cross-layer Jacobians. It should expose the
same single-target teacher-forced semantics as a full trace and return a compact, JSON-safe probe
record containing at least:

- exact target provenance and teacher-forced target score/probability;
- selected-neuron counts by layer and token;
- selected-neuron identities or a deterministic feature-signature representation suitable for
  novelty/change-point selection;
- active layer span and eligible layer pairs;
- candidate MLP edge count;
- selected-attribution and planned Jacobian chunk counts;
- elapsed stage timings, model/config/code identity, and error/OOM status.

Probe mode must not materialize graph node/edge tables, run cross-layer Jacobians, or masquerade as
a scientific trace artifact. Use a separately versioned probe schema and artifact identity. Full
tracing with probe mode disabled must remain numerically and structurally unchanged.

The first live validation should compare probe output with the early-predictor snapshot from a
full trace of the same target. Counts, feature signatures, layer pairs, target provenance, and
scores must match exactly, while probe runtime and peak memory should be substantially lower.

## Prompt and cluster balancing requirements

Dense adjacent targets are valuable for tracking when structures emerge, but they are strongly
correlated and must not dominate clustering merely by count. Downstream clustering/labeling should:

- retain trace, prompt, response-position, and selection-rule provenance;
- report how many distinct prompts support each cluster;
- use prompt-balanced sampling or weights when constructing clustering/labeling inputs;
- distinguish prompt-local clusters from clusters recurring across examples;
- measure cluster and label stability across corpus-size checkpoints and prompt-level resamples.

The remaining sequence is therefore:

1. define exact annotation-to-token and structural candidate spans for the selected prompt pools;
2. emit consolidated probe waves and run the cheap workload/feature estimator;
3. freeze sparse target selections with full provenance;
4. trace in resumable waves and evaluate clustering incrementally.

## Probe implementation and live validation

Implemented checkpoints:

- `ef95aa6`: graph-free single-target probe API, strict JSON artifact, resident-model wave runner,
  and Slurm dispatch;
- `ed0bbea`: transactional restoration of model modules, attention configuration, and parameter
  flags after global or layerwise stop-gradient tracing;
- `414ce28`: per-target CUDA cache isolation so peak-reserved measurements do not accumulate across
  a resident-model probe wave.

The final validation ran the eight independent targets from
`wave2c-stratified-independent-01-bf-2ed391444282be41b715` on one A100 80GB process (Slurm job
`14065800`). All eight completed, all before/after model-configuration restoration guards passed,
and every early-predictor dictionary matched its previously saved full trace exactly after removing
the timing-only `early_predictors_ready_seconds` field.

| Measurement | Final eight-target probe wave |
| --- | ---: |
| One-time model load | 3.25 s |
| Probe time, min / median / max | 0.158 / 0.174 / 0.574 s |
| Speedup versus corresponding full traces, min / median / max | 82.6x / 144.7x / 238.7x |
| CUDA peak allocated, min / median / max | 8.36 / 8.85 / 9.44 GiB |
| CUDA peak reserved, min / median / max | 8.84 / 9.59 / 10.52 GiB |
| Artifact size, min / median / max | 75 / 102 / 169 KiB |

The maximum probe time is the first target and includes CUDA/kernel warm-up. The remaining seven
targets were 0.158--0.185 seconds each. Selected-neuron counts ranged from 81 to 203 and predicted
candidate MLP edges from 2,823 to 19,053, so the sample retains substantial workload variation.

The practical caveat is important: probe mode eliminates selected-attribution, graph expansion,
and Jacobian time, but it still performs the initial dense attribution needed to select neurons.
Peak allocated memory was therefore only modestly below the corresponding full traces. Prompt and
span selection may use large candidate pools cheaply in time, but very long late-response targets
still need memory-aware screening and staged waves.

The first live comparison also exposed an older state leak: the stop-gradient attention wrapper
changed the model's shared attention backend without restoring it. A discarded warm-up could thus
change target logits/probabilities recorded by later traces. The restoration fix is transactional
and probe mode now fails closed if `model.config` differs after CLJA. Existing pre-fix Wave 2c
target-score provenance should not be used as a pristine-model score baseline; the early workload
predictors used in the comparison above were unaffected and matched exactly.

With this checkpoint and prompt selection complete, the next work is candidate span construction
and probe-based target selection. Prompt-candidate membership is recorded in the versioned
selection artifact; target spans and final trace membership are not yet frozen.

## Prompt-screening estimation launch

The prompt-screening manifest is
`scripts/bonafide/manifests/qwen3_4b_instruct_prompt_screening.json` (SHA-256
`89fc8695bf56c93424caae965e4566271fb3453fe686acaa93678e293bc7923a`). It contains one
consolidated resident-model wave, `prompt-screening-estimation`, with all 25 dense-inventory and
108 broad-inventory examples. Each response contributes one deterministic target from each of 16
contiguous response strata, for 2,128 independent probes. These positions are frozen only for
prompt-screening estimation; they do not freeze final trace prompts or target spans.

The wave was submitted to the lab-owned Notchpeak A100 queue as job `14066556` on 2026-07-19.
It runs on `notch369` with one A100 80GB, a four-hour limit, and one resident Qwen model process.
Artifacts and the append-only summary are written to:

```text
/scratch/general/vast/u1653998/circuits/results/bonafide/probes/prompt-screening-v1/
```

The job completed successfully in 10m29s with exit code 0. All 2,128 probes completed; there were
no errors or OOMs, all 133 examples have 16 distinct target positions, and every probe reports both
model-configuration restoration and graph-work skipping. The saved corpus occupies approximately
336 MiB and contains 2,128 atomic artifact directories with no temporary-directory leftovers.

| Measurement | Prompt-screening result |
| --- | ---: |
| Probe wall time, median / p90 / max | 0.195 / 0.239 / 0.569 s |
| Total unit wall time, median / p90 / max | 0.250 / 0.313 / 0.639 s |
| CUDA peak allocated, median / p90 / max | 10.01 / 12.60 / 15.71 GiB |
| CUDA peak reserved, median / p90 / max | 11.48 / 16.08 / 21.89 GiB |
| Selected occurrences, median / p90 / max | 138 / 254 / 1,432 |
| Candidate MLP edges, median / p90 / max | 8,662 / 29,942 / 934,125 |

The candidate-edge distribution is strongly heavy-tailed: its p99 is approximately 140,512 while
the maximum is 934,125. Prompt selection for full tracing must therefore use target-level workload
distributions rather than response length alone. The runner emits a compact progress record every
100 items, validates and skips complete artifacts on resume, and aborts rather than reusing the
model if a failed probe reports model-configuration leakage.

## Frozen prompt corpus and target refinement

Prompt membership is now frozen by
`scripts/bonafide/manifests/qwen3_4b_instruct_refinement_probes.json` (SHA-256
`db75a0226ef4fa36694c36c1ff0160d9410aefd6aa4250adbee67ecb48920d72`). The corpus has:

- 11 dense discovery responses: the reviewed ten-response core plus
  `bf-d8f174d2963759f617ca`;
- 24 broad discovery responses with exactly six examples per CoT phenotype, four per hint format,
  eight per response-length bin, eight per screened-workload bin, 24 distinct base-question
  families, 12 SimpleQA, seven DDXPlus, and five HLE examples;
- eight family-locked confirmatory responses, exactly two per phenotype, which may be assigned to
  frozen clusters but must not contribute to cluster fitting or label generation.

The added dense response shares its prompt, validator hint, question, and hinted answer with
`bf-6145690c43f611af97cb`, but the two teacher-forced responses differ: one contains commission and
the other omission plus commission. This is the cleanest within-corpus response-conditioned
contrast. It does not by itself establish causality; a later intervention must occur before answer
commitment during free generation.

Final target positions were not inherited blindly from the 16-point prompt screen. A second
graph-free refinement manifest selected every dense position and at most 64 semantic/boundary
candidates per broad response. Notchpeak job `14074080` ran the resulting 4,131 targets in one
resident-model process on an A100 80GB. It completed in 21m26s with exit code 0, no errors or OOMs,
4,131 unique atomic artifacts, and a clean source revision `c8052dd`.

| Refinement measurement | Result |
| --- | ---: |
| Dense / broad probes | 2,083 / 2,048 |
| Probe wall time, median / p90 / max | 0.187 / 0.238 / 0.651 s |
| CUDA peak reserved, median / p90 / max | 11.07 / 16.09 / 22.50 GiB |
| Candidate MLP edges, median / p90 / max | 8,493 / 29,678 / 81,461,593 |
| Saved size | 657 MiB |

The append-only summary is at
`/scratch/general/vast/u1653998/circuits/results/bonafide/probes/prompt-refinement-v1/probe-summary.jsonl`
(SHA-256 `099e1e80cf79c8b364df69793329f80e8489d264a7aa6cf4fd947daf759d365d`).
Final selection treats validated probe directories as authoritative because a safe resumed probe
run may append metric-free `skipped_complete` rows for already completed targets.

## Frozen final traces

The final runner manifest is
`scripts/bonafide/manifests/qwen3_4b_instruct_final_traces.json` (SHA-256
`706143579c8ebcbd05e1fee150d2f3facf5f3f7e7de372c40e399b83a01687e2`). It contains 2,595
independent width-one traces:

| Role | Prompts | Targets | Clustering role |
| --- | ---: | ---: | --- |
| Dense discovery | 11 | 2,083 | Fit/label, with prompt-balanced weights |
| Broad discovery | 24 | 384 | Fit/label contextual support |
| Broad confirmatory holdout | 8 | 128 | Assignment, stability, and steering checks only |

Each broad response has exactly 16 targets after deduplication: three around the first reliable
hinted-answer commitment, three around a curated BonaFide source/fabrication span or a conservative
fallback, three around the final-answer commitment, four phase controls, one low-probability
semantic token, one large adjacent-candidate feature change, and one median-workload control.
Overlapping reasons remain attached to the same target. Short numeric answer strings and weak
substring source markers are not trusted as semantic anchors; curated BonaFide spans take
precedence. Every final target records its token text, logit, probability, workload measurements,
source probe identity, and probe/metrics hashes. The selector also fails closed if artifacts mix
model, ADAG configuration, or code revision.

The 2,595 selections are split into 47 runnable waves: one routine wave per response plus four
isolated extreme-workload preflight waves. The isolated targets remain scientifically selected,
but they must not hold the routine corpus hostage:

| Example / response position | Token | Candidate MLP edges | Why retained |
| --- | --- | ---: | --- |
| `bf-2981baca0442c8e8021f` / 113 | ` indeed` | 850,981 | Dense all-token contract |
| `bf-89f277b79caf27f7f6ad` / 134 | ` given` | 2,816,921 | Dense all-token contract |
| `bf-72c58caa775145db5022` / 418 | ` specifically` | 1,966,660 | Largest local feature-change control |
| `bf-4f3bab852b0bea33fe6d` / 663 | `Given` | 81,461,593 | Start of an explicit faithful hint/source acknowledgment |

The last target is unusually valuable and unusually dangerous: its neighboring selected tokens
at positions 662 and 664 require only 7,366 and 11,061 candidate edges, so the source-commitment
event remains represented even if position 663 proves impractical.

### Resource interpretation before launch

A rough linear fit on the 72 existing instrumented full traces (candidate edges, planned Jacobian
chunks, and input length; in-sample R-squared 0.877) projects about 31.7 A100-hours for the 2,591
routine targets. Two lab A100s would make that roughly a 16-hour perfectly utilized lower-bound
wall time, before queueing and failures. This is within the original one-to-two-day target.

Do not treat the corresponding 52.1 A100-hour total including all four extreme targets as a reliable
forecast. The fit was trained only through 91,410 candidate edges and 591 planned chunks, while the
`Given` target has 81.5 million edges and 5,353 chunks. Its extrapolated contribution is about 19
A100-hours, almost 900 times beyond the training edge range, and graph materialization may fail or
scale much worse. Run the four isolated waves as explicit preflights; the routine corpus is ready
without waiting for them.

## Frozen compound execution plan

No full trace was launched while freezing the selection. The separate operational plan is now
`scripts/bonafide/manifests/qwen3_4b_instruct_final_execution_plan.json` (file SHA-256
`bbd4df3593f79ec83c2a84947cd6087f103aff340efebc73525d801594986402`; logical plan SHA-256
`9a3788a3e8f500dd458b8e890fb287f1a7375f28fd3798bb129f62ef70b2fcc7`). It references rather
than rewrites the 2,595 frozen scientific work items and is byte-for-byte reproducible from four
hash-bound inputs:

- the final trace manifest;
- the exact Qwen trace run config;
- the 72 completed Wave 2c instrumented traces used to fit the scheduling heuristic;
- the 4,131-target refinement summary used to attach workload predictors.

The ordinary least-squares scheduling model uses candidate MLP edges, planned Jacobian chunks,
and input tokens. Its in-sample R-squared is 0.877. This is a load-balancing heuristic, not a
runtime guarantee: the training data stop at 91,410 candidate edges and 591 chunks, so every
isolated extreme is an extrapolation.

| Array tasks | Kind | Targets per task | Estimated A100 time per task | Default launch policy |
| --- | --- | ---: | ---: | --- |
| 0--11 | Routine LPT shards | 215--218 | 2.640--2.641 h | One `0-11%4` array, 6 h each |
| 12 | 850,981-edge preflight | 1 | 0.216 h | Submit and inspect separately |
| 13 | 1,966,660-edge preflight | 1 | 0.494 h | Submit after the prior result |
| 14 | 2,816,921-edge preflight | 1 | 0.689 h | Submit after the prior result |
| 15 | 81,461,593-edge pathological target | 1 | 19.020 h | Manual opt-in only; do not launch under the current 12 h partition limit |

Routine targets are assigned by deterministic longest-processing-time-first balancing and execute
high-cost-first within each task. The four manifest-marked extremes cannot enter the routine
array. Task 15 additionally requires `ALLOW_PATHOLOGICAL=1`; this guard is not evidence that the
target will fit in memory or finish before the partition limit.

The compound runner preserves the scientific artifact contract:

- validate the plan, every source hash, all target references, and every existing artifact before
  loading the model;
- load one resident model per non-empty array task, then process independent targets at batch size
  one without merging graphs;
- retain each target's original wave ID, artifact identity, and output directory;
- checkpoint at one atomic compact artifact per target and checksum-validate it on resume;
- keep one append-only summary per task attempt, never one JSONL shared by array writers;
- establish an atomic plan-level cohort lock over config, code revision, and runtime environment;
- stop the whole task and return nonzero after OOM, a resource gate, an ordinary error, or the
  five-minute Slurm `SIGUSR1` warning.

Regenerate and byte-compare the plan with:

```bash
.venv/bin/python -m scripts.bonafide.execution_plan \
  --manifest scripts/bonafide/manifests/qwen3_4b_instruct_final_traces.json \
  --config scripts/bonafide/configs/qwen3_4b_instruct.json \
  --historical-summary /scratch/general/vast/u1653998/circuits/results/bonafide/performance/wave2c-instrumented-summary.jsonl \
  --refinement-summary /scratch/general/vast/u1653998/circuits/results/bonafide/probes/prompt-refinement-v1/probe-summary.jsonl \
  --output /tmp/qwen3_4b_instruct_final_execution_plan.json
cmp scripts/bonafide/manifests/qwen3_4b_instruct_final_execution_plan.json \
  /tmp/qwen3_4b_instruct_final_execution_plan.json
```

Inspect any task without loading the model or writing a run summary:

```bash
PLAN="$PWD/scripts/bonafide/manifests/qwen3_4b_instruct_final_execution_plan.json"
.venv/bin/python -m scripts.bonafide.runner \
  --config scripts/bonafide/configs/qwen3_4b_instruct.json \
  --manifest scripts/bonafide/manifests/qwen3_4b_instruct_final_traces.json \
  --execution-plan "$PLAN" \
  --execution-task-index 0 \
  --artifact-root /scratch/general/vast/u1653998/circuits/results/bonafide/final-traces \
  --dry-run
```

The routine launch, after inspection, is:

```bash
PLAN="$PWD/scripts/bonafide/manifests/qwen3_4b_instruct_final_execution_plan.json"
sbatch --export=ALL,EXECUTION_PLAN="$PLAN" scripts/bonafide/final_trace_array.sbatch
```

The launcher defaults to `--array=0-11%4`, one A100 80GB, 64GB host RAM, and six hours on the
lab-owned `marasovic-gpu-np` partition. Tasks 12--14 should be submitted one at a time for genuine
stage-gated inspection with `ALLOW_EXTREMES=1`; an array concurrency cap of one does not itself
guarantee index order. A Notchpeak `sbatch --test-only` check accepted the frozen routine array and
resolved it to `notch369`; the reported test identifier was confirmed absent from the live queue.
