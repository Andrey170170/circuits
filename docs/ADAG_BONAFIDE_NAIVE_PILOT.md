# ADAG–BonaFide naive feasibility pilot

Status: first-pass experiment plan. This document intentionally stops before designing a
general faithfulness detector or a text/internal atom comparator.

The staged resource benchmark and compact trace-artifact boundary are specified separately in
`docs/TRACING_PERFORMANCE_BENCHMARK.md`. In particular, tracing does not construct or save a
response-wide mega-graph; graph construction is a downstream consumer of target-indexed traces.

## Goal

Test a narrow empirical question:

> When a BonaFide example gives us a known hint-reliance or required-bottleneck structure,
> can we see a corresponding, reasonably separable structure in an ADAG attribution graph
> over raw MLP neurons?

The desired output is a small set of inspectable reconstructed graphs, cluster assignments,
and controlled comparisons. We want to learn whether the proposed internal representation is
even useful at a naive level: whether recognizable structures appear, whether they are diffuse
or clean, whether they survive minor perturbations, and whether ADAG's clustering makes them
more legible.

This is a representation-feasibility pilot, not a benchmark result. A positive result would
justify investing in response-wide tracing and better provenance. A negative result would help
locate the failure mode: the model may not organize the computation cleanly in the raw-neuron
basis, or the tracing/pruning/clustering procedure may fail to recover it.

## Conceptual grounding

BonaFide adopts a mechanistic reading of faithfulness: a step is faithful when it accurately
describes a process that occurred in the model, and a complete CoT must include a complete path
the model followed while containing no unfaithful steps. Its task construction gives partial
ground truth by making some computations necessary or making a misleading hint the only
credible source of a selected wrong answer. It does **not** make every internal computation
observable. See `papers/2605.25052_src/neurips_2026.tex` and the selected records in
`BonaFide.csv`.

ADAG first constructs a pruned attribution graph, then characterizes raw MLP neurons using two
non-local profiles:

- input attribution: which preceding tokens contributed to a neuron's activation;
- output contribution: which target output logits the neuron promotes or suppresses.

It groups neurons into supernodes with multi-view spectral clustering over those profiles and
can use an explainer/simulator pipeline to describe the supernodes. See
`papers/2604.07615_src/colm2026_conference.tex` and the repository `README.md`.

These facts motivate the pilot, but they also bound its interpretation. ADAG uses a locally
linear replacement backward pass, gradient-based attribution, and pruning. Its output is an
approximate subgraph for selected logits, not a complete transcript of model computation.

## Non-goals

This pilot will not:

- build the final textual-atom extractor;
- define or train an atom-to-atom comparator;
- classify general CoTs as faithful or unfaithful;
- treat cluster labels as ground truth about what the model computed;
- compare raw-neuron and transcoder methods comprehensively;
- claim that a missing structure proves that the underlying computation did not occur;
- solve response-wide graph storage, scaling, or multi-GPU execution in general.

The supplied BonaFide hint, annotated step, and required bottleneck are temporary probes. They
let us inspect the internal representation without first solving textual atom extraction.

## Local setup

The project uses Python 3.12 and a `uv` lock restricted to the CHPC Linux x86_64 target. From a
compute allocation, initialize the reusable scratch-backed environment with:

```bash
module load python/3.12.12 uv/0.11.14
source scripts/chpc_env.sh
uv sync --frozen
```

`scripts/chpc_env.sh` keeps the Python environment and experiment artifacts under
`/scratch/general/vast/$USER/circuits`, while reusing the existing Hugging Face and `uv` caches
under `/scratch/general/vast/$USER/nlp_research_project`. It also loads `.env` when present, so
credentials do not need to appear in shell commands. Temporary files use allocation-local
scratch when `SLURM_JOB_ID` is available. The initial local model cache contains the exact Qwen
Instruct and Thinking checkpoints named below; description-model weights can be added later if
automatic labeling is included in the pilot.

The cached revisions verified offline for this pilot are:

- `Qwen/Qwen3-4B-Instruct-2507` at `cdbee75f17c01a7cc42f958dc650907174af0554`;
- `Qwen/Qwen3-4B-Thinking-2507` at `768f209d9ea81521153ed38c47d515654e938aea`.

Each snapshot has all three indexed weight shards (7.49 GiB total). Pin these revisions in the
run manifest rather than relying on a moving repository default.

## Initial anchors

Use exact prompts and stored CoTs from `BonaFide.csv`. Keep Qwen Instruct and Qwen Thinking
artifacts completely separate: their neurons and cluster identities are not comparable.

| BonaFide row ID | Model | Probe | Why start here |
|---|---|---|---|
| `f44d85b57fe09ddc` | `Qwen/Qwen3-4B-Instruct-2507` | Professor hint says *The Zero-Sum Game*; the CoT fabricates discography support for that wrong answer. | Short commission-style case. Tests whether hint/value-copying and fabricated-recall-looking activity are distinguishable. |
| `23068c9e9e56a270` | `Qwen/Qwen3-4B-Instruct-2507` | Professor hint says *Baltimore, United States*; the CoT explicitly acknowledges and follows it. | Short faithful hint-reliance contrast for the same model family and hint format. |
| `a92b84d0920c5100` | `Qwen/Qwen3-4B-Instruct-2507` | Hidden error text supplies *Pennsylvania*; the CoT claims support from reliable sources. | A second commission case with a different prompt channel and source claim. |
| `f2d81f1889e8f0df` | `Qwen/Qwen3-4B-Thinking-2507` | Collatz from 64; first required step is `64 / 2 = 32`. | Clean outright-task bottleneck that occurs early. Trace a prefix ending shortly after the first step rather than the repetitive full CoT initially. |

The first three form an Instruct-model pilot set. The Collatz case is a separate Thinking-model
pilot, not a fourth context in the same clustering run.

### Controlled variants

Clustering and separability cannot be judged from one isolated graph. For each anchor, build a
small family of controlled contexts while preserving the exact original as the primary record.

For hinting cases, useful conditions are:

- original misleading hint;
- no hint;
- paraphrased hint with the same answer;
- substituted hinted answer;
- where useful, a truthful hint.

For Collatz, use nearby powers of two such as 32 and 128, plus wording paraphrases. This tests
whether a cluster tracks the halving computation rather than the literal tokens `64` and `32`.

Generated variants are experimental controls, not new BonaFide-labeled examples. Their
provenance must state how they were derived.

## Proposed stages

### Stage 0: freeze inputs and semantics

Create a machine-readable run manifest containing the dataset row ID, exact model and tokenizer
revision, prompt, stored CoT or traced prefix, tokenization, target token positions, condition,
code commit, tracing configuration, and hashes of all text fields.

Before tracing, render the chat template and verify the assistant-token boundaries for both
Qwen variants. Record whether a run uses the exact stored response, a prefix, or newly generated
text.

### Stage 1: single-target smoke traces

For each anchor, trace one or a few deliberately chosen next-token logits around the known event:

- the hinted answer/value and the first explicit hint acknowledgment or source claim;
- the first Collatz transition and its result token(s).

Use `batch_size=1`, inspect the unclustered graph first, and start with short response prefixes.
The purpose is to validate model loading, Qwen gradient hooks, assistant-position detection, graph
serialization, and visualization before spending compute on every response token.

### Stage 2: short-window, per-target traces

Trace every response target token in a short window around the event **independently** and save
one compact trace artifact per target. Do not merge those artifacts in the tracing pipeline.
Downstream analyses may load a chosen collection while preserving each artifact's provenance.

An aggregate trace over all response logits is acceptable as a smoke test, but it is not the
primary analysis artifact. Aggregation can make unrelated token pathways appear to be one
mechanism and can hide when a structure first appears.

### Stage 3: controlled-context circuit and clustering

Build a circuit over the controlled contexts for one model at a time. Cluster raw-neuron basis
features from their input-attribution and output-contribution profiles, then map clusters back to
position-specific occurrences in each graph.

Sweep at least:

- pruning threshold or an equivalent retained-attribution budget;
- number of clusters;
- target-token/window choice;
- position-collapsed versus position-aware analysis, once the latter is implemented correctly.

Do not select the visually nicest setting after the fact without showing the alternatives.

### Stage 4: inspect and label

First inspect graphs and profile exemplars manually with cluster IDs only. Then optionally run the
ADAG description pipeline. Human-readable labels help exploration, but conclusions should remain
tied to the underlying profiles, member neurons, graph paths, and controlled contrasts.

The repository exposes an API attribution-description backend for workflows where its bundled
local explainer/simulator is not applicable; that local description stack is tied to a Llama
tokenizer. Automatic labels are therefore not required for the first feasibility result.

### Stage 5: limited causal spot checks

If a candidate cluster is recognizable and stable, perform a small position-restricted ablation
or patching check. Compare against layer-, size-, and attribution-matched random neuron groups.
This can test whether the proposed cluster affects the relevant token or answer, but causal
importance must not be treated as equivalent to BonaFide faithfulness.

## What the repository already supports

The current clone already contains most of the ADAG backbone:

- raw MLP-neuron tracing for Llama and Qwen-family models;
- RelP-style gradient handling, node selection, edge construction, and attribution/contribution
  profiles in `circuits/tracing/`;
- a `CircuitData` artifact with node and edge tables, tokenized inputs, attention masks, target
  logits/probabilities, labels, model ID, timestamp, and tracing config;
- per-example labels encoded as `label___N`, plus token positions in node and edge records;
- dataset/config-driven single- and multi-GPU circuit preparation in
  `scripts/circuit_prep/prep.py`;
- multi-view spectral clustering, cluster state persistence, description hooks, steering, and
  graph visualization/export;
- Python 3.12 and a locked `uv` environment.

There is an existing response-rollout path at API level, but it always generates at least one
extra token and does not provide the exact teacher-forced target contract needed here. It is
useful implementation context, not the pilot interface: the benchmark requires a no-generation
path with explicit response-relative target positions.

## Required implementation work

### Required for the first credible pilot

1. **BonaFide manifest/loader.** Select rows by stable ID, deduplicate repeated annotations of the
   same CoT, support response-prefix boundaries, and emit exact original/control conditions.
2. **Expose response tracing.** Add pilot-runner or CLI support for `use_rollout`,
   `max_new_tokens`, selected response target positions, and explicit teacher-forced targets.
3. **Fix or prohibit variable-length rollout batches.** The trace path constructs target-position
   lists per example, but currently returns only the first example's target positions. Require
   `batch_size=1` for the pilot until unequal assistant lengths are serialized and consumed
   correctly.
4. **Preserve target provenance.** The tracing core computes some per-target contribution maps,
   but its scalar attribution objective sums selected logits and dataframe conversion further
   aggregates values. Generate one reusable artifact per target with explicit response position,
   prediction position, token ID, and token text. Multi-target traces are benchmark-only and
   cannot be losslessly split afterward.
5. **Add a complete run manifest.** `CircuitData` stores token IDs and labels but not the original
   prompt, seed response, condition derivation, model/tokenizer revision, or text hashes.
6. **Make occurrence identity explicit.** Distinguish a basis feature such as
   `(model, layer, neuron, polarity)` from an occurrence such as
   `(example, response_position, layer, neuron, polarity)`.
7. **Fix/verify position-aware clustering.** Clustering defaults to summing a neuron over token
   positions. Although `sum_over_tokens=False` is exposed, the current cluster-map expansion
   looks up `token=-1`, so the position-aware path needs a focused test and likely a mapping fix.
8. **Produce inspection diagnostics.** Record graph sizes, retained attribution mass (or a clear
   proxy), threshold, target count, cluster overlap, and stability across the planned sweeps.

### Needed only if the pilot advances

- remove hard-coded 32-layer assumptions in remaining steering/export paths before relying on
  Qwen steering results;
- add position-restricted intervention support and matched controls;
- define a durable response-level graph merge schema;
- add automatic Qwen-compatible description configuration;
- add larger-scale job planning and resumable shard handling.

## Artifact and provenance contract

The smallest trustworthy occurrence-level record should preserve:

```text
run_id
example_id / BonaFide row ID
condition and parent condition
model_id, model_revision, tokenizer_revision
prompt_hash, response_hash, code_commit
target_response_position, target_token_id, target_token_text
source_layer, source_position, source_neuron, source_polarity
target_layer, target_position, target_neuron, target_polarity
attribution_method and full tracing config
signed attribution, edge weight, pruning decision
```

Cluster assignments should reference basis features, while graph nodes and edges reference
occurrences. Any merged or summarized weight must retain the IDs of the contributing target
slices. Store raw graphs and manifests before cluster labels so later clustering choices do not
rewrite experimental provenance.

## Naive evaluation criteria

This is exploratory, but the criteria should be written down before graph inspection.

### Evidence that the idea is promising

At least one known structure should satisfy several of the following:

- **Recoverability:** a compact path or cluster has input attribution concentrated on the known
  hint/bottleneck tokens and contributes to the relevant response/answer token.
- **Timing:** the structure is present before or around the relevant decision or verbalized step,
  not only after the answer has been copied into the response context.
- **Contrast:** its profile weakens or changes appropriately without the hint, and follows a
  substituted hint or arithmetic operand rather than a fixed lexical item.
- **Separability:** it is distinguishable from generic instruction following, output formatting,
  token copying, and broad answer-token promotion.
- **Stability:** its members or profile remain recognizably similar across paraphrases and
  reasonable pruning/cluster-count choices.
- **Graph coherence:** position-specific occurrences form an interpretable directed route rather
  than being connected only after target aggregation.

No single criterion is enough. In particular, a cluster that merely copies the hinted answer
from the prompt is expected but does not by itself distinguish faithful acknowledgment from a
fabricated justification.

### Evidence against the current representation

The naive representation is not promising if, after reasonable target and threshold checks:

- relevant attribution remains diffuse across many unrelated neurons and clusters;
- clusters are dominated by token identity, position, or formatting and do not generalize to
  controlled variants;
- cluster membership changes radically under small perturbations or modest hyperparameter
  changes;
- the only recognizable structures occur after the response itself has introduced the relevant
  content;
- full-window graphs become interpretable only after provenance-destroying aggregation;
- faithful hint acknowledgment, unfaithful source fabrication, and ordinary answer copying look
  indistinguishable at every inspected level.

This would reject the current raw-neuron/ADAG representation for this purpose, not prove that the
model lacks an internal computation corresponding to the BonaFide structure.

## Main risks and confounds

- **Teacher-forcing contamination.** Later response-token traces condition on the model's earlier
  written reasoning. A late fabricated justification may become a real causal input even though
  it did not describe the original source of the answer. Focus on early positions and preserve
  time.
- **Answer-copying confound.** In hinting tasks the wrong answer appears verbatim in the prompt.
  A clean copy circuit is not yet a clean hint-reliance or misattribution circuit.
- **Approximation and pruning.** RelP/CLJA choices and thresholds can omit, distort, or redistribute
  paths. Missing graph structure must initially be reported as unresolved.
- **Target aggregation.** Summing many response logits rewards neurons useful across generic
  language generation and can merge distinct computations.
- **Small-context clustering.** Spectral clustering with manually chosen `k` can manufacture
  apparent groups when there are too few varied contexts.
- **Raw-neuron basis.** Relevant computation may be distributed or superposed and therefore not
  cleanly separable into neuron clusters.
- **Prompt/model fidelity.** Exact Qwen model/tokenizer versions and chat templates matter. A
  nearby checkpoint is not a valid replication of a BonaFide record.
- **Partial BonaFide ground truth.** The task design identifies selected necessary processes; it
  does not annotate every legitimate internal route or every step in the response.
- **Interpretation bias.** Knowing the expected hint/bottleneck makes post-hoc pattern matching
  easy. Controlled variants and predeclared criteria are essential.

## Open ADAG questions for the implementation discussion

1. What exactly should the traced objective be for a response: each emitted token logit, a short
   semantic token set, the answer token, or the current sum over many teacher-forced logits?
2. How does the percentage threshold scale when the objective is a sum over response positions?
   Should comparisons instead match retained attribution mass or node budgets?
3. In the intended ADAG semantics, is a feature the layer/neuron identity across all positions,
   or should position-specific occurrences be eligible for distinct clusters in this experiment?
4. When token positions are summed before clustering, which temporal distinctions are lost, and
   how should cluster labels be mapped back to response-time occurrences?
5. How should signed activation polarity be handled across examples and positions? Is treating
   positive and negative occurrences as distinct basis features sufficient?
6. What is the correct interpretation of an edge under the RelP replacement backward pass and
   stop-gradient choices? Which claims require a later intervention rather than attribution?
7. How should contexts where a neuron was pruned be treated: zero activity, missing observation,
   or unknown? How sensitive is spectral similarity to this choice?
8. How many controlled contexts are needed before multi-view spectral clustering is meaningful,
   and how should `k` be selected without inspecting the desired answer first?
9. If a later downstream analysis needs a response-wide representation, what representation can
   consume independent per-target artifacts without discarding their provenance? This is outside
   the tracing/performance implementation.
10. Can position-restricted steering test the candidate mechanism without the current blunt
    intervention of changing a cluster at every token position?
11. For Qwen, which description path is acceptable for this pilot, and can we postpone language
    labels until graph separability is established?
12. Which result would distinguish "the mechanism is not cleanly present" from "this tracing
    configuration failed to recover it" strongly enough to guide the next method comparison?

## Immediate implementation order after discussion

1. Freeze the four anchor manifests and controlled conditions.
2. Add a small response-window pilot runner with explicit target selection.
3. Save one compact artifact per reusable target with full run and target provenance.
4. Smoke-test one early Instruct target and the first Collatz bottleneck.
5. Export unclustered graphs and verify token/time alignment.
6. Run small threshold and target-window sweeps.
7. Cluster controlled contexts, inspect stability, and only then add automatic descriptions or
   causal spot checks.
