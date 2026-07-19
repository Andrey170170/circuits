# BonaFide tracing corpus plan

Status: agreed next-stage plan after the Wave 2c resource sample. Prompt selection and target-span
selection are deliberately deferred until the cheap probe path is implemented and validated.

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

The immediate sequence is therefore:

1. implement and validate `probe_only`;
2. select the dense prompt set for label/intervention diversity;
3. define annotation and structural candidate spans on a broader prompt pool;
4. run probes and freeze sparse target selections with full provenance;
5. trace in resumable waves and evaluate clustering incrementally.
