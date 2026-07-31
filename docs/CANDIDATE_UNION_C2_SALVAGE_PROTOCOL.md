# Candidate-union C2 post-hoc salvage protocol

Status: frozen before computing the post-hoc salvage endpoints.

## Question and claim boundary

The frozen C2 utility gate failed because equal-weight multiview next-bin
retrieval was worse than width-one retrieval. This post-hoc analysis asks a
narrower question: whether candidate profiles nevertheless provide useful
standalone or missing-support fallback information.

This is exploratory reuse of the completed 245-target C2 discovery cohort. It
cannot retroactively pass C2, authorize the 2,594-position matched run, or
support a confirmatory claim. A positive result may justify a newly frozen
holdout confirmation. The confirmatory holdout remains untouched.

## Frozen inputs and pair table

The analysis reuses the exact C2 selection, candidate-union plan, width-one
artifacts, and assembled two-pass artifacts. It revalidates their hashes and
provenance using the C2 analyzer before constructing profiles.

The 210 phase-bin 0--5 anchors are compared with next-bin targets from the 35
discovery responses. Same-family alternative responses remain excluded. This
produces 7,338 planned anchor-target pairs. For every pair, retain:

- width-one cosine and common signed-base support;
- flattened five-channel candidate-contrast cosine and support;
- one cosine and support diagnostic for each model-rank channel 1--5;
- explicit invalidity from fewer than 16 common bases or zero norm; and
- the true-continuation indicator only for final evaluation.

Signed bases, occurrence reduction, five rank-aligned contrasts, cosine
normalization, common-support threshold, family/response/target weighting, and
stable response-ID tie handling are unchanged from C2. A structurally zero
rank channel is invalid, not zero evidence.

## Primary exploratory endpoints

All population outcomes use the fixed equal-family, then equal-response, then
equal-anchor weights across all 210 eligible anchors. If a method has no score
for the true target, its reciprocal rank and top-one value are zero.

### E1: candidate standalone signal

Rank each anchor's valid candidate-profile pairs. Report weighted zero-filled
MRR and top-one accuracy across all anchors. Conditional scored-anchor MRR is a
diagnostic only. For a valid pool of size `n`, also report the descriptive
uniform-rank expectations `H_n / n` for reciprocal rank and `1 / n` for top
one, using average midranks if exact score ties occur.

### E2: width-one-missing rescue

Define the observed rescue subset as anchors whose true continuation lacks a
valid width-one score. On that subset, report candidate coverage, zero-filled
MRR, top one, scored-only MRR, and the varying-pool analytic chance diagnostic.
Because subset membership is truth-dependent, inferential permutations must
recompute the subset for every pseudo-truth assignment.

### E3: label-free width-one-preferred backoff

Within each anchor and view, convert valid pair scores to mid-empirical-CDF
percentiles. For each pair, use the width-one percentile when it is valid;
otherwise use the candidate percentile; otherwise leave the pair invalid.
This rule is fixed before seeing salvage ranks, uses no true-label information,
and avoids treating raw cosine scales as interchangeable.

Compare backoff zero-filled MRR with width-one zero-filled MRR over the same 210
anchors. Raw-cosine width-one-then-candidate backoff is reported only as a
prespecified sensitivity. No blend weight, support threshold, or gate is tuned
on C2 outcomes.

## Null inference and multiplicity

Inference uses 100,000 deterministic whole-response trajectory permutations.
Responses are sorted stably. Each replicate draws one uniform permutation of
the 35 response trajectories, rejects mappings from a response to its
same-family alternative, permits fixed points, and uses the same mapping across
all six phase transitions. Scores, supports, missingness, and candidate pools
remain fixed. A pseudo-truth with no method score receives reciprocal rank
zero.

For every replicate compute:

1. candidate all-anchor zero-filled MRR;
2. candidate zero-filled MRR on the recomputed width-one-invalid subset; and
3. backoff-minus-width-one zero-filled MRR.

One-sided empirical p-values use `(1 + count(null >= observed)) / 100001`.
Holm correction controls the three primary exploratory tests. The seed is
derived deterministically from the audited C2 report SHA-256 and the analysis
schema ID; the numeric seed, NumPy version, and rejection count are recorded.

Separately report deterministic 10,000-replicate family-block bootstrap
intervals and leave-one-family-out estimates for observed minus
permutation-null family effects. The one family containing two responses stays
as one block. These stability summaries do not replace the permutation test.

## Diagnostics and interpretation

For each individual model-rank channel 1--5, report score coverage, zero-filled
MRR/top one, scored-only values, support, and analytic varying-pool chance.
These five channel results are descriptive and cannot be selected as a new
primary endpoint from this cohort.

A candidate standalone or rescue endpoint is considered promising only when
its MRR lift over the permutation null is at least `0.03`, its Holm-adjusted
one-sided p-value is below `0.05`, and its family-bootstrap lower bound is above
zero. Backoff additionally requires at least `0.02` zero-filled MRR improvement
over width one, a Holm-adjusted p-value below `0.05`, and a positive bootstrap
lower bound. LOFO stability determines whether a positive endpoint is worth a
new holdout protocol. Regardless of outcome, the report must retain the failed
C2 decision and explicitly distinguish broader coverage from better-than-null
ranking information.
