# Candidate-union C2 post-hoc salvage protocol v2

Status: frozen before computing any post-hoc salvage endpoint.

This version supersedes `docs/CANDIDATE_UNION_C2_SALVAGE_PROTOCOL.md`. The v1
implementation review found two issues before execution and before any salvage
rank was inspected: its truth-dependent rescue denominator did not align with
the family-bootstrap estimand, and its tie language conflicted with average
midranks. V2 fixes the rescue endpoint to use all 210 anchors and freezes
average-rank tie handling with fractional top-one credit.

## Question and claim boundary

The frozen C2 utility gate failed because equal-weight multiview next-bin
retrieval was worse than width-one retrieval. This post-hoc analysis asks
whether candidate profiles nevertheless provide useful standalone or
missing-support fallback information.

This is exploratory reuse of the completed 245-target C2 discovery cohort. It
cannot retroactively pass C2, authorize the 2,594-position matched run, or
support a confirmatory claim. A positive result may justify a newly frozen
holdout confirmation. The confirmatory holdout remains untouched.

## Frozen inputs and pair table

The analysis reuses the exact C2 selection, candidate-union plan, width-one
artifacts, assembled two-pass artifacts, and audited failed C2 report. The
analyzer must bind this protocol's path and SHA-256 in its output and derive its
random seed from the audited report hash, protocol hash, and analysis schema.
The only accepted audited C2 report SHA-256 is
`9ea1123685e73bc45f8c93490429a2a309ed62953406d61509d0014730ef6530`.

The 210 phase-bin 0--5 anchors are compared with next-bin targets from the 35
discovery responses. Same-family alternative responses remain excluded. This
produces 7,338 planned anchor-target pairs. For every pair, retain:

- width-one cosine and common signed-base support;
- flattened five-channel candidate-contrast cosine and support;
- one cosine and support diagnostic for each model-rank channel 1--5;
- explicit invalidity from fewer than 16 common bases or zero norm; and
- the true-continuation indicator only for final evaluation.

Signed bases, occurrence reduction, five rank-aligned contrasts, cosine
normalization, common-support threshold, and family/response/target weighting
are unchanged from C2. Exact score ties use average ranks. If `t` candidates
tie for first, each receives top-one credit `1/t`. This explicitly supersedes
C2's stable response-ID tie break. A structurally zero rank channel is invalid,
not zero evidence.

## Primary exploratory endpoints

All primary outcomes use fixed equal-family, then equal-response, then
equal-anchor weights across all 210 eligible anchors. If a method has no score
for the designated true target, reciprocal rank and top-one credit are zero.

### E1: candidate standalone signal

Rank each anchor's valid candidate-profile pairs. Report weighted zero-filled
MRR and top-one accuracy across all anchors. Conditional scored-anchor MRR is a
diagnostic only. Also report the exact uniform-truth expectation over each
anchor's valid midranks and planned pool.

### E2: width-one-missing rescue contribution

For anchor `i`, define

```text
rescue_i = I(width-one true pair is invalid) * candidate reciprocal rank_i
```

where an invalid candidate true pair contributes zero. The primary rescue
statistic is the hierarchical mean of `rescue_i` across all 210 anchors. The
fixed denominator makes its global, permutation, family-bootstrap, and LOFO
estimands agree. Every permutation recomputes width-one invalidity for its
pseudo-truth.

Separately report the observed width-one-invalid subset size, candidate
coverage, zero-filled MRR/top one within that subset, scored-only MRR, support,
and varying-pool chance. Those conditional subset summaries are diagnostics
and are not used for the E2 gate.

### E3: label-free width-one-preferred backoff

Within each anchor and view, convert valid pair scores to
`(ascending average midrank - 0.5) / valid_pair_count`. For each pair, use the
width-one percentile when valid; otherwise use the candidate percentile;
otherwise leave the pair invalid. This fixed rule uses no true-label
information and avoids treating raw cosine scales as interchangeable.

Compare backoff zero-filled MRR with width-one zero-filled MRR over the same 210
anchors. Raw-cosine width-one-then-candidate backoff is a prespecified
descriptive sensitivity only. No blend weight, support threshold, or gate is
tuned on C2 outcomes.

## Null inference and multiplicity

Inference uses 100,000 deterministic whole-response trajectory permutations.
Responses are sorted stably. Each replicate draws one uniform permutation of
the 35 response trajectories, rejects mappings from a response to its
same-family alternative, permits fixed points, and uses the same mapping across
all six phase transitions. Scores, supports, missingness, and candidate pools
remain fixed.

For every replicate compute:

1. candidate all-anchor zero-filled MRR;
2. all-anchor width-one-missing rescue contribution; and
3. backoff-minus-width-one zero-filled MRR.

One-sided empirical p-values use `(1 + count(null >= observed)) / 100001`.
Holm correction controls the three primary exploratory tests. Record the exact
seed derivation, numeric seed, NumPy version, and rejected-proposal count.

Separately report deterministic 10,000-replicate family-block bootstrap
intervals and leave-one-family-out estimates for observed minus
permutation-null family effects. All 34 families remain in each primary
estimand; the one family containing two responses stays one block. Assert that
the mean centered family effect matches the corresponding global lift.

## Diagnostics and interpretation

For each individual model-rank channel 1--5, report true-score coverage,
zero-filled MRR/top one, scored-only values, valid-pair support, and exact
varying-pool chance. These channel results are descriptive and cannot be
selected as a new primary endpoint from this cohort.

E1 or E2 is promising only when its MRR lift over the permutation null is at
least `0.03`, its Holm-adjusted one-sided p-value is below `0.05`, and its
family-bootstrap lower bound is above zero. E3 additionally requires at least
`0.02` observed zero-filled MRR improvement over width one, a Holm-adjusted
p-value below `0.05`, and a positive bootstrap lower bound. LOFO stability
determines whether a positive endpoint is worth a new holdout protocol.

Regardless of outcome, the report must retain the failed C2 decision and
explicitly distinguish broader coverage from better-than-null ranking
information.
