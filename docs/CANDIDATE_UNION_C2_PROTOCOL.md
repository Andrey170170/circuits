# Candidate-union C2 scientific-utility protocol

Status: frozen before C2 rank screening and tracing.

## Question and claim boundary

C2 asks whether the frozen `model_top5_plus_observed` candidate-union view adds
non-degenerate and reproducible discovery-only trajectory information beyond the completed
width-one attribution traces. It does not test faithfulness, validate generated cluster labels,
or authorize a full matched top-five corpus.

Every C2 artifact remains an independent trace of one teacher-forced response position. Candidate
graphs are not merged across response positions. The two-pass contract is the C0/C1 contract:
independently select each candidate topology, take the exact node/edge union, then measure every
union node and edge for every applicable candidate.

## Frozen cohort design

- Source: the frozen Qwen3-4B Instruct final-trace manifest.
- Partition: discovery only; confirmatory holdout and isolated extreme-workload waves are excluded.
- Membership: all 35 discovery responses from all 34 base-question families.
- Final size: exactly 245 targets, seven per response.
- Temporal coverage: sort each response's regular width-one targets and divide them into seven
  contiguous ordinal bins; select one target from every response/bin cell.
- Rank-screen pool: two deterministic targets per cell (490 total): the lowest stored observed
  probability and a distinct target nearest the bin center.
- Final selection after graph-free screening: prefer realized width six in cells for which
  `(sorted_response_index + phase_bin)` is even and width five otherwise. If the preferred width
  is unavailable, use the other width. Within the chosen width prefer the temporal-center slot,
  then stable source-artifact ID. This rule, the response membership, and final count are fixed
  before rank evidence is observed.

The graph-free screen is selection evidence, not a scientific trace artifact. The exact 245 source
IDs and their candidate token IDs are frozen in a new selection and launch bundle before tracing.

## Frozen feature contract

Only MLP neurons are used in the primary profile. Occurrences of the same
`(model, revision, layer, neuron, observed-attribution polarity)` within one target are reduced by
signed sum. Support is explicit; absence from another target is missing, not zero.

The width-one view is the signed attribution scalar from the completed width-one artifact. The
candidate view uses five semantically aligned contrasts:

```text
contribution(model full-distribution rank r) - contribution(observed token), r = 1..5
```

When the observed token has model rank `r <= 5`, that contrast is exactly zero by construction.
This preserves five aligned model-rank axes for both realized widths without inventing a duplicate
candidate. Candidate attribution, raw contribution, activation, applicability, and independent
selection masks remain in the feature store as diagnostics.

Directional comparisons use L2 normalization on the intersection of supported signed bases. A
pair with fewer than 16 common non-boundary MLP bases or a zero norm has no directional score.
Target weights follow the existing equal-family, then equal-response, then equal-target contract.

## Predeclared C2 comparison

The primary trajectory task is next-bin response retrieval. For each target in phase bins 0--5,
rank the targets in the next phase bin from all other discovery responses plus the true
continuation. Same-family alternative responses are excluded as distractors. Report mean
reciprocal rank (MRR), top-one accuracy, scored-anchor coverage, and common-basis support.

Three similarities are reported:

1. width-one attribution only;
2. candidate contribution-contrast only;
3. equal-weight multi-view similarity: the mean of the valid width-one and candidate similarities.

The primary utility contrast is multi-view MRR minus width-one MRR. Uncertainty is a deterministic
family-block bootstrap. The utility gate passes only if:

- the absolute MRR improvement is at least `0.03`;
- the 95% family-block bootstrap interval has a lower bound above zero; and
- at least 80% of leave-one-family-out estimates retain a positive improvement.

The candidate profile non-degeneracy gate additionally requires:

- finite values and complete candidate applicability for every retained union measurement;
- median target-level effective rank of the five contrast channels at least `2.0`;
- at least 90% of targets with effective rank at least `1.5`; and
- a positive median fraction of signed bases exhibiting both positive and negative raw candidate
  contributions.

All thresholds are fixed using the C0/C1 engineering evidence, before C2 rank or trace results.
Failure stops the full top-five extension; it does not invalidate the bounded C0/C1 result or the
width-one downstream program.
