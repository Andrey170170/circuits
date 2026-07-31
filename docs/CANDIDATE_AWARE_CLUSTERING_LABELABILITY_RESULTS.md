# C2 candidate-aware clustering and labelability result

Status: label-free clustering decision complete; matched evidence-only labeling comparison in
preparation.

## Bound artifacts

The analysis uses only the 245 C2 discovery targets frozen by
`CANDIDATE_AWARE_CLUSTERING_LABELABILITY_PROTOCOL.md`. The immutable artifacts are:

- input bundle:
  `/scratch/general/vast/$USER/circuits/results/bonafide/downstream/candidate-aware-clustering-c2-v1/inputs`,
  manifest SHA-256
  `a1870accbd8630471eb58daeff2e1cfc19c0a46d17b37d2bde3cbb6b06b11fff`;
- fitted baseline:
  `/scratch/general/vast/$USER/circuits/results/bonafide/downstream/candidate-aware-clustering-c2-v1/baseline-v1`,
  manifest SHA-256
  `67e99f2d4e19382518daacd3144cd36bcf71af2fc784d87582c2f57dc5ce2509`;
- held-out labelability report:
  `/scratch/general/vast/$USER/circuits/results/bonafide/downstream/candidate-aware-clustering-c2-v1/labelability-evaluation-v1.json`,
  manifest SHA-256
  `79a434f99125c86f5e95c79a812b5dfe7e061d0d7053b93286fe16d585bd9f52`.

All three deep loaders revalidated their source files and independently recomputed the persisted
diagnostics. No labels, model descriptions, confirmatory outcomes, or confirmatory holdout records
were opened.

## Fitted states

The common valid resolution is 64 clusters. The chosen medoid fits have the following structural
diagnostics:

| State | Assigned | Mean/min seed ARI | Modularity | Affinity enrichment | Ready clusters |
| --- | ---: | ---: | ---: | ---: | ---: |
| `W` width-one | 100.0% | 0.7800 / 0.7687 | 0.2358 | 8.1645 | 35/64 |
| `C` candidate direction | 100.0% | 0.4940 / 0.4546 | 0.1354 | 7.2170 | 56/64 |
| `F` calibrated fusion | 99.28% | 0.7361 / 0.7134 | 0.3217 | 8.6015 | 42/64 |
| `S` support-only control | 100.0% | 0.6801 / 0.6569 | 0.2192 | 7.2625 | 36/64 |

Readiness is conservative: a cluster must satisfy the frozen generation, selection-scoring, and
audit witness requirements in both width-one and candidate evidence. `F` is the only candidate-
aware fit with promising structural stability, but its 65.6% ready-cluster fraction is below the
frozen 80% guardrail. `C` has wider witness coverage but fails seed stability and modularity.

## Held-out directional result

Candidate-direction coherence is evaluated on one fixed occurrence intersection scoreable under
`W`, `C`, `F`, and `S`. It covers all eight families in each held-out discovery partition.

| Comparison | Selection lift over `W` | 95% family bootstrap | Audit lift over `W` | 95% family bootstrap |
| --- | ---: | ---: | ---: | ---: |
| `C - W` | 0.04744 | [0.03111, 0.06356] | 0.07829 | [0.05511, 0.10326] |
| `F - W` | 0.04818 | [0.02523, 0.07243] | 0.07201 | [0.04253, 0.10491] |

Both candidate states have positive effects in all eight families and beat the support-only
control on both partitions. This is useful evidence that the five-channel directions contain
local competitive-token structure. It is not enough to select a candidate clustering state:
both selection lifts narrowly miss the prospectively frozen `>= 0.05` threshold. `F` also loses
0.07840 width-one coherence on selection, beyond the allowed loss of 0.05, although its audit loss
of 0.04140 is within the guardrail.

## Decision and early stopping

No `C` or `F` state can pass the frozen candidate-clustering decision. The failed selection,
stability, width-preservation, and readiness conditions cannot be repaired by the remaining
direction-null or generation-family-jackknife computations. The 100 null refits and 18 jackknife
refits were therefore not launched. This is a fail-closed early stop, not an assumption that those
uncomputed diagnostics pass.

The result separates two questions:

1. Candidate measurements improve held-out local competitive-direction coherence.
2. Reclustering on those measurements does not meet the frozen standard for replacing `W`.

The eligible next experiment is consequently the protocol's evidence-only comparison on unchanged
`W64` clusters:

1. width-one source-attribution evidence;
2. the identical clusters and witnesses with rank-one-through-five candidate evidence added.

The old width-one v2.1 pilot anchors are not reusable: they belong to different cluster fits, and
the old dense store covers only 77 of the 245 matched C2 targets. New anchors, witness identities,
and evidence hashes must be frozen before any model call. Candidate descriptions remain
exploratory; the fixed simulator validates only the input-localization hypothesis.

## Frozen evidence-only comparison

Revision `eea52ab` published the provider-neutral, pre-model-call comparison at:

```text
/scratch/general/vast/$USER/circuits/results/bonafide/downstream/
candidate-aware-clustering-c2-v1/labeling-comparison-v1
```

Its manifest SHA-256 is
`227cde5658f1381963b94df192b8e86e1188ca13e28c334003a5a100d3496b55`. The deterministic W64
anchors, in the frozen 3-by-4 target-point order, are:

```text
61, 5, 42, 34, 41, 21, 59, 13, 43, 58, 49, 47
```

The artifact contains 601 generation evidence rows, 530 prompt-ineligible selection/audit scoring
rows, and 24 arm handoffs: one width-only and one width-plus-candidate handoff for each anchor. The
two arms have identical W clusters and generation witness IDs. Candidate slots and numeric
signatures are additive fields only in the combined arm contract. Selection and audit records are
physically separate, marked prompt-ineligible, and reserved for later input-localization scoring
and blinded review.

Every exact teacher-forced prefix, observed token, source-attribution profile, top-16 source-token
highlights, model-rank slots, and candidate signature is persisted before model calls. Candidate
signatures retain the full-precision occurrence count, sum, mean, norm, and unit direction; the
width-five observed-rank channel is checked as a structural zero. A post-publication deep reload
recomputed the entire artifact from the bound input, clustering, labelability, and tokenizer
sources and reproduced the manifest and row counts.

This artifact is deliberately not launchable. It retains all supported generation witnesses and
sets `renderer_frozen=false`; a separate committed renderer must freeze a bounded, identical
witness subset for both arms, six-significant-digit prompt formatting, typed output parsing, and
the Opus/Terra request plan before any paid API call.

## Remaining reporting boundary

The persisted evaluator covers numerical validity, structural metrics, held-out candidate
coherence, width-one preservation, and conservative labeling readiness. Per-state/per-resolution
recurrence and phase-concentration tables requested by the protocol are not yet persisted. They
are non-blocking for the two unchanged-`W` evidence arms, but must be added before describing the
entire label-free reporting section as complete.
