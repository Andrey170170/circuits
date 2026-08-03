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

The provider-neutral evidence artifact itself remains deliberately non-launchable and retains all
supported generation witnesses. The separate renderer was frozen at revision `38d83d4` and
published under:

```text
/scratch/general/vast/$USER/circuits/results/bonafide/downstream/
candidate-aware-clustering-c2-v1/labeling-renderer-v1
```

Its manifest SHA-256 is
`a0c49b86fc8fc53710aee8c6e88709da1391cdb8fa3fce255aac7ef62032c13c`. It selects exactly eight
generation witnesses for each W64 anchor using width-only greedy diversity over family, response,
phase, and observed-token identity, with width salience and case ID as deterministic tie-breakers.
The identical 96 witness occurrences and order feed both arms. Arm 1 contains no candidate slots or
signature; arm 2 adds exactly five rank slots and the unclipped rank-aligned signature rendered to
six significant digits.

The renderer contains 24 logical prompts and a provider/model-unresolved historical stage plan for
120 semantic samples, 24 rewrites, and 24 conservative controls. Its frozen provider-role labels
are Opus, Opus, and Terra respectively and are not being rewritten. Rewriters may see only the
original generation prompt and its five generation-only semantic samples; controls see only the
original generation prompt. Selection/audit evidence, automatic scores, and held-out measurements
are forbidden inputs to every generation stage. The manifest records `calls_made=false`.

A post-publication deep reload reproduced the comparison manifest
`227cde5658f1381963b94df192b8e86e1188ca13e28c334003a5a100d3496b55`, all `601/530/24`
comparison rows, the 24 rendered prompts, and the `120/24/24` request plan. Cross-snapshot
validation uses the recorded Git commit, tree, and blobs and requires the current comparison and
labelability runtime bytes to match those recorded sources before deterministic recomputation; it
does not depend on the older scratch worktree remaining present.

The 24 serialized prompt payloads contain 1,170,746 characters. The frozen Qwen tokenizer gives a
provider-independent proxy of 383,583 input tokens total (12,780--18,944 per logical prompt).
The additive execution adapter uses generic stage identities. OpenAI is the iteration default
because its current Luna/Terra path is materially cheaper: Luna supplies the high-volume semantic
samples, while Terra performs rewriting/summarization and the independent conservative control.
Anthropic remains a matched full-evaluation comparison after fake-backend and small OpenAI
validation rule out simple request, parsing, resume, and firewall defects. Provider/model IDs,
output ceilings, request files, and cost bindings live in versioned execution recipes; endpoint
resolution and every paid submission remain explicit later gates. No paid API or Transluce call
was made at this checkpoint.

## Remaining reporting boundary

The next code artifact is a provenance-bound C2/dense-multiplex assessment. It retains all 245 C2
targets and marks the exact 77-target dense overlap rather than silently inner-joining. Its primary
measurement grain is target plus signed basis because the candidate profile is a signed sum over
raw node occurrences; a dense occurrence projection may carry the cluster identity and a foreign
key to that measurement, but not invent per-token candidate values. It also distinguishes the C2
`W64` state used by the matched labeling anchors from the separate dense primary 64-cluster state.

The persisted evaluator covers numerical validity, structural metrics, held-out candidate
coherence, width-one preservation, and conservative labeling readiness. Per-state/per-resolution
recurrence, phase/rank concentration, projection coverage, and within-cluster heterogeneity remain
to be persisted before describing the entire label-free reporting section as complete.
