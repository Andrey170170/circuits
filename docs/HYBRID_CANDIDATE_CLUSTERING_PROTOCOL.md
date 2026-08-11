# Hybrid candidate-union, paper-style clustering protocol

Status: exploratory discovery analysis. This protocol does not authorize labeling, opening the
confirmatory holdout, full-corpus tracing, or a causal or general faithfulness claim.

The purpose of this analysis is to keep the cancellation-resistant information in the completed
candidate-union traces while moving the clustering semantics closer to ADAG: compare signed bases
on context-local input-attribution and output-contribution profiles, fuse the two similarities
inside each target, and only then aggregate across targets. The analysis fits fresh clusters. It
does not treat an existing width-one or W64 assignment as the answer to recover.

## Discovery firewall

Only the 18 frozen generation families from the existing C2 `18/8/8` family split may affect:

- signed-basis and pair eligibility;
- target-local similarities and hierarchical aggregation;
- affinity construction;
- cluster fitting, seed-medoid choice, or grid-cell validity; and
- any subsequent choice among the predeclared clustering states.

Selection-scoring and audit families are held out from every fit operation. They may be used only
by a separately frozen evaluation stage after the clustering artifact is immutable. All three C2
partitions remain discovery evidence. The distinct confirmatory holdout stays closed and must not
be resolved, loaded, summarized, or used for troubleshooting.

Input construction may authenticate the frozen all-partition target inventory and artifact
identities, then extract the profiles needed for later held-out evaluation. It must preserve the
partition field, and the fit entry point must select `family_partition == "generation"` before
constructing eligibility or pair evidence. A mixed-partition fit fails closed.

## Tracing and signed-basis contract

This analysis does not recreate the paper's summed-top-five tracing objective. It reuses the
completed `adag.bonafide.candidate-union.v1` family:

1. each candidate was traced independently as a specified-token, width-one objective;
2. the exact candidate-selected nodes and edges were unioned; and
3. every candidate was rescored on the fixed union without a second pruning threshold.

This topology can retain a basis that has opposing effects on different logits even when those
effects would cancel in a summed-logit objective. That difference is intentional. The downstream
similarity and fusion rules below, rather than the topology construction, are the paper-style part
of this experiment.

Use only non-boundary internal MLP nodes. Signed-basis identity is
`(model_id, model_revision, layer, neuron_index, sign(observed_activation))`; zero observed
activation is unsupported. Candidate activations for an internal node must equal the observed
activation under `rtol=1e-6, atol=1e-7`, or input publication fails. Repeated token occurrences of
the same signed basis in one target are reduced by signed sum.

For input attribution, recover the observed-candidate fixed-union refinement artifact and sum its
missing-aware `attr_map` by signed basis. Missing token coordinates remain missing and are never
filled with zero. For output contribution, sum the union artifact's raw
`candidate_contribution` values by signed basis. Preserve the explicit candidate axis, logits,
model ranks, observed flag, candidate count, and the union and refinement artifact identities.

## Predeclared profile representations

All candidate coordinates are target-local. Candidate rank or token identity is never assumed to
have the same meaning across unrelated targets.

The primary representation is **`raw` (`raw_top5_plus_observed.v1`)**:

- use the missing-aware observed-refinement `attr_map` without scaling; and
- use every raw contribution coordinate in the candidate-union artifact.

The candidate axis therefore has width five when the observed token is in the model top five and
width six otherwise. In the width-six case it contains the observed candidate plus all five model
top candidates. This is the primary test of whether the extra information preserved by independent
candidate tracing produces useful clusters.

Three predeclared sensitivities or diagnostics are fit independently:

1. **`paper_normalized` (`paper_normalized_model_top5.v1`)** divides each input-attribution value
   by the signed-basis activation and each candidate contribution by that candidate's raw logit,
   then restricts the output view to the five model-top-ranked coordinates. For either denominator
   `d`, use `d` when
   `abs(d) > 1e-10`; otherwise use the fallback denominator `1.0`. No coordinate is dropped because
   its denominator is small, and no absolute value is substituted for the signed denominator.
2. **`top5` (`raw_model_top5.v1`)** uses the raw input profile and only the five coordinates with
   full-distribution ranks one through five. When the observed token is outside the model top five,
   its extra coordinate is excluded.
3. **`contrast` (`top5_minus_observed.v1`)** uses the raw input profile and the five-vector
   `contribution(model-rank r) - contribution(observed)` for `r = 1..5`.

The normalized, top-five-only, and contrast states are diagnostics, not silent fallbacks for an
invalid primary state. Candidate ordering is reconstructed from recorded rank and observed flags,
not inferred from array position.

## Target-local similarity and fusion

For every pair of signed bases co-occurring in one generation target:

1. compute input-profile cosine over only token coordinates supported by both bases;
2. compute output-profile cosine over that target's complete candidate vector;
3. mark a view invalid when its restricted norm is zero or nonfinite;
4. require both views to be valid, clamp each cosine to `max(cosine, 0)`; and
5. compute their harmonic mean inside the target,
   `2 * input * output / (input + output)`, using zero when both clamped values are zero.

Thus a measured zero remains valid recurrence evidence, while a missing overlap or zero-norm view
does not increment recurrence counts. Harmonic fusion happens before any cross-target averaging;
input and output similarities must not first be averaged separately.

Aggregate the fused scalar similarities with the frozen hierarchical target weight
`1/F * 1/R_f * 1/T_r`, giving equal total weight to each generation family, then to each response
within a family, then to each target within a response. A basis is eligible only with valid support
in at least three targets, two responses, and two families. A pair enters the recurring affinity
only with valid overlap in at least two targets, two responses, and two families. After weighted
averaging, only positive off-diagonal affinity is retained.

## Fit grid and numerical gates

Fit every Cartesian grid cell separately:

- representations: `raw`, `paper_normalized`, `top5`, and `contrast`;
- affinity modes: full positive affinity (primary) and positive 32-nearest-neighbor affinity with
  deterministic `union_max` symmetrization (sensitivity);
- cluster counts: `32`, `64`, and `96`; and
- random seeds: `17`, `29`, and `43`.

Spectral fits use self-loop weight `1.0` and eigensolver tolerance `1e-6`. Within one
representation, affinity mode, and cluster count, choose the seed medoid that maximizes its mean
assignment ARI to the other seeds, breaking ties by the smaller seed.

A grid cell is numerically valid only when:

- the affinity is finite and symmetric;
- at least `n_clusters + 1` eligible bases are active;
- connected components do not exceed the requested cluster count;
- every seed fit converges without an eigensolver fallback;
- every seed realizes exactly the requested nonempty cluster IDs; and
- every seed assigns at least 95% of the eligible basis universe.

Invalid scientific grid cells are expected outcomes. Persist each invalid cell with its complete
grid key, stable reason code, human-readable error, and any diagnostics computed before failure;
continue fitting the remaining cells. Do not omit failed cells, replace them with another state, or
abort the artifact merely because one scientific cell is invalid. Input-integrity, partition-
firewall, schema, hash, or provenance failures remain fatal and must prevent publication of the
entire output.

For each valid medoid state, report at minimum assignment coverage, active basis count, connected
components, cluster sizes, all pairwise seed ARIs, mean/minimum seed ARI, modularity, and within-
cluster affinity enrichment. Apply the existing exploratory structural guardrails without relaxing
them after inspection: at least 95% assignment, largest cluster fraction at most 15%, mean/minimum
seed ARI at least `0.72/0.70`, modularity at least `0.20`, affinity enrichment at least `1.25`, and
at least 80% of clusters meeting the frozen labeling-witness support rule. A structural pass does
not itself authorize labeling or select a scientific winner.

No grid cell is required to agree with W64. W64 assignments must not enter profile construction,
similarity, affinity, fitting, seed choice, or validity. Any later W64 overlap is a descriptive
diagnostic only.

## Decision boundary and subsequent evaluation

The immutable clustering artifact is exploratory and records `labeling_authorized=false` and
`confirmatory_holdout_opened=false`. Before model-generated labels are inspected, a separate
evaluation protocol must freeze how primary `raw` and eligible alternative states will be compared
on selection and audit evidence. That comparison should cover held-out candidate-direction
coherence, input-profile coherence, family sensitivity, witness readiness, and paper-style
explainer/simulator performance using actual context-local candidate tokens and scores.

An invalid `raw` cell is a result, not permission to promote a diagnostic. Likewise, a diagnostic
outperforming `raw` in an inspected discovery analysis does not establish a confirmatory result.
Opening the distinct confirmatory holdout requires a new immutable decision and remains out of
scope here.

## Artifact and code provenance

Input and fit artifacts must be immutable, atomically published, and deep-loadable. Their manifests
must bind at least:

- this protocol's SHA-256;
- the clean Git commit and the hashes of the executable input, clustering, and persistence modules;
- the source candidate-aware input manifest path, schema, and SHA-256;
- the complete candidate-union payload-hash set and fixed-union refinement payload-hash set;
- model and tokenizer identity/revision, response position, case, response, base-question family,
  and frozen family partition for every target;
- the canonical signed-basis index and representation/fusion version identifiers;
- recurrence gates, hierarchical weighting, complete grid, numerical-gate versions, and random
  seeds;
- every produced file's SHA-256, row/count summaries, and a canonical manifest self-hash; and
- `exploratory=true`, `labeling_authorized=false`, and
  `confirmatory_holdout_opened=false`.

The loader must recompute manifest and payload hashes and reject source drift, mixed model
identities, altered partitions, missing artifacts, malformed candidate axes, or schema drift. It
must also verify that the executable code revision and protocol hash match the values bound at
publication. Generated traces, cluster outputs, scheduler logs, and credentials remain outside the
repository.
