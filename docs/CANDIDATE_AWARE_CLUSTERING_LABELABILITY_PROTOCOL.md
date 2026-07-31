# C2 candidate-aware clustering and labelability salvage protocol

Status: frozen before candidate-aware cluster fitting, cluster selection, or description
generation.

## Question and decision boundary

This post-hoc discovery analysis asks whether the completed C2 candidate-union measurements can
produce more coherent local signed-basis clusters and more evidence-backed labels than matched
width-one input attribution alone.

It does not reinterpret the failed C2 trajectory gate. Equal-weight candidate/width-one next-bin
retrieval was worse than width one, and the candidate-only salvage signal was driven largely by
support applicability. This analysis cannot retroactively pass C2, authorize the 2,594-position
matched run, open the confirmatory holdout, or establish causal or BonaFide faithfulness claims.
Success means only that non-degenerate local alternative-token contribution measurements improve
stable, held-out, corpus-bounded descriptions of the traced signed bases.

## Frozen inputs

Reuse exactly the 245 completed C2 discovery targets: seven phase bins from each of 35 responses
covering 34 base-question families. The cohort contains 235 width-five and ten width-six candidate
sets. The observed token is candidate zero; every model rank from one through five is present.

The analysis binds:

- C2 selection file SHA-256
  `67cd4b2d5bf2fe2558b2abe55060b75d098856ccdb71be1df771b2278ca869b1`;
- candidate-union plan file SHA-256
  `7d8a79144a6616a30ac67802d2975ef8a7fe5f92bfc1f879ed1a980fbe0a7724` and
  canonical SHA-256 `0390409e9488fa822be50c0114242369bfa50ff27559d6da40ba182e6d91ae68`;
- audited C2 report SHA-256
  `9ea1123685e73bc45f8c93490429a2a309ed62953406d61509d0014730ef6530`;
- C2 artifact payload-set SHA-256
  `9e73332c93c3dcf3b8ea9c8f0f5c107f5e51fe3a0bba21114b1372cbfa64ef94`;
- post-hoc salvage report SHA-256
  `3ce0f45c1d05e97310481ec4bc42462b09c7490108d9e921a3490fa3620fa0b3`.

Width-one source-token attribution maps are rebuilt directly from the 245 bound compact traces.
The existing dense compacted feature store contains only 77 of these targets and therefore cannot
serve as the matched baseline. Candidate measurements come only from the corresponding validated
candidate-union artifacts. No model inference or tracing is needed.

## Signed-basis identity and views

All scientific comparisons use the same non-boundary MLP signed-basis universe and the same
target/family weights. Polarity is the sign of the observed-candidate activation, matching the
production cluster-profile and labeling identity. A zero activation is unsupported. Repeated
token occurrences of the same signed basis are reduced by signed sum.

Build four views:

1. `W`, matched width-one input view: the missing-aware source-token `attr_map` used by the dense
   atlas, not the scalar attribution used in the C2 retrieval analysis.
2. `C`, candidate-direction view: for model ranks `r=1..5`, use
   `contribution(rank r) - contribution(observed)` as a five-channel vector. A zero-norm vector is
   missing; a structural zero channel when the observed token occupies a top-five rank is retained.
3. `F`, late fusion: construct `W` and `C` similarities independently. After the recurring-pair
   gates, replace each positive off-diagonal similarity by its ascending empirical midrank divided
   by the number of positive similarities in that view. Average the two calibrated values only for
   pairs supported in both views, then apply the common kNN rule. There is no missing-view fallback
   and no raw feature concatenation.
4. `S`, support-only control: candidate-union weighted target-support Jaccard/co-occurrence with no
   directional contribution values.

The primary matched universe contains only bases eligible in both `W` and `C`. Candidate-union
bases outside that universe are reported as an expanded-coverage diagnostic and cannot make the
primary comparison pass.

## Fit grid and deterministic state choice

Use hierarchical equal-family, then equal-response, then equal-target weights. Primary basis
eligibility requires support in at least three targets, two responses, and two families in both
scientific views. A pair requires valid directional overlap in at least two targets, two responses,
and two families; `F` requires those gates independently in both views.

For `W`, `C`, and `F`, fit normalized sparse spectral clustering with:

- cluster counts `32, 64, 96`;
- seeds `17, 29, 43`;
- positive 32-nearest-neighbor affinity;
- deterministic `union_max` symmetrization;
- self-loop weight `1.0` and eigen tolerance `1e-6`.

The primary matched comparison uses 64 clusters if all three views yield valid states. Otherwise
it uses the smallest tested count valid for all three; if none is common, the matched comparison
fails. Within one view/count, the medoid seed maximizes mean assignment ARI to the other seeds,
with the smaller seed breaking ties. `S` uses the chosen common count and the same spectral
settings.

## Support and directional nulls

The support-only state is mandatory because the C2 salvage signal was largely explained by
candidate-score applicability.

Also run 100 deterministic candidate-direction null refits. Within each target, permute the
five-channel vectors among bases within layer, activation polarity, and candidate-vector L2-mass
decile. This preserves topology/support, the candidate set, layer/polarity composition, and vector
magnitude while breaking signed-basis-to-competitive-direction identity. Seeds derive from the
protocol file hash plus replicate index.

Each null refit uses the permuted vectors to construct its cluster state, but its assignments are
scored against the original unpermuted vectors with the same held-out-family coherence estimator.
A candidate result is directional only when its held-out candidate-coherence statistic exceeds
both `S` and the 95th percentile of that null distribution on common scoreable occurrences.

## Label-free evaluation

Report for every state and resolution:

- assigned-basis coverage, cluster-size entropy/Gini, largest/tiny/singleton fractions;
- seed ARI and common-basis ARI across views;
- affinity enrichment, modularity, and conductance;
- response/family/target recurrence and phase concentration;
- labeling-support counts under the frozen family partitions;
- held-out-family candidate-direction coherence.

Held-out-family coherence omits one family at a time from cluster prototype construction. For each
cluster, L2-normalize every supported five-channel basis-target vector and average the vectors from
the remaining families, then L2-normalize that centroid. For an omitted-family occurrence, the
primary score is cosine to its assigned-cluster centroid minus the maximum cosine to any other
nonempty cluster centroid. Compute the hierarchical mean first within target, response, and family;
comparisons between states use only occurrences scoreable in both states and report that common
coverage. Report own-cluster cosine, the primary between-cluster margin, coverage, family-block
bootstrap intervals, and family effects. Candidate clustering passes the functional gate only if
`C` or `F`:

1. improves family-weighted candidate-direction coherence over `W` by at least `0.05`;
2. has a 95% family-block-bootstrap lower bound above zero;
3. has positive improvement in at least 80% of family omissions;
4. exceeds `S` and the candidate-direction-null 95th percentile; and
5. for `F`, reduces width-one input-profile coherence by no more than `0.05`.

As secondary structural guardrails, require at least 95% assignment among the primary eligible
universe, largest cluster fraction at most 15%, mean/minimum seed ARI at least `0.72/0.70`,
modularity at least `0.20`, within-cluster affinity enrichment at least `1.25`, and at least 80%
of clusters with the frozen minimum labeling support. Failure remains visible; thresholds are not
relaxed after outcomes.

## Frozen discovery partitions

Before fitting or description generation, create a deterministic family-level `18/8/8` split:

- 18 generation families;
- eight selection-scoring families;
- eight audit families.

The family containing two responses remains intact. Sort families by the bytewise SHA-256 of
`candidate-aware-labelability-v1`, a NUL byte, and the canonical family ID; assign the first 18 to
generation, the next eight to selection, and the final eight to audit. The generated manifest
records the exact IDs, response/condition balance diagnostics, the namespace, and a self-hash.
Phase coverage is identical because every response has all seven C2 bins. No outcome-dependent
rebalancing is permitted. All three partitions remain discovery evidence; `audit` here is not the
untouched confirmatory holdout.

A labeling-ready cluster requires at least four generation, two selection, and two audit families,
and at least `8/4/4` target witnesses in those partitions.

## Factorial labeling pilot

If a candidate state passes the label-free functional gate, select 12 deterministic matched
cluster triplets at the common resolution. Match `W`, `C`, and `F` by signed-basis overlap using a
maximum-weight assignment, then stratify the retained triplets across member-size and witness-
support quantiles. Freeze the IDs before generating descriptions.

Run five arms to distinguish better evidence from better clusters:

1. `W` clusters with width-one evidence;
2. `W` clusters with width-one plus candidate evidence;
3. `C` clusters with combined evidence;
4. `F` clusters with combined evidence;
5. `S` clusters with combined evidence as the support control.

Combined witnesses show the local prefix and observed token, all five ranked competitors with
their probabilities, the cluster's signed rank-aligned contribution-difference vector, and the
width-one source-attribution highlights as separate evidence fields. Generation prompts never
contain selection or audit measurements. Prompts require a bounded local feature hypothesis and
forbid response-identity, causality, selectivity, generality, or faithfulness claims.

Use Opus as the fixed semantic generator/rewriter and Terra as the conservative abstention
control. Qwen may be added later without changing this comparison.

## Label validation and success

The fixed Transluce simulator continues to score only the input-localization hypothesis on the
selection and audit source-token attribution records. It is not a candidate-contribution
simulator. Candidate-direction coherence is validated separately by the label-free held-out
measurement gate above. Every label bundle attaches the measured numeric candidate signature;
natural-language candidate-effect wording remains exploratory and requires blinded human review.

A label is retained only when it:

- is not `insufficient_evidence`;
- has input-localization correlation at least `0.15` on selection and `0.10` on audit with the
  same sign;
- beats the best frozen generic local-token/formatting control by at least `0.05` on audit;
- belongs to a state that passed candidate-direction coherence when candidate evidence is claimed;
- passes blinded review for literal support, specificity, and claim boundaries.

The pilot succeeds only if `C` or `F`, relative to `W` with the same combined evidence:

- yields at least three additional retained labels among the 12 matched clusters;
- lowers abstention by at least 20 percentage points; and
- loses no more than one label that was valid on width-one input localization.

Comparing the two `W` arms isolates evidence-only benefit; comparing `C` or `F` with
`W + combined evidence` isolates reclustering benefit. A pass justifies a newly frozen
confirmatory replication or targeted intervention study. It does not automatically authorize
full-corpus candidate tracing.
