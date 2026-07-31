# C2 candidate-aware clustering and labelability salvage protocol

Status: frozen before candidate-aware cluster fitting, cluster selection, or description
generation.

Every executable input manifest must bind this file's SHA-256 and a clean Git commit containing it;
dirty, missing, or mismatched protocol provenance fails closed.

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

This deliberately differs from the attribution-sign identity used inside the completed C2
retrieval analyzer. Neither of its profile dictionaries is reused. Recompute both views from raw
artifacts, require every retained internal node's activation to be candidate-invariant, and emit an
attribution-sign to activation-sign crosswalk with agreement/disagreement counts and weights. Any
candidate-varying internal activation fails input construction. The generated input manifest binds
the new basis index, crosswalk, feature files, and payload hashes.

Candidate invariance uses NumPy-style `allclose(candidate, candidate_zero, rtol=1e-6,
atol=1e-7)` for every applicable internal-node activation. Record maximum absolute and relative
deviation, using `max(abs(candidate_zero), 1e-12)` in the relative denominator, plus comparison and
violation counts. Any violation blocks input publication.

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
4. `S`, support-only control: weighted candidate-union target-support Jaccard with no directional
   contribution values. For bases `i,j`, its affinity is
   `sum_t w_t I(i and j supported) / sum_t w_t I(i or j supported)` over generation targets. A
   candidate-union occurrence counts as support even when its direction vector has zero norm. Apply
   the same primary universe, minimum two co-supported targets/responses/families, positive 32-NN,
   and `union_max` rules as the scientific states; do not percentile-calibrate `S`.

The primary matched universe contains only bases eligible in both `W` and `C`. Candidate-union
bases outside that universe are reported as an expanded-coverage diagnostic and cannot make the
primary comparison pass.

## Fit grid and deterministic state choice

Use hierarchical equal-family, then equal-response, then equal-target weights. Primary basis
eligibility requires support in at least three targets, two responses, and two families in both
scientific views. A pair requires valid directional overlap in at least two targets, two responses,
and two families; `F` requires those gates independently in both views.

Only the 18 frozen generation families enter basis eligibility, pair evidence, empirical-CDF
calibration, affinity construction, seed-medoid selection, or assignments. Selection and audit
families never influence a fitted cluster state.

For `W`, `C`, and `F`, fit normalized sparse spectral clustering with:

- cluster counts `32, 64, 96`;
- seeds `17, 29, 43`;
- positive 32-nearest-neighbor affinity;
- deterministic `union_max` symmetrization;
- self-loop weight `1.0` and eigen tolerance `1e-6`.

The primary matched comparison uses 64 clusters if all three views yield numerically valid states.
Otherwise it uses the smallest tested count valid for all three; if none is common, the matched
comparison fails. A state is numerically valid only when the generation-only affinity is finite,
exactly symmetric, has at least `n_clusters + 1` active bases, has no more connected components
than requested clusters, converges without eigensolver fallback, and assigns at least 95% of the
primary eligible universe. Disconnected or unassigned bases remain explicit and are never
force-assigned. Within one view/count, the medoid seed maximizes mean assignment ARI to the other
seeds, with the smaller seed breaking ties. `S` uses the chosen common count and the same spectral
settings.

## Support and directional nulls

The support-only state is mandatory because the C2 salvage signal was largely explained by
candidate-score applicability.

Also run 100 deterministic candidate-direction null refits using generation evidence only. Within
each target/layer/activation-polarity group, assign candidate-vector L2 mass deciles, scan nonempty
deciles in ascending order, and merge consecutive deciles into a disjoint block until it contains
at least four bases. If the final block is smaller than four, merge it backward once into the prior
block; if no prior block exists, leave it fixed. Permute each resulting block exactly once. Report
movable basis-occurrence and hierarchical target-weight fractions. If either fraction is below 80%,
the null is ineffective and no directional result can pass. This preserves topology/support, the
candidate set, layer/polarity composition, and local vector-magnitude distribution while breaking
signed-basis-to-competitive-direction identity.

For replicate `r=0..99`, derive the RNG seed as the unsigned big-endian integer in the first eight
bytes of SHA-256 over the protocol file SHA-256 text, a NUL byte, `direction-null-v1`, a NUL byte,
and the eight-byte big-endian replicate index. All 100 refits must be numerically valid. Define the
null 95th percentile with NumPy `quantile(..., 0.95, method="higher")`; an invalid replicate makes
the directional gate fail rather than being discarded or replaced.

Each null replicate repeats candidate and fusion construction, common-count selection, and seed-
medoid selection, then scores its assignments against original unpermuted selection and audit
vectors. The null statistic is `max(C improvement over W, F improvement over W)` among numerically
valid states, so choosing between `C` and `F` is multiplicity-controlled. A candidate result is
directional only when it exceeds both `S` and the corresponding 95th-percentile max statistic on
common scoreable occurrences, first on selection and independently on audit.

## Label-free evaluation

Report for every state and resolution:

- assigned-basis coverage, cluster-size entropy/Gini, largest/tiny/singleton fractions;
- seed ARI and common-basis ARI across views;
- affinity enrichment, modularity, and conductance;
- response/family/target recurrence and phase concentration;
- labeling-support counts under the frozen family partitions;
- selection and audit candidate-direction coherence.

For each fitted cluster, L2-normalize every supported five-channel generation basis-target vector,
average them under hierarchical occurrence weights, then L2-normalize that generation centroid.
For a selection or audit occurrence, the primary score is cosine to its assigned-cluster centroid
minus the maximum cosine to any other nonempty cluster centroid. Compute the hierarchical mean
first within target, response, and family; comparisons between states use only occurrences
scoreable in both states and report that common coverage. Report own-cluster cosine, the primary
between-cluster margin, coverage, family-block bootstrap intervals, and per-family effects.

Width-one input-profile coherence is computed within each selection or audit target, where source-
token coordinates are aligned: mean cosine for scoreable pairs assigned to the same cluster minus
mean cosine for scoreable pairs assigned to different clusters. Reduce hierarchically by target,
response, and family and compare states only on common scoreable pairs. This is the `F` preservation
metric; input maps are never concatenated or centroided across unrelated target sequences.

Separately, leave out each generation family, rebuild eligibility, pair evidence, calibration,
affinity, all three seed states, and the medoid assignment, then compare that refit with the full
generation state on common assigned bases. Report median and p10 family-jackknife ARI; require at
least `0.60/0.45` for a scientific state. Candidate clustering passes the functional gate only if
`C` or `F`:

1. improves family-weighted candidate-direction coherence over `W` by at least `0.05`;
2. has a 95% family-block-bootstrap lower bound above zero separately on selection and audit;
3. has positive improvement in at least seven of eight families in both partitions;
4. exceeds `S` and the candidate-direction-null 95th percentile; and
5. for `F`, reduces width-one input-profile coherence by no more than `0.05`.

As secondary structural guardrails, require at least 95% assignment among the primary eligible
universe, largest cluster fraction at most 15%, mean/minimum seed ARI at least `0.72/0.70`,
modularity at least `0.20`, within-cluster affinity enrichment at least `1.25`, and at least 80%
of clusters with the frozen minimum labeling support. Failure remains visible; thresholds are not
relaxed after outcomes.

For each partition and state comparison, compute 10,000 paired family-block bootstrap replicates
from the eight fixed family effects. Every family must have common scoreable observations or that
partition fails the functional gate. Draw eight family IDs with replacement and average their
fixed paired effects; no occurrence-level resampling or pool reconstruction occurs. Derive the RNG
seed from the first eight bytes, interpreted unsigned big-endian, of SHA-256 over the protocol file
SHA-256 text, a NUL byte, the partition name, a NUL byte, and `candidate-coherence-bootstrap-v1`.
Use NumPy `quantile` with `method="linear"` for the 2.5th and 97.5th percentiles. No invalid
bootstrap replicate may be dropped or redrawn.

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

Select 12 deterministic labeling-ready `W` anchors using a fixed two-dimensional quantile design.
For each ready `W` cluster, compute ascending midrank percentiles `(midrank - 0.5) / n` separately
for member count and generation-target witness count, breaking exact value ties by cluster ID only
after assigning the shared midrank. Define 12 target points as the Cartesian product of member
coordinates `(1/6, 1/2, 5/6)` and support coordinates `(1/8, 3/8, 5/8, 7/8)`, in that order.
Use Hungarian minimum-cost assignment with squared Euclidean rank distance and cluster ID as the
final deterministic tie-breaker. Fewer than 12 ready `W` clusters fails both label pilots; there is
no sparse-cell fallback. Independently match `W` to `C`, `F`, and `S`
with Hungarian maximum-weight assignment on signed-member-basis Jaccard similarity, ordered first
by `W` cluster ID and then comparison cluster ID for deterministic ties. The 12 `W` anchors are the
paired denominator. A comparison match below Jaccard `0.10`, a missing output, or an unscoreable
output counts as an abstention; it is never replaced after labels are seen. Freeze all IDs and
overlaps before description generation.

Run five arms to distinguish better evidence from better clusters:

1. `W` clusters with width-one evidence;
2. `W` clusters with width-one plus candidate evidence;
3. `C` clusters with combined evidence;
4. `F` clusters with combined evidence;
5. `S` clusters with combined evidence as the support control.

Combined witnesses show the local prefix and observed token; the five model-rank slots with token,
logit, probability, and observed-token flag; the cluster candidate signature; and width-one source-
attribution highlights as separate evidence fields. A width-five case contains four distinct
competitors because one rank slot is the observed token's structural zero; a width-six case has
five competitors.

For target `t` and cluster `k`, include every supported occurrence of an assigned signed basis in
that candidate-union node table. Let `d_b` be its rank-aligned five-vector after within-basis signed
sum over repeated token occurrences. Persist and render, without clipping: member-occurrence count
`m`; `sum_b d_b`; the elementwise mean `(sum_b d_b)/m`; its L2 norm; and the unit direction when
the norm is nonzero. Missing members are omitted and reflected only in `m`. JSON stores full
finite double precision; prompts render six significant digits in rank order one through five.
Generation prompts never contain selection or audit measurements. Prompts require a bounded local
input feature hypothesis, type candidate-effect prose separately as exploratory, and forbid
response-identity, causality, selectivity, generality, or faithfulness claims.

Run arms 1 and 2 regardless of whether `C` or `F` passes the functional clustering gate; this tests
whether richer evidence alone reduces abstention on unchanged clusters. Run arms 3 through 5 only
after at least one candidate scientific state passes. Use Opus as the fixed semantic generator and
rewriter and Terra as the conservative abstention control. Qwen may be added later without changing
this comparison.

## Label validation and success

Every model output is typed as `input_localization_hypothesis`,
`exploratory_candidate_description`, `background_or_confound`, `limitations`, and `status`. The
fixed Transluce simulator scores only `input_localization_hypothesis` on selection and audit
source-token attribution records. It is not a candidate-contribution simulator. Candidate-
direction coherence is validated separately by the held-out measurement gate above. Every bundle
attaches the measured numeric candidate signature. `exploratory_candidate_description` is never
called validated; it may be retained as descriptive text only after blinded literal review.

Freeze these three shared input-localization controls, in order:

1. `tokens near the current response position`;
2. `common punctuation and formatting tokens`;
3. `shared instruction-template and response-boilerplate tokens`.

Their ordered-list canonical SHA-256 is
`2f222eecb6f07350fd9f2f4c0217116b26158af00c50a1f02550c22309e5bf12`. Score all three on the
same selection records, select the highest-correlation control per cluster with list order as the
tie-breaker, and carry that one unchanged to audit. Audit never selects a control.

Blinded literal review uses two independent reviewers. Each sees one randomly identified output,
its exact generation/selection/audit witnesses, and the typed limitations, but not provider, arm,
view, cluster ID, automatic scores, competing-arm outputs, or pass thresholds. For the input
hypothesis each reviewer answers yes/no to: localized evidence is literal; wording is more specific
than a frozen control; limitations preserve the claim boundary. For exploratory candidate text
they separately answer whether the prose literally matches the displayed five-channel numeric
signature. Accept a component only on unanimous yes answers. Any disagreement goes to a third
reviewer under the same blinding, with the majority decision final. The deterministic blinded-ID
mapping and completed forms are hashed before unblinding. Until both primary reviews and any needed
adjudication exist, report automated results as `pending_blinded_review`, not retained labels.

A label is retained only when it:

- is not `insufficient_evidence`;
- has input-localization correlation at least `0.15` on selection and `0.10` on audit with the
  same sign;
- beats the best frozen generic local-token/formatting control by at least `0.05` on audit;
- belongs to a state that passed candidate-direction coherence when its exploratory candidate
  description is reported;
- has its input-localization wording pass blinded review for literal support, specificity, and
  claim boundaries.

The evidence-only pilot succeeds if arm 2, relative to arm 1 on the same 12 `W` anchors:

- yields at least three additional retained input-localization hypotheses;
- lowers abstention by at least 20 percentage points; and
- loses no more than one hypothesis retained in arm 1.

The reclustering pilot succeeds only if `C` or `F`, relative to arm 2 on the same 12 `W`-anchored
matches:

- yields at least three additional retained labels among the 12 matched clusters;
- lowers abstention by at least 20 percentage points; and
- loses no more than one label that was valid on width-one input localization.

Missing or weak matches remain in both abstention and gain/loss denominators. The two decisions are
reported separately: candidate evidence may help labeling even if candidate-based reclustering
fails. A pass justifies a newly frozen confirmatory replication or targeted intervention study. It
does not automatically authorize full-corpus candidate tracing.
