# Candidate Identity Alignment Protocol

## Purpose and claim boundary

This is an exploratory, label-free follow-up to the failed C2-W64 matched-evidence labeling
comparison. The earlier comparison showed that appending five rank-aligned candidate contrasts did
not reduce abstention. This protocol tests whether the same completed traces become more coherent
when competitor identity or exact longitudinal recurrence is preserved across targets.

The experiment does not retrace, alter C2-W64 assignments, infer per-occurrence candidate values
from compact sums, or claim causal/faithfulness structure. Labels, prior model outputs, audit
values/content, outcomes, and the separate confirmatory holdout are forbidden executable inputs.
Any later labeling result is an iterative exploratory result because the earlier 12-anchor outputs
motivated this test.

The feasibility pass inspected schemas and identity-recurrence counts in generation and selection,
but computed no transformed value-aware selection metric. Its Arrow table reads mechanically loaded
target metadata, including serialized candidate-selection metadata, for all three partitions before
filtering; audit rows were not parsed or summarized, and audit candidate-profile values and metrics
were not loaded. Bind the exact exposure in `plans/candidate-identity-feasibility-v1.json`.
Representation code, thresholds, seeds, and the complete variant family must be committed before
the one-shot selection evaluation.

## Frozen inputs and grain

Use the existing C2 input bundle, C2-W64 baseline, and multiplex assessment. The scientific record
grain is one `case_id x activation-signed basis`. Candidate metadata come from the target's frozen
`candidate_selection_json`; values come from the corresponding five-vector of
`contribution(model rank r) - contribution(observed token)`.

For ranks one through five, join by `full_distribution_rank`. The observed token is the reference,
not an additional competitor. If the observed token itself occupies a model-top-five rank, omit
that structural-zero slot from every assembled representation and support control. Token IDs under
the frozen tokenizer revision are authoritative; decoded text is metadata used only by the
deterministic surface key and later rendering.

All feature dictionaries and generation centroids use the 18 generation families only. Selection
contains eight frozen families and is evaluated once for the complete committed variant family.
Transformation/evaluation code never parses, summarizes, or loads audit candidate metadata or
candidate-profile values; the feasibility-only whole-target-table structural exposure is bound by
the receipt above.

## Representations

`R`, the mandatory comparator, uses sparse keys `rank:1` through `rank:5` for non-observed
competitors and the existing signed contrast values.

Test exactly four scientific variants:

1. `T`, competitor identity: key by exact competitor token ID.
2. `P`, ordered competition identity: key by exact
   `(observed_token_id, competitor_token_id)`.
3. `SR`, surface relation: key by the tuple
   `(observed_class, competitor_class, normalized_relation, leading_space_relation)`.
4. `M`, adjacent-phase recurrence motif: key by
   `(competitor_token_id, left_phase, right_phase, endpoint_side)` when the same activation-signed
   basis and same non-observed competitor token occur in adjacent phase bins of one response.

For `SR`, set `n = unicodedata.normalize("NFKC", text).casefold().strip()` in that order. Classify
`n` as `empty` when it is empty; `letters` when every Unicode category starts with `L`; `digits`
when every category is `Nd`; `alphanumeric` when every character is `L` or `Nd` and at least one
of each occurs; `punctuation_symbol` when every category starts with `P` or `S`; and `mixed`
otherwise. The leading-whitespace flag is `bool(text) and text[0].isspace()` under Python Unicode
semantics, and its relation is `same` only when observed and competitor flags agree. Set
`normalized_relation`, in precedence order, to `equal`, `prefix`, `suffix`, or `none`;
prefix/suffix requires both normalized strings to be non-empty, the shorter to contain at least two
Unicode code points, and one string to start/end with the other. Prefix wins when both apply to
unequal strings.

For `M`, every response must contain exactly one target in each phase zero through six. An edge
exists only between phases `p` and `p+1`, only for the exact same signed basis, and only for an
exact recurring competitor token. Anchor the motif to the left endpoint record
`(left_case_id, signed_basis_index)` and retain `right_case_id` as provenance. Its sparse vector
contains the left endpoint contrast at the `left` coordinate and the right endpoint contrast at the
`right` coordinate. Phase six and every unavailable/non-left record remain in the common
selection denominator with reciprocal rank zero. This gives `M` an explicit maximum 6/7 temporal
coverage before other missingness. These are recurrence annotations, not causal or contribution
paths. Do not use cluster identity to construct an edge or attach the same motif to both endpoints.

The generation dictionary is the union of applicable non-observed event keys, including events
whose delta is exactly zero. Drop selection-only keys identically from scientific and support
views. Within a scientific row, duplicate coordinates signed-sum. Within a support row, each
applicable event contributes `1.0`, so duplicate keys count events. Canonically sort and serialize
all keys. A scientific projection with no finite nonzero value is missing, never an all-zero
scientific vector; its support view may remain nonzero.

## Mandatory controls

For `R`, `T`, `P`, `SR`, and `M`, construct `R_support`, `T_support`, `P_support`, `SR_support`, and
`M_support` with the identical dictionary, hierarchy, normalization, and keys but support values as
defined above. This detects gains caused only by applicability or recurrence support. The name
`S` remains reserved for the earlier frozen support-only clustering state.

Construct 100 deterministic direction-null replicates for all four variants jointly. For `T`, `P`,
and `SR`, stratify generation rows by target, layer, polarity, and exact sparse-support signature.
For `M`, stratify the already constructed left-endpoint rows by response, left phase,
right phase, layer, polarity, and exact sparse-support signature. Within each stratum, assign L2
mass deciles by sorting `(L2 mass, signed_basis_index)` and setting the zero-based ordinal rank `q`
among `n` rows to `min(9, floor(10*q/n))`. Scan nonempty deciles in ascending order, merging
consecutive deciles until a block contains at least four rows; merge one short final block backward
when a prior block exists, otherwise leave it fixed. Permute each complete value-vector block
exactly once.

This preserves feature-key support, layer, polarity, local magnitude strata, target/motif coverage,
and untouched selection data while breaking generation signed-basis-to-direction identity. For
each variant, at least 80% of eligible generation rows and 80% of hierarchical generation weight
must belong to genuinely movable blocks; otherwise that variant cannot pass. All 100 replicates
must be valid. Invalid replicates fail rather than being dropped or redrawn.

Derive each seed as the first eight unsigned big-endian bytes of SHA-256 over the committed
protocol file SHA-256 text, a NUL byte, `candidate-identity-direction-null-v1`, a NUL byte, the
variant name, a NUL byte, and the eight-byte big-endian replicate index. Selection and support
controls remain fixed, and `R` remains the unpermuted comparator. For every replicate form two
joint statistics over `T`, `P`, `SR`, and `M`: the maximum null MRR improvement over `R`, and the
maximum null MRR improvement over the corresponding support control. Define each 95th percentile
with NumPy `quantile(..., 0.95, method="higher")`.

## Generation centroids and recurrence

L2-normalize each nonzero basis-target vector. For every frozen W64 cluster, average bases within
target, targets within response, responses within family, and then the 18 family means equally.
Do not renormalize intermediate means; L2-normalize only the final cluster centroid.

Separately compute recurrent-basis consistency for each scientific variant. Unit-normalize each
anchored row; for each signed basis with at least three anchored cases, two responses, and two
families, mean target -> response -> family without intermediate normalization. Consistency is the
L2 norm of the final mean. For `M`, anchored cases are left endpoints only. Record recurrence
coverage and the full distribution. Generation scoreable weight is the hierarchical weight of rows
with a nonzero projected vector and an available true-cluster centroid.

## One-shot selection metric

For every W64-assigned candidate-supported selection basis-target record, cosine-rank all available
generation centroids. The true cluster is its frozen W64 assignment. Use descending cosine and
one-based average rank across exact ties. Reciprocal rank is zero when the assembled vector is
missing, the true centroid is unavailable, or no competitor centroid is available.

Reduce reciprocal rank equally over basis occurrences within target, targets within response,
responses within family, then the eight families. Report zero-filled MRR, scoreable hierarchical
weight, target/response/family coverage, and per-family effects relative to `R`.

Use 10,000 paired family-block bootstrap replicates. For each comparison, draw the eight fixed
family effects with replacement and average them. Derive the seed as the first eight unsigned
big-endian bytes of SHA-256 over the committed protocol file SHA-256 text, a NUL byte,
`candidate-identity-selection-bootstrap-v1`, a NUL byte, and the comparison name. Report linear
2.5/97.5 percentile intervals. Invalid replicates fail rather than being dropped or redrawn.

## Evidence-assembly gate

A variant passes only if every condition holds:

1. all values and receipts are provenance-valid and all eight selection families are present;
2. generation scoreable hierarchical weight is at least 80% of `R`;
3. selection scoreable hierarchical weight is at least 80% of `R`;
4. generation median recurrent-basis consistency is at least `0.55`;
5. zero-filled selection MRR exceeds `R` by at least `0.03`;
6. at least seven of eight family MRR effects are positive;
7. the paired family-bootstrap lower bound for variant minus `R` is above zero;
8. value-aware MRR exceeds the variant's support-only MRR with a paired-bootstrap lower bound
   above zero; and
9. the observed variant-minus-`R` effect exceeds the 95th percentile of its joint max-null; and
10. the observed variant-minus-support effect exceeds the 95th percentile of its separate joint
    max-null.

The `0.55` consistency and `0.03` lift thresholds reuse earlier frozen C2 precedents but define a
new endpoint here. Failure is evidence against these four assemblies, not against top-five tracing
in general.

If multiple variants pass, choose the largest MRR improvement, then the largest value-minus-support
effect, then the fixed simplicity order `T`, `P`, `SR`, `M`. No representation or threshold changes
are allowed after selection results are computed. Call this choice the `offline_winner`. Separately
choose the `local_labeling_winner` by the same rule restricted to passing variants in `T`, `P`, and
`SR`. An `M` offline win neither blocks a passing local variant from labeling nor authorizes the
single-target renderer.

## Conditional labeling

If and only if a `local_labeling_winner` exists, publish a new immutable evidence renderer on the
existing 12 W64 anchors. Compare width-one evidence against the identical witnesses plus that
assembled candidate evidence. Preserve the existing local/single-target wording, typed outputs,
five Luna samples, Terra rewrite, Terra conservative control, native Batch receipts, and hard
cumulative cost guard. Do not mutate or reuse the old rank-slot renderer as if it were the new
cohort.

`M` is longitudinal and contains right-endpoint values, so an `M` pass cannot authorize the
single-target renderer. It instead requires a separately frozen longitudinal paired-cohort
protocol, left/right witnesses shown identically to both arms, generation-only support-ready anchor
selection, sampled-adjacent-phase wording, and a new success rule before any API call. An `M` pass
therefore authorizes that non-billable renderer-design step, not immediate labeling spend.

The earlier labeling success rule remains unchanged: at least three additional retained local
hypotheses, abstention lower by at least 20 percentage points, and no more than one lost width-only
hypothesis. If no representation passes the offline gate, make no API calls.

Winner-based reclustering is a separate later decision. It is not part of this test and requires
the existing structural, stability, selection, support-control, null, width-preservation, and
label-readiness gates before labels.
