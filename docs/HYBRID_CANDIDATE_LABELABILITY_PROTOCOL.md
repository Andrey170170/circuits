# Hybrid candidate clustering labelability protocol

Status: frozen exploratory evaluation. This protocol may authorize an exploratory labeling run;
it cannot establish confirmatory scientific promotion. The distinct confirmatory holdout remains
closed.

## Question and fixed states

The evaluation asks whether the richer candidate-union tracing states frozen by
`HYBRID_CANDIDATE_CLUSTERING_PROTOCOL.md` are stable across generation families and have enough
held-out-from-fit local evidence to support exploratory labels.

Before selection-scoring or audit artifacts are opened, freeze exactly these states from the
immutable hybrid fit:

- `primary`: `raw_top5_plus_observed.v1`, `full_positive`, `K=64`, medoid seed `17`;
- `alternative`: `paper_normalized_model_top5.v1`, `full_positive`, `K=64`, medoid seed `29`.

No state, affinity, resolution, seed, recurrence gate, or threshold may change after held-out
inspection. W64 overlap is descriptive only and cannot satisfy a gate because its basis universe
and extraction contract differ.

## Partition firewall and basis universe

Generation is the only fit partition. Selection-scoring and audit are held out from fitting but
have been inspected by earlier C2 discovery analyses, so results are exploratory discovery
evidence rather than a fresh confirmatory test.

Reopen each selection-scoring and audit target only through the validated candidate-union artifact
and its observed-candidate fixed-union refinement. Apply the same identity, payload, candidate-axis,
activation, and node-binding checks as hybrid input construction. Map every occurrence by the
canonical signed-basis identity into the immutable 11,585-basis generation universe. Report and
ignore held-out-only bases; never expand the universe, refit on held-out targets, or decode labels,
task outcomes, or prior generated descriptions.

## Target-local coherence

Candidate axes and source-token axes are target-local. Do not concatenate or centroid them across
unrelated sequences. For each target and state, compute cosine similarity for scoreable pairs of
assigned bases separately for:

- the state's input representation;
- the state's candidate representation; and
- a shared raw model-top-five-minus-observed contrast representation used only for a fair
  primary-versus-alternative diagnostic.

For each view, enumerate all unordered pairs of mapped bases; no pair sampling is permitted. A pair
enters the fixed common pool only when both bases are assigned in both states and both vectors have
finite, nonzero norm on their jointly supported coordinates in both states. Construct this
intersection before classifying the pair as same-cluster or different-cluster separately in each
state. The primary native candidate cosine uses its complete target-local width-five or width-six
axis; the alternative native cosine uses its target-local model-top-five axis. These differently
sized vectors are never compared to one another: only pair identities and validity are
intersected. The shared contrast diagnostic uses the same target-local five model-rank coordinates
for both states. Missing support, an empty overlap, or a zero norm is unscoreable and is never
zero-filled.

For each state and view, the target statistic is mean same-cluster cosine minus mean
different-cluster cosine. A target is scoreable only when both sets are nonempty. Reduce target
means equally within response, response means equally within family, and family means equally
within partition. Report common-pool, same/different, target, response, and family coverage.

For each state, apply 10,000 family-block bootstrap replicates independently in selection-scoring
and audit. Sample the eight frozen family IDs with replacement and average their fixed family
statistics. Derive the seed from SHA-256 over this protocol's file SHA-256 text, a NUL byte, the
partition, a NUL byte, the state role, a NUL byte, the view, and a NUL byte followed by
`hybrid-coherence-bootstrap-v1`. Use NumPy linear 2.5th and 97.5th percentiles. Invalid replicates
fail rather than being redrawn.

## Generation-family jackknife

For each of the 18 generation families, leave that family out and rebuild the state's recurrence
eligibility, full-positive affinity, all three frozen seed fits, and medoid selection at `K=64`.
Compare the jackknife medoid with the frozen full-data medoid on common assigned bases using ARI.
The common set must contain at least 80% of the bases assigned by the frozen full-data state;
otherwise that replicate and the entire state fail. Any invalid refit, unrealized cluster, missing
seed, nonfinite statistic, or coverage failure fails the entire state and is never excluded,
replaced, or converted to zero for a median computed over fewer replicates. Require all 18 valid
replicates, median ARI at least `0.60`, and p10 ARI at least `0.45` independently for both states.

## Witness readiness

Count candidate and input-profile support from newly validated hybrid occurrences mapped to the
frozen assignments. One target is a joint witness only when at least one assigned cluster member
has both a finite nonzero candidate vector and a finite nonzero input vector on nonempty supported
coordinates in that same reduced target occurrence. Repeated node occurrences are reduced before
this check. Count distinct target and family IDs, never occurrence rows. Candidate zero norms,
input zero norms, and empty input support do not count. A cluster is ready only when its joint
witnesses contain at least:

- generation: eight targets from four families;
- selection-scoring: four targets from two families;
- audit: four targets from two families.

At least 52 of 64 clusters must be ready in each state (`ceil(0.80 * 64)`). Audit witnesses remain
scoring-only and must never enter prompts.

Before an API call, materialize and hash each ready cluster's exact generation, selection-scoring,
and audit witness inventory and verify nonempty source profiles and tokenizer alignment. From the
jointly supported inventory, freeze exactly `8/4/4` generation/selection-scoring/audit witnesses.
Within each partition, hash-sort families by the protocol hash, state role, cluster ID, partition,
and family ID; take one hash-first target from each of the required `4/2/2` families, then fill the
remaining `4/2/2` positions from still-unselected targets in global hash order. Persist the exact
ordered target IDs and hashes. Only generation witnesses enter candidate-generation prompts;
generation plus selection-scoring enter summary prompts; audit never enters an API prompt. Actual
Transluce scoring is post-label evaluation, not a pre-label gate.

## Exploratory labeling authorization

A state authorizes exploratory labeling only when all conditions pass:

1. the frozen structural guardrails pass;
2. generation-family jackknife median/p10 ARI pass;
3. at least 80% of clusters are witness-ready;
4. all eight families are scoreable in both held-out-from-fit partitions; and
5. both input-view and native candidate-view coherence are positive, their family-bootstrap 95%
   lower bounds are above zero, and at least seven of eight family effects are positive in both
   partitions.

Candidate-view coherence is required because the hybrid prompt exposes candidate-union evidence.
No minimum lift over W64 is used.

If neither state passes, make no labeling calls. If exactly one passes, label only that state. If
both pass, label both. Use the OpenAI recipe and the existing cost guard; start with 12 ready
clusters per passing state. Select those 12 before any model call using the frozen two-dimensional
quantile rule from the earlier C2 protocol: ascending midrank percentiles for member count and
generation joint-witness target count; target points are the Cartesian product of member
coordinates `(1/6, 1/2, 5/6)` and support coordinates `(1/8, 3/8, 5/8, 7/8)`; use Hungarian
minimum squared-distance assignment and the lexicographically smallest cluster-ID tuple among
global optima. Persist the ordered IDs. The labeling result remains exploratory until local
simulator scoring and label-quality assessment complete.

## Persistence and provenance

Publish evaluation and labeling-input artifacts atomically. Bind the exact source input and fit
manifest hashes, this protocol hash, clean Git commit and executable file hashes, fixed state
identities, family partitions, target/artifact payload hashes, signed-basis mapping, coverage,
bootstrap seeds, jackknife fits, readiness records, and every output file hash. Deep loaders must
recompute these bindings and fail closed on drift. Persist separate booleans for
`exploratory_labeling_authorized` and `scientific_promotion_authorized`; the latter remains false.
