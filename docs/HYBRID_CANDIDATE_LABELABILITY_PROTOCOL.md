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

For each view, the target statistic is mean same-cluster cosine minus mean different-cluster
cosine. A target is scoreable only when both sets are nonempty. Use one fixed pair pool common to
both states within a view. Reduce target means equally within response, response means equally
within family, and family means equally within partition. Report coverage and per-family effects.

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
An invalid refit is a failed replicate. Require median ARI at least `0.60` and p10 ARI at least
`0.45` independently for both states.

## Witness readiness

Count candidate and input-profile support from newly validated hybrid occurrences mapped to the
frozen assignments. A cluster is ready only when both evidence types contain at least:

- generation: eight targets from four families;
- selection-scoring: four targets from two families;
- audit: four targets from two families.

At least 80% of the 64 clusters must be ready in each state. Audit witnesses remain scoring-only
and must never enter prompts.

Before an API call, materialize and hash each ready cluster's exact generation, selection-scoring,
and audit witness inventory and verify nonempty source profiles and tokenizer alignment. Actual
Transluce scoring is post-label evaluation, not a pre-label gate.

## Exploratory labeling authorization

A state authorizes exploratory labeling only when all conditions pass:

1. the frozen structural guardrails pass;
2. generation-family jackknife median/p10 ARI pass;
3. at least 80% of clusters are witness-ready;
4. all eight families are scoreable in both held-out-from-fit partitions; and
5. input-view coherence is positive, its family-bootstrap 95% lower bound is above zero, and at
   least seven of eight family effects are positive in both partitions.

Candidate-view coherence is reported under the same diagnostics. It is required to describe a
state as candidate-direction coherent, but it is not required merely to run exploratory
input-localization labeling when the input-view gate passes. No minimum lift over W64 is used.

If neither state passes, make no labeling calls. If exactly one passes, label only that state. If
both pass, label both. Use the OpenAI recipe and the existing cost guard; start with 12
deterministically spread ready clusters per passing state. The labeling result remains exploratory
until local simulator scoring and label-quality assessment complete.

## Persistence and provenance

Publish evaluation and labeling-input artifacts atomically. Bind the exact source input and fit
manifest hashes, this protocol hash, clean Git commit and executable file hashes, fixed state
identities, family partitions, target/artifact payload hashes, signed-basis mapping, coverage,
bootstrap seeds, jackknife fits, readiness records, and every output file hash. Deep loaders must
recompute these bindings and fail closed on drift. Persist separate booleans for
`exploratory_labeling_authorized` and `scientific_promotion_authorized`; the latter remains false.
