# Candidate-union C0 results

The two-pass C0 candidate-union run completed on 2026-07-29 from clean commit
`19c236be0750fa2d0296d5790d309d900e13c3ca`.

## Execution

- Smoke job: `14357105`, completed in 3m56s.
- Dense wave: `14357398`, completed in 23m50s.
- Broad wave: `14357399`, completed in 18m22s.
- Output root:
  `/scratch/general/vast/u1653998/circuits/results/bonafide/downstream/candidate-union-v1`
- All 55 candidate-specific refinement artifacts and all 10 assembled union
  artifacts passed checksum, schema, topology, and numerical validation.
- The output root occupies 58 MiB: 17 MiB of assembled unions and 37 MiB of
  independently resumable refinement traces, plus summaries.

The median complete-case runtime was 243.2 seconds, the maximum was 390.8
seconds, and the maximum reserved GPU memory was 20.76 GiB on an A100 80 GB.

## Dense measurements

Across the ten cases, the exact pass-one unions contain 5,417 node rows and
272,206 edge rows. Candidate applicability expands these to:

- 29,578 node-candidate measurements;
- 1,499,091 edge-candidate measurements.

Pass two adds measurements that were not selected in the corresponding
candidate's pass-one graph:

- 5,208 node measurements, 17.6% of applicable node entries;
- 880,789 edge measurements, 58.8% of applicable edge entries.

There are 1,579 union-node rows with both positive and negative candidate
contributions. Node activations and internal edge weights are exactly invariant
across the pass-two candidate axis; the candidate-dependent quantities are node
contributions and node/edge attributions.

No applicable pass-two value was exactly zero in this cohort, although the
artifact contract preserves zero values when they occur.

## What missing meant

For nodes, the threshold interpretation is a good approximation but not an
exact contract. Of 5,208 newly measured MLP node entries, 5,135 (98.6%) have
absolute normalized attribution below `0.005`. The remaining 73 range from
`0.0050245` to `0.0079044`. ADAG selects nodes using a raw goal-relative
threshold before dataframe normalization, so the later normalized value does
not provide a strict cutoff certificate.

For edges, the cutoff interpretation is wrong in practice. Every one of the
880,789 newly measured edge entries was absent because at least one endpoint
node was absent in pass one. Only five have absolute weight below `0.01`;
880,784 have absolute weight at or above the configured edge threshold. Thus a
missing edge generally cannot be treated as a small edge with a `0.01` upper
bound. The node-pruning decision prevented it from being considered for that
candidate.

## Reproduction and numerical sensitivity

All pass-one selected node attributions, contributions, and activations are
reproduced exactly in pass two.

Of 618,302 selected edge entries, 617,560 are reproduced exactly. The remaining
742 entries (0.12%) all occur in `c0-broad-faithful-width6-p38`: 634 embedding
edges and 108 internal edges. Their maximum absolute attribution difference is
`0.0064655` and maximum absolute weight difference is `0.84375`.

This one-case discrepancy is consistent with topology-dependent bfloat16
batching: pass one computes selected-neuron gradients and Jacobians over each
candidate's smaller selected set, while pass two computes them over the shared
union. Pass-two internal weights are nevertheless identical across candidates
on that shared basis. For downstream candidate comparison, the pass-two values
should therefore be treated as the canonical common-topology profile while the
pass-one membership masks and reference values remain available for audit.

## Interpretation

The two-pass system achieves the intended comparison contract:

- independent candidate objectives still define pass-one topology and preserve
  provenance;
- the union does not invent all-pairs topology;
- candidate profiles are dense over every applicable union node and exact union
  edge;
- terminal logit edges remain candidate-specific;
- missingness and measured zero remain distinguishable.

The result is still a pruned union. A node or edge absent from every independent
pass-one graph is outside the refinement set and remains unmeasured.

## Decision

C0 passes. The project locks `model_top5_plus_observed` plus independent
candidate-specific `k=1` topology, exact edge/node union, and fixed-union
candidate rescoring as the production approach.

The next gate is C1: freeze and run a 24--48-target,
family/response-balanced discovery cohort to establish end-to-end resource,
storage, numerical-health, realized-width-five, and realized-width-six bounds
for this exact two-pass contract. C1 does not authorize C2 or the full matched
corpus by itself.
