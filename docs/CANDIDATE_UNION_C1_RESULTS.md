# Candidate-union C1 results

The two-pass C1 policy/resource run completed on 2026-07-29. Pass one is bound
to clean commit `630265f3e8f98d4da70a571280dfd93ece955d43`; pass two
is bound to clean commit `d8896c0da5e00df15bea1903345f5a7ee57e7db2`.

## Frozen cohort and execution

The cohort contains 32 discovery targets:

- 16 dense and 16 broad targets;
- 26 responses and 25 base-question families, with at most three targets per
  response and four per family;
- all four CoT phenotypes and all six screened hint types;
- input lengths from 124 to 912 tokens;
- 17 realized-width-five targets and 15 realized-width-six targets.

The observed teacher-forced token was outside the model top five for 15/32
targets (46.9%), with observed ranks from 6 to 46 in that stratum. The frozen
candidate rule remained `model_top5_plus_observed`: width five when the
observed token was in model top five and width six otherwise.

The launch bundle is
`scripts/bonafide/manifests/qwen3_4b_instruct_topk_c1_launch_bundle_v1.json`
(SHA-256
`aefc05ebfeb18e60930ea9730ff59503f05de98e7329249a23b4c7b75baf104f`).
The pass-two plan is
`scripts/bonafide/manifests/qwen3_4b_instruct_candidate_union_c1_plan_v1.json`
(SHA-256
`97838138bd4e4b2e205d7aadd240d1dd7c67d9701b01fe6511e66d430210ab2c`).
It binds all 175 references by artifact ID, payload hash, source target,
response position, and candidate order.

Pass-one jobs `14359219` through `14359230` all completed with exit code zero.
Pass-two jobs `14360263` through `14360266` all completed with exit code zero.
Their summed scheduler GPU time was 2.66 and 2.71 A100-hours, respectively,
or 5.37 A100-hours total.

The output roots are:

- pass one:
  `/scratch/general/vast/u1653998/circuits/results/bonafide/downstream/top5-c1-v1`;
- pass two:
  `/scratch/general/vast/u1653998/circuits/results/bonafide/downstream/candidate-union-c1-v1`.

They occupy 123 MiB and 202 MiB. All 175 pass-one traces, 175 pass-two
refinement traces, and 32 assembled unions passed checksum, schema, topology,
candidate-order, provenance, and finite-value validation.

## Runtime and resources

Per target, excluding the small once-per-wave model-load overhead:

| Measurement | Median | p95 | Maximum |
| --- | ---: | ---: | ---: |
| Pass-one candidate traces, summed | 273.1 s | 483.9 s | 520.5 s |
| Pass-two union refinement | 287.0 s | 455.1 s | 477.7 s |
| End-to-end two-pass compute | 571.2 s | 930.8 s | 998.2 s |

Realized width affected cost as expected:

| Width | Targets | Median pass two | Median end to end |
| --- | ---: | ---: | ---: |
| 5 | 17 | 261.2 s | 555.1 s |
| 6 | 15 | 329.8 s | 643.7 s |

Pass-one individual candidate traces had 49.1 s median, 88.5 s p95, and
175.3 s maximum wall time. The maximum pass-one reserved HBM was 27.33 GiB.
The maximum pass-two reserved HBM was 26.28 GiB, leaving at least 52.97 GiB
headroom on an A100 80 GB. Maximum host RSS was 5.93 GiB. No OOM, resource
gate, scheduler timeout, ordinary trace error, or numerical failure occurred.

An assembled union contains a median 548 nodes and 34,499 edges, with maxima
of 1,038 nodes and 67,311 edges. Its median serialized size is 1.77 MB and its
maximum is 3.32 MB. Independently resumable pass-two refinements use a median
3.95 MB and maximum 6.71 MB per case.

## Dense measurement result

Across all 32 cases, the exact pass-one unions contain 19,287 node rows and
1,129,582 edge rows. Candidate applicability expands these to:

- 105,293 node-candidate measurements;
- 6,119,326 edge-candidate measurements.

Pass two adds measurements absent from the corresponding candidate's pass-one
graph:

- 16,882 node entries, 16.0% of applicable node measurements;
- 3,361,742 edge entries, 54.9% of applicable edge measurements.

Of the newly measured nodes, 16,714 (99.0%) have absolute normalized
attribution below `0.005`; the maximum is `0.0079044`. As in C0, this is a
strong empirical pattern but not a strict normalized-threshold certificate
because ADAG prunes against a raw goal-relative value.

Of the newly measured edges, 3,361,740 have at least one endpoint absent from
that candidate's pass-one graph. Only two have both endpoints present, and
only 24 of all newly measured edges have absolute weight below `0.01`.
Therefore missing edge membership still overwhelmingly means that upstream
node pruning prevented the edge from being considered, not that the edge
weight was below the edge threshold.

No newly measured applicable node attribution, edge attribution, or edge
weight is exactly zero in this cohort. The format nevertheless preserves
measured zero separately from inapplicable null.

## Candidate-profile information

Every raw node-contribution profile matrix has full candidate rank:

- all 17 width-five cases have rank five;
- all 15 width-six cases have rank six.

After centering across candidates, every matrix has the maximum possible
contrastive rank: four for width five and five for width six. Entropy effective
rank ranges from 1.57 to 4.72, with median 3.16. There are 5,693 union-node
rows with both positive and negative candidate contributions.

The profiles are therefore non-degenerate at the resource gate. This does not
establish scientific utility; C2 must test whether these candidate differences
add reproducible information beyond width-one attribution and temporal
features.

## Reproduction, serialization, and resume

All selected pass-one node attributions, contributions, and activations
reproduce exactly in pass two. Internal node activations and internal edge
weights are exactly invariant across the pass-two candidate axis.

Of 2,757,584 selected edge entries, 2,753,959 reproduce exactly. The remaining
3,625 entries (0.13%) occur across six cases. Their maximum absolute
attribution difference is `0.0064655` and maximum absolute weight difference
is `0.84375`. This matches the C0 topology-dependent bfloat16 batching
sensitivity: pass one computes over each candidate's selected set, whereas
pass two uses the common union. Pass-two values remain the canonical
common-topology measurements; pass-one values and membership masks remain
available for audit.

Resume job `14361225` reran completed case `c1-001`. It exited zero in 33
seconds with `skipped_complete`, reproduced the exact artifact identity and
topology hash, did not load the model, and did not rewrite the artifact.

## Decision

C1 passes the policy/resource gate for the locked two-pass contract:

1. `model_top5_plus_observed`;
2. independent specified-token pass-one traces;
3. exact node and edge union without induced topology;
4. candidate-specific fixed-union node and edge rescoring;
5. one dense artifact retaining applicability and pass-one membership.

This authorizes freezing and explicitly reviewing a C2 scientific-utility
pilot. It does not authorize a full matched corpus. The C2 cohort, feature
contract, downstream weighting, resource plan, and launch still require their
own versioned review.
