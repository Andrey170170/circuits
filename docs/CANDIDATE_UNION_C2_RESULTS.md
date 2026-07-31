# Candidate-union C2 results

The two-pass C2 scientific-utility run completed on 2026-07-31. All 245 selected
discovery targets completed both passes and passed artifact, checksum, topology,
candidate-order, provenance, applicability, and finite-value validation.

## Frozen cohort and measurement scale

The cohort contains seven temporal-bin targets from each of 35 discovery
responses spanning all 34 base-question families. Confirmatory holdout targets
and isolated source-workload extremes were excluded by the frozen C2 protocol.
The realized candidate widths are 235 width-five and 10 width-six targets.

The assembled artifacts contain:

- 145,614 exact-union node rows and 7,613,281 exact-union edge rows;
- 728,745 applicable node-candidate measurements;
- 38,264,204 applicable edge-candidate measurements; and
- 1,235 independently selected pass-one traces and 1,235 fixed-union
  refinements.

The pass-one output root is
`/scratch/general/vast/u1653998/circuits/results/bonafide/downstream/top5-c2-v1`.
The refinement and assembled-union root is
`/scratch/general/vast/u1653998/circuits/results/bonafide/downstream/candidate-union-c2-v1`.
The audited analysis report is `analysis/c2-scientific-utility-v1.json` under
that union root. Its SHA-256 is
`9ea1123685e73bc45f8c93490429a2a309ed62953406d61509d0014730ef6530`;
the analysis module SHA-256 recorded inside it is
`d1875f02c4c79ee66ec90e9a085411ffc5fbfa5b0b95bc264daf64ff23359031`.

## Primary trajectory result

The primary task retrieves each target's true next-bin continuation from the
other discovery responses. Same-family alternative responses are excluded.
Features use only MLP neurons, reduce repeated signed bases by signed sum, and
compare targets by cosine similarity on at least 16 shared non-boundary signed
bases. The candidate view is the flattened five-channel profile

```text
contribution(model rank r) - contribution(observed token), r = 1..5.
```

Equal-family, then equal-response, then equal-target weighting gives:

| View | MRR | Top one | Weighted scored-anchor coverage |
| --- | ---: | ---: | ---: |
| Width-one attribution | 0.589798 | 0.431609 | 0.384804 |
| Candidate contrasts | 0.176885 | 0.068137 | 0.933824 |
| Equal-weight multiview | 0.205333 | 0.081863 | 0.933824 |

The frozen primary contrast is therefore
`0.205333 - 0.589798 = -0.384465`, not the required improvement of at
least `+0.03`. Its deterministic 10,000-replicate family-block bootstrap
interval is `[-0.488618, -0.288354]`. All 34 leave-one-family-out estimates
remain negative. The point-estimate, interval, and LOFO requirements therefore
all fail.

The width-one and multiview coverage differ substantially. Two sensitivity
analyses reach the same decision:

- assigning zero reciprocal rank to missing true scores gives a difference of
  `-0.024778`; and
- restricting both views to the 81 commonly scored anchors gives a
  family-weighted difference of `-0.261241`, with a 95% family-block bootstrap
  interval of `[-0.379702, -0.145639]` and zero positive leave-one-family-out
  estimates among 29 represented families.

The candidate view is measurable on many more anchor pools than width one, but
its directional information does not identify the true response trajectory in
this task. Higher score coverage is not sufficient to establish utility.

## Non-degeneracy

Using entropy effective rank over normalized singular values of each target's
bases-by-five contrast matrix:

- median effective rank is `3.206573`;
- 243/245 targets (`99.184%`) have effective rank at least `1.5`; and
- the median fraction of signed bases with both positive and negative raw
  candidate contributions is `0.392593` (minimum `0.070896`).

This convention passes the frozen non-degeneracy thresholds. A squared-singular-
value entropy sensitivity gives median `2.003666`, but only 188/245 targets
(`76.735%`) reach `1.5`. The protocol did not specify which entropy convention
defines effective rank. That omission should be resolved before reusing such a
gate, although it cannot change the C2 decision because the utility gate fails
strongly under either convention.

The protocol also did not fully freeze the bootstrap seed and replicate count,
tie handling, missing-score estimator, or whether resampling changes only
estimator blocks or also the retrieval pool. The primary point estimate and the
reported sensitivity analyses are negative enough that these omissions do not
create a plausible pass interpretation.

## Runtime and counterfactual matched-corpus estimate

C2 consumed approximately 23.8 A100-hours in pass one and 20.9 A100-hours in
pass two when measured as summed per-case unit time. Summed scheduler allocation
was approximately 46.2 A100-hours. Scaling these measured costs to the 2,594
successfully completed width-one positions gives a counterfactual estimate of:

| Stage | Estimated A100-hours | Four-GPU queue-free wall time |
| --- | ---: | ---: |
| Independent candidate traces | 252--256 | 2.7--3.0 days |
| Exact-union refinement | 221--233 | 2.4--2.7 days |
| Total | about 473 compute / 489 allocated | 5.1--5.6 days |

A response-block bootstrap and measured overhead support a planning envelope of
roughly 450--550 A100-hours. At the usual four-GPU cap, this means about six
execution days or seven calendar days with modest queue and retry allowance.
The measured-equivalent persistent storage is about 25 GiB; reserving 35 GiB
persistent and 50 GiB free during execution would be prudent.

Using C2's 4.08% width-six rate, that run would produce approximately 13,076
independent candidate traces, the same number of fixed-union refinements, and
2,594 assembled union artifacts. Queue-free total wall time would be about 20
days on one GPU, 10--11 days on two, 5--6 days on four, or 2.6--3 days on eight.

This estimate excludes the still-unrun pathological position with 81.5 million
early candidate edges. The matched 2,594-position set includes three completed
source-workload extremes that C2 deliberately excluded. They would require
isolated pass-one and pass-two handling, so the estimate is not an authorization
or a guarantee for those cases.

## Decision

C2 fails the frozen scientific-utility gate. The two-pass artifacts remain
useful bounded evidence about candidate-specific contribution profiles, but the
candidate view did not add the predeclared reproducible trajectory information
beyond width-one attribution. Under the frozen execution plan, do not promote
this design to the full 2,594-position matched top-five corpus.
