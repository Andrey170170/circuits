# Candidate-union C2 post-hoc salvage results

The post-hoc C2 salvage analysis completed on 2026-07-31 from clean commit
`de256933a20cf493726e77dd91ad77e94944af2c`. The V2 protocol was frozen before
execution in commit `f2eae40` after implementation review corrected the rescue
denominator and exact-tie rule. The confirmatory holdout was not touched.

This analysis cannot retroactively pass C2 or authorize the 2,594-position
matched run. It asks only whether the completed candidate traces retain a
narrower exploratory use.

## Audited inputs and output

The analysis revalidated the same 245 C2 width-one and assembled-union
artifacts and constructed all 7,338 planned next-bin anchor-target pairs. It
used 100,000 deterministic whole-response trajectory permutations, Holm
correction over three endpoints, and 10,000 family-block bootstrap replicates.

The report is
`/scratch/general/vast/u1653998/circuits/results/bonafide/downstream/candidate-union-c2-v1/analysis/c2-salvage-v2.json`
with SHA-256
`3ce0f45c1d05e97310481ec4bc42462b09c7490108d9e921a3490fa3620fa0b3`.
It binds:

- audited C2 report SHA-256
  `9ea1123685e73bc45f8c93490429a2a309ed62953406d61509d0014730ef6530`;
- V2 protocol SHA-256
  `f8eb538f956ac8f3ce69b8bbe98bf46e96ab7435a47bf7b3644aaef25a27bd49`;
- analysis-module SHA-256
  `c541213622bb8baa7f56f66d119ac61aeffca657def8357923f31d6fb73bc7e1`;
  and
- pair-table SHA-256
  `e01c0253ec0ec2898f49833fab8cdcf3fb9f31190027466d72d081b5deb87d2e`.

## Primary exploratory results

All observed and null MRRs below use the same fixed equal-family,
equal-response, equal-anchor weights. Missing true pairs receive reciprocal
rank zero.

| Endpoint | Observed | Null mean | Lift | Holm p | Bootstrap 95% | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Candidate standalone MRR | 0.165588 | 0.106691 | +0.058898 | 0.000120 | [0.022823, 0.099699] | Promising |
| Width-one-missing rescue contribution | 0.070848 | 0.071363 | -0.000515 | 1.000000 | [-0.014138, 0.014372] | Fail |
| Calibrated backoff minus width one | -0.012515 | 0.055082 | -0.067596 | 1.000000 | [-0.101859, -0.036555] | Fail |

Candidate standalone is the only endpoint that passes the frozen exploratory
criteria. Its unadjusted one-sided permutation p-value is `0.000040`; all 34/34
leave-one-family-out lifts remain positive.

## What the positive result means

Candidate profiles score the true continuation for 196/210 anchors, giving
93.38% weighted coverage. Their conditional scored-anchor MRR is `0.176885`;
their all-anchor zero-filled MRR is `0.165588`. Width one remains more accurate
where it is available: 38.48% coverage, `0.589798` conditional MRR, and
`0.218199` zero-filled MRR.

The above-null candidate-profile MRR appears descriptively dominated by
support applicability rather than fine directional ranking:

- a random allowed next-bin pair has 69.19% weighted candidate-score validity;
- the actual true continuation has 93.38% validity;
- among anchors with a valid true candidate score, uniform-rank chance MRR is
  `0.165685`, versus observed `0.176885`.

Support alone and the within-support directional increment were not separately
corrected inferential endpoints. The combined candidate-profile view therefore
provides a stable exploratory high-coverage response-trajectory signature in
this discovery cohort, but it is not evidence that the candidate view is a
better general retriever than width one.

## What did not salvage

There are 129 anchors whose true width-one pair is invalid. Candidate profiles
cover 115 of them, but their conditional MRR is only `0.133874`, and the fixed-
denominator rescue contribution is indistinguishable from its permutation
null. It therefore did not demonstrate above-null recovery on the cases width
one misses in this cohort.

The label-free percentile-calibrated backoff also fails. Its zero-filled MRR is
`0.205684`, below width one's `0.218199`; every LOFO effect relative to the null
is negative.

Two prespecified descriptive diagnostics are interesting but not inferential:

- raw-cosine backoff reaches `0.279219` zero-filled MRR, `+0.061020` over width
  one, but the protocol deliberately did not test this uncalibrated sensitivity
  against a null; and
- the individual model-rank-five channel reaches `0.200116` zero-filled MRR,
  followed by rank four at `0.187658` and rank three at `0.184096`. Channel
  selection was not a primary corrected test.

These diagnostics may define hypotheses for a new holdout protocol; they are
not positive findings from this cohort.

## Decision

The completed top-five traces are scientifically useful as a bounded,
high-coverage candidate-profile signature and as material for descriptive
alternative-token contribution analysis. They did not demonstrate above-null
missing-width-one rescue or a safe automatic fusion layer, and the failed C2
promotion decision remains unchanged.

Before tracing all 2,594 positions, the next defensible step is a small newly
frozen holdout replication of candidate standalone support-aware retrieval. A
raw-cosine backoff hypothesis may be included prospectively as a separate
secondary test. Rank-channel findings should remain diagnostic unless a new
protocol explicitly corrects for their selection.
