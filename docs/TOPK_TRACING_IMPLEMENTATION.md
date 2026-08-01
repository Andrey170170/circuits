# Contribution-aware same-position candidate tracing

Status: C0, C1, and C2 are complete and production topology semantics are frozen. CPU fixtures,
observed-token parity, candidate-policy smokes, the ten-target C0 comparison, the two-pass C0
rerun, and the 32-target C1 policy/resource gate pass. C2 fails its frozen scientific-utility
gate, so a matched top-five corpus remains unauthorized.

The scientific and launch contract remains Section 10 of
`plans/2026-07-26-adag-bonafide-downstream-execution-plan.md`.

## Locked trace semantics

The production candidate family is no longer a scalar candidate-joint graph. C0 established the
following target-local two-pass contract:

```text
one example and response position
        |
model_top5_plus_observed (realized width 5 or 6)
        |
one independent specified-token k=1 trace per candidate
        |
exact union of independently selected nodes and exact edges
        |
candidate-specific fixed-union node and edge rescoring
        |
dense candidate-union artifact with applicability and selection masks
```

No induced all-pairs edge is added merely because two endpoints appear in the node union.
Internal and embedding-to-MLP union edges are applicable to every candidate. A terminal logit edge
is applicable only to the candidate whose logit it targets. Measured zero and inapplicable null
remain distinct.

The original joint top-k schema remains readable and useful for the completed C0 raw-sum and
contrastive diagnostics. It is not the production C1/C2 topology family.

## Legacy joint-trace semantics

A top-k artifact represents one teacher-forced response target and one shared prediction
position. Candidate logits form a separate output-contribution axis:

```text
one example
one response position
one prediction position
one frozen candidate policy
one named joint topology objective
N raw candidate-logit contribution columns
```

It is not a multi-response-position trace and is not five independently selected graphs.
Scalar node/edge attribution fields use the named joint objective. `contrib_map` retains one raw
gradient-times-activation value per candidate.

Implemented policies are:

- `observed_token`: the explicit observed-token `k=1` parity mode;
- `specified_token`: one frozen token ID for C0 independent-candidate references;
- `model_top5_plus_observed`: the frozen C1 policy; observed token at candidate index zero plus
  every distinct model-top-five token, realizing width five or six;
- `observed_plus_top4_alternatives`: observed token at candidate index zero plus four
  deterministic alternatives, retained for C0 smoke compatibility;
- `model_top5`: five deterministic highest-logit candidates.

Full-distribution rank is one-based. Candidate-vector index is zero-based. Descending logit is
ordered with ascending token ID as the exact-tie rule. The variable-width policy freezes
`candidate_count_min=5`, `candidate_count_max=6`, and
`candidate_count_rule=5_if_observed_in_model_top5_else_6`; each artifact records its realized
width and never duplicates the observed token.

Implemented objectives are:

- `raw_logit_sum`: unit weight for every candidate, preserving legacy `k=1` semantics;
- `observed_vs_alternatives`: observed logit minus the mean alternative logit.

The frozen tracing config uses a percentage-of-goal neuron threshold. Raw objectives preserve the
legacy signed-goal threshold reference for parity. The contrastive objective records and uses the
absolute joint-objective magnitude so a negative contrast cannot turn the absolute-attribution
threshold negative.

## Artifact and execution boundary

Top-k artifacts use `adag.compact-trace.topk-position.v1`. They have a separate payload type,
validator, saver, loader, checksum, and resume path. The legacy
`adag.compact-trace.v1` loader rejects them.

`scripts.bonafide.topk_runner`:

- validates one immutable policy/objective family per manifest;
- verifies every work item against the hashed width-one source manifest;
- binds source, top-k manifest, model/config, code, environment, and warm-up policy into runtime
  identity;
- loads one resident model per wave;
- saves one atomic artifact per response target;
- validates checksums before resume;
- fails after ordinary errors, OOM, model-configuration leakage, resource gates, or the Slurm
  pre-timeout signal;
- writes candidate norms, effective rank, sign counts, runtime, HBM, RSS, graph size, and
  instrumentation diagnostics.

The legacy joint/reference GPU launcher is `scripts/bonafide/topk_tracing.sbatch`. It requires an
absolute top-k manifest and a lane-specific absolute `UV_PROJECT_ENVIRONMENT`. It is limited to
an explicitly reviewed parity or C0 wave; it does not authorize C1, C2, or a full corpus.

The C0 fixed-union launcher is `scripts/bonafide/candidate_union_refinement.sbatch`. The
corresponding C1-only launchers are `scripts/bonafide/topk_c1_tracing.sbatch` and
`scripts/bonafide/candidate_union_c1_refinement.sbatch`. Their runners checksum-validate the
independently saved references, freeze exact node and edge unions, apply no pruning thresholds
during measurement, preserve zero-valued measurements, save each candidate rescore independently
for resume, and atomically assemble the final union artifact.

`scripts/bonafide/topk_rank_screen.sbatch` is a discovery-only selection-evidence launcher. It
loads the frozen model once, measures observed-token rank for low-probability discovery targets,
and records whether the union policy would realize width five or six. It does not construct
graphs or save scientific trace artifacts. C0/C1 manifests should use its measured ranks rather
than infer rank from the stored observed-token probability.

## GPU policy smoke evidence

The frozen `model_top5_plus_observed` policy was checked on an A100 80 GB:

- job `14314883` completed two raw-sum traces at realized width five (observed ranks one and two);
- rank-screen job `14314916` measured 32 low-probability discovery targets and found 17
  width-five and 15 width-six cases;
- job `14314944` completed the deliberately selected width-six target
  `trace-source-c6a2b2d04df3ec93a97b764d` at observed rank eight.

The width-six artifact is `topk-trace-8b79cc86b8b9fd8975793954`, bound to code revision
`0bf7f71f678051b89daf9e625b2da9ef3ce93fbb`. It contains 409 candidate-profile rows with matrix
rank six, 409 graph nodes, 5,902 graph edges, 33.05 seconds trace wall time, 16.26 GiB peak
reserved HBM, and 62.99 GiB headroom. Its compact payload SHA-256 is
`8b06a1ab8ba9093384e175f1fd0219e7a2bfee12ae04b5e4f73ba6787ab5321b`.

These checks established executable width-five/width-six policy behavior and initial resource
feasibility. C0 subsequently completed 55 independent reference traces and 55 fixed-union
rescoring traces over ten targets. All ten union artifacts passed integrity and topology
validation. See `docs/CANDIDATE_UNION_C0_RESULTS.md`.

C0 recovered 5,208 previously missing node-candidate measurements and 880,789 previously missing
edge-candidate measurements. The result locks candidate-specific union refinement as the C1/C2
approach. It authorizes planning and explicit review of C1; it does not authorize C2 or a matched
corpus.

C1 completed 175 independent traces and 175 fixed-union rescoring traces over 32 balanced
discovery targets. All 32 dense union artifacts passed integrity, topology, numerical, resource,
serialization, and resume checks. C1 recovered 16,882 node-candidate and 3,361,742 edge-candidate
measurements absent from the corresponding independent graphs. Every raw contribution matrix had
full candidate rank and every centered matrix had the maximum possible contrastive rank. See
`docs/CANDIDATE_UNION_C1_RESULTS.md`.

## Gate workflow

1. Freeze an observed-token `k=1` manifest over a representative discovery-only set.
2. Dry-run the manifest and launcher inputs.
3. Run and save the new `k=1` artifacts.
4. Compare them with the exact frozen width-one artifacts using
   `scripts.bonafide.topk_parity`.
5. Stop on any unexplained structural, provenance, numerical, contribution, or instrumentation
   mismatch.
6. Freeze and run C0 joint raw/contrastive manifests plus fixed-candidate `k=1` references for ten
   discovery targets. Complete.
7. Compare joint graphs with the independent candidate union. Complete; joint objectives were too
   lossy to become the primary family.
8. Run exact-union fixed-topology node and edge refinement and audit the result. Complete.
9. Freeze a 24--48-target, family/response-balanced C1 resource cohort using the same candidate
   policy and two-pass contract. Complete: 32 targets.
10. Measure total and per-candidate runtime, HBM, RSS, graph/union size, storage, numerical health,
    observed-token rank, and realized width. Complete.
11. Freeze and explicitly review the C2 scientific-utility cohort, feature contract, weighting,
    resource plan, and launch before execution. Complete.
12. Run C2 and apply the frozen non-degeneracy and trajectory-utility gates. Complete: the
    candidate profiles are non-degenerate under the primary effective-rank convention, but the
    multiview next-bin MRR is lower than width one and the utility gate fails. See
    `docs/CANDIDATE_UNION_C2_RESULTS.md`.

The matched 2,594-position corpus remains blocked. C2 does not provide the required scientific
utility evidence, and no full-corpus manifest or Slurm launch is authorized.

The frozen post-hoc salvage analysis subsequently found above-null candidate-only trajectory
signal. Descriptively, support applicability appears to contribute most of the lift, but it was not
a separate corrected endpoint. The analysis did not demonstrate above-null width-one-missing
rescue or useful percentile-calibrated backoff. The result supports bounded exploratory reuse and
a possible small holdout replication, not full-corpus promotion. See
`docs/CANDIDATE_UNION_C2_SALVAGE_RESULTS.md`.
