# Contribution-aware same-position candidate tracing

Status: engineering implementation checkpoint. CPU fixtures and compatibility tests pass. No
Qwen GPU trace, observed-token parity cohort, C0 candidate-reference cohort, C1/C2 probe, or
matched corpus is authorized or recorded by this document.

The scientific and launch contract remains Section 10 of
`plans/2026-07-26-adag-bonafide-downstream-execution-plan.md`.

## Trace semantics

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

The GPU launcher is `scripts/bonafide/topk_tracing.sbatch`. It requires an absolute top-k
manifest and a lane-specific absolute `UV_PROJECT_ENVIRONMENT`. It is limited to an explicitly
reviewed parity or C0 wave; it does not authorize C1, C2, or a full corpus.

## Gate workflow

1. Freeze an observed-token `k=1` manifest over a representative discovery-only set.
2. Dry-run the manifest and launcher inputs.
3. Run and save the new `k=1` artifacts.
4. Compare them with the exact frozen width-one artifacts using
   `scripts.bonafide.topk_parity`.
5. Stop on any unexplained structural, provenance, numerical, contribution, or instrumentation
   mismatch.
6. Freeze C0 joint raw/contrastive manifests and fixed-candidate `k=1` reference manifests for
   8--12 discovery targets.
7. Build the C0 topology report with `scripts.bonafide.topk_c0_compare`.
8. Review per-candidate/union node and edge recall, retained absolute node-attribution mass,
   source-to-logit path recall, exact omitted-path witnesses, candidate effective rank, sign
   conflicts, numerical health, and resources.

C1 and C2 remain blocked until the applicable earlier gates pass. A matched 2,594-position corpus
remains blocked until all Section 10.7 gates and explicit Slurm review pass.
