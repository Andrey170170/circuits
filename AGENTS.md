# Repository guide

## Purpose

This repository is a fork of ADAG, a raw-MLP-neuron attribution and circuit-tracing library for
Llama- and Qwen-family models. The upstream pipeline traces selected logits, clusters neurons from
input-attribution and output-contribution profiles, labels the resulting groups, and supports
visualization and steering.

The active research project is an exploratory BonaFide faithfulness feasibility study using
`Qwen/Qwen3-4B-Instruct-2507`. We first save independent, compact traces for selected response
tokens, then ask whether downstream ADAG clustering and labeling expose recognizable
source-, hint-, or bottleneck-like structure. This is not yet a general text-atom/model-atom
comparator or a validated faithfulness detector.

Read these before changing the experiment:

- `docs/ADAG_BONAFIDE_NAIVE_PILOT.md` defines the scientific question and claim boundaries.
- `docs/TRACING_PERFORMANCE_BENCHMARK.md` records runtime/resource measurements.
- `docs/TRACING_CORPUS_PLAN.md` records prompt/target selection and the frozen execution plan.

## Scientific contract

- Treat an ADAG graph as a pruned, locally approximate attribution subgraph for selected logits,
  not a complete transcript of model computation.
- Treat clusters and generated labels as exploratory evidence. Stronger causal claims require
  controlled interventions and matched baselines.
- Preserve the identity and provenance of every target trace. Do not merge response graphs in the
  tracing pipeline; aggregation and clustering are downstream stages.
- Keep prompt, response position, model/tokenizer revision, trace configuration, code revision,
  and artifact hashes attached to results.
- Never silently alter a frozen manifest, execution plan, model revision, tokenizer input, or trace
  config. Regenerate a new version and document the change.
- Do not mix Qwen Instruct and Thinking artifacts or neuron identities.

## CHPC environment and storage

Use Python 3.12 and the locked `uv` environment:

```bash
module load python/3.12.12 uv/0.11.14
source scripts/chpc_env.sh
uv sync --frozen
```

After sourcing `scripts/chpc_env.sh`, invoke Python through
`$UV_PROJECT_ENVIRONMENT/bin/python` or `uv run`. The script loads the untracked `.env`, reuses the
scratch-backed Hugging Face and `uv` caches, and places the project environment and working results
under `/scratch/general/vast/$USER/circuits`. VAST scratch is subject to CHPC's 60-day inactivity
purge; copy important completed outputs to group storage or Pando for long-term retention.
Reproducible jobs use the already cached model with Hugging Face/Transformers offline mode; do not
download or upgrade weights inside a run.

Large or high-I/O artifacts belong in `$CIRCUITS_RESULTS_DIR`, not the repository or home-backed
`results/`. Allocation-local scratch is temporary and must not hold the only copy of a result.
Scheduler stdout/stderr and GPU-monitor output belong in `logs/`; generated log files stay ignored.

## Running work

- Check `hostname`, `SLURM_JOB_ID`, `mychpc batch`, and the relevant repo status before launching.
- Keep login-node work to inspection, editing, and small tests. Run model/GPU experiments through
  the project `sbatch` launchers directly; do not wrap reproducible jobs in an interactive helper.
- `scripts/bonafide/benchmark_tracing.sbatch` supports `EXECUTION_MODE=trace` for compact
  single-target graph artifacts and `EXECUTION_MODE=probe` for graph-free workload estimation.
  Probe output is selection evidence, not a scientific trace artifact.
- `scripts/bonafide/final_trace_array.sbatch` executes the frozen compound plan. Its routine shape is
  `0-11%4`, one A100 and 64 GiB host memory per six-hour task. Validate hashes and use
  `sbatch --test-only` before a changed launch.
- Tasks 12-14 are isolated extremes and require explicit `ALLOW_EXTREMES=1` review one at a time.
  Task 15 also requires `ALLOW_PATHOLOGICAL=1` and does not fit the current partition walltime.
- Never edit the executable tracing tree, change `HEAD`, or relocate files being written while a
  provenance-bound array is active. Later shards must see the same cohort identity as earlier ones.
- Fail closed on provenance drift, model-configuration leakage, corrupt resume artifacts, OOM,
  resource gates, ordinary task errors, or the Slurm pre-timeout signal.

## Development and Git hygiene

- Run focused tests for touched code first and use `bash -n` for edited shell launchers. Record any
  unrelated full-suite collection failure instead of describing the suite as green.
- Keep changes small and commit cohesive checkpoints after validation. Do not rewrite or discard
  user changes in a dirty worktree.
- Preserve untracked research inputs such as `BonaFide.csv` and `papers/` unless the user explicitly
  asks to version or remove them.
- Never commit `.env`, credentials, caches, model weights, generated traces, scheduler logs, or GPU
  monitor logs.
- The intended `origin` is `https://github.com/Andrey170170/circuits.git`; verify before pushing.
