# Coarse proposal-bank production v1 runbook

Status: **frozen offline; blocked before provider launch**.

This campaign produces graph-blind coarse proposals for trace-sampling enrichment. Fine and broad
votes are not semantic truth, adequacy labels, motif labels, correctness judgments, or evidence of
internal computation.

## Frozen artifact

Bundle root:

`/scratch/general/vast/u1653998/circuits/results/process_witness/coarse_annotation/process-witness-coarse-openai-production-v1/full-corpus-proposal-bank-v1`

Manifest self-hash: `e1fb95057a2ac0eb7a67b4dd6bd96047c3bf9c6061796fd17c7ae166ddfd75c9`.
The bundle binds code commit `e2c0899a5232f19ff5198fd165c7d78ff3baf669`, the v9
workstation SHA-256 `95a5627768b2e5f05920aaab91f6e1cd9c00688560f9d821ffc2d8d69c7ceeea`,
the protocol config, the price snapshot, every unit/window/request, and every Batch shard.
The strict loader reconstructed the full topology after freeze.

Exact census:

- 188 responses and 94,546 atomic units;
- 74,860 provider-pending semantic atoms, 19,500 deterministic surface/control atoms, and 186
  deterministic terminal atoms;
- 12,557 response-local windows of at most six targets;
- 37,671 physical requests: three body-identical replicas per window;
- six response-affinity shards totaling 1,002,935,940 bytes. Shard sizes are 179,922,075,
  179,684,685, 179,983,719, 179,999,574, 179,808,231, and 103,537,656 bytes. Every shard is
  strictly below the internal 180,000,000-byte guard.

Cost views are deliberately separate. Direct extrapolation from the completed v4 run is
$18.740845334. A one-write/two-read cache-pattern planning model is $32.10, but cache behavior is
not assumed. The conservative no-cache/full-16,384-output ceiling is $514.45756665. The empirical
physical-input forecast is 276,968,462 tokens. None of the forecasts is a spend authorization.

## Launch gates

Do not upload or submit until both gates are filled:

1. fresh run-specific spend authorization, including how calibration may stop the remainder; and
2. the active OpenAI API tier's Batch queued-input-token limit.

Initialization records the exact queue limit and a requested maximum shard concurrency. The sum of
the largest concurrently permitted frozen shards must fit that limit. Submission also rechecks
active queued tokens, collected cost, source hashes, and prior submission intent. The default and
safest first wave is one shard at a time.

`shard-005` is the proposed calibration shard: it is the smallest queue load (about 28.5M empirical
input tokens), contains all three response-source lanes, and does not change the already-frozen
request universe. It is not assumed to predict prompt-cache performance for the five larger shards;
reforecast with its receipt-derived usage before authorizing the remainder.

Network-free initialization after fresh authorization:

```bash
module load python/3.12.12 uv/0.11.14
source scripts/chpc_env.sh
$UV_PROJECT_ENVIRONMENT/bin/python \
  scripts/bonafide/process_witness_coarse_openai_batch_production_v1.py initialize \
  --bundle-root /scratch/general/vast/u1653998/circuits/results/process_witness/coarse_annotation/process-witness-coarse-openai-production-v1/full-corpus-proposal-bank-v1 \
  --run-root RUN_ROOT \
  --maximum-authorized-cost-usd AUTHORIZED_USD \
  --authorization-note 'EXACT USER AUTHORIZATION' \
  --provider-queued-input-token-limit ACTIVE_TIER_LIMIT \
  --maximum-concurrent-shards 1
```

Provider-mutating submission is a separate explicit command:

```bash
$UV_PROJECT_ENVIRONMENT/bin/python \
  scripts/bonafide/process_witness_coarse_openai_batch_production_v1.py submit-shard \
  --run-root RUN_ROOT --shard-id shard-005
```

Use `status-shard` and `collect-shard` for receipt-bound observation and collection. A submission
intent exists before upload/create. If provider state becomes ambiguous, automatic retry is
forbidden; use `recover-shard-submission`, which requires exactly one metadata-matched Batch or the
immediate create snapshot.

After all primary shards are collected, `prepare-recovery` creates one failed-only recovery wave,
partitioned under the same byte guard. It excludes every successful request and preserves each
failed provider body byte-for-byte. The corresponding operations are `submit-recovery-shard`,
`status-recovery-shard`, `collect-recovery-shard`, and `recover-recovery-submission`. A second
recovery wave is forbidden.

`finalize` requires exactly one successful effective result for every frozen physical request,
complete receipt pricing within the authorization, and three votes for every provider-pending atom.
It preserves physical votes, projects broad selection families, makes a separate sampling-group
index, copies exact event/collection evidence, and makes the result tree read-only. Traces remain
per target; grouping never merges trace artifacts or alters atomic annotations.
