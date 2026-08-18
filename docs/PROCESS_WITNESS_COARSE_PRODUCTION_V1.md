# Coarse proposal-bank production v1 runbook

Status: **v6 frozen offline; shard-005-only primary calibration authorized, but blocked on the
active API-tier queue limit and durable archival**.

This campaign produces graph-blind coarse proposals for trace-sampling enrichment. Fine and broad
votes are not semantic truth, adequacy labels, motif labels, correctness judgments, or evidence of
internal computation.

## Authoritative frozen artifact

Bundle root:

`/scratch/general/vast/u1653998/circuits/results/process_witness/coarse_annotation/process-witness-coarse-openai-production-v1/full-corpus-proposal-bank-v6`

The authoritative manifest self-hash is
`3b67f89e14ef703d3222fc4fdfebfd6b60336571d5cdee1164dd1a91fe78ab1c`; the manifest file SHA-256
is `be80325b3daf05c3390fdbf9b47cd0f58951019b7914156b58372c608464fb89`. The bundle binds commit
`6133b5624844b6d0636215b306c6edb4082d64fa`, tree
`09ccda553036472f3fabae22272817c85597352e`, the v9 workstation SHA-256
`95a5627768b2e5f05920aaab91f6e1cd9c00688560f9d821ffc2d8d69c7ceeea`, the protocol and price
snapshots, the locked Python/OpenAI environment inputs, every unit/window/request, and every Batch
shard. The strict loader reconstructed the full topology and verified read-only `0555` directories
and `0444` files after freeze.

Exact census:

- 188 responses and 94,546 atomic units;
- 74,860 provider-pending semantic atoms, 19,500 deterministic surface/control atoms, and 186
  deterministic terminal atoms;
- 12,557 response-local windows of at most six targets;
- 37,671 physical requests: three body-identical replicas per window;
- six response-affinity shards totaling 1,002,935,940 bytes. Shard sizes are 179,922,075,
  179,684,685, 179,983,719, 179,999,574, 179,808,231, and 103,537,656 bytes. Every shard is
  strictly below the internal 180,000,000-byte guard.

The cost-plan self-hash is
`d7e00efe88c49d65321a3272e52ab6727434eaa4c046119cc2656a13ce2cb5b2`; its file SHA-256 is
`7bbb72b687eaf7996a5e2ba9575432fd7f7b401a6d9eca26839d1b6ebc339d42`. Cost views are deliberately
separate:

- direct extrapolation from the completed v4 run: $18.740845334;
- one-write/two-read cache-pattern planning forecast: $32.10, without assuming cache behavior;
- strict primary no-cache/full-16,384-output exposure: $514.45756665;
- empirical physical-input forecast: 276,968,462 tokens.

The first two values are forecasts, not hard caps. The strict exposure is an upper exposure for the
frozen primary request universe, not advance authorization for a recovery wave.

V1 through v5 are superseded and must not be launched. V1/v2 failed lifecycle review; v3 lacked the
calibration-only primary-authorization scope; v4 carried misleading recovery-preauthorization
wording; and v5 corrected that wording but allowed an operational config change to rename every
request. V6 separately binds the full operational config and a validated stable v1 request-identity
namespace. Its six Batch files are byte-identical to v3, including every `custom_id`, request body,
and shard hash. All earlier versions remain immutable provenance only.

## Storage gate

VAST scratch is subject to CHPC's 60-day inactivity purge. V6 is currently the only recorded
authoritative copy and no durable copy was made during preparation. Before relying on the bundle or
any eventual result for scientific analysis, make a verified content-hash-preserving copy to group
storage or Pando and record its location. Do not treat read-only permissions on VAST as archival.

## Primary launch gates

The user authorized only the `shard-005` primary calibration wave:

- exact primary forecast budget: $3.207300838;
- acknowledged strict primary exposure: $79.53101355;
- primary-wave actual spend must remain strictly below $100;
- maximum concurrency: one.

This does **not** authorize any other primary shard, the $514.45756665 full campaign, recovery, or
full-corpus finalization. Recovery remains unavailable unless failures are first frozen from the
authorized shard and the user later supplies a fresh recovery-specific forecast and exact strict
exposure acknowledgement.

Do not initialize, upload, or submit until the active API tier's Batch queued-input-token limit is
recorded and is at least the frozen `shard-005` forecast of 28,514,690 tokens. Initialization then
records all of the following:

1. the exact `shard-005`-only authorization and notes above;
2. the active OpenAI API tier's Batch queued-input-token limit; and
3. maximum concurrency one.

The forecast budget stops later submissions when known cost plus conservative reservations exceeds
it; it cannot stop an already submitted Batch and is not advertised as a hard maximum. Finalization
separately refuses primary actual cost above the acknowledged primary strict exposure.

`shard-005` is the proposed calibration shard. It has the smallest physical file, an empirical
forecast cost of $3.207300838, and a strict primary exposure of $79.53101355. It contains all three
response-source lanes, but it is not assumed to predict prompt-cache behavior for the five larger
shards. Collect its receipt-derived per-request usage before deciding whether to authorize further
submissions.

Network-free initialization after replacing `ACTIVE_TIER_LIMIT_AT_LEAST_28514690` with the verified
integer limit:

```bash
module load python/3.12.12 uv/0.11.14
source scripts/chpc_env.sh
$UV_PROJECT_ENVIRONMENT/bin/python \
  scripts/bonafide/process_witness_coarse_openai_batch_production_v1.py initialize \
  --bundle-root /scratch/general/vast/u1653998/circuits/results/process_witness/coarse_annotation/process-witness-coarse-openai-production-v1/full-corpus-proposal-bank-v6 \
  --run-root RUN_ROOT \
  --authorized-primary-shard-id shard-005 \
  --forecast-budget-usd 3.207300838 \
  --forecast-budget-authorization-note 'AUTHORIZED PRIMARY CALIBRATION FORECAST FOR SHARD-005 ONLY: USD 3.207300838' \
  --acknowledged-strict-worst-case-exposure-usd 79.53101355 \
  --strict-exposure-acknowledgement-note 'ACKNOWLEDGED SHARD-005 PRIMARY STRICT EXPOSURE: USD 79.53101355; FULL CAMPAIGN NOT AUTHORIZED' \
  --primary-actual-spend-limit-usd 100 \
  --provider-queued-input-token-limit ACTIVE_TIER_LIMIT_AT_LEAST_28514690 \
  --maximum-concurrent-shards 1
```

Do not run the command with the placeholder. Initialization makes no provider call. It copies only
`shard-005`, freezes the non-expandable primary subset, and binds the exact Python/OpenAI SDK
versions and optional
`OPENAI_PROJECT_ID`/`OPENAI_ORG_ID` hashes; every later campaign command rechecks those values,
tracked source files, lockfiles, bundle identity, and copied shard inputs.

Provider-mutating submission is a separate explicit command:

```bash
$UV_PROJECT_ENVIRONMENT/bin/python \
  scripts/bonafide/process_witness_coarse_openai_batch_production_v1.py submit-shard \
  --run-root RUN_ROOT --shard-id shard-005
```

Use `status-shard` and `collect-shard` for receipt-bound observation and collection. Submission and
collection use atomic campaign/shard gates. An upload receipt is persisted before Batch creation.
If a provider call becomes ambiguous, do not resubmit blindly: use `recover-shard-submission`.
Recovery reconciles exactly one metadata-matched Batch, or proves zero matching Batches before a
safe re-upload after an unknown/orphan upload state. Collection is resumable and retains raw
provider snapshots, output/error JSONL, and receipt chains.

Every primary mutation command refuses nonauthorized shard IDs. Collection fails closed if actual
primary cost exceeds the acknowledged $79.53101355 strict exposure or reaches the separate $100
primary-wave limit. No provider call, upload, initialization, or submission has occurred for v6.

## Failed-only recovery authorization

After the authorized primary shard is terminal and collected, `prepare-recovery` may freeze at most
one failed-only recovery wave. It excludes successful requests and independently revalidates every
failed request against the frozen `shard-005` body, path, order, cost, strict exposure, and queue
binding. Primary authorization does **not** authorize this wave.

Inspect the generated recovery manifest, then obtain a fresh recovery-specific forecast budget and
acknowledgement of its exact `strict_no_cache_full_output_exposure_usd`. Record them before any
recovery upload:

```bash
$UV_PROJECT_ENVIRONMENT/bin/python \
  scripts/bonafide/process_witness_coarse_openai_batch_production_v1.py authorize-recovery \
  --run-root RUN_ROOT \
  --recovery-forecast-budget-usd AUTHORIZED_RECOVERY_FORECAST_USD \
  --forecast-budget-authorization-note 'EXACT USER RECOVERY FORECAST AUTHORIZATION' \
  --acknowledged-strict-worst-case-exposure-usd EXACT_FROZEN_RECOVERY_EXPOSURE_USD \
  --strict-exposure-acknowledgement-note 'EXACT USER RECOVERY STRICT-EXPOSURE ACKNOWLEDGEMENT'
```

Only then use `submit-recovery-shard`, `status-recovery-shard`,
`collect-recovery-shard`, or `recover-recovery-submission`. Recovery authorization is validated on
both ordinary submission and reconciliation. A second recovery wave is forbidden.

## Finalization

`finalize` requires exactly one successful effective result for every frozen physical request,
complete per-request or defensibly reconciled aggregate pricing, separately bounded primary and
recovery actual costs, and three votes for every provider-pending atom. Completed/partial missing
rows remain cost-incomplete; zero pricing is allowed only for provider-evidenced pre-execution whole
Batch failure.

A shard-subset calibration run is explicitly forbidden from invoking full-corpus `finalize`.

The final artifact is built in a temporary tree, semantically validated before publication, then
made read-only and validated again. It contains the exact campaign bundle and request bodies,
campaign/recovery authorization, upload/create/submission/status/collection receipts, raw provider
snapshots and JSONL, effective events, proposals, sampling groups, and a hash-bound inventory.
Traces remain per target; grouping never merges trace artifacts or alters atomic annotations.
