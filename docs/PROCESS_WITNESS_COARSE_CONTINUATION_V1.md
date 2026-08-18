# Coarse proposal-bank full-corpus continuation v1

Status: **implementation validated offline; production build and provider submission not yet
performed**.

This is an additive continuation of the immutable v6 `shard-005` calibration. It does not alter or
rerun the 6,433 valid calibration requests. It adopts the complete calibration receipt tree,
reconciles its eight `credit_balance_exhausted` rows as zero-usage failures from exact Batch
aggregate/row sums, labels original v6 shards `000`–`004`, and defers one failed-only recovery until
all new primary tranches are collected. The recovery therefore contains the 14 inherited failures
plus any new non-successful primary rows, and no successful row.

These outputs remain graph-blind sampling proposals. They are not semantic truth, trace selection,
an ADAG adequacy result, a motif, a witness, a computation, a faithfulness result, or causal
evidence.

The lifecycle is intentionally additive instead of refactoring production v1: the completed
calibration revalidates hashes of its original v1 sources, so changing those files would invalidate
the evidence being adopted. This module reuses v1 parsing, pricing, provider-validation, and
write-or-verify primitives while defining a separately versioned receipt state machine for the new
attempt topology.

## Frozen operational policy

- Source bundle: `full-corpus-proposal-bank-v6`, manifest self-hash
  `3b67f89e14ef703d3222fc4fdfebfd6b60336571d5cdee1164dd1a91fe78ab1c`.
- Inherited calibration: 6,447 requests, 6,433 successes and 14 non-successes; known complete
  adopted cost `$3.99951985` after zero-usage reconciliation.
- Remaining exact request universe: 31,224 original rows from shards `000`–`004`. Every JSONL line
  and `custom_id` is copied byte-for-byte; original request/shard/body/window/response/replica
  identities are retained.
- Queue: recorded Tier-3 limit 40,000,000 input tokens, internal empirical tranche cap 30,000,000,
  concurrency one. Response affinity is preserved. The calibration aggregate was 28,834,177 input
  tokens versus a 28,514,690 forecast, an actual/forecast ratio of 1.011204295.
- Spend: sticky warning at cumulative actual cost `>= $20`; no new submission when known actual is
  `>= $40` or known actual plus active and candidate calibrated reservations is `> $40`. Equality
  at `$40` is permitted. Because usage is known only after a Batch finishes, one already admitted
  attempt can cross `$40`; its data remain valid, the crossing is recorded, and all later
  submissions are blocked.
- Candidate reservation:

  ```text
  raw = 1.47010105 * body_bytes / 102828486
      + 2.52941880 * request_count / 6439
  scaled_direct = v6_direct_forecast * (3.99951985 / 3.207300838)
  reservation = 1.25 * max(raw, scaled_direct)
  ```

  `$1.47010105` is the calibration's all-input/cache cost and `$2.52941880` is its output cost.
  The original no-cache/full-output strict exposure is retained as theoretical metadata, not used
  as the operational admission reservation.
- Any explicit provider submission rejection (`400`, `403`, `409`, `422`, or `429`) stops this
  campaign version and makes no scientific result; `429` is additionally recorded as a possible
  queue rejection. Any later smaller-layout amendment must be separately versioned, reviewed, and
  adopt every valid result already collected; that amendment is outside this implementation. An
  ambiguous upload/create state is reconciled through bound provider evidence, never blind
  resubmission.

The final committed builder must be run once offline before production. Record its exact tranche
count, maximum queue reservation, maximum calibrated cost reservation, aggregate direct forecast,
and aggregate calibrated reservation from the resulting manifest. These are planning observations,
not provider submissions or actual full-run costs.

## Production build

Build only from a dedicated clean worktree at the committed, reviewed source revision. Preparation
fails if tracked files are dirty; it binds the commit, tree, relevant Git blobs, and a complete Git
archive plus source inventory. This preserves the main worktree's unrelated research edits.
`RUN_ROOT` must not already exist. Source `scripts/chpc_env.sh` from the main checkout before entering
the clean worktree so the same untracked provider environment is bound for later operations.

```bash
module load python/3.12.12 uv/0.11.14
source scripts/chpc_env.sh

BUNDLE_ROOT=/scratch/general/vast/$USER/circuits/results/process_witness/coarse_annotation/process-witness-coarse-openai-production-v1/full-corpus-proposal-bank-v6
CALIBRATION_ROOT=/scratch/general/vast/$USER/circuits/results/process_witness/coarse_annotation/process-witness-coarse-openai-production-v1/run-shard-005-calibration-v1
RUN_ROOT=/scratch/general/vast/$USER/circuits/results/process_witness/coarse_annotation/process-witness-coarse-openai-production-v1/run-full-continuation-v1

$UV_PROJECT_ENVIRONMENT/bin/python \
  scripts/bonafide/process_witness_coarse_openai_batch_continuation_v1.py prepare \
  --bundle-root "$BUNDLE_ROOT" \
  --calibration-run-root "$CALIBRATION_ROOT" \
  --run-root "$RUN_ROOT" \
  --provider-queued-input-token-limit 40000000 \
  --tranche-empirical-queue-cap 30000000 \
  --maximum-concurrent-attempts 1 \
  --authorized-forecast-budget-usd 40 \
  --warning-spend-threshold-usd 20 \
  --hard-campaign-stop-usd 40 \
  --authorization-note 'AUTHORIZED FULL COARSE LABELING CONTINUATION: WARNING AT USD 20; HARD SUBMISSION STOP AT USD 40; ONE IN-FLIGHT BATCH MAY OVERSHOOT POST HOC' \
  --calibration-observed-input-tokens 28834177 \
  --calibration-forecast-input-tokens 28514690
```

Preparation is network-free. Inspect and record `continuation-manifest.json`,
`inherited-cost-reconciliation.json`, every `attempts/*/binding.json`, and the initial
`cost-status/receipt-0000.json` before the first provider call.

## Sequential primary execution

For each `primary-tranche-NNN` in manifest order, submit, wait until terminal, and collect before
submitting the next tranche:

```bash
$UV_PROJECT_ENVIRONMENT/bin/python \
  scripts/bonafide/process_witness_coarse_openai_batch_continuation_v1.py submit-attempt \
  --run-root "$RUN_ROOT" --attempt-id primary-tranche-NNN

$UV_PROJECT_ENVIRONMENT/bin/python \
  scripts/bonafide/process_witness_coarse_openai_batch_continuation_v1.py status-attempt \
  --run-root "$RUN_ROOT" --attempt-id primary-tranche-NNN

$UV_PROJECT_ENVIRONMENT/bin/python \
  scripts/bonafide/process_witness_coarse_openai_batch_continuation_v1.py collect-attempt \
  --run-root "$RUN_ROOT" --attempt-id primary-tranche-NNN
```

If upload/create state is ambiguous, use this exact attempt reconciliation command:

```bash
$UV_PROJECT_ENVIRONMENT/bin/python \
  scripts/bonafide/process_witness_coarse_openai_batch_continuation_v1.py recover-attempt-submission \
  --run-root "$RUN_ROOT" --attempt-id primary-tranche-NNN
```

Collection is intent-first and resumable: retained raw snapshots/files must match byte-for-byte on
retry. It appends a hash-chained cost-status receipt. Do not submit the next tranche unless the
prior collection is present and cost-complete.

## Deferred failed-only recovery and finalization

After every primary tranche is collected, freeze the one recovery wave:

```bash
$UV_PROJECT_ENVIRONMENT/bin/python \
  scripts/bonafide/process_witness_coarse_openai_batch_continuation_v1.py prepare-failed-only-recovery \
  --run-root "$RUN_ROOT"
```

Inspect `failed-only-recovery/manifest.json`. Then operate its attempt ID
`failed-only-recovery-000` with the same `submit-attempt`, `status-attempt`, and `collect-attempt`
commands. No second failed-only recovery wave is permitted. Finalization fails closed if this wave
does not resolve every inherited and new failure.

```bash
FINAL_ROOT=/scratch/general/vast/$USER/circuits/results/process_witness/coarse_annotation/process-witness-coarse-openai-production-v1/proposal-bank-continuation-v1

$UV_PROJECT_ENVIRONMENT/bin/python \
  scripts/bonafide/process_witness_coarse_openai_batch_continuation_v1.py finalize \
  --run-root "$RUN_ROOT" --destination "$FINAL_ROOT"
```

Finalization requires one successful effective event for all 37,671 original requests and exactly
three replica votes for every provider-pending unit. It copies v6 and all inherited/new provider
evidence into the result, writes a complete hash inventory, freezes all modes read-only, strictly
reloads before publication, renames atomically, and strictly reloads again. The resulting artifact
can be validated without either original VAST source root.
