# Frozen labeling comparison pipeline

Status: implementation-ready; hosted-provider smoke tests passed; the corrected 12-cluster
comparison pilot is active.

## Comparison contract

All three recipes use the same frozen primary/alternative cluster bundle, prompt renderer,
generation/selection/audit partitions, and local evaluator:

| Recipe | Candidate generator | Candidate selection and label audit | Final summarizer |
| --- | --- | --- | --- |
| `qwen-only-v1` | configured OpenAI-compatible Qwen endpoint | `Transluce/llama_8b_simulator` | same Qwen endpoint |
| `openai-5.6-v1` | `gpt-5.6-luna` | `Transluce/llama_8b_simulator` | `gpt-5.6-terra` |
| `anthropic-original-upgraded-v1` | `claude-haiku-4-5-20251001` | `Transluce/llama_8b_simulator` | `claude-opus-5` |

Changing the hosted provider does not change the judge. The simulator is loaded inside a circuits
GPU job because it consumes per-token logits and is not equivalent to an ordinary chat endpoint.
Qwen remains a separate HTTP service and does not share the tracing environment.

The evidence flow is:

```text
generation witnesses -> five candidate descriptions
selection_scoring witnesses -> fixed local simulator -> candidate ranking
ranked candidates + frozen structure -> one cluster summary
audit witnesses -> fixed local simulator -> final-label audit
```

The confirmatory holdout is never loaded by this runtime.

## Configuration

Recipes and the explicit dated price snapshot live in
`scripts/bonafide/configs/labeling/`. Provider, model, endpoint, reasoning mode, concurrency,
timeout, retry policy, and generation limits are configuration fields.

Expected environment variables:

- OpenAI: `OPENAI_API_KEY`;
- Anthropic: `ANTHROPIC_API_KEY`;
- Qwen endpoint: `QWEN_BASE_URL`;
- Qwen authentication, when the server requires it: `QWEN_API_KEY`.

Secrets are read only from the environment. They are never put in requests, run manifests,
telemetry, command arguments, or endpoint identities.

The price snapshot is not silently refreshed. A new official price date requires a new file and
recipe revision. New runs use `prices-2026-07-30.json`, based on the official
[OpenAI pricing](https://developers.openai.com/api/docs/pricing),
[OpenAI Batch](https://developers.openai.com/api/docs/guides/batch), and
[Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing) pages as read on
2026-07-30. Older run manifests retain their original dated snapshot and telemetry; later price
changes are recorded as additive reconciliations rather than retroactive artifact edits.

## Prepare a smoke or pilot

Source the CHPC environment first:

```bash
module load python/3.12.12 uv/0.11.14
source scripts/chpc_env.sh
```

Use a new output root for every recipe. For the two-cluster live smoke:

```bash
"$UV_PROJECT_ENVIRONMENT/bin/python" scripts/bonafide/labeling_pipeline.py \
  prepare-candidates \
  --frozen-root /scratch/general/vast/$USER/circuits/results/bonafide/clustering/cluster-selection-d5a771f-v1 \
  --recipe scripts/bonafide/configs/labeling/openai.json \
  --output-root /scratch/general/vast/$USER/circuits/results/bonafide/labeling/openai-smoke-v1 \
  --cluster-limit 2 \
  --transport-override live
```

Use `--cluster-limit 12` for the comparison pilot. The limit is the total across requested states:
12 becomes six primary and six alternative clusters, while the two-cluster smoke becomes one from
each. Omit the limit only for the full 150-ready-cluster comparison. The deterministic selector
spreads each state's allocation across its sorted ready cluster IDs. A scientific pilot should
record and, if needed, replace this automatic sample with an explicitly reviewed stratified
cluster list before submission.

Preparation:

- validates the master and state self-hashes plus every frozen file hash;
- reconstructs one polarity-aligned mean cluster attribution profile per exact witness trace;
- pads the known missing leading/BOS attribution entry using the existing labeler convention;
- renders the generation witnesses with the strongest signed token highlights;
- persists all three partitions for later local scoring;
- writes five logical requests per cluster;
- fails on a dirty labeling source tree unless `--allow-dirty` is explicitly used for development.

## Execute model requests

For a Qwen endpoint or a deliberately tiny hosted-provider live smoke:

```bash
"$UV_PROJECT_ENVIRONMENT/bin/python" scripts/bonafide/labeling_pipeline.py \
  execute-live \
  --run-root /scratch/general/vast/$USER/circuits/results/bonafide/labeling/openai-smoke-v1
```

For hosted production work, use native batches:

```bash
python_bin="$UV_PROJECT_ENVIRONMENT/bin/python"
run_root=/absolute/run/root

"$python_bin" scripts/bonafide/labeling_pipeline.py prepare-batch \
  --run-root "$run_root" --stage candidate_generation
"$python_bin" scripts/bonafide/labeling_pipeline.py submit-batch \
  --run-root "$run_root" --stage candidate_generation
"$python_bin" scripts/bonafide/labeling_pipeline.py batch-status \
  --run-root "$run_root" --stage candidate_generation
"$python_bin" scripts/bonafide/labeling_pipeline.py collect-batch \
  --run-root "$run_root" --stage candidate_generation
```

Submission is an explicit separate command because it creates billable external work. Status
polling does not occupy a Slurm job. OpenAI inputs use one `/v1/responses` JSONL line per logical
request and unique `custom_id`; Anthropic inputs use one Message Batch request per logical
request. Multiple clusters are never combined into one semantic prompt.

## Local scoring and summarization

The simulator checkpoint must be staged before submission and the frozen labeling environment
must contain exact revision `63919a3fe41f88d91ef764213ae9018e1f8a578e`. Scoring resolves the
repository/revision pair to its cached snapshot path before loading, so the production job remains
strictly offline and does not depend on Hugging Face metadata access.

Candidate scoring:

```bash
sbatch --test-only \
  --export=ALL,RUN_ROOT=/absolute/run/root,LABELING_ENV=/absolute/frozen/env,PHASE=candidate_selection \
  scripts/bonafide/labeling_score.sbatch

sbatch \
  --export=ALL,RUN_ROOT=/absolute/run/root,LABELING_ENV=/absolute/frozen/env,PHASE=candidate_selection \
  scripts/bonafide/labeling_score.sbatch
```

The scorer re-tokenizes the frozen Qwen-token profile for the Transluce tokenizer by deterministic
character overlap and records coverage diagnostics for every exemplar. It selects on the
`selection_scoring` partition only.

After all candidate score files exist:

```bash
"$UV_PROJECT_ENVIRONMENT/bin/python" scripts/bonafide/labeling_pipeline.py \
  prepare-summaries --run-root /absolute/run/root
```

Run or batch `cluster_summary` through the same commands used for generation. Then audit final
labels:

```bash
sbatch \
  --export=ALL,RUN_ROOT=/absolute/run/root,LABELING_ENV=/absolute/frozen/env,PHASE=summary_audit \
  scripts/bonafide/labeling_score.sbatch
```

For the full comparison, the natural concurrency is three candidate-scoring jobs—one per recipe—
followed by three summary-audit jobs. Each job loads the simulator once and processes its run’s
clusters; one GPU job per cluster would waste model-load time.

## Outputs and resumption

Each logical request owns independent atomic files:

```text
manifest.json
profiles/<state>/cluster-####.json
requests/<stage>.jsonl
provider_batches/<stage>/
results/<stage>/<request_id>.json
telemetry/<stage>/<request_id>.json
scores/candidate_selection/<state>/cluster-####.json
scores/summary_audit/<state>/cluster-####.json
telemetry/local_scoring/<phase>-<job>-<task>.json
stages/cluster_summary/manifest.json
```

A retry skips only requests having both a result and telemetry file. A lone file is treated as a
partial failure and stops the run. Provider-batch collection maps by request ID rather than output
order.

Per-request telemetry includes requested/resolved model, provider request ID, prompt/evidence and
source hashes, endpoint identity without credentials, generation parameters, attempts, latency,
stop reason, parse status, response hash, uncached/cache-read/cache-write/input/output/reasoning
tokens, and a cost estimate tied to the price snapshot. Unknown rates make the estimate incomplete
instead of zero. Local telemetry keeps API dollars and GPU-hours separate and records elapsed
time, allocated GPUs, peak HBM, and peak host RSS.

## Current verification

On 2026-07-27:

- focused schema, pricing, provider formatting, prompt, retokenization, resumption, and fake-backend
  tests passed;
- a real frozen primary cluster produced all three profile partitions, five request records, one
  shared prompt identity, and no holdout access;
- a minimal `gpt-5.6-luna` Responses request parsed successfully and reported usage;
- a minimal `claude-haiku-4-5-20251001` Messages request parsed successfully and reported usage;
- no Terra, Opus, native batch, Qwen endpoint, or Transluce simulator production call was made.

The prepared 12-cluster pilot is:

```text
/scratch/general/vast/$USER/circuits/results/bonafide/labeling/
comparison-pilot-fcb2549-v1
```

It contains six primary clusters (`0, 12, 24, 38, 50, 62`) and six alternative clusters
(`0, 17, 37, 54, 73, 94`). Each recipe has 60 candidate requests and 12 profile files. The sets of
12 rendered prompt hashes are identical across Qwen, OpenAI, and Anthropic. OpenAI and Anthropic
native-batch inputs are prepared but have not been submitted.

### Corrected active pilot

Offline scorer preflight exposed a Transformers metadata lookup when the simulator was loaded by
repository name. Commit `755e37a` pins simulator revision
`63919a3fe41f88d91ef764213ae9018e1f8a578e` and resolves it to the local snapshot path before
loading. Because this changes recorded scorer provenance, the original `fcb2549` pilot remains
unsubmitted and a new pilot was prepared at:

```text
/scratch/general/vast/$USER/circuits/results/bonafide/labeling/
comparison-pilot-755e37a-v1
```

On 2026-07-29, the OpenAI Luna and Anthropic Haiku candidate batches each completed 60 requests.
All Luna outputs parsed successfully. One Haiku output was invalid JSON and was retried once
through the live Messages endpoint; the failed batch artifact and telemetry remain archived under
that request's `provider_batches/candidate_generation/retries/` directory. Both recipes therefore
entered candidate scoring with 60 valid descriptions.

The initial A800 jobs `1676775` and `1676776` and first owner-A100 attempts `14382311` and
`14382312` failed before scoring because CUDA memory telemetry was reset before the process had
initialized its CUDA context. They wrote no score artifacts. Recovery jobs `14382647` (OpenAI)
and `14382648` (Anthropic) explicitly initialized the context and completed all 12 clusters and
60 candidate correlations per recipe with no null scores. They used 0.038759 and 0.023972
A100 GPU-hours, respectively. Commit `579a71b` codifies the initialization for future jobs; the
active run manifests remain bound to the unchanged `755e37a` source snapshot. Three clusters per
recipe have a negative best candidate correlation, so their generated summaries remain
provisional rather than being silently discarded.

On 2026-07-30, the 12-request Terra summary batch
`batch_6a6bc0cecb2481908f73edf4e3854077` completed with 12 valid labels. The 12-request Opus batch
`msgbatch_01Dzn8SKncTvvYUNh5vvwf6P` completed at the provider but hit its 300-token cap on ten
responses. Commit `e2859fd` added provenance-preserving live retries; all ten 1,200-token retries
parsed successfully while retaining the original result, telemetry, requests, hashes, and retry
manifests. Terra's 12-label audit completed in job `14383331`; its correlations range from
-0.093977 to 0.150438, reinforcing that these are exploratory labels rather than accepted
mechanistic explanations. Opus audit job `14383798` also completed all 12 labels, using
0.005594 A100 GPU-hours; its correlations range from -0.038592 to 0.220686.

The active manifests still pin `prices-2026-07-27.json`. A separate 2026-07-30 reconciliation uses
the reduced OpenAI native-Batch rates per million uncached-input/cache-read/output tokens:
Luna `$0.10/$0.01/$0.60` and Terra `$1.00/$0.10/$6.00`. The measured Luna usage reprices to
`$0.01113894`; measured Terra usage (21,310 input and 754 output tokens) reprices to `$0.025834`.
The pilot's OpenAI path is therefore `$0.03697294` at current rates. Scaling the maximum-output
pilot envelope to all 150 ready clusters gives a conservative estimate of about `$0.68`; a `$1`
full-run OpenAI guardrail leaves ample contingency. Historical telemetry is not rewritten.

The Qwen pilot root retains the identical cluster selection and waits for its separately managed
endpoint.

### Width-one-aware v2 pilot

Manual inspection of the first hosted pilot found a corpus-context confound: 11 of 12 Terra labels
and all 12 Opus labels named hint, injection, security, or audit context even though that context
is shared by every frozen exemplar and therefore is not cluster-discriminative. Several localized
attribution peaks instead fell on generic reasoning transitions, factual lookup, verification, or
named-entity continuation. Fluent JSON and model-reported confidence were consequently not treated
as label quality.

The additive `width_one_v2` prompt policy leaves the v1 recipes and artifacts unchanged. It:

- states that each witness is an independent single-observed-target, width-one attribution trace;
- records that contribution evidence is shallow and provides neither a non-degenerate contribution
  profile nor a top-k target comparison;
- treats hint/security/audit wording and step-by-step style alone as shared corpus background;
  localized attribution on those spans may support only a corpus-bounded association, not
  selectivity or generality without matched controls;
- requires candidate outputs to separate localized evidence, background/confounds, and limitations;
- gives the final summarizer the exact highlighted `generation` and `selection_scoring` witnesses,
  with their profile and score hashes, while excluding `audit` witnesses;
- permits an explicit `insufficient_evidence` result and raises summary output limits to 1,200
  tokens;
- writes a deterministic post-run quality bundle in which non-positive or nonfinite selection
  scores and model-declared insufficiency fail closed, while every remaining label is
  `review_required`, never automatically accepted;
- records audit correlation separately for evaluation and never uses it to rewrite or accept a
  label.

The rerun uses explicit cluster IDs rather than recomputing a size-limited sample: primary
`0, 12, 24, 38, 50, 62` and alternative `0, 17, 37, 54, 73, 94`. The v1 pilot tree remains an
immutable comparison predecessor.
