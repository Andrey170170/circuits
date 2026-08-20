# Coarse proposal post-campaign analysis and sampling v1/v2

Status: **campaign analysis frozen; three candidate sampling designs frozen; no sampling policy or
trace target set selected; tracing not launched**.

This document records what can be concluded from the completed full-corpus graph-blind coarse
proposal campaign and how those proposals were converted into auditable sampling candidates. The
coarse labels remain model proposals used only for selection. They are not semantic truth,
correctness judgments, internal computations, ADAG adequacy labels, motifs, witnesses,
faithfulness results, or causal evidence.

## Campaign outcome

The immutable campaign universe contains 37,671 physical requests and 94,546 atomic units across
188 responses. Effective evidence contains 37,656 successful requests and 15 residual
`invalid_output` requests across 12 windows and 12 responses. Total receipt-derived cost was
$30.283011425.

The 15 residual rows are identity failures rather than missing provider responses: each returned
six otherwise schema-valid decisions, exactly five of which used an expected focal unit ID. The
strict lane therefore leaves 72 provider-pending atoms short of three votes:

- 74,788 atoms have three strict votes;
- 60 have two;
- six have one; and
- six have zero.

The primary analysis lane is a conservative **exact-ID salvage**. It retains only schema-valid
decisions whose unit ID exactly equals an expected focal ID and never maps or repairs an unknown
ID. That leaves only 12 provider-pending atoms incomplete: 74,848 have three votes, ten have two,
one has one, and one has zero. The original 15 invalid request statuses and raw responses remain
unchanged. A unique-complement predicate is recorded only as a diagnostic and performs zero ID
substitutions.

The frozen v1 analysis source is:

`/scratch/general/vast/u1653998/circuits/results/process_witness/coarse_annotation/process-witness-coarse-openai-production-v1/post-campaign-analysis-v1`

Its manifest self-hash is
`610e8765095551bbaadea643ca372e03254416f286bbfb8475b7df14ab501a0b`; its inventory self-hash is
`d33d1ca4215668a42ec80e943175a6e09019b186511f4d4ab05c80daf224a821`.

## Proposal census

Under conservative exact-ID salvage, all 94,546 atoms partition as:

- 38,398 `process_bearing` proposals;
- 55,994 `contextual` proposals;
- 142 `unresolved` proposals; and
- 12 explicit `missing_proposal` atoms.

The 19,500 deterministic surface/control atoms and 186 deterministic terminal atoms are inventory
routes, not stochastic model votes. Among the 74,848 provider atoms with three exact-ID votes,
59,864 have a unanimous fine label, 14,308 have a 2-1 fine split, and 676 have a 1-1-1 fine split.
These body-identical replicas measure model stability; they are not independent annotators and do
not establish label accuracy.

Across provider votes, confidence was 211,636 high, 12,620 medium, and 309 low. There were 25,389
vote-level boundary concerns: 15,703 `merge_next`, 9,167 `merge_previous`, 1,256 `split_needed`,
and 425 `meaning_unclear`. These are proposal/boundary diagnostics, not human-confirmed defects.

The process-bearing share rises across response position deciles, from 2,294 atoms in the first
decile to 5,589 in the last. This makes response position an explicit sampling dimension rather
than evidence that later tokens contain the true process. Length, source, prompt, response, task,
fragment status, and missingness tables are retained in `analysis-tables.json`.

## Sampling v2 design

The additive v2 artifact leaves the v1 evidence immutable and converts it into five overlapping
sampling mechanisms. Fragment components are the mandatory sampling PSUs, so a long computation
or list split by the 96-token cap is not given extra mass merely because it produced multiple
atoms. Incomplete atoms are hard barriers and remain eligible through uncertainty and uniform
coverage.

The exact frames are:

| mechanism | PSUs | atoms | token positions |
|---|---:|---:|---:|
| process enrichment | 20,330 | 20,373 | 304,951 |
| evaluation/commitment | 17,441 | 17,459 | 171,041 |
| diversity | 74,979 | 75,046 | 820,236 |
| uncertainty/missing | 30,842 | 30,908 | 284,695 |
| uniform reserve | 94,479 | 94,546 | 842,007 |

Mechanism eligibility overlaps. Independent nested Poisson streams select group, atom, and token
position; exact route marginals are combined into an exact union inclusion probability. First-owner
attribution uses `uniform -> uncertainty -> evaluation/commitment -> process -> diversity`, while
the union probability and inverse-probability weight do not depend on that reporting attribution.
The uniform mechanism has positive and equal support over all 842,007 response-token positions.

Three candidate policies are frozen at expected unique-position budgets of 30k, 35k, and 40k:

| policy | process | evaluation | diversity | uncertainty | uniform |
|---|---:|---:|---:|---:|---:|
| balanced | 20% | 20% | 20% | 20% | 20% |
| process weighted | 40% | 20% | 15% | 15% | 10% |
| uncertainty weighted | 25% | 20% | 15% | 30% | 10% |

The analytic expected unique totals are exactly 30,000, 35,000, and 40,000. Fixed hash-derived
streams produce nested candidate realizations without truncation; realized totals fluctuate around
the expectations:

| policy | 30k tier | 35k tier | 40k tier |
|---|---:|---:|---:|
| balanced | 29,936 | 34,927 | 39,939 |
| process weighted | 30,070 | 35,045 | 40,012 |
| uncertainty weighted | 29,898 | 34,931 | 39,891 |

Every realized row retains its target, PSU, atom, position, mechanism memberships, first owner,
exact marginal inclusion probability, inverse-probability weight, and resource-context metadata.
The tiers are candidate-only: no policy is selected and no target is authorized for tracing.

The immutable v2 sampling artifact is:

`/scratch/general/vast/u1653998/circuits/results/process_witness/coarse_annotation/process-witness-coarse-openai-production-v1/post-campaign-sampling-v2`

It is bound to source commit `1399ccf578f3f088b1c05e4b464a060de59e61d1`. Manifest self-hash is
`5d2a49a14123ed819ab404c3da8b4633eab55d8e30cf6996c7e9544c3bfc7089`; manifest file SHA-256 is
`824c5eef7267299ab17d0aeef4017c4d15de182c1ae805c486da1668af594abd`.
Inventory self-hash is
`d6ded745d84c2b59129f32beefed5ea2ba7e31c9d485eaa5bea8c4ebebf5e94c`; inventory file SHA-256 is
`7a4a7d0d31c66a96b054c9a6d470dbc2fae88e6cea5804339fdc21e54f67c574`.
The tree is 148 MiB, has `0555` directories and `0444` files, contains no symlinks, and passed
built-in prepublication, built-in postpublication, and separate CLI strict validation.

## Resource and audit gates

The rendered target-context census spans 176 to 10,767 tokens. Only 102,019 of 842,007 positions
are within the previously measured <=1,268-token T5 envelope; 739,988 exceed it. The source receipt
needed to promote that historical envelope into a strict per-target resource qualification is not
copied and bound here. Consequently every candidate retains its context count, but
`resource_qualified` remains null and `trace_ready` remains false.

The v2 audit supplement freezes diagnostic pool membership only. It does not silently select a
human-review sample or acceptance threshold. Pool sizes are 253 low-confidence PSUs, 154
unresolved/incomplete PSUs, 53 segmentation-cap PSUs, 409 long computation-syntax PSUs, 4,812
evaluation PSUs, 853 final-answer PSUs, 11,777 intermediate-commitment PSUs, and 114 uncertain
PSUs; their union is 18,046 PSUs. A reveal-free review packet, weighted estimator, sample sizes,
and numeric acceptance bounds remain pending predeclaration.

Before tracing, separately:

1. predeclare and execute the blind production audit;
2. bind strict T5 resource receipts and qualify every admitted context tier;
3. choose one policy and budget only after reviewing audit and resource results;
4. freeze a separately versioned trace manifest with exact target identities and probabilities.

No tracing command, GPU job, or provider request was launched while building either analysis
artifact. Both currently live only on purgeable VAST scratch and still require a verified durable
copy before long-term scientific reliance.

## Validation

From source commit `1399ccf`:

```bash
module load python/3.12.12 uv/0.11.14
source scripts/chpc_env.sh
export PYTHONPATH="$PWD"

"$UV_PROJECT_ENVIRONMENT/bin/python" \
  scripts/bonafide/build_process_witness_coarse_post_campaign_v2.py validate \
  --root /scratch/general/vast/$USER/circuits/results/process_witness/coarse_annotation/process-witness-coarse-openai-production-v1/post-campaign-sampling-v2 \
  --parent-v1-root /scratch/general/vast/$USER/circuits/results/process_witness/coarse_annotation/process-witness-coarse-openai-production-v1/post-campaign-analysis-v1
```

Validation is read-only and performs deterministic scientific recomputation in addition to checking
the manifest, inventory, file modes, symlinks, copied Git blobs, parent bindings, and hashes.
