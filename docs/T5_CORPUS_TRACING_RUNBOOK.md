# Qwen T5 corpus tracing runbook

Status: corpus intake is prepared and validated; no rank-screen or tracing GPU job has been
submitted.

## Frozen source and scope

The generated source is
`/scratch/rai/vast1/u1653998/bonafide/campaigns/qwen3-4b-instruct-2507-circuits-corpus-v1/attempts/generation-v1.csv`
at SHA-256 `7dec2ed942fcb6f5d5be281d61f3184db7f37689163b11e2b9c0c86af4cdcf62`.
The completed generation run contains 1,864 requests and 1,864 independently retained atomic
attempts. Every atomic file hash matches the generation run metadata, the canonical attempt rows
match the CSV, and the job completed with exit code zero.

The trace-compatible mechanical screen selects one response draw per eligible cell:

| Role | Responses | Targets per response | T5 target positions |
| --- | ---: | ---: | ---: |
| New primary discovery | 456 | 20 | 9,120 |
| Discovery bridge calibration | 23 | 20 | 460 |
| Total | 479 | 20 | 9,580 |

Eligible alternate draws are retained in the screening record but are not part of this tracing
profile. The two-draw generation design used draw one as a fallback/replicate, not as a second
primary response. Adding replicate tracing would be a separately frozen sensitivity profile.

Each response is divided into 20 contiguous token strata. One response position is selected from
each stratum by the existing SHA-256 rejection sampler with seed
`qwen3-4b-instruct-2507-t5-corpus-v1`. This provides broad temporal coverage without choosing
targets from the outcomes of an attribution graph. Every target remains an independent
single-position artifact.

## Terminal assistant-suffix correction

The generation backend retained Qwen's terminal `<|im_end|>` token (`151645`) in all 1,520
natural-EOS completion-token arrays. The frozen screen described that terminal token as excluded,
and the tracing tokenizer correctly treats it as part of the assistant suffix rather than response
content. Applying the unmodified screen would therefore overstate response and total lengths by
one token and would disagree with tracing at every eligible response.

The tracing tokenizer also gives 32 decoded responses an equivalent but different BPE
segmentation from the generation-time IDs; six of those responses are selected. Target positions
and token IDs therefore come from re-encoding the frozen response text through the authenticated
trace tokenizer, not from indexing the generation-time arrays. Generation log probabilities are
retained only when this segmentation is identical, because otherwise they have no exact
trace-token alignment. The final audit re-tokenized all 479 selected responses and matched all
9,580 target positions and IDs exactly.

The preparation profile fails closed unless the retained terminal token is one of the completion's
declared default stop IDs, removes exactly that suffix, and then reapplies all six frozen
mechanical predicates to the tokenizer-exact content IDs. This changes the primary cell yield from
the naive 458 to 456. It is recorded as
`qwen3-circuits-mechanical-screen.trace-compatible-v1`; the frozen base rule and implementation
hash remain attached.

The immutable prepared artifact is:

```text
/scratch/general/vast/u1653998/circuits/results/bonafide/downstream/
  t5-corpus-v1/preparation-v5/
```

Important files:

| File | SHA-256 | Contents |
| --- | --- | --- |
| `preparation-receipt.json` | `98e29b25b7cfda2aaf6fc7d427d4ea4fe54188fe32d09074b012051d39d0fda2` | Input validation, correction, counts, and output hashes |
| `screening-records.csv` | `0de4e435a79a27a24ca90ba0c467b94e23fb20c810dc94e1a5a939c7fcc958a1` | All 1,864 outcomes and corrected predicate evidence |
| `selected-responses.jsonl` | `03f02ece00791eb9db0279fb7f3fdb108a23bb0e3f99e59daa6400e8ac15713b` | The 479 selected response identities and exact trace-token arrays |
| `t5-source-targets.json` | `8050d0b4775ca5354674b4c63b2ac7d8a09aba95212093ece8385d869fa06a25` | 9,580 independent target work items |

Rebuilding requires a new empty output directory:

```bash
source scripts/chpc_env.sh
"$UV_PROJECT_ENVIRONMENT/bin/python" -m scripts.bonafide.prepare_t5_corpus \
  --profile scripts/bonafide/configs/qwen3_4b_instruct_t5_corpus_v1.json \
  --output-dir "$CIRCUITS_RESULTS_DIR/bonafide/downstream/t5-corpus-v1/preparation-v6"
```

## T5 semantics and execution stages

T5 here retains the project's frozen `model_top5_plus_observed` semantics. Candidate zero is the
observed teacher-forced response token. All five model-top-ranked alternatives are retained, so a
position has five candidates when the observed token is already in the model top five and six
otherwise.

The production representation remains the cancellation-resistant two-pass candidate union:

1. measure the candidate set/rank at every target without tracing a graph;
2. trace each realized candidate independently as a specified-token `k=1` objective;
3. form the exact union of independently retained nodes and exact retained edges;
4. rescore every candidate on that frozen union without a second pruning threshold; and
5. assemble one dense candidate-union artifact per target.

The summed-top-five joint graph is not substituted for this procedure.

### Stage 1: rank screen

Rank screening now groups the 20 targets from one response into one causal model forward pass.
This preserves each position's logits while reducing the full corpus from 9,580 forward passes to
479. After freezing a clean source snapshot/commit, launch one GPU job with:

```bash
SOURCE_MANIFEST=/scratch/general/vast/u1653998/circuits/results/bonafide/downstream/t5-corpus-v1/preparation-v5/t5-source-targets.json \
OUTPUT=/scratch/general/vast/u1653998/circuits/results/bonafide/downstream/t5-corpus-v1/rank-screen-v1.json \
ALL_ITEMS=1 \
PROGRESS_EVERY=500 \
UV_PROJECT_ENVIRONMENT=/absolute/frozen/t5/environment \
sbatch scripts/bonafide/topk_rank_screen.sbatch
```

The output must contain exactly 9,580 unique results and bind the source-manifest hash before the
pass-one bundle is built.

### Stage 2: build pass-one profiles

This is CPU-only and must use a new output directory:

```bash
"$UV_PROJECT_ENVIRONMENT/bin/python" -m scripts.bonafide.build_t5_corpus_bundle \
  --profile scripts/bonafide/configs/qwen3_4b_instruct_t5_corpus_v1.json \
  --source-manifest /scratch/general/vast/u1653998/circuits/results/bonafide/downstream/t5-corpus-v1/preparation-v5/t5-source-targets.json \
  --rank-screen /scratch/general/vast/u1653998/circuits/results/bonafide/downstream/t5-corpus-v1/rank-screen-v1.json \
  --output-dir /scratch/general/vast/u1653998/circuits/results/bonafide/downstream/t5-corpus-v1/pass1-bundle-v1
```

The builder emits six specified-candidate manifests, an exact case selection, and a task table.
Each wave contains at most 64 traces and is deterministically load-balanced by causal input length.
Candidate-five waves contain only realized width-six targets. Even if all targets have width six,
the role-separated bundle has at most 906 tasks, below CHPC's current `MaxArraySize=1000` and the
QOS submit limit of 1,000 jobs. The actual rank screen can only reduce that count.

### Stage 3: pass-one array

Read `execution_profile.task_count` from `t5-pass1-bundle.json`, then test and submit the exact
range with concurrency four. The launcher requires the bundle hash and clean launch commit:

```bash
sbatch --test-only --array="0-$((TASK_COUNT - 1))%4" \
  --export=ALL,BUNDLE="$BUNDLE",BUNDLE_SHA256="$BUNDLE_SHA256",EXPECTED_GIT_COMMIT="$COMMIT",UV_PROJECT_ENVIRONMENT="$T5_ENV" \
  scripts/bonafide/t5_corpus_pass1.sbatch

sbatch --array="0-$((TASK_COUNT - 1))%4" \
  --export=ALL,BUNDLE="$BUNDLE",BUNDLE_SHA256="$BUNDLE_SHA256",EXPECTED_GIT_COMMIT="$COMMIT",UV_PROJECT_ENVIRONMENT="$T5_ENV" \
  scripts/bonafide/t5_corpus_pass1.sbatch
```

Submission remains an explicit later action. The preparation step does not authorize it.

## Resource implication

This full profile is much larger than C2. C2 measured 245 target cases at approximately 23.8
A100-hours for independent pass one and 20.9 A100-hours for fixed-union pass two. Scaling by target
count gives an early planning estimate near 930 A100-hours for pass one and 820 A100-hours for pass
two, before retry/queue overhead. At four-way concurrency that is roughly 18 queue-free days for
both passes. The exact realized width-five/six mix and rank-screen output should replace this
estimate before submission.

Therefore, preparation covers every selected response, but execution should remain checkpointed:
first confirm the rank screen and one pass-one smoke wave, then decide whether to execute the full
all-response profile or freeze a smaller 96/128-response discovery checkpoint. A smaller profile
must be a new outcome-blind selection artifact; it must not silently edit this all-response source
manifest.
