# Width-one labeling v2.1 pilot report

Status: completed diagnostic comparison on 2026-07-31.

## Scope

This pilot labels the same 12 explicitly frozen clusters from the primary and alternative
width-one clustering states. Each witness is an independent, single-observed-target attribution
trace. Contribution evidence is shallow: there is no non-degenerate contribution profile, top-k
target comparison, matched no-hint control, or intervention. Every retained label is therefore a
corpus-bounded association, not a causal, contribution, faithfulness, selectivity, or generality
claim.

Run root:

```text
/scratch/general/vast/$USER/circuits/results/bonafide/labeling/
comparison-pilot-adbfff4-v2
```

The provider paths contain immutable requests, responses, score artifacts, telemetry, and both
the original candidate-gated assessment and the corrected additive
`assessments/label_quality_v2/` bundle.

## What v2 fixed

The v1 label names mentioned hint, audit, security, injection, leak, metadata, or unauthorized
access in 23 of 24 cases even though all exemplars share that corpus context. V2 reduces that count
to zero and makes the width-one limitations explicit.

The first v2 assessment exposed a second issue: candidate correlations evaluated Luna/Haiku text,
but Terra/Opus often rewrote the final label. V2.1 therefore:

1. treats literal `insufficient_evidence` as control flow rather than simulator text;
2. scores the exact final label on `selection_scoring`;
3. uses that final-label score as the only automatic consistency gate;
4. keeps the best candidate score as a diagnostic;
5. reports audit separately and never uses it for automatic acceptance or rewriting.

The positive gate is intentionally minimal. `review_required` means only that a substantive final
label has a finite positive selection correlation and still needs semantic review. There is no
automatic acceptance state.

## Corrected results

Terra returned 11 explicit abstentions and one `fact/facts token association`; the latter had a
final-label selection correlation of `-0.0500`. Its corrected bundle is therefore 12/12
`insufficient_evidence`. Terra is useful as a conservative disagreement baseline but is too
abstention-heavy to be the sole labeler.

Opus returned eight provisional labels and four abstentions. Final-label scoring yields six
`review_required` and six `insufficient_evidence`; human review retains four bounded hypotheses.

| State | Cluster | Final-label selection r | Audit r | Human disposition | Label or reason |
| --- | ---: | ---: | ---: | --- | --- |
| Primary | 0 | 0.0617 | -0.0224 | Downgrade | Generic target-local recency; target identity and phase are confounded |
| Primary | 12 | N/A | N/A | Insufficient | Model abstained; no coherent localized pattern |
| Primary | 24 | 0.2566 | 0.0958 | Retain as background | Shared instruction template, especially the `last line` directive |
| Primary | 38 | 0.1743 | 0.2041 | Retain provisional | Information/data-reference nouns in verification-style reasoning spans |
| Primary | 50 | 0.3988 | 0.4225 | Retain provisional | Head nouns completing evidential-source phrases; strongest pilot label |
| Primary | 62 | N/A | N/A | Insufficient | Model abstained; polarity and highlighted spans are inconsistent |
| Alternative | 0 | N/A | N/A | Insufficient | Model abstained; no stable localized class |
| Alternative | 17 | 0.2010 | -0.1305 | Downgrade | Ubiquitous function words and answer-slot position explain the pattern |
| Alternative | 37 | 0.0865 | 0.2159 | Retain as background | Shared reasoning boilerplate and assistant-response onset |
| Alternative | 54 | -0.0599 | 0.0012 | Insufficient | Repetition hypothesis does not pass final-label selection |
| Alternative | 73 | N/A | N/A | Insufficient | Model abstained; sparse heterogeneous spans |
| Alternative | 94 | -0.0072 | 0.1279 | Insufficient; diagnostic | Prompt-span subword continuation is coherent visually but fails selection |

The primary state produced three retained hypotheses among its six sampled clusters; the
alternative state produced one. This pilot is too small and was not sampled to estimate a
state-level labelability rate, so it does not replace the label-free clustering-quality decision
or justify discarding the alternative state.

## Cost and execution ledger

The retry-aware rollup includes successful canonical calls, archived truncated originals, and
distinct retry attempts exactly once.

| Path | Real API attempts | Input tokens | Output tokens | API cost |
| --- | ---: | ---: | ---: | ---: |
| OpenAI | 72 | 212,207 | 22,119 | $0.09524426 |
| Anthropic | 124 | 591,243 | 109,479 | $2.36251450 |
| Total | 196 | 803,450 | 131,598 | $2.45775876 |

Anthropic cost is dominated by archived live retries after 40 Haiku candidates hit the 700-token
cap and all 12 adaptive-Opus summaries hit the 1,200-token cap. The successful retry limits were
1,400 and 3,200 tokens, now recorded as the future Anthropic v2 defaults.

Successful simulator telemetry records `0.095270` A100 GPU-hours of model processing. The six
successful Slurm jobs occupied `0.3236` A100-hours including imports and checkpoint loading. A
stale inherited Hugging Face cache path caused one failed and one cancelled launch, adding
`0.1028` A100-hours; no score artifacts were written by those attempts, and the launcher now pins
the intended cache.

## Recommendation

Use Opus as the primary semantic rewriter and Terra as an explicit conservative/abstention
comparison. Carry primary 38 and 50 as the best provisional hypotheses; carry primary 24 and
alternative 37 only as background/template descriptors. Keep all other pilot clusters at
`insufficient_evidence`.

Do not scale the hosted recipes to all 150 ready clusters yet. First run the identically selected
Qwen comparison, then decide whether the four retained hypotheses and provider disagreements
justify a larger label run. Before stronger scientific interpretation, add matched controls or
top-k traces that separate lexical/positional association from genuine contribution structure.
