# Raw-graph observatory

This directory records the separate exploratory lane defined by
[`plans/2026-08-20-raw-graph-observatory-v1.md`](../../plans/2026-08-20-raw-graph-observatory-v1.md).
It does not modify or admit evidence into the frozen Qwen process-witness campaign.

## Candidate review v1

`outright-task-review-v1/review.html` is the self-contained human-selection page for labeled
outright-task completions. Its source and exclusions are bound in the adjacent `manifest.json`.

- `src_type` is exactly `complex` or `graph`.
- Annotation rows are grouped by `(target_model, prompt, cot)` without dropping their labels,
  reasons, or spans.
- Model identifiers containing `qwen`, case-insensitively, are absent from the packet payload.
- Browser selections are local exploratory state. An exported selection is only a proposal until
  it passes the audit and manifest-freeze step in the plan.

The generated HTML and manifest are checked in so the initial review set is inspectable at the
branch commit. Rebuilds must use a fresh destination or explicitly replace this version with a new
review-packet version; do not silently mutate a reviewed packet.

From the repository root, rebuild a new version with the locked environment and module entrypoint:

```bash
source scripts/chpc_env.sh
"$UV_PROJECT_ENVIRONMENT/bin/python" -m scripts.bonafide.build_outright_task_review \
  --destination experiments/raw_graph_observatory/outright-task-review-v1-rebuild
```

## Candidate and token-target review v2

`outright-task-review-v2/review.html` is the all-model exploratory browser defined by
[`plans/2026-08-21-raw-graph-observatory-review-v2.md`](../../plans/2026-08-21-raw-graph-observatory-review-v2.md).
It leaves v1 unchanged, restores Qwen, shows exact reconstructed response-token statistics, and
lets a reviewer save multiple trace targets with optional comments.

V2 token positions are rebuilt under the exact pinned profiles recorded in its manifest. They are
not claimed to be recovered generation token IDs. The exported JSON is a review/discussion
artifact; re-tokenization and a separate immutable tracing manifest are required before launch.
Open the checked-in HTML directly in a current Chromium- or Firefox-family browser; its embedded
payload is decoded locally and selections persist in browser storage scoped to the payload hash.

Rebuild v2 from the repository root after sourcing the locked environment:

```bash
source scripts/chpc_env.sh
"$UV_PROJECT_ENVIRONMENT/bin/python" -m scripts.bonafide.build_outright_target_review
```

## First Qwen width-one trace wave

The first approved trace set is frozen independently of the browser export in
`scripts/bonafide/selections/qwen3_4b_thinking_raw_graph_observatory_v1.json`.
It contains response positions `65, 88, 120, 135, 162, 181, 184` from one
Qwen3-4B-Thinking completion. Each position is a separate observed-token-logit
trace; the tracing pipeline does not merge their graphs. Position 120 is the
`4` subtoken in the displayed value `45` (position 121 is the `5` subtoken).

The checked manifest binds the review-payload, frozen-selection, tokenizer,
chat-template, system-prompt, assistant-prefix, and response-token hashes. The
runner fails closed if live historical-thinking serialization disagrees with
those identities. Rebuild and compare the manifest with:

```bash
source scripts/chpc_env.sh
"$UV_PROJECT_ENVIRONMENT/bin/python" \
  -m scripts.bonafide.build_outright_trace_manifest \
  --output /tmp/qwen3_4b_thinking_raw_graph_observatory_v1.json
cmp /tmp/qwen3_4b_thinking_raw_graph_observatory_v1.json \
  scripts/bonafide/manifests/qwen3_4b_thinking_raw_graph_observatory_v1.json
```

Submit all seven independent units in one model-resident A100 job from the
repository root:

```bash
source scripts/chpc_env.sh
sbatch --export=ALL,\
MANIFEST="$PWD/scripts/bonafide/manifests/qwen3_4b_thinking_raw_graph_observatory_v1.json",\
CONFIG="$PWD/scripts/bonafide/configs/qwen3_4b_thinking_raw_graph_observatory_v1.json",\
WAVE=raw-observatory-qwen-modular-q1-width1-v1,\
ARTIFACT_ROOT="$CIRCUITS_RESULTS_DIR/bonafide/raw-graph-observatory-qwen-selected-v1" \
  scripts/bonafide/benchmark_tracing.sbatch
```

The first execution completed on 2026-08-21 as Notchpeak job `14774593` from
commit `e7d5aeaf7eb4505c5d12c624b7d784c541cf6281`. All seven compact artifacts
passed checksum and frozen target-identity validation. The job used one A100
80GB on `notch370`, exited `0:0` after 7m44s, and wrote 4,774,093 artifact bytes
under:

```text
/scratch/general/vast/u1653998/circuits/results/bonafide/raw-graph-observatory-qwen-selected-v1
```

The execution summary SHA-256 is
`be73bac9ad9d761b9796b2bb31754f011040087e1db112b9e7c21c096ad00537`.
Per-target trace times were 48.3--74.4 seconds and peak reserved GPU memory was
11.85--13.70 GiB. These measurements establish successful trace production,
not graph adequacy or semantic interpretation.

## Persistent Trace Observatory viewer

`circuits.observatory` projects the seven compact artifacts into a lossless, safe-JSON viewer
bundle and serves a Neuronpedia-inspired local interface. It preserves each target as an
independent graph, starts from a target-connected upstream focus view, exposes signed raw evidence,
and keeps label A/B overlays plus saved workspace notes separate from the source traces.

The checked implementation and exact CHPC launch/tunnel commands are documented in
[`docs/TRACE_OBSERVATORY_RUNBOOK.md`](../../docs/TRACE_OBSERVATORY_RUNBOOK.md). The current bundle
is under `raw-graph-observatory-viewer-v1` beside the source result root; persistent workspace state
is under `/uufs/chpc.utah.edu/common/home/u1653998/projects/circuits-observatory-state`.
