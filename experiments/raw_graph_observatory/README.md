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
  --destination experiments/raw_graph_observatory/outright-task-review-v2
```
