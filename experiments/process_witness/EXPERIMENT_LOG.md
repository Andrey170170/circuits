# BonaFide process-witness experiment log

The canonical record is `experiment_log.jsonl`. This Markdown file is its concise reading view.

## 2026-08-11 — New process-witness protocol

The project question changed from a broad Qwen-Instruct feasibility pilot to a positive-control
study of whether a frozen ADAG atlas can recover stable witnesses for BonaFide-required processes.
Qwen3-4B-Thinking-2507, historical reasoning reconstruction, strict upstream top-five tracing
(`T5`), task-conditional broad completions, and complete dense trajectories became the governing
design. Earlier Instruct/CU5 artifacts remained excluded from the new atlas.

## 2026-08-12 — Step-0 T5 gate and broad generation

Historical Thinking serialization and T5 top-five objective parity passed focused validation. The
bounded Collatz landmark wave passed through 1,268 total input tokens; this established only a
bounded target-context resource tier. Broad-generation job `1791893` then completed all 558 frozen
physical draws for 186 logical slots. Integrity checks passed, but the frozen mechanical selector
resolved only 182 slots; four slots failed all three draws because of immediate repeated-token
blocks.

## 2026-08-13 — Mechanically conditioned backfill and corpus freeze

A separately versioned prompt-pooled backfill generated 28 attempts in jobs `1795117`, `1795254`,
and `1795307`. Frozen ordering selected four admissible backfills without using correctness,
faithfulness, interestingness, response length, or traceability. The authoritative read-only
cohort `qwen3-thinking-process-witness-atlas-responses-backfilled-v2` contains 188 records: 182
original full assistant responses, four full backfill responses, and two historical dense
reasoning-only reconstructions. It has 47 prompt hashes with exactly four responses each. Manifest
SHA-256 is `d4cbb862333b62d5ae108fdc2d02aab8ab47f4b729438d80f1b880f21d1d76f6`.

## 2026-08-13 — Proposed pre-witness polysemanticity validation

Before inspecting dense trajectories for candidate witnesses, test whether ADAG's global
signed-neuron clustering and one-label-per-cluster representation remains meaningful under
controlled changes in semantic composition. The proposed design uses graph-blind multi-axis token
annotations, balanced trace panels, matched atlas mixtures, held-out evaluation, and explicit
algorithmic, sampling, annotation, labeling, and composition-null noise floors. This is a proposed
protocol adjustment pending refinement; no graph or clustering output has been opened for it.

