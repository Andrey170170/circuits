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

## 2026-08-13 — Polysemanticity validation accepted; first annotation pass authorized

The pre-witness quality gate is now part of the central protocol. Annotation uses independent
semantic, process-role, discourse, representation, surface, and token-position axes rather than a
single forced class. Automatic rules provide reviewable suggestions and must be inspected against
raw responses before scaling. Exact measurements may be refined after trace fields and coverage
are known, but must freeze before cluster assignments, labels, semantic-composition comparisons,
or dense witnesses are opened. T5 remains primary. The exact frozen target bank may also be traced
with CU5 in parallel under separate, sealed artifacts for later method-development evidence; CU5
does not itself establish that ADAG modification is necessary.

## 2026-08-13 — Automatic annotation bootstrap and strict-review stop

Automatic drafts through `process-witness-graph-blind-auto-v3` were built for all 188 frozen
responses using the exact Qwen Thinking tokenizer and response-relative character spans projected
onto verified continuation token identities. It contains 842,007 response tokens and 786,932
overlapping suggestions. The structural audit passed. Raw per-rule inspection caught and removed
avoidable first-pass mistakes including list bullets crossing newlines into subtraction matches,
Markdown asterisks treated as multiplication, decimal dots treated as sentence terminators,
backticks treated as quotes, hard-coded JSON answer keys, and missed attached-modulo and Unicode
operator forms.

Strict review then blocked human use of those artifacts. It found stale response context in the
review UI, missing prompt/task context, underspecified ontology and review provenance, excessive
apostrophe-as-quote and percentage-as-modulo matches, and insufficient UI scaling/resume checks.
Versions 1–3 are preserved but superseded.

The implementation checkpoint fixes those findings and passes seven focused tests plus Ruff,
Python compilation, and diff checks. No replacement artifact was built after the fixes, no human
review began, and no trace, graph, cluster, or label output was opened. The next action is an
interactive UI smoke test followed by a newly versioned automatic-corpus rebuild and independent
review; only that later version may begin human annotation.
