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

## 2026-08-13 — Canonical automatic annotations and token-painting workstation

The replacement `process-witness-graph-blind-auto-v5` artifact is now the canonical automatic
draft for human review. It contains all 188 frozen responses, 842,007 authoritative response
tokens, and 786,924 graph-blind suggestions. Independent validation reproduced every source,
token, record, compact projection, payload, implementation, and manifest identity with zero
errors. The compact 32.9 MB workstation bundle is the sole canonical browser input; individual
record loading remains diagnostic.

The v5 workstation implements one-response-at-a-time exact-token painting, independent axis
layers, machine paint plus manual overrides, clear and revert-to-machine brushes, prompt/task
context, navigation/search, append-only review events, per-response/axis completion, and
provenance-bound export/resume. A real Chrome run loaded all 188 responses in 3.213 seconds and
passed painting, axis switching, clear/revert/undo, review completion, export, and fresh-page
resume with no console errors. The earlier v4 artifact is preserved but superseded because its UI
performed quadratic Unicode slicing during bundle import. No human response/axis has yet been
reviewed, and these machine suggestions are not accepted semantic labels or process truth.

## 2026-08-13 — Dense semantic draft rejected after quality audit

The v6 automatic draft fixed workstation scrolling and expanded broad semantic coverage, but it
was not promoted for human annotation. Although its 188-response artifact passed full provenance,
tokenization, projection, permission, and browser-interaction audits, a deterministic
response-stratified semantic review found avoidable false positives in final-result,
intermediate-result, state-transition, lookup, encoding, verification, and instruction-versus-work
rules. Version 6 remains an immutable diagnostic artifact only. No human review or trace-target
selection used it.

## 2026-08-13 — Conservative v7 draft rejected after post-build sampling

The replacement `process-witness-graph-blind-auto-v7` draft used a conservative v4 ontology and
token-painting UI v7. It passed full provenance and real-browser QA, but the post-build
response-stratified sample still found generic discourse shortcuts that mislabeled derived
conclusions beginning with "Given that" and one active list lookup. Version 7 is therefore
preserved as another diagnostic artifact and was not used for human review.

## 2026-08-13 — Conservative v8 automatic draft frozen for review

The final `process-witness-graph-blind-auto-v8` draft uses ontology v5 and token-painting UI v8.
It preserves exact local operator cues separately from broad candidate event spans, adds explicit
instruction, lookup, schema-relation, state-transition, arithmetic, verification, correction, and
answer layers, and leaves `usage` and `event_status` human-only. Strict replay abstains on task
descriptions, lookup imperatives, background or planned encoding, ambiguous motion,
classification and parameter-assignment arrows, negated verification, generic "given that", and
bare coefficients while retaining explicit arithmetic results, performed transitions, and active
lookups.

The frozen artifact contains 188 responses, 842,007 tokens, and 1,038,936 graph-blind automatic
suggestions. Its 39.5 MB compact bundle is the sole canonical input for real review. Automatic
coverage is deliberately layered: `discourse_phase` 97.4%, `process_span` 35.7%,
`event_operation` 33.7%, exact `operation` 3.1%, and `process_role` 17.9%. These are detector yields,
not accuracy or evidence of internal computation. Human review, `usage` assignment, and the
response-balanced trace-target conversion remain pending.
