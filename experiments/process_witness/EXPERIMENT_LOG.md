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

## 2026-08-13 — Conservative v8 draft rejected after contextual audit

The `process-witness-graph-blind-auto-v8` draft passed full mechanical and browser audits, but its
post-build contextual audit exposed instruction-versus-execution and schema-versus-outcome errors
that sentence-local rules could not resolve. Version 8 remains diagnostic and was not opened for
human review.

## 2026-08-13 — Context-aware v9 automatic draft frozen for review

The final `process-witness-graph-blind-auto-v9` draft uses ontology v6 and token-painting UI v9.
It preserves exact local operator cues separately from broad candidate event spans, adds bounded
recent-execution and inventory-reading context, and keeps `usage` and `event_status` human-only.
The frozen known-error regression packet covers all systematic cases found during inspection:
task descriptions, lookup imperatives, active lookups, background or planned encoding, ambiguous
motion, sequence recaps, classification and parameter-assignment arrows, negated verification,
generic "given that", bare coefficients, performed transitions, and compact comparison outcomes.

The frozen artifact contains 188 responses, 842,007 tokens, and 1,038,919 graph-blind automatic
suggestions. Its 39.5 MB compact bundle is the sole canonical input for real review. Automatic
coverage is deliberately layered: `discourse_phase` 97.4%, `process_span` 35.7%,
`event_operation` 33.7%, exact `operation` 3.1%, and `process_role` 17.9%. These are detector yields,
not accuracy or evidence of internal computation. Human review, `usage` assignment, and the
response-balanced trace-target conversion remain pending.

## 2026-08-16 — Global-atlas adequacy separated from motif/witness study

The next experiment is now a standalone **global-atlas adequacy test**: whether one frozen signed-
neuron-to-cluster mapping retains locally useful distinctions across the heterogeneous outright-
process distribution. Its trace bank is selected only for ordered adequacy panels: repeatability
floor, best-case homology, natural transport, role-versus-mechanism separation, collision stress,
matched non-process specificity, and surface specificity. Each panel has a distinct inferential
job and must not be collapsed into one score.

Motif-dataset construction, dense witness trajectories, graph-motif selection, and witness
thresholds are deferred until the adequacy gate records `robust`. A `brittle` result redirects the
project to a newly versioned representation; an `inconclusive` result admits only additional
adequacy data. Existing target-local traces may later be reused when they meet a separately frozen
motif-study manifest, but the adequacy bank is not optimized for a witness hypothesis.

## 2026-08-16 — Coarse-first trace sequence and mechanism-specific adequacy battery

The adequacy study now uses two graph-blind annotation products. A frozen **coarse selection
layer** supplies broad functional regime, process family, surface form, event/response identity,
uncertainty, and provenance for stratified target selection. A richer **descriptive annotation
layer** may be refined in immutable versions while T5 tracing runs; it cannot inspect graphs or
retroactively change why a target entered the trace bank. Sampling will retain both balanced
diagnostic and natural-frequency views, explicit inclusion weights, and prompt/response/event
blocking.

The working production envelope is roughly 30,000–40,000 independent T5 targets completed within
about one week, but the exact count remains unfrozen pending coarse-label yield, target-context
resource tiers, panel support, throughput, and resume-policy design.

After the coarse target manifest freezes, three lanes proceed concurrently: T5 trace production,
graph-blind descriptive refinement, and synthetic tests of ADAG similarity/clustering. The
synthetic and real-data adequacy battery separately tests pairwise co-occurrence censoring,
bridge-induced merging, mean masking and support loss, missing-versus-incompatible ambiguity,
hard-assignment/resolution failure, position/mixture aliasing, and cluster-description dilution.

A source audit found that the paper harmonic-fuses attribution and contribution similarities
inside each context before averaging contexts, whereas released `combine="harmonic"` code averages
each view first and the callable default is arithmetic mean. The study records paper-faithful and
released-code conditions separately and will freeze their gate relationship before outcomes.
Existing Leiden and concatenated-profile clusterers are predeclared localization comparators, not
alternatives chosen because their labels look better; the same bounded grid runs on compatible
synthetic fixtures. Surface references remain excluded from the primary process atlas but may enter
one separately named nuisance-contamination fit after the primary state freezes. Only a robust
adequacy verdict permits a separately frozen motif study. Brittle outcomes are localized into
controllable, repairable, representation-level, or fundamental failures before any revised method
is attempted.

## 2026-08-16 — Coarse tags restricted to sampling; staged refinement retained

The pre-trace coarse artifact is now explicitly a **sampling instrument**, not an early semantic
ontology. Each bounded unit receives one exclusive tag: `active_task_work`,
`evaluation_or_revision`, `intermediate_commitment`, `final_answer`, `other_semantic_text`,
`surface_or_control`, or `uncertain`. The tags are retained only to reproduce and audit why targets
entered wave one. They cannot define adequacy strata, motif classes, or scientific endpoints.

Wave-one selection will be a frozen priority-weighted mixture rather than stratified-uniform
sampling: process enrichment, evaluation/commitment coverage, prompt/response/unit diversity, a
uniform reserve, and an uncertainty reserve. Target conversion may use observable anchors,
unit boundaries, sampled interiors, and a small local halo. Exact quotas and the approximately
30,000–40,000 target count remain pending the coarse-yield census and resource gates.

Coarse semantics will use direct structured LLM API calls on deterministic bounded units/windows,
with full request/response provenance and graph-blind human acceptance auditing. Rich annotation
will compare raw narrow API calls first, then a reproducibly qualified managed multi-turn or
tool-calling API workflow, and only then a Pi-based agentic harness if simpler approaches are
insufficient. Luna is the expected low-cost coarse candidate; its exact model and API contract are
not yet frozen.

After descriptive refinement, a smaller second trace wave may fill refined-label coverage deficits.
That decision must use only graph-blind annotation coverage and freeze before any ADAG graph,
cluster, or generated label outcome is opened. The wave remains separately identified; no silent
extension of wave one is permitted.

## 2026-08-16 — Coarse qualification smoke prepared; external send pending

The committed `process-witness-coarse-openai-v1` implementation partitions all 842,007 frozen
response tokens exactly once. The real-corpus census contains 94,384 units: 74,698 semantic units
pending LLM classification, 19,500 structural/control units assigned deterministically, and 186
exact terminal answer-serialization units. The two historical reasoning-only responses receive no
fabricated final-answer unit.

The first Luna packet is a deliberately small qualification smoke, not an error-rate study: 12
unique windows, six focal semantic units each, and four interleaved body-identical repeats. Its 72
unique focal units cross complex/graph sources with all four hidden v9 sampling hints and balance
early/middle/late response positions. V9 suggestions select strata only and never enter provider
input. The request body contains the original task prompt plus a bounded response window, and the
strict output asks only for one coarse tag, confidence, and boundary concerns per focal unit.

The immutable offline bundle has manifest SHA-256
`c32d1e111128afc8b78137df0897b973e8bc76872d7ac2e4a12899121e2ca5c9` and binds commit
`81f46bea1b4f79f05a0d1e9713a822e8d380aa9e`. Its conservative live ceiling is $0.0964352; the
runner records pre-call intents, raw provider receipts, resolved model, normalized usage, per-call
cost, and cumulative cost, with SDK retries disabled. No API request was sent at this checkpoint:
the external-send safety gate requires explicit approval to transmit BonaFide-derived task prompts
and bounded response windows to `https://api.openai.com/v1`.

## 2026-08-16 — Coarse qualification executed; scale-up held for blind review

After explicit approval, all 16 predeclared direct Responses requests completed successfully with
resolved model `gpt-5.6-luna`. Every receipt, intent, parsed record, decision, event, and manifest
binding validates, and all 96 physical decisions exactly cover their requested focal units. The
immutable run manifest SHA-256 is
`88a6e279d7aba0111a8c3d9386da77c87ca5fde80fedb9706ab95c3fc83cb59d`.

The run exposed a cache-write accounting defect without invalidating the provider evidence. Raw
receipts contain 28,432 total input tokens partitioned into 48 ordinary, 7,084 cache-read, and
21,300 cache-write tokens, plus 6,873 output tokens. The frozen runner recorded $0.01265888 because
it priced cache writes as ordinary input. A separate self-hashed audit, leaving the run untouched,
corrects the total to $0.01372388 (delta +$0.001065). Future Responses usage normalization now
preserves all three input buckets.

The semantic smoke does not yet justify unattended scale-up. Across four body-identical repeats,
tag agreement is 18/24 (75%), exact decision agreement is 16/24 (66.7%), and confidence agreement
is 19/24 (79.2%). Luna emitted no `uncertain` tags and no boundary concerns, while 70/72 unique
units received high confidence. Because no semantic threshold was frozen in advance, this is a
conservative hold rather than a formal statistical failure.

The 72 unique focal units are frozen in blind-first review packet
`process-witness-coarse-review-v1-efd0b5d8b3ad2af6`, manifest SHA-256
`14f9a4438a6967abd727a97920a436aad87d91d601bf28ce31806e74f59b924f`. The reviewer must lock a
human tag before either Luna decision or repeat disagreement is revealed; later corrections remain
separate from the blind judgment. Full-corpus labeling and wave-one target selection remain blocked
until this review determines whether the v1 prompt is acceptable or needs a new version.

## 2026-08-16 — Full-context markup arms executed; neither protocol qualified automatically

The v2 qualification reused the exact 12 v1 windows and 72 focal units in two matched arms. In
`target_only_markup`, the full task prompt and complete raw response were shown, but only the six
focal units were wrapped as targets. In `full_unit_markup`, every coarse unit in the same complete
response was wrapped as target or context. Both arms used `gpt-5.6-luna`, medium reasoning,
`max_output_tokens=16384`, strict structured output, and four exact repeats. The comparison plan was
hash-frozen before submission and included within-arm repeats, cross-arm primary decisions, and
each arm against the completed v1 baseline.

All 32 native-Batch requests completed with zero provider or validation failures and exact coverage
of 144 arm-by-target decisions. The immutable collection manifest SHA-256 is
`1c3f9c8c8ffb670399a77ea836a6b8348b8084264b89d79203ec341c148fca15`; actual receipt-priced spend
was $0.12362749. Target-only cost $0.01770586, while full-unit markup cost $0.10592163 because the
unit tags expanded input from 108,757 to 1,088,487 tokens. Cache-read totals were 29,311 and 312,063
tokens respectively.

Neither arm improved exact-repeat stability enough to qualify automatically. Target-only repeat tag
agreement was 18/24 (75.0%) and exact-decision agreement 16/24 (66.7%). Full-unit repeat tag
agreement was 17/24 (70.8%) and exact-decision agreement 16/24 (66.7%). Across the two primary arms,
tag agreement was 50/72 (69.4%), exact-decision agreement 43/72 (59.7%), confidence agreement 61/72
(84.7%), and boundary agreement 68/72 (94.4%). Against v1 primaries, tag agreement was 38/72 for
target-only and 45/72 for full-unit; these are protocol differences, not correctness measurements.

The immutable comparison bundle manifest SHA-256 is
`71865ca5afbc51e2b5ee31b11421c128b86d2bed0ab999a1ca91b85da9f33854`. It preserves all metric
disagreements and a deterministic agreement sample without interpreting either side as correct.
The result supports the concern that dense markup is a costly and potentially distracting
presentation, but blind human review is still required to determine whether either arm is more
semantically sensible. Full-corpus labeling and target-bank freeze remain blocked.

## 2026-08-17 — Human review retained as development evidence; matched few-shot test accepted

The uploaded full-context ledger contains 72/72 locked blind judgments and has SHA-256
`2b4cf65ea8bf92662b261b691c2baa3638f220bca2de5a57a9f7518cbaa2b0bc`. Blind tag counts are 31
other semantic, 20 active work, 11 evaluation/revision, seven intermediate commitments, and three
surface/control. Nine items have separately preserved post-reveal corrections. Because the human
reviewer learned the ontology during this ordered pass and saw model decisions after each lock,
these records are development evidence rather than a clean estimate of model accuracy.

The next qualification freezes labels by their visible trajectory effect and retains genuine
ambiguity. Three identical-protocol Luna decisions form an ordered vote profile; 3-0, 2-1, and
1-1-1 measure stability and never become ground truth merely through majority vote. A matched
fresh holdout will compare refined zero-shot instructions with the same instructions plus short
contrastive micro-context demonstrations. Both arms use the full task prompt, complete response,
target-only inline markup, medium reasoning, strict output, and identical target groups. The
holdout excludes the development units and remains graph-blind and globally human-blind until its
review finishes. Dense all-unit markup is retained only as completed sensitivity evidence, not the
new primary protocol.

The frozen design target is 24 unique prompt/response windows and 144 semantic units, exactly one
window per source-by-position-by-hidden-hint cell. Two arms and three identical-body replicas yield
144 physical Batch requests and 864 physical unit decisions. Few-shot is rejected if it introduces
more than two additional human process-bearing false negatives; it is preferred only with at least
five net paired admissible-agreement wins and no increase in stable high-confidence errors.
Otherwise the arms tie and zero-shot remains the parsimonious production proposal.

## 2026-08-17 — Refined zero-shot/few-shot Luna qualification completed

The user granted project-wide authorization to send public BonaFide-derived task prompts and model
responses to the OpenAI API; the durable scope and safeguards are recorded in
`docs/BONAFIDE_EXTERNAL_API_AUTHORIZATION.md`. The run-specific immutable intent also records the
authorization verbatim and enforced a $3.00 hard ceiling.

The fresh v3 qualification used 24 unique prompt/response windows, six consecutive semantic units
per window, two target-only full-context arms, and three identical-body replicas per arm/window.
All 144 native-Batch requests completed successfully, yielding exact coverage of 864 physical unit
decisions and 288 unique arm-by-target decisions. There were zero provider or validation failures,
and all 144 provider response identities were distinct.

The immutable collection manifest SHA-256 is
`9d748de8fbf24881348eeceaea7ad92144adb08d56ff6329b61e827f95c49783`.
Receipt-derived usage was 1,061,454 input tokens, including 707,348 cache-read and 353,674
cache-write tokens, plus 119,525 output tokens of which 73,117 were reasoning tokens. Only 432 input
tokens were neither cache reads nor cache writes. Actual cost was $0.12304093, well below the
$1.96969785 conservative no-cache/full-output estimate and the $3.00 hard ceiling.

The model outputs remain concealed from the human reviewer. The globally blind packet contains no
model decisions or reveal payload; all 144 human judgments must be sealed before a separate,
identity-bound comparison/reveal artifact can be built. Therefore the completed run establishes
protocol execution and exact label proposals, but not which arm is more accurate or suitable for
production sampling.

## 2026-08-17 — Full-context blind review packet v2 frozen

The first v3 human-review packet (`qualification-refined-zero-vs-few-shot-v1-human-review-v1`)
contained invalid generated JavaScript and opened as an empty page. It is preserved for provenance
but is superseded and must not be used for human review.

The authoritative replacement is
`qualification-refined-zero-vs-few-shot-v1-human-review-v2`, built from commit `cd681d0`. It restores
the v2-style three-column presentation: the full task prompt and exact full response with the six
request targets highlighted, the blind judgment form, and a continuously visible reference panel
containing all seven coarse-label definitions and four boundary-concern definitions. The page is
self-contained; initial review requires no JSON import. Import is only for resuming an exported
progress ledger.

The replacement remains globally blind and contains no model outputs or reveal payload. Its packet
ID is `process-witness-coarse-review-v3-b34f4ade7b74ed26`; manifest self-hash is
`4dc3b0b741574b43573b623e499b939a1708f61c5e627c44484160f4c905d6c5`; and `review.html` SHA-256 is
`edb331bf6b6dae71a58aad7a0bf75c9d4a6cdeeda4d253fc106d412821d0a957`. Exact-artifact browser QA
passed full-response rendering, focus navigation, empty-filter behavior, decision persistence,
global sealing, reload, export enablement, sticky references, and desktop-width scrolling with zero
console or runtime errors. No human judgments have yet been collected in this replacement packet,
so no arm is selected and no labeling, tracing, adequacy, motif, witness, faithfulness, or causal
claim follows from this freeze.
