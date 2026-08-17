# BonaFide process-witness discovery plan

Status: **central protocol; the architecture and historical-reconstruction choice are accepted.
Step-0 input inventories are frozen in version 1, the bounded T5 landmark gate passed, and the
balanced 188-response atlas-response cohort is frozen as
`qwen3-thinking-process-witness-atlas-responses-backfilled-v2`, and its graph-blind v9 automatic
annotation draft is frozen for human review. The cohort is not yet a fitted atlas. The next study
is a standalone global-atlas adequacy test, not a combined adequacy-and-witness experiment. It will
test whether ADAG's global signed-neuron clustering and cluster labels remain useful under
controlled semantic mixture. A frozen coarse selection layer is required before target selection;
the richer descriptive annotation layer and synthetic ADAG tests then proceed concurrently with
production tracing. Witness-specific dataset construction, dense trajectory production, and motif
testing begin only after a robust adequacy verdict. Production tracing remains gated by the coarse
graph-blind selection layer, response balancing, and target-context resource tests.**

This is the governing plan for the new process-witness campaign. Where it conflicts with
`docs/ADAG_BONAFIDE_NAIVE_PILOT.md` or `docs/TRACING_CORPUS_PLAN.md`, this plan governs the new
campaign; those documents continue to describe the earlier Qwen Instruct feasibility pilot and its
frozen execution history.

`CONTEXT.md` is the canonical glossary for outright process tasks, task-required process events,
process families and signatures, adequacy panels, motif candidates, and witness claim levels.

This is a new experiment identity. It does not modify or extend the frozen Qwen Instruct
width-one or candidate-union corpora. Those artifacts remain engineering evidence and possible
comparators, but they are not scientific inputs to this study.

## Grand goal and present question

The long-term goal is a general detector of chain-of-thought faithfulness under BonaFide's
mechanistic definition. In loose terms, that detector must eventually test two directions:

1. **Verbalization soundness:** when the CoT says that process `P` occurred, the model computed a
   compatible `P` somewhere in the relevant trajectory.
2. **Relative completeness:** when the model used a decision-relevant process `P`, the CoT
   verbalized a compatible `P` somewhere in the response.

The computation and its verbalization need not occur at the same token position. Exact temporal
alignment is not assumed.

The program retains the following process-recovery question:

> When BonaFide tells us that a particular answer-relevant process must have occurred, can a
> frozen ADAG atlas recover a stable and recognizable witness of that process in an ordered
> series of attribution graphs?

That question is now explicitly conditional on a prior experiment:

> **Global-atlas adequacy test:** Can one global signed-neuron-to-cluster mapping retain locally
> useful process distinctions across the full outright-process distribution?

Run the adequacy test first and decide it independently. Do not construct a witness-optimized motif
dataset merely because the same traces might later be reusable. A robust adequacy verdict permits a
separately frozen positive-control process-recovery study. A brittle verdict redirects work toward
a context-conditional or otherwise revised representation. An inconclusive verdict permits only
additional adequacy data. Neither experiment by itself establishes either direction of a general
faithfulness detector.

## Scientific object

For a stored response `y`, response position `t`, and selected model `M`, define the teacher-forced
prefix and graph as:

```text
prefix(r, t) = chat_template(prompt, y[:t])
G(r, t)      = one independently pruned ADAG graph at that prefix
```

The dense trajectory is the ordered collection:

```text
G(r, 0), G(r, 1), ..., G(r, T - 1)
```

It is a sequence of target-local attribution snapshots under growing prefixes. It is not one
causal graph through generation time, and edges from different target positions must never be
spliced into a purported within-graph path.

The atlas has two frozen mappings plus their evidence:

```text
(layer, raw MLP neuron, polarity) -> cluster ID -> semantic label or abstention
```

The cluster assignment is the primary state. A generated label is a bounded interpretation of a
cluster, not an intrinsic neuron ground truth.

## Trace-family names

The project will use these names consistently:

- **T5:** upstream ADAG's default top-five trace. At each prefix, select the model's five
  highest-logit next tokens, sum those five selected logits into one scalar objective, and build
  one graph using upstream-style nonlinear pruning. The initial percentage threshold is `0.005`.
- **CU5:** the fork's production candidate-union method: independently trace the observed token
  and model top five, take the exact topology union, and rescore on the fixed union. Realized
  width is five or six.

T5 is the primary trace family for this study. CU5 is not presumed superior and is not a primary
input. Historical artifact paths and scheduler names that called CU5 "T5" retain their immutable
names, but all new prose and manifests must record their actual semantics.

Strict T5 does not guarantee that the stored teacher-forced token `y[t]` is one of the five
targets. Every target record must therefore retain the observed token, its model rank, the five
selected token IDs/logits, and whether the observed token was included. The trajectory describes
the model's top-five continuation objective at each stored prefix, not necessarily attribution to
the literal stored next token.

## Reconciled starting state

- The prior Qwen3-4B-Instruct traces are dominated by hinting cases and use width-one or CU5
  semantics. They do not instantiate the new bottleneck/T5 atlas.
- The existing provenance, atomic-artifact, resume, signed-basis, clustering, labeling, and
  response-time multiplex infrastructure is reusable after it is bound to a new model and corpus.
- The previous cluster states and labels are not reusable as this atlas because raw-neuron
  identities and corpus-fitted cluster meanings are model- and corpus-specific.
- The historical CU5 array job `14543695` was held on 2026-08-11. At the last successful scheduler
  check, four already-running tasks were allowed to finish and 515 pending tasks were held. This
  is an operational snapshot, not a current-state guarantee. Do not restart or repurpose that job
  as part of this plan.

## BonaFide data interpretation

`BonaFide.csv` rows are annotations, not unique completions. Completion identity must be deduplicated
on at least `(target_model, prompt, cot)` while retaining all source row IDs and exact spans.

The initial process pool is:

- `src_type=complex`: outright tasks with required intermediate computations;
- `src_type=graph`: outright stateful graph tasks with required transitions;
- `src_type=complex_hints`: computational diversionary hints, kept as a distinct later stratum.

`FAITHFUL_STEP` is a localized positive statement that a span verbalizes a required computation.
`UNFAITHFUL_COT` may represent a claimed omission and has no localized absent-process span. Manual
audit found several apparent missing-step labels whose supposedly missing arithmetic is visibly
present in the CoT. Therefore:

- positive required-step spans are usable after deterministic process-ledger audit;
- absence of `FAITHFUL_COT` is not evidence that a completion is globally unfaithful;
- existing missing-step labels are not accepted as negative controls without re-audit;
- the initial study may use faithful positive cases only.

## Step 0: select and freeze the model, processes, and corpus

Step 0 ends only when a versioned, hashed selection bundle has passed the gates below. No atlas or
dense production trace starts before that freeze.

### 0A. Model selection

The relevant current shortlist is:

| Model | Outright process completions (`complex` + `graph`) | Length | Trace readiness |
| --- | ---: | --- | --- |
| `Qwen/Qwen3-4B-Thinking-2507` | 43 currently available outright completions; 55 after adding `complex_hints`, spanning 46 process-pool questions; 8 outright completions at or below 2,500 response tokens | Exact outright median 4,027 response tokens; minimum 1,081 | Cached exact revision; dense Qwen3 tracer already exercised |
| `meta-llama/Llama-3.3-70B-Instruct` | 21 | Exact median 508 response tokens; minimum 248 | Llama architecture is conceptually supported, but 70B dense tracing needs a separate multi-GPU feasibility proof and uncached weights |
| `allenai/Olmo-3-7B-Instruct` | 16 | About 314 whitespace words at the median | Attractive size/length, but the current tracer does not dispatch OLMo3 modules and the checkpoint is not cached |
| `Qwen/Qwen3-30B-A3B-Instruct-2507` | 24 | About 383 whitespace words at the median | MoE routing requires a separate neuron-identity and tracing design; not an initial candidate |

**Proposed primary:** `Qwen/Qwen3-4B-Thinking-2507`, cached revision
`768f209d9ea81521153ed38c47d515654e938aea`.

All 55 of its current process-pool completions have at least one localized `FAITHFUL_STEP`; none
has a reliable `UNFAITHFUL_STEP` for the required bottleneck itself. This is sufficient for the
initial positive-control discovery phase.

This recommendation prioritizes reaching interpretable full trajectories with the existing
tracer. It is not frozen until a bounded T5 compatibility/resource smoke test passes. If the
shortest full Qwen trajectories are infeasible, reopen selection rather than silently truncating
the scientific cases. The next alternatives are an OLMo3-7B port or a Llama-70B multi-GPU pilot;
each is a new engineering decision.

### 0B. Initial full-completion candidates

For the proposed Qwen Thinking model, begin with faithful outright cases:

| Role | Question/process | BonaFide annotation IDs | Response tokens | Reason |
| --- | --- | --- | ---: | --- |
| Dense discovery A | Collatz from 64; six repeated halving transitions | `f2d81f1889e8f0df` | 1,081 | Clean repeated homologous computation within one trajectory |
| Dense discovery B | Nested modular arithmetic, question 1 | `d788e356498626c3`, `7012fbabf46fcc16` | 1,185 | Multiple localized required operations in a short, richer process |
| Dense reserve A | Nested modular arithmetic, question 20 | `2f95917ead26521c` | 1,107 | Same process family with changed operands; initial transport check |
| Label-audit only | Nested modular arithmetic, question 23 | `49e4685ad33f17ff`, `060e04986919d95f` | 1,427 | The CSV claims an omission despite explicit arithmetic; not a valid negative |

The first hand-exploration pass should focus on discovery A and B as complete responses. Reserve A
must not influence the first witness story or thresholds. The label-audit case is evidence for
the Step-0 audit requirement and must not be presented as unfaithful.

### 0C. Process ledger

Before graph inspection, create a machine-readable process ledger for every dense case containing:

- process ID and family;
- exact prompt, response, model, tokenizer, and source annotation IDs;
- model answer, reference answer, and independently checked correctness;
- required inputs/state and deterministic intermediate transitions;
- why the task construction makes each transition necessary;
- acceptable algorithmic or verbal variants;
- audited positive verbalization spans and token positions;
- first answer-commitment position;
- ambiguity, alternative-strategy, and annotation-audit notes;
- all text, tokenization, and source-file hashes.

This ledger defines what BonaFide tells us. It does not prescribe which neurons, clusters, paths,
or exact time points must realize the process.

### 0D. Broad atlas corpus

Generate new completions from the selected model on the same or closely related outright-task
distribution. These completions provide task-conditional context for clustering and labeling;
they do not need BonaFide faithful/unfaithful labels.

The primary atlas-fit distribution is outright tasks only (`complex` and `graph`). This already
provides variation across arithmetic, number theory, cryptography, text processing, logical tasks,
and multiple graph-traversal skins while keeping every context tied to a known bottleneck-task
construction. Do not add generic hinting or unrelated background merely to increase topic
diversity. `complex_hints`, ordinary hinting, and unrelated controls may later be projected through
the frozen atlas as out-of-distribution or specificity checks; they do not fit the first atlas.

Freeze:

- exact base-question inventory and family split;
- model and tokenizer revision;
- exact rendered system/user conversation and chat template;
- generation/decoding settings and random seeds;
- per-attempt store, accepted response text, token counts, and hashes;
- length and family selection rules;
- human region annotations, token-target conversion policy, and surface-reference selectors;
- exclusion and failure rules.

For newly generated Thinking-model completions, the authoritative response is the exact serialized
assistant token sequence captured by generation, including thinking delimiters and the final-answer
segment. Do not reconstruct those inputs from parsed columns when raw completion token IDs exist.

The three published historical dense cases are a documented exception. The public CSV retained
the interior CoT and parsed answer but discarded raw assistant serialization, token IDs, row-level
system provenance, seed, finish reason, and runtime manifest. The released BonaFide generator's
initial commit nevertheless recovers a strong intended conversation contract: the exact answer-only
system prompt, Qwen Thinking generation prefix, and decoding defaults. Version 1 therefore freezes
these cases as **historical reasoning-segment reconstructions**, not byte-identical recovered runs.
It renders the released system/user generation prefix ending in `<think>\n`, appends the stored CoT
body as its raw continuation, derives and hashes the token IDs, and traces only those CoT tokens.
Unknown closing-tag/JSON material follows the reasoning and cannot causally affect earlier
autoregressive CoT predictions. Final-answer/delimiter tracing is excluded for these historical
cases. Remaining uncertainty about an unpreserved row-level system message, leading stripped
whitespace, and the exact historical revision remains attached to every artifact and claim.

Generation/tracing consistency is necessary but does not recover missing historical provenance.
For newly generated responses, trace the exact system/user prefix and assistant token sequence used
at generation. For historical dense responses, first recover the original rendered context. If it
cannot be recovered, generate and audit fresh dense completions under the new frozen conversation
contract rather than silently replaying old text under a different system message.

Broad context must be prompt- and response-balanced. Hundreds of targets from one long dense
response must not dominate a cluster merely because adjacent tokens are repeated measurements.
Balance alone is insufficient, however: a uniform temporal sample of a full response overrepresents
whitespace, punctuation, delimiters, discourse glue, and other token-prediction structure that is
not the intended process domain. Increasing the cluster count can separate some of that structure,
but cannot repair a surface-form-dominated fit distribution.

The primary atlas is therefore a **process-targeted atlas**, closer to the paper's task-specific
targeting than to an arbitrary sample of response tokens. Before tracing, annotate purpose-selected
regions in the raw broad and dense completions without access to traces, graphs, neuron identities,
cluster states, or generated labels. The annotation interface must support at least:

1. final-answer span onsets;
2. intermediate result, transformed-value, or extracted-value span onsets;
3. graph/state-transition outputs and running-state updates;
4. analogous cipher, logical-state, and text-processing outputs for non-arithmetic outright tasks;
5. explicit surface-reference regions such as punctuation, whitespace/control delimiters,
   function words, and discourse filler.

The annotation system must not collapse these judgments into one flat class. It records
independent, reviewable axes at both event/span and token level:

- semantic domain and task family;
- operation or state transition, allowing multiple values and `unknown`;
- process role, including input/operand, operation cue, intermediate result, state update,
  verification/correction, and final result;
- discourse role and answer-commitment status;
- representation type, surface form, and serialization/syntactic role;
- token position within an annotated event: onset, interior, terminal, or following separator;
- correctness or attempted/ambiguous status, kept separate from semantic type;
- suggestion source, review state, annotator revision, and ambiguity notes.

The plan distinguishes two graph-blind annotation products with different freeze times and
different scientific roles:

1. **Coarse selection layer.** This is a sampling instrument only. Each text unit receives exactly
   one coarse sampling tag: `active_task_work`, `evaluation_or_revision`,
   `intermediate_commitment`, `final_answer`, `other_semantic_text`, `surface_or_control`, or
   `uncertain`. The artifact also retains event/unit, response, prompt, token, confidence, and
   annotation provenance. These tags enrich the first tracing wave and remain attached only as
   selection provenance; they do not define adequacy strata, motif classes, semantic ground truth,
   or scientific endpoints. This layer, its audit, and its target-conversion and sampling policy
   freeze before the first-wave target manifest or any scientific trace.
2. **Descriptive annotation layer.** This retains and refines the richer ontology above, including
   process signatures, detailed roles, operation/state types, modality, correctness/status, and
   event-relative structure. It may be refined in new graph-blind versions while tracing runs. It
   must never inspect graphs, neurons, clusters, generated ADAG descriptions, or adequacy outcomes.
   Every version remains linked to the same immutable response and token identities. The exact
   descriptive version and fields used by an adequacy measurement freeze before that measurement
   opens cluster assignments or labels.

The frozen v9 automatic draft is evidence available to both annotation efforts, not reviewed
semantic truth and not the coarse schema itself. Exhaustive
manual painting of every rich axis over all 842,007 tokens is not a prerequisite for tracing.
The coarse tags may use direct structured LLM API calls followed by graph-blind human audit of
balanced examples, conflicts, rare tags, boundary cases, and low-confidence cases. Later
descriptive versions may add information to already traced targets, but they cannot retroactively
change why a target entered the frozen trace bank or use coarse tags as semantic evidence.
Luna is the expected low-cost initial annotator candidate, but the exact provider/model snapshot,
API surface, and decoding contract freeze only after a small qualification pass.

For the initial coarse annotator, prefer a narrow reproducible pipeline over a general agent:

1. deterministically recover exact token offsets, serialization boundaries, surface/control
   regions, and bounded sentence/clause/line units;
2. issue direct structured API calls that ask for only the exclusive coarse tag on bounded units or
   windows, with task/prompt context and limited neighboring text;
3. run programmatic gap, overlap, boundary, and low-confidence repair calls where needed; and
4. preserve the exact model identity, prompts, request/response bodies, unit identities,
   confidence, validation outcome, and final proposal.

The refined semantic task is deliberately tested under increasing annotation machinery rather than
assuming an agentic harness is required. First measure raw structured API performance with one
narrow axis or chunk per call. Then, if available and reproducibly qualified, test managed
multi-turn or tool-calling API orchestration that incrementally writes and validates an annotation
ledger. Build a Pi-based agentic annotator with frozen tools, prompts, iteration limits, stopping
conditions, and full edit provenance only if the simpler approaches are insufficient. All three
conditions remain graph-blind, preserve uncertainty, and require the same held-out human audit;
none may inspect traces or ADAG outputs.

Automatic annotation may classify observable form broadly but is not semantic ground truth. Use
high-precision regular expressions, lexical rules, or another deterministic local classifier to
suggest numbers, operators, equations, JSON structure, delimiters, punctuation, whitespace,
answer markers, and explicit operation words. A human reviews those suggestions and supplies or
corrects high-impact cases. Structured LLM annotation may propose semantic-heavy regimes, process
families, roles, and event boundaries at scale; preserve the model, prompt, response, confidence,
and raw proposal. Human review is required on balanced samples, conflicts, rare strata,
low-confidence cases, and any manually selected anchors, but exhaustive rich-axis correction is not
a pre-trace requirement. Preserve proposal and reviewed values separately; permit `unknown`,
`ambiguous`, and multi-label annotations rather than forcing complete semantic coverage. Audit a
stratified sample before accepting each draft so avoidable token-boundary, semantic shortcut,
sign, decimal, Markdown, JSON, delimiter, and operation-word errors are corrected before use.

Manual selection is permitted for this exploratory study. Annotators may use the frozen task
schema, raw prompt/completion, BonaFide annotations, and audited process ledger. Broad attempted
computations remain eligible whether their values are correct or incorrect. Annotation may not use
ADAG evidence or be revised because a preferred graph, label, or witness appears.

Painting a region does not automatically admit every token in that region to atlas fitting. Freeze
a deterministic conversion policy such as result onset, value-token set, or a fixed stratified
quota per event. If more eligible positions exist than the response quota, select among them
deterministically across the response. Do not fill a process-target shortfall with punctuation or
filler; apply a frozen retry, exclusion, or lower-yield rule instead.

Every annotated or derived target receives exactly one usage class:

- **`process_atlas_fit`:** eligible to fit and label the primary process atlas;
- **`surface_reference`:** explicit surface-form comparison positions, excluded from all atlas
  fitting, hyperparameter choice, exemplars, and labels for the primary process atlas. They may
  enter only a separately named, predeclared nuisance-contamination sensitivity after the primary
  state and hyperparameters freeze;
- **`trajectory_only`:** the unselected or unclassified remainder, which is not assumed to be a
  negative example of computation.

The complement of the painted process regions is never automatically called nuisance. It may
contain unspoken, delayed, or unrecognized computation. Surface references are likewise comparison
contexts rather than certified no-computation negatives. A uniformly stratified all-token atlas may
be built later as an explicitly named composition-sensitivity analysis, not as the primary atlas.

Use a small token-aware local annotation page for this work. It must render the exact stored response
and tokenizer boundaries, allow span painting and usage/event assignment, and export a reviewable
manifest containing at least response ID, character span, token span, usage class, process/event ID,
target-conversion policy, annotator, revision, source-response hash, and tokenizer hash. The page
must not load graph or clustering artifacts. Freeze and hash the reviewed coarse selection manifest
before production tracing or atlas fitting. Richer descriptive reviews are separately versioned and
need not all finish before tracing. At selection freeze, every dense position not explicitly
assigned to `process_atlas_fit` or `surface_reference` is recorded as `trajectory_only` rather than
omitted.

The initial trace budget is deliberately not frozen in this draft. First inventory annotated
process-target yield and measure T5 cost on the selected model, then freeze the response-balancing
rule, process quotas, nested process-only enlargement tiers if needed, and surface-reference quotas
in a new corpus manifest. For the adequacy study, trace the selected process targets and frozen
surface references across broad responses. Dense-discovery responses may contribute only
predeclared adequacy targets and small fixed position-sensitivity neighborhoods at this stage. Full
all-token dense trajectories are deferred until a robust adequacy verdict and the separate Step-2
motif-study freeze.

The annotation-yield pilot must resolve response balancing before the manifest freezes. Prefer a
common `Q_process` supported by every admitted response, with a frozen retry or exclusion rule for
shortfalls. If that is scientifically too restrictive, use per-response `Q_process,i` only in a
separately specified balancing method; do not silently give high-yield responses more upstream
contexts. Until this gate resolves, the atlas-fit context count is
`sum_i Q_process,i`, not a fixed product.

The source contains 48 distinct outright-task prompts targeted for new generation: 22 `complex`
and 26 `graph`. This prompt inventory is distinct from the 43 currently available Qwen Thinking
outright completions reported in the model table.
Hold out the modular reserve-A prompt completely, leaving 47 atlas-fit prompt cells. Allocate four
admitted response slots to every fit prompt. For the two dense-discovery prompts, the audited dense
response occupies one slot and three new broad completions fill the others. Generate four broad
completions for each of the other 45 prompts, including fresh responses to the label-audit prompt;
the contradictory historical completion itself is not admitted as a negative. This gives 186 new
broad responses plus two dense responses. The final atlas-fit context count remains
`sum_i Q_process,i` until the annotation-yield gate freezes a common quota or an explicit balancing
method; it is not provisionally fixed at 20 arbitrary response positions.

Generate deterministic attempts for every required slot and retain every raw attempt. Broad
eligibility may use only frozen requirements such as natural termination, nondegeneration, and a
valid Thinking/final-answer serialization; do not select broad responses by correctness,
faithfulness, perceived interestingness, or an attractive response length. Freeze a
maximum-attempt/yield failure rule before generation so prompts with poor mechanical yield cannot
be handled ad hoc.

The initial draft described a resource-derived **completion-token cap**. The Step-0 landmark run
showed that this is not the right scientific resource variable: T5 cost is determined by the exact
teacher-forced context at a selected target (rendered prompt plus preceding assistant tokens), not
by total completion length alone. A response-only cutoff would also alter the broad corpus sharply:
only 22 of the 43 historical outright Thinking completions are at or below 4,096 tokens. The raw
generation campaign therefore retains the released historical `max_tokens=32768` envelope and
every predeclared attempt. Mechanical response selection requires natural termination and valid,
nondegenerate serialization but does not impose a traceability cutoff on the whole response.

After graph-blind region annotation converts spans to candidate target positions, freeze a
**total target-context token gate** and apply it to those positions. The first valid resource tier
is the largest total context demonstrated by the Step-0 landmark run; larger tiers require their
own strict-T5 resource trace before use. If a prompt's rendered prefix already exceeds a passing
tier, its targets are not silently admitted or its completion selectively replaced: either a new
resource tier must pass or a new corpus version must predeclare the prompt-cell exclusion and
rebalance rule.

### 0E. Technical and freeze gates

Run only bounded, disposable checks before freezing:

1. load the exact offline checkpoint and verify model/tokenizer/chat-template identity;
2. trace one short-prefix T5 target and confirm upstream summed-top-five semantics;
3. trace a few early/middle/late landmarks from the shortest dense completion;
4. measure wall time, peak GPU memory, graph size, and non-finite behavior;
5. verify atomic save, reload, provenance, and deterministic target selection;
6. estimate the complete broad-plus-dense workload without launching it.

Step 0 freezes a selection bundle containing the model identity, process ledger, broad generation
corpus, dense discovery/reserve roles, T5 configuration, corpus weights, code revision, resource
plan, failure policy, and content hashes. Changes after that point require a new version.

The version-1 input freeze is materialized by
`scripts/bonafide/manifests/qwen3_4b_thinking_process_witness_step0_v1.json` (SHA-256
`d00aa083474d34fdaf0936df5705d00a9192705dbab9e7a9629f95aaf9effc34`). It contains all 48
outright prompt cells, the 47/1 fit-reserve split, 186 deterministic new broad request slots, the
three historical dense reasoning replays, process ledgers, token identities, and the historical
conversation/decoding evidence. Its original status is `inputs_frozen_resource_gate_pending`. The
subsequent strict-T5 landmark run passed at total input-context lengths through 1,268 tokens. It
establishes a first traceable context tier, not a response-length cap and not permission to
extrapolate to the longest broad prompt. The bounded source and strict-T5 smoke
manifests have SHA-256 identities
`e5b85c3463d8a325b31e515dd6c2c6150f883d1cddfec1eb92a6e81e42ac5e94` and
`d125da744b0ad25f2e907424bc30ee67b6b93c693b3ac4010f8256915eb41259`, respectively.

The four Collatz landmarks at response positions 103, 130, 580, and 1,079 all completed with valid
content-addressed artifacts. The three scaling positions used 319, 769, and 1,268 total input
tokens; their T5 trace times were 53.4, 98.0, and 229.2 seconds, and their peak reserved CUDA
memory was 16.1, 29.5, and 49.0 GB on an 80-GB A100. The last retained 36.1 GB of device headroom.
These measurements support the 1,268-token context tier only. Broad historical-format prompt
prefixes range from 175 to 2,631 tokens, so at least one additional long-prompt resource gate is
required before every frozen prompt cell can contribute atlas-fit targets.

## Step 1: run the standalone global-atlas adequacy study

Step 1 has three concurrent work lanes after the coarse selection manifest freezes:

```text
T5 trace production  ||  graph-blind descriptive annotation  ||  synthetic ADAG tests
```

Tracing is expected to be the long pole. Synthetic similarity/clustering tests and descriptive
label refinement should use that time; neither waits for all traces, and neither may inspect partial
scientific cluster outcomes. Real-data atlas fitting begins only when the required trace tranche and
the exact descriptive annotation version for that measurement are both frozen.

### 1A. Freeze the coarse selection layer and first-wave trace bank

Assign each bounded text unit one exclusive coarse sampling tag:

- **`active_task_work`:** actively carries out a task-relevant transformation, traversal,
  retrieval, selection, comparison, count, calculation, or state update;
- **`evaluation_or_revision`:** checks, verifies, backtracks, corrects, or reconsiders work;
- **`intermediate_commitment`:** explicitly settles or reports a non-final result/state;
- **`final_answer`:** states or serializes the response's final commitment;
- **`other_semantic_text`:** planning, explanation, restatement, task-description text, commentary,
  or a process mention that is not currently executed;
- **`surface_or_control`:** formatting, punctuation-only, whitespace/control, tags, JSON syntax, or
  other structural material; and
- **`uncertain`:** the exclusive fallback when the unit or its boundary cannot be assigned
  responsibly.

These are intentionally broader than process families and roles. For example, following a graph
edge and dividing two values are both `active_task_work`; their distinction belongs only to the
descriptive layer. A verification sentence containing division remains `evaluation_or_revision`.

Construct the target bank from a frozen mixture of proposal routes rather than stratified-uniform
sampling:

1. a process-enriched route favors `active_task_work` and observable anchors such as values,
   entities, operators, relation symbols, and explicit task-process cues;
2. an evaluation/commitment route gives rare checks, revisions, intermediate commitments, and final
   answers adequate coverage;
3. a diversity route limits domination by long responses or frequent task families and balances
   prompt, response, coarse unit, position, and span length;
4. a uniform reserve samples ordinary semantic and surface/control positions without enrichment;
   and
5. an uncertainty reserve samples ambiguous units that may hide unusual process forms.

Within a selected unit, sample a bounded combination of observable anchors, unit onset/terminal,
interior positions, and a small local halo rather than either tracing the whole span or choosing one
uniform token. Freeze all proposal routes, caps, random seeds, inclusion probabilities, and target
weights after the coarse-yield census. Preserve hierarchical prompt/response/unit blocking so
nearby token graphs are not treated as independent experimental evidence. The ordered manifest
should prioritize useful early coverage without turning partial completion into a new selection
decision.

The working planning envelope is approximately 30,000–40,000 independent T5 targets with a goal of
finishing production within about one week. This is neither a frozen count nor launch authority.
The exact count follows the coarse-label yield inventory, target-context resource tiers, achievable
array throughput, per-panel minimum support, and failure/resume policy. Reduce or expand only through
the frozen nested target-bank tiers, never by inspecting partial scientific outcomes.

The coarse tags define selection only. They do not enter the ADAG similarity or clustering
objective, define adequacy panels or motif classes, condition scientific endpoint estimates, or
assert that the model internally performed the tagged work. Later analyses use only separately
frozen descriptive annotations. The coarse tag and proposal route remain available solely to audit
how the trace sample was obtained.

After descriptive refinement, audit the already traced bank against the refined label inventory.
If important graph-blind semantic cells are too sparse or distorted, a smaller second wave is
permitted. Its deficit rule, targets, and manifest must freeze before any graph, cluster, or ADAG
label outcome is inspected; it may use refined textual annotations but never preliminary ADAG
results. Record wave identity and selection probability for every target and never silently append
wave two to wave one. If a deficit is discovered only after scientific outcomes are opened, any
new targets are exploratory follow-up data under a new protocol.

### 1B. Trace the atlas-fit corpus

- Trace the frozen `process_atlas_fit` targets and `surface_reference` panels selected for the
  adequacy study with T5; keep surface-reference artifacts sealed from primary fitting, primary
  hyperparameter choice, and labeling.
- Permit dense discovery A and B to contribute only targets that satisfy a frozen adequacy-panel
  rule. Do not trace every response token in Step 1.
- Before tracing, freeze deterministic `process_atlas_fit` panels under the chosen response-
  balancing rule, nested process-only enlargement tiers, and a separately marked surface-reference
  panel.
- Preserve one independent graph artifact per response position.
- Do not merge target-local graphs in the tracing pipeline.
- Record observed-token rank/membership at every position.

Strict T5 remains the primary scientific trace family. After one graph-blind union target bank is
frozen, an exact-target CU5 sidecar may trace the same bank in parallel under a separate manifest,
artifact namespace, and resource plan. CU5 output is retained as method-development information
and cannot be used to claim that ADAG needs modification, select the T5 quality endpoint, or repair
a disappointing T5 atlas post hoc. Any T5/CU5 comparison opens only after its comparison protocol
and the T5 cluster/label-quality measurements are frozen.

The candidate atlases follow upstream's uniform-over-context aggregation, but their contexts are the
frozen `process_atlas_fit` panels rather than uniform all-token samples. Use the response-balancing
rule frozen at the annotation-yield gate. Audit projection coverage over the held-out adequacy
panels without examining candidate witnesses. Predeclare assignment gates using signed-node
coverage, attribution-mass coverage, process-ledger landmark coverage, and cluster stability. If
the smallest process-only panel fails those gates, advance to the next already frozen process-only
tier. Choose the smallest passing tier; if none passes, stop or version a new method rather than
silently admitting every token.

This response-balanced subset is preferable to custom weights for the primary study because the
upstream implementation uniformly averages contexts and has no family/response-weight interface.
A separately named hierarchically weighted all-dense atlas may be implemented later as a method
sensitivity, but it is not upstream-equivalent T5 ADAG. Cluster-label evidence must likewise use
frozen prompt/response exemplar caps. Full dense traces are produced only in conditional Step 2.
Surface-reference traces are opened for projection and comparison only after each candidate atlas
state freezes. A separately predeclared nuisance-contamination fit may then deliberately add them
at matched budget to measure how punctuation, filler, or serialization contexts alter clustering;
it cannot replace, relabel, or tune the primary process-only state.

While this production lane runs, continue graph-blind descriptive annotation in immutable versions.
In parallel, execute the synthetic ADAG battery in Section 1C. Neither activity changes the frozen
trace target membership.

### 1C. Run synthetic ADAG mechanism tests concurrently

Before interpreting real clusters, test the exact similarity and clustering implementations on
engineered attribution/contribution profiles and co-occurrence masks with known structure. At
minimum include:

1. a stable monosemantic block control;
2. high `A-B` similarity on shared contexts plus a distinct `A`-only regime;
3. disjoint `A-B` and `B-C` context families with no direct `A-C` evidence;
4. many positive co-occurrences plus a rare severe disagreement regime;
5. equal mean affinity supported by one versus many contexts;
6. balanced versus frequency-skewed mixtures; and
7. token-position-specific roles before and after the default token-collapse operation.

For every case retain the complete per-context similarities, overlap counts, one-sided occurrence,
affinity matrix, spectral embedding, assignments, and sensitivity to `k`, seed, and mixture
proportion. Synthetic outcomes locate structural behavior of the algorithm; they do not estimate
how often that behavior occurs in the BonaFide-derived corpus.

Run the frozen paper-faithful and released-code similarity conditions plus the predeclared spectral,
Leiden, and concatenated-profile clustering comparators on the same compatible synthetic fixtures.
This is required to distinguish failures caused by pair construction from failures caused by the
global partition. Do not expand the comparator grid after seeing a fixture outcome.

### 1D. Fit matched candidate atlases label-blind

Construct input-attribution and output-contribution profiles using only the frozen atlas-fit
corpus. Select normalization, cluster state, and stability settings without looking for attractive
process labels or opening the dense reserve case.

The primary algorithmic reference must be named precisely. The paper defines **paper-faithful
multi-view spectral ADAG** as: clamp and harmonic-fuse attribution/contribution cosine similarity
inside each co-occurring context, uniformly average those fused scores across contexts, then apply
spectral clustering to the non-negative affinity. The released implementation differs: its
`combine="harmonic"` mode averages each view across contexts before harmonic fusion, while the
callable default is arithmetic `combine="mean"`. Record and test both; do not call either one the
other. A paper-faithful implementation requires its own parity fixtures before scientific use.

Treat clustering implementation as a predeclared diagnostic axis rather than an unrestricted model
search. The repository already exposes:

- multi-view spectral clustering with normalized or unnormalized affinity, non-negative clipping
  or linear shift, arithmetic or released-code harmonic fusion, and a `k` sweep;
- Leiden community detection on a k-nearest-neighbor sparsification of the same multi-view
  affinity; and
- concatenated-profile baselines using agglomerative, K-means, bisecting K-means, or RBF spectral
  clustering.

The adequacy report must keep paper-faithful and released-code multi-view spectral ADAG as distinct
unmodified conditions. Before opening outcomes, freeze whether the gate requires both, designates
one primary with the other as a fidelity comparator, or permits a split verdict; do not choose this
after seeing which condition looks better. Use Leiden and the concatenated-profile clusterers to
localize whether a failure comes from pair construction, global spectral partition, hard fixed-`k`
assignment, or the underlying profiles; they may not be promoted merely because their labels look
cleaner. Optional absence-aware importance profiles, hierarchical weights, position-preserving
identities, robust aggregation, or support thresholds are explicitly modified methods and enter
only as named repair/sensitivity conditions after the unmodified conditions are recorded.

To evaluate global-atlas adequacy, fit an equal-budget composition series from the same saved trace
bank. The exact viable cells and counts freeze after the graph-blind annotation-yield inventory,
but before any cluster or label output is opened. The intended progression is:

1. one well-supported process signature at one matched role;
2. the same process signature across roles or representations;
3. multiple process signatures at one matched role;
4. balanced process families and roles within one well-supported domain or task family;
5. balanced process families, roles, and domains across the outright-process distribution;
6. same-surface/different-role and same-role/different-surface process-target mixtures; and
7. a separately named nuisance-contamination sensitivity that injects actual
   `surface_reference` contexts only after the primary process-only state freezes.

Hold total contexts and prompt/response contribution fixed within each direct comparison. Sample
hierarchically by prompt, response, process event, then token; adjacent subtokens from one event are
not independent contexts. All sibling completions of one prompt stay in the same evaluation fold.
Count prompt-level support and use prompt-held-out projection for semantic evaluation.

Use a predeclared cluster-count sweep rather than assuming the paper's task-specific value transfers
to this broader outright-task corpus. Increasing `k` is a complementary resolution choice, not a
substitute for process-targeted sampling. Freeze the candidate grid after the observed signed-basis
and support audit, then choose the primary `k` label-blind using resampling stability, minimum
cluster support, and profile coherence within the process-target fit corpus. Retain one predeclared
higher-resolution fit as a sensitivity view if support permits; do not use surface references,
manually attractive math labels, or candidate witnesses to choose `k`.

Report task-family and process-family support for every cluster. If support permits, predeclare one
single-domain or single-family refit of the identical saved process-target traces as a
corpus-composition sensitivity. It is not a replacement primary atlas and may not be promoted
because it produces more coherent labels; it tests whether legitimate process diversity is
consuming the available clustering resolution.

Retain:

- signed-basis identities and polarity;
- cluster assignments and profile prototypes;
- normalization and missing-support semantics;
- seeds, hyperparameters, stability results, and corpus weights;
- exact exemplars and their response/family provenance;
- unseen-feature application rule.

### 1E. Label candidate states

Generate cluster descriptions from discovery evidence, retain abstentions and uncertainty, and
audit proposed labels against exact profile/graph exemplars. Candidate labels are not yet a witness
vocabulary. Map them, blind to held-out cluster occurrences, to the frozen annotation ontology or
abstention and test whether their specificity is supported on held-out prompts.

### 1F. Diagnose global-atlas aliasing and label quality

ADAG gives each signed `(layer, neuron, polarity)` identity one global cluster assignment. More
clusters can regroup identities but cannot give one neuron different memberships in different
contexts. Before dense witness inspection, test whether that representation supports trustworthy
task-conditional labels when operations, roles, and domains are mixed.

Call this the **global-atlas adequacy test**. Context-dependent neuron reuse is one possible cause
of failure, but the gate does not claim to measure intrinsic neuron polysemanticity. It measures
whether the fitted global atlas remains useful for this process corpus.

The primary risk is **context-conditioned role aliasing under a global hard partition**, not
polysemanticity in the abstract. Separate and report these mechanisms:

1. **Pairwise co-occurrence censoring:** `A-B` similarity is computed only where both identities
   survive pruning; `A`-only and `B`-only contexts do not directly disconfirm the pair.
2. **Bridge-induced merging:** positive `A-B` and `B-C` affinity from disjoint context families can
   connect `A` and `C` in a global partition without direct `A-C` evidence.
3. **Mean masking:** a high mean can conceal low-support, high-variance, minority-disagreement, or
   family-reversed pair behavior.
4. **Missing-versus-incompatible ambiguity:** zero affinity can mean no overlap or observed
   incompatibility; the partition does not preserve that distinction.
5. **Hard-assignment and resolution failure:** every retained signed identity receives one cluster
   at the chosen resolution even when its conditional neighborhood changes across strata.
6. **Position and mixture aliasing:** default token collapse and unequal context support may merge
   position-specific roles or let frequent prompts, responses, and events dominate.
7. **Description dilution:** cluster construction may be coherent while one aggregate natural-
   language label is broad, dominant-regime-specific, or overconfident; clustering and labeling
   failure are separate verdict dimensions.

Retain raw evidence needed to test these mechanisms instead of relying on the final silhouette or
label score: pair overlap support, one-sided occurrence, the full per-context similarity
distribution, family-conditioned means and lower tails, context-conditioned nearest-neighbor
turnover, same-cluster pairs without direct evidence, bridge-node dependence, prompt-blocked
cluster stability, unknown signed-node coverage, and unknown attribution mass. Text annotations
define evaluation strata; a neuron appearing under two semantic labels is not itself proof of two
functions unless its conditional attribution/contribution or relational evidence also differs.

The adequacy trace bank is selected for this gate only. It may later supply reusable target-local
graphs, but it is not a motif dataset and is not optimized around a conjectured witness shape.
Trace selected process contexts and matched references needed by the panels below. Do not produce
full dense trajectories merely to anticipate Step 2. Small fixed event-relative neighborhoods are
allowed only when predeclared as position-sensitivity measurements for the adequacy gate.

#### Ordered adequacy panels and their inferential jobs

The panels form an evidence ladder. Each panel answers a different question and must not be pooled
into one undifferentiated score.

0. **Repeatability floor — how much movement occurs without changing the semantic problem?**
   Repeat clustering seeds, same-composition prompt-balanced samples, prompt-blocked resamples,
   frozen repeat traces where available, and label/exemplar resamples. This is the perturbation
   floor against which every later degradation is measured. It does not test semantic recovery.
1. **Best-case homology — can ADAG recover a process when given the cleanest fair conditions?**
   Use one well-supported process signature or tightly related task-required events, balanced over
   independent responses, with minimally varying operands, entities, or states. Fit and evaluate
   on separate examples of that same process. This is a necessary lower-bound feasibility test,
   not evidence of broad generalization.
2. **Natural transport — does the recovered organization survive ordinary task variation?**
   Hold the process family or normalized signature fixed while varying prompt, response, operands,
   entities, state, representation, and task skin. This measures the range over which a process
   description can legitimately transport.
3. **Role-versus-mechanism separation — what semantic distinction is the atlas preserving?**
   Compare the same mechanism in different textual/process roles and different mechanisms in the
   same role. This distinguishes an atlas organized around process mechanism from one organized
   mainly around result position, answer commitment, representation, or discourse role.
4. **Collision stress — how much does heterogeneous process composition damage the global map?**
   Mix distinct process signatures, families, and domains at fixed budget and prompt/response
   balance, especially where surface forms or roles overlap. Excess degradation beyond Panels 0–3
   is evidence that one global mapping cannot represent the required contextual distinctions.
5. **Matched non-process specificity — are apparent process labels actually generic reasoning or
   reporting machinery?** Compare performed/derived process spans with matched planning,
   instruction/restatement, hypothetical work, lookup/copying, verification, correction, answer
   serialization, and nearby unclassified discourse. This estimates semantic false-positive
   behavior; the complement of a labeled process span is not automatically a negative.
6. **Surface-reference specificity — are results explained by token-production form?**
   Project matched punctuation, whitespace, delimiters, JSON syntax, function words, and numeric or
   entity surface forms through the frozen candidate atlases. This measures surface leakage and is
   never an atlas-fit source for the primary process comparisons.

The labels required to build these panels are hierarchical. Retain task-requiredness, event
modality, broad process family, example-specific process signature, process role, representation,
and correctness/status as separate fields. `presented_as_executed_or_derived` is a textual claim,
not evidence that the model performed an internal computation. The frozen v9 automatic draft is a
suggestion source, not reviewed truth; semantic-heavy fields may use an LLM annotator followed by
graph-blind human audit of balanced samples, conflicts, and low-confidence cases.

Establish the natural perturbation floor first using repeated clustering seeds, independently
sampled prompt-balanced panels with identical semantic composition, prompt-blocked resampling,
frozen repeat traces, label/exemplar resampling, matched surface references, and prompt/event-
blocked semantic permutations. Refine and freeze the exact metrics after trace artifacts establish
their available occurrence/profile fields and coverage, but before opening any cluster assignments,
generated labels, composition comparison, or dense trajectory. Candidate measurements include
shared-neuron assignment stability, held-out semantic concentration and prediction, label
calibration and over-specificity, prompt-level support, unassigned attribution mass, and direct
operation/role collision audits for recurring signed neurons.

Interpret the gate as follows:

- **robust:** semantic mixture remains within the predeclared same-composition perturbation margin
  and labels stay calibrated at an appropriately broad or specific ontology level;
- **brittle:** mixture produces excess instability, semantic dilution, or systematically
  over-specific labels, or context-dependent reuse cannot be represented by the global mapping;
- **inconclusive:** prompt replication, top-five observed-token coverage, shared-neuron coverage, or
  semantic/surface separation is insufficient.

This gate evaluates the pruned T5 plus corpus-fitted ADAG representation. It does not establish
intrinsic neuron monosemanticity or polysemanticity. Only after the gate may the project freeze:

```text
signed basis -> cluster
cluster -> label/evidence bundle or abstention
```

Do not change the cluster state because a label is disappointing, and do not relabel after viewing
the dense trajectories. If the gate is brittle, first establish that modification is required and
version a new method; CU5 or a context-conditional representation may motivate later engineering
but is not an automatic replacement. The complete passing, frozen object is the **atlas**.

### 1G. Adequacy decision

Conclude Step 1 with exactly one verdict:

- **robust:** proceed to a separately designed motif/witness study; reuse adequate Step-1 traces
  where they meet its later frozen inclusion rules, and add labels or traces only under a new
  manifest;
- **brittle:** stop the witness study under this atlas version and localize the failure before
  choosing a response. Record whether it is controllable by declared corpus conditioning or
  multiple purpose-built atlases, repairable by support-aware/robust aggregation, soft or
  context-conditional membership, a different clustering backend, or attributable to an
  inadequate underlying trace/profile representation. Every repair is a separately versioned
  method that reruns the same synthetic and real adequacy battery. If the required distinctions
  are absent even from favorable raw profiles, compare a different representation or tracing
  method rather than tuning ADAG labels;
- **inconclusive:** collect only the additional support needed to resolve the adequacy gate.

Do not inspect dense witness trajectories, choose motif representations, mine recurrent subgraphs,
or tune graph-similarity thresholds before recording this verdict.

## Step 2, conditional on robust adequacy: construct motif-study data and dense trajectories

Step 2 is a new, separately frozen experimental phase. First use the adequacy results to specify
which process families, event windows, matched controls, annotations, and additional traces the
motif study requires. The Step-1 adequacy bank may be reused without retracing exact target
positions, but its existence does not determine the Step-2 sampling distribution.

The initial motif hypothesis may treat a candidate as a connected attributed subgraph of labeled
supernodes that recurs across target-local graphs. Connectedness, the graph representation,
cluster-alignment method, similarity function, event-level aggregation, and thresholds remain
discovery questions until actual adequacy graphs establish what objects are available. Freeze those
choices after discovery and before opening a separately declared reserve. Similarity must be
invariant to graph-node ordering and handle cluster-ID permutation across refits; exact paths must
remain occurrence-continuous inside one target-local graph.

Project every token-position graph in each dense discovery response onto the frozen atlas. Later,
trace and project dense reserve A under the identical T5 and atlas configuration without refitting.

The surface-reference panel is a frozen comparison set, not a second coequal atlas. After the
process atlas freezes, project surface-reference graphs through the same signed-neuron mapping and
report, for each cluster, normalized context prevalence and attribution mass in process versus
surface-reference contexts. Attach this surface-support score to projected dense nodes and
clusters. Neurons absent from the process atlas remain explicitly unassigned; do not force them
into a punctuation ontology or subtract them merely because they also occur in generic token
production.

For each response produce three linked views:

1. exact labeled-plus-unknown graph `G(r, t)` for every dense target position;
2. an ordered token-level timeline of cluster/node/path presence and attribution mass;
3. an event-relative view aligned to audited process verbalizations and answer commitment.

Build an unknown-preserving trajectory viewer rather than relying only on the upstream static graph
rendering. At every response position, show graph size, the observed-token/top-five relationship,
the frozen usage/event annotations, and attribution mass separated into (a) assigned clusters with
semantic labels, (b) assigned clusters whose label abstained, and (c) raw signed neurons absent from
the atlas. The default timeline must include
`process_atlas_fit`, `surface_reference`, and `trajectory_only` positions rather than filtering to
atlas-fit regions. Abstained clusters and out-of-atlas signed neurons may be grouped into distinct,
reversible visual clouds, with stable cluster/raw identities and exact local edges available on
drill-down. Never drop unknown nodes, merge their identities irreversibly, or draw edges between
clouds from different target-local graphs.

Every aggregated cluster path must retain at least one exact target-local, occurrence-continuous
path witness. Similar structures in successive graphs are longitudinal recurrence, not a single
causal path across time.

Record first appearance, peak, recurrence, and last appearance relative to both verbalization and
answer commitment. A witness may precede its verbalization. A structure seen only after the text
has stated the computation is ambiguous because teacher forcing makes that text causally available
to later prefixes.

## Step 3: discover, then define and test a process witness

### 3A. Manual discovery first

Inspect the complete trajectories for dense discovery A and B by hand. The initial purpose is to
learn what the labeled ADAG objects look like and whether any stable computation-like structure is
even visible.

For each candidate observation, record before revising it:

- process and response;
- cluster IDs/labels and exact signed-basis members;
- exact target-local graph/path witnesses;
- relevant input-token attribution;
- relevant top-five output contribution;
- temporal extent and relation to verbalization/commitment;
- plausible lexical, numeric-copying, formatting, or generic-reasoning confounds;
- whether the pattern recurs in another operation or operand instance.

This phase is intentionally exploratory and transductive: the dense discovery responses helped fit
the atlas. Projecting every dense position and preserving unknowns prevents atlas-target selection
from censoring the inspected timeline, but it does not make same-response discovery independent of
atlas fitting. Its output is a candidate witness hypothesis, not a detector performance estimate.

If labeled clusters are broadly present in `surface_reference` or `trajectory_only` positions,
record the specificity diagnostic against the frozen atlas version. Flat, process-independent
prevalence may indicate generic continuation machinery; a peak near a process event, including
while predicting punctuation, may instead reflect temporal misalignment. Use predeclared prevalence
and attribution-mass comparisons rather than token class alone. Redesign may use that discovery
evidence only in a newly versioned atlas and must be evaluated on a fresh unopened reserve; do not
silently refit or relabel the same atlas until the discovery trajectories look clean.

### 3B. Formalize only after seeing the objects

If recognizable candidates exist, define a machine-checkable witness rule using some combination
of frozen clusters, target-local motifs/paths, process-relevant source attribution, output
contribution, and temporal behavior. Define "stable" at the family/response level, not by counting
many adjacent tokens as independent repetitions.

Freeze the rule and thresholds before opening dense reserve A or any newly generated contrast.
Apply the frozen atlas and witness rule to the reserve case without refitting.

Opening reserve A consumes it for that exact atlas and witness-rule version. If discovery evidence
causes an atlas, label, or witness redesign, the redesigned version requires another unopened
reserve frozen in advance; reserve A cannot be reused as independent validation.

If no recognizable candidate exists, report this ADAG/raw-neuron/T5 procedure as inconclusive for
the selected process class. Do not infer that the model failed to perform the BonaFide-required
computation.

### 3C. Seek unfaithful contrasts only after a positive witness candidate

The initial absence of trustworthy unfaithful bottleneck completions is acceptable. If Step 3A
finds a candidate witness:

1. resample the same selected model on the same frozen prompt, with versioned decoding attempts;
2. seek completions that omit, misstate, or fabricate the required process;
3. independently audit each completion against the deterministic process ledger;
4. freeze accepted contrasts and their roles before tracing them;
5. project them through the unchanged atlas and witness rule.

These are newly reviewed experimental contrasts, not original BonaFide labels. Distinguish at
least: required process verbalized, required process omitted, incorrect process verbalized, and
unrelated attribution/tool fabrication.

### 3D. Causal follow-up

Only candidates that recur under the frozen rule advance to intervention. Prefer position- and
layer-restricted ablation or activation patching before answer commitment, with size-, layer-,
activation-, attribution-, and sparsity-matched controls plus off-window interventions.

A causal effect supports participation in the measured output under that intervention. It does
not by itself prove that a CoT statement is faithful.

## Claim ladder

The study should advance one claim at a time:

```text
BonaFide-required process
    -> recognizable candidate in discovery trajectories
    -> recurrence under a frozen atlas and witness rule
    -> separation from audited contrasts and surface-form references
    -> process-specific causal effect under matched intervention
    -> possible component of a later general faithfulness detector
```

At every stage, a missing ADAG witness is evidence about this recovery method, pruning choice,
atlas, and raw-neuron basis. It is not evidence that the internal computation did not occur.

## Immediate next actions

### 2026-08-12 execution checkpoint

Strict T5 parity and historical Thinking-continuation tokenization are validated. The four-position
scaling wave completed and its compact artifacts passed integrity checks. At target context lengths
319, 769, and 1,268 tokens, peak reserved GPU memory was respectively 16.10, 29.52, and 48.98 GB;
the 1,268-token trace took 229 seconds on one A100. Resource admission is therefore a gate on the
exact prompt-plus-response-prefix context of each annotated target, not a completion-length screen.
Generation remains blind to that later target-context gate.

The Thinking broad-generation lane is frozen at 47 fit prompts, 186 logical response slots, and
three predeclared draws per slot (558 physical requests), using the recovered historical system
message and Qwen Thinking template. Protocol version 2 resolves the graph instruction conflict
mechanically: a graph response may end in either its exact prompt-declared two-key JSON schema or
the exact historical-system `final_answer` schema; complex responses remain `final_answer` only.
The rule checks serialization and nonempty values but never correctness, faithfulness,
interestingness, or response length. Protocol v1 and request bundles v1/v2 remain immutable.

The production request bundle is immutable v3: request-grid SHA-256
`77aab64b05cff413da1c6b450a0e02e78026f9d39775e6a96178414a9f762f81` and manifest SHA-256
`9b96044e8b9dd7d8f19e49611388a97e706acac7f067ce6e619becaa611d4323`.
The protocol-v2 canonical SHA-256 is
`4e3949f33818b36a38b573a48015480bad79bace5e5699282fc489267dff07cd`.
The L40S runtime is frozen for every physical draw in this protocol version; the launcher and batch
script fail closed on a different GPU profile or unhealthy accelerator.

Non-scientific protocol-v2 smoke job 1791871 completed on a healthy L40S at source commit
`7a5dac95c5bf456caccb3a51acb8137bcd5a242e`: four rows, 9,921 completion tokens, four
content-addressed attempt records, natural EOS for every response, and captured aligned token IDs
and logprobs. Exact frozen-row joins, completion and attempt identities, prompt tokenization,
answer-schema admissibility, run provenance, and GPU-health hashes passed. The output CSV SHA-256
is `b91b9ef1cef5463fb404563b6772e9650e2b5f7e40c4a74611c10e31add32dc3`. Saved completion IDs
include the final natural-EOS token `151645`; the persisted raw response intentionally omits the
decoded `<|im_end|>` text. The smoke therefore clears broad generation for submission under
protocol v2 and immutable request bundle v3.

Production broad-generation job 1791893 completed on 2026-08-12 with scheduler state `COMPLETED`,
exit code 0, and elapsed time 2:08:19 on `grn009`. It generated all 558 frozen rows, containing
2,513,759 completion tokens, on a healthy L40S. Its
provenance directory is
`/scratch/rai/vast1/u1653998/bonafide/runs/qwen3-4b-thinking-2507-process-witness-broad-v1/1791893`,
and its materialized output target is
`/scratch/rai/vast1/u1653998/bonafide/campaigns/qwen3-4b-thinking-2507-process-witness-broad-v1/attempts/protocol-v2-production.csv`.
The output CSV SHA-256 is
`1c341b41fb13635ce8a9ed84a905f479966641e9e55fb8ac88c0336a2b477b7b`; its 558-record
content-addressed attempt-store tree SHA-256 is
`4818180508c5d60206a814d8e777f200dc5a8a205fb95ff4505377432b74e7df`.
Independent validation recomputed every completion and attempt identity, exact CSV/store join,
token/logprob alignment, and frozen request/protocol/source binding. One physical attempt reached
the 32,768-token generation limit, but the other two attempts for that slot ended naturally.

Applying the frozen protocol-v2 mechanical rule finds 526 admissible and 32 rejected physical
attempts. First-admissible selection succeeds for 182 of 186 logical slots: attempt index 0 supplies
175 slots, index 1 supplies six, and index 2 supplies one. Four slots fail closed because all three
attempts contain the frozen immediate-block repetition witness. Three failures belong to the same
`complex` string-decoding prompt (leaving one of its four slots), and one belongs to a `graph`
prompt (leaving three of its four slots). The selector's Python CSV field-size limit was raised and
covered by regression tests without changing protocol-v2 admissibility.

### 2026-08-13 atlas-response cohort freeze

The four failed slots were resolved by a separately versioned, prompt-pooled staged backfill. This
is an explicitly adaptive, mechanically conditioned repair of the prompt-balanced cohort; it is
not part of the original fixed-three-draw protocol-v2 sample and must not be described as an
unconditional or equal-effort sample from model generations. The backfill used the identical model
revision, historical system/chat template, decoding parameters, serialization contracts,
protocol-v2 admissibility predicate, and L40S runtime. It did not use correctness, faithfulness,
interestingness, answer content, response length, or traceability to select responses.

The complete candidate universe was frozen before execution in `candidate-plan.json`: 128 ordered
candidates for the deficient `complex` prompt in stages of eight, and eight ordered candidates for
the deficient `graph` prompt in stages of four. Each started stage was generated in full. Sampling
stopped only after a completed stage satisfied the prompt-level deficits of three and one,
respectively; hard-cap exhaustion would have failed the 188-response freeze closed. Candidate-plan
file SHA-256 is `5816779ee5daabd84045aefa2500d9afbe7f7bff77b1f63bc30fe5c59f081a6b`
and canonical SHA-256 is
`5ce09f3303d1c2258e3db0c56f7c9c8627d51ace0e0582da35c0f90e9ab79b67`.

Three generation stages completed on healthy L40S accelerators:

- job `1795117`: 12 attempts (eight complex and four graph), output SHA-256
  `ee160aa774346afe497cf6ddc194f219c68a0c501723d424f1cb8673ac2cd0d2`;
- job `1795254`: eight complex attempts, output SHA-256
  `7debcd0dde5d9aa78519fb8260d484c09fa413bdc5720b91bc363fab18184632`;
- job `1795307`: eight complex attempts, output SHA-256
  `186f121dbcf3ca9e991ff0e063021b2575841192e7f4b4adada91ae3a276ec2c`.

All 28 generated backfill attempts and their content-addressed records remain retained. The frozen
ordering selected graph candidate 0 and complex candidates 8, 11, and 20, then assigned the three
complex responses to the missing logical slots in original Step-0 slot order. The 182 original
first-admissible selections were not reopened or replaced.

The downstream response corpus is frozen read-only at:

```text
/scratch/rai/vast1/u1653998/bonafide/campaigns/qwen3-4b-thinking-2507-process-witness-broad-v1/cohorts/atlas-responses-backfilled-v2
```

Its cohort ID is `qwen3-thinking-process-witness-atlas-responses-backfilled-v2`. Manifest SHA-256
is `d4cbb862333b62d5ae108fdc2d02aab8ab47f4b729438d80f1b880f21d1d76f6`; index SHA-256 is
`c13cdf45a28b350ade8ec578ec4adc0eac740ecc5dec1f32869439ae7c5d9cbc`. Independent validation
verified all 377 payload files and hashes, 188 unique response records, 47 prompt hashes with
exactly four responses each, and 186 unique generated completion IDs. The exact source split is:

- 182 original protocol-v2 mechanically selected full assistant serializations;
- four prompt-pooled mechanically selected backfill full assistant serializations;
- two historical dense reasoning-segment reconstructions.

The historical dense records retain `trace_scope=reasoning_only`; they are not normalized or
represented as recovered full assistant responses. The generated 186 records retain exact raw
assistant text plus links to their authoritative generation records. The response cohort is now a
frozen input to annotation and tracing, but the atlas-fit target set, `Q_process`, target-context
tiers, cluster state, and labels remain unfrozen.

The first frozen cohort directory (`atlas-responses-backfilled-v1`, manifest SHA-256
`aa233d0121f69729656564d93f727f2a94bf252a3609483eef4b9224d83095e3`) remains preserved but is
superseded. Version 2 adds exact freeze-implementation identities and source-run bindings; only
version 2 is admitted downstream.

1. Build and audit the seven-value coarse sampling-tag layer with deterministic structure plus
   direct structured LLM API calls; use it only to inventory and enrich wave-one target selection,
   without requiring exhaustive rich-axis manual review.
2. Freeze the ordered, prompt/response/event-blocked adequacy target bank, inclusion weights,
   composition cells, and exact T5 target-context resource tiers.
3. Start primary T5 production; optionally trace the identical bank with CU5 in a separate sealed
   sidecar. While production runs, refine the graph-blind descriptive layer and execute the
   synthetic ADAG mechanism battery.
4. Implement and parity-test the paper-faithful per-context harmonic condition; freeze the
   released-code fidelity condition, bounded clustering comparator grid, full pair-evidence
   outputs, metrics, and perturbation floors before opening real cluster outcomes.
5. Run the matched real-data composition series and decide `robust`, `brittle`, or `inconclusive`,
   localizing any brittle result before changing the method.
6. Only after a robust verdict, design and freeze the separate motif-study sampling, graph
   representation, recurrence rule, reserve, and additional trace requirements.

### 2026-08-13 annotation implementation checkpoint

Automatic bootstrap artifacts `process-witness-graph-blind-auto-v1` through `auto-v3` span all 188
responses and 842,007 response tokens, but are superseded and inadmissible for human review. Their
structural integrity passed; strict review nevertheless found stale UI response context, absent
prompt/task display, incomplete ontology and provenance validation, excessive apostrophe/quote and
percentage/modulo matches, and inadequate pagination/resume safeguards.

The code checkpoint fixes those findings: documents carry prompt/task context and an expanded
multi-axis ontology; the review UI binds selections to response identity, uses Unicode code-point
coordinates, validates ontology values, paginates, imports prior JSONL decisions, and exports
append-only provenance-rich events; rules distinguish apostrophes and guard non-arithmetic
percentages; terminal JSON detection fails closed. Seven focused tests, Ruff, Python compilation,
and diff checks pass.

The canonical automatic draft is now v9. The read-only artifact is at
`/scratch/general/vast/u1653998/circuits/results/process_witness/annotations/process-witness-graph-blind-auto-v9`.
It contains 188 responses, 842,007 tokens, and 1,038,919 suggestions; manifest SHA-256 is
`634509e5ff9b7a9dd6f859fbd281aa696a17f2c7f24a7dcf8d76f8e4441dd2af`, and compact workstation
bundle SHA-256 is `95a5627768b2e5f05920aaab91f6e1cd9c00688560f9d821ffc2d8d69c7ceeea`.
Ontology v6 preserves exact token cues separately from broader candidate event spans, adds bounded
context for active execution and lookup, and leaves the downstream `usage` and `event_status` axes
human-only. Versions 1–8 remain preserved but are superseded: v5 was too semantically sparse,
while denser v6–v8 drafts were blocked by response-stratified and contextual false-positive audits.
Only v9 `workstation-bundle.json` is canonical for human annotation. No human review, target
selection, or tracing has yet begun.

### 2026-08-16 adequacy sequencing checkpoint

The current execution order is coarse selection, target-bank freeze, then three concurrent lanes:
T5 tracing, graph-blind descriptive refinement, and synthetic ADAG tests. Real-data adequacy fits
follow when the required traces and annotation version are frozen. The source audit defining the
failure mechanisms and paper/code discrepancy is
`experiments/process_witness/ADAG_POLYSEMANTICITY_AND_CLUSTERING_AUDIT.md`.

The coarse schema is now a seven-value sampling-only instrument. The next concrete design task is
its exact unit segmentation, direct-API annotation protocol, human acceptance audit, and
priority-weighted wave-one sampling matrix. Exact trace count, proposal-route quotas, inclusion
weights, partial-wave ordering, and resource tiers remain to be frozen from the yield inventory.
Refined annotation may justify a smaller, separately frozen second tracing wave before any ADAG
outcomes are opened. No atlas cluster, description, motif, or dense trajectory has been opened
under this design.

### 2026-08-17 coarse-label refinement and matched few-shot qualification

The completed full-context human review is development evidence, not an exchangeable accuracy
sample. Its 72 blind decisions were made in a fixed order while the reviewer learned the coarse
boundaries, and model decisions were revealed after each lock. Preserve the original blind event
and every post-reveal correction separately; do not rewrite either into ground truth. The uploaded
ledger SHA-256 is `2b4cf65ea8bf92662b261b691c2baa3638f220bca2de5a57a9f7518cbaa2b0bc`.

Coarse labels now use a trajectory-effect rule. `active_task_work` creates new task state or
evidence; `evaluation_or_revision` assesses or changes an existing candidate;
`intermediate_commitment` reports a settled non-final state without performing the operation in
that unit; and `other_semantic_text` plans, explains, restates, quotes, or comments without
changing task state. The fixed composite-unit precedence is final, evaluation, active,
intermediate, other, then surface. `uncertain` positively represents a defensible tie or
insufficient unit boundary rather than a failure to force a label.

The next qualification uses a fresh graph-blind holdout and only the full-response
`target_only_markup` presentation. It compares two matched arms:

1. the refined definitions with no demonstrations;
2. the identical definitions plus a frozen pack of short contrastive micro-context examples.

The example pack may use synthetic text or the completed development set but cannot contain a
holdout unit. Both arms use `gpt-5.6-luna`, medium reasoning, strict structured output, and the same
frozen target groups. Each arm makes three predeclared identical-protocol requests per group. The
complete ordered decisions remain a replica vote profile: 3-0 stable, 2-1 mixed, and 1-1-1
disputed. A majority decision never replaces the physical votes and is not semantic truth or an
independent-voter confidence estimate.

The comparison freezes before submission: human blind agreement, paired arm wins/losses,
replica-profile counts, tag confusions, boundary concerns, abstention, broad selection-family
stability, usage, cache buckets, and cost. The fresh human holdout remains globally blind until
review completion. Few-shot prompting is preferred only if its human agreement and stability
improve without suppressing legitimate uncertainty or causing a material per-family regression.
The qualification freezes no production coarse corpus, target bank, trace, adequacy result, motif,
or witness.
