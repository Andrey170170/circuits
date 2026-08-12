# BonaFide process-witness discovery plan

Status: **central draft protocol; the overall architecture is accepted, Step 0 remains open, and no
new generation or tracing run is authorized.**

This is the governing plan for the new process-witness campaign. Where it conflicts with
`docs/ADAG_BONAFIDE_NAIVE_PILOT.md` or `docs/TRACING_CORPUS_PLAN.md`, this plan governs the new
campaign; those documents continue to describe the earlier Qwen Instruct feasibility pilot and its
frozen execution history.

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

The present experiment asks a narrower prerequisite question:

> When BonaFide tells us that a particular answer-relevant process must have occurred, can a
> frozen ADAG atlas recover a stable and recognizable witness of that process in an ordered
> series of attribution graphs?

This is a positive-control process-recovery study. It does not yet establish either direction of
a general faithfulness detector.

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

For Thinking-model completions, the authoritative response is the exact serialized assistant
token sequence captured by generation, including thinking delimiters and the final-answer segment.
Do not reconstruct a trace input by concatenating a parsed `cot` column with a parsed answer when
raw completion token IDs are available. Before freezing the historical dense cases, either recover
their original assistant serialization or document and validate the reconstruction rule; otherwise
the original-inference-context claim remains unproven.

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
  fitting, hyperparameter choice, exemplars, and labels;
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
must not load graph or clustering artifacts. Freeze and hash the reviewed annotation manifest before
production tracing or atlas fitting. At freeze, every dense position not explicitly assigned to
`process_atlas_fit` or `surface_reference` is recorded as `trajectory_only` rather than omitted.

The initial trace budget is deliberately not frozen in this draft. First inventory annotated
process-target yield and measure T5 cost on the selected model, then freeze the response-balancing
rule, process quotas, nested process-only enlargement tiers if needed, and surface-reference quotas
in a new corpus manifest. For broad responses, trace the selected process targets and frozen
surface references. For dense discovery responses, trace every response position. Only dense
targets marked `process_atlas_fit` participate in the primary atlas fit; all dense targets are
projected after the atlas freezes.

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
resource-derived length cap; do not select broad responses by correctness, faithfulness, or
perceived interestingness. Freeze a maximum-attempt/yield failure rule before generation so prompts
with poor mechanical yield cannot be handled ad hoc.

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

## Step 1: build and freeze the atlas

### 1A. Trace the atlas-fit corpus

- Trace the frozen `process_atlas_fit` targets and `surface_reference` panels from the broad
  completions with T5; keep surface-reference artifacts sealed from fitting and labeling.
- Trace every response token in dense discovery A and B with T5.
- Before tracing, freeze deterministic `process_atlas_fit` panels under the chosen response-
  balancing rule, nested process-only enlargement tiers, and a separately marked surface-reference
  panel.
- Preserve one independent graph artifact per response position.
- Do not merge target-local graphs in the tracing pipeline.
- Record observed-token rank/membership at every position.

The primary atlas follows upstream's uniform-over-context aggregation, but its contexts are the
frozen `process_atlas_fit` panels rather than uniform all-token samples. Use the response-balancing
rule frozen at the annotation-yield gate. Audit projection coverage over the complete dense
trajectories without examining cluster labels, surface-reference results, or candidate witnesses.
Predeclare assignment gates using signed-node coverage, attribution-mass coverage, process-ledger
landmark coverage, and cluster stability. If the smallest process-only panel fails those gates,
advance to the next already frozen process-only tier. Choose the smallest passing tier; if none
passes, stop or version a new method rather than silently admitting every dense token.

This response-balanced subset is preferable to custom weights for the primary study because the
upstream implementation uniformly averages contexts and has no family/response-weight interface.
A separately named hierarchically weighted all-dense atlas may be implemented later as a method
sensitivity, but it is not upstream-equivalent T5 ADAG. Cluster-label evidence must likewise use
frozen prompt/response exemplar caps. Full dense traces are produced once and projected in Step 2
rather than redundantly retraced. Surface-reference traces are opened for projection and comparison
only after the primary atlas freezes.

### 1B. Fit clusters label-blind

Construct input-attribution and output-contribution profiles using only the frozen atlas-fit
corpus. Select normalization, cluster state, and stability settings without looking for attractive
process labels or opening the dense reserve case.

Use a predeclared cluster-count sweep rather than assuming the paper's task-specific value transfers
to this broader outright-task corpus. Increasing `k` is a complementary resolution choice, not a
substitute for process-targeted sampling. Freeze the candidate grid after the observed signed-basis
and support audit, then choose the primary `k` label-blind using resampling stability, minimum
cluster support, and profile coherence within the process-target fit corpus. Retain one predeclared
higher-resolution fit as a sensitivity view if support permits; do not use surface references,
manually attractive math labels, or candidate witnesses to choose `k`.

Report task-family support for every cluster. Because the initial dense cases are arithmetic or
algorithmic while the primary corpus spans all outright tasks, optionally predeclare one arithmetic/
algorithmic-only refit of the identical saved process-target traces as a corpus-composition
sensitivity. It is not a replacement primary atlas and may not be promoted because it produces a
more attractive witness; it tests whether legitimate non-math process diversity is consuming the
available clustering resolution.

Retain:

- signed-basis identities and polarity;
- cluster assignments and profile prototypes;
- normalization and missing-support semantics;
- seeds, hyperparameters, stability results, and corpus weights;
- exact exemplars and their response/family provenance;
- unseen-feature application rule.

### 1C. Label and freeze

Generate cluster descriptions from discovery evidence, retain abstentions and uncertainty, and
audit proposed labels against exact profile/graph exemplars. Then freeze:

```text
signed basis -> cluster
cluster -> label/evidence bundle or abstention
```

Do not change the cluster state because a label is disappointing, and do not relabel after viewing
the reserve trajectory. The complete frozen object is the **atlas**.

## Step 2: build dense bottleneck graph trajectories

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

1. Accept or revise the proposed Qwen Thinking primary, exact conversation contract, and three dense
   roles.
2. Recover and validate the authoritative serialized assistant tokens for the historical dense
   cases; regenerate them under the new contract if recovery fails.
3. Freeze the broad generation manifest and cost envelope, then seek separate authorization for the
   generation-only run.
4. Build the deterministic process ledgers and token-aware region-annotation page.
5. Paint and review process and surface-reference regions, derive token targets, and freeze the
   hashed three-class annotation manifest and `Q_process` tiers.
6. Implement and validate a strict upstream T5 teacher-forced runner distinct from CU5, then run the
   bounded Step-0 smoke/resource gate.
7. Specify the unknown-preserving trajectory-viewer schema and validate its three attribution-mass
   states on small synthetic or smoke-test artifacts.
8. Freeze the versioned selection and execution bundle; only then seek authorization for a
   production atlas and dense-trace campaign.
