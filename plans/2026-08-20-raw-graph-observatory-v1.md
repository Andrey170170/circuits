# Raw-graph observatory v1

Status: **accepted initial version**. This is a separate exploratory lane. It freezes the
selection and inspection workflow below, but it does not select a completion, authorize tracing,
define a neuron label as ground truth, or change the central Qwen process-witness protocol.

## Purpose

Inspect target-local attribution graphs over raw MLP neurons before ADAG clustering, and learn
whether the saved graphs support useful human hypotheses about computation. The immediate outcome
is an inspectable set of exact graphs and conservative neuron evidence cards, not an ADAG atlas or
a faithfulness detector.

The observatory asks:

> Given an independently labeled outright-task completion and a graph-blind target near a known
> process event, what structure is visible in the pruned raw-neuron attribution graph before any
> corpus-level clustering or semantic supernode construction?

## Isolation from the central campaign

This lane is intentionally separated from the frozen Qwen process-witness and global-atlas
adequacy work.

- Candidate review v1 physically excludes every model whose identifier contains `qwen`, matched
  case-insensitively. Excluded records are absent from the embedded HTML payload, not merely hidden
  by a browser filter.
- No response, annotation, target, trace, graph, cluster, or label from the frozen Qwen cohort is an
  input to this v1 selection review.
- Findings from this lane are exploratory method-development evidence. They cannot tune target
  selection, clustering, adequacy rules, motif rules, or claim thresholds for an unopened main-lane
  evaluation.
- A later Qwen observatory requires a new version and an explicit decision about which cases become
  permanently exploratory rather than unopened evidence.
- One existing first-result Qwen Collatz trace received a prior read-only schema/top-row spot check.
  It is not an input to this v1 page and must not be described as wholly unseen.

Working in a separate Git worktree protects code and branch history; the scientific firewall above
protects outcome access. Both boundaries matter.

## Candidate-source contract

The v1 review packet is built from the released `BonaFide.csv` whose SHA-256 is
`5833b500c378bbdcc7103340987749efda10b5944897168e10aed2be4538e13e`.

- **Outright task:** `src_type` is exactly `complex` or `graph`.
- **Completion identity:** deduplicate annotation rows on
  `(target_model, prompt, cot)` and retain every source annotation ID, label, reason, and span.
- **Model exclusion:** omit every completion whose lower-cased `target_model` contains `qwen`.
- **Displayed evidence:** exact model, task/prompt, stored reasoning, model and correct answers,
  source type, question IDs, and all attached annotations.
- **Selection:** the human reviewer selects completion identities, not individual annotation rows.
  Selection state is local to the browser and exports as a source-bound JSON file.

The page must remain filterable by model, exact task, source type, broad faithful/unfaithful
membership, exact label type, selection state, and free-text search. Broad labels are navigation
helpers derived from the exact source labels:

- `faithful_only`: at least one `FAITHFUL_*` annotation and no `UNFAITHFUL_*` annotation;
- `contains_unfaithful`: at least one `UNFAITHFUL_*` annotation;
- `mixed`: both faithful and unfaithful annotations occur on the completion;
- `other`: neither prefix occurs.

These buckets do not replace or reinterpret the annotation rows. In particular,
`UNFAITHFUL_COT` may encode an omission claim without a localized unfaithful step, and the presence
of `FAITHFUL_STEP` does not establish that the complete CoT is faithful.

## Frozen workflow

### 1. Human candidate selection

Review the self-contained non-Qwen packet, select promising completions, and export the exact
selection JSON. Favor short, legible process episodes with localized step evidence and a target
whose tokenization can be inspected precisely. Do not optimize for an attractive future graph.

The user makes the candidate decision. Automated ranking may summarize length or provenance but
must not replace the review or silently drop difficult cases.

### 2. Candidate audit and manifest freeze

Before tracing, audit each selected completion against its original annotation rows and freeze a
new observatory manifest containing:

- source CSV hash and annotation IDs;
- exact model and tokenizer revisions;
- exact prompt, stored response, chat serialization, and their hashes;
- label scope (`*_STEP` versus `*_COT`) and exact character spans;
- tokenization and response-relative event positions;
- graph-blind target-selection reason;
- code revision and tracing configuration.

Models remain separate neuron-identity spaces. Cross-model comparison may compare viewing or
method behavior, never raw neuron IDs.

### 3. Independent target-local traces

For a selected event, begin with independent width-one observed-token traces at a small declared
neighborhood:

1. one pre-event or operator/bridge target;
2. the first subtoken of the event result, predicted from the prefix immediately before it;
3. one immediate post-result target;
4. optionally, the later answer commitment as a temporal control.

Multi-token results are multiple independent targets. Never merge edges from different targets into
one causal graph through response time. A strict summed-top-five trace may be added at the identical
prefix as a separately named sidecar; it does not replace the observed-token trace.

### 4. Raw-graph inspection before clustering

The primary viewer should reveal evidence progressively rather than render the complete graph as a
hairball. It must preserve:

- exact target token, probability/rank, prefix boundary, and objective;
- signed node attribution, activation, layer, token position, neuron index, and polarity;
- signed edge attribution separately from edge weight;
- input-token attribution and output-contribution profiles;
- pruning configuration, graph-size diagnostics, and artifact hashes;
- exact occurrence identity and stable signed basis identity;
- side-by-side target slices without cross-slice path splicing.

Start with source tokens and highest-evidence paths, then allow layer, position, sign, attribution,
and retained-mass filters. A visually clean path is a hypothesis, not proof of computation.

### 5. Conservative neuron evidence cards

Do not assign semantic names from one occurrence. Build cards only for a small set of recurring,
high-evidence signed identities. Each card should include:

- `(model revision, layer, neuron, polarity)`;
- exact graph occurrences and target provenance;
- high and ordinary activation examples from an independent context pool;
- input-attribution exemplars and counterexamples;
- output contributions, including promoted and suppressed targets;
- recurrence across tasks, prompts, and event roles;
- one or more provisional descriptions, uncertainty, and an abstention option;
- held-out observational score when enough examples exist.

Generated prose is a searchable hypothesis layer. It must remain linked to the underlying examples
and may not be called a faithful, unfaithful, bottleneck, or computation neuron without additional
evidence.

### 6. Limited causal spot checks

For a stable candidate hypothesis, run a small position-restricted ablation or patching check with
layer-, size-, polarity-, and attribution-matched controls. Report effects on the declared target
and relevant intermediate nodes separately. Causal importance for a token is not equivalent to
BonaFide faithfulness.

## Version-1 completion criteria

V1 is complete when:

1. the source-bound, physically non-Qwen review packet is built and browser-validated;
2. the user exports a reviewed candidate selection;
3. the selected records pass annotation, serialization, and tokenization audit;
4. a new immutable tracing manifest is reviewed before any GPU launch;
5. at least one selected event can be inspected across independent target slices with exact
   provenance;
6. any neuron descriptions are presented as evidence cards with uncertainty or abstention;
7. exploratory findings and failures are recorded without altering the main campaign.

## Open decisions after human selection

- Which selected model should be traced first, considering current architecture support and
  resource feasibility?
- Which exact annotation/event spans are scientifically useful after manual audit?
- Whether a selected completion needs a matched control or a newly generated same-model variant.
- The first viewer's retained-mass and path-display defaults.
- The independent exemplar corpus and scoring rule for neuron cards.
- Whether any intervention is warranted after raw inspection.

These remain open deliberately. The review page selects candidates; it does not silently freeze the
later tracing or labeling protocol.
