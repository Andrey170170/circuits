# ADAG–BonaFide downstream execution plan

Date: 2026-07-26  
Status: proposed execution plan; no new corpus, clustering, labeling, or model-serving job is
authorized by this document  
Current checkout at plan creation: `main` at
`b0dc9fa30678f04d320d7ae3c33f24cc794940dc`, two commits ahead of `origin/main`, with the
research inputs `BonaFide.csv` and `papers/` intentionally untracked  
Scientific status: exploratory representation-feasibility work, not a validated faithfulness
detector  
Revision focus: dense-first response-time multiplex analysis, family-level inference, label-free
feasibility gating, and optional evidence-gated contribution/model-serving extensions

## 1. Purpose

This document is the durable plan for the work after the completed Qwen3-4B width-one trace
corpus. Its primary scientific object is the dense response-time multiplex: the independently
traced target graphs for every response position in the 11 dense responses, kept as target-indexed
slices and analyzed jointly across generation time. It combines four related but distinct
workstreams:

1. build and validate a provenance-preserving, dense-first atlas from the completed width-one
   traces;
2. define and construct a derived response-time multiplex, informally called the "mega-graph,"
   without collapsing its target-specific graph slices;
3. implement and run a new contribution-aware trace family with five candidate logits at one
   response position;
4. make the explanation, simulation, and cluster-summary pipeline provider-neutral, then evaluate
   a locally served large Qwen model against the existing Transluce and Anthropic paths.

The first feasibility decision is narrower than completion of all four workstreams:

> Do signed basis features or frozen clusters form recognizable, stable, target-witnessed
> trajectories and paths around hint use, source commitment, fabrication, bottlenecks, or answer
> commitment across dense response generation?

The dense multiplex supplies longitudinal evidence about emergence, persistence, source
attribution, and path recurrence. The broad discovery traces test recurrence outside the dense
responses and may contribute to a separately versioned combined-atlas sensitivity analysis. The
confirmatory responses remain sealed until the dense-first discovery state and its evaluation rule
are frozen.

The intended execution pattern is:

1. make the shared code and schema changes first;
2. pass focused correctness and provenance tests;
3. build the source inventory, dense atlas inputs, and small dense response-time multiplex before
   any model-generated labels;
4. freeze one immutable checkout, environment, manifest, and output namespace per executable
   workstream;
5. run small probes and make the dense, label-free feasibility decision;
6. only after the relevant scientific and resource probes pass, launch independent processing
   workstreams concurrently;
7. continue non-overlapping development in the main checkout while those immutable checkouts run.

Two graph constructions must never be conflated:

```text
response-time multiplex
    independent width-one target graphs across response positions
    longitudinal correspondence is observational/identity-based, not a causal edge

candidate-union graph
    one response position with five or six candidate output logits
    topology is the exact union of independent candidate-specific k=1 graphs
    dense candidate values come from fixed-union node and edge rescoring
```

This plan deliberately does not contain a ready-to-paste full-corpus `sbatch` command. Before any
resource-mutating launch, the exact model revision, GPU type and count, partition, account, QoS
where applicable, wall time, host memory, array shape, concurrency, input hashes, code commit, and
output root must be shown for explicit review.

## 2. Completion definition

The first dense feasibility phase is complete when:

- the 2,594 existing compact trace artifacts have been revalidated and indexed without modifying
  them;
- all 2,083 dense traces can be reconstructed as 11 exact target-indexed response-time
  multiplexes;
- the dense-first width-one atlas can be deterministically rebuilt from hashes and assigns frozen
  cluster IDs to eligible signed-basis occurrences;
- input-profile and temporal-trajectory feature views have explicit mathematical definitions,
  missing-support semantics, family/response/target weights, and round-trip tests;
- the dense label-free report measures emergence, persistence, source-attribution trajectory,
  commitment-relative timing, target-witnessed path recurrence, and family-level stability;
- a predeclared dense feasibility decision has been made before model-generated labels are used;
- confirmatory holdout traces were excluded from normalization, hyperparameter choice, fitting,
  exemplar selection, and label generation;
- graph export retains target support and cannot create a path by joining edges from different
  targets;
- any width-one engineering labels are based on discovery attribution and temporal evidence only,
  with the contribution limitation stated explicitly.

The optional downstream extension phase is complete only when all authorized extensions that pass
their own go/no-go gates are complete:

- the top-five trace path passes observed-token `k=1` parity against the completed width-one
  family;
- a small candidate-specific reference set establishes the topology loss of candidate-joint
  graphs and validates exact-union refinement;
- a versioned candidate-selection rule and candidate-union contract have been frozen for the
  top-five family;
- representative top-five probes establish runtime, host-memory, GPU-memory, graph-size, and
  numerical-health bounds, and a family/response-balanced scientific pilot establishes
  non-degenerate added information, before a matched corpus launch;
- the model-facing description pipeline works through a provider-neutral interface and an
  OpenAI-compatible vLLM endpoint;
- locally served Qwen, the specialized Transluce simulator where available, and a small external
  API control can be compared on the same frozen cluster/exemplar set;
- every generated artifact has a stable logical/content identity, source hashes, schema version,
  code revision, environment identity, and versioned recorded storage locations;
- reports separate engineering correctness, clustering stability, graph coherence, labeling
  quality, intervention evidence, and scientific interpretation.

The work is not complete merely because clustering or labeling scripts execute. Each phase must
pass its applicable acceptance gates in Section 14. Failure of an optional extension gate stops
that extension; it does not invalidate a completed dense feasibility result.

## 3. Frozen baseline

### 3.1 Completed trace family

The current scientific source is the frozen
`Qwen/Qwen3-4B-Instruct-2507` width-one, teacher-forced trace family.

| Item | Frozen value |
| --- | --- |
| Model | `Qwen/Qwen3-4B-Instruct-2507` |
| Model/tokenizer revision | `cdbee75f17c01a7cc42f958dc650907174af0554` |
| Execution cohort commit | `32f1084` |
| Final selection manifest | `scripts/bonafide/manifests/qwen3_4b_instruct_final_traces.json` |
| Final selection manifest SHA-256 | `706143579c8ebcbd05e1fee150d2f3facf5f3f7e7de372c40e399b83a01687e2` |
| Frozen execution plan | `scripts/bonafide/manifests/qwen3_4b_instruct_final_execution_plan.json` |
| Execution plan file / logical SHA-256 | `bbd4df3593f79ec83c2a84947cd6087f103aff340efebc73525d801594986402` / `9a3788a3e8f500dd458b8e890fb287f1a7375f28fd3798bb129f62ef70b2fcc7` |
| Frozen planned targets | 2,595 |
| Completed, checksum-validated artifacts | 2,594 |
| Dense discovery | 2,083 targets from 11 responses |
| Broad discovery | 384 targets from 24 responses |
| Discovery fit/label total | 2,467 planned and 2,466 completed |
| Confirmatory holdout | 128 targets from 8 family-locked responses |
| Routine execution | job `14112800`, 2,591 artifacts |
| Isolated completed extremes | tasks 12–14, 3 artifacts |
| Excluded pathological target | `bf-4f3bab852b0bea33fe6d`, response position 663, token `Given` |
| Pathological screening workload | 81,461,593 candidate MLP edges |
| Artifact root | `/scratch/general/vast/u1653998/circuits/results/bonafide/final-traces` |
| Current storage footprint | approximately 1.8 GiB |

The exact completeness statement is:

> 2,594 of 2,595 selected width-one traces are complete; one pathological discovery target was
> deliberately excluded before submission.

Task 15 is neither a failed run nor an ordinary missing shard. It remains outside the downstream
input denominator unless a separate memory/runtime strategy is approved. The two neighboring
targets at response positions 662 and 664 preserve local coverage of the source-commitment event.
No downstream step may silently retrace task 15 with a different configuration merely to make the
count look complete.

### 3.2 Current artifact semantics

Each reusable artifact is one independent, teacher-forced response target. It is a pruned,
locally approximate attribution subgraph for one selected logit, not a complete transcript of the
model's computation and not a response-wide graph.

The current contribution vector has width one because each artifact has one selected output logit.
This does not imply that its node/edge graph is small. It means only that the within-artifact
output-contribution profile is one-dimensional. The width-one corpus is therefore suitable for:

- loader, schema, integrity, indexing, graph-export, and holdout-firewall testing;
- input-attribution-based clustering and provisional attribution labels;
- estimating shared neuron occurrence and target-support structure;
- developing the multiplex and visualization contracts.

It is not automatically sufficient for a meaningful contribution-view atlas. That question must
be measured, not assumed. A top-five candidate-logit family is a separate contribution-aware
trace family.

### 3.3 Current downstream implementation status

At plan creation:

- no atlas artifact has been generated;
- no discovery-only cluster state has been frozen;
- no response multiplex has been generated;
- no top-five BonaFide trace manifest or corpus exists;
- no Qwen3.5 serving environment or weight snapshot is frozen;
- no model-output usage ledger or complete API cost estimate exists.

The repository does contain useful building blocks:

- compact artifact save/load validation;
- `CircuitData.merge()` for concatenating circuit tables and reindexing circuit inputs;
- multi-view spectral clustering;
- in-process `VLLMExplainer`;
- an in-process, specialized Transluce `FinetunedSimulator`;
- Anthropic contribution and attribution explainers/scorers;
- Opus-based final cluster summarization.

Those building blocks are not yet the required downstream architecture.

## 4. Non-negotiable scientific and provenance rules

### 4.1 Trace identity

- Never rewrite or augment the completed compact source artifacts.
- Preserve prompt, response position, prediction position, target token, model/tokenizer revision,
  trace configuration, source manifest hash, artifact payload hash, and execution cohort.
- Use the existing deterministic `artifact_id` as `trace_unit_id` for the width-one v1 corpus. If a
  future schema needs distinct identifiers, it must define and test their relationship explicitly.
- Treat logical/content identity as distinct from physical location. Record source and current
  artifact locations as locations; never make an absolute VAST path the scientific identity.
- Recover tokenizer identity from the frozen selection manifest's tokenizer revision, tokenizer
  file-manifest aggregate hash, and chat-template hash. Do not invent a payload-local tokenizer
  field where the compact artifact did not store one.
- Do not mix Qwen Instruct and Thinking traces, neuron identities, atlases, or labels.
- Do not treat a multi-target trace as separable into exact single-target graphs after tracing.
- Do not describe a graph absence as evidence that a computation did not occur.

### 4.2 Basis identity versus occurrence identity

Use separate keys:

```text
basis_key =
    (model_id, model_revision, layer, neuron_index, polarity)

occurrence_key =
    (trace_unit_id, token_position, layer, neuron_index, polarity)
```

`polarity` is explicit rather than inferred later. A shared cluster is attached to a signed basis
feature. That cluster is then projected onto position-specific occurrences in each target trace.
An occurrence never loses its trace-unit identity.

No scientific key may use `NeuronId.to_string()` or another representation that omits polarity.
The signed tuple is serialized through a versioned typed schema.

### 4.3 Statistical unit and weighting

The primary independent unit is the BonaFide base-question family, not a target token. The analysis
hierarchy is:

```text
base-question family
    response/condition
        target response position
            signed basis occurrence
```

The default fit weight is equal total weight per base-question family, then equal weight per
response within a family, then equal weight per target within a response. Any alternative weighting
is a named sensitivity analysis. Resampling, cross-validation, uncertainty intervals, and
generation/selection/audit partitions operate on whole base-question families. Target-level
measurements remain useful descriptive observations but are not treated as independent samples.

Cluster support is reported at all levels:

```text
signed basis occurrences
target positions
responses
base-question families
conditions and corpus roles
```

Clusters supported by fewer than a frozen minimum number of discovery families are marked
`prompt_local` or `insufficient_cross_family_support`, rather than being presented as recurrent
features. The minimum is selected from discovery-only structural preflight evidence before
semantic graph inspection and is stored in the cluster state.

### 4.4 Response-time multiplex and target-indexed graph semantics

The "mega-graph" is a derived response-time multiplex. It is not a scalar union of all response
graphs. For dense response `r`, it contains the independently traced slices
`G(r, 0), ..., G(r, T)`. A shared signed basis or cluster may be linked across slices by an explicit
`same_basis_at_next_target` or `same_cluster_at_next_target` correspondence. Such a correspondence
means recurrence across teacher-forced target graphs; it is not a model-computation edge and may
never participate in an ordinary causal/path query.

For every derived node and edge, retain the supporting target or targets. A path
`e1, e2, ..., en` is valid only when:

```text
support(e1) ∩ support(e2) ∩ ... ∩ support(en) is nonempty
```

A graph query must also preserve exact endpoint continuity: the target occurrence of `ei` must be
the source occurrence of `ei+1` inside a surviving target trace. It is invalid to join:

- one edge from response target A to another edge from response target B; or
- two edges from the same target through different occurrences merely because those occurrences
  share a basis neuron or cluster.

A basis-level or cluster-level path is valid only as a projection of at least one witnessed,
occurrence-continuous per-target path. The query must return the exact witnessing target IDs and
occurrence paths.

Longitudinal queries are separate from path queries:

```text
query_trajectory(basis_or_cluster, response_filter)
    -> ordered per-target measurements and support

query_recurrent_path(projected_path, response_filter)
    -> independently witnessed per-target paths and their target positions
```

`query_recurrent_path` may show that a path pattern recurs across generation time, but it must not
join the endpoint of one target graph to the start of another and call the result one path.

`CircuitData.merge()` may be used only in a tightly tested adapter to reuse table preparation. It
must not be treated as the multiplex data model, provenance index, or path engine.

### 4.5 Holdout firewall

The eight family-locked confirmatory responses and their 128 targets are protected as follows:

| Operation | Discovery traces | Holdout traces |
| --- | --- | --- |
| Artifact/schema validation | Yes | Yes |
| Feature-space inventory used for fitting | Yes | No |
| Normalization/statistics estimation | Yes | No |
| Cluster-count or hyperparameter selection | Yes | No |
| Cluster fitting | Yes | No |
| Stability-driven model selection | Yes | No |
| Exemplar selection for descriptions | Yes | No |
| Label generation or label revision | Yes | No |
| Frozen cluster assignment | Yes | Yes |
| Unseen-basis reporting | N/A | Yes |
| Predeclared coverage/transport/trajectory metrics | Internal validation | Confirmatory evaluation |
| Position-restricted steering with matched controls | Later development | Allowed after freeze |

If a basis feature appears only in the holdout, it is reported as unseen/unassigned unless a
predeclared out-of-sample assignment rule can assign it without refitting. The holdout must not
cause a new cluster to be created or an existing label to be edited.

Exact basis lookup is not called assignment accuracy or evidence of generalization. Confirmatory
evaluation reports, at minimum:

```text
unweighted seen/unseen signed-basis coverage
attribution-mass-weighted seen/unseen coverage
holdout profile-to-frozen-cluster-prototype coherence on supported dimensions
frozen confidence/rejection rate where a prototype assignment rule exists
commitment-relative temporal-role recurrence
target-witnessed graph/path recurrence
per-family outcomes and family-blocked uncertainty
```

The eight holdout responses/families are the confirmatory sampling units; their 128 target traces
are repeated measurements, not 128 independent trials.

### 4.6 Claim boundaries

All atlas clusters and generated labels remain exploratory. Required wording:

- "clustered attribution/contribution profiles," not "the model's concepts";
- "target-supported derived path," not "the model's complete computation path";
- "intervention changed the selected output under this control," not "the explanation is
  faithful";
- "confirmatory holdout coverage/transport/trajectory result," not "validated faithfulness
  detection."

Any causal spot check requires size-, layer-, attribution-, and position-matched controls. Causal
importance is not equivalent to BonaFide faithfulness.

## 5. Artifact families and namespaces

No track writes into another track's namespace.

| Family | Proposed schema ID | Source | Purpose |
| --- | --- | --- | --- |
| Frozen compact trace | existing compact schema | Qwen3-4B width-one tracing | Immutable numerical source |
| Atlas index | `adag.bonafide.atlas-index.v1` | Validated compact traces | Target, circuit-input, basis, and occurrence provenance |
| Atlas feature store | `adag.bonafide.atlas-features.v1` | Discovery atlas index | Sparse input-profile and temporal-trajectory views with discovery-only normalization |
| Frozen cluster state | `adag.bonafide.cluster-state.v1` | Discovery features | Cluster mapping, fit configuration, stability evidence |
| Label bundle | `adag.bonafide.cluster-labels.v1` | Frozen clusters and discovery exemplars | Model outputs, parsed labels, scores, usage, provenance |
| Response-time multiplex | `adag.bonafide.response-time-multiplex.v1` | Compact traces plus frozen clusters | Exact target slices, target-supported paths, longitudinal trajectories, and summaries |
| Joint top-five diagnostic trace | `adag.compact-trace.topk-position.v1` | C0 diagnostic execution | Joint topology plus named candidate-logit contributions |
| Candidate-union trace | `adag.compact-candidate-union.v1` | Independent `k=1` references plus fixed-union rescores | Exact independent topology union and dense candidate node/edge profiles |
| Model server manifest | `adag.label-server.v1` | Immutable serving environment | Model/revision/runtime/hardware/endpoint contract |
| Model evaluation | `adag.label-model-eval.v1` | Frozen pilot prompts and outputs | Quality, reliability, latency, GPU, token, and cost evidence |

Proposed large-output roots are under `$CIRCUITS_RESULTS_DIR`, for example:

```text
bonafide/downstream/dense-atlas-width1-v1/
bonafide/downstream/dense-response-multiplex-width1-v1/
bonafide/downstream/top5-probes-v1/
bonafide/downstream/candidate-union-v1/
bonafide/downstream/labels-pilot-v1/
bonafide/downstream/model-eval-v1/
```

The concrete roots must be frozen in manifests before execution. Completed results that must be
retained beyond VAST scratch's purge window must be copied to an approved durable location with a
checksummed inventory.

## 6. Dependency and parallelism model

```text
shared identity, signed-basis, partition, and target-slice contracts
    |
    +--> width-one artifact inventory and dense-first atlas index
    |       |
    |       +--> dense input-profile and trajectory-feature preflight
    |       |       |
    |       |       +--> frozen dense-first width-one cluster state
    |       |               |
    |       |               +--> dense label-free feasibility decision
    |       |               +--> optional attribution/trajectory labeling pilot
    |       |               +--> broad recurrence analysis
    |       |               +--> frozen holdout transport evaluation
    |       |
    |       +--> dense response-time multiplex assembly
    |               |
    |               +--> trajectory and recurrent-path analysis
    |               +--> project frozen cluster IDs and optional labels
    |
    +--> observed-token k=1 compatibility trace
    |       |
    |       +--> candidate-specific k=1 reference traces
    |               |
    |               +--> top-five candidate-policy/resource probes
    |               |
    |               +--> family/response-balanced top-five scientific pilot
    |                       |
    |                       +--> frozen top-five manifest and execution plan
    |                               |
    |                               +--> conditionally authorized matched top-five corpus
    |                                       |
    |                                       +--> contribution-aware atlas
    |
    +--> provider-neutral model interface
            |
            +--> local Qwen server readiness and smoke tests
            +--> Transluce simulator reference
            +--> external API control and cost estimator
                    |
                    +--> frozen labeling-model comparison
```

The following can run concurrently after their immediate shared contracts pass:

- dense width-one atlas and response-time multiplex feature construction;
- top-five representative probes;
- Qwen serving-environment preparation and model-load smoke testing;
- multiplex schema and query testing on small frozen fixtures.

The following must wait:

- holdout evaluation waits for a frozen dense-first discovery cluster state, broad recurrence
  rule, and confirmatory metric contract;
- final labels wait for frozen clusters and a chosen model path;
- label projection into the multiplex waits for frozen cluster IDs;
- a full top-five corpus waits for candidate-policy, `k=1` parity, resource, and launch approval
  gates plus a family/response-balanced scientific-utility gate;
- a contribution-aware atlas waits for enough validated top-five traces.

Provider-neutral model interfaces and large-model serving are not prerequisites for the dense,
label-free feasibility decision.

## 7. Shared code tranche before parallel launches

### 7.1 Inventory and validation layer

Implement one read-only corpus inventory command that:

1. loads the frozen final selection and execution plan;
2. enumerates every expected target;
3. resolves the compact artifact path;
4. validates manifest schema, payload size, payload SHA-256, numerical finiteness, model identity,
   target count, and `scientifically_reusable`;
5. classifies the target as discovery, holdout, excluded pathological, missing, corrupt, or
   unexpected;
6. emits a deterministic inventory sorted by frozen target identity;
7. writes the inventory atomically with its own canonical hash.

Required summary counters:

```text
planned
completed
discovery_planned
discovery_completed
holdout_planned
holdout_completed
excluded_pathological
missing
corrupt
unexpected
```

Acceptance requires `2,594 complete`, `1 excluded_pathological`, and zero corrupt or unexpected
scientific artifacts.

### 7.2 Atlas sidecar index

Build a sidecar rather than placing provenance into modified `CircuitData` objects. At minimum,
each target record contains:

```text
atlas_trace_index
trace_unit_id
artifact_id
artifact_path
artifact_manifest_sha256
artifact_payload_sha256
source_selection_manifest_sha256
source_execution_plan_sha256
example_id
corpus_role
cluster_fit_eligible
condition and family IDs
selection_reasons
response_position
prediction_position
target_token_id
target_token_text
target_logit
target_probability
model_id and revision
tokenizer revision
trace configuration identity
code revision and tracing source-tree hash
```

Each circuit-input/label reindex operation must have an explicit mapping:

```text
(trace_unit_id, local_ci_index, local_label) -> global_atlas_ci_index
```

Do not rely on an array offset that cannot be reconstructed from the sidecar.

### 7.3 Basis and occurrence adapters

Create typed conversion functions for:

- raw node row to signed basis key;
- raw node row to occurrence key;
- raw edge row to a target-indexed occurrence edge;
- occurrence to frozen cluster assignment;
- cluster assignment back to per-target graph export.

Tests must cover:

- positive and negative polarity;
- identical basis neurons at different token positions;
- identical basis neurons in different trace units;
- boundary/embed/unembed nodes;
- local-to-global circuit-input relabeling;
- unknown holdout basis features;
- serialization round trips.

The downstream scientific path must not reuse the current polarity-losing cluster-label
aggregation unchanged. In particular, no mapping may collapse `(layer, neuron, polarity)` to
`(layer, neuron)`, and no cluster exemplar may replace member polarity with a constant sign.
Add an end-to-end fixture in which the same raw neuron appears with both polarities and receives
distinct indexing, cluster, exemplar, label-evidence, serialization, and multiplex projections.

### 7.4 Fit/apply split

Refactor clustering orchestration so that these are explicit, separately persisted operations:

1. `build_features(discovery_only)`;
2. `fit_cluster_state(features, config)`;
3. `apply_cluster_state(atlas, frozen_state)`;
4. `evaluate_assignments(assignments, partition)`.

The current convenience behavior that clusters and optionally fetches descriptions in one call
must not be the production orchestration path. `get_desc=False` is required during fitting.

The cluster state records:

```text
feature schema and hash
discovery target and prompt hashes
normalization parameters
similarity construction
missing-overlap handling
random seed
cluster algorithm and version
cluster count
all clustering hyperparameters
basis-to-cluster mapping
frozen cluster prototypes for supported input-profile and temporal views
prototype-distance/similarity definition
confidence and rejection thresholds where out-of-sample prototype assignment is enabled
unassigned rule
software lock and code revision
fit diagnostics and stability report hashes
```

### 7.5 Provider-neutral model interfaces

Separate the following capabilities:

```text
ExplanationGenerator
    generate_attribution_explanations(...)
    generate_contribution_explanations(...)

ActivationSimulator
    score_attribution_explanations(...)
    score_contribution_explanations(...)

ClusterSummarizer
    summarize_cluster(...)
```

Implement adapters for:

- existing in-process vLLM explainer;
- existing Transluce finetuned simulator;
- existing Anthropic endpoints;
- a generic OpenAI-compatible chat-completions endpoint;
- a deterministic fake backend for tests.

The generic remote adapter must accept endpoint URL, served model name, generation configuration,
reasoning mode, concurrency, timeout, retry policy, and a run identifier through configuration.
Do not hard-code a vendor class into the atlas/labeling orchestration.

### 7.6 Usage ledger and dry-run estimator

Every model call writes an append-only record:

```text
request_id
run_id
cluster_id
role
evidence_partition_id
backend
endpoint identity without credentials
model name and frozen revision where known
prompt-template version/hash
input hash
generation parameters
total input tokens
uncached input tokens where derivable
cache-read tokens where reported
cache-write tokens where reported
reasoning tokens where reported
output tokens
latency and retries
parse status
response hash and output artifact reference
provider price snapshot identity where applicable
estimated monetary cost
GPU allocation identity and elapsed GPU-seconds for local serving
```

Provide a dry-run mode that renders every prompt, counts tokens with the matching tokenizer, and
computes expected API cost from an explicitly dated price table. Do not infer cost from call count
alone.

### 7.7 Immutable-run manifest

All executable tracks use a common run envelope containing:

```text
run family and schema
created timestamp
Git commit
dirty-status hash
relevant source-tree hash
environment or container identity
Python/CUDA/PyTorch/runtime versions
input paths and hashes
configuration paths and hashes
output root
Slurm job and array identity when applicable
host/GPU inventory
resume policy
```

The run fails before substantial work if the checkout is dirty relative to the recorded state or
if any frozen input/config hash differs.

## 8. Workstream A: dense-first width-one atlas

### 8.1 Goal

Build the least-change, dense-first downstream atlas from the completed width-one artifacts. Its
main scientific purpose is to determine whether signed basis features or clusters form stable,
recognizable trajectories and target-witnessed paths across the 11 densely traced responses
without retracing. Its engineering purpose is to validate the downstream data contract before any
larger contribution-aware corpus or model-labeling investment.

No tracing model inference is required to:

- validate and load compact artifacts;
- construct the atlas index and sparse feature store;
- fit clusters;
- assign clusters;
- export graph tables and diagnostics.

A tokenizer may be loaded for exact decoding if stored token text is insufficient, but the traced
Qwen model itself should not be loaded. Explanation/label generation is a separate model-serving
step.

### 8.2 Input partition

Use the partitions in stages:

1. dense core:
   - 2,083 completed traces from 11 responses;
   - primary response-time feature construction, clustering, trajectory analysis, label-free
     feasibility decision, and later exemplar generation;
2. broad discovery recurrence:
   - 383 completed traces from 24 responses/families;
   - apply the frozen dense-first state and report coverage, prototype coherence, and trajectory/
     path recurrence outside the dense responses;
   - a combined dense-plus-broad fit is allowed only as a separately versioned sensitivity
     analysis and never replaces the frozen dense-first result silently;
3. confirmatory holdout:
   - 128 traces from eight family-locked responses;
   - opened only after the dense state, broad recurrence rule, any labels, and confirmatory metrics
     are frozen;
4. excluded:
   - zero data from the pathological target.

This staging deliberately uses broad discovery as an internal transport check for the dense-first
scientific object. Because its corpus role permits discovery fitting, a later combined-atlas
sensitivity run is scientifically allowed, but its outputs receive a new cluster-state identity.

Every report carries target, response, and family denominators. At minimum:

```text
discovery: 2,466 of 2,467 planned
dense core: 2,083 of 2,083 planned targets, across 11 responses
broad discovery recurrence: 383 of 384 planned targets, across 24 responses/families
holdout: 128 of 128 planned
overall: 2,594 of 2,595 planned
```

### 8.3 Streaming feature build

Do not begin by unpickling and concatenating every graph into one unconstrained in-memory
`CircuitData`. Use a two-pass or resumable streaming builder:

1. inventory pass:
   - validate each artifact;
   - count nodes, edges, circuit inputs, basis keys, occurrences, and nonzero profile entries;
   - estimate dense and sparse memory requirements;
   - write per-artifact stats;
2. feature pass:
   - assign deterministic signed-basis and feature-column indices from the dense core only;
   - write sparse input-profile, temporal-trajectory, support, and width-one contribution blocks;
   - checkpoint by trace or prompt;
   - verify row/column checksums and totals;
3. compaction pass:
   - combine blocks into a read-only sparse feature store;
   - verify that reconstructing a sampled trace matches the compact source rows.

For the frozen dense run, feature and response-time multiplex construction share the expensive
artifact-load prefix. One response-array task checksum-validates, decompresses, unpickles,
normalizes, and round-trip-validates each compact trace once, then emits two separately identified
and atomically resumable response shards under the feature and multiplex output roots. Their plan
hashes, manifests, schemas, compactors, and downstream scientific roles remain separate. Buffer
Parquet rows across targets before flushing, and record per-stage wall time, call counts, sink
flush counts, and peak RSS in each response-shard manifest. The frozen joint array is `0-10%4`;
retain the single-lane launcher as a checksum-aware recovery path.

Before allocating any dense pairwise similarity matrix, report:

```text
number of fitted signed basis features
number of sparse profile columns
nonzero count and density
estimated matrix bytes by dtype
peak host-memory estimate
candidate sparse/chunked alternative
```

If the dense similarity estimate does not fit with conservative headroom, use a sparse nearest-
neighbor or blockwise similarity construction. Do not solve a memory problem by dropping prompts
or targets without versioning a new analysis subset.

For the fastest end-to-end smoke, a bounded, family/response-balanced subset may pass through the
`CircuitData.merge()` and clustering code with the atlas sidecar attached. That smoke validates
the adapters and finds immediate clustering bugs. It is not the full atlas representation and
must not be used for response-level path queries. The inventory-based memory estimate decides
whether the full run can safely reuse the current in-memory clustering path or needs the
sparse/blockwise builder.

### 8.4 Feature views and mathematical contract

The first atlas is position-collapsed at the shared basis level. "Position-collapsed" means that
cluster identity is attached to the signed basis feature rather than to an individual occurrence;
it does not mean that response-target identity disappears.

Construct and retain separately:

- input-attribution view keyed by response target/circuit input and local source-token position;
- temporal-trajectory view keyed by response and ordered target position;
- output-contribution view keyed by trace target and candidate output index;
- optional attribution-magnitude view, only if explicitly enabled;
- support/overlap masks so missing observations are not treated as numerical zeros.

Before implementation, freeze a short mathematical feature specification with these defaults:

1. within one target slice, occurrences of the same signed basis are aggregated by signed summation
   of their raw input-attribution vectors;
2. the raw signed vector, absolute attribution mass, occurrence count, activation summary, and
   support mask are retained separately;
3. profile direction uses L2 normalization or absolute-L1 normalization selected before semantic
   inspection; division by the signed algebraic sum is prohibited;
4. zero- or near-zero-norm profiles contribute no directional similarity and retain only their
   explicit magnitude/support measurements;
5. two signed bases receive within-target profile similarity only where both are supported;
   unsupported is missing, not zero;
6. similarities are accumulated with equal family, then response, then target weight;
7. source-token positions are local to a circuit input. Profiles from unrelated prompts are not
   aligned merely because their integer token positions match;
8. all reducers, norms, epsilon values, overlap thresholds, dtypes, and weighting formulas are part
   of the feature-schema hash.

The temporal-trajectory view records, for each signed basis and dense response target:

```text
supported/present
signed and absolute attribution
activation summary
prompt-attribution fraction
generated-prefix-attribution fraction
hint/source-region attribution fraction where predeclared spans exist
in-degree, out-degree, and named path-role participation inside that exact target graph
```

Raw target positions remain available. Cross-response trajectory comparison uses predeclared
response phases and event-relative coordinates rather than treating the same integer position in
different-length responses as semantically aligned:

```text
early reasoning
pre-commitment
commitment window
post-commitment
final answer
```

Annotated hint acknowledgment, source/fabrication, and answer-commitment anchors supply additional
event-relative offsets where available. Phase/anchor construction is versioned and uses only
frozen corpus metadata, never model-generated labels.

The temporal-trajectory view is never the sole clustering evidence. Graph-degree and path-role
features are pruning-dependent and can overemphasize generic high-degree generation features.
Retain input-profile-only clustering as the primary comparator, standardize named trajectory
components separately, and report sensitivity with graph-role components disabled.

Equal-family/response/target weights prevent adjacent dense targets from masquerading as
independent support. Record the exact weighting and compare at least one unweighted sensitivity
run.

Because each current trace has one candidate output, the contribution view must be diagnosed
before it is used:

- report its variance and effective rank;
- report pairwise overlap;
- compare attribution-only versus attribution-plus-contribution assignments;
- reject contribution-derived descriptions when the view is effectively degenerate;
- do not present width-one contribution labels as equivalent to top-five contribution labels.

### 8.5 Clustering sweep

Run a deterministic engineering sweep on dense discovery only. At minimum vary:

- cluster count;
- attribution-only input-profile versus attribution-plus-temporal-trajectory input;
- hierarchical family/response/target weighting versus unweighted features;
- similarity combination rule;
- clustering random seed;
- retained basis support threshold, if one is introduced.

The initial sweep may retain target-count checkpoints approximating 500, 1,000, 1,500, 2,000, and
the full dense core for engineering continuity, but every checkpoint must be constructed and
reported by whole response/base-question-family blocks where possible. Stability resampling and
uncertainty use whole families, not individual target traces. It must not select a checkpoint by
visual appeal.

For each configuration report:

```text
assigned and unassigned signed basis counts
cluster size distribution
distinct families, responses, and targets supporting each cluster
within-cluster profile coherence
between-seed adjusted Rand information or equivalent on shared eligible bases
family-blocked bootstrap/resample stability
dense-versus-broad support balance
cluster survival across corpus checkpoints
runtime and peak host/GPU memory
```

The choice of a frozen cluster state is based on predeclared stability, support, trajectory, and
coherence criteria, not on human-readable labels. Before semantic inspection, freeze:

```text
primary and secondary metrics
minimum family/response/target support
prompt_local and unassigned rules
cluster matching rule across resamples
stability/coherence thresholds or discovery-only decision bands
failure and stop conditions
```

Because this is an exploratory feasibility study, thresholds may be calibrated on label-free dense
structural preflights. Once frozen, they cannot be revised using semantic labels, broad recurrence
results, or holdout outcomes.

#### 8.5.1 Dense structural preflight and initial sweep freeze

The completed `dense-features/compacted` store contains 2,083 target traces, 16,022 signed basis
features, 461,534 summed target-local profile columns, and 89,226,743 supported cells. A global
dense profile matrix would require 29.58 GB in float32 before working memory, so the production
path uses exact target-block computation followed by sparse global reduction:

```text
for target t:
    X_t = signed input-attribution profiles, with unsupported cells masked
    S_t(i,j) = cosine over support(i,t) intersect support(j,t)
global pair evidence = weighted sums plus distinct target/response/family overlap counts
affinity = positive recurring similarity -> deterministic sparse k-nearest-neighbor graph
clusters = normalized sparse spectral embedding -> deterministic KMeans
```

The exact build was measured on the completed store at 102--109 seconds on four CPU cores with
3.35 GB peak RSS. It produced 4,645,518 upper-triangular pair-evidence entries. Unsupported
coordinates and zero-norm intersections contribute no pair evidence; the stored numerical zero is
never interpreted as scientific absence.

Freeze these label-free structural defaults before semantic inspection:

- exclude embedding/unembedding boundary layers from fitting while retaining them as explicitly
  unassigned atlas features;
- require a basis to recur in at least three targets, two responses, and two base-question
  families for the primary fit;
- require a pair to coexist with valid directional profiles in at least two targets, two
  responses, and two base-question families;
- use hierarchical equal-family/equal-response/equal-target weights;
- retain signed similarities in pair evidence, but use only positive similarity as spectral
  affinity;
- construct a deterministic `union_max` 32-nearest-neighbor graph;
- use cluster counts 32, 64, 96, and 128 and seeds 17, 29, and 43;
- keep descriptions disabled for every fit and sweep state.

This primary rule leaves 4,308 eligible non-boundary bases; all 4,308 have a recurring positive
neighbor, and the 32-nearest-neighbor graph is one connected component. Predeclared one-factor
sensitivities use basis target-support thresholds 2 and 5, neighbor counts 16 and 64, target-pair
overlap 3, and unweighted target similarities. The attribution-plus-temporal view is a second
evidence family after the input-only sweep and does not replace the input-profile primary
comparator.

Run the fully crossed cluster-count/seed grid for both hierarchical and unweighted evidence.
Evaluate the other one-factor sensitivities at the frozen reference seed first; expand them across
seeds only if the reference result changes the structural decision band. These are provisional
sweep states, not the frozen scientific cluster state. Selection still requires the stability,
coherence, recurrence, checkpoint, and family-blocked resampling report specified above.

#### 8.5.2 Frozen label-free state-selection rule

Select states without cluster descriptions, token-text inspection, broad-discovery feedback, or
holdout feedback. The primary candidate pool is limited to the hierarchical, input-profile-only,
support-3, pair-overlap-2, 32-neighbor configurations from Section 8.5.1. For each cluster count,
use the seed whose assignment has the highest mean adjusted Rand index (ARI) to the other two seeds
as that resolution's medoid state. Unweighted and one-factor configurations are sensitivities,
not primary candidates.

A medoid state must pass all of these structural gates:

- at least 95% of eligible bases are assigned;
- no cluster contains more than 15% of assigned bases;
- no more than 2% of clusters are singletons;
- mean seed-pair ARI is at least 0.72 and the minimum seed-pair ARI is at least 0.70;
- sparse-affinity modularity is at least 0.20 and observed within-cluster affinity is at least
  1.25 times its degree-volume null expectation;
- at least 90% of assigned bases and 80% of clusters are labelable, where a labelable cluster has
  at least eight signed bases, twenty dense targets, three responses, and three base-question
  families;
- median agreement with the matching unweighted and one-factor sensitivity states is at least
  ARI 0.50 on shared assigned bases;
- under leave-one-family-out refits, median ARI to the full state is at least 0.60 and the
  10th-percentile ARI is at least 0.45 on shared assigned bases.

The last two gates may be reported as `pending` during structural narrowing, but they must pass
before a state becomes labeling-ready. Failure of every resolution triggers a separately
versioned clustering-method refinement; it does not authorize semantic inspection to choose a
state.

Project every gate-passing medoid onto the dense response-time multiplex. Projection attaches the
frozen signed-basis assignment to target-local occurrences and edges without creating cross-target
causal edges. Report:

- per-cluster target, response, and family recurrence and maximum response concentration;
- response-normalized emergence, persistence density, and five-bin temporal-profile coherence;
- target-witnessed cluster-edge support and the fraction of cluster-edge mass recurring across at
  least two responses and two families;
- cluster size, graph conductance, input-profile coherence, and temporal/edge evidence coverage;
- the fraction of clusters and assigned mass that remain labelable after projection.

Among states passing all gates, compute within-candidate percentile ranks and use this predeclared
weighted score:

```text
25% seed and family-blocked stability
25% sparse-graph modularity, affinity enrichment, and conductance
20% size balance and labelable assigned mass
15% target/response/family recurrence and response-concentration control
15% temporal coherence and recurrent target-witnessed edge support
```

The highest-scoring state is primary. The highest-scoring gate-passing state at a different cluster
count is the alternative. If only one cluster count passes, the alternative is the best
gate-passing unweighted state with the same count; otherwise the run is not labeling-ready. Ties
within numerical tolerance use, in order, higher family-jackknife median ARI, fewer singleton/tiny
clusters, and lower cluster count.

The selected state persists assignments, medoid/prototype members, balanced target exemplars,
projection summaries, all metric inputs, and exact source hashes. A cluster below the labelability
rule remains part of the frozen state but is marked `insufficient_labeling_support` and receives no
generated description. Both selected states remain exploratory ADAG objects rather than validated
faithfulness detectors.

### 8.6 Position-aware follow-up

The repository exposes `sum_over_tokens=False`, but the current cluster-map expansion was designed
around token-summed keys. Before any position-aware result:

1. add a focused regression fixture with the same neuron at multiple token positions;
2. demonstrate distinct expected keys and assignments;
3. ensure no lookup through `token=-1` silently collapses them;
4. compare against the position-collapsed baseline;
5. version the resulting feature and cluster schema.

Position-aware clustering is a follow-up analysis, not a prerequisite for the first width-one
atlas.

### 8.7 Dense label-free feasibility report

Before generating descriptions, produce a cluster-ID-only report over the dense core. It must
measure:

```text
feature/cluster emergence target
persistence duration and recurrence
prompt versus generated-prefix attribution trajectory
hint/source/answer-region attribution enrichment against matched control regions
commitment-relative timing
change points around frozen annotation anchors
independently witnessed cluster-path recurrence
family/response support and family-blocked stability
generic token/formatting and response-position confounds
```

The report predeclares candidate-selection and rejection rules derived from the project criteria:
recoverability, timing, controlled contrast, separability from generic copying/formatting,
stability, and graph coherence. A visually attractive cluster or union-only path is insufficient.

The first scientific go/no-go decision is made from this report:

- `promising_dense_representation`: at least one candidate satisfies several predeclared criteria
  with response/family support and exact target witnesses;
- `engineering_valid_but_scientifically_inconclusive`: the multiplex and atlas are correct but
  evidence is unstable, prompt-local, late-only, or confounded;
- `current_representation_not_promising`: reasonable discovery-only settings fail the frozen
  criteria.

The second and third outcomes do not prove absence of the modeled computation. They bound or reject
the current raw-neuron/ADAG representation for this study. Optional bounded top-five probes may
still test whether contribution information addresses a diagnosed width-one limitation, but no
full contribution-aware corpus is automatic.

### 8.8 Width-one labels

The first label bundle is attribution-only and explicitly marked as engineering/exploratory.
Discovery exemplars must be:

- family/response-balanced;
- phase/condition aware;
- target-position aware;
- diverse in token text and context;
- selected without any holdout example;
- stored verbatim with trace and profile references.

Each exemplar bundle also includes label-free temporal evidence:

```text
responses/families supporting the cluster
phase and event-relative occurrence distribution
emergence and persistence summaries
prompt/generated-prefix/hint/source attribution fractions
exact target-witnessed path roles
matched generic-token/formatting controls
```

Generated labels may describe an observed temporal role such as hint extraction, source
acknowledgment, answer commitment, generated-context maintenance, post-hoc justification, or
formatting only when the underlying frozen evidence supports that wording.

Before any model call, split discovery evidence by prompt/base-question family, never by individual
target alone:

1. generation partition:
   - supplies exemplars used to propose candidate descriptions;
2. selection-scoring partition:
   - supplies disjoint examples used to score and choose among a backend's candidates;
3. audit partition:
   - supplies disjoint examples for cross-backend correlation/R² and parse/reliability reporting.

Use the same frozen family-grouped partitions and scoring examples for every backend. Stratify by
corpus role, phase, condition, and prompt family where support permits. No response, paraphrase,
or family may cross partitions merely through a different target position. If a cluster lacks
enough distinct discovery prompts/families for the three-way split, use a predeclared
family-grouped cross-fit or mark its simulator evaluation `insufficient_support`; never fill the
gap with one of the 128 confirmatory targets.

Generate several candidate descriptions per cluster from the generation partition, score them
where a validated simulator is available on the selection-scoring partition, and report the
chosen candidate on the audit partition. The audit results are still exploratory model-selection
evidence, not a new confirmatory set. Summarize only after recording the underlying candidates,
partition IDs, and scores. A short label never replaces the detailed profile evidence.

### 8.9 Width-one atlas outputs

Core dense-feasibility outputs:

```text
inventory.json
atlas-index.parquet
atlas-index.manifest.json
basis-index.parquet
occurrence-index.parquet
attribution-features/
contribution-features/
temporal-trajectory-features/
feature-manifest.json
clustering-sweep.jsonl
stability-report.json
cluster-state.json
cluster-assignments.parquet
dense-label-free-feasibility-report.json
run-manifest.json
```

Later discovery-transport, optional interpretation, and confirmatory outputs:

```text
broad-recurrence-report.json
discovery-exemplars.jsonl
label-bundle.json
holdout-transport-report.json
```

Every output is immutable. A changed cluster setting creates a new cluster-state identity and
output directory.

## 9. Workstream B: dense response-time multiplex

### 9.1 Goal

Represent the independent width-one target traces across each dense response in one queryable
longitudinal structure while preserving the fact that topology and values are target-specific.
The core deliverable is one logical response-time multiplex per dense response inside a common
typed artifact.

The multiplex is downstream of tracing. It may be rebuilt when aggregation, clustering, or
labeling changes without rerunning the Qwen3-4B tracer.

The multiplex adds organization and longitudinal measurements, not new raw model evidence. Its
scientific value comes from tracking the same signed bases and clusters across generation targets:
when they emerge, persist, change source attribution, participate in independently witnessed
paths, and disappear.

### 9.2 Logical tables

Use normalized target-indexed tables rather than one lossy graph:

```text
targets
    trace_unit_id
    example_id
    response_position
    target token/logit/probability
    corpus role and provenance

basis_nodes
    basis_key
    optional frozen cluster_id
    optional frozen label_id

node_occurrences
    occurrence_key
    trace_unit_id
    basis_key
    activation
    attribution
    local node metadata

edge_occurrences
    trace_unit_id
    source occurrence/key
    target occurrence/key
    attribution
    weight
    local edge metadata

aggregated_node_support
    basis/cluster key
    support target set or indexed support table
    explicitly named reductions

aggregated_edge_support
    source basis/cluster key
    target basis/cluster key
    support target set or indexed support table
    explicitly named reductions

longitudinal_correspondence
    example_id
    left and right target response positions
    basis_key or frozen cluster_id
    left and right occurrence-support references
    correspondence_kind = same_basis | same_cluster
    explicitly_noncausal = true

trajectory_measurements
    example_id
    target response position
    basis_key or frozen cluster_id
    support, attribution, activation, source-region fractions
    named graph-role and path-recurrence measurements
    phase and event-relative coordinates
```

Use Parquet/Arrow-compatible tables for scale and deterministic typed schemas. A compact bitmap
may accelerate target-support operations, but the authoritative mapping must remain exportable as
target IDs.

### 9.3 Aggregation semantics

Raw occurrence values are never silently summed across:

- response positions;
- candidate logits;
- prompts;
- conditions;
- polarity;
- discovery and holdout.

Any aggregate column names its reduction and denominator, for example:

```text
support_target_count
support_prompt_count
mean_abs_attribution_over_supported_targets
median_signed_attribution_over_supported_targets
support_weighted_positive_fraction
```

Missing support is distinct from a supported zero.

No longitudinal correspondence row appears in `edge_occurrences`. Cross-time recurrence is a
separate relation and cannot be consumed accidentally by the ordinary graph traversal engine.

### 9.4 Path-query contract

Implement target filtering first, then graph traversal:

```text
query_path(source, destination, target_filter=None)
    -> projected path plus exact witnessing target IDs and occurrence paths
```

The implementation must intersect target support and preserve exact occurrence-endpoint
continuity at every traversal step. It may optimize target intersection with bitmaps, but it
cannot perform an unrestricted union or contracted-basis traversal and filter only at the end.

Required adversarial fixture:

- target A contains edge `u -> v`;
- target B contains edge `v -> w`;
- neither contains both edges;
- the union graph appears to contain `u -> v -> w`;
- the multiplex query must return no valid path.

Required same-target contraction fixture:

- one target contains edge `u -> v_at_position_1`;
- the same target contains edge `v_at_position_2 -> w`;
- both `v` occurrences map to the same basis neuron or cluster;
- no occurrence-continuous `u -> v -> w` path exists;
- the basis/cluster projection must return no valid path.

Also test:

- a path supported by exactly one target;
- the same path supported by multiple targets;
- opposite signed/polarity occurrences;
- response-position filtering;
- cluster-level contraction with target support retained;
- round-trip export back to an exact sampled per-target graph.

Required longitudinal fixtures:

- one signed basis recurs in adjacent target slices and produces a correspondence but no graph
  edge;
- a cluster recurs through different member bases and is marked `same_cluster`, not `same_basis`;
- an apparent path split over adjacent target slices is reported as a recurrent pattern with two
  witnesses, never as one occurrence-continuous path;
- missing target support creates a trajectory gap rather than a numerical zero;
- response phase and event-relative coordinates round-trip without changing raw target position.

### 9.5 Cluster, trajectory, and label projection

The multiplex stores cluster IDs as a derived annotation:

```text
occurrence -> basis_key -> frozen cluster_id -> frozen label bundle
```

It does not write labels into source traces. Relabeling produces a new projection manifest, not a
new raw multiplex.

The data model and viewer can be developed with synthetic or placeholder cluster IDs. Scientific
cluster projection waits for the frozen discovery atlas. Holdout nodes receive assignments only
through the frozen apply rule.

After cluster projection, derive but do not overwrite:

```text
cluster emergence and last-supported target
supported-target persistence and gap statistics
prompt/generated-prefix/hint/source attribution trajectories
commitment-relative peak position
per-target path roles
independently witnessed recurrent cluster paths
```

These derived trajectory summaries are clustering and labeling evidence only when their generating
feature/cluster state was frozen before the evaluation partition was opened.

### 9.6 Visualization

Provide three explicit views:

1. per-target graph:
   - exact source topology and values;
   - occurrence nodes;
   - cluster colors/labels as annotations;
2. response/atlas summary:
   - support counts and named reductions;
   - filters for example, response position, phase, condition, cluster, polarity, and role;
   - path queries that display surviving targets.
3. response-time trajectory:
   - ordered target slices for one dense response;
   - emergence, persistence, source-attribution, and path-role tracks;
   - raw position plus phase/event-relative alignment;
   - an explicit visual distinction between within-target graph edges and cross-target
     correspondence.

The response summary must visually distinguish:

- one target;
- several targets with shared support;
- an aggregate with no single jointly supporting target.

The third case cannot be drawn as an ordinary causal path.

Likewise, a sequence of corresponding bases or clusters across target slices cannot be drawn with
the same edge style or described as a model-computation path.

### 9.7 Dense response-time multiplex outputs

Required dense outputs:

```text
multiplex-manifest.json
targets.parquet
basis-nodes.parquet
node-occurrences.parquet
edge-occurrences.parquet
aggregated-node-support.parquet
aggregated-edge-support.parquet
longitudinal-correspondence.parquet
trajectory-measurements.parquet
recurrent-path-witnesses.parquet
phase-and-event-alignment.json
path-query-validation-report.json
per-target-round-trip-report.json
viewer-export/
run-manifest.json
```

Cluster and label projections are separate derived manifests so the raw target-slice multiplex can
be reused when clustering or labeling changes.

## 10. Workstream C: contribution-aware top-five tracing

### 10.1 Scope

"Top five" means five candidate output logits at the same teacher-forced prediction position. It
does not mean five different response-token positions and does not create a response-wide trace.

Each top-five artifact remains associated with exactly one:

```text
example
response position
prediction position
teacher-forced prefix
candidate set
independent candidate references
exact union and fixed-union measurement contract
```

If C0–C2 pass and a full matched corpus is explicitly authorized, it should cover the same 2,594
completed target positions as the width-one corpus. The pathological missing position 663 remains
excluded. Under the frozen model-top-five-plus-observed policy, candidate zero supplies paired
actual-token evidence without changing prompt or target-position selection. The realized
candidate width is five when the observed token is already in the model top five and six
otherwise.

### 10.2 Candidate-policy decision

The primary candidate rule is:

1. `model_top5_plus_observed`:
   - preserve all five highest-logit tokens at the shared prediction position;
   - always include the stored teacher-forced response token;
   - realize width five when those sets overlap and width six otherwise.

For this policy record:

```text
candidate rank under full model distribution
candidate index within the stored vector
token ID
token text
logit
probability
whether it is the observed response token
observed token's full-distribution rank
tie-breaking rule
```

Place the observed token at index zero and order the remaining model-top-five tokens by stable
descending logit with token ID as the deterministic tie-breaker:

```text
candidate[0] = observed teacher-forced token
candidate[1:] = model top-five tokens excluding the observed token
candidate_count = 5 if observed rank <= 5 else 6
```

The manifest freezes `candidate_count_min=5`, `candidate_count_max=6`, and the explicit rule
`5_if_observed_in_model_top5_else_6`; every artifact records its realized count. The C1 report
must state how frequently the observed token falls outside the model top five and stratify
resources and topology results by realized width.

Pure `model_top5` remains a discovery-only sensitivity rule. The earlier
`observed_plus_top4_alternatives` rule remains readable for C0 smoke compatibility but is not the
C1 production policy because it drops the fifth-ranked model candidate when the observed token
falls outside the top five. Do not mix policies under one trace-family ID.

### 10.3 Frozen topology and contribution semantics

C0 closed the topology decision. Raw unweighted-logit-sum and
observed-versus-alternatives joint graphs remain versioned diagnostic artifacts, but neither is
the production candidate-comparison family. The locked approach is a separately versioned
candidate-specific union:

```text
candidate[0..N-1]
    -> N independent specified-token k=1 traces
    -> exact union of independently retained nodes and exact retained edges
    -> N candidate-specific fixed-union rescoring traces
    -> one dense candidate-union artifact
```

The contract is:

- topology membership is established only by the independent candidate traces;
- the union contains exact retained edges, not an induced all-pairs graph over union nodes;
- pass two bypasses node and edge pruning on that frozen union and preserves measured zero;
- internal and embedding-to-MLP union edges are applicable to every candidate;
- a terminal logit edge is applicable only to its corresponding candidate;
- each node and edge stores candidate-specific measurements, applicability, and original
  `selected_by_candidate` membership;
- the independent references and fixed-union rescoring artifacts remain separately checksummed
  and resumable.

Changing the candidate policy, union rule, applicability rule, or refinement measurement
semantics creates a new trace-family and schema version. It may not silently revise this family.

C0 found substantial opposing candidate effects and substantial topology loss in the scalar joint
graphs. Across the locked union artifacts, pass two recovered 5,208 node-candidate and 880,789
edge-candidate measurements that were absent from the corresponding independent graph. This is
why missing observations remain missing in pass one and are measured explicitly in pass two
rather than filled with zero or inferred from a pruning cutoff.

For candidate comparisons, pass-two values are the canonical common-topology profiles. Pass-one
membership and values remain audit evidence. The C0 audit reproduced every selected node value
exactly. A one-case bfloat16/topology-batching sensitivity affected 742 of 618,302 selected edge
entries; it is recorded in `docs/CANDIDATE_UNION_C0_RESULTS.md` and must remain visible in later
numerical-health reports.

### 10.4 Observed-token `k=1` parity gate

Before a top-five probe, run a new compatibility mode with exactly the stored observed response
token as the sole candidate. On a deterministic representative set compare it to the existing
width-one artifacts:

- target and candidate provenance;
- selected node identities;
- selected edge identities;
- scalar node and edge values;
- target logit and probability;
- contribution vector shape and values;
- graph/dataframe ordering after canonicalization;
- compact serialization and reloading;
- instrumentation fields that should remain invariant.

Define tolerances by dtype and existing numerical behavior. Structural mismatch, provenance
mismatch, unexplained value mismatch, or serialization drift blocks top-five probing.

This parity mode is not `model_top1`; it deliberately selects the observed token so the old and
new interfaces address the same logit.

### 10.5 Top-five schema

The implementation must add a candidate axis that is distinct from the response-target-position
axis. It is invalid to represent several candidates by repeating the same prediction position
through the current multi-position interface. The compact artifact still has one scientific
response target position, while realized `candidate_count` describes the five- or six-column
output-profile width.

Every artifact adds:

```text
trace_family_id
candidate_policy_id
candidate_policy_version
joint_objective_id
joint_objective_formula
shared_prediction_position
shared_response_position
observed_token_id/text/rank
candidates[]
    candidate_index
    full_distribution_rank
    token_id/text
    logit/probability
    observed-token membership
candidate_contribution_schema
```

Validation rejects:

- duplicate candidates;
- nondeterministic ordering;
- candidate records from different prediction positions;
- a contribution vector whose width differs from the candidate count;
- mixed policy/objective IDs;
- multi-response-position artifacts marked scientifically reusable;
- missing model/tokenizer/chat-template provenance.

Expected implementation touchpoints are:

- `circuits/tracing/trace.py` for same-position candidate selection, scoring, provenance, and
  dataframe conversion;
- `circuits/tracing/clja.py` and its attribution helpers for a candidate axis that is not confused
  with target position, and for candidate-resolved neuron contribution maps;
- `circuits/tracing/artifact.py` for a new schema and reuse rule that distinguishes one target
  position from five candidates;
- `scripts/bonafide/runner.py` plus manifest/execution-plan validation for the new family;
- `circuits/analysis/circuit_ops.py` and labeling consumers for named candidate metadata rather
  than an anonymous positional vector.

Legacy artifacts must remain readable under their existing schema, while old consumers must fail
clearly rather than silently misread a new candidate-axis artifact.

### 10.6 Candidate-specific reference and staged probe design

Select a versioned, paired probe set from the completed width-one targets containing:

- short, medium, and long input contexts;
- dense and broad discovery targets;
- low, median, high, extreme, and near-pathological screened workloads;
- early, middle, and late response positions;
- semantic, punctuation, commitment, source/fabrication, and control tokens;
- cases where the observed token is inside and outside model top five.

All candidate-policy, objective, resource, and label-stability probes are discovery-only. The 128
holdout positions may enter the matched top-five corpus only after the candidate policy, joint
objective, trace configuration, execution plan, discovery cluster procedure, and labeling
procedure are frozen. Holdout traces then receive the same frozen processing and remain
coverage/transport/trajectory/steering evaluation only.

Use three gates:

1. C0 candidate-specific reference:
   - 8–12 deliberately selected discovery targets;
   - trace each selected candidate independently with `k=1`;
   - compare raw-sum, centered/contrastive, and candidate-specific-union topology semantics;
   - completed on ten targets; the candidate-specific union was selected and fixed-union
     node/edge refinement was validated;
2. C1 policy/resource probe:
   - 24–48 family/response-balanced discovery targets;
   - use the frozen `model_top5_plus_observed` policy;
   - deliberately include both realized-width-five and realized-width-six cases;
   - execute the locked independent-trace plus exact-union-refinement contract;
   - retain a small pure-`model_top5` sensitivity subset only if separately versioned and
     explicitly approved, never as a competing production policy;
   - establish end-to-end and per-candidate runtime, HBM, RSS, independent/union graph size,
     serialization, resume, and numerical bounds;
3. C2 scientific-utility pilot:
   - approximately 200–300 family/response-balanced discovery targets, with the exact count and
     family/response membership frozen before execution;
   - construct a pilot contribution feature store;
   - test whether candidate profiles add non-degenerate, reproducible information beyond
     width-one attribution and temporal trajectories.

No full matched corpus is authorized by C0 or C1 alone.

For every paired width-one/top-five target report:

```text
total and per-stage wall time
peak GPU allocated/reserved and headroom
peak host RSS
candidate MLP edges and Jacobian chunks
raw and compact node/edge counts
retained-attribution proxy
candidate-vector effective rank and sign diversity
observed-token rank
independent selection membership and fixed-union dense-measurement coverage
artifact size
numerical validity
```

Do not extrapolate from width-one runtime by multiplying by five. Measure it.

### 10.7 Full top-five launch gates

All must pass:

- observed-token `k=1` parity;
- candidate ordering and schema tests;
- opposing-effect/cancellation fixture and C0 empirical candidate-specific comparison;
- exact independent-union topology and fixed-union node/edge refinement validation;
- representative probe completion without OOM or integrity failure;
- runtime and memory model with clearly bounded extrapolation;
- C2 contribution-vector effective-rank/sign-diversity report;
- C2 evidence that contribution profiles add or materially change a predeclared discovery-only
  stability, coherence, separability, or trajectory metric;
- C2 evidence that the frozen candidate-union profile is numerically stable and scientifically
  useful under the predeclared comparison;
- frozen prompt/target membership;
- frozen candidate policy and candidate-union contract;
- new config, manifest, execution plan, artifact root, and cohort lock;
- successful dry run and `sbatch --test-only`;
- explicit review of exact Slurm parameters.

Failure of the C2 scientific-utility gate stops the full top-five extension. It does not block the
dense width-one atlas, response-time multiplex, broad recurrence analysis, or limited candidate
probe report.

The new run must inherit the existing fail-closed behavior:

- validate all hashes before model load;
- one resident model per task;
- one atomic artifact per target;
- checksum validation on resume;
- one append-only summary per task attempt;
- no shared multi-writer JSONL;
- stop after OOM, resource gate, ordinary error, or pre-timeout signal;
- retry failed rows only after root-cause review.

### 10.8 Contribution-aware atlas

Do not overwrite the width-one cluster state. The top-five family produces a new feature-store and
cluster-state identity. Required comparisons:

- width-one attribution-only clusters versus top-five attribution-only clusters;
- top-five attribution-only versus top-five multi-view clusters;
- cluster stability and support;
- whether contribution profiles add separability beyond attribution;
- mapping overlap between width-one and top-five clusters without forcing one-to-one identity;
- label stability across candidate policies on the probe subset.

The contribution-aware atlas remains an optional extension. It is not required to declare the
dense response-time feasibility phase complete.

### 10.9 Candidate-aware local clustering salvage

The failed C2 temporal-retrieval gate does not answer whether non-degenerate candidate profiles
improve local functional clustering or labelability. The completed candidate view has substantial
effective rank and sign diversity, while its post-hoc standalone retrieval signal is above null
but largely explained by support applicability. Before rejecting the local contribution view, run
one separately frozen post-hoc comparison on the existing 245 C2 discovery targets only.

The comparison rebuilds the production width-one source-attribution view directly from those
compact traces and compares matched width-one, candidate-direction, late-fusion, and support-only
cluster states. It requires shared signed-basis eligibility, held-out-family directional
coherence, a support-only state, and vector-permutation nulls before any descriptions are accepted.
If the label-free functional gate passes, a matched factorial labeling pilot distinguishes the
effect of adding candidate evidence from the effect of reclustering with that evidence.

The full frozen contract, thresholds, prompt boundary, and success rule are in
`docs/CANDIDATE_AWARE_CLUSTERING_LABELABILITY_PROTOCOL.md`. This salvage neither changes the C2
decision nor authorizes more tracing; it first tests whether already-computed local competition
measurements solve the present cluster-labeling evidence problem.

## 11. Workstream D: explanation, simulation, and Qwen serving

This workstream uses Qwen3.5/Qwen3.6 only as downstream explanation, prompted-simulation, and
summarization models. It does not authorize tracing those models or comparing their neurons to the
frozen Qwen3-4B corpus. Tracing a Qwen3.5 hybrid-MoE model would be a separate scientific and
engineering project with a new tracer compatibility audit, manifest, model revision, neuron/basis
schema, corpus, and resource plan.

This workstream is not on the critical path for the dense label-free feasibility result. Manual
inspection with frozen cluster IDs and evidence tables comes first. Model-generated descriptions
are attempted only after the dense cluster state and exemplar contract are frozen.

### 11.1 Current model roles

The current downstream description stack has several roles that must not be conflated.

| Role | Current default | What it does |
| --- | --- | --- |
| Attribution explanation generation | `Transluce/llama_8b_explainer` through in-process vLLM, or Haiku API | Reads highlighted input-token exemplars and proposes neuron/cluster descriptions |
| Attribution explanation scoring | `Transluce/llama_8b_simulator`, or Haiku API | Predicts per-token activation patterns from a candidate description, then computes correlation/R² |
| Contribution explanation generation | Haiku API | Reads prompt/continuation candidates with normalized contribution scores and proposes descriptions |
| Contribution explanation scoring | Haiku API | Predicts contribution scores for held-out exemplars and supports correlation/R² selection |
| Final cluster summarization | Opus API | Combines attribution descriptions, contribution descriptions, neuron descriptions, and exemplars into one short label per cluster |

Haiku is therefore doing both generative interpretation and numeric simulation/scoring in the
API path. Opus is doing a final compression/judgment task, one call per cluster, not tracing or
clustering.

These tasks can be replaced, but replacement difficulty differs:

- explanation generation is ordinary structured language-model inference and is the easiest role
  to move to a strong local instruction model;
- final summarization is also ordinary generation and probably does not require an Opus-class
  model once the input evidence is structured;
- numeric simulation/scoring is the demanding role because reliability is measured by predicted
  activation/contribution values, not label fluency;
- the specialized Transluce simulator remains a valuable reference because it is trained for the
  scoring protocol.

### 11.2 Why the current simulator is not a stock vLLM-server drop-in

`FinetunedSimulator` does not simply generate text. It:

- uses a specific Llama tokenizer and special-token remapping;
- applies left-padding and custom position IDs;
- reads raw logits at each input token position;
- interprets vocabulary positions 0 through 10 as activation scores;
- computes expected activations on a 0–10 scale.

A stock OpenAI-compatible chat endpoint does not expose that full per-input-position raw-logit
protocol. Keep the current simulator in process as a specialized reference, or replace it with a
prompted numeric simulator and validate the replacement against held-out correlation/R². Do not
route it through ordinary chat completions and call the result equivalent.

### 11.3 Local model candidates

Current model names and external benchmark figures are time-sensitive and must be refreshed when
the evaluation is frozen.

First operational candidate:

- `Qwen/Qwen3.6-35B-A3B`;
- substantially smaller total and active parameter count;
- use it first to validate the endpoint, prompt schemas, parsing, generation, and summarization at
  lower deployment cost.

Capability escalation:

- `Qwen/Qwen3.5-122B-A10B`;
- the official name is 122B total / 10B active, not 112B;
- stage it after the 35B path works and use it only to test whether the larger deployment buys a
  measurable explanation, simulation, or summarization improvement.

Capability ceiling:

- `Qwen/Qwen3.5-397B-A17B-FP8`;
- use only if an appropriate full H200-class node is available and the 122B pilot leaves a real
  unresolved quality gap;
- this is not required for the first production labeling pass.

Specialized reference:

- `Transluce/llama_8b_simulator` for attribution simulation;
- `Transluce/llama_8b_explainer` where tokenizer/model compatibility is acceptable.

External controls:

- Haiku for attribution/contribution generation and scoring on a small frozen subset;
- Opus for final summarization on a small frozen subset;
- external calls are comparison evidence, not the default full-corpus path.

As of 2026-07-26, Artificial Analysis reports an Intelligence Index of 32 for the reasoning modes
of both Qwen3.5-122B-A10B and Qwen3.6-35B-A3B. This general benchmark does not establish
superiority on activation simulation or cluster labeling. The project-specific frozen comparison
is the decision-maker.

### 11.4 Separate serving environment

The tracing environment pins `vllm==0.7.3`. The official Qwen3.5 model card currently requires a
main/nightly vLLM build. Therefore:

- do not upgrade the locked tracing environment;
- first check the available CHPC `vllm/current` module and capture its exact vLLM, CUDA, PyTorch,
  driver, and Python versions;
- if compatible, freeze the module/environment identity in the server manifest;
- otherwise build a separate uv environment or immutable Apptainer image under scratch/group
  storage;
- never download or upgrade model/runtime dependencies inside a production Slurm run;
- stage and hash model/tokenizer files before launch;
- keep server logs, generated outputs, and model caches out of the repository.

At plan creation, CHPC exposes a `vllm/current` module, but compatibility has not been tested.
None of Qwen3.5-122B-A10B, Qwen3.6-35B-A3B, Qwen3.5-397B-A17B-FP8, or the two Transluce 8B
description models was found in the configured project Hugging Face cache. No Anthropic or OpenAI
API key was present in the sourced environment. These are readiness observations, not failures.

### 11.5 Server contract

Serve through an OpenAI-compatible endpoint with:

- text-only mode (`--language-model-only`) to omit the vision encoder;
- a deliberately capped initial context, likely 16K–32K rather than the advertised maximum;
- tensor/expert parallel settings chosen from a measured load probe;
- prefix caching benchmarked enabled and disabled because the hybrid-cache support is still
  described as experimental;
- reasoning parser enabled where needed;
- per-request control of reasoning/thinking mode;
- fixed chat template and tokenizer revision;
- health, model-info, and deterministic smoke endpoints;
- no credentials in command arguments or manifests.

The model card's eight-GPU, 262K-context command is an example, not our initial resource request.
Our workload should first measure the actual rendered exemplar lengths and cap context
accordingly.

### 11.6 Hardware staging

The following are hypotheses to test, not frozen requests:

| Model/checkpoint | First load probe | Reason |
| --- | --- | --- |
| Qwen3.6-35B-A3B | one large-memory GPU if supported | Cheap endpoint/client and prompt validation |
| Qwen3.5-122B-A10B FP8 | two H200s or an evidence-backed H100 configuration | Main capability candidate with runtime headroom |
| Qwen3.5-122B-A10B BF16 | at least a multi-GPU H200/H100 probe | Weight memory alone is not a safe capacity estimate |
| Qwen3.5-397B-A17B-FP8 | full appropriate H200 node; official vLLM recipe is verified on eight H200s | Capability ceiling only |

At each launch decision:

1. run the CHPC GPU recommender with the intended GPU count and wall time;
2. capture live queue and availability evidence;
3. request explicit host memory, not only default memory-per-core;
4. verify `CUDA_VISIBLE_DEVICES` and `nvidia-smi` inside the allocation;
5. monitor utilization and GPU memory;
6. record model-load peak, KV-cache allocation, steady-state headroom, tokens/s, and failures;
7. revise tensor parallelism or context length from measurements.

Do not assume that a checkpoint fitting by raw weight bytes leaves adequate runtime/KV/cache
headroom.

### 11.7 Prompted Qwen roles

Use non-reasoning mode first for:

- attribution description generation;
- contribution description generation;
- final short cluster labels.

Use and compare reasoning mode for:

- prompted numeric attribution simulation;
- prompted numeric contribution simulation;
- difficult ambiguity adjudication where the non-reasoning output is unstable.

All numeric outputs use strict JSON schemas, bounded numeric ranges, token-index alignment, and
parse/retry limits. Preserve raw model text even when parsing fails.

### 11.8 Frozen model-comparison pilot

Select 8–16 discovery clusters after the dense-first width-one cluster state and label-free report
are frozen. Stratify them by:

- cluster size;
- distinct-prompt support;
- attribution coherence;
- broad/dense balance;
- apparent ease or ambiguity under manual inspection;
- positive/negative polarity;
- early/middle/late layer.

Use identical prompts/exemplars and a fixed sampling schedule. First validate the complete
evaluation path with Qwen3.6-35B-A3B and the specialized simulator. Then compare:

1. Qwen3.5-122B-A10B;
2. Qwen3.6-35B-A3B;
3. the Transluce specialized simulator where applicable;
4. Haiku generation/scoring on a smaller matched control;
5. Opus summaries on a smaller matched control.

The comparison reuses the frozen discovery-only, family-grouped generation,
selection-scoring, and audit partitions from Section 8.8. Each backend receives identical
partition membership and equivalent rendered evidence. Backend choice and prompt revisions may
use the audit result only with the explicit status "exploratory model selection"; the
confirmatory 128-target partition remains sealed.

Metrics:

```text
schema/parse success
refusal and empty-output rate
retry rate
attribution simulator correlation and R²
contribution simulator correlation and R²
description score under the same evaluator
label agreement and stability across seeds/exemplar resamples
blind human preference and error categories
latency and throughput
peak HBM and host memory
GPU-hours
input/output/reasoning token totals
measured or estimated API cost
```

The local winner must meet a quality floor, not merely be free of API charges. A sensible outcome
may be:

- local Qwen for high-volume generation and summarization;
- specialized Transluce simulator for attribution scoring;
- a small external API sample for calibration/audit;
- no Opus full-corpus dependency unless it shows a material blind-evaluation advantage.

### 11.9 Cost comparison

For every backend compute:

```text
API cost =
    uncached_input_tokens / 1e6 * input_price
  + cache_read_tokens / 1e6 * cache_read_price
  + cache_write_tokens / 1e6 * cache_write_price
  + output_tokens / 1e6 * output_price
  + any provider-specific reasoning or storage charges

local compute =
    allocated_GPU_count * allocation_elapsed_hours
```

Cached-token categories are mutually exclusive accounting buckets, not additions to a total input
count that already includes them. When a provider reports only total input plus cache reads,
derive uncached input as `total - cache_read` only when the provider's usage semantics explicitly
support that calculation; otherwise report an interval or unknown component rather than
double-counting.

Keep local GPU-hours and monetary API cost as separate quantities unless an official internal GPU
rate is supplied. Also report useful-output cost:

```text
cost per parsed explanation
cost per scored candidate
cost per accepted cluster label
GPU-hours per accepted cluster label
```

The full-run estimate is derived from actual pilot token counts and success rates. Do not estimate
from vendor benchmark verbosity alone.

Time-sensitive reference snapshot:

- official Qwen3.5-122B model card:
  <https://huggingface.co/Qwen/Qwen3.5-122B-A10B>
- official vLLM Qwen recipe:
  <https://github.com/vllm-project/recipes/blob/main/Qwen/Qwen3.5.md>
- Artificial Analysis Qwen3.5-122B:
  <https://artificialanalysis.ai/models/qwen3-5-122b-a10b/>
- Artificial Analysis Qwen3.6-35B:
  <https://artificialanalysis.ai/models/qwen3-6-35b-a3b>

## 12. Immutable checkout and execution strategy

### 12.1 Development phase

Implement and review shared changes on a normal development branch. Until the shared schemas and
tests stabilize:

- do not launch a provenance-bound full atlas, top-five corpus, or production label run;
- small read-only inventory checks and tiny deterministic fixtures are allowed;
- model/GPU probes use explicit project launchers, not a mutable interactive development tree.

Commit cohesive checkpoints after focused validation.

### 12.2 Freeze points

Create clean immutable worktrees only after each track reaches code-complete status. Suggested
logical identities:

```text
dense-downstream-width1-v1
top5-trace-v1
qwen-labeling-v1
```

The joint dense worktree carries two independent production build plans and output roots:
`dense-atlas-width1-v1` and `response-time-multiplex-v1`. Sharing an executable snapshot and
artifact-read pass does not merge their derived scientific identities.

The actual directories and commits are recorded in run manifests. A worktree is frozen when:

- its commit is named;
- `git status --porcelain` is empty;
- relevant source-tree hash is recorded;
- a lane-specific environment or immutable container is built from that checkout's lock;
- environment/config/input hashes are recorded;
- output root is new and empty or contains only checksum-valid resumable artifacts;
- launcher dry-run passes.

An immutable worktree is not sufficient by itself. `scripts/chpc_env.sh` defaults every checkout
to the shared environment
`/scratch/general/vast/$USER/circuits/envs/circuits-py312`. A `uv sync` from the main checkout
could therefore mutate the executable environment of an otherwise frozen worktree. Before
sourcing the helper for a frozen lane, set a unique absolute `UV_PROJECT_ENVIRONMENT`, for example:

```text
/scratch/general/vast/$USER/circuits/envs/dense-downstream-width1-v1-<lock-hash>
/scratch/general/vast/$USER/circuits/envs/top5-trace-v1-<lock-hash>
```

Build it with the lane's frozen `uv.lock`, record the lock hash and installed-package inventory,
and do not run `uv sync` against it after the run manifest is sealed. The uv download cache may
remain shared; the installed environment may not. The Qwen serving lane uses its separately
frozen serving environment or container.

The Hugging Face cache may be shared only for exact revision-pinned, hash-validated snapshots.
The server/trace manifest records the resolved snapshot path and hashes, and active runs fail if
that identity changes.

### 12.3 Active-run rule

While a provenance-bound run is active:

- do not edit its executable checkout;
- do not change its branch/HEAD;
- do not upgrade, sync, or otherwise mutate its lane-specific environment;
- do not replace cached weights/tokenizer files;
- do not relocate or rewrite artifacts being produced;
- do not have two writers target one summary or artifact directory.

The main checkout may continue other work, including Qwen deployment/client development, because
the active jobs execute from immutable worktrees. Before borrowing shared caches or output roots,
verify that the change cannot affect the running cohort.

### 12.4 Parallel launch pattern

After code and probe gates:

| Lane | Immutable input | Initial executable work | Can overlap with |
| --- | --- | --- | --- |
| A: dense atlas | Completed dense trace corpus | inventory, input/trajectory features, clustering checkpoints | B, C, D |
| B: top-five extension | New top-five worktree/config | `k=1` parity, C0 reference, C1 resource probe, later C2 | A, C, D |
| C: response-time multiplex | Completed dense trace corpus and schema | fixture tests, small assembly, then full dense assembly | A, B, D |
| D: optional model serving | Frozen server env/model revision | 35B endpoint smoke, later quality pilot/122B escalation | A, B, C |

Concurrency is conditional on independent GPU, CPU, host-memory, and I/O capacity. In particular:

- atlas sparse-feature construction may be CPU/RAM/I/O heavy;
- spectral clustering may be memory heavy even without model inference;
- top-five tracing loads Qwen3-4B and uses tracing GPUs;
- Qwen3.5 serving may reserve multiple H100/H200 GPUs;
- simultaneous jobs must not create VAST I/O contention that invalidates timing or risks
  incomplete writes.

Resource-aware staggering is preferable to nominal concurrency when the same filesystem or H200
node is the bottleneck.

## 13. Proposed milestone order

### Milestone 0: baseline audit

Deliver:

- deterministic 2,594/2,595 inventory;
- source artifact integrity report;
- exact discovery/holdout/excluded lists;
- current checkout and environment snapshot.

Gate: no corrupt, unexpected, or mixed-provenance input.

### Milestone 1: shared contracts

Deliver:

- atlas index schema and builder;
- signed basis/occurrence adapters and polarity-preserving aggregation;
- family/response/target weighting and partition contract;
- holdout partition enforcement;
- response-time target-slice, support, longitudinal-correspondence, and trajectory schema;
- exact mathematical feature specification;
- deterministic fixtures and focused tests.

Gate: all provenance, signed-polarity, feature-reducer, missing-support, per-target round-trip,
false-cross-target-path, false-cross-time-path, and holdout-firewall tests pass.

Top-five provenance schemas, provider-neutral model interfaces, and the usage ledger may be
developed in parallel, but they are not part of this milestone's dense critical-path gate.

### Milestone 2: dense preflight and optional extension preflights

Lane A:

- stream a small response-balanced dense subset;
- measure feature dimensions and memory;
- run input-profile-only and input-plus-temporal clustering with descriptions disabled.

Lane B:

- observed-token `k=1` parity and C0 candidate-specific union refinement are complete;
- freeze and run C1 only after its exact discovery cohort and resource contract are reviewed;
- do not authorize C2 or a full corpus from C1 resource success alone.

Lane C:

- assemble a small dense response-time multiplex;
- prove exact per-target round trip, support-intersection path behavior, longitudinal
  correspondence, and trajectory-gap semantics.

Lane D:

- determine whether `vllm/current` supports Qwen3.5;
- if this optional lane is prioritized, stage and smoke the 35B control first;
- do not stage the 122B snapshot until a frozen label-pilot need and quality comparison justify it.

Gate: Lane A and Lane C independently demonstrate correctness and measured resource headroom.
Failure or deferral of Lane B or D does not block the dense feasibility phase.

### Milestone 3: dense response-time atlas and label-free feasibility

Deliver:

- full 11-response dense target-slice multiplex;
- dense input-profile and temporal-trajectory feature stores;
- clustering sweep and stability report;
- frozen dense-first width-one cluster state;
- cluster projection and trajectory/recurrent-path summaries;
- dense label-free feasibility report and explicit go/no-go classification.

Gate: cluster state is selected without generated labels, broad-recurrence feedback, or holdout
feedback; no false cross-target/cross-time path exists; and the report applies the predeclared
recoverability, timing, contrast, separability, stability, and graph-coherence criteria.

### Milestone 4: discovery transport, optional labels, and holdout

Deliver:

- broad-discovery recurrence/transport report under the frozen dense-first state;
- separately versioned dense-plus-broad sensitivity fit only if predeclared;
- manual cluster-ID review with exact evidence;
- optional frozen model comparison and attribution/trajectory label bundle;
- frozen confirmatory metric contract;
- holdout transport report with coverage, prototype coherence, temporal recurrence, witnessed graph
  recurrence, and eight per-family outcomes.

Gate: broad results and generated labels do not revise the dense cluster state; holdout does not
affect fitting, hyperparameters, exemplars, labels, or thresholds; exact basis lookup is not
misreported as assignment accuracy.

### Milestone 5: optional contribution-aware C0–C2 decision

Deliver:

- observed-token `k=1` parity report;
- C0 independent-candidate versus joint-topology report;
- C1 candidate-policy/resource report;
- C2 family/response-balanced scientific-utility feature and stability report;
- frozen candidate-policy and candidate-union decision, or an explicit stop decision.

Gate: all Section 10.7 scientific, topology, numerical, and resource gates pass. Otherwise stop the
full top-five extension while retaining the bounded probe evidence.

### Milestone 6: conditionally authorized matched top-five corpus

Before launch, present exact Slurm parameters for explicit approval.

Deliver:

- frozen candidate-policy decision;
- frozen manifest/execution plan/cohort;
- validated matched top-five artifacts;
- performance and completeness report;
- failed-only retry record if necessary.

Gate: integrity and numerical-health audit; no silent mixing with width-one artifacts; the corpus
uses exactly the C2-approved candidate policy and objective.

### Milestone 7: optional contribution-aware atlas

Deliver:

- top-five sparse feature store;
- attribution-only and multi-view clustering comparison;
- contribution-aware labels;
- mapping/stability comparison to the width-one atlas;
- updated multiplex projection as a new derived version.

Gate: the full contribution view reproduces the C2 non-degeneracy/utility result and improves or
meaningfully changes a predeclared metric.

### Milestone 8: limited causal checks

Only for stable, interpretable candidates:

- predeclare position and target;
- choose size/layer/attribution-matched random controls;
- run position-restricted ablation or patching;
- report effect sizes and null/control distribution;
- preserve the distinction between causal importance and faithfulness.

## 14. Acceptance matrix

| Layer | Required evidence | Blocking failure |
| --- | --- | --- |
| Source integrity | All expected completed payload hashes and metadata validate | Corrupt, unexpected, or mixed source |
| Atlas indexing | Deterministic logical IDs, signed basis keys, and exact local/global mapping | Lost target, prompt, CI, polarity, or logical/content identity |
| Statistical unit | Equal family/response/target weights and family-blocked resampling/partitions | Adjacent targets treated as independent inferential samples |
| Feature construction | Frozen reducers/norms/weights, sparse round trip, and explicit missing-data semantics | Signed-sum normalization instability or missing treated as zero |
| Holdout | Automated firewall and audit log | Any holdout-derived fit, label, or hyperparameter input |
| Clustering | Dense family-level stability/support, input-profile/trajectory coherence, and reproducibility | Selection based only on attractive labels/plots or target-level pseudoreplication |
| Dense feasibility | Label-free recoverability, timing, contrast, separability, stability, and witnessed graph coherence | Scientific conclusion based on labels, union-only paths, or late answer copying alone |
| Width-one labeling | Attribution/trajectory evidence, exact exemplars, and width-one contribution limitation | Contribution claim from a degenerate view or temporal role unsupported by target witnesses |
| Response-time multiplex | Target-support intersection, exact occurrence continuity, per-target round trip, and explicitly noncausal longitudinal correspondence | False path created by target union, basis/cluster contraction, or cross-time correspondence |
| Holdout transport | Seen-mass coverage, frozen-prototype coherence, temporal/path recurrence, and eight per-family outcomes | Identity lookup reported as accuracy/generalization or 128 targets treated as independent trials |
| Top-five parity | Observed-token `k=1` structural/numerical parity | Unexplained graph or value mismatch |
| Top-five semantics | Frozen `model_top5_plus_observed` policy, independent `k=1` topology union, fixed-union dense measurement, and explicit masks | Mixed positions/policies, joint-objective topology presented as independent, induced edges, or missing treated as zero |
| Top-five utility | C2 non-degenerate profiles and material change in a predeclared discovery metric | Full corpus launched from schema/resource success alone |
| Top-five resources | Measured runtime/HBM/RSS/storage headroom | Unsupported extrapolation or OOM |
| Model server | Frozen revision/runtime, health and deterministic smoke | Mutable model/runtime or silent template drift |
| Prompted simulator | Parse rate and disjoint discovery-audit correlation/R² under frozen family-grouped splits | Fluent labels with poor numerical simulation or reused generation/scoring evidence |
| Cost | Actual pilot token/GPU ledger and dated prices | Call-count-only estimate |
| Claims | Explicit exploratory/confirmatory/causal boundaries | Faithfulness claim unsupported by controls |

## 15. Failure, resume, and revision policy

- Artifacts are created atomically and never overwritten.
- Resume validates the manifest and payload before skipping a completed unit.
- A corrupt existing unit blocks the task; it is not silently replaced.
- Retry only failed units after recording the root cause and confirming that configuration and
  cohort identity remain unchanged.
- Hardware failure is isolated from parameter effects. Do not interpret a failed GPU/node as a
  scientific condition.
- Any changed model revision, tokenizer, prompt template, candidate policy, candidate-union or
  refinement contract, clustering schema, feature normalization, or holdout rule creates a new
  versioned run family. A changed diagnostic joint objective also creates a new diagnostic family.
- If resource probes invalidate the planned hardware or wall time, revise the execution plan while
  keeping workload and scientific semantics fixed.
- If the workload itself must change, create and document a new manifest rather than editing the
  frozen one.
- Task 15 remains separately governed and does not block atlas, multiplex, or matched top-five
  work over the 2,594 completed positions.

## 16. Immediate implementation tranche

The recommended first code changes, before any substantial processing, are:

1. corpus inventory and atlas sidecar:
   - deterministic compact-artifact discovery;
   - exact 2,594/2,595 accounting;
   - stable logical IDs and target/CI/signed-basis/occurrence indexes;
2. downstream fit/apply split:
   - dense-only input-profile and temporal-trajectory feature builder;
   - frozen reducers, norms, family/response/target weights, and support rules;
   - frozen dense-first cluster state;
   - broad discovery recurrence/transport path;
   - explicit holdout apply path;
3. multiplex core:
   - target-indexed node/edge tables;
   - support-intersection query;
   - noncausal longitudinal correspondence and trajectory tables;
   - false-cross-target and false-cross-time path tests;
4. dense label-free report:
   - emergence, persistence, source-attribution, commitment-relative, and recurrent-path metrics;
   - family-blocked stability and explicit feasibility decision;
5. focused validation and review:
   - run the smallest relevant tests first;
   - review cross-cutting provenance, polarity, statistical-unit, and holdout boundaries;
   - commit a clean checkpoint suitable for immutable dense-atlas/multiplex worktrees.

The following are optional parallel extension tranches and do not block the dense core:

6. top-five trace contract:
   - candidate provenance;
   - observed-token `k=1` compatibility path;
   - deterministic candidate selection;
   - C0 independent-candidate, objective-comparison, and exact-union refinement path;
   - candidate-union schema, applicability, missingness, and cancellation tests;
7. model abstraction:
   - generic OpenAI-compatible generator/scorer/summarizer adapters;
   - fake backend;
   - usage/cost ledger;

Once items 1–5 pass, freeze the shared dense-downstream worktree, its two independent plans, and
begin their joint Milestone 2 preflight. Items 6–7 may proceed in independent worktrees subject to
scientific priority and live resource availability.

### 16.1 Agreed development and parallel-execution order

Use this concrete working order:

1. shared foundation in the main checkout:
   - version this plan;
   - implement deterministic inventory, signed identities, partition enforcement, and hierarchical
     family/response/target weighting;
   - build a two-dense-response vertical slice through target tables, longitudinal
     correspondence, trajectory features, exact per-target reconstruction, and clustering with
     descriptions disabled;
   - pass focused provenance, polarity, missing-support, false-cross-target-path, and
     false-cross-time-path tests;
2. freeze independent executable lanes:
   - one `dense-downstream-width1-v1` executable snapshot with separate
     `dense-atlas-width1-v1` and `response-time-multiplex-v1` plans and output roots;
   - `top5-trace-v1`;
   - `qwen-labeling-v1`;
3. first parallel job wave:
   - one-pass dense feature and response-time multiplex construction by response in an
     `0-10%4` joint array, followed by separate deterministic compactors and multiplex query
     validation;
   - top-five observed-token `k=1` parity and C0 candidate-union work are complete;
   - optional Qwen3.6-35B serving-environment/load smoke;
4. second parallel job wave after first-wave gates:
   - clustering sweep arrays over the immutable dense feature store;
   - broad recurrence application under the frozen dense state;
   - top-five C1 candidate-policy/resource probes;
   - complete 35B prompt/schema/usage validation and consider 122B staging only if justified;
5. continued main-checkout development while immutable jobs run:
   - trajectory and recurrent-path queries;
   - response-time viewer/export work;
   - clustering report and state-selection logic;
   - label-evidence rendering and provider-neutral clients;
   - C1 selection/resource analysis and later C2 launchers.

Stagger dense feature-store and response-time-multiplex compaction if both would create substantial
VAST I/O. Do not launch C2, a full top-five corpus, holdout evaluation, production labeling, or
122B serving merely because an earlier engineering smoke succeeds.

## 17. Decisions to make after the first code tranche

These decisions are intentionally evidence-gated:

1. Candidate policy:
   - `model_top5_plus_observed` is the frozen BonaFide candidate policy;
   - observed token is candidate zero and every distinct model-top-five token is retained;
   - pure model top five is sensitivity-only under a separately named family.
2. Width-one cluster input:
   - input-attribution profile only;
   - input attribution plus temporal trajectory if the trajectory view passes support/stability
     checks;
   - width-one contribution does not enter the primary state unless it unexpectedly passes the
     frozen effective-rank/variance checks.
3. Similarity implementation:
   - existing dense method;
   - sparse k-nearest-neighbor affinity plus sparse spectral/Leiden method after the
     inventory-based memory estimate;
   - blockwise similarity alone is not sufficient if clustering still requires dense
     eigendecomposition.
4. Local model:
   - Qwen3.6-35B-A3B for the first complete serving/prompt path;
   - Qwen3.5-122B-A10B only if the smaller model leaves a measured quality gap;
   - 397B ceiling only if the smaller models leave a measured gap.
5. Simulator:
   - keep the specialized Transluce simulator for attribution;
   - adopt prompted Qwen numeric simulation only if held-out score quality is competitive.
6. External API scope:
   - small frozen calibration/control set;
   - larger use only if it materially outperforms local models under blind evaluation.
7. Parallel scheduling:
   - actual number of concurrent lanes after live GPU, host-memory, queue, and VAST-I/O checks.
8. Dense-plus-broad sensitivity:
   - retain the dense-first cluster state as primary;
   - optionally fit a separately versioned combined state after the broad recurrence report.

None of these decisions requires changing the frozen width-one corpus.

## 18. Canonical project references

Read with this plan:

- `docs/ADAG_BONAFIDE_NAIVE_PILOT.md` for the scientific question and claim boundaries;
- `docs/TRACING_PERFORMANCE_BENCHMARK.md` for the compact-artifact and downstream boundary;
- `docs/TRACING_CORPUS_PLAN.md` for the frozen selection, execution plan, exact denominators, and
  completed tracing record;
- `circuits/tracing/artifact.py` for compact artifact validation and atomic persistence;
- `circuits/tracing/trace.py` for current `CircuitData.merge()` behavior;
- `circuits/analysis/circuit_ops.py` for current clustering orchestration;
- `circuits/descriptions/vllm_backend.py` for the in-process explainer and specialized simulator;
- `circuits/descriptions/api_backend.py` and `circuits/descriptions/label.py` for current Anthropic
  generation, scoring, and summarization roles.

This plan extends those documents downstream. It does not modify their frozen trace selection or
execution record.

## 19. Dense clustering selection execution record

This section is an append-only record of the 2026-07-27 execution. It does not revise the frozen
selection rule in Section 8.5.2.

The initial immutable sweep at code revision `039bf2c` produced 44 valid sparse cluster states.
Label-free structural evaluation retained the hierarchical medoid states at 64, 96, and 128
clusters. Their dense-multiplex projection and family-blocked/checkpoint evaluation ran from the
immutable `7b3558d` worktree:

```text
run root:
  /scratch/general/vast/u1653998/circuits/results/bonafide/clustering/
  cluster-selection-7b3558d-v1

structural report:
  control/structural-report.json
  report_sha256 =
    157a8ee183f67c1e5cd6dbe7c1a5af3d478441f9769186f96f108fe8cf5f3751

projection manifest:
  projection/manifest.json
  manifest_sha256 =
    fdf51d35236372c0522adae0b3e3f0817798de306a95a7230ce91c0d9e46c6ee

resampling report:
  control/resample-report.json
  report_sha256 =
    58789bdf151d68944eb6219f5eab3ba495d6fc417356c37c1c762b4f075ac181
```

The resampling workload comprised ten leave-one-family-out evidence builds, three whole-family
checkpoints, and 39 candidate refits at four-way concurrency. All three candidate resolutions
passed the frozen family-jackknife gate:

| Candidate | Family-jackknife median ARI | Family-jackknife p10 ARI | Checkpoint median ARI |
| ---: | ---: | ---: | ---: |
| 64 | 0.627 | 0.584 | 0.444 |
| 96 | 0.634 | 0.595 | 0.442 |
| 128 | 0.665 | 0.626 | 0.435 |

The checkpoints contain 364, 1,060, and 1,476 dense targets because whole families, rather than
individual targets, are the resampling unit. Agreement rises with corpus size but remains only
moderate at the largest checkpoint: ARI 0.528, 0.522, and 0.538 for 64, 96, and 128 clusters,
respectively. This is not a frozen rejection gate, but it is a material sample-size sensitivity to
retain in downstream interpretation.

The deterministic selector at code revision `d5a771f` assigned the following frozen
within-candidate percentile-rank composites:

| Candidate | Composite | Stability | Graph | Balance/labelability | Recurrence | Temporal/edges |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 0.699 | 0.250 | 0.667 | 1.000 | 0.800 | 1.000 |
| 96 | 0.475 | 0.500 | 0.500 | 0.375 | 0.500 | 0.500 |
| 128 | 0.326 | 0.750 | 0.333 | 0.125 | 0.200 | 0.000 |

The frozen labeling-ready states are therefore:

- primary: the 64-cluster medoid, source task 4;
- alternative: the 96-cluster medoid, source task 6.

The selected-state bundle is:

```text
/scratch/general/vast/u1653998/circuits/results/bonafide/clustering/
cluster-selection-d5a771f-v1

master manifest_sha256 =
  e85db200868f3c611f3577f4c81300e4f9d969ba00ce6f688197312e94977d23
```

Each state persists the exact 16,022-row signed-basis assignment, five-member
within-cluster-affinity prototypes, dense multiplex summaries, target-witnessed recurrent edges,
and verbatim target exemplars. The shared family-grouped labeling split contains four generation,
three selection-scoring, and three audit families. Exemplar selection is response/family-disjoint
within each partition and deterministically prefers phase, condition, and target-token diversity.

Of the primary clusters, 62 of 64 have both the frozen labelability support and exemplars in all
three partitions; the other two are retained but marked `insufficient_partition_support`. Of the
alternative clusters, 88 of 96 are ready, six are marked `insufficient_labeling_support`, and two
are marked `insufficient_partition_support`. No descriptions were generated and no holdout was
opened during fitting, evaluation, selection, or evidence packaging.

The first finalizer submission, job `1664577`, failed before analysis because its launcher was
given the wrong resampling-report path; it wrote no selected-state output. Corrected job `1664623`
completed the bundle above. Post-run validation recomputed the master/state canonical hashes and
every persisted file hash, checked contiguous cluster evidence and exact 16,022-row assignments,
confirmed the shared disjoint family partitions, and passed.

### 19.1 Labeling comparison runtime implementation

On 2026-07-27, commits beginning at `10c9bb5` implemented the provider-neutral comparison runtime
without launching a production labeling run. The implementation adds:

- config-driven OpenAI, Anthropic, OpenAI-compatible, and deterministic fake adapters;
- live request execution and hosted-provider native batch prepare/submit/status/collect stages;
- atomic per-request results and telemetry with a dated, explicit price snapshot;
- the three frozen recipes: Qwen-only, Luna/Terra, and Haiku/Opus 5;
- exact selected-bundle validation and a deterministic bridge from frozen trace witnesses plus
  signed-basis assignments to polarity-aligned cluster attribution profiles;
- character-overlap retokenization diagnostics for the fixed Transluce simulator;
- selection scoring on `selection_scoring`, summary generation from ranked candidates, and final
  label audit on `audit`;
- a GPU scoring launcher that loads the simulator once per recipe/phase rather than once per
  cluster.

The primary and alternative prompts share the same renderer and partition semantics across
providers. Hosted transport batching does not combine multiple clusters into one semantic model
request. API monetary cost and local GPU-hours remain separate accounting quantities.

Minimal live checks confirmed `gpt-5.6-luna` through the Responses API and
`claude-haiku-4-5-20251001` through the Messages API with normalized usage records. The first
deliberately 64-token Haiku response ended at the token cap and was correctly classified as
invalid JSON; a 128-token retry completed and parsed. No Terra, Opus, provider batch, Qwen
endpoint, or production simulator call was made.

The remaining execution dependency is staging `Transluce/llama_8b_simulator` in a frozen circuits
labeling environment. It was not present in the configured Hugging Face cache at implementation
time. Qwen endpoint variables were also not present in this thread's environment; that endpoint is
owned by the separate serving track. The operating procedure and output layout are recorded in
`docs/LABELING_COMPARISON_PIPELINE.md`.

After fixing `--cluster-limit` to mean a total across requested states, code revision `fcb2549`
prepared the non-billable 12-cluster pilot at:

```text
/scratch/general/vast/u1653998/circuits/results/bonafide/labeling/
comparison-pilot-fcb2549-v1
```

The shared selection is primary clusters `0, 12, 24, 38, 50, 62` and alternative clusters
`0, 17, 37, 54, 73, 94`. Each recipe contains 60 candidate requests and 12 persisted cluster
profiles. The 12 unique rendered prompt hashes are identical across all three recipes.

Run manifests:

| Recipe | Run ID | Manifest SHA-256 |
| --- | --- | --- |
| Qwen-only | `labeling-93fb184ea79b2318` | `e6768c018b3811eeecf9a32740c86f7c717fa8f2f72fd47357246893c52b82a6` |
| OpenAI 5.6 | `labeling-180c5fe58531ed36` | `509823a3e6af310dc1e0a7fcb584628cfc9576d0ac6f89567f5105c488a53739` |
| Anthropic upgraded | `labeling-f242fcaab41aca70` | `883384599da9e3fc35b5763d4e84ea13bd8b5a70274ef98f07d32b666227229b` |

The OpenAI and Anthropic native candidate-batch payloads were prepared but not submitted. Their
input SHA-256 values are `771528757ee34f0346deb6df080f817ec1bd28c0f7aaa50c4dfdde1c629f8783`
and `973e2180a2303b724f17e1e893d8b6b0337de64906f0c504770e9f36a34d5180`,
respectively. No billable provider batch or Slurm scoring job was created.

On 2026-07-29, offline simulator preflight required a new provenance cohort. Commit `755e37a`
pins and locally resolves Transluce simulator revision
`63919a3fe41f88d91ef764213ae9018e1f8a578e`; the corrected comparison pilot lives under
`comparison-pilot-755e37a-v1`. Luna and Haiku candidate generation completed with 60 valid
descriptions per recipe after one archived-and-recorded live retry of a malformed Haiku batch
response.

The initial A800 scorer jobs `1676775` and `1676776` and first owner-A100 attempts `14382311` and
`14382312` failed before producing score artifacts because the CUDA context was not initialized
before memory telemetry reset. Recovery jobs `14382647` and `14382648` completed all 12 clusters
and 60 candidate correlations per hosted recipe, using a combined 0.062731 A100 GPU-hours. The
recovery launch explicitly initialized CUDA; commit `579a71b` codifies that ordering for future
runs without changing the active manifests' `755e37a` provenance.

On 2026-07-30, Terra batch `batch_6a6bc0cecb2481908f73edf4e3854077` produced 12 valid summaries.
Opus batch `msgbatch_01Dzn8SKncTvvYUNh5vvwf6P` produced two valid summaries and ten truncated
responses at the frozen 300-token cap. Commit `e2859fd` added a fail-closed retry command; ten
explicit live retries at 1,200 tokens completed successfully with original artifacts and hashes
archived. Terra audit job `14383331` completed 12/12 labels with correlations from -0.093977 to
0.150438. Opus audit job `14383798` completed 12/12 labels with correlations from -0.038592 to
0.220686. Weak or negative scores remain visible and are not regenerated or excluded through
semantic inspection.

OpenAI reduced Luna and Terra prices on 2026-07-30. New recipes bind the additive
`prices-2026-07-30.json` snapshot; active manifests and telemetry retain the frozen 2026-07-27
snapshot. At the new native-Batch rates, measured pilot usage reconciles to `$0.01113894` for Luna
and `$0.025834` for Terra, or `$0.03697294` for the OpenAI path. A maximum-output extrapolation to
all 150 ready clusters is about `$0.68`, so the forward full-run OpenAI guardrail is `$1`.

The identically selected Qwen recipe remains endpoint-gated.

### 19.2 Width-one-aware labeling v2

Human review of the 12-cluster v1 outputs found that the hosted summarizers often named the shared
hinting/security corpus condition instead of the localized attribution pattern. This does not
invalidate the frozen clustering or trace corpus, but it prevents the v1 labels from serving as an
accepted production bundle.

The v2 comparison is therefore a new provenance cohort over the same explicit 12 cluster IDs. It
does not reinterpret width-one evidence as the fuller contribution view used by upstream ADAG:
each source artifact targets one observed response token, contribution evidence remains shallow,
and there is no non-degenerate top-k target comparison. Candidate and summary prompts require
localized attribution evidence, separate universal corpus context, and allow
`insufficient_evidence`. Localized activity on hint/security spans may support a corpus-bounded
association, but shared context alone cannot establish selectivity or generality. Summaries receive
exact generation and selection-scoring witnesses but
never audit witnesses. Profile and candidate-score hashes are recorded in the summary-stage
manifest.

The deterministic v2 quality report has no automatic acceptance state. A non-positive/nonfinite
best selection correlation or model-declared insufficiency yields `insufficient_evidence`; every
other label is `review_required`. Audit correlation is reported independently and cannot revise,
accept, or reject the generated label. The original v1 artifacts and telemetry remain unchanged.

The hosted v2.1 pilot completed on 2026-07-31 under
`comparison-pilot-adbfff4-v2` using the same explicitly frozen six primary and six alternative
cluster IDs as v1. Luna and Haiku generated five candidates per cluster, the fixed Transluce
simulator ranked them on `selection_scoring`, Terra and Opus generated final labels, and the
simulator then scored the exact final labels separately on `selection_scoring` and `audit`.
Candidate and summary artifacts bind source revision `adbfff4`; the additive final-label scoring,
fail-closed quality gate, and retry-aware telemetry correction bind revision `7c65e74`.

The corrected automatic result is 12/12 `insufficient_evidence` for Terra and six
`review_required` plus six `insufficient_evidence` for Opus. Semantic review retains four bounded
hypotheses: primary 38 and 50 as the strongest provisional lexical/phrase-role associations, and
primary 24 plus alternative 37 as background/template descriptors. Primary 0 and alternative 17
are downgraded despite positive in-sample scores because target-local recency or ubiquitous
position/function-word structure is a better explanation and their audit correlations are
negative. This is diagnostic evidence from single-target traces; none is a causal, contribution,
faithfulness, selectivity, or generality claim.

The retry-aware ledger records 196 real hosted API attempts, 803,450 input tokens, 131,598 output
tokens, and `$2.45775876` total API cost (`$0.09524426` OpenAI and `$2.36251450` Anthropic).
Successful simulator work used `0.095270` measured A100 GPU-hours; six successful Slurm
allocations occupied `0.3236` A100-hours including startup. The initial stale-cache failure and
cancelled companion launch added `0.1028` A100-hours and produced no scores. The launcher now pins
the simulator cache. The complete per-cluster dispositions and provenance ledger are recorded in
`docs/LABELING_WIDTH_ONE_V2_PILOT.md`.

Use Opus as the semantic labeler and Terra as a conservative abstention/disagreement baseline for
the next comparison. Do not yet scale either hosted recipe across all 150 labeling-ready clusters:
first run the identically selected Qwen cohort, then use provider agreement and the four retained
hypotheses to decide whether a larger pass is informative. Matched controls or top-k traces remain
necessary before interpreting these width-one associations as contribution structure.

### 19.3 Candidate-evidence matched labeling comparison

The C2 candidate-aware clustering experiment did not promote `C` or `F`: both narrowly missed the
frozen selection lift, while `C` also failed stability/modularity and `F` failed selection
width-preservation and the 80% readiness guardrail. Direction-null and generation-family
jackknife arrays were not launched because those later computations cannot repair the already
failed necessary conditions. The result and bounded interpretation are recorded in
`docs/CANDIDATE_AWARE_CLUSTERING_LABELABILITY_RESULTS.md`.

The remaining eligible labeling experiment holds the W64 clusters fixed and compares width-one
evidence against the identical clusters/witnesses with candidate evidence added. Revision
`eea52ab` froze 12 deterministic anchors and published the provider-neutral evidence bundle under
`candidate-aware-clustering-c2-v1/labeling-comparison-v1`, manifest SHA-256
`227cde5658f1381963b94df192b8e86e1188ca13e28c334003a5a100d3496b55`. It contains all 601
generation witness rows, 530 prompt-ineligible held-out scorer rows, and 24 paired arm handoffs.

Execution order from this checkpoint is:

1. freeze a bounded renderer that selects one identical, W-only-determined generation witness set
   per anchor for both arms and renders candidate numbers to six significant digits;
2. prepare the full token/cost estimate and exact Opus-generation, Opus-rewrite, and Terra-control
   batch parameters for approval;
3. run the fixed Transluce simulator only on `input_localization_hypothesis`, first on selection
   and then on audit without changing the label or selected generic control;
4. freeze blinded review IDs/forms before review and unblind only after the required decisions;
5. decide the evidence-only pilot by the prospectively frozen retained-label, abstention, and loss
   thresholds. Candidate text remains exploratory even if the evidence-only pilot passes.

No model call is authorized directly by the evidence bundle: its manifest deliberately records
`renderer_frozen=false` and requires the renderer/request cohort to be committed first.
