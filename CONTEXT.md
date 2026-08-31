# BonaFide Process-Witness Study

This context defines the scientific language for testing whether ADAG can represent required
model processes and, only after that prerequisite passes, whether recurring graph structure can
serve as evidence about chain-of-thought faithfulness.

## Tasks and textual processes

**Outright process task**:
A BonaFide `complex` or `graph` task whose answer construction requires one or more constrained
intermediate processes.
_Avoid_: Arithmetic task, bottleneck dataset

**Task-required process event**:
One process step required by the construction of a particular outright process task. Its identity
is example-specific even when it belongs to a reusable process family.
_Avoid_: Operation, computation token

**Process family**:
A reusable class of processes, such as transformation, state transition, traversal, comparison,
selection, encoding, extraction, aggregation, or verification.
_Avoid_: Arithmetic operation

**Process signature**:
A normalized description of the specific mechanism instantiated by one or more task-required
process events, such as selecting the minimum-cost outgoing edge.
_Avoid_: Label, process family

## ADAG representation

**Global atlas**:
One frozen mapping from signed neuron identity to cluster and from cluster to a bounded semantic
description or abstention, fitted across a declared corpus composition.
_Avoid_: Neuron ground truth, witness

**Global-atlas adequacy test**:
A prerequisite study of whether one global atlas retains locally useful process distinctions across
the outright-process distribution within measured perturbation and specificity floors.
_Avoid_: Intrinsic polysemanticity test, witness test

**Adequacy panel**:
A graph-blind, balanced group of target contexts with one declared inferential job, such as
measuring repeatability, best-case recovery, natural transport, collision, or specificity.
_Avoid_: Motif dataset, witness panel

**Adequacy trace bank**:
The reusable collection of independent target-local T5 graphs selected to evaluate the adequacy
panels, without being optimized for a later motif hypothesis.
_Avoid_: Witness corpus, dense trajectory corpus

**Semantic-composition stair**:
An ordered adequacy comparison that holds total target-context evidence approximately fixed while
increasing the declared semantic diversity of the atlas-fit corpus.
_Avoid_: Dataset-size ablation, broad-versus-narrow anecdote

**Evidence-support stair**:
An ordered adequacy comparison that holds semantic composition fixed while varying the hierarchical
support profile of prompt cells, responses per prompt, and targets per response or event.
_Avoid_: Token-budget sweep, independent-context count

**Evidence-support profile**:
The ordered tuple of independent prompt cells, responses per prompt, and target contexts per
response or event; profiles with the same total graph count are not scientifically interchangeable.
_Avoid_: Sample size, compute budget

**Adequacy cell**:
One declared semantic composition, evidence-support profile, algorithm condition, and replicate or
resample for which an atlas and its cell-local measurements are produced.
_Avoid_: Dataset, independent experiment

**Admission profile**:
A stage-positioned, cell-level or split-level account of whether enough relevant evidence exists to
attempt and interpret the next analysis. It does not measure pipeline quality, semantic richness,
or whether the resulting atlas has an appropriate resolution.
_Avoid_: Atlas-quality score, dataset-quality score, global pass

**Formation admission**:
The part of an admission profile concerning whether an adequacy cell contains enough independent
breadth, recurrence, and relational support to constrain atlas formation.
_Avoid_: Cluster-quality score, minimum cluster count

**Projection admission**:
The part of an admission profile concerning whether an atlas-fit split provides enough coverage to
make projection onto an independent evaluation split informative rather than predominantly unknown.
_Avoid_: Projection accuracy, held-out cluster quality

**Description admission**:
The post-clustering part of an admission profile concerning whether the frozen clusters collectively
provide enough independent and diverse evidence to attempt meaningful descriptions.
_Avoid_: Description-fidelity quality, label confidence

**Cell-local measurement**:
An admission, failure-incidence, cluster-quality, or description-fidelity measurement defined
within one adequacy cell before comparison with another cell.
_Avoid_: Stair effect, final score

**Adequacy contrast**:
A declared comparison between adequacy cells, interpreted relative to the appropriate
same-composition perturbation or other frozen reference.
_Avoid_: Raw score, independent replication

**Stair-dynamics summary**:
An optional higher-level description of how cell-local measurements or adequacy contrasts change
across semantic composition or evidence support; it is not a separate quality class.
_Avoid_: Overall adequacy score, fifth metric class

**Coarse sampling tag**:
One mutually exclusive, graph-blind region category used only to enrich the first tracing-wave
sample. The allowed tags are active task work, evaluation or revision, intermediate commitment,
final answer, other semantic text, surface or control, and uncertain. A tag is retained as
selection provenance but is not an adequacy label, motif label, or claim about internal work.
_Avoid_: Process family, semantic ground truth, ADAG label

**Trajectory effect**:
The primary change a bounded text unit makes to the visible reasoning trajectory: producing new
task state or evidence, assessing or revising an existing candidate, reporting a settled state,
committing the final answer, adding semantic context without changing state, or adding only
surface/control form. Coarse tags classify this effect rather than mere topical relevance.
_Avoid_: Internal model computation, all task-related text

**Replica vote profile**:
The complete ordered set and histogram of coarse-tag decisions from predeclared identical-protocol
model requests for one unit. A 3-0, 2-1, or 1-1-1 profile measures proposal stability; it is not a
probability, independent-voter confidence interval, or semantic ground truth. Majority labels may
guide sampling only while the full profile remains attached as provenance.
_Avoid_: Gold label, posterior probability

**Coarse selection layer**:
The frozen set of coarse sampling tags, structural identities, and sampling metadata used to build
the first tracing wave. It determines where traces are collected but does not define later
adequacy strata, motif classes, or scientific endpoints.
_Avoid_: Descriptive annotation layer, adequacy ontology, target truth

**Descriptive annotation layer**:
A versioned graph-blind set of finer token and event annotations used to condition adequacy
measurements after target selection. It may grow while tracing runs but cannot retroactively change
the frozen selection reason for a target.
_Avoid_: Selection manifest, neuron interpretation

**Trace wave**:
A separately frozen set of target positions selected under one declared annotation and sampling
policy. Wave one uses only coarse sampling tags; a smaller wave two may repair refined-label
coverage deficits before graph or cluster outcomes are inspected.
_Avoid_: Retry, silent extension, outcome-guided target addition

**ADAG cluster description**:
A bounded natural-language summary or abstention for one fitted cluster, evaluated separately from
the cluster assignment itself.
_Avoid_: Text annotation, neuron ground truth, witness

**Evidence-degradation calibration**:
A labeler test in which controlled losses of cluster-evidence resolution should produce
correspondingly broader descriptions or abstention rather than equally specific process claims.
_Avoid_: Bad-cluster labeling, label quality by confidence

**Unsupported description specificity**:
A cluster description that asserts a process distinction finer than the frozen cluster evidence
and held-out occurrences support.
_Avoid_: Detailed label, confident label

**Cluster-retention quality**:
The extent to which a frozen clustering and its projected graphs preserve the measured distinctions
and relations present in the admitted trace evidence.
_Avoid_: Cluster interpretability, attractive partition

**Description-fidelity quality**:
The extent to which a cluster description faithfully and at appropriate specificity summarizes
the evidence available for its frozen cluster, including calibrated abstention.
_Avoid_: Label confidence, label detail

**Atlas quality**:
The joint adequacy of one frozen cluster mapping, its projected graph representation, and its
bounded cluster descriptions for the declared corpus composition and downstream use.
_Avoid_: Motif quality, causal validity

**Exploratory adequacy characterization**:
An outcome-open analysis that measures and helps interpret atlas-quality dimensions without an
independently validated pass threshold.
_Avoid_: Adequacy verdict, failed confirmation

**Confirmatory adequacy gate**:
A frozen decision rule applied to evidence not used to invent or calibrate that rule, yielding the
declared robust, brittle, or inconclusive verdict.
_Avoid_: Retrospective threshold, exploratory pass

**Context-conditioned role aliasing**:
Loss of distinctions when one global signed-neuron assignment represents occurrences whose
attribution, contribution, or relational role changes across declared context strata.
_Avoid_: Proven neuron polysemanticity

**Pairwise co-occurrence censoring**:
The absence of one-sided contexts from the direct similarity evidence for a neuron pair because the
pair is compared only where both signed identities are observed after tracing and pruning.
_Avoid_: Evidence that an absent neuron was inactive

**Bridge-induced merging**:
A possible global-partition outcome in which context-specific affinities connect otherwise
unsupported or incompatible neuron groups through intermediate identities.
_Avoid_: Guaranteed transitivity

**Mean masking**:
Loss of support, dispersion, minority disagreement, or context-family reversal when a distribution
of pair evidence is reduced to one aggregate affinity.
_Avoid_: Polysemanticity by itself

## Witnesses

**Process motif candidate**:
An exploratory attributed graph pattern that recurs across target-local ADAG graphs associated with
a declared process; recurrence alone does not establish specificity, computation, or causality.
_Avoid_: Witness, temporal path

**Structural candidate witness**:
A frozen motif, matcher, event-level aggregation rule, and threshold that generalizes to unopened
process instances and separates them from matched controls.
_Avoid_: Attractive cluster, repeated label

**Causally supported process witness**:
A structural candidate witness whose targeted intervention produces a selective process-relevant
effect under matched controls.
_Avoid_: Faithfulness detector, proof of internal computation

**Dense graph trajectory**:
An ordered series of independently traced target-local graphs under growing teacher-forced
prefixes. Correspondence across positions is recurrence, not an edge or path through time.
_Avoid_: Mega-graph, temporal causal graph
