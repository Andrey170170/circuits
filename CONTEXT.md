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

**Coarse selection layer**:
A frozen graph-blind annotation projection used to stratify and sample adequacy targets. It records
broad functional regime, process family, surface form, event and response identity, uncertainty,
and provenance without claiming an internal computation.
_Avoid_: Ground truth, final ontology, ADAG label

**Descriptive annotation layer**:
A versioned graph-blind set of finer token and event annotations used to condition adequacy
measurements after target selection. It may grow while tracing runs but cannot retroactively change
the frozen selection reason for a target.
_Avoid_: Selection manifest, neuron interpretation

**ADAG cluster description**:
A bounded natural-language summary or abstention for one fitted cluster, evaluated separately from
the cluster assignment itself.
_Avoid_: Text annotation, neuron ground truth, witness

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
