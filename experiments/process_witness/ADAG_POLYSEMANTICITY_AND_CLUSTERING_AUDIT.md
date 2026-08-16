# ADAG polysemanticity and clustering-semantics audit

Date: 2026-08-16
Mode: read-only primary-source audit; no experiment result
Repository revision inspected: `362680a2b3d25351c32b35bd5617ac5ffef61b40`
Upstream ADAG release used for comparison: `2d215d4ba016ba8602b69d71fc0f1ad139a427b7`

## Bottom line

The concern is real, but **polysemanticity is not the same thing as sparsity**, and the cited raw-MLP paper does not establish that a neuron performs at most one function within a trace or across traces. Its main quantitative claim is that a small task-level circuit in the pre-down-projection MLP-activation basis can preserve behavior about as well as an SAE-basis circuit at comparable circuit size. That is a claim about the sparsity and causal adequacy of a selected circuit, not a monosemanticity guarantee.

For paper-style ADAG clustering, the specific risks are:

1. pair similarity is **conditioned on both signed neuron identities surviving tracing in the same context**;
2. heterogeneous similarities inside those co-occurring contexts are reduced to one mean;
3. spectral clustering uses the whole affinity graph, so indirect `A--B--C` connectivity can group neurons lacking direct `A--C` evidence;
4. the final label summarizes aggregate cluster profiles, so a cluster containing context-dependent roles can receive a broad label or a label dominated by the best-supported role.

These are properties worth testing in a mixed-context atlas. They are not, by themselves, proof that the atlas fails.

## Corrections to the working language

### 1. “Similar activations” should be “similar attribution profiles”

ADAG does not cluster raw activation magnitudes. For every context it compares two views by cosine similarity:

- the input-attribution vector over prefix tokens; and
- the output-contribution vector over the selected output logits.

The paper defines these profiles and their pairwise cosine similarities in `papers/2604.07615_src/colm2026_conference.tex:248-253,266-280`. The implementation computes the per-context cosine matrices in `circuits/analysis/multiview_cluster.py:123-163`.

The default pipeline collapses token-position occurrences before clustering. Its stable identity is a **signed basis identity** `(layer, neuron, polarity)`, while individual `(layer, token, neuron, polarity)` occurrences receive the resulting assignment afterward (`circuits/analysis/cluster.py:46-53,74-109`; `circuits/analysis/circuit_ops.py:133-169,302-367`). Positive and negative occurrences of the same MLP coordinate are therefore separate clustering identities. This does not solve multiple same-sign functions.

Here a clustering “context” is one uniquely indexed circuit example/target stored in the `label` column (`circuits/analysis/process_circuits.py:297-350`), not an informal semantic category.

### 2. “Sparse” does not mean “one function”

The cited paper is *Language Model Circuits Are Sparse in the Neuron Basis* ([arXiv 2601.22594](https://arxiv.org/html/2601.22594)). It:

- defines a circuit as a sparse subgraph for a **specific task or dataset**, with input-dependent nodes and edges (arXiv HTML lines 165-178);
- selects a task-level circuit on 300 SVA training pairs and evaluates faithfulness/completeness on 40 held-out pairs (lines 180-203);
- reports near-perfect SVA faithfulness/completeness with about 200 RelP-selected MLP-activation neurons (lines 220-244); and
- states the principal comparison against **SAE circuits**, not a quantitative monosemanticity comparison against transcoders (lines 101-121, 205-245).

The paper qualitatively reproduces a cross-layer-transcoder multi-hop case study with neuron circuits (lines 289-324), but that is different from showing that each raw neuron has only one function. Its sparsity metric asks how few nodes preserve/mediate the task behavior. A neuron can participate in two rare, unrelated functions and still be compatible with sparse activations or sparse task circuits. The paper does not report a cross-behavior semantic-purity or polysemanticity evaluation.

Therefore this proposed inference is unsupported:

> “Pre-down-projection neurons are sparse, therefore a selected neuron is active for at most one function within a trace.”

A defensible replacement is:

> “On the evaluated tasks, RelP can select small, causally effective circuits in the pre-down-projection MLP-activation basis. This does not establish semantic uniqueness of their neurons within or across contexts.”

### 3. ADAG’s stated scope is task/distribution-conditional, but not formally single-behavior-only

The ADAG paper says supernodes should play similar roles over a “particular dataset of examples” and the “distribution of interest” (`papers/2604.07615_src/colm2026_conference.tex:255-264`). Mathematically, its context set `C` may contain any contexts; the paper does not impose a one-behavior-only rule.

However, the validations are narrow, structured context distributions:

- state-capital questions, fit on the entire capitals dataset (`:355-405`);
- semantically related variants of one harmful-advice prompt (`:409-425`); and
- 10,000 two-digit addition examples (`:609-615`).

Thus “ADAG is defined to work only on one fine-grained behavior” is too strong. The accurate claim is:

> ADAG learns task-conditional supernodes over the supplied context distribution. Its published evidence does not establish that one mapping preserves fine-grained roles across a heterogeneous mixture such as our outright-process corpus.

## Audit of the proposed failure cases

### Failure case 1.1: one-sided contexts are absent from a pair’s similarity

**Confirmed, with a nuance.**

For features `i,j`, the paper defines `C_ij` as only the contexts where both co-occur after pruning and averages their similarity over `C_ij`; no co-occurrence gives affinity zero (`papers/2604.07615_src/colm2026_conference.tex:270-280`). The implementation likewise updates a pair’s numerator and denominator only within labels containing both signed identities, then divides by overlap count (`circuits/analysis/multiview_cluster.py:123-175`).

Consequently, if `A` and `B` look similar wherever both survive, contexts containing `A` but not `B` do not count as disconfirming evidence for the `A-B` affinity. They may still affect:

- `A`’s affinity to other neurons;
- the global spectral partition; and
- the later description of a cluster containing `A`.

So ADAG is not blind to those contexts altogether; **the direct `A-B` comparison is blind to them**.

The optional implementation flag `use_attributions_as_view=True` adds a full across-context importance profile with zero for absence and can expose different occurrence patterns (`circuits/analysis/multiview_cluster.py:62-66,211-249`). It is off by default and is not part of the paper equation.

### Failure case 1.2: `A-B` and `B-C` evidence can bridge `A` and `C`

**Possible and correctly identified, but not guaranteed in every fit.**

If `A,B` co-occur only in one context set and `B,C` only in another, ADAG can assign positive `A-B` and `B-C` affinities while assigning `A-C` zero because they never co-occur. Spectral clustering then embeds the entire affinity graph using the normalized affinity/Laplacian and applies K-means (`papers/2604.07615_src/colm2026_conference.tex:282-286`; `circuits/analysis/multiview_cluster.py:259-315`). It is a global graph partition, not a rule requiring every same-cluster pair to have direct evidence.

Therefore `B` can act as an indirect bridge and all three may land in one coarse cluster. Whether they actually do depends on all other affinities, the requested `k`, normalization, eigenvectors, and K-means. This should be called **unsupported transitive grouping** or **bridge-induced merging**, not assumed to occur deterministically.

The upstream fit assigns every retained non-boundary signed identity to one of exactly `k` clusters. Pair overlap counts are stored as diagnostics but are not used as minimum-support or abstention gates (`circuits/analysis/circuit_ops.py:337-379`). Thus unsupported or weakly supported identities are not automatically left unknown during fitting.

### Failure case 2: averaging can hide a minority disagreement regime

**Confirmed.**

In the paper, each co-occurring context yields a non-negative, harmonic-fused similarity, and those scalars are uniformly averaged. A small set of near-zero contexts only lowers a large set of positive contexts in proportion to its frequency (`papers/2604.07615_src/colm2026_conference.tex:270-280`). Thus a pair can retain high aggregate affinity while behaving very differently in a rare sub-regime.

There is also a paper/code fidelity caveat:

- the paper takes the harmonic mean of attribution and contribution similarity **inside each context**, then averages across contexts;
- current `combine="harmonic"` code first averages each view across co-occurring contexts, then takes one harmonic mean (`circuits/analysis/multiview_cluster.py:165-201`);
- current `cluster_multiview` defaults to arithmetic `combine="mean"`, although the README example explicitly requests harmonic (`circuits/analysis/circuit_ops.py:302-332`; `README.md:154-170`).

Both implementations compress a context distribution to one pair affinity and can hide minority regimes, but they need not produce the same matrix.

### Label collapse is a second, distinct failure surface

The paper averages member profiles into one supernode representation in each context, then selects natural-language descriptions by global correlation over all contexts (`papers/2604.07615_src/colm2026_conference.tex:288-312`). Current code sums profiles over members per context before explanation (`circuits/analysis/circuit_ops.py:630-670`) and scores descriptions on the prepared records without an explicit held-out context split (`circuits/descriptions/label.py:194-239,399-639`).

Therefore two different failures must be separated:

1. **clustering failure:** distinct context-conditional roles were merged into one supernode;
2. **description failure:** the cluster may contain coherent structure, but the generated label is overly broad, overly narrow, or dominated by a frequent/easy-to-name regime.

Calling the outcome “multiple different supernodes inside one label” is slightly imprecise. The clustering failure creates **one supernode containing functionally heterogeneous signed neurons**; the labeling stage may then conceal that heterogeneity.

## Does a held-out set make sense?

**Yes, but it must be described as partial identity projection rather than ordinary inductive clustering.**

The published ADAG demonstrations are transductive: the full task dataset contributes to clustering and descriptions, and individual graph slices are then displayed from that same corpus (`papers/2604.07615_src/colm2026_conference.tex:403-405,423-425,609-615`). The paper does not report held-out assignment or label transfer.

After fitting, however, the result is a map from token-collapsed signed neuron identities to cluster IDs. It can be frozen and applied to another graph from the **same model** wherever those signed identities recur. The current code expands the map back only over raw occurrences in the fitted `Circuit` object (`circuits/analysis/circuit_ops.py:133-169,361-378`) and serializes those assignments (`:1193-1234`); it does not implement a learned out-of-sample rule for neuron identities never seen during fitting.

A held-out projection therefore has three outcomes:

- recurring signed identity: receives the frozen cluster;
- unseen signed identity: remains unknown/unclustered;
- different model or neuron basis: not projectable.

Coverage and attributed mass assigned versus unknown must be reported. This is still a meaningful evaluation of whether a frozen atlas transports to new contexts. Indeed, the raw-neuron paper itself uses training examples to select task circuits and held-out examples to evaluate them ([arXiv 2601.22594](https://arxiv.org/html/2601.22594), lines 180-203), although that is a circuit-selection evaluation rather than ADAG cluster projection.

The present BonaFide multiplex projector is stricter than this conceptual rule: it requires the projection and fit feature stores to have exactly the same signed-basis identity set and rejects a trajectory basis absent from the fit index (`circuits/analysis/bonafide/clustering_projection.py:316-328,408-436`). It is therefore **not yet a general prompt-held-out projector**. A future held-out adequacy test needs an explicit partial-projection contract and coverage accounting rather than assuming the existing helper already provides it.

So “held-out makes no sense by design” is too strong. The accurate limitation is:

> Upstream ADAG does not provide a general inductive assignment for unseen neuron identities, and its published labels are scored in-sample. Held-out graph projection is partial, identity-based, and must fail visibly on unknown identities.

## What an adequacy test must measure

The adequacy question is narrower than generic polysemanticity:

> Does the pairwise co-occurrence aggregation plus global partition and label aggregation preserve the context-conditional distinctions required for later process-motif analysis?

At minimum, the audit should expose the failure mechanism directly, before any motif search:

1. **Pair support:** `|C_ij|` and the fraction of same-cluster pairs with zero or very small direct co-occurrence support.
2. **Conditional heterogeneity:** the distribution of per-context pair similarities, not only their mean; report lower tail, disagreement frequency, variance, and regime-conditioned means.
3. **One-sided occurrence:** how often `A` occurs without `B` for high-affinity and same-cluster pairs. This is deliberately absent from paper-style pair affinity.
4. **Bridge dependence:** same-cluster pairs with no direct affinity but short/high-weight paths through other neurons; compare direct-evidence-only and ordinary spectral partitions.
5. **Mixture sensitivity:** fit context-pure and progressively mixed atlases at matched support/budget, then align clusters permutation-invariantly and measure splits, merges, and role purity.
6. **Label resolution:** separately score whether labels distinguish annotated context-conditional roles; do not treat clustering stability as label validity.
7. **Frozen-map transfer:** on held-out contexts, report known-identity coverage, unknown attributed mass, and whether the distinctions seen in fit contexts remain observable.

The cleanest diagnostic outcomes are:

- **adequate:** context heterogeneity is low or explicitly represented; bridges do not create materially heterogeneous clusters; labels retain the required distinctions; frozen-map coverage is sufficient;
- **conditionally adequate:** pure/narrow atlases work but mixed atlases collapse distinctions, implying purpose-built or conditioned atlases;
- **inadequate:** even favorable, well-supported regimes cannot form stable and labelable distinctions;
- **inconclusive:** tracing coverage, co-occurrence support, or annotation reliability is insufficient to localize the failure.

## Claim boundary

This audit establishes that the proposed failure modes are structurally possible under ADAG’s published aggregation and spectral clustering. It does **not** establish that they occur at a consequential rate in the BonaFide-derived corpus. That requires the planned context-conditional adequacy experiment.
