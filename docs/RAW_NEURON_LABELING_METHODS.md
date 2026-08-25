# Initial methods for labeling raw Qwen MLP neurons

Date: 2026-08-23

## Decision summary

The first run labels a raw-neuron occurrence's apparent role inside one selected-logit graph. It
does not assign a global neuron meaning. Use deterministic evidence inventories and structured,
evidence-cited graph-role hypotheses first. Delphi remains a later activation-selectivity arm, not
the definition of a correct label and not a drop-in circuit labeler. The later comparison is:

1. deterministic evidence cards and human labels;
2. Delphi-style activation-selectivity labels;
3. ADAG input-attribution and output-contribution labels; and
4. a hybrid presentation that keeps the activation and graph-role hypotheses separate.

The stable object for the first graph-role run is the exact occurrence keyed by trace, token
position, layer, neuron index, and polarity, with its selected target and source trace hashes. A
signed raw-neuron basis `(model revision, layer, neuron index, activation polarity)` becomes the
stable object only for later cross-context activation-selectivity work. Evidence from one
occurrence must never be promoted to a model-wide neuron label.

## Why Delphi is useful but not directly compatible

[EleutherAI Delphi](https://github.com/EleutherAI/delphi) generates and scores natural-language
explanations from feature-activation records. Its maintained entry point is designed for sparse
autoencoder and transcoder features, and its README recommends caching activations over at least
10 million tokens. It supports contrastive explanations with semantically similar non-activating
examples and documents detection, fuzzing, and simulation scorers. The repository also warns that
the paper's reproducible implementation is on the `article_version` branch while `main` continues
to change.

The associated paper, [Automatically Interpreting Millions of Features in Large Language
Models](https://arxiv.org/abs/2410.13928), collects feature activations over a broad corpus,
generates explanations from sampled contexts, and evaluates explanations separately. It also finds
SAE features substantially more interpretable than raw neurons. Consequently, Delphi's protocol is
a strong baseline for this project, while its favorable feature-level results must not be assumed
to transfer to Qwen's raw MLP coordinates.

For this repository, the minimal adapter should stream post-activation inputs to each Qwen
`down_proj` and treat the two activation signs as separate non-negative half-features:
`a+ = max(a, 0)` and `a- = max(-a, 0)`. This matches the observatory's existing signed-basis
identity. Only selected bases need be retained; top-k, quantile, random-reservoir, and hard-negative
contexts can be stored without materializing a dense all-neuron cache. Delphi's record construction,
explainers, and scorers can then be reused or reproduced behind a pinned dependency.

## Candidate methods

### A. Deterministic evidence card and blinded human baseline

For every selected signed basis, show:

- top positive-side activation contexts from a broader corpus, plus quantile and random examples;
- semantically or lexically matched low-activation contexts;
- direct output-vocabulary tendencies from the signed `down_proj` column projected toward the
  unembedding, clearly marked as a local linear diagnostic;
- occurrences in the selected graphs, their source-token attribution maps, output-contribution
  maps, and immediate graph neighbors; and
- support counts, corpus provenance, thresholds, and missing-evidence warnings.

Ask a reviewer to write separate `input selectivity`, `output tendency`, and `trace-local role`
hypotheses, with an abstain/polysemantic option. This is the lowest-assumption baseline and the
best debugging surface for all automated methods.

### B. Delphi-style activation-selectivity explanation

Generate a contrastive explanation from activation examples and matched non-examples for each
signed basis. Score it only on held-out records using at least detection and token-level simulation
or ranking. Keep generation examples, model selection, threshold selection, and final audit
partitions separate. Resample the generation exemplars to measure whether the explanation is
stable.

This arm answers "in which textual contexts is this signed neuron active?" It does not explain why
the neuron is present in a traced path or what downstream behavior it causes. This limitation is
also explicit in OpenAI's original [automated neuron explanation
work](https://openai.com/index/language-models-can-explain-neurons-in-language-models/), which
describes input correlations rather than downstream mechanisms and notes polysemantic and
out-of-distribution failure modes.

### C. ADAG profile explanation

Use the repository's existing profile representation to generate two descriptions independently:

- input attribution: which source-token regions are associated with the basis or cluster in the
  controlled trace corpus; and
- output contribution: which traced output candidates it promotes or suppresses.

The existing ADAG description stack already builds activation records from attribution and
contribution maps. Its bundled Transluce explainer/simulator is Llama-tokenizer-oriented, while the
newer candidate-labeling path includes deterministic character-overlap alignment from frozen Qwen
profiles to the Transluce scorer. That code is useful, but the seven observatory graphs alone do not
provide an independent controlled corpus for a stable profile label.

This arm answers a task-conditional graph question rather than the generic activation-selectivity
question answered by Delphi.

### D. Hybrid two-view explanation

Give an explainer both the held-out-compatible activation evidence from B and the profile evidence
from C, but require it to emit separate fields:

- activation-selectivity hypothesis;
- task-conditional input-attribution hypothesis;
- task-conditional output-effect hypothesis;
- scope, exceptions, and possible polysemantic alternatives.

Do not initially collapse these into a one-to-three-word label. A fluent synthesis can hide a
disagreement between the two evidence views. Compare the hybrid against B and C rather than assuming
that more context improves validity.

### E. Cluster or circuit-role labels, later

Once related traces and matched controls support recurring clusters, label the aggregate profiles
and preserve member identities and paths. A role such as "reads the task constraint and supports
the arithmetic answer" is relational; it cannot be validated solely by the activation contexts of
one neuron. It should therefore remain outside the first raw-basis pilot.

## Initial pilot

The seven current graphs contain 525 raw-neuron occurrences and 264 unique signed bases. Eighty-three
bases occur in at least two graphs and four occur in all seven. Because all seven targets are nested
positions in the same faithful completion, these are correlated witnesses rather than independent
contexts.

Start with the frozen 26 occurrences in the position-120 graph: 15 direct target parents, six
salient or repeated upstream occurrences, and five lower-salience retained occurrences. Build
bounded graph-local evidence, deterministic summaries, and structured role hypotheses with
mandatory evidence citations. Inspect abstentions, unsupported stories, and omissions before
extending the same run system to the other targets. Cross-context signed-basis sampling and an
independent activation corpus for Delphi come later.

For each basis, compare:

1. human evidence-card label;
2. Delphi activation-only label;
3. ADAG profile-only label; and
4. hybrid two-view label.

Primary outcomes should be held-out detection/ranking, activation-simulation correlation, hard-
negative specificity, resample stability, and blinded human usefulness. Report coverage and
abstentions alongside scores. Do not score an explanation on the examples used to generate it.

Only after observational performance and stability are acceptable should a few hypotheses receive
position-restricted interventions with layer-, polarity-, magnitude-, and support-matched controls.
Simulation or detection scores are not causal validation: a rigorous audit of natural-language
neuron explanations found high observational error and little causal efficacy even among highly
scored explanations ([Huang et al., 2023](https://arxiv.org/abs/2309.10312)).

## Immediate implementation boundary

The first implementation defines a versioned graph-local evidence packet, immutable run and method
identities, provider-neutral prompt requests and result ingestion, and a complete observatory
overlay export. It does not call a paid provider or install overlays automatically. A selected-basis
activation collector and thin Delphi adapter are later work. Do not launch a whole-model Delphi
run, modify the frozen seven-target manifest, or introduce SAE features in this first step.
