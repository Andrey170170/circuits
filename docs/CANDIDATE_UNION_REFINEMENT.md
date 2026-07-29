# Candidate-union refinement

Status: locked production approach for C1, C2, and any later explicitly
authorized matched candidate corpus.

This experiment keeps the C0 candidate traces independent while making their
node and edge profiles comparable.

## Decision

C0 rejected a scalar joint-logit graph as the primary candidate-comparison
artifact. Raw-sum and contrastive joint objectives remain diagnostic families,
but neither recovered the independent candidate topology well enough to replace
it.

Moving forward, one response target uses:

1. the frozen `model_top5_plus_observed` candidate policy;
2. one independent specified-token `k=1` trace for each realized candidate;
3. the exact union of those independently selected nodes and edges;
4. one candidate-specific fixed-union rescore for nodes and edges;
5. one assembled dense candidate-union artifact retaining applicability and
   original-selection masks.

Changing the candidate policy, topology construction, applicability semantics,
or fixed-union measurement rule requires a new versioned trace family and a new
decision record. It must not silently alter this family.

## Two-pass contract

Pass one is the existing set of five or six independent, specified-token ADAG
traces for one teacher-forced response position. Their exact graph union defines
the candidate set for measurement:

- MLP nodes are the union of nodes retained by any independent trace.
- Edges are the union of edges retained by any independent trace.
- No induced edge is added merely because both endpoints are in the node union.
- Each node and edge keeps a `selected_by_candidate` mask recording pass-one
  membership.

Pass two reruns every candidate as an independent one-logit objective. It
bypasses node and edge pruning but only materializes the frozen pass-one union.
Zero attribution or weight is preserved instead of being dropped.

Internal and embedding-to-MLP edges are applicable to every candidate.
Candidate-to-logit edges are applicable only to the candidate whose logit they
target. Inapplicable entries are stored as null, not zero.

## Result

The assembled artifact contains dense candidate vectors for each applicable
union element:

- nodes: attribution, contribution, and activation;
- edges: attribution and weight;
- both: applicability and pass-one selection masks.

This supports downstream contribution comparisons without changing the
scientific identity of the independent traces. It does not make the union a
complete computation graph: structure absent from every pass-one trace remains
unmeasured.

## C0 execution

The frozen plan is
`scripts/bonafide/manifests/qwen3_4b_instruct_candidate_union_c0_v1.json`.
It binds all 55 independent C0 references by artifact ID, payload hash, source
target, response position, and candidate token order. The two approved waves
are `candidate-union-c0-dense` and `candidate-union-c0-broad`.
