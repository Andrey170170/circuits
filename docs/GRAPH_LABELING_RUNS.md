# Graph-local occurrence labeling runs

This package creates immutable labeling runs for raw-MLP-neuron occurrences in one selected-logit
observatory graph. Its primary object is an occurrence in one trace, not a global neuron meaning.
Every label remains an exploratory role hypothesis under the graph claim boundary; it is not a
complete computation transcript, causal evidence, or a faithfulness verdict.

The first frozen study is
`scripts/bonafide/configs/graph_labeling/qwen-position-120-occurrence-role-v1.json`. It binds the
position-120 Qwen trace and 26 explicit occurrence IDs: 15 direct target parents, six salient or
repeated upstream occurrences, and five lower-salience retained occurrences. No transformed
controls are included in v1.

## Stages

Prepare validates the complete observatory bundle and exact source hashes, then writes one
content-addressed evidence packet per occurrence. It also materializes provider-neutral prompt
requests for the structured LLM method and records the scoped Git/source-tree revision. It never
calls a provider.

```bash
python -m circuits.graph_labeling prepare \
  --spec scripts/bonafide/configs/graph_labeling/qwen-position-120-occurrence-role-v1.json \
  --run-root /scratch/general/vast/$USER/circuits/labels/qwen-position-120-role-v1
```

Scientific identity covers the source hashes, explicit occurrences, evidence policy, prompt,
model, and generation parameters. Execution identity is separate and covers only operational
choices. Credentials, endpoints, headers, and secret-like provider parameters are rejected from
scientific specs and artifacts. A minimal execution file is:

```json
{"schema_version":"adag.graph-labeling.execution.v1","mode":"local"}
```

The provider-free baseline can then run locally:

```bash
python -m circuits.graph_labeling execute \
  --run-root /scratch/general/vast/$USER/circuits/labels/qwen-position-120-role-v1 \
  --method-id deterministic-evidence-summary-v1 \
  --execution execution-local.json
```

`structured-llm-graph-role-v1` is materialized only. Passing `mode: local` fails before making any
provider call. Provider transport is external in v1. An external or manual executor returns one
JSONL row per frozen request with the exact request, logical-request, evidence, and method hashes;
the raw payload hash is also mandatory. Complete results are ingested through the same validation
seam:

```bash
python -m circuits.graph_labeling ingest-results \
  --run-root /scratch/general/vast/$USER/circuits/labels/qwen-position-120-role-v1 \
  --method-id structured-llm-graph-role-v1 \
  --results-jsonl collected-results.jsonl
```

Duplicate, missing, extra, drifted, or invented-evidence results fail closed. Ingestion writes one
immutable label-result manifest and labels file atomically. Execution failures remain execution
telemetry and are not converted into scientific labels.

Export expands the sparse 26-result file into a complete observatory label-set v1. Every other
node in every trace receives an explicit `not_selected` record. Export writes a standalone JSON
overlay; it does not mutate the observatory site.

```bash
python -m circuits.graph_labeling status \
  --run-root /scratch/general/vast/$USER/circuits/labels/qwen-position-120-role-v1

python -m circuits.graph_labeling export-overlay \
  --run-root /scratch/general/vast/$USER/circuits/labels/qwen-position-120-role-v1 \
  --label-set-id occ-role-<identity-from-status> \
  --site-root /scratch/general/vast/$USER/circuits/results/bonafide/raw-graph-observatory-viewer-v1 \
  --destination /scratch/general/vast/$USER/circuits/labels/deterministic-evidence-summary-v1.json
```

The label-set ID is derived from the study semantic hash and method semantic hash; method and study
names are aliases and do not affect it. The source viewer manifest, catalog, and selected trace
must still have the exact hashes frozen at preparation.

Install the exported overlay into a new derived observatory site:

```bash
python -m circuits.graph_labeling install-overlay \
  --source-site /scratch/general/vast/$USER/circuits/results/bonafide/raw-graph-observatory-viewer-v1 \
  --label-set /scratch/general/vast/$USER/circuits/labels/occ-role-<identity>.json \
  --destination-site /scratch/general/vast/$USER/circuits/results/bonafide/raw-graph-observatory-labeled-v1
```

Installation never mutates the source site. It validates the complete label-set binding, builds
and validates a new viewer bundle atomically, and is idempotent only for the identical source and
label-set derivation. A reused label-set ID with different content fails closed.

## Evidence semantics

Each packet retains only the exact observed tokens through the target and derived causal prefix,
never the full response field or post-target text. It also records top positive and negative source-token
attributions, strongest incoming and outgoing graph edges, and every direct edge to the target
logit. Stable facts cover subject, node, target, target contribution, and coverage. A bounded
directed search records target-connected paths using a declared numerical display ranking; it does
not infer a semantic bottleneck. Counts, search bounds, retained absolute mass, and truncation
flags remain explicit.

The position-120 trace has 296 observed tokens because it includes the observed target `4`, while
its causal attribution profiles contain 295 entries and end at prediction position 294. The packet
records both lengths and explicitly marks the observed target as excluded from the causal profile;
it never shifts or pads the profile.

`reads_from` is human-readable. Structured labels separately carry `cited_evidence_ids` and an
explicit claim-to-citations mapping for label, reads-from, apparent role, target effect, rationale,
and every nonempty alternative hypothesis. Reads-from citations are limited to source-attribution
or incoming-edge facts from that occurrence's packet. Provisional labels without these citations
and labels citing another packet fail validation. Each imported structured label also binds its
`result_sha256` exactly to the matching request receipt's raw-response hash. The scientific
statuses are `provisional_label` and `insufficient_evidence`; exported non-subject nodes use
`not_selected`.

Selection, evidence construction, and labeling method are versioned choices behind the run
module. V1 deliberately supports only canonicalized explicit occurrence groups and `none_v1`
controls. New selectors or transformed controls require new named implementations and identities;
they must not silently alter this study.
