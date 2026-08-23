# Raw-graph observatory viewer v1

Status: **proposed implementation plan**. This plan defines a persistent, single-user viewer for
the seven selected Qwen traces. It does not interpret a graph, assign neuron semantics, run a
labeler, or alter the frozen trace artifacts.

## Decision

Build a small CHPC-local **Trace Observatory** that closely follows Neuronpedia's graph-browsing
experience, but do not deploy or fork the full Neuronpedia application.

Use the MIT-licensed Anthropic attribution-graph frontend already vendored under
`circuits/frontend/assets` as the initial rendering substrate. Adapt its layout and interaction
code toward Neuronpedia's current four-pane experience:

1. token-by-layer overview;
2. incoming and outgoing connections;
3. focused, zoomable subgraph;
4. selected-neuron evidence details.

Add a trace catalog, exact target context, provenance, signed raw measurements, and replaceable
label overlays around those panes. Use our own name and visual identity. If individual current
Neuronpedia components are copied rather than independently recreated, pin their source commit and
retain the required Apache-2.0 and third-party notices.

The primary-source reuse assessment is recorded in
`docs/NEURONPEDIA_VIEWER_REUSE_RESEARCH.md`.

## Designs considered

| Design | Public interface | Strength | Main cost | Decision |
| --- | --- | --- | --- | --- |
| Static immutable bundle | `build_viewer(...)`, `serve_viewer(...)` | Smallest surface and safest serving process | Durable notes and live label comparison arrive awkwardly | Use its two-command public shape |
| Workspace-document engine | `inspect(spec)`, `execute(command)`, `export(request)` | Most expressive multi-pane and revision model | Too much abstraction before the first real viewer exists | Borrow its declarative workspace and evidence-layer concepts later |
| Explicit ports and adapters | catalog, trace, label, review, and transport ports | Clean isolation of storage, labels, and mutable review state | Too many interfaces if all are public | Keep only proven variation points as internal seams |
| Full Neuronpedia fork | Neuronpedia's existing web application | Closest visual parity immediately | Inherits database, auth, storage, APIs, deployment, and model-specific semantics | Reject for v1 |

The recommended hybrid keeps the common caller extremely small—`sync` and `serve`—while hiding
compact-trace, label-set, renderer, and workspace adapters inside one deep module. A new interface
is introduced only where at least two behaviors are already required: raw versus later trace
formats, multiple label approaches, writable versus read-only workspace stores, and authoritative
documents versus the legacy renderer compatibility format.

## Why not deploy all of Neuronpedia

Neuronpedia is open source, but its graph page is part of a full Next.js/React application coupled
to Postgres/Prisma, authentication, cloud storage, and Neuronpedia feature APIs. Those systems do
not help display seven already-generated local traces. The existing repository viewer already has
the relevant D3/dagre graph interactions and runs behind a small Python standard-library server.

The existing renderer can be reused, but its current export contract cannot be the observatory's
scientific record:

- node `influence` is the absolute value of attribution, which loses attribution sign;
- links expose edge weight but omit signed edge attribution;
- label lookup can collapse `(layer, neuron, polarity)` to `(layer, neuron)`;
- raw-neuron nodes are described with inherited transcoder terminology;
- saved view state is written back into generated graph JSON;
- the server binds all network interfaces and implicitly calls an external exemplar service.

V1 therefore inserts a lossless, renderer-neutral graph document between compact traces and the
borrowed visual code.

## Scientific and data invariants

- One `adag.compact-trace.v1` artifact is one independent target-local causal graph.
- No endpoint, cache, renderer, comparison, or path search may splice edges from different target
  positions.
- Every node preserves an exact occurrence identity and a stable signed-basis identity containing
  model revision, layer, neuron, and polarity.
- Activation polarity and attribution sign are different quantities and remain different fields.
- Node attribution, activation, input-attribution profile, and output-contribution profile remain
  separate.
- Edge attribution and edge weight remain separate and independently inspectable.
- Display filtering is a reversible projection. It records original and displayed counts and a
  retained-attribution measure; it never becomes scientific pruning silently.
- Labels, cluster assignments, human notes, pins, and layout are overlays. They never modify a
  compact trace or its projection of raw evidence.
- Missing label coverage is `unknown`; labeler abstention is explicit; neither is numerical zero.
- Qwen model and revision identity spaces cannot mix.
- A recurrence across independently traced positions is correspondence, not a causal or temporal
  edge.

The viewer must always display the warning that a graph is a pruned local attribution
approximation for one selected logit, not a complete computation transcript.

## Deep module and public interface

Create one deep `circuits.observatory` module. Its normal human interface is two commands:

```bash
uv run python -m circuits.observatory sync \
  --trace-root /scratch/general/vast/u1653998/circuits/results/bonafide/raw-graph-observatory-qwen-selected-v1 \
  --site-root <rebuildable-site-root> \
  --state-root <persistent-state-root>

uv run python -m circuits.observatory serve \
  --site-root <rebuildable-site-root> \
  --state-root <persistent-state-root> \
  --host 127.0.0.1 \
  --port 8032
```

`sync` validates trusted compact artifacts, loads their pickle payloads, creates safe JSON
projections, builds indexes, hashes the complete bundle, and publishes it atomically. `serve`
validates and serves only safe JSON, static assets, and the separate viewer-state store. It must
not import models, require a GPU, unpickle data on an HTTP request, or accept arbitrary paths from
the browser.

The initial browser transport is intentionally small:

```text
GET /api/v1/catalog
GET /api/v1/traces/{artifact_id}
GET /api/v1/traces/{artifact_id}/nodes/{occurrence_id}
GET /api/v1/label-sets
GET /api/v1/label-sets/{label_set_id}/traces/{artifact_id}
GET /api/v1/workspaces/{workspace_id}
PUT /api/v1/workspaces/{workspace_id}
```

The trace endpoint returns exactly one graph. Side-by-side comparison is browser composition of
multiple independent responses, making accidental topology merging harder.

## Canonical viewer document

Define and freeze `adag.observatory.trace-graph.v1`. It contains:

```text
artifact identity, source hashes, and source schema
model, tokenizer, and code revisions
target token, probability/rank, response position, prefix boundary, and objective
prompt/response tokens and role/position mapping
exact node occurrence and signed-basis identities
node attribution, activation, attribution profile, and contribution profile
exact edge endpoints, signed attribution, and signed weight
trace pruning/configuration and size diagnostics
named display projection and retained-mass accounting
```

Large node profiles may be split into lazy-loaded detail files. A small compatibility adapter may
map this document to inherited frontend field names, but the compatibility object is never an
authoritative artifact.

The bundle should be content-addressed and conceptually shaped as:

```text
viewer-manifest.json
traces/index.json
traces/<artifact-id>/graph-core.json.gz
traces/<artifact-id>/node-details.json.gz
label-sets/<label-set-id>.json.gz
assets/...
```

Build into a temporary sibling directory, validate hashes and referential integrity, then rename
atomically. An existing immutable destination is an error rather than an overwrite.

## Label and review seams

### Label sets

A label approach emits a versioned immutable overlay, not viewer-specific strings embedded in the
graph. Each label-set manifest records:

- label-set ID and content hash;
- model and tokenizer revision;
- signed-basis and polarity-rule schema;
- input artifact/corpus hashes;
- method, code version, configuration, prompt, and random seed where applicable;
- cluster assignment separately from cluster or neuron description;
- description, abstention, confidence if defined, and evidence references.

The first implementation includes a raw-identity view and two synthetic label-set fixtures. This
proves that approaches A and B can be shown side by side before a real labeler is chosen. Later
adapters may consume raw-neuron labels, ADAG cluster assignments/descriptions, or evidence-card
artifacts without changing the trace schema or renderer.

### Workspaces and notes

Viewer state is separate from evidence:

- URL or browser storage holds transient filters, pane arrangement, and selection;
- an explicit Save writes a small versioned workspace containing pins, filters, layout, selected
  label sets, and comments;
- compact traces, label artifacts, and generated graph projections remain read-only.

Use atomic versioned JSON for the first single-user workspace store. Add SQLite only when editing
history, search, or concurrent clients create a real need. Updates use optimistic revisions so two
browser tabs cannot silently overwrite each other. A later frozen review export binds notes to
exact trace, occurrence, signed-basis, and label-set hashes.

## V1 screen and interactions

### Left rail: trace trajectory

List the seven response positions `65, 88, 120, 135, 162, 181, 184` in order, with target token,
local token context, user selection comment, target probability, graph size, and validation state.
Changing the selected position loads a different independent graph.

### Top: exact context

Show the properly formatted prompt and response prefix with token boundaries available on hover,
the prediction boundary marked, and the selected target highlighted. Keep response-relative and
absolute token positions visible without forcing token-box styling on normal reading.

### Center: graph panes

Start with the target logit and strongest upstream evidence rather than all edges at once. Support:

- synchronized hover/click and node pinning;
- token-by-layer overview and focused subgraph;
- path expansion upstream and downstream;
- layer, token position, polarity, signed attribution, absolute attribution, edge attribution,
  edge weight, retained mass, and depth filters;
- exact neuron/basis search;
- an explicit, warned `show full graph` action.

The default edge budget is a display-policy setting, not a new trace threshold. The UI always
reports full and displayed graph sizes.

### Right: evidence card

For the selected node show:

- occurrence and signed-basis identities;
- layer, position, neuron, polarity, activation, and signed attribution;
- input-attribution and output-contribution profiles;
- ranked incoming and outgoing edges with attribution and weight in separate columns;
- each selected label approach in its own column, including unknown/abstained results;
- provenance and optional human notes.

There is no implicit external exemplar lookup in v1. A future exemplar source must be an explicit
offline or remote evidence adapter, and unavailable data must never be translated into a dead
neuron.

### Comparison

V1 includes A/B label columns for the same raw node. The next increment adds at most two target
graphs side by side with synchronized display filters. Exact signed-basis recurrence may be shown
later in a separate neutral alignment lane, dashed and without arrowheads; path search must ignore
it.

## CHPC operation and storage

- Site code and schemas live in Git.
- Immutable source traces remain in the existing VAST result root and retain their hashes.
- Rebuildable safe-JSON projections may live on VAST.
- Saved workspaces and review exports use an explicit persistent home- or group-backed state root.
  Running the site does not make the VAST trace artifacts archival.
- `serve` binds `127.0.0.1` by default and is accessed through SSH port forwarding. It has no
  public-hosting, authentication, or multi-user scope.
- Confirm CHPC policy before leaving an HTTP process on a login node. Technically, JSON-only
  serving of these seven traces should be light. If policy disfavors it, run the identical
  loopback service in a one-CPU Slurm job and forward through the login host.
- Run `sync` in a development allocation if importing the locked scientific environment or
  normalizing future bulk traces is nontrivial. The long-running `serve` process remains CPU-only
  and low-I/O.

Local access will have the shape:

```bash
ssh -N -L 8032:127.0.0.1:8032 <chpc-login-host>
# browse http://127.0.0.1:8032
```

If the service runs on a worker, the tunnel command must include the normal CHPC login-to-worker
hop rather than exposing the worker port.

## Failure behavior

- A checksum, schema, endpoint, or identity failure quarantines that trace with a visible error;
  strict sync fails the whole bundle.
- Unsupported, benchmark-only, or multi-target artifacts are refused.
- Two artifacts with one artifact ID and different hashes fail sync.
- A label set bound to another model, revision, basis schema, or source hash is rejected while the
  raw graph remains viewable.
- Duplicate occurrence IDs, missing edge endpoints, non-finite values, or cross-trace edges fail
  projection validation.
- Corrupt derived cache entries are rebuilt; source artifacts are never repaired by the viewer.
- An unwritable state root makes the site visibly read-only unless writable state was explicitly
  required.
- A stale workspace update returns a conflict rather than overwriting.
- Path traversal, arbitrary file reads, and browser-supplied pickle paths are rejected.

## Implementation stages and gates

### Stage 1: lossless bundle and catalog

1. Freeze the viewer graph, label-set, and workspace schemas.
2. Implement compact-trace discovery, verification, and one-trace-at-a-time projection.
3. Add golden synthetic fixtures for polarity, attribution sign, edge attribution versus weight,
   missing values, and cross-slice false paths.
4. Build and validate a bundle over all seven real traces.

Gate: every source hash remains unchanged; all seven targets appear in deterministic order; exact
raw values round-trip through the viewer document.

### Stage 2: persistent raw viewer

1. Add the loopback-only JSON/static server and catalog page.
2. Adapt the existing D3/dagre renderer to the authoritative document.
3. Add target context, progressive graph navigation, evidence cards, filters, and provenance.
4. Remove source-graph writes and the implicit Modal exemplar dependency from this path.

Gate: the user can inspect every trace, including the 11,165-edge case, without GPU/model loading
in the serving process; the browser never changes a compact artifact or projection.

### Stage 3: persistent workspaces and label comparison

1. Add saved workspaces and comments in the separate state store.
2. Add raw identity and two synthetic label adapters.
3. Show A/B label columns, unknowns, abstentions, method provenance, and disagreement.

Gate: changing a label approach changes annotations only; graph topology and raw measurements are
byte-identical; saved state survives a server restart.

### Stage 4: independent target comparison

1. Add two-pane target comparison with synchronized filters.
2. Add optional exact-basis correspondence in a non-causal alignment lane.
3. Add comparison exports that bind source, display projection, label-set, and workspace hashes.

Gate: no causal path can cross target traces, and every screenshot/export identifies its source
graphs and display projection.

### Stage 5: real labels and evidence

Add real label artifacts one method at a time, followed by local exemplar evidence and limited
intervention results only when their provenance contracts exist. Do not make graph generation,
label inference, or steering part of the viewer server.

## V1 acceptance criteria

V1 is complete when stages 1 through 3 pass their gates and:

1. one documented command synchronizes the seven traces into a validated site bundle;
2. one documented CPU-only command serves it on loopback;
3. the site closely matches the useful Neuronpedia graph workflow without adopting unrelated
   product infrastructure or branding;
4. signed raw node and edge evidence, exact identities, target context, and provenance remain
   inspectable;
5. saved notes and viewer state survive restart without modifying scientific inputs;
6. two label approaches can be compared on the same raw graph through versioned overlays;
7. focused unit, contract, HTTP-safety, and browser tests pass on the largest real graph;
8. license and attribution files cover every reused upstream source file.

## Deliberately deferred

- public hosting, accounts, authentication, and multi-user collaboration;
- live graph generation, model loading, label inference, or steering from the site;
- a Neuronpedia-compatible feature database or remote exemplar dependency;
- automatic semantic neuron names;
- a response-wide merged causal graph;
- SQLite until a concrete persistence requirement exceeds versioned JSON;
- WebGL or a frontend framework rewrite unless measured browser performance requires it.

There are no blocking design questions for stage 1. The plan assumes private single-user use over
an SSH tunnel, immutable trace inputs, and a close interaction-level resemblance to Neuronpedia
rather than a branded clone.
