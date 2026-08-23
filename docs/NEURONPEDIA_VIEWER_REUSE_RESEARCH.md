# Neuronpedia viewer reuse research

Date: 2026-08-23

## Scope and pinned sources

This note asks whether Neuronpedia can be reused as the basis of a persistent,
CHPC-local viewer for the raw-neuron attribution traces. It uses only official
source repositories and the code already vendored in this repository.

- Neuronpedia was inspected at commit
  [`ebb8426fc550ad5d1f76d52c271a624af3c08a57`](https://github.com/hijohnnylin/neuronpedia/tree/ebb8426fc550ad5d1f76d52c271a624af3c08a57).
- Anthropic's attribution-graphs frontend was inspected at its final commit,
  [`18ed0d606b2c8bd068c253321bf7bb7f517e54e3`](https://github.com/anthropics/attribution-graphs-frontend/tree/18ed0d606b2c8bd068c253321bf7bb7f517e54e3).
  GitHub marks that repository archived and read-only.

## Bottom line

Neuronpedia's website, including its current graph viewer, is open source and
reusable. The current repository license is Apache-2.0. However, adopting the
whole Neuronpedia site would also adopt a Next.js/React application coupled to
Postgres/Prisma, authentication, S3-backed uploads, and many services that this
read-only observatory does not need.

The better initial route is to retain this repository's existing, lightweight
Anthropic-derived viewer and server, then port or independently reimplement the
specific Neuronpedia interactions we want. This is not merely a hypothetical
shortcut: the existing code already exports ADAG circuits to the compatible
`metadata + qParams + nodes + links` contract and already adds raw-neuron-specific
views for attribution and contribution maps.

## What is open source and reusable

Neuronpedia's public monorepo contains the complete Next.js/React webapp, graph
page, graph-generation service, Prisma schema, APIs, and related interpretability
services. The official README calls the services standalone and explicitly says
the webapp frontend can be paired with a different inference API. See the
[architecture and service boundaries](https://github.com/hijohnnylin/neuronpedia/blob/ebb8426fc550ad5d1f76d52c271a624af3c08a57/README.md#services-are-standalone-apps).

The relevant graph UI is source-visible under
[`apps/webapp/app/[modelId]/graph`](https://github.com/hijohnnylin/neuronpedia/tree/ebb8426fc550ad5d1f76d52c271a624af3c08a57/apps/webapp/app/%5BmodelId%5D/graph).
Its useful pieces include:

- a token-by-layer link graph using hybrid SVG/canvas rendering, hover/click
  synchronization, pinning, and pruning in
  [`link-graph.tsx`](https://github.com/hijohnnylin/neuronpedia/blob/ebb8426fc550ad5d1f76d52c271a624af3c08a57/apps/webapp/app/%5BmodelId%5D/graph/link-graph.tsx);
- a force-directed, groupable and zoomable focused graph in
  [`subgraph.tsx`](https://github.com/hijohnnylin/neuronpedia/blob/ebb8426fc550ad5d1f76d52c271a624af3c08a57/apps/webapp/app/%5BmodelId%5D/graph/subgraph.tsx);
- connected-node and feature-detail panes, including editable labels, in
  [`node-connections.tsx`](https://github.com/hijohnnylin/neuronpedia/blob/ebb8426fc550ad5d1f76d52c271a624af3c08a57/apps/webapp/app/%5BmodelId%5D/graph/node-connections.tsx)
  and
  [`feature-detail.tsx`](https://github.com/hijohnnylin/neuronpedia/blob/ebb8426fc550ad5d1f76d52c271a624af3c08a57/apps/webapp/app/%5BmodelId%5D/graph/feature-detail.tsx);
- a four-pane composition (link graph, connections, focused subgraph, feature
  details) in
  [`wrapper.tsx`](https://github.com/hijohnnylin/neuronpedia/blob/ebb8426fc550ad5d1f76d52c271a624af3c08a57/apps/webapp/app/%5BmodelId%5D/graph/wrapper.tsx).

The graph JSON contract is public and validated by
[`graph-schema.json`](https://github.com/hijohnnylin/neuronpedia/blob/ebb8426fc550ad5d1f76d52c271a624af3c08a57/apps/webapp/app/api/graph/graph-schema.json).
A graph consists of metadata, saved view state (`qParams`), nodes, and signed
links. Metadata may point either to a Neuronpedia source set or to independently
hosted feature-detail JSON files. This is an appropriate interchange shape for
our viewer, even if we do not reuse Neuronpedia's application shell.

## Why the full site is the wrong first deployment

The official local setup uses Node.js 22+, Postgres 16+ with pgvector, and
recommends at least 16 GB of RAM; the webapp runs on port 3000. See the
[requirements and local webapp instructions](https://github.com/hijohnnylin/neuronpedia/blob/ebb8426fc550ad5d1f76d52c271a624af3c08a57/README.md#requirements).
Its
[`package.json`](https://github.com/hijohnnylin/neuronpedia/blob/ebb8426fc550ad5d1f76d52c271a624af3c08a57/apps/webapp/package.json)
includes Next.js 15, React 19, Prisma, NextAuth, D3, Radix components, AWS and
Vercel clients, Sentry, and many unrelated product features.

The graph page is not a published standalone component. Its graph provider
depends on Next navigation, session and global providers, generated Prisma
types, database-backed graph metadata, and Neuronpedia feature APIs. It fetches
each graph from a URL stored in `GraphMetadata.url`; that model is defined in
the
[`Prisma schema`](https://github.com/hijohnnylin/neuronpedia/blob/ebb8426fc550ad5d1f76d52c271a624af3c08a57/apps/webapp/prisma/schema.prisma#L167-L196).
The stock upload path additionally obtains a signed URL, uploads to S3, and
persists metadata for an authenticated user; see
[`upload-graph-modal.tsx`](https://github.com/hijohnnylin/neuronpedia/blob/ebb8426fc550ad5d1f76d52c271a624af3c08a57/apps/webapp/app/%5BmodelId%5D/graph/upload-graph-modal.tsx).

The GPU graph server is separable and not required to display already-generated
graphs. Its official README describes a FastAPI service for generating new
graphs with circuit-tracer or a complete-replacement-model backend. See
[`apps/graph/README.md`](https://github.com/hijohnnylin/neuronpedia/blob/ebb8426fc550ad5d1f76d52c271a624af3c08a57/apps/graph/README.md).
We already have the seven trace artifacts, so adopting that service would not
help the first viewer milestone.

## Comparison with the viewer already in this repository

This repository already vendors and substantially adapts Anthropic's static
MIT-licensed frontend under [`circuits/frontend/assets`](../circuits/frontend/assets/README.md).
The upstream README and LICENSE are byte-identical to the copies here, while
the graph JavaScript, CSS, and entrypoint have local changes. The local copy
also bundles the browser dependencies that the upstream repository expected a
host page to supply.

The current local stack is much smaller:

- [`circuits/frontend/server.py`](../circuits/frontend/server.py) is a Python
  standard-library HTTP server. It serves local JSON, gzip-compresses large
  responses, saves `qParams`, and optionally proxies neuron exemplars.
- [`circuits/frontend/graph_models.py`](../circuits/frontend/graph_models.py)
  defines the compatible metadata, node, link, and saved-state models.
- [`Circuit.export_to_circuit_tracer`](../circuits/analysis/circuit_ops.py)
  already emits graph JSON and `graph-metadata.json` in the viewer's contract.
- The adapted feature panes already display raw `attr_map` and `contrib_map`
  arrays, which the stock Anthropic and Neuronpedia schemas do not model.

This is a static browser application plus a small CPU server, not a database
application. That is a better operational fit for an SSH-port-forwarded viewer
on a CHPC login node. One safety change should precede persistent use: the
current server binds to all interfaces (`""`), while a port-forward-only
service should default to loopback.

## Semantic compatibility and required adapter

The container format is compatible, but the node meanings are not identical.
Neuronpedia assumes transcoder/CLT features, embeddings, logits, reconstruction
errors, and now some attention-feature types. Its TypeScript contract is
documented in
[`graph-types.ts`](https://github.com/hijohnnylin/neuronpedia/blob/ebb8426fc550ad5d1f76d52c271a624af3c08a57/apps/webapp/app/%5BmodelId%5D/graph/graph-types.ts).
Our compact traces contain raw MLP neurons. The existing exporter currently
encodes those as `cross layer transcoder` nodes to fit the inherited renderer.

The persistent viewer should therefore add an explicit raw-neuron node type and
a deterministic adapter from `adag.compact-trace.v1`; it should not silently
describe a raw neuron as a transcoder feature. Graph provenance should remain
immutable. Labels and human annotations should be separate, versioned overlays
keyed by graph identity and node identity, so multiple labeling approaches can
be switched and compared without rewriting trace artifacts.

## Licensing and attribution obligations

Neuronpedia's current
[`LICENSE`](https://github.com/hijohnnylin/neuronpedia/blob/ebb8426fc550ad5d1f76d52c271a624af3c08a57/LICENSE)
is Apache-2.0. If we redistribute copied or modified Neuronpedia code, Apache
2.0 section 4 requires us to provide the license, mark modified files, retain
pertinent source notices, and reproduce relevant notices from its
[`NOTICE`](https://github.com/hijohnnylin/neuronpedia/blob/ebb8426fc550ad5d1f76d52c271a624af3c08a57/NOTICE).
The license does not grant permission to use Neuronpedia's trademarks or
product branding except to describe the origin of the code. We should use our
own name and logo.

Neuronpedia's NOTICE specifically records that its graph viewer includes a
TypeScript port of Anthropic's fork of `d3-jetpack`, ultimately BSD-3-Clause;
that notice must remain with copied code. The root NOTICE also explains that
older Neuronpedia releases were MIT-licensed, but current `main` should be
treated as Apache-2.0 rather than assuming the old license.

Anthropic's official
[`attribution-graphs-frontend`](https://github.com/anthropics/attribution-graphs-frontend/tree/18ed0d606b2c8bd068c253321bf7bb7f517e54e3)
is MIT-licensed. Its copyright and permission notice must remain in copies or
substantial portions; the vendored copy already includes that
[`LICENSE`](../circuits/frontend/assets/LICENSE). Any added third-party browser
libraries also need their own notices audited before distribution. This is an
engineering summary, not legal advice.

## Recommended implementation direction

1. Keep the existing static viewer and lightweight Python server as the
   foundation.
2. Add a provenance-preserving compact-trace adapter and a manifest-driven
   graph selector for the seven traces.
3. Bring over the Neuronpedia interaction model in small pieces: synchronized
   hover/click state, pruning controls, a clearer connections pane, focused
   subgraph, and feature-details pane. Port source only where it saves real
   work; otherwise extend the already-adapted MIT code.
4. Define a label-provider interface immediately. Store imported label sets
   immutably and save human annotations separately; use local versioned JSON at
   first, adding SQLite only if indexing, editing history, or concurrent access
   warrants it.
5. Run the viewer CPU-only on loopback and access it with SSH port forwarding.
   Graph generation and model inference remain separate scheduled workflows.

## Unresolved gaps before implementation

- Specify and test the exact mapping from each compact-trace table and
  provenance field to viewer nodes, edges, positions, and display weights.
- Decide which raw quantity controls initial pruning; Neuronpedia's `influence`
  semantics come from circuit-tracer and must not be assumed equivalent to
  every ADAG attribution field.
- Decide whether the first milestone needs persisted interactive layouts and
  annotations, or only importable/exportable JSON overlays.
- Pin any individual Neuronpedia files actually copied. `main` is active and
  should not become an unrecorded moving dependency.
- Confirm the local CHPC policy for a long-lived login-node HTTP process. The
  expected viewer workload is light, but operational permission is separate
  from technical resource use.
