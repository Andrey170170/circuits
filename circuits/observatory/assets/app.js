import { GraphView } from "./graph-view.js";
import { profileDisplay } from "./profile-data.js";

const TRACE_SCHEMA = "adag.observatory.trace-graph.v1";
const WORKSPACE_SCHEMA = "adag.observatory.workspace.v1";
const DEFAULT_FILTERS = Object.freeze({
  layer: "all",
  position: "all",
  inputTokens: "hide",
  polarity: "all",
  magnitude: "0",
  mass: "0.9",
  edgeBudget: "100",
});

const $ = (selector) => document.querySelector(selector);
const elements = {
  app: $("#app"),
  modelValue: $("#model-value"),
  targetSelect: $("#target-select"),
  targetList: $("#target-list"),
  labelPrimary: $("#label-primary-select"),
  labelA: $("#label-a-select"),
  labelB: $("#label-b-select"),
  saveButton: $("#save-workspace"),
  saveState: $("#save-state"),
  contextHeading: $("#context-heading"),
  contextToggle: $("#context-toggle"),
  tokenContext: $("#token-context"),
  layerFilter: $("#layer-filter"),
  positionFilter: $("#position-filter"),
  inputTokenFilter: $("#input-token-filter"),
  polarityFilter: $("#polarity-filter"),
  magnitudeFilter: $("#magnitude-filter"),
  massFilter: $("#mass-filter"),
  edgeBudgetFilter: $("#edge-budget-filter"),
  neuronSearch: $("#neuron-search"),
  clearFilters: $("#clear-filters"),
  graphStatus: $("#graph-status"),
  graphSvg: $("#graph-svg"),
  fitGraph: $("#fit-graph"),
  zoomIn: $("#zoom-in"),
  zoomOut: $("#zoom-out"),
  resetGraph: $("#reset-graph"),
  evidenceEmpty: $("#evidence-empty"),
  evidenceContent: $("#evidence-content"),
  occurrenceId: $("#occurrence-id"),
  basisId: $("#basis-id"),
  identityFields: $("#identity-fields"),
  activationValue: $("#activation-value"),
  activationBar: $("#activation-bar"),
  attributionValue: $("#attribution-value"),
  attributionBar: $("#attribution-bar"),
  attrProfile: $("#attr-profile"),
  contribProfile: $("#contrib-profile"),
  incoming: $("#incoming-connections"),
  outgoing: $("#outgoing-connections"),
  labelAName: $("#label-a-name"),
  labelAValue: $("#label-a-value"),
  labelBName: $("#label-b-name"),
  labelBValue: $("#label-b-value"),
  syntheticWarning: $("#synthetic-label-warning"),
  neuronComment: $("#neuron-comment"),
  pinNode: $("#pin-node"),
  provenanceDrawer: $("#provenance-drawer"),
  provenanceToggle: $("#provenance-toggle"),
  provenanceContent: $("#provenance-content"),
  provArtifact: $("#prov-artifact"),
  provHash: $("#prov-hash"),
  rawNodeCount: $("#raw-node-count"),
  rawEdgeCount: $("#raw-edge-count"),
  displayNodeCount: $("#display-node-count"),
  displayEdgeCount: $("#display-edge-count"),
  retainedMass: $("#retained-mass"),
  projectionNote: $("#projection-note"),
  trajectoryRail: $("#trajectory-rail"),
  evidenceRail: $("#evidence-rail"),
  trajectoryToggle: $("#trajectory-toggle"),
  evidenceToggle: $("#evidence-toggle"),
  trajectoryClose: $("#trajectory-close"),
  evidenceClose: $("#evidence-close"),
  drawerScrim: $("#drawer-scrim"),
  toastRegion: $("#toast-region"),
};

const state = {
  catalog: [],
  site: null,
  labelSets: [],
  overlayCache: new Map(),
  selectedArtifactId: null,
  traceDocument: null,
  graph: null,
  projection: null,
  selectedNodeId: null,
  labels: { a: "raw", b: "none" },
  filters: { ...DEFAULT_FILTERS },
  workspace: {
    revision: 0,
    comments: {},
    pinnedNodeIds: [],
    readOnly: false,
  },
  dirty: false,
  contextExpanded: false,
  loadController: null,
  expandedProfiles: new Set(),
};

let graphView;
let noteTimer;

class HttpError extends Error {
  constructor(message, response, body = null) {
    super(message);
    this.name = "HttpError";
    this.status = response.status;
    this.body = body;
  }
}

function firstDefined(...values) {
  return values.find((value) => value !== undefined && value !== null);
}

function finiteNumber(value, fallback = null) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function textOrNull(value) {
  return value === undefined || value === null || value === "" ? null : String(value);
}

function safeTokenText(text, tokenId) {
  if (text !== undefined && text !== null) return String(text);
  return tokenId !== undefined && tokenId !== null ? `[token id ${tokenId}]` : "[token text unavailable]";
}

function normalizePolarity(value) {
  if (value === undefined || value === null || value === "") return "unknown";
  if (typeof value === "number") return value < 0 ? "negative" : "positive";
  const lowered = String(value).trim().toLowerCase();
  if (["+", "+1", "1", "pos", "positive"].includes(lowered)) return "positive";
  if (["-", "−", "-1", "neg", "negative"].includes(lowered)) return "negative";
  return lowered;
}

function recordId(record, fallback) {
  if (!record || typeof record !== "object") return fallback;
  return String(
    firstDefined(
      record.id,
      record.occurrence_id,
      record.basis_id,
      record.canonical_id,
      record.identity,
      fallback,
    ),
  );
}

function formatNumber(value, digits = 3) {
  const number = finiteNumber(value);
  if (number === null) return "—";
  if (number === 0) return "0";
  const magnitude = Math.abs(number);
  if (magnitude >= 1000 || magnitude < 0.001) return number.toExponential(digits).replace("e+", "e");
  return number.toLocaleString(undefined, { maximumFractionDigits: Math.max(digits, 4) });
}

function formatMagnitude(value) {
  const number = finiteNumber(value);
  return number === null ? "—" : formatNumber(Math.abs(number));
}

function formatSigned(value) {
  const number = finiteNumber(value);
  if (number === null) return "—";
  if (number > 0) return `+${formatNumber(number)}`;
  if (number < 0) return `−${formatNumber(Math.abs(number))}`;
  return "0";
}

function formatPercent(value) {
  const number = finiteNumber(value);
  return number === null ? "—" : `${(number * 100).toFixed(number >= 0.9995 ? 0 : 1)}%`;
}

function formatCount(value) {
  const number = finiteNumber(value);
  return number === null ? "—" : Math.trunc(number).toLocaleString();
}

function shortHash(value) {
  const text = textOrNull(value);
  if (!text) return "";
  return text.length > 22 ? `${text.slice(0, 12)}…${text.slice(-8)}` : text;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    ...options,
    headers: { Accept: "application/json", ...options.headers },
  });
  let body = null;
  const contentType = response.headers.get("content-type") ?? "";
  if (contentType.includes("json")) {
    body = await response.json().catch(() => null);
  } else if (!response.ok) {
    body = await response.text().catch(() => null);
  }
  if (!response.ok) {
    const detail = body?.detail ?? body?.error ?? body?.message ?? response.statusText;
    throw new HttpError(`${response.status}: ${detail}`, response, body);
  }
  return body;
}

function normalizeCatalog(payload) {
  const traces = Array.isArray(payload) ? payload : payload?.traces;
  if (!Array.isArray(traces)) throw new Error("Catalog response does not contain a traces array.");
  return traces
    .map((trace, index) => {
      const artifactId = textOrNull(firstDefined(trace.artifact_id, trace.id, trace.slug));
      if (!artifactId) return null;
      return {
        ...trace,
        artifact_id: artifactId,
        response_position: firstDefined(trace.response_position, trace.target?.response_position, index),
        prediction_position: firstDefined(trace.prediction_position, trace.target?.prediction_position),
        token_id: firstDefined(trace.token_id, trace.target_token_id, trace.target?.token_id),
        token_text: firstDefined(trace.token_text, trace.target_token, trace.target?.token_text),
        node_count: finiteNumber(firstDefined(trace.node_count, trace.diagnostics?.node_count), 0),
        edge_count: finiteNumber(firstDefined(trace.edge_count, trace.diagnostics?.edge_count), 0),
        model_id: firstDefined(trace.model_id, trace.model?.id, trace.model?.model_id),
        model_revision: firstDefined(trace.model_revision, trace.model?.revision),
      };
    })
    .filter(Boolean)
    .sort((a, b) => Number(a.response_position) - Number(b.response_position));
}

function normalizeLabelSets(payload) {
  const sets = Array.isArray(payload) ? payload : payload?.label_sets ?? payload?.sets ?? [];
  return sets
    .map((item) => {
      const id = textOrNull(firstDefined(item.id, item.label_set_id, item.slug));
      if (!id) return null;
      return {
        ...item,
        id,
        name: firstDefined(item.name, item.display_name, item.title, id),
        synthetic: Boolean(firstDefined(item.synthetic, item.is_synthetic, item.fixture, false)),
      };
    })
    .filter(Boolean);
}

function normalizeNode(node, index) {
  const occurrence = node.occurrence && typeof node.occurrence === "object" ? node.occurrence : {};
  const basis = node.basis && typeof node.basis === "object" ? node.basis : {};
  const fallbackId = `node-${index}`;
  const id = String(firstDefined(node.id, node.occurrence_id, occurrence.id, occurrence.occurrence_id, fallbackId));
  const layer = firstDefined(node.layer, occurrence.layer, basis.layer, node.layer_index);
  const position = firstDefined(
    node.position,
    node.token_position,
    occurrence.position,
    occurrence.token_position,
    occurrence.context_position,
  );
  const neuron = firstDefined(
    node.neuron,
    node.neuron_index,
    occurrence.neuron,
    occurrence.neuron_index,
    basis.neuron,
    basis.neuron_index,
    node.feature,
  );
  const polarity = normalizePolarity(
    firstDefined(node.activation_polarity, node.polarity, occurrence.polarity, basis.polarity),
  );
  return {
    ...node,
    id,
    occurrence,
    basis,
    occurrenceId: recordId(occurrence, id),
    basisId: String(
      firstDefined(
        node.basis_id,
        recordId(basis, `${textOrNull(layer) ?? "?"}:${textOrNull(neuron) ?? "?"}:${polarity}`),
      ),
    ),
    layer,
    position,
    neuron,
    polarity,
    kind: String(firstDefined(node.kind, occurrence.kind, node.type, node.feature_type, "neuron")),
    attribution: finiteNumber(firstDefined(node.attribution, node.influence)),
    activation: finiteNumber(node.activation),
    attributionMap: firstDefined(node.attribution_map, node.attr_map, []),
    contributionMap: firstDefined(node.contribution_map, node.contrib_map, []),
  };
}

function endpointId(value) {
  if (value && typeof value === "object") {
    return String(firstDefined(value.id, value.occurrence_id, value.node_id, ""));
  }
  return String(value ?? "");
}

function normalizeGraph(payload) {
  if (!payload || typeof payload !== "object") throw new Error("Trace response is not a JSON object.");
  const target = payload.target ?? {};
  const nodes = (Array.isArray(payload.nodes) ? payload.nodes : []).map((sourceNode, index) => {
    const node = normalizeNode(sourceNode, index);
    if (/logit|output|target/i.test(node.kind)) {
      node.token_text = firstDefined(node.token_text, target.token_text);
      node.token_id = firstDefined(node.token_id, target.token_id);
    }
    return node;
  });
  const nodeById = new Map(nodes.map((node) => [node.id, node]));
  const edges = (Array.isArray(payload.edges) ? payload.edges : payload.links ?? [])
    .map((edge, index) => ({
      ...edge,
      id: String(firstDefined(edge.id, edge.edge_id, edge.ordinal !== undefined ? `edge-${edge.ordinal}` : `edge-${index}`)),
      source: endpointId(edge.source),
      target: endpointId(edge.target),
      attribution: finiteNumber(edge.attribution),
      weight: finiteNumber(edge.weight),
    }))
    .filter((edge) => nodeById.has(edge.source) && nodeById.has(edge.target));
  return { nodes, edges, nodeById };
}

function normalizeContext(document) {
  const context = document.context ?? {};
  const target = document.target ?? {};
  let sourceTokens = Array.isArray(context.tokens) ? context.tokens : [];
  const tokenIds = Array.isArray(context.token_ids) ? context.token_ids : [];
  const tokenTexts = Array.isArray(context.token_texts) ? context.token_texts : [];
  const roles = Array.isArray(context.roles) ? context.roles : [];
  if (sourceTokens.length === 0 && (tokenIds.length || tokenTexts.length)) {
    sourceTokens = Array.from({ length: Math.max(tokenIds.length, tokenTexts.length) }, (_, index) => ({
      token_id: tokenIds[index],
      text: tokenTexts[index],
      role: roles[index],
    }));
  }

  const tokens = sourceTokens.map((token, index) => {
    const record = token && typeof token === "object" ? token : { text: token };
    const tokenId = firstDefined(record.token_id, record.id, tokenIds[index]);
    return {
      id: tokenId,
      text: safeTokenText(firstDefined(record.text, record.token_text, record.token), tokenId),
      index,
      absolutePosition: firstDefined(record.absolute_position, record.position, context.start_position !== undefined ? Number(context.start_position) + index : index),
      responsePosition: firstDefined(record.response_position, record.output_position),
      role: firstDefined(record.role, roles[index]),
    };
  });

  const predictionPosition = finiteNumber(firstDefined(target.prediction_position, context.prediction_position));
  const absolutePosition = finiteNumber(
    firstDefined(target.observed_absolute_position, target.absolute_position, target.position),
  );
  let selectedIndex = tokens.findIndex(
    (token) => absolutePosition !== null && finiteNumber(token.absolutePosition) === absolutePosition,
  );
  if (selectedIndex < 0 && predictionPosition !== null && predictionPosition >= 0 && predictionPosition < tokens.length) {
    selectedIndex = predictionPosition;
  }
  if (selectedIndex < 0 && target.token_id !== undefined && tokens.length) {
    selectedIndex = tokens.findLastIndex((token) => String(token.id) === String(target.token_id));
  }

  if (selectedIndex < 0 && target.token_id !== undefined) {
    const index = predictionPosition !== null ? Math.min(Math.max(predictionPosition, 0), tokens.length) : tokens.length;
    tokens.splice(index, 0, {
      id: target.token_id,
      text: safeTokenText(target.token_text, target.token_id),
      index,
      absolutePosition: absolutePosition ?? "?",
      responsePosition: target.response_position,
      role: "assistant",
    });
    selectedIndex = index;
  }

  return { tokens, selectedIndex, predictionPosition, absolutePosition };
}

function graphNodeLabel(node) {
  if (!node) return "unknown";
  const kind = node.kind.toLowerCase();
  if (kind.includes("logit") || kind.includes("output")) return `OUT:${safeTokenText(node.token_text, node.neuron)}`;
  if (kind.includes("embed") || kind.includes("input")) return `IN:${textOrNull(node.position) ?? "?"}`;
  return `L${textOrNull(node.layer) ?? "?"}:N${textOrNull(node.neuron) ?? "?"}`;
}

function exactNeuronAliases(node) {
  const layer = textOrNull(node.layer);
  const neuron = textOrNull(node.neuron);
  const aliases = new Set([node.id, node.occurrenceId, node.basisId, graphNodeLabel(node)]);
  if (layer && neuron) {
    aliases.add(`L${layer}:${neuron}`);
    aliases.add(`L${layer}:N${neuron}`);
    aliases.add(`${layer}:${neuron}`);
    const sign = node.polarity === "negative" ? "-" : node.polarity === "positive" ? "+" : node.polarity;
    aliases.add(`L${layer}:N${neuron}:${sign}`);
    aliases.add(`${layer}:${neuron}:${sign}`);
  }
  return new Set([...aliases].filter(Boolean).map((value) => String(value).trim().toLowerCase()));
}

function selectedCatalogTrace() {
  return state.catalog.find((item) => item.artifact_id === state.selectedArtifactId) ?? null;
}

function selectedNode() {
  return state.graph?.nodeById.get(state.selectedNodeId) ?? null;
}

function markDirty() {
  state.dirty = true;
  if (!state.workspace.readOnly) {
    elements.saveState.textContent = "Unsaved changes";
    elements.saveButton.disabled = false;
  }
}

function showToast(message, kind = "info") {
  const toast = document.createElement("div");
  toast.className = `toast ${kind}`;
  toast.textContent = message;
  elements.toastRegion.append(toast);
  window.setTimeout(() => toast.remove(), 4200);
}

function setGraphStatus(kind, message, retry = false) {
  elements.graphStatus.hidden = kind === "ready";
  elements.graphStatus.classList.toggle("error", kind === "error");
  if (kind === "ready") return;
  elements.graphStatus.replaceChildren();
  if (kind === "loading") {
    const spinner = document.createElement("div");
    spinner.className = "spinner";
    spinner.setAttribute("aria-hidden", "true");
    elements.graphStatus.append(spinner);
  }
  const paragraph = document.createElement("p");
  paragraph.textContent = message;
  elements.graphStatus.append(paragraph);
  if (retry) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = "Retry";
    button.addEventListener("click", () => loadTrace(state.selectedArtifactId));
    elements.graphStatus.append(button);
  }
}

function populateCatalog() {
  elements.targetList.replaceChildren();
  elements.targetSelect.replaceChildren();
  if (state.catalog.length === 0) {
    const empty = document.createElement("li");
    empty.className = "rail-empty";
    empty.innerHTML = "<p>No trace artifacts are available in this catalog.</p>";
    elements.targetList.append(empty);
    const option = new Option("No traces", "");
    elements.targetSelect.append(option);
    elements.targetSelect.disabled = true;
    return;
  }

  state.catalog.forEach((trace, index) => {
    const token = safeTokenText(trace.token_text, trace.token_id);
    const option = new Option(`position ${trace.response_position} · ${token}`, trace.artifact_id);
    elements.targetSelect.append(option);

    const row = document.createElement("li");
    row.className = "target-row";
    row.dataset.artifactId = trace.artifact_id;
    const button = document.createElement("button");
    button.type = "button";
    button.innerHTML = `
      <span class="target-order">${index + 1}</span>
      <span class="target-position">${escapeHtml(trace.response_position)}</span>
      <span class="target-token" title="${escapeHtml(token)}">${escapeHtml(token)}</span>
      <span class="target-meta">
        <span>${formatCount(trace.node_count)}n · ${formatCount(trace.edge_count)}e</span>
        <span>${trace.probability !== undefined ? `p ${formatNumber(trace.probability)}` : "p —"}</span>
        <span class="target-comment" title="${escapeHtml(trace.comment ?? "No selection comment")}">${escapeHtml(trace.comment ?? "No selection comment")}</span>
      </span>`;
    button.addEventListener("click", () => selectArtifact(trace.artifact_id));
    row.append(button);
    elements.targetList.append(row);
  });
  elements.targetSelect.disabled = false;
  updateCatalogSelection();
}

function updateCatalogSelection() {
  elements.targetSelect.value = state.selectedArtifactId ?? "";
  elements.targetList.querySelectorAll(".target-row").forEach((row) => {
    const selected = row.dataset.artifactId === state.selectedArtifactId;
    row.classList.toggle("selected", selected);
    row.querySelector("button")?.setAttribute("aria-current", selected ? "true" : "false");
  });
}

function populateLabelSets() {
  const options = [
    { id: "raw", name: "Raw identity" },
    ...state.labelSets,
  ];
  const fill = (select, includeNone) => {
    select.replaceChildren();
    if (includeNone) select.append(new Option("None", "none"));
    options.forEach((item) => select.append(new Option(item.name, item.id)));
    select.disabled = false;
  };
  fill(elements.labelPrimary, false);
  fill(elements.labelA, false);
  fill(elements.labelB, true);
  if (!state.labels.b || state.labels.b === "none") state.labels.b = state.labelSets[0]?.id ?? "none";
  elements.labelPrimary.value = options.some((item) => item.id === state.labels.a) ? state.labels.a : "raw";
  elements.labelA.value = elements.labelPrimary.value;
  elements.labelB.value = ["none", ...state.labelSets.map((item) => item.id)].includes(state.labels.b) ? state.labels.b : "none";
  state.labels.a = elements.labelA.value;
  state.labels.b = elements.labelB.value;
}

function renderContext() {
  elements.tokenContext.replaceChildren();
  const context = normalizeContext(state.traceDocument);
  const catalog = selectedCatalogTrace();
  const roles = new Set(context.tokens.map((token) => token.role).filter(Boolean));
  elements.contextHeading.textContent = roles.size === 1 && roles.has("assistant") ? "Reasoning tokens (model output)" : "Prompt and reasoning tokens";

  const localRadiusBefore = 48;
  const localRadiusAfter = 8;
  const canCollapse = context.tokens.length > localRadiusBefore + localRadiusAfter + 1;
  const windowStart = canCollapse && !state.contextExpanded
    ? Math.max(0, context.selectedIndex - localRadiusBefore)
    : 0;
  const windowEnd = canCollapse && !state.contextExpanded
    ? Math.min(context.tokens.length, context.selectedIndex + localRadiusAfter + 1)
    : context.tokens.length;
  const visibleTokens = context.tokens.slice(windowStart, windowEnd);
  elements.contextToggle.hidden = !canCollapse;
  elements.contextToggle.textContent = state.contextExpanded ? "Show target window" : "Show full prefix";
  elements.contextToggle.setAttribute("aria-expanded", String(state.contextExpanded));

  if (windowStart > 0) {
    const ellipsis = document.createElement("span");
    ellipsis.className = "context-ellipsis";
    ellipsis.textContent = `… ${windowStart} earlier tokens`;
    elements.tokenContext.append(ellipsis);
  }

  let previousRole = visibleTokens[0]?.role ?? null;
  visibleTokens.forEach((token) => {
    if (previousRole && token.role && token.role !== previousRole) {
      const breakElement = document.createElement("span");
      breakElement.className = "token-role-break";
      breakElement.setAttribute("aria-hidden", "true");
      elements.tokenContext.append(breakElement);
    }
    const button = document.createElement("button");
    button.type = "button";
    button.className = "context-token";
    button.textContent = token.text;
    button.dataset.position = String(token.absolutePosition ?? index);
    const isSelected = token.index === context.selectedIndex;
    const isPredicted = token.index === context.selectedIndex || (context.predictionPosition !== null && token.index === context.predictionPosition);
    button.classList.toggle("selected", isSelected);
    button.classList.toggle("predicted", isPredicted);
    button.title = [
      `token id: ${token.id ?? "unavailable"}`,
      `absolute position: ${token.absolutePosition ?? "unavailable"}`,
      `response position: ${token.responsePosition ?? catalog?.response_position ?? "unavailable"}`,
      token.role ? `role: ${token.role}` : null,
      isPredicted ? `prediction prefix length: ${state.traceDocument.target?.prediction_position ?? "unavailable"}` : null,
    ]
      .filter(Boolean)
      .join("\n");
    elements.tokenContext.append(button);
    previousRole = token.role ?? previousRole;
  });

  if (windowEnd < context.tokens.length) {
    const ellipsis = document.createElement("span");
    ellipsis.className = "context-ellipsis";
    ellipsis.textContent = `${context.tokens.length - windowEnd} later tokens …`;
    elements.tokenContext.append(ellipsis);
  }

  if (context.tokens.length === 0) {
    const empty = document.createElement("p");
    empty.className = "connection-empty";
    empty.textContent = "Token context is unavailable for this trace.";
    elements.tokenContext.append(empty);
  }
}

function populateGraphFilters() {
  const preserve = { layer: state.filters.layer, position: state.filters.position };
  const layers = [...new Set(state.graph.nodes.map((node) => node.layer).filter((value) => value !== undefined && value !== null))]
    .sort((a, b) => Number(a) - Number(b));
  const positions = [...new Set(state.graph.nodes.map((node) => node.position).filter((value) => value !== undefined && value !== null))]
    .sort((a, b) => Number(a) - Number(b));

  elements.layerFilter.replaceChildren(new Option("All", "all"));
  layers.forEach((layer) => elements.layerFilter.append(new Option(`Layer ${layer}`, String(layer))));
  elements.positionFilter.replaceChildren(new Option("All", "all"));
  positions.forEach((position) => elements.positionFilter.append(new Option(String(position), String(position))));
  if ([...elements.layerFilter.options].some((option) => option.value === preserve.layer)) elements.layerFilter.value = preserve.layer;
  else state.filters.layer = "all";
  if ([...elements.positionFilter.options].some((option) => option.value === preserve.position)) elements.positionFilter.value = preserve.position;
  else state.filters.position = "all";
  syncFilterControls();
}

function syncFilterControls() {
  elements.layerFilter.value = state.filters.layer;
  elements.positionFilter.value = state.filters.position;
  elements.inputTokenFilter.value = state.filters.inputTokens;
  elements.polarityFilter.value = state.filters.polarity;
  elements.magnitudeFilter.value = state.filters.magnitude;
  elements.massFilter.value = state.filters.mass;
  elements.edgeBudgetFilter.value = state.filters.edgeBudget;
}

function nodePassesFilters(node) {
  if (state.filters.inputTokens === "hide" && node.kind === "input_token") return false;
  if (state.filters.layer !== "all" && String(node.layer) !== state.filters.layer) return false;
  if (state.filters.position !== "all" && String(node.position) !== state.filters.position) return false;
  if (state.filters.polarity !== "all" && node.polarity !== state.filters.polarity) return false;
  if (Math.abs(node.attribution ?? 0) < Number(state.filters.magnitude)) return false;
  return true;
}

function edgeHigherPriority(left, right) {
  const difference = Math.abs(left.attribution ?? 0) - Math.abs(right.attribution ?? 0);
  if (difference !== 0) return difference > 0;
  return String(left.id) < String(right.id);
}

function heapPush(heap, edge) {
  heap.push(edge);
  let index = heap.length - 1;
  while (index > 0) {
    const parent = Math.floor((index - 1) / 2);
    if (!edgeHigherPriority(heap[index], heap[parent])) break;
    [heap[index], heap[parent]] = [heap[parent], heap[index]];
    index = parent;
  }
}

function heapPop(heap) {
  if (heap.length === 0) return null;
  const first = heap[0];
  const last = heap.pop();
  if (heap.length > 0) {
    heap[0] = last;
    let index = 0;
    while (true) {
      const left = index * 2 + 1;
      const right = left + 1;
      let next = index;
      if (left < heap.length && edgeHigherPriority(heap[left], heap[next])) next = left;
      if (right < heap.length && edgeHigherPriority(heap[right], heap[next])) next = right;
      if (next === index) break;
      [heap[index], heap[next]] = [heap[next], heap[index]];
      index = next;
    }
  }
  return first;
}

function createProjection() {
  const eligibleNodes = state.graph.nodes.filter(nodePassesFilters);
  const eligibleNodeIds = new Set(eligibleNodes.map((node) => node.id));
  const eligibleEdges = state.graph.edges.filter(
    (edge) => eligibleNodeIds.has(edge.source) && eligibleNodeIds.has(edge.target),
  );
  const totalMass = eligibleEdges.reduce((sum, edge) => sum + Math.abs(edge.attribution ?? 0), 0);
  const requestedMass = Number(state.filters.mass);
  const budget = state.filters.edgeBudget === "all" ? Infinity : Number(state.filters.edgeBudget);
  const displayedEdges = [];
  const displayedEdgeIds = new Set();
  const targetNodeIds = eligibleNodes
    .filter((node) => /logit|output|target/i.test(node.kind))
    .map((node) => node.id);
  let displayedMass = 0;

  if (targetNodeIds.length > 0) {
    const incoming = new Map(eligibleNodes.map((node) => [node.id, []]));
    eligibleEdges.forEach((edge) => incoming.get(edge.target)?.push(edge));
    const frontier = [];
    const expandedNodes = new Set();
    const enqueueIncoming = (nodeId) => {
      if (expandedNodes.has(nodeId)) return;
      expandedNodes.add(nodeId);
      (incoming.get(nodeId) ?? []).forEach((edge) => heapPush(frontier, edge));
    };
    targetNodeIds.forEach(enqueueIncoming);
    while (frontier.length > 0 && displayedEdges.length < budget) {
      if (displayedEdges.length > 0 && totalMass > 0 && displayedMass / totalMass >= requestedMass) break;
      const edge = heapPop(frontier);
      if (!edge || displayedEdgeIds.has(edge.id)) continue;
      displayedEdgeIds.add(edge.id);
      displayedEdges.push(edge);
      displayedMass += Math.abs(edge.attribution ?? 0);
      enqueueIncoming(edge.source);
    }
  } else {
    const sortedEdges = [...eligibleEdges].sort((a, b) => (
      edgeHigherPriority(a, b) ? -1 : edgeHigherPriority(b, a) ? 1 : 0
    ));
    for (const edge of sortedEdges) {
      if (displayedEdges.length >= budget) break;
      if (displayedEdges.length > 0 && totalMass > 0 && displayedMass / totalMass >= requestedMass) break;
      displayedEdges.push(edge);
      displayedMass += Math.abs(edge.attribution ?? 0);
    }
  }

  const displayedNodeIds = new Set(targetNodeIds);
  displayedEdges.forEach((edge) => {
    displayedNodeIds.add(edge.source);
    displayedNodeIds.add(edge.target);
  });
  if (state.selectedNodeId && eligibleNodeIds.has(state.selectedNodeId)) displayedNodeIds.add(state.selectedNodeId);
  if (displayedNodeIds.size === 0) {
    eligibleNodes
      .sort((a, b) => Math.abs(b.attribution ?? 0) - Math.abs(a.attribution ?? 0))
      .slice(0, 80)
      .forEach((node) => displayedNodeIds.add(node.id));
  }

  return {
    nodes: state.graph.nodes.filter((node) => displayedNodeIds.has(node.id)),
    edges: displayedEdges,
    eligibleNodeCount: eligibleNodes.length,
    eligibleEdgeCount: eligibleEdges.length,
    retainedMass: totalMass > 0 ? displayedMass / totalMass : eligibleEdges.length === 0 ? 1 : 0,
    displayedMass,
    totalMass,
    truncatedByBudget: displayedEdges.length >= budget && displayedEdges.length < eligibleEdges.length,
    targetAnchored: targetNodeIds.length > 0,
  };
}

function renderGraph({ fit = true } = {}) {
  if (!state.graph) return;
  state.projection = createProjection();
  const { nodes, edges } = state.projection;
  updateProvenance();
  if (nodes.length === 0) {
    graphView.render([], []);
    setGraphStatus("empty", "No neurons match the current filters. Clear filters to restore the graph.");
    return;
  }
  setGraphStatus("ready");
  graphView.render(nodes, edges, { selectedNodeId: state.selectedNodeId, fit });
}

function traceDiagnostics() {
  return state.traceDocument?.diagnostics ?? {};
}

function updateProvenance() {
  const document = state.traceDocument ?? {};
  const catalog = selectedCatalogTrace() ?? {};
  const artifact = document.artifact ?? {};
  const diagnostics = traceDiagnostics();
  const artifactId = firstDefined(artifact.id, artifact.artifact_id, catalog.artifact_id, state.selectedArtifactId);
  const hash = firstDefined(
    artifact.source_hash,
    artifact.data_sha256,
    artifact.sha256,
    document.data_sha256,
    catalog.data_sha256,
    catalog.source_hash,
  );
  elements.provArtifact.textContent = textOrNull(artifactId) ?? "—";
  elements.provArtifact.title = textOrNull(artifactId) ?? "";
  elements.provHash.textContent = hash ? `sha256 ${shortHash(hash)}` : "hash unavailable";
  elements.provHash.title = hash ?? "";
  elements.rawNodeCount.textContent = formatCount(
    firstDefined(
      diagnostics.source_node_count,
      diagnostics.raw_node_count,
      diagnostics.node_count,
      catalog.node_count,
      state.graph?.nodes.length,
    ),
  );
  elements.rawEdgeCount.textContent = formatCount(
    firstDefined(
      diagnostics.source_edge_count,
      diagnostics.raw_edge_count,
      diagnostics.edge_count,
      catalog.edge_count,
      state.graph?.edges.length,
    ),
  );
  elements.displayNodeCount.textContent = formatCount(state.projection?.nodes.length);
  elements.displayEdgeCount.textContent = formatCount(state.projection?.edges.length);
  elements.retainedMass.textContent = formatPercent(state.projection?.retainedMass);
  const notes = ["of eligible edge |attribution|"];
  if (state.projection?.targetAnchored) notes.push("target-connected upstream projection");
  if (state.filters.inputTokens === "hide") notes.push("input nodes hidden; full profiles remain in neuron evidence");
  if (state.projection?.truncatedByBudget) notes.push(`limited to ${formatCount(state.projection.edges.length)} strongest edges`);
  if (state.filters.edgeBudget === "all") notes.push("full eligible graph requested");
  elements.projectionNote.textContent = notes.join(" · ");
}

function setSignedEvidence(valueElement, barElement, value, scale) {
  const number = finiteNumber(value);
  valueElement.className = "signed-value";
  barElement.className = "signed-bar";
  if (number === null) {
    valueElement.textContent = "—";
    barElement.style.setProperty("--bar-width", "0%");
    return;
  }
  const sign = number < 0 ? "negative" : "positive";
  valueElement.classList.add(`${sign}-text`);
  valueElement.textContent = formatMagnitude(number);
  barElement.classList.add(sign);
  barElement.style.setProperty("--bar-width", `${Math.min(50, Math.abs(number) / (scale || 1) * 50)}%`);
  barElement.style.setProperty("--bar-color", sign === "negative" ? "#f04f5f" : "#175cff");
}

function identityRows(node) {
  const rows = [
    ["Kind", node.kind],
    ["Layer", node.layer],
    ["Token position", node.position],
    ["Neuron", node.neuron],
    ["Polarity", node.polarity],
  ];
  const seen = new Set(["id", "occurrence_id", "basis_id", "kind", "layer", "position", "token_position", "neuron", "neuron_index", "polarity"]);
  for (const [prefix, record] of [["Occurrence", node.occurrence], ["Basis", node.basis]]) {
    Object.entries(record).forEach(([key, value]) => {
      if (seen.has(key) || value === undefined || value === null || typeof value === "object") return;
      rows.push([`${prefix} ${key.replaceAll("_", " ")}`, value]);
    });
  }
  return rows;
}

function renderIdentity(node) {
  elements.occurrenceId.textContent = node.occurrenceId;
  elements.occurrenceId.title = JSON.stringify(node.occurrence);
  elements.basisId.textContent = node.basisId;
  elements.basisId.title = JSON.stringify(node.basis);
  elements.identityFields.replaceChildren();
  identityRows(node).forEach(([key, value]) => {
    const dt = document.createElement("dt");
    dt.textContent = key;
    const dd = document.createElement("dd");
    dd.textContent = textOrNull(value) ?? "—";
    if (key === "Polarity" && ["positive", "negative"].includes(value)) dd.classList.add(`${value}-text`);
    elements.identityFields.append(dt, dd);
  });
}

function profileKeyLabel(key, axis) {
  if (axis === "output") {
    const target = state.traceDocument?.target ?? {};
    const token = safeTokenText(target.token_text, target.token_id);
    return `target ${token} · response position ${target.response_position ?? "?"}`;
  }
  const index = Number(key);
  const context = normalizeContext(state.traceDocument);
  const token = Number.isInteger(index) ? context.tokens[index] : null;
  if (!token) return String(key);
  const clean = token.text.replaceAll("\n", "↵");
  return `${key} ${clean || "␠"}`;
}

function renderProfile(container, profile, axis) {
  const expansionKey = `${state.selectedArtifactId}::${state.selectedNodeId}::${axis}`;
  const expanded = state.expandedProfiles.has(expansionKey);
  const { rows, total } = profileDisplay(profile, { expanded });
  container.replaceChildren();
  if (rows.length === 0) {
    const empty = document.createElement("p");
    empty.className = "profile-empty";
    empty.textContent = "No profile values in this trace.";
    container.append(empty);
    return;
  }
  const max = Math.max(...rows.map((row) => Math.abs(row.value)), 1e-30);
  const fragment = document.createDocumentFragment();
  rows.forEach((row) => {
    const sign = row.value < 0 ? "negative" : "positive";
    const element = document.createElement("div");
    element.className = "profile-row";
    element.title = `${profileKeyLabel(row.key, axis)}: ${formatSigned(row.value)}`;
    element.innerHTML = `
        <span class="profile-key">${escapeHtml(profileKeyLabel(row.key, axis))}</span>
        <span class="profile-track"><span class="${sign}" style="--width:${Math.abs(row.value) / max * 50}%;--color:${sign === "negative" ? "#f04f5f" : "#175cff"}"></span></span>
        <span class="profile-value">${formatSigned(row.value)}</span>`;
    fragment.append(element);
  });
  container.append(fragment);
  if (total > 18) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "profile-toggle";
    button.setAttribute("aria-expanded", String(expanded));
    button.textContent = expanded ? "Show top 18" : `Show all ${formatCount(total)} values`;
    button.title = expanded
      ? "Collapse this profile to its 18 strongest values"
      : `Reveal all ${formatCount(total)} finite values stored in this profile`;
    button.addEventListener("click", () => {
      if (expanded) state.expandedProfiles.delete(expansionKey);
      else state.expandedProfiles.add(expansionKey);
      renderProfile(container, profile, axis);
    });
    container.append(button);
  }
}

function connectedEdges(node, direction) {
  const key = direction === "incoming" ? "target" : "source";
  return state.graph.edges
    .filter((edge) => edge[key] === node.id)
    .sort((a, b) => Math.abs(b.attribution ?? 0) - Math.abs(a.attribution ?? 0))
    .slice(0, 8);
}

function renderConnections(container, edges, direction) {
  container.replaceChildren();
  if (edges.length === 0) {
    const empty = document.createElement("p");
    empty.className = "connection-empty";
    empty.textContent = "No connections recorded.";
    container.append(empty);
    return;
  }
  const endpoint = direction === "incoming" ? "source" : "target";
  const table = document.createElement("table");
  table.className = "connection-table";
  table.innerHTML = `<thead><tr><th>${direction === "incoming" ? "Source" : "Target"}</th><th>Attr</th><th>Weight</th></tr></thead>`;
  const body = document.createElement("tbody");
  edges.forEach((edge) => {
    const row = document.createElement("tr");
    const node = state.graph.nodeById.get(edge[endpoint]);
    const label = graphNodeLabel(node) || edge[endpoint];
    row.innerHTML = `
      <td><button type="button" data-node-id="${escapeHtml(edge[endpoint])}" title="Inspect ${escapeHtml(edge[endpoint])}">${escapeHtml(label)}</button></td>
      <td class="${(edge.attribution ?? 0) < 0 ? "negative-text" : "positive-text"}">${formatMagnitude(edge.attribution)}</td>
      <td>${formatSigned(edge.weight)}</td>`;
    body.append(row);
  });
  table.append(body);
  table.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-node-id]");
    if (button) selectNode(button.dataset.nodeId, true);
  });
  container.append(table);
}

function commentKey(nodeId = state.selectedNodeId) {
  return `${state.selectedArtifactId}::${nodeId}`;
}

function updateNoteControls() {
  if (!state.selectedNodeId) return;
  elements.neuronComment.value = state.workspace.comments[commentKey()] ?? "";
  elements.pinNode.checked = state.workspace.pinnedNodeIds.includes(commentKey());
}

async function renderEvidence(node) {
  if (!node) {
    elements.evidenceEmpty.hidden = false;
    elements.evidenceContent.hidden = true;
    return;
  }
  elements.evidenceEmpty.hidden = true;
  elements.evidenceContent.hidden = false;
  renderIdentity(node);
  const activationScale = Math.max(...state.graph.nodes.map((item) => Math.abs(item.activation ?? 0)), 1e-30);
  const attributionScale = Math.max(...state.graph.nodes.map((item) => Math.abs(item.attribution ?? 0)), 1e-30);
  setSignedEvidence(elements.activationValue, elements.activationBar, node.activation, activationScale);
  setSignedEvidence(elements.attributionValue, elements.attributionBar, node.attribution, attributionScale);
  renderProfile(elements.attrProfile, node.attributionMap, "input");
  renderProfile(elements.contribProfile, node.contributionMap, "output");
  renderConnections(elements.incoming, connectedEdges(node, "incoming"), "incoming");
  renderConnections(elements.outgoing, connectedEdges(node, "outgoing"), "outgoing");
  updateNoteControls();
  await renderLabels(node);
}

function labelSetDescriptor(id) {
  return state.labelSets.find((item) => item.id === id) ?? null;
}

function overlayUrl(descriptor) {
  const encodedArtifact = encodeURIComponent(state.selectedArtifactId);
  const template = firstDefined(descriptor.trace_url_template, descriptor.url_template, descriptor.trace_url);
  if (template) {
    return String(template)
      .replaceAll("{artifact_id}", encodedArtifact)
      .replaceAll("{label_set_id}", encodeURIComponent(descriptor.id));
  }
  const traceUrls = descriptor.trace_urls ?? descriptor.traces;
  if (traceUrls && typeof traceUrls === "object" && typeof traceUrls[state.selectedArtifactId] === "string") {
    return traceUrls[state.selectedArtifactId];
  }
  return `/api/v1/label-sets/${encodeURIComponent(descriptor.id)}/traces/${encodedArtifact}`;
}

async function loadOverlay(id) {
  if (["raw", "none"].includes(id)) return null;
  const descriptor = labelSetDescriptor(id);
  if (!descriptor) return null;
  const key = `${id}::${state.selectedArtifactId}`;
  if (state.overlayCache.has(key)) return state.overlayCache.get(key);
  if (descriptor.labels || descriptor.nodes || descriptor.entries) {
    state.overlayCache.set(key, descriptor);
    return descriptor;
  }
  const promise = fetchJson(overlayUrl(descriptor)).catch((error) => {
    state.overlayCache.delete(key);
    throw error;
  });
  state.overlayCache.set(key, promise);
  const overlay = await promise;
  state.overlayCache.set(key, overlay);
  return overlay;
}

function findLabelRecord(overlay, node) {
  if (!overlay) return null;
  const source = firstDefined(overlay.labels, overlay.nodes, overlay.entries, overlay.annotations, overlay);
  if (Array.isArray(source)) {
    return (
      source.find((entry) =>
        [entry.occurrence_id, entry.node_id, entry.id, entry.basis_id]
          .filter(Boolean)
          .some((id) => [node.id, node.occurrenceId, node.basisId].includes(String(id))),
      ) ?? null
    );
  }
  if (source && typeof source === "object") {
    return source[node.id] ?? source[node.occurrenceId] ?? source[node.basisId] ?? null;
  }
  return null;
}

function renderLabelValue(container, record, descriptor, node) {
  if (descriptor?.id === "raw") {
    const sign = node.polarity === "negative" ? "-" : node.polarity === "positive" ? "+" : node.polarity;
    const model = firstDefined(node.basis?.model_id, state.traceDocument?.model?.model_id, "unknown model");
    const revision = firstDefined(node.basis?.model_revision, state.traceDocument?.model?.model_revision);
    const identity = `${graphNodeLabel(node)}:${sign}`;
    container.innerHTML = `<span class="mono">${escapeHtml(identity)}</span><span class="label-confidence" title="${escapeHtml(node.basisId)}">Exact signed basis · ${escapeHtml(model)}${revision ? ` @ ${escapeHtml(shortHash(revision))}` : ""}</span>`;
    return;
  }
  if (!record) {
    container.innerHTML = '<span class="label-status">Unknown — no coverage</span>';
    return;
  }
  if (typeof record === "string") {
    container.textContent = record;
    return;
  }
  const status = String(firstDefined(record.status, record.state, "")).toLowerCase();
  if (record.abstained || status === "abstained" || status === "abstain") {
    container.innerHTML = '<span class="label-status">Abstained</span>';
    return;
  }
  const value = firstDefined(record.description, record.label, record.text, record.title, record.name);
  if (value === undefined || value === null) {
    container.innerHTML = '<span class="label-status">Unknown — no description</span>';
    return;
  }
  const confidence = finiteNumber(firstDefined(record.confidence, record.score));
  container.innerHTML = `${escapeHtml(value)}${confidence !== null ? `<span class="label-confidence">confidence ${formatNumber(confidence)}</span>` : ""}`;
}

async function renderLabels(node) {
  const requestArtifact = state.selectedArtifactId;
  const slots = [
    { id: state.labels.a, heading: elements.labelAName, value: elements.labelAValue },
    { id: state.labels.b, heading: elements.labelBName, value: elements.labelBValue },
  ];
  slots.forEach((slot) => {
    const descriptor = slot.id === "raw" ? { id: "raw", name: "Raw identity" } : labelSetDescriptor(slot.id);
    slot.heading.textContent = slot.id === "none" ? "—" : descriptor?.name ?? slot.id;
    slot.value.innerHTML = slot.id === "none" ? "—" : '<span class="label-status">Loading…</span>';
  });
  const descriptors = slots.map((slot) => labelSetDescriptor(slot.id)).filter(Boolean);
  elements.syntheticWarning.hidden = !descriptors.some((descriptor) => descriptor.synthetic);
  await Promise.all(
    slots.map(async (slot) => {
      if (slot.id === "none") return;
      const descriptor = slot.id === "raw" ? { id: "raw", name: "Raw identity" } : labelSetDescriptor(slot.id);
      try {
        const overlay = await loadOverlay(slot.id);
        if (requestArtifact !== state.selectedArtifactId || node.id !== state.selectedNodeId) return;
        const record = findLabelRecord(overlay, node);
        renderLabelValue(slot.value, record, descriptor, node);
        if (descriptor?.synthetic || record?.synthetic) elements.syntheticWarning.hidden = false;
      } catch (error) {
        slot.value.innerHTML = `<span class="label-status">Unavailable: ${escapeHtml(error.message)}</span>`;
      }
    }),
  );
}

function selectNode(nodeId, focus = false, dirty = true) {
  const node = nodeId ? state.graph?.nodeById.get(nodeId) : null;
  state.selectedNodeId = node?.id ?? null;
  if (focus && node && !state.projection?.nodes.some((item) => item.id === node.id)) {
    renderGraph({ fit: false });
  }
  if (focus && node) graphView.focus(node.id);
  else graphView.select(state.selectedNodeId);
  renderEvidence(node);
  updateUrl();
  if (dirty) markDirty();
  if (node && window.innerWidth < 760) setDrawer("evidence", true);
}

function exactNeuronSearch() {
  const query = elements.neuronSearch.value.trim().toLowerCase();
  if (!query) return;
  const matches = state.graph.nodes.filter((node) => exactNeuronAliases(node).has(query));
  if (matches.length === 0) {
    showToast(`No exact neuron identity matches “${elements.neuronSearch.value.trim()}”.`, "warning");
    return;
  }
  if (matches.length > 1) {
    showToast(`${matches.length} occurrences match. Add polarity (+/−) or use an occurrence/basis ID.`, "warning");
    return;
  }
  if (!nodePassesFilters(matches[0])) {
    state.filters.layer = "all";
    state.filters.position = "all";
    state.filters.polarity = "all";
    state.filters.magnitude = "0";
    syncFilterControls();
    renderGraph({ fit: false });
  }
  selectNode(matches[0].id, true);
}

function renderTopBar() {
  const catalog = selectedCatalogTrace();
  const model = state.traceDocument?.model ?? {};
  const modelId = firstDefined(model.id, model.model_id, catalog?.model_id, state.site?.model_id, "Unknown model");
  const revision = firstDefined(model.revision, catalog?.model_revision);
  elements.modelValue.textContent = revision ? `${modelId} @ ${shortHash(revision)}` : String(modelId);
  elements.modelValue.title = revision ? `${modelId}\nrevision ${revision}` : String(modelId);
  if (catalog) {
    elements.targetSelect.value = catalog.artifact_id;
    document.title = `position ${catalog.response_position} · Trace Observatory`;
  }
}

function readWorkspacePayload(payload) {
  const workspace = payload?.workspace ?? payload?.state ?? payload ?? {};
  state.workspace.revision = finiteNumber(firstDefined(payload?.revision, workspace.revision), 0);
  state.workspace.readOnly = Boolean(firstDefined(payload?.read_only, workspace.read_only, false));
  state.workspace.comments = workspace.comments && typeof workspace.comments === "object" ? { ...workspace.comments } : {};
  state.workspace.pinnedNodeIds = Array.isArray(firstDefined(workspace.pinned_node_ids, workspace.pinnedNodeIds))
    ? [...firstDefined(workspace.pinned_node_ids, workspace.pinnedNodeIds)]
    : [];
  const savedFilters = workspace.filters;
  if (savedFilters && typeof savedFilters === "object") {
    state.filters = { ...DEFAULT_FILTERS, ...savedFilters };
  }
  state.labels.a = firstDefined(workspace.label_set_ids?.[0], workspace.labels?.a, state.labels.a);
  state.labels.b = firstDefined(workspace.label_set_ids?.[1], workspace.labels?.b, state.labels.b);
  if (!new URL(location.href).searchParams.has("trace")) {
    state.selectedArtifactId = firstDefined(workspace.selected_artifact_id, workspace.artifact_id, state.selectedArtifactId);
  }
  if (!new URL(location.href).searchParams.has("node")) {
    state.selectedNodeId = firstDefined(workspace.selected_node_id, state.selectedNodeId);
  }
}

function workspaceBody() {
  return {
    expected_revision: state.workspace.revision,
    workspace: {
      schema_version: WORKSPACE_SCHEMA,
      selected_artifact_id: state.selectedArtifactId,
      selected_node_id: state.selectedNodeId,
      filters: { ...state.filters },
      label_set_ids: [state.labels.a, state.labels.b].filter((id) => id && id !== "none"),
      labels: { ...state.labels },
      comments: { ...state.workspace.comments },
      pinned_node_ids: [...state.workspace.pinnedNodeIds],
      saved_at: new Date().toISOString(),
    },
  };
}

async function saveWorkspace() {
  if (state.workspace.readOnly) return;
  window.clearTimeout(noteTimer);
  if (state.selectedNodeId) {
    const key = commentKey();
    const value = elements.neuronComment.value;
    if (value) state.workspace.comments[key] = value;
    else delete state.workspace.comments[key];
  }
  elements.saveButton.disabled = true;
  elements.saveState.textContent = "Saving…";
  try {
    const payload = await fetchJson("/api/v1/workspaces/default", {
      method: "PUT",
      headers: {
        "Content-Type": "application/json",
        "If-Match": String(state.workspace.revision),
      },
      body: JSON.stringify(workspaceBody()),
    });
    state.workspace.revision = finiteNumber(firstDefined(payload?.revision, payload?.workspace?.revision), state.workspace.revision + 1);
    state.dirty = false;
    elements.saveState.textContent = `Saved · revision ${state.workspace.revision}`;
    showToast("Workspace saved.");
  } catch (error) {
    elements.saveButton.disabled = false;
    if (error instanceof HttpError && error.status === 409) {
      elements.saveState.textContent = "Save conflict";
      showToast("Workspace changed in another tab. Reload before saving again.", "error");
    } else {
      elements.saveState.textContent = "Save failed";
      showToast(`Could not save workspace: ${error.message}`, "error");
    }
  }
}

function applyUrlState() {
  const params = new URL(location.href).searchParams;
  state.selectedArtifactId = params.get("trace") ?? state.selectedArtifactId;
  state.selectedNodeId = params.get("node") ?? state.selectedNodeId;
  const mappings = {
    layer: "layer",
    position: "position",
    inputs: "inputTokens",
    polarity: "polarity",
    magnitude: "magnitude",
    mass: "mass",
    edges: "edgeBudget",
  };
  Object.entries(mappings).forEach(([parameter, key]) => {
    if (params.has(parameter)) state.filters[key] = params.get(parameter);
  });
  state.labels.a = params.get("labelA") ?? state.labels.a;
  state.labels.b = params.get("labelB") ?? state.labels.b;
}

function updateUrl() {
  const url = new URL(location.href);
  const setOrDelete = (name, value, defaultValue = null) => {
    if (value === null || value === undefined || value === "" || value === defaultValue) url.searchParams.delete(name);
    else url.searchParams.set(name, value);
  };
  setOrDelete("trace", state.selectedArtifactId);
  setOrDelete("node", state.selectedNodeId);
  setOrDelete("layer", state.filters.layer, DEFAULT_FILTERS.layer);
  setOrDelete("position", state.filters.position, DEFAULT_FILTERS.position);
  setOrDelete("inputs", state.filters.inputTokens, DEFAULT_FILTERS.inputTokens);
  setOrDelete("polarity", state.filters.polarity, DEFAULT_FILTERS.polarity);
  setOrDelete("magnitude", state.filters.magnitude, DEFAULT_FILTERS.magnitude);
  setOrDelete("mass", state.filters.mass, DEFAULT_FILTERS.mass);
  setOrDelete("edges", state.filters.edgeBudget, DEFAULT_FILTERS.edgeBudget);
  setOrDelete("labelA", state.labels.a, "raw");
  setOrDelete("labelB", state.labels.b, "none");
  history.replaceState(null, "", url);
}

async function selectArtifact(artifactId) {
  if (!artifactId || artifactId === state.selectedArtifactId && state.traceDocument) {
    closeDrawers();
    return;
  }
  state.selectedArtifactId = artifactId;
  state.selectedNodeId = null;
  updateCatalogSelection();
  updateUrl();
  closeDrawers();
  markDirty();
  await loadTrace(artifactId);
}

async function loadTrace(artifactId) {
  state.loadController?.abort();
  const controller = new AbortController();
  state.loadController = controller;
  setGraphStatus("loading", "Loading trace graph…");
  elements.evidenceEmpty.hidden = false;
  elements.evidenceContent.hidden = true;
  try {
    const payload = await fetchJson(`/api/v1/traces/${encodeURIComponent(artifactId)}`, { signal: controller.signal });
    if (controller.signal.aborted) return;
    state.traceDocument = payload;
    state.contextExpanded = false;
    state.graph = normalizeGraph(payload);
    if (payload.schema_version && payload.schema_version !== TRACE_SCHEMA) {
      showToast(`Trace schema ${payload.schema_version} is not ${TRACE_SCHEMA}; using compatible fields.`, "warning");
    }
    renderTopBar();
    renderContext();
    populateGraphFilters();
    renderGraph();
    const requestedNode = state.selectedNodeId;
    if (requestedNode && state.graph.nodeById.has(requestedNode)) selectNode(requestedNode, false, false);
    else selectNode(null, false, false);
    await refreshSelectedLabels();
  } catch (error) {
    if (error.name === "AbortError") return;
    state.traceDocument = null;
    state.graph = null;
    setGraphStatus("error", `This trace could not be loaded. ${error.message}`, true);
    showToast(`Trace load failed: ${error.message}`, "error");
  }
}

async function refreshSelectedLabels() {
  const node = selectedNode();
  if (node) await renderLabels(node);
}

function resetFilters() {
  state.filters = { ...DEFAULT_FILTERS };
  elements.neuronSearch.value = "";
  syncFilterControls();
  renderGraph();
  updateUrl();
  markDirty();
}

function setDrawer(which, open) {
  const rail = which === "trajectory" ? elements.trajectoryRail : elements.evidenceRail;
  const other = which === "trajectory" ? elements.evidenceRail : elements.trajectoryRail;
  const toggle = which === "trajectory" ? elements.trajectoryToggle : elements.evidenceToggle;
  const otherToggle = which === "trajectory" ? elements.evidenceToggle : elements.trajectoryToggle;
  other.classList.remove("open");
  otherToggle.setAttribute("aria-expanded", "false");
  rail.classList.toggle("open", open);
  toggle.setAttribute("aria-expanded", String(open));
  elements.drawerScrim.hidden = !open;
}

function closeDrawers() {
  elements.trajectoryRail.classList.remove("open");
  elements.evidenceRail.classList.remove("open");
  elements.trajectoryToggle.setAttribute("aria-expanded", "false");
  elements.evidenceToggle.setAttribute("aria-expanded", "false");
  elements.drawerScrim.hidden = true;
}

function bindEvents() {
  elements.targetSelect.addEventListener("change", () => selectArtifact(elements.targetSelect.value));
  const filterEntries = [
    [elements.layerFilter, "layer"],
    [elements.positionFilter, "position"],
    [elements.inputTokenFilter, "inputTokens"],
    [elements.polarityFilter, "polarity"],
    [elements.magnitudeFilter, "magnitude"],
    [elements.massFilter, "mass"],
    [elements.edgeBudgetFilter, "edgeBudget"],
  ];
  filterEntries.forEach(([element, key]) => {
    element.addEventListener("change", () => {
      if (key === "edgeBudget" && element.value === "all") {
        const confirmed = window.confirm(
          `Show all ${formatCount(state.graph?.edges.length)} raw edges? Layout may be slow. This changes display only; source trace values remain unchanged.`,
        );
        if (!confirmed) {
          element.value = state.filters.edgeBudget;
          return;
        }
      }
      state.filters[key] = element.value;
      renderGraph();
      updateUrl();
      markDirty();
    });
  });
  elements.clearFilters.addEventListener("click", resetFilters);
  elements.contextToggle.addEventListener("click", () => {
    state.contextExpanded = !state.contextExpanded;
    renderContext();
    requestAnimationFrame(() => graphView.fit());
  });
  elements.neuronSearch.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      exactNeuronSearch();
    }
  });
  elements.neuronSearch.addEventListener("search", () => {
    if (elements.neuronSearch.value) exactNeuronSearch();
  });
  elements.fitGraph.addEventListener("click", () => graphView.fit());
  elements.zoomIn.addEventListener("click", () => graphView.zoomBy(1.3));
  elements.zoomOut.addEventListener("click", () => graphView.zoomBy(1 / 1.3));
  elements.resetGraph.addEventListener("click", () => graphView.reset());
  elements.saveButton.addEventListener("click", saveWorkspace);

  const updateLabels = async (slot, value) => {
    state.labels[slot] = value;
    if (slot === "a") {
      elements.labelPrimary.value = value;
      elements.labelA.value = value;
    }
    updateUrl();
    markDirty();
    await refreshSelectedLabels();
  };
  elements.labelPrimary.addEventListener("change", () => updateLabels("a", elements.labelPrimary.value));
  elements.labelA.addEventListener("change", () => updateLabels("a", elements.labelA.value));
  elements.labelB.addEventListener("change", () => updateLabels("b", elements.labelB.value));

  elements.neuronComment.addEventListener("input", () => {
    window.clearTimeout(noteTimer);
    noteTimer = window.setTimeout(() => {
      if (!state.selectedNodeId) return;
      const key = commentKey();
      const value = elements.neuronComment.value;
      if (value) state.workspace.comments[key] = value;
      else delete state.workspace.comments[key];
      markDirty();
    }, 180);
  });
  elements.pinNode.addEventListener("change", () => {
    const key = commentKey();
    const pins = new Set(state.workspace.pinnedNodeIds);
    if (elements.pinNode.checked) pins.add(key);
    else pins.delete(key);
    state.workspace.pinnedNodeIds = [...pins];
    markDirty();
  });

  elements.provenanceToggle.addEventListener("click", () => {
    const collapsed = elements.app.classList.toggle("provenance-collapsed");
    elements.provenanceToggle.setAttribute("aria-expanded", String(!collapsed));
    markDirty();
    requestAnimationFrame(() => graphView.fit());
  });
  elements.trajectoryToggle.addEventListener("click", () => setDrawer("trajectory", !elements.trajectoryRail.classList.contains("open")));
  elements.evidenceToggle.addEventListener("click", () => setDrawer("evidence", !elements.evidenceRail.classList.contains("open")));
  elements.trajectoryClose.addEventListener("click", closeDrawers);
  elements.evidenceClose.addEventListener("click", closeDrawers);
  elements.drawerScrim.addEventListener("click", closeDrawers);
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeDrawers();
  });
  window.addEventListener("beforeunload", (event) => {
    if (!state.dirty) return;
    event.preventDefault();
  });
}

async function loadOptionalWorkspace() {
  try {
    const payload = await fetchJson("/api/v1/workspaces/default");
    readWorkspacePayload(payload);
    elements.saveState.textContent = `Workspace revision ${state.workspace.revision}`;
  } catch (error) {
    if (error instanceof HttpError && error.status === 404) {
      elements.saveState.textContent = "New workspace";
      return;
    }
    state.workspace.readOnly = true;
    elements.saveState.textContent = "Read-only workspace";
    showToast(`Workspace storage is unavailable: ${error.message}`, "warning");
  }
}

async function initialize() {
  try {
    graphView = new GraphView(elements.graphSvg, {
      onSelect: (node) => selectNode(node?.id ?? null, false),
    });
    bindEvents();
    const [catalogPayload, labelPayload] = await Promise.all([
      fetchJson("/api/v1/catalog"),
      fetchJson("/api/v1/label-sets").catch((error) => {
        showToast(`Label sets are unavailable; raw identity remains usable. ${error.message}`, "warning");
        return { label_sets: [] };
      }),
      loadOptionalWorkspace(),
    ]);
    state.site = catalogPayload?.site ?? catalogPayload?.model ?? null;
    state.catalog = normalizeCatalog(catalogPayload);
    state.labelSets = normalizeLabelSets(labelPayload);
    applyUrlState();
    if (!state.catalog.some((item) => item.artifact_id === state.selectedArtifactId)) {
      state.selectedArtifactId = state.catalog[0]?.artifact_id ?? null;
    }
    populateCatalog();
    populateLabelSets();
    syncFilterControls();
    elements.saveButton.disabled = state.workspace.readOnly;
    elements.app.setAttribute("aria-busy", "false");
    if (!state.selectedArtifactId) {
      setGraphStatus("empty", "No trace artifacts are available. Run observatory sync, then reload this page.");
      return;
    }
    updateUrl();
    await loadTrace(state.selectedArtifactId);
  } catch (error) {
    elements.app.setAttribute("aria-busy", "false");
    setGraphStatus("error", `Trace Observatory could not start. ${error.message}`, false);
    showToast(error.message, "error");
  }
}

initialize();
