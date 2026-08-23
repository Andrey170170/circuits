const COLORS = {
  positive: "#175cff",
  negative: "#f04f5f",
};

function numericMagnitude(value) {
  const number = Number(value);
  return Number.isFinite(number) ? Math.abs(number) : 0;
}

function signClass(value) {
  return Number(value) < 0 ? "negative" : "positive";
}

function safeText(value, fallback = "unknown") {
  return value === undefined || value === null || value === "" ? fallback : String(value);
}

function nodeLabel(node) {
  if (node.kind === "logit" || node.kind === "target_logit" || node.kind === "output") {
    return `OUT:${safeText(node.token_text ?? node.neuron ?? node.id, "?")}`;
  }
  if (node.kind === "embedding" || node.kind === "input" || node.kind === "input_token") {
    return `IN:${safeText(node.position, "?")}`;
  }
  const layer = safeText(node.layer, "?");
  const neuron = safeText(node.neuron ?? node.feature ?? node.basis?.neuron, "?");
  return `L${layer}:N${neuron}`;
}

function nodeAriaLabel(node) {
  const sign = Number(node.attribution) < 0 ? "negative" : "positive";
  return `${nodeLabel(node)}, token position ${safeText(node.position)}, ${sign} attribution ${safeText(node.attribution)}`;
}

export class GraphView {
  constructor(svgElement, { onSelect, onHover } = {}) {
    if (!globalThis.d3 || !globalThis.dagre) {
      throw new Error("Graph libraries did not load. Expected d3 and dagre globals.");
    }

    this.svgElement = svgElement;
    this.svg = d3.select(svgElement);
    this.onSelect = onSelect ?? (() => {});
    this.onHover = onHover ?? (() => {});
    this.nodes = [];
    this.edges = [];
    this.nodeById = new Map();
    this.incoming = new Map();
    this.outgoing = new Map();
    this.selectedNodeId = null;
    this.hoveredNodeId = null;
    this.viewport = null;
    this.contentBounds = null;

    this.zoom = d3
      .zoom()
      .scaleExtent([0.08, 6])
      .on("zoom", (event) => {
        this.viewport?.attr("transform", event.transform);
      });
    this.svg.call(this.zoom).on("dblclick.zoom", null);
  }

  render(nodes, edges, { selectedNodeId = null, fit = true } = {}) {
    this.nodes = nodes;
    this.edges = edges;
    this.nodeById = new Map(nodes.map((node) => [node.id, node]));
    this.selectedNodeId = this.nodeById.has(selectedNodeId) ? selectedNodeId : null;
    this.#buildAdjacency();
    this.svg.selectAll("*").remove();

    if (nodes.length === 0) {
      this.viewport = null;
      return;
    }

    this.#createMarkers();
    this.viewport = this.svg.append("g").attr("class", "graph-viewport");

    const layout = new dagre.graphlib.Graph({ multigraph: true, compound: false });
    layout.setGraph({
      rankdir: "BT",
      marginx: 78,
      marginy: 36,
      nodesep: nodes.length > 400 ? 9 : 18,
      edgesep: 5,
      ranksep: nodes.length > 400 ? 45 : 66,
      acyclicer: "greedy",
      ranker: "tight-tree",
    });
    layout.setDefaultEdgeLabel(() => ({}));

    nodes.forEach((node) => {
      const label = nodeLabel(node);
      layout.setNode(node.id, {
        width: Math.max(72, Math.min(130, 18 + label.length * 6.5)),
        height: 29,
        node,
        label,
      });
    });

    edges.forEach((edge, index) => {
      if (!this.nodeById.has(edge.source) || !this.nodeById.has(edge.target)) return;
      layout.setEdge(edge.source, edge.target, { edge }, edge.id ?? `edge-${index}`);
    });

    dagre.layout(layout);
    const graphInfo = layout.graph();
    const graphWidth = Math.max(400, graphInfo.width || 400);
    const graphHeight = Math.max(260, graphInfo.height || 260);
    this.contentBounds = { x: 0, y: 0, width: graphWidth, height: graphHeight };

    this.#renderLayerBands(layout, graphWidth, graphHeight);
    this.#renderEdges(layout);
    this.#renderNodes(layout);
    this.updateHighlight();

    if (fit) {
      requestAnimationFrame(() => this.fit());
    }
  }

  #createMarkers() {
    const defs = this.svg.append("defs");
    ["positive", "negative"].forEach((sign) => {
      defs
        .append("marker")
        .attr("id", `arrow-${sign}`)
        .attr("viewBox", "0 -5 10 10")
        .attr("refX", 9)
        .attr("refY", 0)
        .attr("markerWidth", 6)
        .attr("markerHeight", 6)
        .attr("orient", "auto")
        .attr("markerUnits", "strokeWidth")
        .append("path")
        .attr("d", "M0,-4L9,0L0,4")
        .attr("fill", COLORS[sign]);
    });
  }

  #renderLayerBands(layout, graphWidth, graphHeight) {
    const layerGroups = d3.group(
      this.nodes.map((node) => ({ node, layout: layout.node(node.id) })),
      ({ node }) => safeText(node.layer, node.kind ?? "other"),
    );
    const bands = [...layerGroups.entries()]
      .map(([layer, entries]) => ({
        layer,
        center: d3.mean(entries, ({ layout: item }) => item?.y ?? 0),
      }))
      .sort((a, b) => a.center - b.center);

    bands.forEach((band, index) => {
      const previous = bands[index - 1];
      const next = bands[index + 1];
      const top = previous ? (previous.center + band.center) / 2 : 0;
      const bottom = next ? (band.center + next.center) / 2 : graphHeight;
      band.top = Math.max(0, top);
      band.bottom = Math.min(graphHeight, bottom);
    });

    const group = this.viewport.append("g").attr("class", "layer-bands");
    group
      .selectAll("rect")
      .data(bands)
      .join("rect")
      .attr("class", "layer-band")
      .attr("x", 36)
      .attr("y", (band) => band.top)
      .attr("width", Math.max(0, graphWidth - 36))
      .attr("height", (band) => Math.max(1, band.bottom - band.top));

    group
      .selectAll("line")
      .data(bands)
      .join("line")
      .attr("class", "layer-rule")
      .attr("x1", 36)
      .attr("x2", graphWidth)
      .attr("y1", (band) => band.center)
      .attr("y2", (band) => band.center);

    group
      .selectAll("text")
      .data(bands)
      .join("text")
      .attr("class", "layer-label")
      .attr("x", 4)
      .attr("y", (band) => band.center + 4)
      .text((band) => this.#formatLayer(band.layer));
  }

  #formatLayer(layer) {
    const lowered = String(layer).toLowerCase();
    if (lowered === "-1" || lowered === "e" || lowered.includes("embed") || lowered.includes("input")) return "in";
    if (lowered.includes("logit") || lowered.includes("output") || lowered === "out") return "out";
    return `L${layer}`;
  }

  #renderEdges(layout) {
    const maxMagnitude = d3.max(this.edges, (edge) => numericMagnitude(edge.attribution)) || 1;
    const widthScale = d3.scaleSqrt().domain([0, maxMagnitude]).range([0.55, 4.8]);
    const line = d3.line().x((point) => point.x).y((point) => point.y).curve(d3.curveBasis);
    const edgeRows = [];
    layout.edges().forEach((descriptor) => {
      const layoutEdge = layout.edge(descriptor);
      if (!layoutEdge?.edge) return;
      edgeRows.push({
        ...layoutEdge.edge,
        descriptor,
        points: layoutEdge.points ?? [],
      });
    });

    this.edgeSelection = this.viewport
      .append("g")
      .attr("class", "graph-edges")
      .selectAll("path")
      .data(edgeRows, (edge) => edge.id)
      .join("path")
      .attr("class", (edge) => `graph-edge ${signClass(edge.attribution)}`)
      .attr("data-edge-id", (edge) => edge.id)
      .attr("data-source", (edge) => edge.source)
      .attr("data-target", (edge) => edge.target)
      .attr("d", (edge) => line(edge.points))
      .attr("stroke-width", (edge) => widthScale(numericMagnitude(edge.attribution)))
      .attr("marker-end", (edge) => `url(#arrow-${signClass(edge.attribution)})`);

    this.edgeSelection
      .append("title")
      .text(
        (edge) =>
          `${edge.source} → ${edge.target}\nattribution ${safeText(edge.attribution)}\nweight ${safeText(edge.weight)}`,
      );
  }

  #renderNodes(layout) {
    const rows = this.nodes.map((node) => ({ ...node, layout: layout.node(node.id) }));
    this.nodeSelection = this.viewport
      .append("g")
      .attr("class", "graph-nodes")
      .selectAll("g")
      .data(rows, (node) => node.id)
      .join("g")
      .attr("class", (node) => `graph-node ${signClass(node.attribution)}`)
      .attr("data-node-id", (node) => node.id)
      .attr("tabindex", 0)
      .attr("role", "button")
      .attr("aria-label", nodeAriaLabel)
      .attr("transform", (node) => `translate(${node.layout.x},${node.layout.y})`)
      .on("mouseenter", (_event, node) => {
        this.hoveredNodeId = node.id;
        this.updateHighlight();
        this.onHover(node);
      })
      .on("mouseleave", () => {
        this.hoveredNodeId = null;
        this.updateHighlight();
        this.onHover(null);
      })
      .on("focus", (_event, node) => {
        this.hoveredNodeId = node.id;
        this.updateHighlight();
      })
      .on("blur", () => {
        this.hoveredNodeId = null;
        this.updateHighlight();
      })
      .on("click", (event, node) => {
        event.stopPropagation();
        this.select(node.id, true);
      })
      .on("keydown", (event, node) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          this.select(node.id, true);
        }
      });

    this.nodeSelection
      .append("rect")
      .attr("x", (node) => -node.layout.width / 2)
      .attr("y", (node) => -node.layout.height / 2)
      .attr("width", (node) => node.layout.width)
      .attr("height", (node) => node.layout.height);

    this.nodeSelection
      .append("text")
      .attr("y", 3.5)
      .text((node) => node.layout.label);

    this.nodeSelection
      .append("text")
      .attr("class", "node-sign")
      .attr("x", (node) => node.layout.width / 2 - 8)
      .attr("y", -6)
      .text((node) => (Number(node.attribution) < 0 ? "−" : "+"));

    this.nodeSelection
      .append("title")
      .text(
        (node) =>
          `${nodeAriaLabel(node)}\nactivation ${safeText(node.activation)}\nclick to inspect connected paths`,
      );

    this.svg.on("click", () => this.select(null, true));
  }

  #buildAdjacency() {
    this.incoming = new Map(this.nodes.map((node) => [node.id, []]));
    this.outgoing = new Map(this.nodes.map((node) => [node.id, []]));
    this.edges.forEach((edge) => {
      if (this.outgoing.has(edge.source)) this.outgoing.get(edge.source).push(edge);
      if (this.incoming.has(edge.target)) this.incoming.get(edge.target).push(edge);
    });
  }

  #connectedPaths(nodeId) {
    if (!nodeId || !this.nodeById.has(nodeId)) return { nodes: new Set(), edges: new Set() };
    const connectedNodes = new Set([nodeId]);
    const connectedEdges = new Set();

    const walk = (seed, adjacency, nextKey) => {
      const queue = [seed];
      while (queue.length) {
        const current = queue.shift();
        for (const edge of adjacency.get(current) ?? []) {
          connectedEdges.add(edge.id);
          const next = edge[nextKey];
          if (!connectedNodes.has(next)) {
            connectedNodes.add(next);
            queue.push(next);
          }
        }
      }
    };

    walk(nodeId, this.incoming, "source");
    walk(nodeId, this.outgoing, "target");
    return { nodes: connectedNodes, edges: connectedEdges };
  }

  updateHighlight() {
    if (!this.nodeSelection || !this.edgeSelection) return;
    const focusId = this.hoveredNodeId ?? this.selectedNodeId;
    const connected = this.#connectedPaths(focusId);
    const hasFocus = Boolean(focusId);

    this.nodeSelection
      .classed("selected", (node) => node.id === this.selectedNodeId)
      .classed("hovered", (node) => node.id === this.hoveredNodeId)
      .classed("connected", (node) => hasFocus && connected.nodes.has(node.id))
      .classed("muted", (node) => hasFocus && !connected.nodes.has(node.id));

    this.edgeSelection
      .classed("connected", (edge) => hasFocus && connected.edges.has(edge.id))
      .classed("muted", (edge) => hasFocus && !connected.edges.has(edge.id));
  }

  select(nodeId, notify = false) {
    this.selectedNodeId = nodeId && this.nodeById.has(nodeId) ? nodeId : null;
    this.updateHighlight();
    if (notify) this.onSelect(this.selectedNodeId ? this.nodeById.get(this.selectedNodeId) : null);
  }

  focus(nodeId) {
    const row = this.nodeSelection?.filter((node) => node.id === nodeId);
    if (!row || row.empty()) return false;
    this.select(nodeId, true);
    const node = this.nodeById.get(nodeId);
    const layoutDatum = row.datum()?.layout;
    if (layoutDatum) {
      const bounds = this.svgElement.getBoundingClientRect();
      const transform = d3.zoomIdentity
        .translate(bounds.width / 2, bounds.height / 2)
        .scale(1.45)
        .translate(-layoutDatum.x, -layoutDatum.y);
      this.svg.transition().duration(this.#motionDuration()).call(this.zoom.transform, transform);
    }
    row.node()?.focus({ preventScroll: true });
    return true;
  }

  fit() {
    if (!this.contentBounds || !this.viewport) return;
    const bounds = this.svgElement.getBoundingClientRect();
    if (!bounds.width || !bounds.height) return;
    const padding = 34;
    const scale = Math.max(
      0.08,
      Math.min(
        1.2,
        (bounds.width - padding * 2) / this.contentBounds.width,
        (bounds.height - padding * 2) / this.contentBounds.height,
      ),
    );
    const x = (bounds.width - this.contentBounds.width * scale) / 2;
    const y = (bounds.height - this.contentBounds.height * scale) / 2;
    this.svg
      .transition()
      .duration(this.#motionDuration())
      .call(this.zoom.transform, d3.zoomIdentity.translate(x, y).scale(scale));
  }

  zoomBy(factor) {
    this.svg.transition().duration(this.#motionDuration()).call(this.zoom.scaleBy, factor);
  }

  reset() {
    this.select(null, true);
    this.fit();
  }

  #motionDuration() {
    return globalThis.matchMedia?.("(prefers-reduced-motion: reduce)").matches ? 0 : 160;
  }
}
