const OUTPUT_KIND = /logit|output|target/i;
const INPUT_KIND = /^(embedding|input|input_token)$/i;

const DIMENSIONS = Object.freeze({
  marginLeft: 58,
  marginRight: 34,
  marginTop: 22,
  marginBottom: 22,
  minimumCoreWidth: 880,
  layerHeight: 48,
  nodeHeight: 29,
  nodeGap: 18,
  inputNodeWidth: 54,
  inputNodeHeight: 21,
  inputColumnGap: 7,
  inputRowGap: 7,
  inputHeaderHeight: 34,
  inputBottomPadding: 14,
  inputFitPreviewRows: 2,
});

function safeText(value, fallback = "unknown") {
  return value === undefined || value === null || value === "" ? fallback : String(value);
}

function nodeLabel(node) {
  if (OUTPUT_KIND.test(String(node.kind ?? ""))) {
    return `OUT:${safeText(node.token_text ?? node.neuron ?? node.id, "?")}`;
  }
  if (INPUT_KIND.test(String(node.kind ?? ""))) {
    return `IN:${safeText(node.position, "?")}`;
  }
  return `L${safeText(node.layer, "?")}:N${safeText(node.neuron ?? node.feature ?? node.basis?.neuron, "?")}`;
}

function classifyNode(node) {
  const kind = String(node.kind ?? "");
  if (OUTPUT_KIND.test(kind)) return { key: "output", label: "out", role: "output", order: Infinity };
  if (INPUT_KIND.test(kind)) return { key: "input", label: "input tokens", role: "input", order: -Infinity };
  const layer = Number(node.layer);
  if (Number.isFinite(layer) && layer >= 0) {
    return { key: `layer:${layer}`, label: `L${layer}`, role: "layer", order: layer };
  }
  const label = safeText(node.layer, "other");
  return { key: `other:${label}`, label, role: "other", order: -1 };
}

function compareBands(left, right) {
  if (left.role === "output") return right.role === "output" ? 0 : -1;
  if (right.role === "output") return 1;
  if (left.role === "input") return right.role === "input" ? 0 : 1;
  if (right.role === "input") return -1;
  if (left.role === "layer" && right.role === "layer") return right.order - left.order;
  if (left.role === "layer") return -1;
  if (right.role === "layer") return 1;
  return left.label.localeCompare(right.label);
}

function stableNodeCompare(left, right) {
  const leftPosition = Number(left.position);
  const rightPosition = Number(right.position);
  if (Number.isFinite(leftPosition) && Number.isFinite(rightPosition) && leftPosition !== rightPosition) {
    return leftPosition - rightPosition;
  }
  const leftNeuron = Number(left.neuron ?? left.feature ?? left.basis?.neuron);
  const rightNeuron = Number(right.neuron ?? right.feature ?? right.basis?.neuron);
  if (Number.isFinite(leftNeuron) && Number.isFinite(rightNeuron) && leftNeuron !== rightNeuron) {
    return leftNeuron - rightNeuron;
  }
  return String(left.id).localeCompare(String(right.id));
}

function nodeWidth(node, role) {
  if (role === "input") return DIMENSIONS.inputNodeWidth;
  return Math.max(72, Math.min(130, 18 + nodeLabel(node).length * 6.5));
}

function orderWithinBands(bands, edges) {
  const bandByNode = new Map();
  bands.forEach((band) => band.nodes.forEach((node) => bandByNode.set(node.id, band)));
  const neighbors = new Map();
  const addNeighbor = (nodeId, neighborId) => {
    if (!bandByNode.has(nodeId) || !bandByNode.has(neighborId)) return;
    if (!neighbors.has(nodeId)) neighbors.set(nodeId, []);
    neighbors.get(nodeId).push(neighborId);
  };
  edges.forEach((edge) => {
    addNeighbor(edge.source, edge.target);
    addNeighbor(edge.target, edge.source);
  });

  const positions = new Map();
  const refreshPositions = () => {
    bands.forEach((band) => {
      const denominator = Math.max(1, band.nodes.length - 1);
      band.nodes.forEach((node, index) => positions.set(node.id, index / denominator));
    });
  };
  const sweep = (orderedBands) => {
    orderedBands.forEach((band) => {
      if (band.role === "input") return;
      band.nodes = band.nodes
        .map((node, originalIndex) => {
          const connected = neighbors.get(node.id) ?? [];
          const values = connected.map((id) => positions.get(id)).filter(Number.isFinite);
          return {
            node,
            originalIndex,
            barycenter: values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null,
          };
        })
        .sort((left, right) => {
          if (left.barycenter === null && right.barycenter === null) return left.originalIndex - right.originalIndex;
          if (left.barycenter === null) return 1;
          if (right.barycenter === null) return -1;
          return left.barycenter - right.barycenter || left.originalIndex - right.originalIndex;
        })
        .map((entry) => entry.node);
      refreshPositions();
    });
  };

  refreshPositions();
  for (let iteration = 0; iteration < 3; iteration += 1) {
    sweep(bands);
    sweep([...bands].reverse());
  }
}

function rounded(value) {
  return Math.round(value * 100) / 100;
}

function routeEdge(source, target) {
  const upwards = target.y < source.y;
  const sourceY = source.y + (upwards ? -source.height / 2 : source.height / 2);
  const targetY = target.y + (upwards ? target.height / 2 : -target.height / 2);
  const middleY = (sourceY + targetY) / 2;
  return `M${rounded(source.x)},${rounded(sourceY)} C${rounded(source.x)},${rounded(middleY)},${rounded(target.x)},${rounded(middleY)},${rounded(target.x)},${rounded(targetY)}`;
}

export function buildLayeredLayout(nodes, edges) {
  const grouped = new Map();
  nodes.forEach((node) => {
    const descriptor = classifyNode(node);
    if (!grouped.has(descriptor.key)) grouped.set(descriptor.key, { ...descriptor, nodes: [] });
    grouped.get(descriptor.key).nodes.push(node);
  });
  const bands = [...grouped.values()].sort(compareBands);
  bands.forEach((band) => band.nodes.sort(stableNodeCompare));
  orderWithinBands(bands, edges);

  const nonInputBands = bands.filter((band) => band.role !== "input");
  const widestInternalRow = Math.max(
    DIMENSIONS.minimumCoreWidth,
    ...nonInputBands.map((band) => (
      band.nodes.reduce((sum, node) => sum + nodeWidth(node, band.role), 0)
      + Math.max(0, band.nodes.length - 1) * DIMENSIONS.nodeGap
    )),
  );
  const graphWidth = DIMENSIONS.marginLeft + widestInternalRow + DIMENSIONS.marginRight;
  const nodeRows = [];
  let cursorY = DIMENSIONS.marginTop;
  let fitBottom = null;

  bands.forEach((band) => {
    band.top = cursorY;
    if (band.role === "input") {
      const pitchX = DIMENSIONS.inputNodeWidth + DIMENSIONS.inputColumnGap;
      const columns = Math.max(1, Math.floor((widestInternalRow + DIMENSIONS.inputColumnGap) / pitchX));
      const rowCount = Math.max(1, Math.ceil(band.nodes.length / columns));
      const pitchY = DIMENSIONS.inputNodeHeight + DIMENSIONS.inputRowGap;
      band.center = band.top + DIMENSIONS.inputHeaderHeight / 2;
      band.labelY = band.center;
      band.nodes.forEach((node, index) => {
        const row = Math.floor(index / columns);
        const column = index % columns;
        const countInRow = Math.min(columns, band.nodes.length - row * columns);
        const rowWidth = countInRow * DIMENSIONS.inputNodeWidth + Math.max(0, countInRow - 1) * DIMENSIONS.inputColumnGap;
        const rowStart = DIMENSIONS.marginLeft + (widestInternalRow - rowWidth) / 2;
        nodeRows.push({
          ...node,
          role: band.role,
          bandKey: band.key,
          label: nodeLabel(node),
          width: DIMENSIONS.inputNodeWidth,
          height: DIMENSIONS.inputNodeHeight,
          x: rowStart + column * pitchX + DIMENSIONS.inputNodeWidth / 2,
          y: band.top + DIMENSIONS.inputHeaderHeight + row * pitchY + DIMENSIONS.inputNodeHeight / 2,
        });
      });
      band.bottom = band.top + DIMENSIONS.inputHeaderHeight + rowCount * pitchY + DIMENSIONS.inputBottomPadding;
      fitBottom = band.top
        + DIMENSIONS.inputHeaderHeight
        + Math.min(rowCount, DIMENSIONS.inputFitPreviewRows) * pitchY
        + DIMENSIONS.inputBottomPadding;
    } else {
      band.center = band.top + DIMENSIONS.layerHeight / 2;
      band.labelY = band.center;
      const widths = band.nodes.map((node) => nodeWidth(node, band.role));
      const rowWidth = widths.reduce((sum, width) => sum + width, 0) + Math.max(0, widths.length - 1) * DIMENSIONS.nodeGap;
      let cursorX = DIMENSIONS.marginLeft + (widestInternalRow - rowWidth) / 2;
      band.nodes.forEach((node, index) => {
        const width = widths[index];
        nodeRows.push({
          ...node,
          role: band.role,
          bandKey: band.key,
          label: nodeLabel(node),
          width,
          height: DIMENSIONS.nodeHeight,
          x: cursorX + width / 2,
          y: band.center,
        });
        cursorX += width + DIMENSIONS.nodeGap;
      });
      band.bottom = band.top + DIMENSIONS.layerHeight;
    }
    cursorY = band.bottom;
  });

  const graphHeight = cursorY + DIMENSIONS.marginBottom;
  const nodeById = new Map(nodeRows.map((node) => [node.id, node]));
  const edgeRows = edges.flatMap((edge) => {
    const source = nodeById.get(edge.source);
    const target = nodeById.get(edge.target);
    return source && target ? [{ ...edge, path: routeEdge(source, target) }] : [];
  });
  const publicBands = bands.map(({ nodes: _nodes, ...band }) => band);
  const bounds = { x: 0, y: 0, width: graphWidth, height: graphHeight };
  const fitHeight = fitBottom === null
    ? graphHeight
    : Math.min(graphHeight, fitBottom + DIMENSIONS.marginBottom);
  return {
    bands: publicBands,
    nodeRows,
    nodeById,
    edgeRows,
    bounds,
    fitBounds: { ...bounds, height: fitHeight },
  };
}
