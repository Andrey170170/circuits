const OUTPUT_KIND = /logit|output|target/i;
const INPUT_KIND = /^(embedding|input|input_token)$/i;

const DIMENSIONS = Object.freeze({
  marginLeft: 58,
  marginRight: 34,
  marginTop: 62,
  marginBottom: 22,
  minimumColumnWidth: 88,
  columnGap: 12,
  cellPadding: 8,
  tokenLabelCharacterWidth: 7,
  nodeHeight: 29,
  nodeGap: 8,
  inputNodeWidth: 54,
  inputNodeHeight: 21,
  layerHeight: 48,
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

function numericPosition(node) {
  if (node.position === undefined || node.position === null || node.position === "") return null;
  const position = Number(node.position);
  return Number.isFinite(position) ? position : null;
}

function positionKey(node) {
  const position = numericPosition(node);
  return position === null ? "unknown" : `position:${position}`;
}

function stableNodeCompare(left, right) {
  const leftPosition = numericPosition(left);
  const rightPosition = numericPosition(right);
  if (leftPosition !== null && rightPosition !== null && leftPosition !== rightPosition) {
    return leftPosition - rightPosition;
  }
  if (leftPosition === null && rightPosition !== null) return 1;
  if (leftPosition !== null && rightPosition === null) return -1;
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

function tokenTextAt(tokenTextByPosition, position) {
  if (position === null || tokenTextByPosition === null || tokenTextByPosition === undefined) return null;
  if (tokenTextByPosition instanceof Map) {
    return tokenTextByPosition.get(position) ?? tokenTextByPosition.get(String(position)) ?? null;
  }
  return tokenTextByPosition[position] ?? tokenTextByPosition[String(position)] ?? null;
}

function visibleTokenText(value) {
  if (value === null || value === undefined) return null;
  const visible = String(value)
    .replaceAll(" ", "␠")
    .replaceAll("\n", "↵")
    .replaceAll("\t", "⇥");
  if (visible.length === 0) return "∅";
  return visible.length > 18 ? `${visible.slice(0, 17)}…` : visible;
}

function buildColumns(bands, tokenTextByPosition) {
  const columnsByKey = new Map();
  bands.forEach((band) => {
    band.nodes.forEach((node) => {
      const key = positionKey(node);
      if (!columnsByKey.has(key)) {
        const position = numericPosition(node);
        columnsByKey.set(key, { key, position, cells: new Map() });
      }
      const column = columnsByKey.get(key);
      if (!column.cells.has(band.key)) column.cells.set(band.key, []);
      column.cells.get(band.key).push(node);
    });
  });

  const bandByKey = new Map(bands.map((band) => [band.key, band]));
  const columns = [...columnsByKey.values()].sort((left, right) => {
    if (left.position === null) return right.position === null ? 0 : 1;
    if (right.position === null) return -1;
    return left.position - right.position;
  });
  let cursorX = DIMENSIONS.marginLeft;
  columns.forEach((column) => {
    column.tokenText = tokenTextAt(tokenTextByPosition, column.position);
    column.displayTokenText = visibleTokenText(column.tokenText);
    column.label = column.position === null ? "position ?" : `position ${column.position}`;
    let widestCell = 0;
    column.cells.forEach((cellNodes, bandKey) => {
      const band = bandByKey.get(bandKey);
      cellNodes.sort(stableNodeCompare);
      const width = cellNodes.reduce((sum, node) => sum + nodeWidth(node, band.role), 0)
        + Math.max(0, cellNodes.length - 1) * DIMENSIONS.nodeGap;
      widestCell = Math.max(widestCell, width);
    });
    const headerWidth = Math.max(column.label.length, column.displayTokenText?.length ?? 0)
      * DIMENSIONS.tokenLabelCharacterWidth
      + DIMENSIONS.cellPadding * 2;
    column.width = Math.max(
      DIMENSIONS.minimumColumnWidth,
      widestCell + DIMENSIONS.cellPadding * 2,
      headerWidth,
    );
    column.left = cursorX;
    column.right = cursorX + column.width;
    column.x = cursorX + column.width / 2;
    cursorX = column.right + DIMENSIONS.columnGap;
  });
  return columns;
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

export function buildLayeredLayout(nodes, edges, { tokenTextByPosition = null } = {}) {
  const grouped = new Map();
  nodes.forEach((node) => {
    const descriptor = classifyNode(node);
    if (!grouped.has(descriptor.key)) grouped.set(descriptor.key, { ...descriptor, nodes: [] });
    grouped.get(descriptor.key).nodes.push(node);
  });
  const bands = [...grouped.values()].sort(compareBands);
  bands.forEach((band) => band.nodes.sort(stableNodeCompare));
  const columns = buildColumns(bands, tokenTextByPosition);

  const graphWidth = columns.length === 0
    ? DIMENSIONS.marginLeft + DIMENSIONS.marginRight
    : columns.at(-1).right + DIMENSIONS.marginRight;
  const nodeRows = [];
  let cursorY = DIMENSIONS.marginTop;

  bands.forEach((band) => {
    band.top = cursorY;
    band.center = band.top + DIMENSIONS.layerHeight / 2;
    band.labelY = band.center;
    band.bottom = band.top + DIMENSIONS.layerHeight;
    columns.forEach((column) => {
      const cellNodes = column.cells.get(band.key) ?? [];
      const widths = cellNodes.map((node) => nodeWidth(node, band.role));
      const cellWidth = widths.reduce((sum, width) => sum + width, 0)
        + Math.max(0, widths.length - 1) * DIMENSIONS.nodeGap;
      let cursorX = column.x - cellWidth / 2;
      cellNodes.forEach((node, index) => {
        const width = widths[index];
        nodeRows.push({
          ...node,
          role: band.role,
          bandKey: band.key,
          columnKey: column.key,
          label: nodeLabel(node),
          width,
          height: band.role === "input" ? DIMENSIONS.inputNodeHeight : DIMENSIONS.nodeHeight,
          x: cursorX + width / 2,
          y: band.center,
        });
        cursorX += width + DIMENSIONS.nodeGap;
      });
    });
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
  const publicColumns = columns.map(({ cells: _cells, ...column }) => column);
  const bounds = { x: 0, y: 0, width: graphWidth, height: graphHeight };
  return {
    bands: publicBands,
    columns: publicColumns,
    nodeRows,
    nodeById,
    edgeRows,
    bounds,
    fitBounds: bounds,
  };
}
