function firstPresent(...values) {
  return values.find((value) => value !== undefined && value !== null);
}

export function findLabelRecord(overlay, node) {
  if (!overlay) return null;
  const source = firstPresent(
    overlay.labels,
    overlay.nodes,
    overlay.entries,
    overlay.annotations,
    overlay,
  );
  const occurrenceIds = [node.id, node.occurrenceId].filter(Boolean).map(String);

  if (Array.isArray(source)) {
    const exact = source.find((entry) =>
      [entry.occurrence_id, entry.node_id, entry.id]
        .filter(Boolean)
        .map(String)
        .some((id) => occurrenceIds.includes(id)),
    );
    if (exact) return exact;
    return source.find((entry) =>
      entry.basis_id && node.basisId && String(entry.basis_id) === String(node.basisId),
    ) ?? null;
  }
  if (source && typeof source === "object") {
    for (const id of occurrenceIds) {
      if (source[id] !== undefined && source[id] !== null) return source[id];
    }
    return node.basisId ? source[node.basisId] ?? null : null;
  }
  return null;
}
