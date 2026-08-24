function finiteNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function normalizeProfile(profile) {
  if (Array.isArray(profile)) {
    return profile.map((entry, index) => {
      if (entry && typeof entry === "object") {
        return {
          key: entry.key ?? entry.label ?? entry.position ?? entry.token_position ?? index,
          value: finiteNumber(entry.value ?? entry.attribution ?? entry.contribution ?? entry.score),
        };
      }
      return { key: index, value: finiteNumber(entry) };
    });
  }
  if (profile && typeof profile === "object") {
    return Object.entries(profile).map(([key, value]) => ({ key, value: finiteNumber(value) }));
  }
  return [];
}

export function profileRows(profile) {
  return normalizeProfile(profile)
    .filter((row) => row.value !== null)
    .sort((left, right) => Math.abs(right.value) - Math.abs(left.value));
}

export function profileDisplay(profile, { expanded = false, limit = 18 } = {}) {
  const allRows = profileRows(profile);
  const rows = expanded ? allRows : allRows.slice(0, limit);
  return {
    rows,
    total: allRows.length,
    hiddenCount: allRows.length - rows.length,
  };
}
