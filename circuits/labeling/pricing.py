"""Explicit, dated API price snapshots and usage-based cost estimates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from circuits.labeling.schema import CostEstimate, Usage


def load_price_snapshot(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unreadable price snapshot: {path}") from error
    if value.get("schema_version") != "adag.labeling.prices.v1":
        raise ValueError(f"unsupported price snapshot schema: {path}")
    return value


def estimate_cost(
    snapshot: dict[str, Any],
    *,
    provider: str,
    model: str,
    transport: str,
    usage: Usage,
) -> CostEstimate:
    snapshot_id = str(snapshot["snapshot_id"])
    try:
        rates = snapshot["rates"][provider][model][transport]
    except KeyError:
        return CostEstimate(
            price_snapshot_id=snapshot_id,
            complete=False,
            missing_components=[f"rate:{provider}/{model}/{transport}"],
        )

    components: dict[str, tuple[int | None, str]] = {
        "input": (usage.uncached_input_tokens, "input_per_million"),
        "cache_read": (usage.cache_read_tokens, "cache_read_per_million"),
        "cache_write": (usage.cache_write_tokens, "cache_write_per_million"),
        "output": (usage.output_tokens, "output_per_million"),
    }
    costs: dict[str, float | None] = {}
    missing: list[str] = []
    for component, (tokens, rate_name) in components.items():
        if tokens in (None, 0):
            costs[component] = 0.0 if tokens == 0 else None
            if tokens is None and component in ("input", "output"):
                missing.append(f"usage:{component}")
            continue
        rate = rates.get(rate_name)
        if rate is None:
            costs[component] = None
            missing.append(f"rate:{component}")
        else:
            costs[component] = float(tokens) / 1_000_000 * float(rate)

    known = [cost for cost in costs.values() if cost is not None]
    complete = not missing
    return CostEstimate(
        price_snapshot_id=snapshot_id,
        input_cost=costs["input"],
        cache_read_cost=costs["cache_read"],
        cache_write_cost=costs["cache_write"],
        output_cost=costs["output"],
        total_cost=sum(known) if complete else None,
        complete=complete,
        missing_components=missing,
    )
