from __future__ import annotations

from pathlib import Path

import pytest

from circuits.labeling.config import load_recipe
from circuits.labeling.pricing import estimate_cost, load_price_snapshot
from circuits.labeling.schema import Usage

CONFIG_ROOT = Path("scripts/bonafide/configs/labeling")


@pytest.mark.parametrize("name", ["qwen-only.json", "openai.json", "anthropic.json"])
def test_comparison_recipes_are_valid(name: str) -> None:
    recipe = load_recipe(CONFIG_ROOT / name)
    assert recipe.candidate_samples == 5
    assert recipe.scorer.backend == "transluce_finetuned"
    assert recipe.scorer.model == "Transluce/llama_8b_simulator"


def test_recipes_keep_local_scorer_fixed() -> None:
    recipes = [
        load_recipe(CONFIG_ROOT / name)
        for name in ("qwen-only.json", "openai.json", "anthropic.json")
    ]
    scorer_configs = {recipe.scorer.model_dump_json() for recipe in recipes}
    assert len(scorer_configs) == 1


def test_cost_estimate_uses_mutually_exclusive_usage_buckets() -> None:
    snapshot = load_price_snapshot(CONFIG_ROOT / "prices-2026-07-27.json")
    estimate = estimate_cost(
        snapshot,
        provider="openai",
        model="gpt-5.6-luna",
        transport="live",
        usage=Usage(
            input_tokens=1_200_000,
            uncached_input_tokens=1_000_000,
            cache_read_tokens=200_000,
            cache_write_tokens=0,
            output_tokens=100_000,
        ),
    )
    assert estimate.complete
    assert estimate.total_cost == pytest.approx(1.62)


def test_unknown_rate_fails_open_as_incomplete_not_zero() -> None:
    snapshot = load_price_snapshot(CONFIG_ROOT / "prices-2026-07-27.json")
    estimate = estimate_cost(
        snapshot,
        provider="openai_compatible",
        model="local",
        transport="live",
        usage=Usage(uncached_input_tokens=100, output_tokens=10),
    )
    assert not estimate.complete
    assert estimate.total_cost is None
