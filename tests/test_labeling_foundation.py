from __future__ import annotations

import json
from pathlib import Path

import pytest
from circuits.labeling.api import openai_usage
from circuits.labeling.config import LabelingRecipe, load_recipe
from circuits.labeling.pricing import estimate_cost, load_price_snapshot
from circuits.labeling.schema import Usage

CONFIG_ROOT = Path("scripts/bonafide/configs/labeling")


@pytest.mark.parametrize(
    "name",
    [
        "qwen-only.json",
        "openai.json",
        "anthropic.json",
        "qwen-only-width-one-v2.json",
        "openai-width-one-v2.json",
        "anthropic-width-one-v2.json",
    ],
)
def test_comparison_recipes_are_valid(name: str) -> None:
    recipe = load_recipe(CONFIG_ROOT / name)
    assert recipe.candidate_samples == 5
    assert recipe.scorer.backend == "transluce_finetuned"
    assert recipe.scorer.model == "Transluce/llama_8b_simulator"
    assert recipe.scorer.model_revision == "63919a3fe41f88d91ef764213ae9018e1f8a578e"


@pytest.mark.parametrize(
    "name",
    [
        "qwen-only-width-one-v2.json",
        "openai-width-one-v2.json",
        "anthropic-width-one-v2.json",
    ],
)
def test_v2_recipes_enable_width_one_policy_and_safe_summary_budget(name: str) -> None:
    recipe = load_recipe(CONFIG_ROOT / name)
    assert recipe.prompt_policy == "width_one_v2"
    assert recipe.recipe_id.endswith("width-one-v2")
    assert recipe.cluster_summarizer.max_output_tokens >= 1200
    for role in (recipe.candidate_generator, recipe.cluster_summarizer):
        if role.provider in {"openai", "anthropic"} and role.reasoning:
            assert role.temperature is None


def test_v2_recipe_rejects_short_summary_budget() -> None:
    raw = json.loads((CONFIG_ROOT / "openai-width-one-v2.json").read_text())
    raw["cluster_summarizer"]["max_output_tokens"] = 1199
    with pytest.raises(ValueError, match="at least 1200"):
        LabelingRecipe.model_validate(raw)


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


def test_current_openai_batch_prices_include_cache_buckets() -> None:
    snapshot = load_price_snapshot(CONFIG_ROOT / "prices-2026-07-30.json")
    estimate = estimate_cost(
        snapshot,
        provider="openai",
        model="gpt-5.6-terra",
        transport="native_batch",
        usage=Usage(
            input_tokens=1_300_000,
            uncached_input_tokens=1_000_000,
            cache_read_tokens=200_000,
            cache_write_tokens=100_000,
            output_tokens=100_000,
        ),
    )
    assert estimate.complete
    assert estimate.total_cost == pytest.approx(1.745)


def test_live_openai_cache_write_receipt_uses_three_input_price_buckets() -> None:
    snapshot = load_price_snapshot(CONFIG_ROOT / "prices-2026-07-30.json")
    raw_usage = json.loads(
        Path("tests/fixtures/openai_responses_usage_cache_write.json").read_text(
            encoding="utf-8"
        )
    )

    estimate = estimate_cost(
        snapshot,
        provider="openai",
        model="gpt-5.6-luna",
        transport="live",
        usage=openai_usage(raw_usage),
    )

    assert estimate.complete
    assert estimate.input_cost == pytest.approx(0.0000006)
    assert estimate.cache_read_cost == 0
    assert estimate.cache_write_cost == pytest.approx(0.00028875)
    assert estimate.output_cost == pytest.approx(0.0004812)
    assert estimate.total_cost == pytest.approx(0.00077055)


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
