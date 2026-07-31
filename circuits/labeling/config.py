"""Configuration loader for model-provider labeling recipes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from circuits.labeling.schema import ProviderKind, StrictModel, TransportKind


class RetryConfig(StrictModel):
    max_attempts: int = Field(default=3, ge=1, le=10)
    initial_backoff_seconds: float = Field(default=1.0, ge=0)
    max_backoff_seconds: float = Field(default=30.0, ge=0)


class ModelRoleConfig(StrictModel):
    provider: ProviderKind
    model: str
    transport: TransportKind = "live"
    api_key_env: str | None = None
    base_url_env: str | None = None
    base_url: str | None = None
    max_output_tokens: int = Field(ge=1)
    temperature: float | None = Field(default=None, ge=0, le=2)
    reasoning: dict[str, Any] = Field(default_factory=dict)
    provider_parameters: dict[str, Any] = Field(default_factory=dict)
    concurrency: int = Field(default=8, ge=1, le=256)
    timeout_seconds: float = Field(default=120.0, gt=0)
    retry: RetryConfig = Field(default_factory=RetryConfig)

    @model_validator(mode="after")
    def validate_endpoint(self) -> "ModelRoleConfig":
        if self.base_url and self.base_url_env:
            raise ValueError("set only one of base_url and base_url_env")
        if self.provider == "openai_compatible" and not (
            self.base_url or self.base_url_env
        ):
            raise ValueError(
                "openai_compatible providers require base_url or base_url_env"
            )
        return self


class LocalScorerConfig(StrictModel):
    backend: Literal["transluce_finetuned"] = "transluce_finetuned"
    model: str = "Transluce/llama_8b_simulator"
    model_revision: str = "63919a3fe41f88d91ef764213ae9018e1f8a578e"
    gpu_index: int = Field(default=0, ge=0)
    source_tokenizer: str = "Qwen/Qwen3-4B-Instruct-2507"
    source_tokenizer_revision: str = "cdbee75f17c01a7cc42f958dc650907174af0554"
    aggregate: Literal["polarity_aligned_mean"] = "polarity_aligned_mean"
    alignment: Literal["character_overlap"] = "character_overlap"


class LabelingRecipe(StrictModel):
    schema_version: Literal["adag.labeling.recipe.v1"] = "adag.labeling.recipe.v1"
    recipe_id: str
    description: str
    prompt_policy: Literal["legacy_v1", "width_one_v2"] = "legacy_v1"
    candidate_samples: int = Field(default=5, ge=1, le=20)
    candidate_generator: ModelRoleConfig
    scorer: LocalScorerConfig = Field(default_factory=LocalScorerConfig)
    cluster_summarizer: ModelRoleConfig
    price_snapshot: str

    @model_validator(mode="after")
    def validate_width_one_policy(self) -> "LabelingRecipe":
        if self.prompt_policy != "width_one_v2":
            return self
        if self.cluster_summarizer.max_output_tokens < 1200:
            raise ValueError(
                "width_one_v2 summaries require at least 1200 output tokens"
            )
        for name, role in (
            ("candidate_generator", self.candidate_generator),
            ("cluster_summarizer", self.cluster_summarizer),
        ):
            if (
                role.provider in {"openai", "anthropic"}
                and role.reasoning
                and role.temperature is not None
            ):
                raise ValueError(
                    f"width_one_v2 {name} cannot set temperature with provider reasoning"
                )
        return self


def load_recipe(path: Path) -> LabelingRecipe:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unreadable labeling recipe: {path}") from error
    return LabelingRecipe.model_validate(raw)
