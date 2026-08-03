"""Live provider adapters with normalized usage and retry telemetry."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Literal, override

from circuits.labeling.config import ModelRoleConfig
from circuits.labeling.schema import GenerationRequest, GenerationResult, Usage

logger = logging.getLogger(__name__)

ParseStatus = Literal["success", "empty", "invalid_json", "provider_error"]


def parse_json_output(
    text: str, stage: str, prompt_template_version: str | None = None
) -> tuple[dict[str, Any] | None, ParseStatus]:
    stripped = text.strip()
    if not stripped:
        return None, "empty"
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL)
    if fenced:
        stripped = fenced.group(1)
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            return None, "invalid_json"
        try:
            value = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError:
            return None, "invalid_json"
    if not isinstance(value, dict):
        return None, "invalid_json"
    required = {
        "candidate_generation": ("description",),
        "cluster_summary": ("label", "rationale", "confidence"),
    }[stage]
    if any(key not in value for key in required):
        return None, "invalid_json"
    if prompt_template_version == "bonafide-width-one-cluster-candidate-v2":
        fields = (
            "description",
            "localized_evidence",
            "background_or_confound",
            "limitations",
        )
        if set(value) != set(fields) or any(
            not isinstance(value.get(field), str) or not value[field].strip()
            for field in fields
        ):
            return None, "invalid_json"
    if prompt_template_version == "bonafide-width-one-cluster-summary-v2":
        fields = ("label", "rationale", "background_or_confound", "limitations")
        confidence = value.get("confidence")
        status = value.get("status")
        expected_fields = {*fields, "confidence", "status"}
        if (
            set(value) != expected_fields
            or any(
                not isinstance(value.get(field), str) or not value[field].strip()
                for field in fields
            )
            or isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not 0 <= confidence <= 1
            or len(value["label"].split()) > 12
            or status not in {"provisional_label", "insufficient_evidence"}
            or (
                status == "insufficient_evidence"
                and value.get("label") != "insufficient_evidence"
            )
            or (
                status == "provisional_label"
                and value.get("label") == "insufficient_evidence"
            )
        ):
            return None, "invalid_json"
    return value, "success"


def _hash_text(text: str | None) -> str | None:
    return (
        hashlib.sha256(text.encode("utf-8")).hexdigest() if text is not None else None
    )


def _get(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _integer(value: Any) -> int | None:
    return (
        int(value) if isinstance(value, int) and not isinstance(value, bool) else None
    )


def openai_stop_reason(raw: Any) -> str | None:
    """Prefer the actionable reason attached to an incomplete response."""

    incomplete_details = _get(raw, "incomplete_details", {})
    return _get(incomplete_details, "reason") or _get(raw, "status")


def openai_usage(raw: Any) -> Usage:
    input_tokens = _integer(_get(raw, "input_tokens"))
    output_tokens = _integer(_get(raw, "output_tokens"))
    input_details = _get(raw, "input_tokens_details", {})
    output_details = _get(raw, "output_tokens_details", {})
    cached = _integer(_get(input_details, "cached_tokens")) or 0
    reasoning = _integer(_get(output_details, "reasoning_tokens"))
    return Usage(
        input_tokens=input_tokens,
        uncached_input_tokens=(
            max(0, input_tokens - cached) if input_tokens is not None else None
        ),
        cache_read_tokens=cached,
        cache_write_tokens=0,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning,
    )


def anthropic_usage(raw: Any) -> Usage:
    input_tokens = _integer(_get(raw, "input_tokens"))
    cache_read = _integer(_get(raw, "cache_read_input_tokens")) or 0
    cache_write = _integer(_get(raw, "cache_creation_input_tokens")) or 0
    output_tokens = _integer(_get(raw, "output_tokens"))
    return Usage(
        input_tokens=input_tokens,
        # Anthropic reports ordinary, cache-read, and cache-created tokens as
        # separate billable buckets.
        uncached_input_tokens=input_tokens,
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
        output_tokens=output_tokens,
        reasoning_tokens=None,
    )


class AsyncGenerationBackend(ABC):
    def __init__(self, config: ModelRoleConfig):
        self.config = config

    @property
    @abstractmethod
    def endpoint_identity(self) -> str: ...

    @abstractmethod
    async def _generate_once(self, request: GenerationRequest) -> GenerationResult: ...

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        started = time.monotonic()
        attempt = 0
        while True:
            attempt += 1
            try:
                result = await self._generate_once(request)
                result.attempt_count = attempt
                result.latency_seconds = time.monotonic() - started
                return result
            except Exception as error:
                if attempt >= self.config.retry.max_attempts:
                    return GenerationResult(
                        request_id=request.request_id,
                        provider=request.provider,
                        model_requested=request.model,
                        parse_status="provider_error",
                        error_type=type(error).__name__,
                        error_message=str(error)[:2000],
                        attempt_count=attempt,
                        latency_seconds=time.monotonic() - started,
                    )
                delay = min(
                    self.config.retry.max_backoff_seconds,
                    self.config.retry.initial_backoff_seconds * (2 ** (attempt - 1)),
                )
                delay *= random.uniform(0.8, 1.2)
                logger.warning(
                    "retrying request %s after %s (attempt %d)",
                    request.request_id,
                    type(error).__name__,
                    attempt,
                )
                await asyncio.sleep(delay)


class OpenAIResponsesBackend(AsyncGenerationBackend):
    def __init__(self, config: ModelRoleConfig):
        super().__init__(config)
        import os

        from openai import AsyncOpenAI

        key_name = config.api_key_env or "OPENAI_API_KEY"
        api_key = os.environ.get(key_name)
        if not api_key:
            raise ValueError(
                f"required API key environment variable is missing: {key_name}"
            )
        self.client = AsyncOpenAI(api_key=api_key, timeout=config.timeout_seconds)

    @property
    @override
    def endpoint_identity(self) -> str:
        return "https://api.openai.com/v1/responses"

    @override
    async def _generate_once(self, request: GenerationRequest) -> GenerationResult:
        kwargs: dict[str, Any] = {
            "model": request.model,
            "input": [message.model_dump() for message in request.messages],
            "max_output_tokens": request.max_output_tokens,
            **request.provider_parameters,
        }
        if request.temperature is not None and not request.reasoning:
            kwargs["temperature"] = request.temperature
        if request.reasoning:
            kwargs["reasoning"] = request.reasoning
        response = await self.client.responses.create(**kwargs)
        text = response.output_text or ""
        parsed, status = parse_json_output(
            text, request.stage, request.prompt_template_version
        )
        return GenerationResult(
            request_id=request.request_id,
            provider="openai",
            provider_request_id=response.id,
            model_requested=request.model,
            model_resolved=response.model,
            raw_text=text,
            raw_response_sha256=_hash_text(text),
            parsed=parsed,
            parse_status=status,
            usage=openai_usage(response.usage),
            stop_reason=openai_stop_reason(response),
        )


class AnthropicMessagesBackend(AsyncGenerationBackend):
    def __init__(self, config: ModelRoleConfig):
        super().__init__(config)
        import os

        from anthropic import AsyncAnthropic

        key_name = config.api_key_env or "ANTHROPIC_API_KEY"
        api_key = os.environ.get(key_name)
        if not api_key:
            raise ValueError(
                f"required API key environment variable is missing: {key_name}"
            )
        self.client = AsyncAnthropic(api_key=api_key, timeout=config.timeout_seconds)

    @property
    @override
    def endpoint_identity(self) -> str:
        return "https://api.anthropic.com/v1/messages"

    @override
    async def _generate_once(self, request: GenerationRequest) -> GenerationResult:
        system_parts = [
            message.content for message in request.messages if message.role == "system"
        ]
        messages = [
            message.model_dump()
            for message in request.messages
            if message.role != "system"
        ]
        kwargs: dict[str, Any] = {
            "model": request.model,
            "system": "\n\n".join(system_parts),
            "messages": messages,
            "max_tokens": request.max_output_tokens,
            **request.provider_parameters,
        }
        if request.temperature is not None and not request.reasoning:
            kwargs["temperature"] = request.temperature
        if request.reasoning:
            kwargs["thinking"] = request.reasoning
        response = await self.client.messages.create(**kwargs)
        text = "".join(
            str(_get(block, "text", ""))
            for block in response.content
            if _get(block, "type") == "text"
        )
        parsed, status = parse_json_output(
            text, request.stage, request.prompt_template_version
        )
        return GenerationResult(
            request_id=request.request_id,
            provider="anthropic",
            provider_request_id=response.id,
            model_requested=request.model,
            model_resolved=response.model,
            raw_text=text,
            raw_response_sha256=_hash_text(text),
            parsed=parsed,
            parse_status=status,
            usage=anthropic_usage(response.usage),
            stop_reason=response.stop_reason,
        )


class OpenAICompatibleBackend(AsyncGenerationBackend):
    def __init__(self, config: ModelRoleConfig):
        super().__init__(config)
        import os

        from openai import AsyncOpenAI

        base_url = config.base_url or os.environ.get(config.base_url_env or "")
        if not base_url:
            raise ValueError("OpenAI-compatible base URL is missing")
        key_name = config.api_key_env
        api_key = os.environ.get(key_name, "") if key_name else ""
        self.base_url = base_url.rstrip("/")
        self.client = AsyncOpenAI(
            api_key=api_key or "local-not-secret",
            base_url=self.base_url,
            timeout=config.timeout_seconds,
        )

    @property
    @override
    def endpoint_identity(self) -> str:
        return f"{self.base_url}/chat/completions"

    @override
    async def _generate_once(self, request: GenerationRequest) -> GenerationResult:
        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": [message.model_dump() for message in request.messages],
            "max_tokens": request.max_output_tokens,
            **request.provider_parameters,
        }
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.reasoning:
            kwargs["extra_body"] = {"reasoning": request.reasoning}
        response = await self.client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        text = choice.message.content or ""
        parsed, status = parse_json_output(
            text, request.stage, request.prompt_template_version
        )
        usage = response.usage
        prompt_tokens = _integer(_get(usage, "prompt_tokens"))
        cached = (
            _integer(_get(_get(usage, "prompt_tokens_details", {}), "cached_tokens"))
            or 0
        )
        completion_tokens = _integer(_get(usage, "completion_tokens"))
        normalized = Usage(
            input_tokens=prompt_tokens,
            uncached_input_tokens=(
                max(0, prompt_tokens - cached) if prompt_tokens is not None else None
            ),
            cache_read_tokens=cached,
            cache_write_tokens=0,
            output_tokens=completion_tokens,
            reasoning_tokens=_integer(
                _get(_get(usage, "completion_tokens_details", {}), "reasoning_tokens")
            ),
        )
        return GenerationResult(
            request_id=request.request_id,
            provider="openai_compatible",
            provider_request_id=response.id,
            model_requested=request.model,
            model_resolved=response.model,
            raw_text=text,
            raw_response_sha256=_hash_text(text),
            parsed=parsed,
            parse_status=status,
            usage=normalized,
            stop_reason=choice.finish_reason,
        )


class FakeBackend(AsyncGenerationBackend):
    @property
    @override
    def endpoint_identity(self) -> str:
        return "fake://deterministic"

    @override
    async def _generate_once(self, request: GenerationRequest) -> GenerationResult:
        if request.stage == "candidate_generation":
            value: dict[str, Any] = {
                "description": (
                    f"Deterministic candidate for {request.state} cluster "
                    f"{request.cluster_id}, sample {request.sample_index}."
                )
            }
            if (
                request.prompt_template_version
                == "bonafide-width-one-cluster-candidate-v2"
            ):
                value.update(
                    localized_evidence="Deterministic highlighted-token evidence.",
                    background_or_confound="Shared corpus context is not localized evidence.",
                    limitations="Single-target width-one attribution only.",
                )
        else:
            value = {
                "label": f"cluster-{request.cluster_id}",
                "rationale": "Deterministic fake summary for pipeline validation.",
                "confidence": 0.5,
            }
            if (
                request.prompt_template_version
                == "bonafide-width-one-cluster-summary-v2"
            ):
                value.update(
                    status="provisional_label",
                    background_or_confound="Shared corpus context is excluded.",
                    limitations="Single-target width-one attribution only.",
                )
        text = json.dumps(value, sort_keys=True)
        input_tokens = sum(len(message.content.split()) for message in request.messages)
        return GenerationResult(
            request_id=request.request_id,
            provider="fake",
            provider_request_id=f"fake-{request.request_id}",
            model_requested=request.model,
            model_resolved=request.model,
            raw_text=text,
            raw_response_sha256=_hash_text(text),
            parsed=value,
            parse_status="success",
            usage=Usage(
                input_tokens=input_tokens,
                uncached_input_tokens=input_tokens,
                cache_read_tokens=0,
                cache_write_tokens=0,
                output_tokens=len(text.split()),
                reasoning_tokens=0,
            ),
            stop_reason="stop",
        )


def create_backend(config: ModelRoleConfig) -> AsyncGenerationBackend:
    if config.provider == "openai":
        return OpenAIResponsesBackend(config)
    if config.provider == "anthropic":
        return AnthropicMessagesBackend(config)
    if config.provider == "openai_compatible":
        return OpenAICompatibleBackend(config)
    return FakeBackend(config)
