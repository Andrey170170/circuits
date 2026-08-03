"""Stable request, result, and telemetry schemas for labeling model calls."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from circuits.analysis.bonafide.canonical import canonical_sha256

ProviderKind = Literal["openai", "anthropic", "openai_compatible", "fake"]
RequestStage = Literal["candidate_generation", "cluster_summary"]
TransportKind = Literal["live", "native_batch"]


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ChatMessage(StrictModel):
    role: Literal["system", "user", "assistant"]
    content: str


class GenerationRequest(StrictModel):
    schema_version: Literal["adag.labeling.request.v1"] = "adag.labeling.request.v1"
    request_id: str
    run_id: str
    recipe_id: str
    stage: RequestStage
    state: Literal["primary", "alternative"]
    cluster_id: int
    sample_index: int | None = None
    evidence_partition_id: str
    provider: ProviderKind
    model: str
    transport: TransportKind
    messages: list[ChatMessage]
    max_output_tokens: int
    temperature: float | None = None
    reasoning: dict[str, Any] = Field(default_factory=dict)
    provider_parameters: dict[str, Any] = Field(default_factory=dict)
    prompt_template_version: str
    prompt_sha256: str
    evidence_sha256: str
    source_manifest_sha256: str

    def logical_payload(self) -> dict[str, Any]:
        """Return the provider-independent content used for prompt identity checks."""

        return {
            "messages": [message.model_dump() for message in self.messages],
            "max_output_tokens": self.max_output_tokens,
            "temperature": self.temperature,
            "reasoning": self.reasoning,
            "provider_parameters": self.provider_parameters,
            "prompt_template_version": self.prompt_template_version,
            "prompt_sha256": self.prompt_sha256,
            "evidence_sha256": self.evidence_sha256,
        }


class Usage(StrictModel):
    input_tokens: int | None = None
    uncached_input_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None


class GenerationResult(StrictModel):
    schema_version: Literal["adag.labeling.result.v1"] = "adag.labeling.result.v1"
    request_id: str
    provider_request_id: str | None = None
    provider: ProviderKind
    model_requested: str
    model_resolved: str | None = None
    raw_text: str | None = None
    raw_response_sha256: str | None = None
    parsed: dict[str, Any] | None = None
    parse_status: Literal["success", "empty", "invalid_json", "provider_error"]
    usage: Usage = Field(default_factory=Usage)
    stop_reason: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    attempt_count: int = 1
    latency_seconds: float | None = None
    created_at: str = Field(default_factory=utc_now)


class CostEstimate(StrictModel):
    currency: Literal["USD"] = "USD"
    price_snapshot_id: str
    input_cost: float | None = None
    cache_read_cost: float | None = None
    cache_write_cost: float | None = None
    output_cost: float | None = None
    total_cost: float | None = None
    complete: bool
    missing_components: list[str] = Field(default_factory=list)


class TelemetryRecord(StrictModel):
    schema_version: Literal["adag.labeling.telemetry.v1"] = "adag.labeling.telemetry.v1"
    request_id: str
    run_id: str
    recipe_id: str
    stage: RequestStage
    state: Literal["primary", "alternative"]
    cluster_id: int
    sample_index: int | None = None
    evidence_partition_id: str
    backend: ProviderKind
    endpoint_identity: str
    model_requested: str
    model_resolved: str | None = None
    transport: TransportKind
    prompt_template_version: str
    prompt_sha256: str
    evidence_sha256: str
    source_manifest_sha256: str
    generation_parameters: dict[str, Any]
    logical_request_sha256: str
    provider_request_id: str | None = None
    usage: Usage = Field(default_factory=Usage)
    latency_seconds: float | None = None
    queue_seconds: float | None = None
    attempt_count: int = 1
    stop_reason: str | None = None
    parse_status: str
    response_sha256: str | None = None
    result_artifact: str
    error_type: str | None = None
    cost: CostEstimate | None = None
    slurm_job_id: str | None = None
    slurm_array_task_id: str | None = None
    host: str | None = None
    created_at: str = Field(default_factory=utc_now)

    @classmethod
    def from_request_result(
        cls,
        request: GenerationRequest,
        result: GenerationResult,
        *,
        endpoint_identity: str,
        result_artifact: str,
        cost: CostEstimate | None,
        slurm_job_id: str | None,
        slurm_array_task_id: str | None,
        host: str | None,
    ) -> TelemetryRecord:
        generation_parameters = {
            "max_output_tokens": request.max_output_tokens,
            "temperature": request.temperature,
            "reasoning": request.reasoning,
            "provider_parameters": request.provider_parameters,
        }
        return cls(
            request_id=request.request_id,
            run_id=request.run_id,
            recipe_id=request.recipe_id,
            stage=request.stage,
            state=request.state,
            cluster_id=request.cluster_id,
            sample_index=request.sample_index,
            evidence_partition_id=request.evidence_partition_id,
            backend=request.provider,
            endpoint_identity=endpoint_identity,
            model_requested=request.model,
            model_resolved=result.model_resolved,
            transport=request.transport,
            prompt_template_version=request.prompt_template_version,
            prompt_sha256=request.prompt_sha256,
            evidence_sha256=request.evidence_sha256,
            source_manifest_sha256=request.source_manifest_sha256,
            generation_parameters=generation_parameters,
            logical_request_sha256=canonical_sha256(request.logical_payload()),
            provider_request_id=result.provider_request_id,
            usage=result.usage,
            latency_seconds=result.latency_seconds,
            attempt_count=result.attempt_count,
            stop_reason=result.stop_reason,
            parse_status=result.parse_status,
            response_sha256=result.raw_response_sha256,
            result_artifact=result_artifact,
            error_type=result.error_type,
            cost=cost,
            slurm_job_id=slurm_job_id,
            slurm_array_task_id=slurm_array_task_id,
            host=host,
        )
