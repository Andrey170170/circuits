"""Versioned schemas for graph-local occurrence labeling runs."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from circuits.analysis.bonafide.canonical import canonical_sha256

_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,199}\Z")


def require_safe_id(value: str, field: str) -> str:
    if not _SAFE_ID.fullmatch(value):
        raise ValueError(f"{field} is not a safe identifier: {value!r}")
    return value


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceBundleSpec(StrictModel):
    site_root: Path
    viewer_manifest_sha256: str
    catalog_sha256: str


class TraceSelection(StrictModel):
    artifact_id: str
    response_position: int = Field(ge=0)
    artifact_source_sha256: str
    trace_file_sha256: str

    @model_validator(mode="after")
    def validate_artifact_id(self) -> TraceSelection:
        require_safe_id(self.artifact_id, "artifact_id")
        return self


class ExplicitOccurrenceSelection(StrictModel):
    policy: Literal["explicit_occurrence_groups_v1"] = "explicit_occurrence_groups_v1"
    groups: dict[str, list[str]]

    @model_validator(mode="after")
    def validate_occurrences(self) -> ExplicitOccurrenceSelection:
        if not self.groups or any(not values for values in self.groups.values()):
            raise ValueError("every occurrence-selection group must be nonempty")
        values = [item for group in self.groups.values() for item in group]
        if len(values) != len(set(values)):
            raise ValueError("occurrence IDs must be unique across selection groups")
        for group, occurrence_ids in self.groups.items():
            require_safe_id(group, "selection group")
            for occurrence_id in occurrence_ids:
                require_safe_id(occurrence_id, "occurrence_id")
        self.groups = {
            group: sorted(occurrence_ids)
            for group, occurrence_ids in sorted(self.groups.items())
        }
        return self

    @property
    def occurrence_ids(self) -> list[str]:
        return [item for group in self.groups.values() for item in group]


class EvidenceSpec(StrictModel):
    policy: Literal["graph_local_occurrence_evidence_v1"] = (
        "graph_local_occurrence_evidence_v1"
    )
    top_positive_sources: int = Field(default=8, ge=1, le=64)
    top_negative_sources: int = Field(default=8, ge=1, le=64)
    top_incoming_edges: int = Field(default=12, ge=1, le=128)
    top_outgoing_edges: int = Field(default=12, ge=1, le=128)


class LabelerSpec(StrictModel):
    provider: Literal["openai", "anthropic", "openai_compatible"]
    model: str
    max_output_tokens: int = Field(ge=1)
    temperature: float | None = Field(default=None, ge=0, le=2)
    reasoning: dict[str, Any] = Field(default_factory=dict)
    provider_parameters: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def reject_operational_or_secret_parameters(self) -> LabelerSpec:
        if self.reasoning and self.temperature is not None:
            raise ValueError("reasoning and temperature cannot be configured together")
        forbidden = {
            "api_key",
            "api_key_env",
            "authorization",
            "auth",
            "base_url",
            "endpoint",
            "extra_headers",
            "headers",
            "password",
            "secret",
        }

        def inspect(value: Any, path: str) -> None:
            if isinstance(value, dict):
                for key, nested in value.items():
                    normalized = str(key).lower().replace("-", "_")
                    if normalized in forbidden or normalized.endswith(
                        ("_api_key", "_password", "_secret", "_token")
                    ):
                        raise ValueError(
                            "operational or secret provider parameter is forbidden: "
                            f"{path}{key}"
                        )
                    inspect(nested, f"{path}{key}.")
            elif isinstance(value, list):
                for index, nested in enumerate(value):
                    inspect(nested, f"{path}{index}.")

        inspect(self.reasoning, "reasoning.")
        inspect(self.provider_parameters, "provider_parameters.")
        return self


MethodKind = Literal[
    "deterministic_evidence_summary_v1", "structured_llm_graph_role_v1"
]


class MethodSpec(StrictModel):
    method_id: str
    kind: MethodKind
    prompt_version: str
    output_schema: Literal["adag.graph-labeling.occurrence-role-label.v1"] = (
        "adag.graph-labeling.occurrence-role-label.v1"
    )
    labeler: LabelerSpec | None = None

    @model_validator(mode="after")
    def validate_labeler(self) -> MethodSpec:
        require_safe_id(self.method_id, "method_id")
        if self.kind == "structured_llm_graph_role_v1" and self.labeler is None:
            raise ValueError("structured LLM methods require a labeler")
        if (
            self.kind == "deterministic_evidence_summary_v1"
            and self.labeler is not None
        ):
            raise ValueError(
                "deterministic evidence summaries cannot configure a labeler"
            )
        return self

    @property
    def identity_sha256(self) -> str:
        value = self.model_dump(mode="json")
        value.pop("method_id")
        return canonical_sha256(value)


class StudySpec(StrictModel):
    study_name: str
    source: SourceBundleSpec
    trace: TraceSelection
    selection: ExplicitOccurrenceSelection
    evidence: EvidenceSpec = Field(default_factory=EvidenceSpec)
    control_policy: Literal["none_v1"] = "none_v1"
    methods: list[MethodSpec]

    @model_validator(mode="after")
    def validate_methods(self) -> StudySpec:
        aliases = [method.method_id for method in self.methods]
        if not aliases:
            raise ValueError("at least one labeling method is required")
        if len(aliases) != len(set(aliases)):
            raise ValueError("labeling method IDs must be unique")
        semantic_identities = [method.identity_sha256 for method in self.methods]
        if len(semantic_identities) != len(set(semantic_identities)):
            raise ValueError("labeling methods must have unique semantic identities")
        return self

    def scientific_payload(self) -> dict[str, Any]:
        value = self.model_dump(mode="json")
        # Aliases, methods, and host-specific storage paths are separate identities.
        value.pop("study_name")
        value.pop("methods")
        value["source"].pop("site_root")
        return value

    @property
    def identity_sha256(self) -> str:
        return canonical_sha256(self.scientific_payload())


class GraphLabelingSpec(StrictModel):
    schema_version: Literal["adag.graph-labeling.spec.v1"] = (
        "adag.graph-labeling.spec.v1"
    )
    run_name: str
    study: StudySpec

    @model_validator(mode="after")
    def validate_run_name(self) -> GraphLabelingSpec:
        require_safe_id(self.run_name, "run_name")
        return self


class ExecutionSpec(StrictModel):
    schema_version: Literal["adag.graph-labeling.execution.v1"] = (
        "adag.graph-labeling.execution.v1"
    )
    mode: Literal["materialize_only", "local"] = "materialize_only"
    concurrency: int = Field(default=1, ge=1, le=256)
    max_attempts: int = Field(default=1, ge=1, le=10)
    initial_backoff_seconds: float = Field(default=1.0, ge=0)
    max_backoff_seconds: float = Field(default=30.0, ge=0)
    slurm: dict[str, Any] = Field(default_factory=dict)

    @property
    def identity_sha256(self) -> str:
        return canonical_sha256(self.model_dump(mode="json"))


class OccurrenceSubject(StrictModel):
    trace_unit_id: str
    source_trace_sha256: str
    occurrence_id: str
    basis_id: str
    layer: int
    neuron_index: int
    polarity: Literal["+", "-"]
    token_position: int
    selection_group: str
    target: dict[str, Any]


class EvidencePacket(StrictModel):
    schema_version: Literal["adag.graph-labeling.evidence.v1"] = (
        "adag.graph-labeling.evidence.v1"
    )
    evidence_policy: str
    subject: OccurrenceSubject
    claim_boundary: str
    context: dict[str, Any]
    node: dict[str, Any]
    facts: list[dict[str, Any]]
    top_positive_sources: list[dict[str, Any]]
    top_negative_sources: list[dict[str, Any]]
    top_incoming_edges: list[dict[str, Any]]
    top_outgoing_edges: list[dict[str, Any]]
    direct_target_edges: list[dict[str, Any]]
    target_connected_paths: list[dict[str, Any]]
    path_search: dict[str, Any]
    coverage: dict[str, Any]
    evidence_sha256: str

    @model_validator(mode="after")
    def validate_self_hash(self) -> EvidencePacket:
        value = self.model_dump(mode="json")
        recorded = value.pop("evidence_sha256")
        if recorded != canonical_sha256(value):
            raise ValueError("evidence packet content hash mismatch")
        return self


class PromptRequest(StrictModel):
    schema_version: Literal["adag.graph-labeling.prompt-request.v1"] = (
        "adag.graph-labeling.prompt-request.v1"
    )
    request_id: str
    study_sha256: str
    method_id: str
    method_sha256: str
    occurrence_id: str
    evidence_sha256: str
    prompt_version: str
    messages: list[dict[str, Literal["system", "user"] | str]]
    generation: LabelerSpec
    logical_request_sha256: str

    @model_validator(mode="after")
    def validate_logical_hash(self) -> PromptRequest:
        require_safe_id(self.request_id, "request_id")
        require_safe_id(self.method_id, "method_id")
        require_safe_id(self.occurrence_id, "occurrence_id")
        logical = {
            "study_sha256": self.study_sha256,
            "method_sha256": self.method_sha256,
            "occurrence_id": self.occurrence_id,
            "evidence_sha256": self.evidence_sha256,
            "messages": self.messages,
            "generation": self.generation.model_dump(mode="json"),
            "prompt_version": self.prompt_version,
        }
        expected = canonical_sha256(logical)
        if self.logical_request_sha256 != expected:
            raise ValueError("logical request content hash mismatch")
        if self.request_id != f"req-{expected[:24]}":
            raise ValueError("request ID differs from logical request identity")
        return self


LabelStatus = Literal["provisional_label", "insufficient_evidence"]
TargetEffect = Literal["supports", "suppresses", "mixed", "unclear"]


class OccurrenceRoleLabel(StrictModel):
    schema_version: Literal["adag.graph-labeling.occurrence-role-label.v1"] = (
        "adag.graph-labeling.occurrence-role-label.v1"
    )
    method_id: str
    method_sha256: str
    subject: OccurrenceSubject
    status: LabelStatus
    label: str | None
    reads_from: list[str] = Field(default_factory=list)
    cited_evidence_ids: list[str] = Field(default_factory=list)
    claim_citations: dict[str, list[str]] = Field(default_factory=dict)
    apparent_role: str | None = None
    target_effect: TargetEffect = "unclear"
    rationale: str | None = None
    alternative_hypothesis: str | None = None
    limitations: list[str] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0, le=1)
    evidence_sha256: str
    logical_request_sha256: str | None = None
    result_sha256: str | None = None

    @model_validator(mode="after")
    def validate_citation_shape(self) -> OccurrenceRoleLabel:
        if self.status == "provisional_label" and not self.cited_evidence_ids:
            raise ValueError("provisional labels require cited evidence")
        if self.status == "insufficient_evidence" and self.label is not None:
            raise ValueError("insufficient-evidence labels cannot contain a label")
        cited = set(self.cited_evidence_ids)
        for claim, evidence_ids in self.claim_citations.items():
            require_safe_id(claim, "claim citation key")
            if not evidence_ids or not set(evidence_ids).issubset(cited):
                raise ValueError(
                    "claim citations must be nonempty subsets of cited evidence"
                )
        if self.status == "provisional_label":
            required_claims = {
                "label",
                "reads_from",
                "apparent_role",
                "target_effect",
                "rationale",
            }
            if self.alternative_hypothesis:
                required_claims.add("alternative_hypothesis")
            missing = sorted(
                claim
                for claim in required_claims
                if not self.claim_citations.get(claim)
            )
            if missing:
                raise ValueError(
                    "provisional label lacks claim-level citations: "
                    + ", ".join(missing)
                )
        return self


class ExternalResultRow(StrictModel):
    schema_version: Literal["adag.graph-labeling.external-result.v1"] = (
        "adag.graph-labeling.external-result.v1"
    )
    request_id: str
    logical_request_sha256: str
    evidence_sha256: str
    method_sha256: str
    raw_payload: dict[str, Any]
    raw_response_sha256: str

    @model_validator(mode="after")
    def validate_raw_hash(self) -> ExternalResultRow:
        require_safe_id(self.request_id, "request_id")
        if self.raw_response_sha256 != canonical_sha256(self.raw_payload):
            raise ValueError("external-result raw response hash mismatch")
        return self


class RunReceipt(StrictModel):
    run_root: Path
    state: Literal["prepared", "materialized", "completed"]
    study_sha256: str
    method_id: str | None = None
    method_sha256: str | None = None
    label_set_id: str | None = None
    execution_sha256: str | None = None
    occurrence_count: int
    label_count: int = 0
    request_count: int = 0


class ExportReceipt(StrictModel):
    destination: Path
    label_set_id: str
    method_id: str
    content_sha256: str
    selected_count: int
    unselected_count: int
