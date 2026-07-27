"""
High-level circuit tracing: prepare inputs, run CLJA, and produce a CircuitData artifact.
"""

import hashlib
import json
import math
import pickle
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from circuits.tracing.clja import (
    ADAGConfig,
    CLJAProbeSelection,
    get_all_pairs_cl_ja_effects_with_attributions,
)
from circuits.tracing.candidates import (
    CandidateLogitAxis,
    CandidatePolicyId,
    CandidateSelection,
    JointLogitObjective,
    JointObjectiveId,
    build_joint_objective,
    select_candidate_logits,
)
from circuits.tracing.instrumentation import (
    TraceInstrumentation,
    instrumentation_stage,
)
from circuits.tracing.utils import Edge, Node
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer


@dataclass
class CircuitData:
    """Complete output of circuit tracing — everything needed for downstream analysis."""

    df_node: pd.DataFrame
    df_edge: pd.DataFrame
    cis: list[list[int]]
    attention_masks: list[list[int]]
    labels: list[str]
    target_logits: list[list[int]]
    target_logit_probs: list[list[float]]
    k: int
    config: ADAGConfig
    model_id: str = ""
    traced_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    target_logit_values: list[list[float]] = field(default_factory=list)
    target_provenance: list[dict[str, object]] = field(default_factory=list)
    trace_metadata: dict[str, object] = field(default_factory=dict)
    benchmark_only: bool = False

    @classmethod
    def merge(cls, shards: list["CircuitData"]) -> "CircuitData":
        """Merge multiple CircuitData shards (from parallel workers) into one.

        Re-indexes the label suffixes (___N) in df_node/df_edge so they are globally unique.
        """
        if len(shards) == 1:
            return shards[0]

        all_df_node = []
        all_df_edge = []
        global_offset = 0

        for shard in shards:
            shard_size = len(shard.labels)
            if shard_size == 0:
                continue

            # Re-index label suffixes: replace ___<local_idx> with ___<global_idx>
            df_node = shard.df_node.copy()
            df_edge = shard.df_edge.copy()

            def reindex_label(label: str, offset: int) -> str:
                parts = label.rsplit("___", 1)
                if len(parts) == 2:
                    return f"{parts[0]}___{int(parts[1]) + offset}"
                return label

            df_node["label"] = df_node["label"].apply(
                lambda l: reindex_label(l, global_offset)
            )
            df_edge["label"] = df_edge["label"].apply(
                lambda l: reindex_label(l, global_offset)
            )

            all_df_node.append(df_node)
            all_df_edge.append(df_edge)
            global_offset += shard_size

        return cls(
            df_node=pd.concat(all_df_node, ignore_index=True),
            df_edge=pd.concat(all_df_edge, ignore_index=True),
            cis=[ci for shard in shards for ci in shard.cis],
            attention_masks=[am for shard in shards for am in shard.attention_masks],
            labels=[l for shard in shards for l in shard.labels],
            target_logits=[tl for shard in shards for tl in shard.target_logits],
            target_logit_probs=[
                tp for shard in shards for tp in shard.target_logit_probs
            ],
            k=shards[0].k,
            config=shards[0].config,
            model_id=shards[0].model_id,
            target_logit_values=[
                values
                for shard in shards
                for values in getattr(shard, "target_logit_values", [])
            ],
            target_provenance=[
                provenance
                for shard in shards
                for provenance in getattr(shard, "target_provenance", [])
            ],
            trace_metadata={
                "merged_shard_metadata": [
                    getattr(shard, "trace_metadata", {}) for shard in shards
                ]
            },
            benchmark_only=any(
                getattr(shard, "benchmark_only", False) for shard in shards
            ),
        )

    def save_to_pickle(self, path: str) -> None:
        """Save CircuitData to a pickle file."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_from_pickle(cls, path: str) -> "CircuitData":
        """Load CircuitData from a pickle file."""
        with open(path, "rb") as f:
            return pickle.load(f)


TOPK_TRACE_FAMILY_ID = "bonafide.topk-position.v1"
TOPK_CONTRIBUTION_SCHEMA_ID = "adag.candidate-contribution.raw-logit.v1"


@dataclass(frozen=True)
class TopKPositionTrace:
    """One response target with a distinct same-position candidate-logit axis."""

    circuit_data: CircuitData
    trace_family_id: str
    shared_response_position: int
    shared_prediction_position: int
    candidate_selection: CandidateSelection
    joint_objective: JointLogitObjective
    candidate_contribution_schema: dict[str, object]

    @property
    def candidate_count(self) -> int:
        return len(self.candidate_selection.candidates)

    def contract_dict(self) -> dict[str, object]:
        return {
            "trace_family_id": self.trace_family_id,
            "shared_response_position": self.shared_response_position,
            "shared_prediction_position": self.shared_prediction_position,
            "candidate_count": self.candidate_count,
            "candidate_selection": self.candidate_selection.to_dict(),
            "joint_objective": self.joint_objective.to_dict(),
            "candidate_contribution_schema": self.candidate_contribution_schema,
        }


PROBE_SCHEMA_VERSION = "adag.teacher-forced-probe.v1"
PROBE_OCCURRENCE_SCHEMA_VERSION = "adag.selected-neuron-occurrences.v1"
PROBE_FEATURE_BASIS_SCHEMA_VERSION = "adag.selected-neuron-feature-basis.v1"


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _stable_json_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _model_config_for_hash(model: PreTrainedModel) -> dict[str, object]:
    config_to_dict = getattr(model.config, "to_dict", None)
    if callable(config_to_dict):
        return config_to_dict()
    return {
        "_name_or_path": getattr(model.config, "_name_or_path", None),
        "_commit_hash": getattr(model.config, "_commit_hash", None),
        "model_type": getattr(model.config, "model_type", None),
        "_attn_implementation": getattr(model.config, "_attn_implementation", None),
    }


@dataclass(frozen=True)
class TeacherForcedProbeResult:
    """Versioned, graph-free estimate for one teacher-forced target token."""

    target_provenance: dict[str, object]
    selected_occurrences: list[dict[str, int | float]]
    occurrence_signature: dict[str, object]
    feature_basis_signature: dict[str, object]
    instrumentation: dict[str, object]
    trace_metadata: dict[str, object]
    model_identity: dict[str, object]
    adag_config: dict[str, object]
    schema_version: str = PROBE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "target_provenance": self.target_provenance,
            "selected_occurrences": self.selected_occurrences,
            "occurrence_signature": self.occurrence_signature,
            "feature_basis_signature": self.feature_basis_signature,
            "instrumentation": self.instrumentation,
            "trace_metadata": self.trace_metadata,
            "model_identity": self.model_identity,
            "adag_config": self.adag_config,
        }
        # Probe artifacts are deliberately plain JSON rather than pickle-backed
        # CircuitData. Fail here if a caller accidentally adds tensor state.
        _canonical_json_bytes(value)
        return value


def validate_teacher_forced_probe_result(
    probe: TeacherForcedProbeResult | Mapping[str, object],
) -> dict[str, object]:
    """Fail closed on malformed or scientifically ambiguous probe output."""

    value = (
        probe.to_dict() if isinstance(probe, TeacherForcedProbeResult) else dict(probe)
    )
    if value.get("schema_version") != PROBE_SCHEMA_VERSION:
        raise ValueError(f"unsupported probe schema: {value.get('schema_version')!r}")
    provenance = value.get("target_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("probe requires exactly one target_provenance object")
    for field_name in (
        "response_token_position",
        "absolute_token_position",
        "prediction_token_position",
        "token_id",
        "token_text",
        "logit",
        "probability",
    ):
        if field_name not in provenance:
            raise ValueError(f"probe target_provenance is missing {field_name}")
    for field_name in (
        "response_token_position",
        "absolute_token_position",
        "prediction_token_position",
        "token_id",
    ):
        numeric = provenance[field_name]
        if isinstance(numeric, bool) or not isinstance(numeric, int) or numeric < 0:
            raise ValueError(
                f"probe target {field_name} must be a non-negative integer"
            )
    if not isinstance(provenance["token_text"], str):
        raise ValueError("probe target token_text must be a string")
    for field_name in ("logit", "probability"):
        numeric = provenance[field_name]
        if isinstance(numeric, bool) or not isinstance(numeric, (int, float)):
            raise ValueError(f"probe target {field_name} must be numeric")
        if not math.isfinite(float(numeric)):
            raise ValueError(f"probe target {field_name} must be finite")
    if not 0.0 <= float(provenance["probability"]) <= 1.0:
        raise ValueError("probe target probability must be in [0, 1]")

    occurrences = value.get("selected_occurrences")
    if not isinstance(occurrences, list):
        raise ValueError("probe selected_occurrences must be a list")
    occurrence_ids: list[list[int]] = []
    for occurrence in occurrences:
        if not isinstance(occurrence, Mapping):
            raise ValueError("probe selected occurrence must be an object")
        occurrence_id = []
        for field_name in ("layer", "token_position", "neuron"):
            component = occurrence.get(field_name)
            if (
                isinstance(component, bool)
                or not isinstance(component, int)
                or component < 0
            ):
                raise ValueError(
                    f"probe occurrence {field_name} must be a non-negative integer"
                )
            occurrence_id.append(component)
        attribution = occurrence.get("attribution")
        if isinstance(attribution, bool) or not isinstance(attribution, (int, float)):
            raise ValueError("probe occurrence attribution must be numeric")
        if not math.isfinite(float(attribution)):
            raise ValueError("probe occurrence attribution must be finite")
        occurrence_ids.append(occurrence_id)
    if occurrence_ids != sorted(occurrence_ids) or len(occurrence_ids) != len(
        {tuple(item) for item in occurrence_ids}
    ):
        raise ValueError("probe occurrence IDs must be sorted and unique")

    occurrence_signature = value.get("occurrence_signature")
    if not isinstance(occurrence_signature, Mapping):
        raise ValueError("probe occurrence_signature must be an object")
    if occurrence_signature.get("schema_version") != PROBE_OCCURRENCE_SCHEMA_VERSION:
        raise ValueError("unsupported probe occurrence signature schema")
    if occurrence_signature.get("occurrence_ids") != occurrence_ids:
        raise ValueError(
            "probe signature occurrence IDs disagree with selected occurrences"
        )
    if occurrence_signature.get("sha256") != _stable_json_hash(occurrence_ids):
        raise ValueError("probe occurrence signature checksum mismatch")

    feature_ids = sorted({(item[0], item[2]) for item in occurrence_ids})
    feature_ids_json = [[layer, neuron] for layer, neuron in feature_ids]
    feature_basis = value.get("feature_basis_signature")
    if not isinstance(feature_basis, Mapping):
        raise ValueError("probe feature_basis_signature must be an object")
    if feature_basis.get("schema_version") != PROBE_FEATURE_BASIS_SCHEMA_VERSION:
        raise ValueError("unsupported probe feature basis signature schema")
    if feature_basis.get("feature_ids") != feature_ids_json:
        raise ValueError("probe feature basis disagrees with selected occurrences")
    if feature_basis.get("sha256") != _stable_json_hash(feature_ids_json):
        raise ValueError("probe feature basis signature checksum mismatch")

    instrumentation = value.get("instrumentation")
    if not isinstance(instrumentation, Mapping):
        raise ValueError("probe instrumentation must be an object")
    predictors = instrumentation.get("early_predictors")
    if not isinstance(predictors, Mapping):
        raise ValueError("probe instrumentation is missing early predictors")
    if predictors.get("selected_neuron_count") != len(occurrences):
        raise ValueError("probe selected count disagrees with instrumentation")
    forbidden_stages = {
        "selected_attribution_contribution",
        "stop_grad_mlp_attribution_contribution",
        "graph_expansion",
        "activation_collection",
        "logit_graph_materialization",
        "mlp_node_materialization",
        "embedding_graph_materialization",
        "cross_layer_graph_expansion",
        "layer_pair_jacobian",
        "layer_pair_materialization",
        "dataframe_conversion",
    }
    stages = instrumentation.get("stages")
    if not isinstance(stages, Mapping):
        raise ValueError("probe instrumentation stages must be an object")
    leaked = forbidden_stages.intersection(stages)
    if leaked:
        raise ValueError(
            f"probe contains forbidden post-selection stages: {sorted(leaked)}"
        )

    adag_config = value.get("adag_config")
    if not isinstance(adag_config, Mapping) or not isinstance(
        adag_config.get("device"), str
    ):
        raise ValueError("probe adag_config must include a device string")
    metadata = value.get("trace_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("probe trace_metadata must be an object")
    if metadata.get("trace_mode") != "teacher_forced_probe":
        raise ValueError("probe trace_metadata trace_mode is invalid")
    required_hashes = (
        "prompt_sha256",
        "response_sha256",
        "text_bundle_sha256",
        "input_sha256",
        "adag_config_sha256",
        "chat_template_sha256",
    )
    for field_name in required_hashes:
        _validate_sha256(metadata.get(field_name), f"trace_metadata.{field_name}")
    system_hash = metadata.get("system_prompt_sha256")
    if system_hash is not None:
        _validate_sha256(system_hash, "trace_metadata.system_prompt_sha256")
    if metadata.get("adag_config_sha256") != _stable_json_hash(dict(adag_config)):
        raise ValueError("probe ADAG config checksum mismatch")
    integer_metadata = (
        "assistant_prefix_token_count",
        "response_token_count",
        "included_response_token_count",
        "input_token_count",
        "effective_start_layer",
        "effective_end_layer",
    )
    for field_name in integer_metadata:
        item = metadata.get(field_name)
        minimum = -1 if field_name == "effective_start_layer" else 0
        if isinstance(item, bool) or not isinstance(item, int) or item < minimum:
            raise ValueError(f"probe trace_metadata.{field_name} is invalid")
    prefix_count = metadata["assistant_prefix_token_count"]
    response_count = metadata["response_token_count"]
    included_count = metadata["included_response_token_count"]
    input_count = metadata["input_token_count"]
    response_position = provenance["response_token_position"]
    if included_count != response_position + 1 or response_count < included_count:
        raise ValueError("probe response-position metadata is inconsistent")
    if input_count != prefix_count + included_count:
        raise ValueError("probe input-token metadata is inconsistent")
    if provenance["absolute_token_position"] != prefix_count + response_position:
        raise ValueError("probe absolute target position is inconsistent")
    if (
        provenance["prediction_token_position"]
        != provenance["absolute_token_position"] - 1
    ):
        raise ValueError("probe prediction target position is inconsistent")
    if metadata["effective_end_layer"] < max(metadata["effective_start_layer"], 0):
        raise ValueError("probe effective layer bounds are inconsistent")

    model_identity = value.get("model_identity")
    if not isinstance(model_identity, Mapping):
        raise ValueError("probe model_identity must be an object")
    for field_name in ("model_id", "revision", "hash_semantics"):
        if (
            not isinstance(model_identity.get(field_name), str)
            or not model_identity[field_name]
        ):
            raise ValueError(f"probe model_identity.{field_name} is required")
    _validate_sha256(
        model_identity.get("model_config_sha256"),
        "model_identity.model_config_sha256",
    )
    _validate_sha256(model_identity.get("sha256"), "model_identity.sha256")
    unhashed_model_identity = dict(model_identity)
    declared_model_hash = unhashed_model_identity.pop("sha256")
    if declared_model_hash != _stable_json_hash(unhashed_model_identity):
        raise ValueError("probe model identity checksum mismatch")
    _canonical_json_bytes(value)
    return value


def _validate_sha256(value: object, field_name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"probe {field_name} must be a lowercase SHA-256 digest")


# Copied from util.chat_input — removes default system preamble ("Cutting Knowledge Date: ...")
# while keeping the system header structure.
STRIPPED_LLAMA_CHAT_TEMPLATE = '{{- bos_token }}\n{%- if custom_tools is defined %}\n    {%- set tools = custom_tools %}\n{%- endif %}\n{%- if not tools_in_user_message is defined %}\n    {%- set tools_in_user_message = true %}\n{%- endif %}\n{%- if not date_string is defined %}\n    {%- set date_string = "26 Jul 2024" %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n\n{#- This block extracts the system message, so we can slot it into the right place. #}\n{%- if messages[0][\'role\'] == \'system\' %}\n    {%- set system_message = messages[0][\'content\']|trim %}\n    {%- set messages = messages[1:] %}\n{%- else %}\n    {%- set system_message = "" %}\n{%- endif %}\n\n{#- System message + builtin tools #}\n{{- "<|start_header_id|>system<|end_header_id|>\\n\\n" }}\n{%- if builtin_tools is defined or tools is not none %}\n    {{- "Environment: ipython\\n" }}\n{%- endif %}\n{%- if builtin_tools is defined %}\n    {{- "Tools: " + builtin_tools | reject(\'equalto\', \'code_interpreter\') | join(", ") + "\\n\\n"}}\n{%- endif %}\n{%- if tools is not none and not tools_in_user_message %}\n    {{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }}\n    {{- \'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.\' }}\n    {{- "Do not use variables.\\n\\n" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- "\\n\\n" }}\n    {%- endfor %}\n{%- endif %}\n{{- system_message }}\n{{- "<|eot_id|>" }}\n\n{#- Custom tools are passed in a user message with some extra guidance #}\n{%- if tools_in_user_message and not tools is none %}\n    {#- Extract the first user message so we can plug it in here #}\n    {%- if messages | length != 0 %}\n        {%- set first_user_message = messages[0][\'content\']|trim %}\n        {%- set messages = messages[1:] %}\n    {%- else %}\n        {{- raise_exception("Cannot put tools in the first user message when there\'s no first user message!") }}\n{%- endif %}\n    {{- \'<|start_header_id|>user<|end_header_id|>\\n\\n\' -}}\n    {{- "Given the following functions, please respond with a JSON for a function call " }}\n    {{- "with its proper arguments that best answers the given prompt.\\n\\n" }}\n    {{- \'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.\' }}\n    {{- "Do not use variables.\\n\\n" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- "\\n\\n" }}\n    {%- endfor %}\n    {{- first_user_message + "<|eot_id|>"}}\n{%- endif %}\n\n{%- for message in messages %}\n    {%- if not (message.role == \'ipython\' or message.role == \'tool\' or \'tool_calls\' in message) %}\n        {{- \'<|start_header_id|>\' + message[\'role\'] + \'<|end_header_id|>\\n\\n\'+ message[\'content\'] | trim + \'<|eot_id|>\' }}\n    {%- elif \'tool_calls\' in message %}\n        {%- if not message.tool_calls|length == 1 %}\n            {{- raise_exception("This model only supports single tool-calls at once!") }}\n        {%- endif %}\n        {%- set tool_call = message.tool_calls[0].function %}\n        {%- if builtin_tools is defined and tool_call.name in builtin_tools %}\n            {{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' -}}\n            {{- "<|python_tag|>" + tool_call.name + ".call(" }}\n            {%- for arg_name, arg_val in tool_call.arguments | items %}\n                {{- arg_name + \'="\' + arg_val + \'"\' }}\n                {%- if not loop.last %}\n                    {{- ", " }}\n                {%- endif %}\n                {%- endfor %}\n            {{- ")" }}\n        {%- else  %}\n            {{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' -}}\n            {{- \'{"name": "\' + tool_call.name + \'", \' }}\n            {{- \'"parameters": \' }}\n            {{- tool_call.arguments | tojson }}\n            {{- "}" }}\n        {%- endif %}\n        {%- if builtin_tools is defined %}\n            {#- This means we\'re in ipython mode #}\n            {{- "<|eom_id|>" }}\n        {%- else %}\n            {{- "<|eot_id|>" }}\n        {%- endif %}\n    {%- elif message.role == "tool" or message.role == "ipython" %}\n        {{- "<|start_header_id|>ipython<|end_header_id|>\\n\\n" }}\n        {%- if message.content is mapping or message.content is iterable %}\n            {{- message.content | tojson }}\n        {%- else %}\n            {{- message.content }}\n        {%- endif %}\n        {{- "<|eot_id|>" }}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' }}\n{%- endif %}\n'

# Model-specific chat template overrides. Models not listed here use tokenizer.chat_template.
CHAT_TEMPLATES: dict[str, str] = {
    "meta-llama/Llama-3.1-8B-Instruct": STRIPPED_LLAMA_CHAT_TEMPLATE,
}

# Tokens marking the end of a header per model family, used for rollout position detection.
HEADER_END_TOKENS: dict[str, str] = {
    "meta-llama/Llama-3.1-8B-Instruct": "<|end_header_id|>",
}


def get_chat_template(tokenizer: PreTrainedTokenizer) -> str:
    """Get the appropriate chat template for the tokenizer's model."""
    model_id = getattr(tokenizer, "name_or_path", "")
    if model_id in CHAT_TEMPLATES:
        return CHAT_TEMPLATES[model_id]
    # Fall back to the tokenizer's built-in chat template
    return tokenizer.chat_template


def get_header_end_token(tokenizer: PreTrainedTokenizer) -> str | None:
    """Get the header-end token string for rollout position detection, or None."""
    model_id = getattr(tokenizer, "name_or_path", "")
    return HEADER_END_TOKENS.get(model_id)


@dataclass(frozen=True)
class TeacherForcedInput:
    """A single, exactly aligned teacher-forced response trace input.

    ``target_prediction_positions`` are positions whose residual stream predicts
    the corresponding token in ``target_token_ids``.  The input is truncated
    immediately after the last selected response token, so the selected token
    itself is present only as the next-token label, never as causal context for
    its own prediction.
    """

    input_ids: list[int]
    attention_mask: list[int]
    assistant_prefix_token_count: int
    response_token_count: int
    included_response_token_count: int
    selected_response_positions: list[int]
    target_prediction_positions: list[int]
    target_token_ids: list[int]
    target_token_texts: list[str]


@dataclass(frozen=True)
class TokenizedTeacherForcedResponse:
    """Exact chat-template decomposition used by teacher-forced tracing."""

    assistant_prefix_ids: list[int]
    response_ids: list[int]
    assistant_suffix_ids: list[int]


def _stable_text_hash(value: str | None) -> str | None:
    if value is None:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _apply_chat_template_ids(
    tokenizer: PreTrainedTokenizer,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> list[int]:
    ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=add_generation_prompt,
        chat_template=get_chat_template(tokenizer),
    )
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        if len(ids) != 1:
            raise ValueError("teacher-forced tracing requires a single chat input")
        ids = ids[0]
    return [int(token_id) for token_id in ids]


def tokenize_teacher_forced_response(
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    response: str,
    *,
    system_prompt: str | None = None,
) -> TokenizedTeacherForcedResponse:
    """Decompose a frozen chat into exact assistant prefix/content/suffix IDs.

    The assistant boundary and end-of-turn suffix are derived by applying the
    tokenizer's chat template to both an assistant generation prefix and an
    empty assistant turn.  This avoids model-family-specific marker searches.
    Benchmark manifest preparation should use this helper so its response-token
    positions are identical to tracing semantics.
    """
    prompt_messages: list[dict[str, str]] = []
    if system_prompt is not None:
        prompt_messages.append({"role": "system", "content": system_prompt})
    prompt_messages.append({"role": "user", "content": prompt})

    prefix_ids = _apply_chat_template_ids(
        tokenizer, prompt_messages, add_generation_prompt=True
    )
    empty_turn_ids = _apply_chat_template_ids(
        tokenizer,
        [*prompt_messages, {"role": "assistant", "content": ""}],
        add_generation_prompt=False,
    )
    full_ids = _apply_chat_template_ids(
        tokenizer,
        [*prompt_messages, {"role": "assistant", "content": response}],
        add_generation_prompt=False,
    )

    if empty_turn_ids[: len(prefix_ids)] != prefix_ids:
        raise ValueError(
            "chat template does not expose an exact assistant generation-prefix boundary"
        )
    assistant_suffix = empty_turn_ids[len(prefix_ids) :]
    if full_ids[: len(prefix_ids)] != prefix_ids:
        raise ValueError(
            "tokenized response does not share the exact assistant generation prefix"
        )
    if assistant_suffix:
        if full_ids[-len(assistant_suffix) :] != assistant_suffix:
            raise ValueError(
                "chat template assistant suffix changed for non-empty content"
            )
        response_ids = full_ids[len(prefix_ids) : -len(assistant_suffix)]
    else:
        response_ids = full_ids[len(prefix_ids) :]

    if not response_ids:
        raise ValueError("response contains no tokens after applying the chat template")
    return TokenizedTeacherForcedResponse(
        assistant_prefix_ids=prefix_ids,
        response_ids=response_ids,
        assistant_suffix_ids=assistant_suffix,
    )


def prepare_teacher_forced_input(
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    response: str,
    target_response_positions: list[int],
    *,
    system_prompt: str | None = None,
) -> TeacherForcedInput:
    """Tokenize a frozen response and align response-relative trace targets.

    No generation or model forward pass occurs here.
    """
    if not target_response_positions:
        raise ValueError("target_response_positions must contain at least one position")
    if any(
        isinstance(position, bool) or not isinstance(position, int)
        for position in target_response_positions
    ):
        raise TypeError("target_response_positions must contain integers")
    if len(set(target_response_positions)) != len(target_response_positions):
        raise ValueError("target_response_positions must be unique")
    if target_response_positions != sorted(target_response_positions):
        raise ValueError("target_response_positions must be sorted in ascending order")
    if target_response_positions[0] < 0:
        raise ValueError("target_response_positions cannot contain negative positions")

    tokenized = tokenize_teacher_forced_response(
        tokenizer,
        prompt,
        response,
        system_prompt=system_prompt,
    )
    prefix_ids = tokenized.assistant_prefix_ids
    response_ids = tokenized.response_ids
    if target_response_positions[-1] >= len(response_ids):
        raise ValueError(
            "target response position "
            f"{target_response_positions[-1]} is outside the tokenized response "
            f"(length {len(response_ids)})"
        )
    if not prefix_ids:
        raise ValueError("chat template produced an empty assistant prefix")

    included_response_count = target_response_positions[-1] + 1
    input_ids = prefix_ids + response_ids[:included_response_count]
    target_token_ids = [
        response_ids[position] for position in target_response_positions
    ]
    target_prediction_positions = [
        len(prefix_ids) + position - 1 for position in target_response_positions
    ]
    return TeacherForcedInput(
        input_ids=input_ids,
        attention_mask=[1] * len(input_ids),
        assistant_prefix_token_count=len(prefix_ids),
        response_token_count=len(response_ids),
        included_response_token_count=included_response_count,
        selected_response_positions=list(target_response_positions),
        target_prediction_positions=target_prediction_positions,
        target_token_ids=target_token_ids,
        target_token_texts=[
            tokenizer.decode([token_id]) for token_id in target_token_ids
        ],
    )


def _teacher_forced_target_scores(
    model: PreTrainedModel,
    prepared: TeacherForcedInput,
) -> tuple[list[float], list[float]]:
    device = next(model.parameters()).device
    input_ids = torch.tensor([prepared.input_ids], device=device)
    attention_mask = torch.tensor([prepared.attention_mask], device=device)
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[0]

    target_logits: list[float] = []
    target_probs: list[float] = []
    for prediction_position, token_id in zip(
        prepared.target_prediction_positions, prepared.target_token_ids
    ):
        position_logits = logits[prediction_position]
        target_logits.append(float(position_logits[token_id].item()))
        target_probs.append(
            float(torch.softmax(position_logits, dim=-1)[token_id].item())
        )
    return target_logits, target_probs


def _teacher_forced_position_logits(
    model: PreTrainedModel,
    prepared: TeacherForcedInput,
) -> torch.Tensor:
    """Return the full distribution for one prepared prediction position."""

    if len(prepared.target_prediction_positions) != 1:
        raise ValueError("candidate tracing requires exactly one response target")
    device = next(model.parameters()).device
    input_ids = torch.tensor([prepared.input_ids], device=device)
    attention_mask = torch.tensor([prepared.attention_mask], device=device)
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[0]
    return logits[prepared.target_prediction_positions[0]].detach()


def _teacher_forced_target_provenance(
    prepared: TeacherForcedInput,
    target_logit_values: list[float],
    target_probs: list[float],
) -> list[dict[str, object]]:
    return [
        {
            "response_token_position": response_position,
            "absolute_token_position": prepared.assistant_prefix_token_count
            + response_position,
            "prediction_token_position": prediction_position,
            "token_id": token_id,
            "token_text": token_text,
            "logit": logit,
            "probability": probability,
        }
        for (
            response_position,
            prediction_position,
            token_id,
            token_text,
            logit,
            probability,
        ) in zip(
            prepared.selected_response_positions,
            prepared.target_prediction_positions,
            prepared.target_token_ids,
            prepared.target_token_texts,
            target_logit_values,
            target_probs,
        )
    ]


def probe_teacher_forced_response(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    response: str,
    target_response_positions: list[int],
    config: ADAGConfig,
    *,
    system_prompt: str | None = None,
    instrumentation: TraceInstrumentation | None = None,
    model_revision: str | None = None,
) -> TeacherForcedProbeResult:
    """Estimate one frozen-response target without constructing a graph.

    This API deliberately rejects multiple targets. The initial attribution and
    important-neuron mask are objective-dependent, so a combined-target probe
    would not be a scientifically meaningful stand-in for independent probes.
    """

    if len(target_response_positions) != 1:
        raise ValueError("probe mode requires exactly one teacher-forced target")
    declared_model_revision = model_revision or getattr(
        model.config, "_commit_hash", None
    )
    if not isinstance(declared_model_revision, str) or not declared_model_revision:
        raise ValueError(
            "probe mode requires model_revision or model.config._commit_hash"
        )
    recorder = instrumentation or TraceInstrumentation(device=config.device)
    with instrumentation_stage(recorder, "prepare_input"):
        prepared = prepare_teacher_forced_input(
            tokenizer,
            prompt,
            response,
            target_response_positions,
            system_prompt=system_prompt,
        )
    with instrumentation_stage(recorder, "target_scoring"):
        target_logit_values, target_probs = _teacher_forced_target_scores(
            model, prepared
        )
    declared_model_config = _model_config_for_hash(model)
    pre_probe_model_config_sha256 = _stable_json_hash(declared_model_config)
    clja_error: BaseException | None = None
    try:
        with instrumentation_stage(recorder, "clja_probe_total"):
            selection = get_all_pairs_cl_ja_effects_with_attributions(
                model=model,
                tokenizer=tokenizer,
                cis=[prepared.input_ids],
                config=config,
                attention_masks=[prepared.attention_mask],
                focus_logits=[prepared.target_token_ids],
                src_tokens=list(range(len(prepared.input_ids) - 1)),
                tgt_tokens=prepared.target_prediction_positions,
                instrumentation=recorder,
                probe_only=True,
            )
    except BaseException as error:
        clja_error = error
        raise
    finally:
        post_probe_model_config_sha256 = _stable_json_hash(
            _model_config_for_hash(model)
        )
        recorder.set_counter(
            "probe_model_config_restored",
            post_probe_model_config_sha256 == pre_probe_model_config_sha256,
        )
        if post_probe_model_config_sha256 != pre_probe_model_config_sha256:
            if clja_error is not None:
                recorder.set_counter("probe_model_config_leak_during_failed_clja", True)
                clja_error.add_note(
                    "ADAG probe also leaked model.config state while failing; "
                    "the resident model must not be reused"
                )
            else:
                raise RuntimeError(
                    "probe CLJA mutated model.config; refusing a poisoned resident model"
                )
    if not isinstance(selection, CLJAProbeSelection):
        raise TypeError("CLJA probe mode returned an unexpected result")
    recorder.set_counter(
        "probe_selected_occurrence_count", len(selection.selected_occurrences)
    )
    recorder.set_counter("probe_graph_work_skipped", True)

    occurrence_ids = [
        [
            int(occurrence["layer"]),
            int(occurrence["token_position"]),
            int(occurrence["neuron"]),
        ]
        for occurrence in selection.selected_occurrences
    ]
    occurrence_signature: dict[str, object] = {
        "schema_version": PROBE_OCCURRENCE_SCHEMA_VERSION,
        "ordering": "layer_token_neuron_ascending",
        "occurrence_ids": occurrence_ids,
        "sha256": _stable_json_hash(occurrence_ids),
    }
    feature_ids = sorted(
        {
            (int(occurrence["layer"]), int(occurrence["neuron"]))
            for occurrence in selection.selected_occurrences
        }
    )
    feature_ids_json = [[layer, neuron] for layer, neuron in feature_ids]
    feature_basis_signature: dict[str, object] = {
        "schema_version": PROBE_FEATURE_BASIS_SCHEMA_VERSION,
        "ordering": "layer_neuron_ascending_unique",
        "feature_ids": feature_ids_json,
        "sha256": _stable_json_hash(feature_ids_json),
    }
    adag_config = asdict(config)
    model_source = str(getattr(model.config, "_name_or_path", ""))
    model_identity: dict[str, object] = {
        "model_id": model_source,
        "revision": declared_model_revision,
        "model_type": getattr(model.config, "model_type", None),
        "hash_semantics": "declared_source_revision_and_model_config_v1",
    }
    model_identity["model_config_sha256"] = pre_probe_model_config_sha256
    model_identity["sha256"] = _stable_json_hash(model_identity)
    input_identity = {
        "input_ids": prepared.input_ids,
        "attention_mask": prepared.attention_mask,
    }
    trace_metadata: dict[str, object] = {
        "trace_mode": "teacher_forced_probe",
        "prompt_sha256": _stable_text_hash(prompt),
        "response_sha256": _stable_text_hash(response),
        "system_prompt_sha256": _stable_text_hash(system_prompt),
        "text_bundle_sha256": _stable_json_hash(
            {"prompt": prompt, "response": response, "system_prompt": system_prompt}
        ),
        "input_sha256": _stable_json_hash(input_identity),
        "adag_config_sha256": _stable_json_hash(adag_config),
        "assistant_prefix_token_count": prepared.assistant_prefix_token_count,
        "response_token_count": prepared.response_token_count,
        "included_response_token_count": prepared.included_response_token_count,
        "input_token_count": len(prepared.input_ids),
        "chat_template_sha256": _stable_text_hash(get_chat_template(tokenizer)),
        "effective_start_layer": selection.effective_start_layer,
        "effective_end_layer": selection.effective_end_layer,
    }
    result = TeacherForcedProbeResult(
        target_provenance=_teacher_forced_target_provenance(
            prepared, target_logit_values, target_probs
        )[0],
        selected_occurrences=selection.selected_occurrences,
        occurrence_signature=occurrence_signature,
        feature_basis_signature=feature_basis_signature,
        instrumentation=recorder.snapshot(),
        trace_metadata=trace_metadata,
        model_identity=model_identity,
        adag_config=adag_config,
    )
    validate_teacher_forced_probe_result(result)
    return result


def _strip_starting_at_rindex_in_place(arr: list, value: object) -> list:
    """Strips everything including and after the final occurrence of `value` within `arr`."""
    try:
        rindex = arr[::-1].index(value)
        index = len(arr) - 1 - rindex
        del arr[index:]
    except ValueError:
        pass
    return arr


def prepare_ci(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    question: str,
    seed_response: str,
    k: int,
    system_prompt: str | None = None,
    true_answers: list[str] | None = None,
    use_chat_format: bool = True,
    verbose: bool = False,
):
    """
    Prepare a single chat input.
    """
    # Handle [EMPTY] sentinel: treat as a real seed for template purposes,
    # then strip the encoded "[EMPTY]" tokens from the end of the ci.
    is_empty_seed = seed_response == "[EMPTY]"
    if is_empty_seed:
        seed_response = ""

    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})
    has_seed = seed_response is not None and len(seed_response) > 0
    if has_seed or is_empty_seed:
        messages.append({"role": "assistant", "content": seed_response})

    if use_chat_format:
        token_ids: list[int] = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=not has_seed and not is_empty_seed,
            chat_template=get_chat_template(tokenizer),
        )
        if has_seed or is_empty_seed:
            _strip_starting_at_rindex_in_place(token_ids, tokenizer.eos_token_id)
    else:
        token_ids = tokenizer(question)["input_ids"]

    if seed_response is not None and seed_response.endswith(" "):
        space_token = tokenizer.encode(" ")[1]
        token_ids = token_ids + [space_token]
    if true_answers is not None:
        # then we create the topk using the first token of every true answer
        topk = [tokenizer.encode(answer)[1] for answer in true_answers]
        topk_probs = [0.0] * len(topk)  # no probs available for true_answers
    else:
        input_ids = torch.tensor([token_ids], device=next(model.parameters()).device)
        with torch.no_grad():
            logits = model(input_ids).logits[0, -1]
        topk_result = torch.topk(logits, k)
        topk = topk_result.indices.tolist()
        probs = torch.softmax(logits, dim=-1)
        topk_probs = probs[topk_result.indices].tolist()

    if verbose:
        print("Prepared:", question, seed_response, "->", tokenizer.decode(topk[0]))

    return token_ids, topk, topk_probs


def prepare_cis(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    questions: list[str],
    seed_responses: list[str],
    k: int = 5,
    system_prompt: str | None = None,
    true_answers: list[str] | list[None] | None = None,
    use_chat_format: bool = True,
    verbose: bool = False,
):
    """
    Prepare a list of chat inputs.
    """
    if true_answers is None:
        true_answers = [None] * len(questions)
    res = [
        prepare_ci(
            model, tokenizer, q, sr, k, system_prompt, ta, use_chat_format, verbose
        )
        for q, sr, ta in zip(questions, seed_responses, true_answers)
    ]
    cis = [r[0] for r in res]
    topks = [r[1] for r in res]
    topk_probs_list = [r[2] for r in res]
    max_length = max(len(ci) for ci in cis)

    attention_masks = []
    focus_tokens = []
    focus_probs = []
    for topk, probs in zip(topks, topk_probs_list):
        focus_tokens.append(list(topk))
        focus_probs.append(list(probs))

    # pad on left
    starts = []
    padded_cis: list[list[int]] = []
    for ci in cis:
        starts.append(max_length - len(ci))
        attention_mask = [0] * (max_length - len(ci)) + [1] * len(ci)
        padded_cis.append([tokenizer.pad_token_id] * (max_length - len(ci)) + ci)
        attention_masks.append(attention_mask)
    cis = padded_cis

    # keep all tokens
    keep_pos = []
    for i in range(max_length):
        keep_pos.append(i)
    if verbose:
        print(keep_pos)
        print(attention_masks)
        print(focus_tokens)
    return cis, attention_masks, focus_tokens, focus_probs, keep_pos, starts


def prepare_ci_with_rollout(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    question: str,
    seed_response: str | None = None,
    max_new_tokens: int = 1,
    verbose: bool = True,
):
    """
    Prepare a single chat input.
    """
    messages = [{"role": "user", "content": question}]
    has_seed = seed_response is not None
    if has_seed:
        messages.append({"role": "assistant", "content": seed_response})

    token_ids: list[int] = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=not has_seed,
        chat_template=get_chat_template(tokenizer),
    )
    if has_seed:
        _strip_starting_at_rindex_in_place(token_ids, tokenizer.eos_token_id)
    if seed_response is not None and seed_response.endswith(" "):
        space_token = tokenizer.encode(" ")[1]
        token_ids = token_ids + [space_token]

    # generate additional tokens
    input_ids = torch.tensor([token_ids], device=next(model.parameters()).device)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids, max_new_tokens=max_new_tokens, do_sample=False
        )
    rollout_token_ids = output_ids[0].tolist()[len(token_ids) :]

    if len(rollout_token_ids) != max_new_tokens:
        raise ValueError(
            f"rollout token ids length {len(rollout_token_ids)} != max_new_tokens {max_new_tokens}"
        )

    if verbose:
        print("Prepared:", question, "->", tokenizer.decode(rollout_token_ids))

    return token_ids + rollout_token_ids


def prepare_cis_with_rollout(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    questions: list[str],
    seed_responses: list[str] | None = None,
    max_new_tokens: int = 1,
    verbose: bool = True,
):
    """
    Prepare a list of chat inputs.
    """
    if seed_responses is None:
        seed_responses = [None] * len(questions)
    cis = [
        prepare_ci_with_rollout(
            model, tokenizer, q, seed_response, max_new_tokens, verbose
        )
        for q, seed_response in zip(questions, seed_responses)
    ]
    max_length = max(len(ci) for ci in cis)
    all_attention_masks = []
    all_focus_tokens = []
    all_tgt_tokens = []

    new_cis: list[list[int]] = []
    starts = []
    for ci in cis:
        starts.append(max_length - len(ci))
        header_end = get_header_end_token(tokenizer)
        if header_end is not None:
            end_token = tokenizer.encode(header_end)[-1]
            positions = [i for i, t in enumerate(ci) if t == end_token]
            start_assistant = positions[2] + 2
        else:
            # For models without explicit header tokens (e.g. Qwen3), find the start
            # of assistant content by looking for the last assistant turn marker.
            # The assistant content starts after "<|im_start|>assistant\n"
            im_start_token = tokenizer.encode("<|im_start|>")[-1]
            positions = [i for i, t in enumerate(ci) if t == im_start_token]
            start_assistant = positions[-1] + 2  # skip "assistant" and "\n" tokens
        # offset by 1 because next token prediction
        tgt_tokens = [i - 1 for i in range(start_assistant, len(ci))]
        # could be different length
        focus_tokens = [ci[i + 1] for i in tgt_tokens]

        attention_masks = [0] * (max_length - len(ci)) + [1] * len(ci)
        padded_ci = [tokenizer.pad_token_id] * (max_length - len(ci)) + ci
        offset_tgt_tokens = [p + (max_length - len(ci)) for p in tgt_tokens]

        all_attention_masks.append(attention_masks)
        all_focus_tokens.append(focus_tokens)
        all_tgt_tokens.append(offset_tgt_tokens)
        new_cis.append(padded_ci)

    # keep all tokens except the last token due to offset
    keep_pos = []
    for i in range(max_length - 1):
        keep_pos.append(i)

    if verbose:
        print(keep_pos)
        print(all_attention_masks)
        print(all_focus_tokens)
        print(all_tgt_tokens)

    # compute focus probs by running a forward pass on the padded sequences
    all_focus_probs: list[list[float]] = []
    device = next(model.parameters()).device
    input_ids = torch.tensor(new_cis, device=device)
    attn_mask = torch.tensor(all_attention_masks, device=device)
    with torch.no_grad():
        logits = model(input_ids, attention_mask=attn_mask).logits
    for batch_i in range(len(new_cis)):
        probs_for_ci: list[float] = []
        for tgt_pos, focus_tok in zip(
            all_tgt_tokens[batch_i], all_focus_tokens[batch_i]
        ):
            token_probs = torch.softmax(logits[batch_i, tgt_pos], dim=-1)
            probs_for_ci.append(token_probs[focus_tok].item())
        all_focus_probs.append(probs_for_ci)

    # the reason why we return all_tgt_tokens[0] is because tgt token positions are the same for all
    return (
        new_cis,
        all_attention_masks,
        all_focus_tokens,
        all_focus_probs,
        all_tgt_tokens[0],
        keep_pos,
        starts,
    )


def trace_teacher_forced_response(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    response: str,
    target_response_positions: list[int],
    config: ADAGConfig,
    *,
    label: str = "teacher_forced",
    system_prompt: str | None = None,
    ignore_bos: bool = False,
    benchmark_only: bool = False,
    instrumentation: TraceInstrumentation | None = None,
) -> CircuitData:
    """Trace selected tokens from one frozen response without generation.

    A one-target trace is a reusable scientific artifact. Multiple targets are
    intentionally allowed only for explicit systems benchmarks: the current
    dataframe conversion aggregates the target axis and therefore must not be
    treated as a target-resolved scientific result.
    """
    if len(target_response_positions) != 1 and not benchmark_only:
        raise ValueError(
            "multi-target traces currently aggregate the target axis; pass "
            "benchmark_only=True for performance measurements"
        )

    with instrumentation_stage(instrumentation, "prepare_input"):
        prepared = prepare_teacher_forced_input(
            tokenizer,
            prompt,
            response,
            target_response_positions,
            system_prompt=system_prompt,
        )
    with instrumentation_stage(instrumentation, "target_scoring"):
        target_logit_values, target_probs = _teacher_forced_target_scores(
            model, prepared
        )

    with instrumentation_stage(instrumentation, "clja_total"):
        nodes, edges = get_all_pairs_cl_ja_effects_with_attributions(
            model=model,
            tokenizer=tokenizer,
            cis=[prepared.input_ids],
            config=config,
            attention_masks=[prepared.attention_mask],
            focus_logits=[prepared.target_token_ids],
            src_tokens=list(range(len(prepared.input_ids) - 1)),
            tgt_tokens=prepared.target_prediction_positions,
            instrumentation=instrumentation,
        )
    if instrumentation is not None:
        instrumentation.set_counter("raw_node_count", len(nodes))
        instrumentation.set_counter("raw_edge_count", len(edges))
    with instrumentation_stage(instrumentation, "dataframe_conversion"):
        df_node, df_edge = convert_circuit_to_dataframes(
            [nodes],
            [edges],
            [label],
            [[0]],
            bs=1,
            ignore_bos=ignore_bos,
            percentage_threshold=config.percentage_threshold,
        )
    if instrumentation is not None:
        instrumentation.set_counter("final_dataframe_node_count", len(df_node))
        instrumentation.set_counter("final_dataframe_edge_count", len(df_edge))

    target_provenance = _teacher_forced_target_provenance(
        prepared, target_logit_values, target_probs
    )

    trace_metadata: dict[str, object] = {
        "trace_mode": "teacher_forced_response",
        "prompt": prompt,
        "prompt_sha256": _stable_text_hash(prompt),
        "response": response,
        "response_sha256": _stable_text_hash(response),
        "system_prompt": system_prompt,
        "system_prompt_sha256": _stable_text_hash(system_prompt),
        "assistant_prefix_token_count": prepared.assistant_prefix_token_count,
        "response_token_count": prepared.response_token_count,
        "included_response_token_count": prepared.included_response_token_count,
        "input_token_count": len(prepared.input_ids),
        "chat_template_sha256": _stable_text_hash(get_chat_template(tokenizer)),
    }
    if instrumentation is not None:
        trace_metadata["instrumentation"] = instrumentation.snapshot()
    return CircuitData(
        df_node=df_node,
        df_edge=df_edge,
        cis=[prepared.input_ids],
        attention_masks=[prepared.attention_mask],
        labels=[label],
        target_logits=[prepared.target_token_ids],
        target_logit_probs=[target_probs],
        target_logit_values=[target_logit_values],
        target_provenance=target_provenance,
        trace_metadata=trace_metadata,
        benchmark_only=benchmark_only,
        k=len(prepared.target_token_ids),
        config=config,
        model_id=model.config._name_or_path,
    )


def trace_teacher_forced_candidates(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    response: str,
    target_response_position: int,
    config: ADAGConfig,
    *,
    candidate_policy_id: CandidatePolicyId,
    candidate_count: int,
    specified_candidate_token_id: int | None = None,
    joint_objective_id: JointObjectiveId = "raw_logit_sum",
    trace_family_id: str = TOPK_TRACE_FAMILY_ID,
    label: str = "teacher_forced_topk",
    system_prompt: str | None = None,
    ignore_bos: bool = False,
    instrumentation: TraceInstrumentation | None = None,
) -> TopKPositionTrace:
    """Trace several candidate logits at one teacher-forced response position.

    The response target remains singular. Candidate logits form a separate
    output-contribution axis and share one prediction position.
    """

    if config.center_logits:
        raise ValueError(
            "candidate tracing requires an objective with explicit centering; "
            "ADAGConfig.center_logits is not a named candidate objective"
        )
    if not isinstance(trace_family_id, str) or not trace_family_id:
        raise ValueError("trace_family_id must be a non-empty string")

    with instrumentation_stage(instrumentation, "prepare_input"):
        prepared = prepare_teacher_forced_input(
            tokenizer,
            prompt,
            response,
            [target_response_position],
            system_prompt=system_prompt,
        )
    with instrumentation_stage(instrumentation, "candidate_scoring"):
        position_logits = _teacher_forced_position_logits(model, prepared)
        selection = select_candidate_logits(
            position_logits,
            observed_token_id=prepared.target_token_ids[0],
            policy_id=candidate_policy_id,
            candidate_count=candidate_count,
            decode_token=lambda token_id: tokenizer.decode([token_id]),
            specified_token_id=specified_candidate_token_id,
        )
        objective = build_joint_objective(joint_objective_id, selection.candidates)
        realized_candidate_count = len(selection.candidates)
        observed_token_id = prepared.target_token_ids[0]
        observed_logit = float(
            position_logits[observed_token_id].detach().float().cpu().item()
        )
        observed_probability = float(
            torch.softmax(position_logits.float(), dim=-1)[observed_token_id]
            .detach()
            .cpu()
            .item()
        )

    candidate_axis = CandidateLogitAxis(
        prediction_position=prepared.target_prediction_positions[0],
        token_ids_by_batch=(
            tuple(candidate.token_id for candidate in selection.candidates),
        ),
        objective_weights=objective.candidate_weights,
        use_absolute_goal_for_percentage_threshold=(
            objective.percentage_threshold_reference
            == "absolute_joint_objective_magnitude"
        ),
    )
    with instrumentation_stage(instrumentation, "clja_total"):
        nodes, edges = get_all_pairs_cl_ja_effects_with_attributions(
            model=model,
            tokenizer=tokenizer,
            cis=[prepared.input_ids],
            config=config,
            attention_masks=[prepared.attention_mask],
            candidate_axis=candidate_axis,
            src_tokens=list(range(len(prepared.input_ids) - 1)),
            tgt_tokens=[prepared.target_prediction_positions[0]],
            instrumentation=instrumentation,
        )
    if instrumentation is not None:
        instrumentation.set_counter("raw_node_count", len(nodes))
        instrumentation.set_counter("raw_edge_count", len(edges))
        instrumentation.set_counter("candidate_count", realized_candidate_count)
    with instrumentation_stage(instrumentation, "dataframe_conversion"):
        df_node, df_edge = convert_circuit_to_dataframes(
            [nodes],
            [edges],
            [label],
            [[0]],
            bs=1,
            ignore_bos=ignore_bos,
            percentage_threshold=config.percentage_threshold,
        )
    if instrumentation is not None:
        instrumentation.set_counter("final_dataframe_node_count", len(df_node))
        instrumentation.set_counter("final_dataframe_edge_count", len(df_edge))

    target_provenance = _teacher_forced_target_provenance(
        prepared, [observed_logit], [observed_probability]
    )
    contribution_schema: dict[str, object] = {
        "schema_id": TOPK_CONTRIBUTION_SCHEMA_ID,
        "axis": "candidate_index",
        "width": realized_candidate_count,
        "semantics": "gradient_times_activation_for_each_raw_candidate_logit",
        "scalar_graph_attribution_semantics": "named_joint_objective",
    }
    contract = {
        "trace_family_id": trace_family_id,
        "shared_response_position": target_response_position,
        "shared_prediction_position": prepared.target_prediction_positions[0],
        "candidate_count": realized_candidate_count,
        "candidate_selection": selection.to_dict(),
        "joint_objective": objective.to_dict(),
        "candidate_contribution_schema": contribution_schema,
    }
    trace_metadata: dict[str, object] = {
        "trace_mode": "teacher_forced_topk_position",
        "prompt": prompt,
        "prompt_sha256": _stable_text_hash(prompt),
        "response": response,
        "response_sha256": _stable_text_hash(response),
        "system_prompt": system_prompt,
        "system_prompt_sha256": _stable_text_hash(system_prompt),
        "assistant_prefix_token_count": prepared.assistant_prefix_token_count,
        "response_token_count": prepared.response_token_count,
        "included_response_token_count": prepared.included_response_token_count,
        "input_token_count": len(prepared.input_ids),
        "chat_template_sha256": _stable_text_hash(get_chat_template(tokenizer)),
        "candidate_trace_contract": contract,
    }
    if instrumentation is not None:
        trace_metadata["instrumentation"] = instrumentation.snapshot()
    circuit_data = CircuitData(
        df_node=df_node,
        df_edge=df_edge,
        cis=[prepared.input_ids],
        attention_masks=[prepared.attention_mask],
        labels=[label],
        target_logits=[[observed_token_id]],
        target_logit_probs=[[observed_probability]],
        target_logit_values=[[observed_logit]],
        target_provenance=target_provenance,
        trace_metadata=trace_metadata,
        benchmark_only=False,
        k=1,
        config=config,
        model_id=model.config._name_or_path,
    )
    return TopKPositionTrace(
        circuit_data=circuit_data,
        trace_family_id=trace_family_id,
        shared_response_position=target_response_position,
        shared_prediction_position=prepared.target_prediction_positions[0],
        candidate_selection=selection,
        joint_objective=objective,
        candidate_contribution_schema=contribution_schema,
    )


def compute_circuits(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: list[str],
    config: ADAGConfig,
    seed_responses: list[str] | None = None,
    k: int = 1,
    bs: int = 4,
    max_new_tokens: int = 1,
    use_rollout: bool = False,
    system_prompt: str | None = None,
    true_answers: list[str] | None = None,
):
    """
    Compute CLSO graphs for all datapoints in a list of prompts, batched.
    """
    # set up data
    prompts = prompts if isinstance(prompts, list) else [prompts]
    if seed_responses is None:
        seed_responses = [None] * len(prompts)
    seed_responses = (
        seed_responses if isinstance(seed_responses, list) else [seed_responses]
    )

    # storage
    all_nodes, all_edges, all_labels, all_focus, all_starts = [], [], [], [], []
    all_cis, all_attention_masks = [], []
    all_focus_tokens: list[list[int]] = []
    all_focus_probs: list[list[float]] = []

    for i in tqdm(range(0, len(prompts), bs), desc="Processing batches"):
        if use_rollout:
            (
                cis,
                attention_masks,
                focus_tokens,
                focus_probs,
                tgt_tokens,
                keep_pos,
                starts,
            ) = prepare_cis_with_rollout(
                model,
                tokenizer,
                prompts[i : i + bs],
                seed_responses[i : i + bs],
                max_new_tokens=max_new_tokens,
                verbose=config.verbose,
            )
        else:
            cis, attention_masks, focus_tokens, focus_probs, keep_pos, starts = (
                prepare_cis(
                    model,
                    tokenizer,
                    prompts[i : i + bs],
                    seed_responses[i : i + bs],
                    k=k,
                    system_prompt=system_prompt,
                    true_answers=true_answers,
                    verbose=config.verbose,
                )
            )
        nodes, edges = get_all_pairs_cl_ja_effects_with_attributions(
            model=model,
            tokenizer=tokenizer,
            cis=cis,
            config=config,
            attention_masks=attention_masks,
            focus_logits=focus_tokens,
            src_tokens=keep_pos,
            tgt_tokens=(
                [max(keep_pos) for _ in range(k)] if not use_rollout else tgt_tokens
            ),
        )
        all_nodes.append(nodes)
        all_edges.append(edges)
        all_focus.append([_ for _ in range(len(focus_tokens))])
        all_starts.append(starts)
        all_focus_tokens.extend(focus_tokens)
        all_focus_probs.extend(focus_probs)
        all_cis.extend(cis)
        all_attention_masks.extend(attention_masks)
        if config.verbose:
            print("focus_tokens:", focus_tokens)
            print("starts:", starts)

    return (
        all_nodes,
        all_edges,
        all_labels,
        all_focus,
        all_starts,
        all_cis,
        all_attention_masks,
        all_focus_tokens,
        all_focus_probs,
    )


def compute_cohens_d_loo(vals_x: list[float], all_vals: list[float]) -> float:
    # vals_y is all_vals without vals_x
    vals_y = all_vals[::]
    for val in vals_x:
        vals_y.remove(val)

    std_x = np.std(vals_x, ddof=1) if len(vals_x) > 1 else 0
    std_y = np.std(vals_y, ddof=1) if len(vals_y) > 1 else 0
    s = (
        np.sqrt(
            ((len(vals_x) - 1) * std_x + (len(vals_y) - 1) * std_y)
            / (len(all_vals) - 2)
        )
        if len(all_vals) > 2
        else 0
    )
    return (np.mean(vals_x) - np.mean(vals_y)) / s if s != 0 else 0


def convert_circuit_to_dataframes(
    nodes: list[list[Node]],
    edges: list[list[Edge]],
    labels: list[str],
    starts: list[list[int]],
    bs: int = 4,
    ignore_bos: bool = False,
    percentage_threshold: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process CLSO graph data into a clean dataframe.
    """
    dfs_node, dfs_edge = [], []
    for batch_idx in range(len(nodes)):
        actual_bs = len(starts[batch_idx])
        for idx in range(actual_bs):
            start = starts[batch_idx][idx] + (1 if ignore_bos else 0)

            def extract_map(
                map_tensor: torch.Tensor | None, start_idx: int, is_attr: bool
            ) -> list[float] | None:
                """Extract raw attr_map/contrib_map values (no normalization)."""
                if map_tensor is None:
                    return None
                if is_attr:
                    return map_tensor[idx, start_idx:].tolist()
                else:
                    return map_tensor[idx].tolist()

            d = [
                (
                    node.layer,
                    node.token,
                    node.neuron,
                    node.final_attribution[idx].sum().item(),
                    node.activation[idx].item(),
                    extract_map(node.attr_map, start, is_attr=True),
                    extract_map(node.contrib_map, start, is_attr=False),
                )
                for node in nodes[batch_idx]
                if node.token >= start
            ]
            df_node = pd.DataFrame(
                d,
                columns=[
                    "layer",
                    "token",
                    "neuron",
                    "attribution",
                    "activation",
                    "attr_map",
                    "contrib_map",
                ],
            ).assign(label=labels[batch_idx * bs + idx] + f"___{batch_idx * bs + idx}")
            d = [
                (
                    f"{edge.src.layer}->{edge.tgt.layer}",
                    f"{edge.src.token}->{edge.tgt.token}",
                    f"{edge.src.neuron}->{edge.tgt.neuron}",
                    edge.final_attribution[idx].sum().item(),
                    edge.weight[idx].item(),
                )
                for edge in edges[batch_idx]
                if edge.src.token >= start and edge.tgt.token >= start
            ]
            df_edge = pd.DataFrame(
                d, columns=["layer", "token", "neuron", "attribution", "weight"]
            ).assign(label=labels[batch_idx * bs + idx] + f"___{batch_idx * bs + idx}")

            # normalize attribution by sum of goals
            total_last_layer_attribution = df_node[
                df_node.layer == df_node.layer.max()
            ].attribution.sum()
            df_node.loc[:, "attribution"] = (
                df_node.loc[:, "attribution"] / total_last_layer_attribution
            )
            df_edge.loc[:, "attribution"] = (
                df_edge.loc[:, "attribution"] / total_last_layer_attribution
            )

            # add to df
            dfs_node.append(df_node)
            dfs_edge.append(df_edge)

    # merge dfs
    df_node = pd.concat(dfs_node)
    df_edge = pd.concat(dfs_edge)
    # drop nodes below threshold per-item (with batching, neurons important in
    # any batch item get traced for all items, so we need per-item filtering here)
    if percentage_threshold is not None:
        # Only apply threshold to MLP neurons, not embedding (layer -1) or final layer nodes
        max_layer = df_node["layer"].max()
        is_exempt = (df_node["layer"] < 0) | (df_node["layer"] == max_layer)
        df_node = df_node[
            is_exempt | (df_node.attribution.abs() >= percentage_threshold)
        ]
    else:
        df_node = df_node[df_node.attribution != 0]
    df_edge = df_edge[df_edge.attribution != 0].dropna(subset=["attribution"])

    # prune edges whose src or tgt node was removed
    surviving_nodes = set(
        zip(df_node["layer"], df_node["token"], df_node["neuron"], df_node["label"])
    )
    # edge columns are "src->tgt" strings, split to check membership
    edge_src = df_edge["layer"].str.split("->").str[0].astype(int)
    edge_src_tok = df_edge["token"].str.split("->").str[0].astype(int)
    edge_src_neu = df_edge["neuron"].str.split("->").str[0].astype(int)
    edge_tgt = df_edge["layer"].str.split("->").str[1].astype(int)
    edge_tgt_tok = df_edge["token"].str.split("->").str[1].astype(int)
    edge_tgt_neu = df_edge["neuron"].str.split("->").str[1].astype(int)
    edge_label = df_edge["label"]
    src_alive = pd.Series(
        [
            (l, t, n, lb) in surviving_nodes
            for l, t, n, lb in zip(edge_src, edge_src_tok, edge_src_neu, edge_label)
        ],
        index=df_edge.index,
    )
    tgt_alive = pd.Series(
        [
            (l, t, n, lb) in surviving_nodes
            for l, t, n, lb in zip(edge_tgt, edge_tgt_tok, edge_tgt_neu, edge_label)
        ],
        index=df_edge.index,
    )
    df_edge = df_edge[src_alive & tgt_alive]
    return df_node, df_edge


def convert_inputs_to_circuits(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: list[str],
    config: ADAGConfig,
    seed_responses: list[str] | None = None,
    labels: list[str] | None = None,
    num_datapoints: int | None = None,
    batch_size: int = 4,
    max_new_tokens: int = 1,
    k: int = 1,
    # TODO: topk_logits: int = 1,
    ignore_bos: bool = False,
    system_prompt: str | None = None,
    use_rollout: bool = False,
    true_answers: list[str] | None = None,
) -> CircuitData:
    """
    Convert a list of prompts and seed responses into a CircuitData artifact.
    """
    # num datapoints
    if num_datapoints is None:
        num_datapoints = len(prompts)
        assert len(prompts) == len(labels) == len(seed_responses)
    else:
        assert len(prompts) >= num_datapoints
        assert len(labels) >= num_datapoints
        assert len(seed_responses) >= num_datapoints

    # grab inputs
    prompts = prompts[:num_datapoints]
    labels = labels[:num_datapoints]
    if seed_responses is not None and not use_rollout:
        seed_responses = seed_responses[:num_datapoints]

    print("Prompt:", prompts[0])
    if seed_responses is not None and not use_rollout:
        print("Seed response:", seed_responses[0].replace(" ", "_"))
    print("Number of datapoints:", len(prompts))

    # compute circuits
    nodes, edges, _, focus, starts, cis, attention_masks, focus_tokens, focus_probs = (
        compute_circuits(
            model,
            tokenizer,
            prompts,
            config=config,
            seed_responses=seed_responses,
            k=k,
            bs=batch_size,
            max_new_tokens=max_new_tokens,
            use_rollout=use_rollout,
            system_prompt=system_prompt,
            true_answers=true_answers,
        )
    )

    # convert to dataframes
    df_node, df_edge = convert_circuit_to_dataframes(
        nodes,
        edges,
        labels,
        starts,
        bs=batch_size,
        ignore_bos=ignore_bos,
        percentage_threshold=config.percentage_threshold,
    )

    return CircuitData(
        df_node=df_node,
        df_edge=df_edge,
        cis=cis,
        attention_masks=attention_masks,
        labels=labels,
        target_logits=focus_tokens,
        target_logit_probs=focus_probs,
        k=k,
        config=config,
        model_id=model.config._name_or_path,
    )
