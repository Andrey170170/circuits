"""Freeze and strictly validate one nearest-above-10k tracing target."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts.bonafide.build_process_witness_annotations import SYSTEM_PROMPT

from circuits.analysis.bonafide.canonical import canonical_sha256, file_sha256
from circuits.analysis.bonafide.coarse_sampling_post_campaign_v1 import (
    _publish_no_replace,
)
from circuits.analysis.bonafide.coarse_sampling_post_campaign_v2 import (
    load_frozen_post_campaign_sampling_v2,
)
from circuits.analysis.bonafide.process_witness_resource_calibration_v1 import (
    EXPECTED_GENERATION_CENSUS,
    EXPECTED_NON_GENERATION_RESPONSES,
    EXPECTED_SAMPLING_INVENTORY_SHA256,
    EXPECTED_SAMPLING_MANIFEST_SHA256,
    EXPECTED_SOURCE_LITERAL_CENSUS,
    MODEL_ID,
    MODEL_REVISION,
    _artifact_id,
    _candidate_union,
    _exact_runtime_tokenization,
    _hash_text,
    _iter_gzip_jsonl,
    _load_default_tokenizer,
    _load_object,
    _readonly_tree,
    _reject_symlinks,
    _verify_self_hash,
    _write_json,
    _write_jsonl,
)
from circuits.tracing.trace import get_chat_template

SCHEMA_VERSION = "adag.process-witness.near-10k-qualification.v1"
INVENTORY_SCHEMA_VERSION = "adag.process-witness.resource-calibration-inventory.v1"
SELECTION_SCHEMA_VERSION = "adag.process-witness.near-10k-selection.v1"
TRACE_PHASE = "process_witness_resource_calibration_v1"
TRACE_WAVE_ID = "context-gt-10000"
CONTEXT_THRESHOLD_EXCLUSIVE = 10_000
SOURCE_POLICY = "balanced"
SOURCE_BUDGET = 40_000
EXPECTED_SELECTED_CONTEXT = 10_006
EXPECTED_SELECTED_TARGET_ID = "pwcoarsetargetv2-c3452565e87058f965f63871e4157bc7"
CLAIM_BOUNDARY = (
    "This immutable single-target qualification measures strict T5 tracing runtime "
    "and resource feasibility immediately above 10,000 rendered context tokens. It "
    "does not select a production corpus, inspect graph semantics, test ADAG "
    "adequacy, identify a motif or witness, or support a faithfulness claim."
)

# This is deliberately independent from resource_calibration_v1.EXECUTION_SOURCE_PATHS:
# extending that historical tuple would invalidate already-frozen ladder bundles.
EXECUTION_SOURCE_PATHS = (
    "circuits/analysis/bonafide/process_witness_near_10k_qualification_v1.py",
    "scripts/bonafide/build_process_witness_near_10k_qualification_v1.py",
    "circuits/analysis/bonafide/process_witness_resource_calibration_v1.py",
    "circuits/analysis/bonafide/coarse_sampling_post_campaign_v2.py",
    "circuits/analysis/bonafide/coarse_sampling_post_campaign_v1.py",
    "circuits/analysis/bonafide/coarse_sampling_openai_batch_production_v1.py",
    "circuits/analysis/bonafide/canonical.py",
    "circuits/labeling/io.py",
    "circuits/tracing/artifact.py",
    "circuits/tracing/backend_qualification.py",
    "circuits/tracing/clja.py",
    "circuits/tracing/trace.py",
    "scripts/bonafide/cuda_allocator_policy.py",
    "scripts/bonafide/cuda_headroom.py",
    "scripts/bonafide/process_witness_attention_backend_qualification_v1.sbatch",
    "scripts/bonafide/runner.py",
    "scripts/bonafide/topk_manifest.py",
    "scripts/bonafide/topk_runner.py",
    "pyproject.toml",
    "uv.lock",
)


def _bundle_inventory(root: Path) -> dict[str, Any]:
    """Hash every payload file except the two root inventory control files."""

    excluded = {"manifest.json", "inventory.json"}
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    core = {"schema_version": INVENTORY_SCHEMA_VERSION, "files": files}
    return {**core, "inventory_sha256": canonical_sha256(core)}


def select_nearest_strictly_above(
    candidates: Sequence[Mapping[str, Any]],
    *,
    exact_tokenizations: Mapping[str, tuple[list[int], list[int]]],
    threshold_exclusive: int = CONTEXT_THRESHOLD_EXCLUSIVE,
) -> dict[str, Any]:
    """Select the nearest exact-tokenized balanced/40k target above a threshold."""

    if isinstance(threshold_exclusive, bool) or not isinstance(
        threshold_exclusive, int
    ):
        raise ValueError("near-10k threshold must be an integer")
    required_membership = {"policy": SOURCE_POLICY, "budget": SOURCE_BUDGET}
    eligible = []
    for raw in candidates:
        row = dict(raw)
        response_id = str(row.get("response_id", ""))
        tokenization = exact_tokenizations.get(response_id)
        if tokenization is None or required_membership not in row.get(
            "policy_memberships", []
        ):
            continue
        prefix_ids, response_ids = tokenization
        position = int(row.get("token_index", -1))
        context = int(row.get("rendered_total_context_token_count", -1))
        if not 0 <= position < len(response_ids):
            raise ValueError("near-10k candidate response position drift")
        if context != len(prefix_ids) + position + 1:
            raise ValueError("near-10k candidate exact context-count drift")
        if context > threshold_exclusive:
            eligible.append(row)
    if not eligible:
        raise ValueError("no exact-tokenized balanced/40k target above threshold")
    return min(
        eligible,
        key=lambda row: (
            int(row["rendered_total_context_token_count"]),
            str(row["target_id"]),
        ),
    )


def _run_config() -> dict[str, Any]:
    """Return the live-qualified compact/expandable near-capacity configuration."""

    return {
        "schema_version": "bonafide-trace-run-config/v1",
        "artifact_root": "results/bonafide/process-witness-near-10k-qualification-v1",
        "batch_size": 1,
        "continue_on_error": False,
        "cuda_allocator_policy": "expandable_segments_v1",
        "instrumentation": {
            "cuda_allocator_snapshot_telemetry": True,
            "cuda_dense_joint_pressure_telemetry": True,
            "cuda_memory_telemetry": True,
        },
        "trace_warmup": {
            "enabled": False,
            "mode": "first_wave_item_full_trace_discard",
            "wave_id_prefixes": [],
        },
        "wave_limits": {
            "cuda_headroom_action": "warn",
            "cuda_headroom_policy": "allocator_dense_joint_v1",
            "max_trace_seconds": 1800,
            "min_cuda_headroom_bytes": 8_589_934_592,
            "stop_on_oom": True,
        },
        "model": {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "device": "cuda:0",
            "dtype": "bfloat16",
            "local_files_only": True,
            "local_snapshot_path": (
                "${HF_HUB_CACHE}/models--Qwen--Qwen3-4B-Thinking-2507/"
                f"snapshots/{MODEL_REVISION}"
            ),
            "from_pretrained_kwargs": {},
        },
        "adag_config": {
            "verbose": False,
            "parent_threshold": None,
            "edge_threshold": 0.01,
            "node_attribution_threshold": None,
            "topk": None,
            "batch_aggregation": "any",
            "topk_neurons": None,
            "percentage_threshold": 0.005,
            "use_relp_grad": True,
            "disable_half_rule": False,
            "disable_stop_grad": False,
            "ablation_mode": "zero",
            "use_stop_grad_on_mlps": True,
            "return_nodes_only": False,
            "focus_last_residual": False,
            "skip_attr_contrib": False,
            "center_logits": False,
            "ig_steps": None,
            "ig_mode": "ig-inputs",
            "return_only_important_neurons": False,
            "apply_blacklist": True,
            "stop_gradient_attention_backend": "flash_sdpa_causal_v1",
            "stop_gradient_contribution_execution": "source_leaf_v1",
            "stop_gradient_contribution_target_lane_chunk_size": 1,
            "selected_neuron_contribution_target_lane_chunk_size": 1,
            "selected_embed_contribution_target_lane_chunk_size": 1,
            "stop_gradient_embed_contribution_target_lane_chunk_size": 1,
            "selected_attribution_neuron_lane_chunk_size": 1,
            "stop_gradient_selected_attribution_forward_execution": "prefix_stop_v1",
            "stop_gradient_selected_attribution_storage": "terminal_detached_v1",
            "selected_target_logit_execution": "selected_position_logits_v1",
            "post_selection_state_storage": "compact_cpu_v1",
            "embedding_edge_materialization": "vectorized_v1",
            "cross_layer_jacobian_execution": "cached_range_v1",
        },
    }


def _execution_source_revision(repo_root: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    status = git(
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        *EXECUTION_SOURCE_PATHS,
    )
    if status:
        raise ValueError("near-10k execution-source paths are not clean")
    files = []
    for relative in EXECUTION_SOURCE_PATHS:
        path = repo_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"near-10k execution source is absent: {path}")
        files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return {
        "repo_root": str(repo_root.resolve()),
        "git_commit": git("rev-parse", "HEAD"),
        "git_tree": git("rev-parse", "HEAD^{tree}"),
        "clean_scope": "execution_source_paths",
        "binding_scope": "near_10k_selection_validation_and_trace_execution",
        "files": files,
    }


def _load_source_state(
    *, sampling_v2_root: Path, tokenizer: Any, system_prompt: str
) -> dict[str, Any]:
    source = sampling_v2_root.resolve()
    validated = load_frozen_post_campaign_sampling_v2(source)
    source_manifest = validated["manifest"]
    if (
        source_manifest.get("manifest_sha256") != EXPECTED_SAMPLING_MANIFEST_SHA256
        or source_manifest.get("inventory_sha256") != EXPECTED_SAMPLING_INVENTORY_SHA256
    ):
        raise ValueError("sampling-v2 manifest/inventory source drift")
    context_binding = _load_object(source / "context-source-binding.json")
    if context_binding.get("literal_census") != EXPECTED_SOURCE_LITERAL_CENSUS:
        raise ValueError("sampling-v2 canonical literal census drift")

    parent_root = Path(str(source_manifest["parent_v1_root"])).resolve()
    workstation = _load_object(
        parent_root / "source-evidence/bundle/workstation-bundle.json"
    )
    documents = {str(row["response_id"]): row for row in workstation["documents"]}
    context_rows = {
        str(row["response_id"]): row
        for row in _iter_gzip_jsonl(source / "context-count-evidence.jsonl.gz")
    }
    if set(documents) != set(context_rows):
        raise ValueError("sampling-v2 context/workstation response census drift")
    exact: dict[str, tuple[list[int], list[int]]] = {}
    excluded: list[str] = []
    non_generation: list[str] = []
    for response_id in sorted(documents):
        if context_rows[response_id].get("source_kind") != (
            "generation_reproducibility_prompt_token_ids"
        ):
            non_generation.append(response_id)
            continue
        tokenized = _exact_runtime_tokenization(
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            document=documents[response_id],
            context_evidence=context_rows[response_id],
        )
        if tokenized is None:
            excluded.append(response_id)
        else:
            exact[response_id] = tokenized
    observed_generation_census = (
        len(documents) - len(non_generation),
        len(exact),
        len(excluded),
    )
    if (
        len(documents) != EXPECTED_SOURCE_LITERAL_CENSUS["responses"]
        or observed_generation_census != EXPECTED_GENERATION_CENSUS
        or len(non_generation) != EXPECTED_NON_GENERATION_RESPONSES
    ):
        raise ValueError("near-10k canonical runtime tokenization census drift")
    candidates = _candidate_union(
        _iter_gzip_jsonl(source / "realized-candidate-tiers.jsonl.gz")
    )
    selected = select_nearest_strictly_above(candidates, exact_tokenizations=exact)
    return {
        "source": source,
        "source_manifest": source_manifest,
        "documents": documents,
        "exact": exact,
        "excluded": excluded,
        "non_generation": non_generation,
        "selected": selected,
    }


def _selection_binding(
    *,
    selected: Mapping[str, Any],
    document: Mapping[str, Any],
    prefix_ids: list[int],
    response_ids: list[int],
) -> dict[str, Any]:
    target_id = str(selected["target_id"])
    prompt = str(document["task_context"]["prompt"])
    response = str(document["text"])
    position = int(selected["token_index"])
    context = int(selected["rendered_total_context_token_count"])
    identity = {
        "target_id": target_id,
        "response_id": str(selected["response_id"]),
        "psu_id": str(selected["psu_id"]),
        "unit_id": str(selected["unit_id"]),
        "token_index": position,
        "rendered_total_context_token_count": context,
        "balanced_40k_first_owner_mechanism": selected[
            "balanced_40k_first_owner_mechanism"
        ],
        "policy_memberships": selected["policy_memberships"],
    }
    return {
        **identity,
        "target_identity_sha256": canonical_sha256(identity),
        "target_id_utf8_sha256": _hash_text(target_id),
        "prompt_utf8_sha256": _hash_text(prompt),
        "response_utf8_sha256": _hash_text(response),
        "assistant_prefix_ids_sha256": canonical_sha256(prefix_ids),
        "response_ids_sha256": canonical_sha256(response_ids),
        "final_target_token_id": response_ids[position],
    }


def _frozen_payloads(
    *, state: Mapping[str, Any], tokenizer: Any, system_prompt: str
) -> dict[str, Any]:
    selected = state["selected"]
    response_id = str(selected["response_id"])
    document = state["documents"][response_id]
    prefix_ids, response_ids = state["exact"][response_id]
    position = int(selected["token_index"])
    context = int(selected["rendered_total_context_token_count"])
    if (
        not 0 <= position < len(response_ids)
        or context != len(prefix_ids) + position + 1
    ):
        raise ValueError("near-10k selected target token/context identity drift")
    binding = _selection_binding(
        selected=selected,
        document=document,
        prefix_ids=prefix_ids,
        response_ids=response_ids,
    )
    selection = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "wave_id": TRACE_WAVE_ID,
        "selection_basis": "nearest_exact_runtime_tokenized_balanced_40k_strictly_above_threshold_v1",
        "threshold_exclusive": CONTEXT_THRESHOLD_EXCLUSIVE,
        "ordering": [
            "rendered_total_context_token_count_ascending",
            "target_id_ascending",
        ],
        **binding,
    }
    item = {
        "artifact_id": _artifact_id(str(selected["target_id"]), context),
        "example": {
            "example_id": response_id,
            "annotation_row_ids": [],
            "prompt": document["task_context"]["prompt"],
            "response": document["text"],
            "system_prompt": system_prompt,
            "historical_replay_scope": "stored_assistant_serialization",
            "teacher_forced_serialization_mode": "historical_thinking_continuation",
            "token_identity": {
                "schema_version": "adag.teacher-forced-token-identity.v1",
                "hash_encoding": "sha256_utf8_canonical_json_integer_array_v1",
                "assistant_prefix_ids_sha256": canonical_sha256(prefix_ids),
                "response_ids_sha256": canonical_sha256(response_ids),
                "assistant_prefix_token_count": len(prefix_ids),
                "response_token_count": len(response_ids),
            },
        },
        "response_token_count": len(response_ids),
        "target_selection": {
            "kind": "explicit_response_positions",
            "response_token_positions": [position],
            "width": 1,
            "final_target_token_id": response_ids[position],
        },
        "objective": {
            "name": "sum_selected_logits",
            "benchmark_only_multi_target": False,
        },
        "resource_calibration": selection,
    }
    wave = {
        "wave_id": TRACE_WAVE_ID,
        "corpus_role": "resource_method_development",
        "purpose": "single-target nearest-above-10k strict-T5 qualification only",
        "resource_context_bin": {
            "lower_inclusive": CONTEXT_THRESHOLD_EXCLUSIVE + 1,
            "upper_inclusive": EXPECTED_SOURCE_LITERAL_CENSUS[
                "rendered_total_context_token_count_max"
            ],
            "distinct_prompt_block_count": 1,
        },
        "items": [item],
    }
    chat_template = get_chat_template(tokenizer)
    if not isinstance(chat_template, str) or not chat_template:
        raise ValueError("Thinking tokenizer chat template is missing")
    width1 = {
        "schema_version": "bonafide-trace-benchmark/v1",
        "dataset": {
            "path": str(state["source"]),
            "sha256": EXPECTED_SAMPLING_MANIFEST_SHA256,
        },
        "tokenizer": {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "chat_template_sha256": _hash_text(chat_template),
        },
        "execution_contract": {
            "trace_units_are_independent": True,
            "merge_graphs": False,
            "historical_replay_scope": "stored_assistant_serialization",
        },
        "waves": [wave],
    }
    return {"width1": width1, "wave": wave, "selection": selection}


def _trace_manifest(
    *, root: Path, payloads: Mapping[str, Any], system_prompt: str
) -> dict[str, Any]:
    """Construct the complete strict-T5 trace manifest behind one private seam."""

    width1_path = root / "width1-source-manifest.json"
    return {
        "schema_version": "bonafide-topk-trace-manifest/v1",
        "phase": TRACE_PHASE,
        "claim_boundary": CLAIM_BOUNDARY,
        "trace_family": {
            "trace_family_id": "bonafide.t5-upstream-summed-top5.v1",
            "candidate_policy_id": "model_top5",
            "candidate_policy_version": "1",
            "candidate_count": 5,
            "joint_objective_id": "raw_logit_sum",
            "joint_objective_version": "1",
        },
        "teacher_forcing_contract": {
            "serialization_mode": "historical_thinking_continuation",
            "system_prompt_sha256": _hash_text(system_prompt),
            "token_identity_schema_version": "adag.teacher-forced-token-identity.v1",
            "hash_encoding": "sha256_utf8_canonical_json_integer_array_v1",
        },
        "source": {
            "width1_manifest_path": str(width1_path.resolve()),
            "width1_manifest_sha256": file_sha256(width1_path),
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "tokenizer_revision": MODEL_REVISION,
            "chat_template_sha256": payloads["width1"]["tokenizer"][
                "chat_template_sha256"
            ],
        },
        "waves": [payloads["wave"]],
    }


def _runtime_census(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_responses": len(state["documents"]),
        "full_generation_responses": len(state["documents"])
        - len(state["non_generation"]),
        "exact_responses": len(state["exact"]),
        "excluded_responses": len(state["excluded"]),
        "excluded_response_ids": state["excluded"],
        "non_generation_responses": len(state["non_generation"]),
        "non_generation_response_ids": state["non_generation"],
    }


def _manifest_core(
    *,
    root: Path,
    state: Mapping[str, Any],
    selection: Mapping[str, Any],
    inventory: Mapping[str, Any],
    revision: Mapping[str, Any],
) -> dict[str, Any]:
    """Reconstruct every canonical outer-manifest field exactly."""

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "frozen_near_10k_qualification_not_launched",
        "claim_boundary": CLAIM_BOUNDARY,
        "sampling_v2_root": str(state["source"]),
        "sampling_v2_manifest_sha256": EXPECTED_SAMPLING_MANIFEST_SHA256,
        "sampling_v2_inventory_sha256": EXPECTED_SAMPLING_INVENTORY_SHA256,
        "source_literal_census": EXPECTED_SOURCE_LITERAL_CENSUS,
        "runtime_tokenization_census": _runtime_census(state),
        "selection_contract": {
            "policy": SOURCE_POLICY,
            "budget": SOURCE_BUDGET,
            "threshold_exclusive": CONTEXT_THRESHOLD_EXCLUSIVE,
            "exact_runtime_tokenization_required": True,
            "ordering": [
                "rendered_total_context_token_count_ascending",
                "target_id_ascending",
            ],
        },
        "selected_target": selection,
        "width1_source_manifest_sha256": file_sha256(
            root / "width1-source-manifest.json"
        ),
        "trace_manifest_sha256": file_sha256(root / "trace-manifest.json"),
        "run_config_sha256": file_sha256(root / "run-config.json"),
        "selection_sha256": file_sha256(root / "selection.jsonl"),
        "inventory_sha256": inventory["inventory_sha256"],
        "execution_source_revision": dict(revision),
        "selected_sampling_policy": None,
        "selected_trace_corpus": None,
        "semantic_graph_inspection_performed": False,
    }


def build_near_10k_qualification_v1(
    *,
    sampling_v2_root: Path,
    destination: Path,
    tokenizer: Any,
    system_prompt: str = SYSTEM_PROMPT,
) -> dict[str, Any]:
    """Build one immutable, deterministic, nearest-above-10k target bundle."""

    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"near-10k qualification destination exists: {destination}"
        )
    state = _load_source_state(
        sampling_v2_root=sampling_v2_root,
        tokenizer=tokenizer,
        system_prompt=system_prompt,
    )
    selected = state["selected"]
    if (
        int(selected["rendered_total_context_token_count"]) != EXPECTED_SELECTED_CONTEXT
        or selected["target_id"] != EXPECTED_SELECTED_TARGET_ID
    ):
        raise ValueError("canonical nearest-above-10k selected target drift")
    payloads = _frozen_payloads(
        state=state, tokenizer=tokenizer, system_prompt=system_prompt
    )
    temporary = destination.parent / f".{destination.name}.building-{os.getpid()}"
    if temporary.exists():
        raise FileExistsError(
            f"near-10k qualification staging root exists: {temporary}"
        )
    temporary.mkdir(parents=True)
    try:
        revision = _execution_source_revision(Path(__file__).resolve().parents[3])
        _write_json(temporary / "width1-source-manifest.json", payloads["width1"])
        trace = _trace_manifest(
            root=temporary, payloads=payloads, system_prompt=system_prompt
        )
        trace["source"]["width1_manifest_path"] = str(
            (destination / "width1-source-manifest.json").resolve()
        )
        _write_json(temporary / "trace-manifest.json", trace)
        _write_json(temporary / "run-config.json", _run_config())
        _write_jsonl(temporary / "selection.jsonl", [payloads["selection"]])
        inventory = _bundle_inventory(temporary)
        _write_json(temporary / "inventory.json", inventory)
        core = _manifest_core(
            root=temporary,
            state=state,
            selection=payloads["selection"],
            inventory=inventory,
            revision=revision,
        )
        manifest = {**core, "manifest_sha256": canonical_sha256(core)}
        _write_json(temporary / "manifest.json", manifest)
        _readonly_tree(temporary)
        _publish_no_replace(temporary, destination)
        return manifest
    except BaseException:
        if temporary.exists():
            temporary.chmod(0o755)
            for path in temporary.rglob("*"):
                path.chmod(0o755 if path.is_dir() else 0o644)
            shutil.rmtree(temporary)
        raise


def _load_selection_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("near-10k selection row drift")
            rows.append(value)
    return rows


def load_frozen_near_10k_qualification_v1(
    root: Path, *, tokenizer: Any | None = None, system_prompt: str = SYSTEM_PROMPT
) -> dict[str, Any]:
    """Strictly reload the bundle, rerun selection, and reject any drift."""

    _reject_symlinks(root)
    root = root.resolve()
    manifest = _load_object(root / "manifest.json")
    _verify_self_hash(manifest, "manifest_sha256", "near-10k qualification manifest")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != "frozen_near_10k_qualification_not_launched"
        or manifest.get("claim_boundary") != CLAIM_BOUNDARY
        or manifest.get("sampling_v2_manifest_sha256")
        != EXPECTED_SAMPLING_MANIFEST_SHA256
        or manifest.get("sampling_v2_inventory_sha256")
        != EXPECTED_SAMPLING_INVENTORY_SHA256
        or manifest.get("source_literal_census") != EXPECTED_SOURCE_LITERAL_CENSUS
        or manifest.get("selected_sampling_policy") is not None
        or manifest.get("selected_trace_corpus") is not None
        or manifest.get("semantic_graph_inspection_performed") is not False
    ):
        raise ValueError("near-10k qualification claim/status/source drift")
    expected_contract = {
        "policy": SOURCE_POLICY,
        "budget": SOURCE_BUDGET,
        "threshold_exclusive": CONTEXT_THRESHOLD_EXCLUSIVE,
        "exact_runtime_tokenization_required": True,
        "ordering": [
            "rendered_total_context_token_count_ascending",
            "target_id_ascending",
        ],
    }
    if manifest.get("selection_contract") != expected_contract:
        raise ValueError("near-10k selection contract drift")

    inventory = _load_object(root / "inventory.json")
    _verify_self_hash(inventory, "inventory_sha256", "near-10k inventory")
    if inventory.get("schema_version") != INVENTORY_SCHEMA_VERSION or inventory.get(
        "inventory_sha256"
    ) != manifest.get("inventory_sha256"):
        raise ValueError("near-10k inventory binding drift")
    if inventory != _bundle_inventory(root):
        raise ValueError("near-10k canonical inventory drift")
    if root.stat().st_mode & 0o777 != 0o555:
        raise ValueError("near-10k root mode drift")
    for path in root.rglob("*"):
        expected_mode = 0o555 if path.is_dir() else 0o444
        if path.stat().st_mode & 0o777 != expected_mode:
            raise ValueError(f"near-10k mode drift: {path.relative_to(root)}")

    revision = manifest.get("execution_source_revision")
    if not isinstance(revision, Mapping):
        raise ValueError("near-10k execution source drift")
    repo_root = Path(str(revision.get("repo_root", ""))).resolve()
    if _execution_source_revision(repo_root) != revision:
        raise ValueError("near-10k execution source drift")
    run_config = _load_object(root / "run-config.json")
    if run_config != _run_config() or file_sha256(
        root / "run-config.json"
    ) != manifest.get("run_config_sha256"):
        raise ValueError("near-10k run-config drift")
    if tokenizer is None:
        tokenizer = _load_default_tokenizer(run_config)
    state = _load_source_state(
        sampling_v2_root=Path(str(manifest["sampling_v2_root"])),
        tokenizer=tokenizer,
        system_prompt=system_prompt,
    )
    if manifest.get("runtime_tokenization_census") != _runtime_census(state):
        raise ValueError("near-10k runtime tokenization census drift")
    payloads = _frozen_payloads(
        state=state, tokenizer=tokenizer, system_prompt=system_prompt
    )
    if manifest.get("selected_target") != payloads["selection"]:
        raise ValueError("near-10k selected target drift")
    if (
        payloads["selection"]["rendered_total_context_token_count"]
        != EXPECTED_SELECTED_CONTEXT
        or payloads["selection"]["target_id"] != EXPECTED_SELECTED_TARGET_ID
    ):
        raise ValueError("canonical nearest-above-10k selected target drift")
    selection_rows = _load_selection_rows(root / "selection.jsonl")
    if selection_rows != [payloads["selection"]] or file_sha256(
        root / "selection.jsonl"
    ) != manifest.get("selection_sha256"):
        raise ValueError("near-10k frozen selection drift")
    width1 = _load_object(root / "width1-source-manifest.json")
    trace = _load_object(root / "trace-manifest.json")
    expected_trace = _trace_manifest(
        root=root, payloads=payloads, system_prompt=system_prompt
    )
    if (
        width1 != payloads["width1"]
        or trace != expected_trace
        or file_sha256(root / "width1-source-manifest.json")
        != manifest.get("width1_source_manifest_sha256")
        or file_sha256(root / "trace-manifest.json")
        != manifest.get("trace_manifest_sha256")
    ):
        raise ValueError("near-10k trace/source manifest drift")
    from scripts.bonafide.topk_manifest import validate_topk_manifest

    validate_topk_manifest(trace)
    expected_core = _manifest_core(
        root=root,
        state=state,
        selection=payloads["selection"],
        inventory=inventory,
        revision=revision,
    )
    expected_manifest = {
        **expected_core,
        "manifest_sha256": canonical_sha256(expected_core),
    }
    if manifest != expected_manifest:
        raise ValueError("near-10k canonical manifest drift")
    return {
        "manifest": manifest,
        "trace_manifest": trace,
        "run_config": run_config,
        "selected": state["selected"],
    }


__all__ = [
    "CONTEXT_THRESHOLD_EXCLUSIVE",
    "EXPECTED_SELECTED_CONTEXT",
    "EXPECTED_SELECTED_TARGET_ID",
    "TRACE_WAVE_ID",
    "_load_default_tokenizer",
    "_run_config",
    "build_near_10k_qualification_v1",
    "load_frozen_near_10k_qualification_v1",
    "select_nearest_strictly_above",
]
