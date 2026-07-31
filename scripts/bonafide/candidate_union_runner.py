"""Run candidate-specific fixed-topology rescoring and assemble union artifacts."""

from __future__ import annotations

import argparse
import gc
import json
import signal
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from circuits.tracing.artifact import (
    load_topk_compact_trace,
    save_topk_compact_trace,
    validate_topk_compact_trace_integrity,
)
from circuits.tracing.candidate_union import (
    CANDIDATE_UNION_TRACE_FAMILY_ID,
    assemble_candidate_union,
    frozen_union_topologies,
    load_candidate_union_artifact,
    save_candidate_union_artifact,
)
from circuits.tracing.clja import ADAGConfig
from circuits.tracing.instrumentation import TraceInstrumentation
from circuits.tracing.trace import trace_teacher_forced_candidates
from scripts.bonafide.runner import (
    _append_jsonl,
    _directory_size,
    _ensure_execution_cohort,
    _gpu_info,
    _load_model_and_tokenizer,
    _rss_peak_bytes,
    _sha256,
    collect_code_revision,
    collect_runtime_environment,
    load_json,
    validate_run_config,
    validate_runtime_topk_trace_against_item,
)
from scripts.bonafide.topk_runner import _model_config_sha256

PLAN_SCHEMA_VERSION = "bonafide-candidate-union-plan/v1"
REFINEMENT_TRACE_FAMILY_ID = "bonafide.candidate-union-refinement.v1"
REFINEMENT_TRACE_FAMILY = {
    "trace_family_id": REFINEMENT_TRACE_FAMILY_ID,
    "candidate_policy_id": "specified_token",
    "candidate_policy_version": "1",
    "joint_objective_id": "raw_logit_sum",
    "joint_objective_version": "1",
    "candidate_count": 1,
}


def validate_candidate_union_plan(plan: Mapping[str, Any]) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported candidate-union plan schema")
    source = plan.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("candidate-union plan requires source provenance")
    for field in (
        "model_id",
        "model_revision",
        "tokenizer_revision",
        "chat_template_sha256",
    ):
        if not isinstance(source.get(field), str) or not source[field]:
            raise ValueError(f"candidate-union source.{field} is required")
    waves = plan.get("waves")
    if not isinstance(waves, list) or not waves:
        raise ValueError("candidate-union plan requires waves")
    wave_ids = set()
    case_ids = set()
    source_ids = set()
    for wave in waves:
        wave_id = wave.get("wave_id")
        if not isinstance(wave_id, str) or not wave_id or wave_id in wave_ids:
            raise ValueError(f"invalid candidate-union wave ID: {wave_id!r}")
        wave_ids.add(wave_id)
        cases = wave.get("cases")
        if not isinstance(cases, list) or not cases:
            raise ValueError(f"candidate-union wave {wave_id!r} has no cases")
        for case in cases:
            case_id = case.get("case_id")
            source_id = case.get("source_width1_artifact_id")
            if not isinstance(case_id, str) or not case_id or case_id in case_ids:
                raise ValueError(f"invalid candidate-union case ID: {case_id!r}")
            if (
                not isinstance(source_id, str)
                or not source_id
                or source_id in source_ids
            ):
                raise ValueError(
                    f"invalid candidate-union source artifact: {source_id!r}"
                )
            case_ids.add(case_id)
            source_ids.add(source_id)
            item = case.get("source_item")
            if not isinstance(item, Mapping) or item.get("artifact_id") != source_id:
                raise ValueError("candidate-union case source item is inconsistent")
            references = case.get("reference_artifacts")
            if not isinstance(references, list) or len(references) not in {5, 6}:
                raise ValueError("candidate-union case requires five or six references")
            for index, reference in enumerate(references):
                if reference.get("candidate_index") != index:
                    raise ValueError(
                        "candidate-union reference indices are not ordered"
                    )
                for field in ("path", "artifact_id", "payload_sha256"):
                    if (
                        not isinstance(reference.get(field), str)
                        or not reference[field]
                    ):
                        raise ValueError(
                            f"candidate-union reference.{field} is required"
                        )
                if len(reference["payload_sha256"]) != 64:
                    raise ValueError("candidate-union reference hash is invalid")
                token_id = reference.get("token_id")
                if isinstance(token_id, bool) or not isinstance(token_id, int):
                    raise ValueError("candidate-union reference token ID is invalid")


def select_wave(plan: Mapping[str, Any], wave_id: str) -> dict[str, Any]:
    validate_candidate_union_plan(plan)
    matches = [wave for wave in plan["waves"] if wave["wave_id"] == wave_id]
    if len(matches) != 1:
        raise ValueError(f"candidate-union wave not found exactly once: {wave_id}")
    return dict(matches[0])


def _load_references(case: Mapping[str, Any]):
    artifacts = []
    for record in case["reference_artifacts"]:
        path = Path(record["path"])
        manifest = validate_topk_compact_trace_integrity(path)
        if manifest.get("artifact_id") != record["artifact_id"]:
            raise ValueError("candidate-union reference artifact ID drift")
        if manifest.get("data_sha256") != record["payload_sha256"]:
            raise ValueError("candidate-union reference payload hash drift")
        if (
            manifest.get("source_width1_artifact_id")
            != case["source_width1_artifact_id"]
        ):
            raise ValueError("candidate-union reference source drift")
        artifact = load_topk_compact_trace(path)
        candidate = artifact.topk_trace.candidate_selection.candidates[0]
        if candidate.token_id != record["token_id"]:
            raise ValueError("candidate-union reference token drift")
        artifacts.append(artifact)
    return artifacts


def _case_identity(
    case: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    plan_sha256: str,
    wave_id: str,
    code_revision: Mapping[str, Any],
    runtime_environment: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    identity = {
        "source_width1_artifact_id": case["source_width1_artifact_id"],
        "source_work_item_sha256": _sha256(case["source_item"]),
        "reference_artifacts": list(case["reference_artifacts"]),
        "candidate_union_plan_sha256": plan_sha256,
        "wave_id": wave_id,
        "trace_family_id": CANDIDATE_UNION_TRACE_FAMILY_ID,
        "refinement_trace_family_id": REFINEMENT_TRACE_FAMILY_ID,
        "model": dict(config["model"]),
        "adag_config": dict(config["adag_config"]),
        "code_revision": dict(code_revision),
        "runtime_environment": dict(runtime_environment),
    }
    digest = _sha256(identity)
    identity["sha256"] = digest
    return f"candidate-union-{digest[:24]}", identity


def _topology_keys(trace):
    nodes = {
        (int(row.layer), int(row.token), int(row.neuron))
        for _, row in trace.circuit_data.df_node.iterrows()
    }
    edges = set()
    for _, row in trace.circuit_data.df_edge.iterrows():
        values = [
            tuple(int(value) for value in str(row[column]).split("->"))
            for column in ("layer", "token", "neuron")
        ]
        edges.add(
            (
                (values[0][0], values[1][0], values[2][0]),
                (values[0][1], values[1][1], values[2][1]),
            )
        )
    return nodes, edges


def _assert_refinement_topology(trace, topology, candidate_token_id: int) -> None:
    nodes, edges = _topology_keys(trace)
    expected_edges = {
        (tuple(source), tuple(target)) for source, target in topology.edges
    }
    if edges != expected_edges:
        missing = sorted(expected_edges - edges)[:5]
        extra = sorted(edges - expected_edges)[:5]
        raise ValueError(
            f"fixed-topology edge mismatch; missing={missing}, extra={extra}"
        )
    final_layer = max(node[0] for node in nodes)
    expected_mlp = {tuple(node) for node in topology.mlp_nodes}
    actual_mlp = {node for node in nodes if 0 <= node[0] < final_layer}
    if actual_mlp != expected_mlp:
        raise ValueError("fixed-topology MLP node mismatch")
    final_nodes = {node for node in nodes if node[0] == final_layer}
    if len(final_nodes) != 1 or next(iter(final_nodes))[2] != candidate_token_id:
        raise ValueError("fixed-topology refinement has the wrong terminal logit")


def _refinement_identity(
    union_identity: Mapping[str, Any],
    *,
    candidate_index: int,
    candidate_token_id: int,
    topology_sha256: str,
) -> tuple[str, dict[str, Any]]:
    identity = {
        "candidate_union_identity_sha256": union_identity["sha256"],
        "candidate_index": candidate_index,
        "candidate_token_id": candidate_token_id,
        "topology_sha256": topology_sha256,
        "measurement_semantics": "candidate_specific_fixed_topology_rescore",
    }
    digest = _sha256(identity)
    identity["sha256"] = digest
    return f"topk-refinement-{digest[:24]}", identity


def run_candidate_union_wave(
    *,
    config: dict[str, Any],
    plan: dict[str, Any],
    wave_id: str,
    artifact_root: Path,
    summary_jsonl: Path,
    only_case_id: str | None = None,
    dry_run: bool = False,
    _model_bundle=None,
    _code_revision: Mapping[str, Any] | None = None,
    _runtime_environment: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    validate_run_config(config)
    wave = select_wave(plan, wave_id)
    repo_root = Path(__file__).resolve().parents[2]
    code_revision = dict(_code_revision or collect_code_revision(repo_root))
    runtime_environment = dict(_runtime_environment or collect_runtime_environment())
    execution = plan.get("execution")
    if isinstance(execution, Mapping):
        if execution.get("config_canonical_sha256") != _sha256(config):
            raise ValueError("candidate-union execution config hash drift")
        if (
            execution.get("required_clean_worktree") is True
            and code_revision.get("git_dirty") is not False
        ):
            raise ValueError(
                "candidate-union execution requires a clean frozen worktree"
            )
    plan_sha256 = _sha256(plan)
    if plan["source"]["model_id"] != config["model"]["model_id"]:
        raise ValueError("candidate-union model ID disagrees with config")
    if plan["source"]["model_revision"] != config["model"]["revision"]:
        raise ValueError("candidate-union model revision disagrees with config")
    cases = list(wave["cases"])
    if only_case_id is not None:
        cases = [case for case in cases if case["case_id"] == only_case_id]
        if not cases:
            raise ValueError(f"candidate-union case not found: {only_case_id}")

    planned = []
    results = []
    family_root = artifact_root / CANDIDATE_UNION_TRACE_FAMILY_ID / wave_id
    for case in cases:
        references = _load_references(case)
        topology_sha256, topologies = frozen_union_topologies(references)
        artifact_id, identity = _case_identity(
            case,
            config=config,
            plan_sha256=plan_sha256,
            wave_id=wave_id,
            code_revision=code_revision,
            runtime_environment=runtime_environment,
        )
        path = family_root / artifact_id
        base = {
            "wave_id": wave_id,
            "case_id": case["case_id"],
            "source_width1_artifact_id": case["source_width1_artifact_id"],
            "artifact_id": artifact_id,
            "artifact_identity_sha256": identity["sha256"],
            "candidate_count": len(references),
            "topology_sha256": topology_sha256,
            "code_revision": code_revision,
        }
        if path.exists():
            artifact = load_candidate_union_artifact(path)
            if artifact.manifest.get("artifact_identity") != identity:
                raise FileExistsError(
                    f"candidate-union artifact identity mismatch: {path}"
                )
            record = {
                **base,
                "status": "skipped_complete",
                "artifact_path": str(path),
                "artifact_bytes": _directory_size(path),
            }
            results.append(record)
            if not dry_run:
                _append_jsonl(summary_jsonl, record)
            continue
        if dry_run:
            results.append({**base, "status": "planned", "artifact_path": str(path)})
        else:
            planned.append(
                (
                    case,
                    references,
                    topologies,
                    topology_sha256,
                    artifact_id,
                    identity,
                    path,
                    base,
                )
            )
    if dry_run or not planned:
        return results

    _ensure_execution_cohort(
        artifact_root=artifact_root,
        plan_sha256=plan_sha256,
        config=config,
        code_revision=code_revision,
        runtime_environment=runtime_environment,
    )
    if _model_bundle is None:
        load_started = time.perf_counter()
        model, tokenizer = _load_model_and_tokenizer(config)
        model_load_seconds = time.perf_counter() - load_started
    else:
        model, tokenizer = _model_bundle
        model_load_seconds = 0.0
    device = config["model"]["device"]
    uses_cuda = device.startswith("cuda")
    gpu_info = _gpu_info(device)
    adag_config = ADAGConfig(**{**config["adag_config"], "device": device})
    signal_state = {"requested": False}
    previous_handler = signal.getsignal(signal.SIGUSR1)

    def request_stop(_signum, _frame) -> None:
        signal_state["requested"] = True

    signal.signal(signal.SIGUSR1, request_stop)
    try:
        for (
            case,
            references,
            topologies,
            topology_sha256,
            artifact_id,
            identity,
            path,
            base,
        ) in planned:
            if signal_state["requested"]:
                raise RuntimeError("candidate-union wave stopped after Slurm SIGUSR1")
            item = dict(case["source_item"])
            position = item["target_selection"]["response_token_positions"][0]
            example = item["example"]
            refinements = []
            candidate_metrics = []
            started = time.perf_counter()
            if uses_cuda:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            try:
                for index, (reference, topology) in enumerate(
                    zip(references, topologies)
                ):
                    candidate = reference.topk_trace.candidate_selection.candidates[0]
                    refinement_id, refinement_identity = _refinement_identity(
                        identity,
                        candidate_index=index,
                        candidate_token_id=candidate.token_id,
                        topology_sha256=topology_sha256,
                    )
                    refinement_path = (
                        artifact_root
                        / REFINEMENT_TRACE_FAMILY_ID
                        / wave_id
                        / artifact_id
                        / f"candidate-{index}"
                        / refinement_id
                    )
                    if refinement_path.exists():
                        manifest = validate_topk_compact_trace_integrity(
                            refinement_path
                        )
                        if manifest.get("artifact_identity") != refinement_identity:
                            raise FileExistsError(
                                "candidate-union refinement identity mismatch: "
                                f"{refinement_path}"
                            )
                        refinement = load_topk_compact_trace(refinement_path)
                        refinements.append(refinement)
                        candidate_metrics.append(
                            {
                                "candidate_index": index,
                                "candidate_token_id": candidate.token_id,
                                "status": "skipped_complete",
                                "artifact_path": str(refinement_path),
                            }
                        )
                        continue
                    if signal_state["requested"]:
                        raise RuntimeError(
                            "candidate-union wave stopped after Slurm SIGUSR1"
                        )
                    item["specified_candidate_token_id"] = candidate.token_id
                    instrumentation = TraceInstrumentation(
                        device=device, synchronize_cuda=uses_cuda
                    )
                    config_before = _model_config_sha256(model)
                    refinement_started = time.perf_counter()
                    trace = trace_teacher_forced_candidates(
                        model,
                        tokenizer,
                        example["prompt"],
                        example["response"],
                        position,
                        adag_config,
                        candidate_policy_id="specified_token",
                        candidate_count=1,
                        specified_candidate_token_id=candidate.token_id,
                        joint_objective_id="raw_logit_sum",
                        trace_family_id=REFINEMENT_TRACE_FAMILY_ID,
                        label=example["example_id"],
                        instrumentation=instrumentation,
                        frozen_topology=topology,
                        frozen_topology_sha256=topology_sha256,
                    )
                    if _model_config_sha256(model) != config_before:
                        raise RuntimeError(
                            "candidate-union refinement leaked model configuration"
                        )
                    validate_runtime_topk_trace_against_item(
                        trace, item, REFINEMENT_TRACE_FAMILY
                    )
                    _assert_refinement_topology(trace, topology, candidate.token_id)
                    if (
                        trace.circuit_data.trace_metadata.get("chat_template_sha256")
                        != plan["source"]["chat_template_sha256"]
                    ):
                        raise ValueError("candidate-union chat template drift")
                    if uses_cuda:
                        torch.cuda.synchronize()
                    elapsed = time.perf_counter() - refinement_started
                    snapshot = instrumentation.snapshot()
                    trace.circuit_data.trace_metadata["instrumentation"] = snapshot
                    metrics = {
                        "status": "complete",
                        "candidate_index": index,
                        "candidate_token_id": candidate.token_id,
                        "trace_wall_seconds": elapsed,
                        "node_count": len(trace.circuit_data.df_node),
                        "edge_count": len(trace.circuit_data.df_edge),
                        "topology_sha256": topology_sha256,
                        "instrumentation": snapshot,
                    }
                    save_topk_compact_trace(
                        refinement_path,
                        trace,
                        metrics=metrics,
                        manifest={
                            "artifact_id": refinement_id,
                            "artifact_identity": refinement_identity,
                            "source_width1_artifact_id": case[
                                "source_width1_artifact_id"
                            ],
                            "source_target_selection": item["target_selection"],
                            "bonafide_example": example,
                            "candidate_union_plan_sha256": plan_sha256,
                            "reference_artifact": case["reference_artifacts"][index],
                            "model_revision": config["model"]["revision"],
                            "code_revision": code_revision,
                            "runtime_environment": runtime_environment,
                            "gpu": gpu_info,
                        },
                    )
                    refinements.append(load_topk_compact_trace(refinement_path))
                    candidate_metrics.append(
                        {
                            **metrics,
                            "artifact_path": str(refinement_path),
                            "artifact_bytes": _directory_size(refinement_path),
                        }
                    )
                    del trace
                    gc.collect()
                    if uses_cuda:
                        torch.cuda.empty_cache()

                union_trace = assemble_candidate_union(
                    references,
                    refinements,
                    topology_sha256=topology_sha256,
                    source_width1_artifact_id=case["source_width1_artifact_id"],
                )
                selected_node_entries = sum(
                    sum(row) for row in union_trace.df_node["selected_by_candidate"]
                )
                selected_edge_entries = sum(
                    sum(row) for row in union_trace.df_edge["selected_by_candidate"]
                )
                metrics = {
                    "status": "complete",
                    "candidate_count": union_trace.candidate_count,
                    "node_count": len(union_trace.df_node),
                    "edge_count": len(union_trace.df_edge),
                    "dense_applicable_node_measurement_count": sum(
                        sum(row)
                        for row in union_trace.df_node["applicable_by_candidate"]
                    ),
                    "dense_applicable_edge_measurement_count": sum(
                        sum(row)
                        for row in union_trace.df_edge["applicable_by_candidate"]
                    ),
                    "selected_node_membership_count": selected_node_entries,
                    "selected_edge_membership_count": selected_edge_entries,
                    "refinement_wall_seconds": sum(
                        value.get("trace_wall_seconds", 0.0)
                        for value in candidate_metrics
                    ),
                    "candidate_metrics": candidate_metrics,
                    "total_unit_wall_seconds": time.perf_counter() - started,
                    "cuda_peak_allocated_bytes": (
                        torch.cuda.max_memory_allocated() if uses_cuda else 0
                    ),
                    "cuda_peak_reserved_bytes": (
                        torch.cuda.max_memory_reserved() if uses_cuda else 0
                    ),
                    "cuda_headroom_after_peak_bytes": (
                        gpu_info["total_memory_bytes"]
                        - torch.cuda.max_memory_reserved()
                        if uses_cuda and gpu_info is not None
                        else 0
                    ),
                    "rss_peak_after_bytes": _rss_peak_bytes(),
                }
                save_candidate_union_artifact(
                    path,
                    union_trace,
                    metrics=metrics,
                    manifest={
                        "artifact_id": artifact_id,
                        "artifact_identity": identity,
                        "candidate_union_plan_sha256": plan_sha256,
                        "source_target_selection": item["target_selection"],
                        "bonafide_example": example,
                        "model_id": config["model"]["model_id"],
                        "model_revision": config["model"]["revision"],
                        "code_revision": code_revision,
                        "runtime_environment": runtime_environment,
                        "gpu": gpu_info,
                    },
                )
                record = {
                    **base,
                    **metrics,
                    "model_load_seconds": model_load_seconds,
                    "artifact_path": str(path),
                    "artifact_bytes": _directory_size(path),
                    "runtime_environment": runtime_environment,
                    "gpu": gpu_info,
                }
            except BaseException as error:
                record = {
                    **base,
                    "status": "error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "total_unit_wall_seconds": time.perf_counter() - started,
                }
                _append_jsonl(summary_jsonl, record)
                results.append(record)
                raise
            _append_jsonl(summary_jsonl, record)
            results.append(record)
    finally:
        signal.signal(signal.SIGUSR1, previous_handler)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--wave", required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--summary-jsonl", type=Path, required=True)
    parser.add_argument("--only-case-id")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-records", action="store_true")
    args = parser.parse_args()
    config = load_json(args.config)
    plan = load_json(args.plan)
    records = run_candidate_union_wave(
        config=config,
        plan=plan,
        wave_id=args.wave,
        artifact_root=args.artifact_root,
        summary_jsonl=args.summary_jsonl,
        only_case_id=args.only_case_id,
        dry_run=args.dry_run,
    )
    if args.print_records or args.dry_run:
        for record in records:
            print(json.dumps(record, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
