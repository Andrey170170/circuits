"""Fixed local Transluce scoring for candidates and final-label audit."""

from __future__ import annotations

import json
import math
import os
import resource
import socket
import time
from pathlib import Path
from typing import Any, Literal, cast

from circuits.analysis.bonafide.canonical import file_sha256
from circuits.descriptions.types import ActivationRecord
from circuits.descriptions.vllm_backend import (
    FinetunedSimulator,
    score_attr_explanations,
)
from circuits.labeling.config import LabelingRecipe
from circuits.labeling.io import atomic_write_json
from circuits.labeling.profiles import retokenize_for_simulator
from circuits.labeling.runtime import (
    load_run_manifest,
    load_stage_requests,
    resolve_local_snapshot,
)
from circuits.labeling.schema import GenerationResult


def _profile_path(run_root: Path, state: str, cluster_id: int) -> Path:
    return run_root / "profiles" / state / f"cluster-{cluster_id:04d}.json"


def _load_records(
    run_root: Path,
    *,
    state: str,
    cluster_id: int,
    partition: str,
    simulator_tokenizer: Any,
) -> tuple[list[ActivationRecord], list[dict[str, Any]]]:
    value = json.loads(
        _profile_path(run_root, state, cluster_id).read_text(encoding="utf-8")
    )
    records: list[ActivationRecord] = []
    diagnostics: list[dict[str, Any]] = []
    for profile in value["partitions"][partition]:
        source = ActivationRecord.model_validate(profile["record"])
        mapped, alignment = retokenize_for_simulator(source, simulator_tokenizer)
        records.append(mapped)
        diagnostics.append(
            {
                "trace_unit_id": profile["trace_unit_id"],
                "matched_signed_basis_count": profile["matched_signed_basis_count"],
                **alignment,
            }
        )
    return records, diagnostics


def _candidate_texts(
    run_root: Path, state: str, cluster_id: int
) -> list[tuple[str, str, dict[str, Any]]]:
    requests = load_stage_requests(run_root, "candidate_generation")
    values: list[tuple[str, str, dict[str, Any]]] = []
    for request in requests:
        if request.state != state or request.cluster_id != cluster_id:
            continue
        path = (
            run_root / "results" / "candidate_generation" / f"{request.request_id}.json"
        )
        if not path.is_file():
            raise ValueError(f"candidate result is missing: {path}")
        result = GenerationResult.model_validate_json(path.read_text(encoding="utf-8"))
        if result.request_id != request.request_id:
            raise ValueError(f"candidate result request ID mismatch: {path}")
        if result.parse_status != "success" or result.parsed is None:
            continue
        description = result.parsed.get("description")
        if isinstance(description, str) and description.strip():
            values.append(
                (request.request_id, description.strip(), dict(result.parsed))
            )
    return values


def _summary_outputs(
    run_root: Path, state: str, cluster_id: int
) -> list[dict[str, Any]]:
    requests = load_stage_requests(run_root, "cluster_summary")
    values: list[dict[str, Any]] = []
    for request in requests:
        if request.state != state or request.cluster_id != cluster_id:
            continue
        path = run_root / "results" / "cluster_summary" / f"{request.request_id}.json"
        if not path.is_file():
            raise ValueError(f"summary result is missing: {path}")
        result = GenerationResult.model_validate_json(path.read_text(encoding="utf-8"))
        if result.request_id != request.request_id:
            raise ValueError(f"summary result request ID mismatch: {path}")
        if result.parse_status != "success" or result.parsed is None:
            continue
        label = result.parsed.get("label")
        if isinstance(label, str) and label.strip():
            relative = path.relative_to(run_root).as_posix()
            values.append(
                {
                    "request_id": request.request_id,
                    "text": label.strip(),
                    "status": result.parsed.get("status"),
                    "source_result_path": relative,
                    "source_result_sha256": file_sha256(path),
                }
            )
    return values


def _is_insufficient_evidence(*, text: str, status: Any = None) -> bool:
    return text.strip() == "insufficient_evidence" or status == "insufficient_evidence"


def correlation_sort_key(item: dict[str, Any]) -> tuple[float, str]:
    value = item.get("correlation")
    correlation = (
        float(value)
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        else float("-inf")
    )
    return correlation, str(item.get("request_id", ""))


def score_run(
    *,
    run_root: Path,
    phase: Literal["candidate_selection", "summary_selection", "summary_audit"],
    states: set[str] | None = None,
    cluster_ids: set[int] | None = None,
) -> dict[str, int]:
    manifest = load_run_manifest(run_root)
    recipe = LabelingRecipe.model_validate(manifest["recipe"])
    rich_policy = recipe.prompt_policy in {"width_one_v2", "hybrid_candidate_v1"}
    if phase == "summary_selection" and not rich_policy:
        raise ValueError("summary_selection requires an evidence-rich prompt policy")
    selected = {
        state: [
            int(cluster_id)
            for cluster_id in values
            if cluster_ids is None or int(cluster_id) in cluster_ids
        ]
        for state, values in manifest["selected_clusters"].items()
        if states is None or state in states
    }
    if phase == "summary_audit" and rich_policy:
        missing_selection = [
            run_root
            / "scores"
            / "summary_selection"
            / state
            / f"cluster-{cluster_id:04d}.json"
            for state, ids in selected.items()
            for cluster_id in ids
            if not (
                run_root
                / "scores"
                / "summary_selection"
                / state
                / f"cluster-{cluster_id:04d}.json"
            ).is_file()
        ]
        if missing_selection:
            raise ValueError(
                f"{recipe.prompt_policy} summary_audit requires completed summary_selection; "
                f"missing {missing_selection[0]}"
            )
    partition = (
        "selection_scoring"
        if phase in {"candidate_selection", "summary_selection"}
        else "audit"
    )
    output_name = phase
    started = time.monotonic()
    counts = {
        "planned": sum(len(values) for values in selected.values()),
        "completed": 0,
        "skipped": 0,
        "inputs_not_scored": 0,
    }

    import torch

    if torch.cuda.is_available():
        # Some CHPC CUDA/driver combinations reject memory-stat operations until
        # the process has initialized its CUDA context explicitly.
        torch.cuda.init()
        torch.cuda.set_device(recipe.scorer.gpu_index)
        torch.cuda.reset_peak_memory_stats(recipe.scorer.gpu_index)
    simulator_snapshot = resolve_local_snapshot(
        recipe.scorer.model, recipe.scorer.model_revision
    )
    simulator = FinetunedSimulator(
        model_name=str(simulator_snapshot),
        gpu_idx=recipe.scorer.gpu_index,
    )
    try:
        for state, ids in selected.items():
            for cluster_id in ids:
                output = (
                    run_root
                    / "scores"
                    / output_name
                    / state
                    / f"cluster-{cluster_id:04d}.json"
                )
                if output.exists():
                    counts["skipped"] += 1
                    continue
                candidate_outputs = (
                    _candidate_texts(run_root, state, cluster_id)
                    if phase == "candidate_selection"
                    else None
                )
                summary_outputs = (
                    _summary_outputs(run_root, state, cluster_id)
                    if phase != "candidate_selection"
                    else None
                )
                skipped_inputs: list[dict[str, Any]] = []
                if candidate_outputs is not None:
                    scoreable_candidates = candidate_outputs
                    if rich_policy:
                        scoreable_candidates = []
                        for request_id, text, parsed in candidate_outputs:
                            if _is_insufficient_evidence(text=text):
                                skipped_inputs.append(
                                    {
                                        "request_id": request_id,
                                        "text": text,
                                        "reason": "candidate_reported_insufficient_evidence",
                                        "candidate": parsed,
                                    }
                                )
                            else:
                                scoreable_candidates.append((request_id, text, parsed))
                    texts = [
                        {"request_id": request_id, "text": text}
                        for request_id, text, _ in scoreable_candidates
                    ]
                else:
                    assert summary_outputs is not None
                    scoreable_summaries = summary_outputs
                    if rich_policy:
                        scoreable_summaries = []
                        for item in summary_outputs:
                            if _is_insufficient_evidence(
                                text=item["text"], status=item["status"]
                            ):
                                skipped_inputs.append(
                                    {
                                        **item,
                                        "reason": "model_reported_insufficient_evidence",
                                    }
                                )
                            else:
                                scoreable_summaries.append(item)
                    texts = scoreable_summaries
                if not texts and not skipped_inputs:
                    raise ValueError(
                        f"no parsed texts to score for {state} cluster {cluster_id}"
                    )
                alignment: list[dict[str, Any]] = []
                scored = []
                if texts:
                    records, alignment = _load_records(
                        run_root,
                        state=state,
                        cluster_id=cluster_id,
                        partition=partition,
                        simulator_tokenizer=simulator.tokenizer,
                    )
                    explanations = [item["text"] for item in texts]
                    scored = score_attr_explanations(
                        simulator,
                        explanations,
                        records,
                        "pos",
                        use_raw_activations=True,
                        keep_only_top_predictions=False,
                    )
                values = [
                    {
                        **item,
                        "correlation": score.score,
                        "rsquared": score.rsquared,
                    }
                    for item, score in zip(texts, scored, strict=True)
                ]
                if phase == "candidate_selection" and rich_policy:
                    assert candidate_outputs is not None
                    by_request = {
                        request_id: parsed
                        for request_id, _, parsed in candidate_outputs
                    }
                    for value in values:
                        value["candidate"] = by_request[cast(str, value["request_id"])]
                values.sort(key=correlation_sort_key, reverse=True)
                atomic_write_json(
                    output,
                    {
                        "schema_version": "adag.labeling.local-scores.v1",
                        "run_id": manifest["run_id"],
                        "phase": phase,
                        "state": state,
                        "cluster_id": cluster_id,
                        "partition": partition,
                        "simulator": recipe.scorer.model_dump(mode="json"),
                        "alignment_diagnostics": alignment,
                        "scores": values,
                        "skipped": skipped_inputs,
                    },
                )
                counts["completed"] += 1
                counts["inputs_not_scored"] += len(skipped_inputs)
    finally:
        simulator.cleanup()

    elapsed = time.monotonic() - started
    allocated_gpus = int(os.environ.get("SLURM_GPUS_ON_NODE", "1"))
    peak_hbm = (
        int(torch.cuda.max_memory_allocated(recipe.scorer.gpu_index))
        if torch.cuda.is_available()
        else None
    )
    telemetry_path = (
        run_root
        / "telemetry"
        / "local_scoring"
        / f"{phase}-{os.environ.get('SLURM_JOB_ID', 'local')}-"
        f"{os.environ.get('SLURM_ARRAY_TASK_ID', '0')}.json"
    )
    atomic_write_json(
        telemetry_path,
        {
            "schema_version": "adag.labeling.local-scoring-telemetry.v1",
            "run_id": manifest["run_id"],
            "phase": phase,
            "backend": recipe.scorer.backend,
            "model": recipe.scorer.model,
            "model_revision": recipe.scorer.model_revision,
            "elapsed_seconds": elapsed,
            "allocated_gpu_count": allocated_gpus,
            "gpu_hours": allocated_gpus * elapsed / 3600,
            "peak_hbm_bytes": peak_hbm,
            "peak_host_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "host": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "counts": counts,
        },
    )
    return counts
