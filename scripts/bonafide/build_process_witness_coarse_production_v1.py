#!/usr/bin/env python3
"""Build the immutable network-free full-corpus coarse proposal campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any

from circuits.analysis.bonafide.canonical import canonical_sha256, file_sha256
from circuits.analysis.bonafide.coarse_sampling_production_v1 import (
    BUNDLE_SCHEMA,
    assign_response_shards,
    compact_jsonl_bytes,
    load_production_config,
    load_workstation_bundle,
    openai_batch_line,
    production_request,
    production_units,
    response_windows,
)
from circuits.labeling.io import (
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_jsonl,
)

EXPECTED_COUNTS = {
    "responses": 188,
    "units": 94_546,
    "provider_pending_units": 74_860,
    "deterministic_surface_units": 19_500,
    "deterministic_terminal_units": 186,
    "fragment_groups_over_96_tokens": 51,
    "windows": 12_557,
    "physical_requests": 37_671,
    "replica_requests": 25_114,
}

_BOUND_SOURCE_FILES = (
    "circuits/analysis/bonafide/coarse_sampling_production_v1.py",
    "circuits/analysis/bonafide/coarse_sampling_openai_batch_production_v1.py",
    "circuits/analysis/bonafide/coarse_sampling_annotation_v4.py",
    "circuits/analysis/bonafide/coarse_sampling_annotation_v3.py",
    "circuits/analysis/bonafide/coarse_sampling_annotation_v2.py",
    "circuits/analysis/bonafide/coarse_sampling_annotation.py",
    "circuits/analysis/bonafide/process_annotation.py",
    "circuits/analysis/bonafide/canonical.py",
    "circuits/labeling/io.py",
    "scripts/bonafide/build_process_witness_coarse_production_v1.py",
    "scripts/bonafide/process_witness_coarse_openai_batch_production_v1.py",
    "scripts/bonafide/configs/process_witness_coarse_production_v1.json",
    "scripts/bonafide/configs/labeling/prices-2026-08-16-coarse-v2.json",
)


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def _source_revision() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    if Path(_git(root, "rev-parse", "--show-toplevel")) != root:
        raise ValueError("coarse production builder repository root drift")
    if _git(root, "status", "--porcelain=v1", "--untracked-files=no"):
        raise ValueError("coarse production build requires a clean tracked worktree")
    commit = _git(root, "rev-parse", "HEAD")
    files = []
    for relative in _BOUND_SOURCE_FILES:
        if _git(root, "ls-files", "--error-unmatch", "--", relative) != relative:
            raise ValueError(f"coarse production source is untracked: {relative}")
        path = root / relative
        blob = _git(root, "rev-parse", f"{commit}:{relative}")
        committed = subprocess.run(
            ["git", "cat-file", "blob", blob], cwd=root, check=True, capture_output=True
        ).stdout
        expected = hashlib.sha256(committed).hexdigest()
        if file_sha256(path) != expected:
            raise ValueError(f"coarse production source differs from HEAD: {relative}")
        files.append({"path": relative, "git_blob": blob, "sha256": expected})
    return {
        "repo_root": str(root),
        "git_commit": commit,
        "git_tree": _git(root, "rev-parse", "HEAD^{tree}"),
        "tracked_worktree_clean": True,
        "files": files,
    }


def _readonly_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def _request_rows(
    *,
    document: dict[str, Any],
    units: list[dict[str, Any]],
    windows: list[dict[str, Any]],
    config: dict[str, Any],
    physical_start: int,
):
    by_id = {unit["unit_id"]: unit for unit in units}
    physical = physical_start
    for window in windows:
        focal = [by_id[unit_id] for unit_id in window["focal_unit_ids"]]
        primary_id = None
        primary_body = None
        for replica in range(3):
            request = production_request(
                physical_index=physical,
                replica_index=replica,
                window=window,
                document=document,
                focal=focal,
                all_units=units,
                config=config,
                primary_request_id=primary_id,
            )
            if primary_id is None:
                primary_id = request["request_id"]
                primary_body = request["provider_body"]
            elif request["provider_body"] != primary_body:
                raise ValueError("coarse production replica body drift")
            yield request, openai_batch_line(request)
            physical += 1


def _cost_plan(
    *,
    config: dict[str, Any],
    prices: dict[str, Any],
    body_bytes: list[int],
) -> dict[str, Any]:
    provider = config["provider"]
    rates = prices["rates"]["openai"][provider["model"]]["native_batch"]
    overhead = int(provider["input_token_overhead_per_request"])
    request_input_bounds = [value + overhead for value in body_bytes]
    threshold = int(
        prices["long_context"][provider["model"]]["threshold_input_tokens_exclusive"]
    )
    long_indices = [
        i for i, value in enumerate(request_input_bounds) if value > threshold
    ]
    ordinary_input_rate = max(
        float(rates["input_per_million"]), float(rates["cache_write_per_million"])
    )
    ordinary_output_rate = float(rates["output_per_million"])
    live = prices["rates"]["openai"][provider["model"]]["live"]
    multiplier = prices["long_context"][provider["model"]]
    long_input_rate = max(
        float(live["input_per_million"]), float(live["cache_write_per_million"])
    ) * float(multiplier["input_multiplier"])
    long_output_rate = float(live["output_per_million"]) * float(
        multiplier["output_multiplier"]
    )
    strict_input = sum(
        value
        / 1_000_000
        * (long_input_rate if i in long_indices else ordinary_input_rate)
        for i, value in enumerate(request_input_bounds)
    )
    output_per = int(provider["max_output_tokens"])
    strict_output = sum(
        output_per
        / 1_000_000
        * (long_output_rate if i in long_indices else ordinary_output_rate)
        for i in range(len(body_bytes))
    )
    empirical = config["empirical_calibration"]
    direct_v4 = (
        float(empirical["source_actual_cost_usd"])
        * len(body_bytes)
        / int(empirical["source_request_count"])
    )
    token_ratio = float(empirical["source_input_tokens"]) / float(
        empirical["source_provider_body_utf8_bytes"]
    )
    value = {
        "schema_version": "adag.process-witness.coarse-production-cost-plan.v1",
        "price_snapshot_id": prices["snapshot_id"],
        "request_count": len(body_bytes),
        "provider_body_utf8_bytes": sum(body_bytes),
        "physical_input_tokens_empirical_ratio_forecast": round(
            sum(body_bytes) * token_ratio
        ),
        "provider_queued_input_token_limit": None,
        "provider_queued_input_token_limit_launch_gate": "must be recorded from the active API tier before submission",
        "direct_v4_cost_per_request_extrapolation_usd": direct_v4,
        "cache_pattern_forecast_usd": float(empirical["cache_pattern_forecast_usd"]),
        "cache_pattern_forecast_is_authorization_ceiling": False,
        "strict_input_token_upper_bound": sum(request_input_bounds),
        "strict_output_token_upper_bound": len(body_bytes) * output_per,
        "long_context_request_count_by_byte_bound": len(long_indices),
        "strict_no_cache_full_output_input_cost_usd": strict_input,
        "strict_no_cache_full_output_output_cost_usd": strict_output,
        "strict_no_cache_full_output_ceiling_usd": strict_input + strict_output,
        "reference_strict_ceiling_usd": float(
            empirical["strict_no_cache_full_output_reference_usd"]
        ),
        "assumptions": [
            "direct-v4 and cache-pattern values are forecasts, never caps",
            "prompt-cache reads/writes are observed operational behavior, never assumed",
            "strict ceiling prices UTF-8 body bytes plus per-request overhead as input tokens",
            "strict ceiling consumes the full max_output_tokens for every physical request",
            "long-context candidates conservatively use live rather than Batch rates",
            "the active API-tier queued-input-token limit remains an explicit launch gate",
        ],
    }
    value["cost_plan_sha256"] = canonical_sha256(value)
    return value


def build(
    *, workstation_bundle_path: Path, config_path: Path, destination: Path
) -> dict[str, Any]:
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    source_revision = _source_revision()
    config = load_production_config(config_path)
    workstation = load_workstation_bundle(workstation_bundle_path, config)
    documents = list(workstation["documents"])
    all_units: list[dict[str, Any]] = []
    all_windows: list[dict[str, Any]] = []
    response_specs = []
    physical_start = 0
    body_bytes: list[int] = []
    request_index: list[dict[str, Any]] = []
    for response_index, document in enumerate(documents):
        units = production_units(document)
        windows = response_windows(document, units, window_start=len(all_windows))
        block_bytes = 0
        block_request_ids = []
        for request, line in _request_rows(
            document=document,
            units=units,
            windows=windows,
            config=config,
            physical_start=physical_start,
        ):
            encoded = compact_jsonl_bytes(line)
            block_bytes += len(encoded)
            block_request_ids.append(request["request_id"])
            body_bytes.append(
                len(
                    json.dumps(
                        request["provider_body"],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ).encode("utf-8")
                )
            )
            request_index.append(
                {
                    key: request[key]
                    for key in (
                        "schema_version",
                        "request_id",
                        "physical_index",
                        "window_id",
                        "window_index",
                        "response_id",
                        "replica_index",
                        "body_sha256",
                        "config_sha256",
                        "arm_id",
                        "repeat_of_request_id",
                        "focal_unit_ids",
                        "prompt_sha256",
                        "full_response_sha256",
                        "markup_audit",
                    )
                }
            )
        response_specs.append(
            {
                "response_index": response_index,
                "response_id": document["response_id"],
                "bytes": block_bytes,
                "provider_body_utf8_bytes": sum(
                    body_bytes[physical_start : physical_start + len(block_request_ids)]
                ),
                "request_count": len(block_request_ids),
                "request_ids_in_order": block_request_ids,
                "physical_start": physical_start,
                "window_start": len(all_windows),
                "window_count": len(windows),
            }
        )
        physical_start += len(block_request_ids)
        all_units.extend(units)
        all_windows.extend(windows)

    observed = {
        "responses": len(documents),
        "units": len(all_units),
        "provider_pending_units": sum(
            u["assignment_route"] == "openai_pending" for u in all_units
        ),
        "deterministic_surface_units": sum(
            u["assignment_route"] == "deterministic_surface" for u in all_units
        ),
        "deterministic_terminal_units": sum(
            u["assignment_route"] == "deterministic_terminal_serialization"
            for u in all_units
        ),
        "fragment_groups_over_96_tokens": len(
            {u["fragment_of"] for u in all_units if u.get("fragment_of")}
        ),
        "windows": len(all_windows),
        "physical_requests": len(request_index),
        "replica_requests": sum(
            r["repeat_of_request_id"] is not None for r in request_index
        ),
    }
    if observed != EXPECTED_COUNTS:
        raise ValueError(f"coarse production full-corpus census drift: {observed}")
    if len({u["unit_id"] for u in all_units}) != len(all_units):
        raise ValueError("coarse production unit identity collision")
    if len({r["request_id"] for r in request_index}) != len(request_index):
        raise ValueError("coarse production request identity collision")

    shard_blocks = assign_response_shards(
        response_specs, int(config["sharding"]["maximum_batch_input_bytes"])
    )
    response_to_shard = {
        block["response_id"]: shard_index
        for shard_index, blocks in enumerate(shard_blocks)
        for block in blocks
    }
    for row in request_index:
        row["shard_id"] = f"shard-{response_to_shard[row['response_id']]:03d}"

    price_path = (config_path.parent / config["provider"]["price_snapshot"]).resolve()
    prices = json.loads(price_path.read_text(encoding="utf-8"))
    cost_plan = _cost_plan(config=config, prices=prices, body_bytes=body_bytes)
    temporary = destination.parent / f".{destination.name}.building-{uuid.uuid4().hex}"
    temporary.mkdir(parents=True)
    try:
        atomic_write_jsonl(temporary / "units.jsonl", all_units)
        atomic_write_jsonl(temporary / "windows.jsonl", all_windows)
        atomic_write_jsonl(temporary / "request-index.jsonl", request_index)
        shard_records = []
        request_by_id = {row["request_id"]: row for row in request_index}
        for shard_index, blocks in enumerate(shard_blocks):
            shard_id = f"shard-{shard_index:03d}"
            path = temporary / "batch-shards" / f"{shard_id}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
            request_ids = []
            with temp_path.open("xb") as handle:
                for block in blocks:
                    response_index = int(block["response_index"])
                    document = documents[response_index]
                    units = production_units(document)
                    windows = all_windows[
                        int(block["window_start"]) : int(block["window_start"])
                        + int(block["window_count"])
                    ]
                    for request, line in _request_rows(
                        document=document,
                        units=units,
                        windows=windows,
                        config=config,
                        physical_start=int(block["physical_start"]),
                    ):
                        expected = request_by_id[request["request_id"]]
                        if request["body_sha256"] != expected["body_sha256"]:
                            raise ValueError(
                                "coarse production second-pass request drift"
                            )
                        handle.write(compact_jsonl_bytes(line))
                        request_ids.append(request["request_id"])
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, path)
            expected_ids = [
                rid for block in blocks for rid in block["request_ids_in_order"]
            ]
            if request_ids != expected_ids:
                raise ValueError("coarse production shard request order drift")
            if path.stat().st_size != sum(int(block["bytes"]) for block in blocks):
                raise ValueError("coarse production shard byte census drift")
            shard_body_bytes = sum(
                int(block["provider_body_utf8_bytes"]) for block in blocks
            )
            shard_records.append(
                {
                    "shard_id": shard_id,
                    "path": str(path.relative_to(temporary)),
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                    "request_count": len(request_ids),
                    "provider_body_utf8_bytes": shard_body_bytes,
                    "direct_v4_cost_forecast_usd": (
                        float(config["empirical_calibration"]["source_actual_cost_usd"])
                        * len(request_ids)
                        / int(config["empirical_calibration"]["source_request_count"])
                    ),
                    "queued_input_tokens_empirical_forecast": round(
                        shard_body_bytes
                        * float(config["empirical_calibration"]["source_input_tokens"])
                        / float(
                            config["empirical_calibration"][
                                "source_provider_body_utf8_bytes"
                            ]
                        )
                    ),
                    "response_count": len(blocks),
                    "response_ids_in_order": [block["response_id"] for block in blocks],
                    "request_ids_in_order": request_ids,
                }
            )
        atomic_write_json(
            temporary / "shards.json",
            {
                "schema_version": "adag.process-witness.coarse-production-shards.v1",
                "policy_id": config["sharding"]["policy_id"],
                "maximum_batch_input_bytes": config["sharding"][
                    "maximum_batch_input_bytes"
                ],
                "shards": shard_records,
            },
        )
        atomic_write_json(temporary / "cost-plan.json", cost_plan)
        atomic_write_bytes(temporary / "protocol-config.json", config_path.read_bytes())
        atomic_write_bytes(temporary / "price-snapshot.json", price_path.read_bytes())
        launch = {
            "schema_version": "adag.process-witness.coarse-production-launch-gates.v1",
            "status": "blocked_pending_fresh_launch_authorization_and_queue_limit",
            "campaign_frozen": True,
            "network_calls_made": 0,
            "fresh_run_specific_spend_authorization": None,
            "provider_batch_queued_input_token_limit": None,
            "provider_batch_queued_input_token_limit_sufficient": None,
            "recommended_first_wave": "one frozen representative shard for cost calibration",
            "remaining_requests_change_after_calibration": False,
        }
        atomic_write_json(temporary / "launch-gates.json", launch)
        files = [
            {
                "path": str(path.relative_to(temporary)),
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(temporary.rglob("*"))
            if path.is_file() and path.name != "manifest.json"
        ]
        manifest = {
            "schema_version": BUNDLE_SCHEMA,
            "status": "prepared_offline_no_provider_calls",
            "claim_boundary": config["claim_boundary"],
            "config_id": config["config_id"],
            "config_sha256": file_sha256(config_path),
            "source_workstation_bundle": str(workstation_bundle_path.resolve()),
            "source_workstation_bundle_sha256": file_sha256(workstation_bundle_path),
            "source_revision": source_revision,
            "counts": {**observed, "shards": len(shard_records)},
            "shard_bindings": [
                {
                    key: shard[key]
                    for key in (
                        "shard_id",
                        "path",
                        "bytes",
                        "sha256",
                        "request_count",
                        "response_count",
                    )
                }
                for shard in shard_records
            ],
            "cost_plan_sha256": cost_plan["cost_plan_sha256"],
            "network_calls_made": 0,
            "provider_submission_performed": False,
            "prior_v1_v4_artifacts_mutated": False,
            "files": files,
        }
        manifest["manifest_sha256"] = canonical_sha256(manifest)
        atomic_write_json(temporary / "manifest.json", manifest)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary.rename(destination)
        _readonly_tree(destination)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workstation-bundle", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    result = build(
        workstation_bundle_path=args.workstation_bundle.resolve(),
        config_path=args.config.resolve(),
        destination=args.destination.resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
