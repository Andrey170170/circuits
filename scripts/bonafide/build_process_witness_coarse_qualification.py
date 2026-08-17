#!/usr/bin/env python3
"""Build a frozen, network-free Luna coarse-annotation qualification bundle."""

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
from circuits.analysis.bonafide.coarse_sampling_annotation import (
    BUNDLE_SCHEMA,
    build_qualification,
    cost_plan,
    forbidden_provider_input_leaks,
    load_coarse_config,
    load_v9_workstation_bundle,
)
from circuits.labeling.io import atomic_write_json, atomic_write_jsonl

_BOUND_SOURCE_FILES = (
    "circuits/analysis/bonafide/coarse_sampling_annotation.py",
    "circuits/analysis/bonafide/canonical.py",
    "circuits/analysis/bonafide/process_annotation.py",
    "circuits/labeling/io.py",
    "scripts/bonafide/build_process_witness_coarse_qualification.py",
    "scripts/bonafide/configs/process_witness_coarse_openai_v1.json",
    "scripts/bonafide/configs/labeling/prices-2026-07-30.json",
)


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _readonly_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _source_revision() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    if Path(_git(root, "rev-parse", "--show-toplevel")) != root:
        raise ValueError("coarse builder repository root drift")
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=no")
    if status:
        raise ValueError("coarse qualification build requires a clean tracked worktree")
    commit = _git(root, "rev-parse", "HEAD")
    files = []
    for relative in _BOUND_SOURCE_FILES:
        if _git(root, "ls-files", "--error-unmatch", "--", relative) != relative:
            raise ValueError(f"coarse qualification source is untracked: {relative}")
        path = root / relative
        blob = _git(root, "rev-parse", f"{commit}:{relative}")
        committed = subprocess.run(
            ["git", "cat-file", "blob", blob],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
        expected = hashlib.sha256(committed).hexdigest()
        if file_sha256(path) != expected:
            raise ValueError(
                f"coarse qualification source differs from HEAD: {relative}"
            )
        files.append({"path": relative, "git_blob": blob, "sha256": expected})
    return {
        "repo_root": str(root),
        "git_commit": commit,
        "git_tree": _git(root, "rev-parse", "HEAD^{tree}"),
        "tracked_worktree_clean": True,
        "files": files,
    }


def build(
    *, source_bundle: Path, config_path: Path, destination: Path
) -> dict[str, Any]:
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    source_revision = _source_revision()
    config = load_coarse_config(config_path)
    bundle = load_v9_workstation_bundle(source_bundle, config)
    qualification = build_qualification(bundle, config)
    price_path = (config_path.parent / config["provider"]["price_snapshot"]).resolve()
    prices = _load_object(price_path)
    plan = cost_plan(qualification["requests"], config, prices)
    plan.update(
        {
            "price_snapshot_path": str(price_path),
            "price_snapshot_sha256": file_sha256(price_path),
            "source_revision": source_revision,
        }
    )
    plan["cost_plan_sha256"] = canonical_sha256(plan)

    temporary = destination.parent / f".{destination.name}.building-{uuid.uuid4().hex}"
    if temporary.exists():
        raise FileExistsError(f"temporary destination exists: {temporary}")
    temporary.mkdir(parents=True)
    try:
        atomic_write_jsonl(temporary / "units.jsonl", qualification["units"])
        atomic_write_json(temporary / "windows.json", qualification["windows"])
        atomic_write_jsonl(temporary / "requests.jsonl", qualification["requests"])
        units_by_id = {unit["unit_id"]: unit for unit in qualification["units"]}
        documents_by_response = {
            document["response_id"]: document for document in bundle["documents"]
        }
        review_rows = []
        for window in qualification["windows"]:
            document = documents_by_response[window["response_id"]]
            focal_ids = set(window["focal_unit_ids"])
            bounded_context = [
                {
                    "role": "target" if unit_id in focal_ids else "context",
                    "unit_id": unit_id,
                    "text": units_by_id[unit_id]["text"],
                }
                for unit_id in window["context_unit_ids"]
            ]
            for unit_id in window["focal_unit_ids"]:
                unit = units_by_id[unit_id]
                review_rows.append(
                    {
                        "schema_version": "adag.process-witness.coarse-human-review-template.v1",
                        "window_index": window["window_index"],
                        "unit_id": unit_id,
                        "response_id": unit["response_id"],
                        "prompt_sha256": unit["prompt_sha256"],
                        "source_type_stratum": window["source_type_stratum"],
                        "position_stratum": window["position_stratum"],
                        "task_prompt": document["task_context"]["prompt"],
                        "bounded_response_units": bounded_context,
                        "token_span": unit["token_span"],
                        "text": unit["text"],
                        "human_tag": None,
                        "boundary_concerns": [],
                        "notes": "",
                    }
                )
        atomic_write_jsonl(temporary / "human-review-template.jsonl", review_rows)
        atomic_write_json(temporary / "cost-plan.json", plan)

        files = []
        for path in sorted(temporary.iterdir()):
            if path.name == "manifest.json":
                continue
            files.append(
                {
                    "path": path.name,
                    "sha256": file_sha256(path),
                    "bytes": path.stat().st_size,
                }
            )
        leaks = {
            request["request_id"]: forbidden_provider_input_leaks(request)
            for request in qualification["requests"]
        }
        leaks = {request_id: values for request_id, values in leaks.items() if values}
        if leaks:
            raise ValueError(f"forbidden provider-input leakage: {leaks}")
        requests = qualification["requests"]
        unique = [
            request for request in requests if request["repeat_of_request_id"] is None
        ]
        repeated = [request for request in requests if request["repeat_of_request_id"]]
        manifest = {
            "schema_version": BUNDLE_SCHEMA,
            "status": "prepared_offline_no_provider_calls",
            "claim_boundary": config["claim_boundary"],
            "qualification_claim_boundary": config[
                "qualification_claim_boundary"
            ],
            "config_id": config["config_id"],
            "config_path": str(config_path.resolve()),
            "config_sha256": file_sha256(config_path),
            "source_workstation_bundle": str(source_bundle.resolve()),
            "source_workstation_bundle_sha256": file_sha256(source_bundle),
            "price_snapshot_sha256": file_sha256(price_path),
            "counts": {
                "responses": 188,
                "all_units": len(qualification["units"]),
                "semantic_units_pending_api": sum(
                    unit["assignment_route"] == "openai_pending"
                    for unit in qualification["units"]
                ),
                "deterministic_surface_units": sum(
                    unit["assignment_route"] == "deterministic_surface"
                    for unit in qualification["units"]
                ),
                "deterministic_final_units": sum(
                    unit["assignment_route"] == "deterministic_terminal_serialization"
                    for unit in qualification["units"]
                ),
                "unique_windows": len(unique),
                "unique_focal_units": len(qualification["review_units"]),
                "physical_requests": len(requests),
                "repeat_requests": len(repeated),
            },
            "request_bindings_in_order": [
                {
                    "request_id": request["request_id"],
                    "body_sha256": request["body_sha256"],
                    "repeat_of_request_id": request["repeat_of_request_id"],
                }
                for request in requests
            ],
            "files": files,
            "network_calls_made": 0,
        }
        manifest["manifest_sha256"] = canonical_sha256(manifest)
        atomic_write_json(temporary / "manifest.json", manifest)
        os.replace(temporary, destination)
        _readonly_tree(destination)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build(
        source_bundle=args.source_bundle.resolve(),
        config_path=args.config.resolve(),
        destination=args.output.resolve(),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
