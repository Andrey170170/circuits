#!/usr/bin/env python3
"""Build the immutable, network-free full-response coarse v2 Batch bundle."""

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
from circuits.analysis.bonafide.coarse_sampling_annotation_v2 import (
    BUNDLE_SCHEMA,
    build_v2_qualification,
    cost_plan_v2,
    forbidden_provider_input_leaks_v2,
    load_coarse_v2_config,
    load_v1_comparison_baseline,
    load_v1_cost_correction_audit,
)
from circuits.labeling.io import (
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_jsonl,
)

_BOUND_SOURCE_FILES = (
    "circuits/analysis/bonafide/coarse_sampling_annotation_v2.py",
    "circuits/analysis/bonafide/coarse_sampling_annotation.py",
    "circuits/analysis/bonafide/coarse_sampling_openai_run.py",
    "circuits/analysis/bonafide/canonical.py",
    "circuits/labeling/io.py",
    "scripts/bonafide/build_process_witness_coarse_qualification_v2.py",
    "scripts/bonafide/configs/process_witness_coarse_openai_v2.json",
    "scripts/bonafide/configs/labeling/prices-2026-08-16-coarse-v2.json",
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
        ["git", *arguments], cwd=root, check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


def _source_revision() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    if Path(_git(root, "rev-parse", "--show-toplevel")) != root:
        raise ValueError("coarse v2 builder repository root drift")
    if _git(root, "status", "--porcelain=v1", "--untracked-files=no"):
        raise ValueError("coarse v2 build requires a clean tracked worktree")
    commit = _git(root, "rev-parse", "HEAD")
    files = []
    for relative in _BOUND_SOURCE_FILES:
        if _git(root, "ls-files", "--error-unmatch", "--", relative) != relative:
            raise ValueError(f"coarse v2 source is untracked: {relative}")
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
            raise ValueError(f"coarse v2 source differs from HEAD: {relative}")
        files.append({"path": relative, "git_blob": blob, "sha256": expected})
    return {
        "repo_root": str(root),
        "git_commit": commit,
        "git_tree": _git(root, "rev-parse", "HEAD^{tree}"),
        "tracked_worktree_clean": True,
        "files": files,
    }


def build(
    *,
    v1_root: Path,
    v1_completed_run_root: Path,
    v1_cost_correction_audit_path: Path,
    workstation_bundle_path: Path,
    config_path: Path,
    destination: Path,
) -> dict[str, Any]:
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    source_revision = _source_revision()
    config = load_coarse_v2_config(config_path)
    if (
        file_sha256(workstation_bundle_path)
        != config["source"]["workstation_bundle_sha256"]
    ):
        raise ValueError("coarse v2 workstation bundle hash drift")
    if file_sha256(v1_root / "windows.json") != config["source"]["v1_windows_sha256"]:
        raise ValueError("coarse v2 v1 windows hash drift")
    if file_sha256(v1_root / "units.jsonl") != config["source"]["v1_units_sha256"]:
        raise ValueError("coarse v2 v1 units hash drift")
    v1_baseline = load_v1_comparison_baseline(v1_completed_run_root, config)
    v1_cost_audit = load_v1_cost_correction_audit(
        v1_cost_correction_audit_path, config, v1_baseline
    )
    workstation = _load_object(workstation_bundle_path)
    qualification = build_v2_qualification(
        v1_root=v1_root, workstation_bundle=workstation, config=config
    )
    price_path = (config_path.parent / config["provider"]["price_snapshot"]).resolve()
    prices = _load_object(price_path)
    plan = cost_plan_v2(qualification["requests"], config, prices)
    plan.update(
        {
            "price_snapshot_path": str(price_path),
            "price_snapshot_sha256": file_sha256(price_path),
            "source_revision": source_revision,
        }
    )
    plan["cost_plan_sha256"] = canonical_sha256(plan)

    temporary = destination.parent / f".{destination.name}.building-{uuid.uuid4().hex}"
    temporary.mkdir(parents=True)
    try:
        atomic_write_jsonl(
            temporary / "focal-units.jsonl", qualification["focal_units"]
        )
        atomic_write_json(temporary / "windows.json", qualification["windows"])
        atomic_write_jsonl(temporary / "requests.jsonl", qualification["requests"])
        atomic_write_jsonl(
            temporary / "batch-input.jsonl", qualification["batch_lines"]
        )
        atomic_write_json(
            temporary / "matched-arm-bindings.json",
            qualification["matched_arm_bindings"],
        )
        atomic_write_json(temporary / "cost-plan.json", plan)
        atomic_write_bytes(
            temporary / "v1-baseline-run-manifest.json",
            (v1_completed_run_root / "run-manifest.json").read_bytes(),
        )
        atomic_write_bytes(
            temporary / "v1-baseline-events.jsonl",
            (v1_completed_run_root / "events.jsonl").read_bytes(),
        )
        atomic_write_bytes(
            temporary / "v1-baseline-cost-correction-audit.json",
            v1_cost_correction_audit_path.read_bytes(),
        )
        leaks = {
            request["request_id"]: forbidden_provider_input_leaks_v2(request)
            for request in qualification["requests"]
        }
        leaks = {key: value for key, value in leaks.items() if value}
        if leaks:
            raise ValueError(f"forbidden provider-input leakage: {leaks}")
        files = [
            {
                "path": path.name,
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(temporary.iterdir())
            if path.name != "manifest.json"
        ]
        requests = qualification["requests"]
        manifest = {
            "schema_version": BUNDLE_SCHEMA,
            "status": "prepared_offline_no_provider_calls",
            "claim_boundary": config["claim_boundary"],
            "qualification_claim_boundary": config["qualification_claim_boundary"],
            "comparison_plan": config["comparison_plan"],
            "comparison_plan_sha256": canonical_sha256(config["comparison_plan"]),
            "config_id": config["config_id"],
            "config_path": str(config_path.resolve()),
            "config_sha256": file_sha256(config_path),
            "source_v1_qualification_root": str(v1_root.resolve()),
            "source_v1_manifest_sha256": config["source"][
                "v1_qualification_manifest_sha256"
            ],
            "source_v1_completed_run_root": str(v1_completed_run_root.resolve()),
            "source_v1_completed_run_manifest_file_sha256": config["source"][
                "v1_completed_run_manifest_file_sha256"
            ],
            "source_v1_completed_run_manifest_sha256": v1_baseline["manifest"][
                "run_manifest_sha256"
            ],
            "source_v1_completed_events_sha256": config["source"][
                "v1_completed_events_sha256"
            ],
            "source_v1_cost_correction_audit_file_sha256": config["source"][
                "v1_cost_correction_audit_file_sha256"
            ],
            "source_v1_cost_correction_audit_sha256": v1_cost_audit[
                "cost_correction_audit_sha256"
            ],
            "source_workstation_bundle": str(workstation_bundle_path.resolve()),
            "source_workstation_bundle_sha256": file_sha256(workstation_bundle_path),
            "counts": {
                "arms": 2,
                "unique_windows": 12,
                "unique_focal_units": 72,
                "physical_requests": 32,
                "repeat_requests": 8,
                "physical_requests_per_arm": 16,
            },
            "request_bindings_in_order": [
                {
                    "request_id": request["request_id"],
                    "arm_id": request["arm_id"],
                    "body_sha256": request["body_sha256"],
                    "source_v1_request_id": request["source_v1_request_id"],
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
    parser.add_argument("--v1-qualification-root", type=Path, required=True)
    parser.add_argument("--v1-completed-run-root", type=Path, required=True)
    parser.add_argument("--v1-cost-correction-audit", type=Path, required=True)
    parser.add_argument("--workstation-bundle", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    manifest = build(
        v1_root=args.v1_qualification_root.resolve(),
        v1_completed_run_root=args.v1_completed_run_root.resolve(),
        v1_cost_correction_audit_path=args.v1_cost_correction_audit.resolve(),
        workstation_bundle_path=args.workstation_bundle.resolve(),
        config_path=args.config.resolve(),
        destination=args.destination.resolve(),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
