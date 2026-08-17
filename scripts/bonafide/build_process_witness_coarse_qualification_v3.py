#!/usr/bin/env python3
"""Build the immutable network-free refined zero-shot/few-shot bundle."""

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
from circuits.analysis.bonafide.coarse_sampling_annotation_v3 import (
    BUNDLE_SCHEMA,
    build_v3_qualification,
    cost_plan_v3,
    forbidden_provider_input_leaks_v3,
    load_coarse_v3_config,
)
from circuits.labeling.io import (
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_jsonl,
)

_BOUND_SOURCE_FILES = (
    "circuits/analysis/bonafide/coarse_sampling_annotation_v3.py",
    "circuits/analysis/bonafide/coarse_sampling_annotation_v2.py",
    "circuits/analysis/bonafide/coarse_sampling_annotation.py",
    "circuits/analysis/bonafide/canonical.py",
    "circuits/labeling/io.py",
    "scripts/bonafide/build_process_witness_coarse_qualification_v3.py",
    "scripts/bonafide/configs/process_witness_coarse_openai_v3.json",
    "scripts/bonafide/configs/labeling/prices-2026-08-16-coarse-v2.json",
)


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments], cwd=root, check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


def _source_revision() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    if Path(_git(root, "rev-parse", "--show-toplevel")) != root:
        raise ValueError("coarse v3 builder repository root drift")
    if _git(root, "status", "--porcelain=v1", "--untracked-files=no"):
        raise ValueError("coarse v3 build requires a clean tracked worktree")
    commit = _git(root, "rev-parse", "HEAD")
    files = []
    for relative in _BOUND_SOURCE_FILES:
        if _git(root, "ls-files", "--error-unmatch", "--", relative) != relative:
            raise ValueError(f"coarse v3 source is untracked: {relative}")
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
            raise ValueError(f"coarse v3 source differs from HEAD: {relative}")
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


def build(
    *,
    workstation_bundle_path: Path,
    development_v2_root: Path,
    development_human_ledger_path: Path,
    config_path: Path,
    destination: Path,
) -> dict[str, Any]:
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    source_revision = _source_revision()
    config = load_coarse_v3_config(config_path)
    if (
        file_sha256(workstation_bundle_path)
        != config["source"]["workstation_bundle_sha256"]
        or file_sha256(development_v2_root / "focal-units.jsonl")
        != config["source"]["development_v2_focal_units_sha256"]
        or file_sha256(development_v2_root / "windows.json")
        != config["source"]["development_v2_windows_sha256"]
        or file_sha256(development_human_ledger_path)
        != config["source"]["development_human_ledger_sha256"]
    ):
        raise ValueError("coarse v3 frozen source file drift")
    workstation = _load_object(workstation_bundle_path)
    qualification = build_v3_qualification(
        workstation_bundle=workstation,
        development_v2_root=development_v2_root,
        config=config,
    )
    price_path = (config_path.parent / config["provider"]["price_snapshot"]).resolve()
    prices = _load_object(price_path)
    plan = cost_plan_v3(qualification["requests"], config, prices)
    plan.update(
        {
            "price_snapshot_source_path": str(price_path),
            "price_snapshot_relative_path": "price-snapshot.json",
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
        atomic_write_json(
            temporary / "exclusion-audit.json", qualification["exclusion_audit"]
        )
        atomic_write_json(temporary / "cost-plan.json", plan)
        atomic_write_bytes(temporary / "protocol-config.json", config_path.read_bytes())
        atomic_write_bytes(temporary / "price-snapshot.json", price_path.read_bytes())
        if (
            len(development_human_ledger_path.read_text(encoding="utf-8").splitlines())
            != 72
        ):
            raise ValueError("coarse v3 development human ledger cardinality drift")
        atomic_write_bytes(
            temporary / "development-human-ledger.jsonl",
            development_human_ledger_path.read_bytes(),
        )
        leaks = {
            request["request_id"]: forbidden_provider_input_leaks_v3(request)
            for request in qualification["requests"]
        }
        leaks = {request_id: values for request_id, values in leaks.items() if values}
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
            "config_source_path": str(config_path.resolve()),
            "protocol_config_relative_path": "protocol-config.json",
            "config_sha256": file_sha256(config_path),
            "source_workstation_bundle": str(workstation_bundle_path.resolve()),
            "source_workstation_bundle_sha256": file_sha256(workstation_bundle_path),
            "source_development_v2_root": str(development_v2_root.resolve()),
            "source_development_v2_manifest_sha256": config["source"][
                "development_v2_manifest_sha256"
            ],
            "source_development_human_ledger_sha256": file_sha256(
                development_human_ledger_path
            ),
            "development_human_ledger_role": (
                "immutable protocol-development evidence only; excluded from holdout "
                "selection, provider inputs, human accuracy denominator, and gate"
            ),
            "counts": {
                "arms": 2,
                "unique_windows": 24,
                "unique_focal_units": 144,
                "physical_requests": 144,
                "replica_requests": 96,
            },
            "request_bindings_in_order": [
                {
                    "request_id": request["request_id"],
                    "arm_id": request["arm_id"],
                    "replica_index": request["replica_index"],
                    "window_index": request["window_index"],
                    "body_sha256": request["body_sha256"],
                    "repeat_of_request_id": request["repeat_of_request_id"],
                }
                for request in requests
            ],
            "files": files,
            "source_revision": source_revision,
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
    parser.add_argument("--workstation-bundle", type=Path, required=True)
    parser.add_argument("--development-v2-root", type=Path, required=True)
    parser.add_argument("--development-human-ledger", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    manifest = build(
        workstation_bundle_path=args.workstation_bundle.resolve(),
        development_v2_root=args.development_v2_root.resolve(),
        development_human_ledger_path=args.development_human_ledger.resolve(),
        config_path=args.config.resolve(),
        destination=args.destination.resolve(),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
