#!/usr/bin/env python3
"""Build the immutable network-free v4 segmentation compatibility bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any

from circuits.analysis.bonafide.canonical import canonical_sha256, file_sha256
from circuits.analysis.bonafide.coarse_sampling_annotation_v4 import (
    BUNDLE_SCHEMA,
    build_v4_qualification,
    cost_plan_v4,
    forbidden_provider_input_leaks_v4,
    load_coarse_v4_config,
)
from circuits.labeling.io import (
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_jsonl,
)

_BOUND_SOURCE_FILES = (
    "circuits/analysis/bonafide/coarse_sampling_annotation_v4.py",
    "circuits/analysis/bonafide/coarse_sampling_annotation_v3.py",
    "circuits/analysis/bonafide/coarse_sampling_review_v3.py",
    "circuits/analysis/bonafide/coarse_sampling_annotation_v2.py",
    "circuits/analysis/bonafide/coarse_sampling_annotation.py",
    "circuits/analysis/bonafide/canonical.py",
    "circuits/analysis/bonafide/process_annotation.py",
    "circuits/labeling/io.py",
    "scripts/bonafide/build_process_witness_coarse_qualification_v4.py",
    "scripts/bonafide/configs/process_witness_coarse_openai_v4.json",
    "scripts/bonafide/configs/labeling/prices-2026-08-16-coarse-v2.json",
    "experiments/process_witness/v3_human_review_post_seal_corrections_v1.jsonl",
)


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def _source_revision() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    if Path(_git(root, "rev-parse", "--show-toplevel")) != root:
        raise ValueError("coarse v4 builder repository root drift")
    if _git(root, "status", "--porcelain=v1", "--untracked-files=no"):
        raise ValueError("coarse v4 build requires a clean tracked worktree")
    commit = _git(root, "rev-parse", "HEAD")
    files = []
    for relative in _BOUND_SOURCE_FILES:
        if _git(root, "ls-files", "--error-unmatch", "--", relative) != relative:
            raise ValueError(f"coarse v4 source is untracked: {relative}")
        path = root / relative
        blob = _git(root, "rev-parse", f"{commit}:{relative}")
        committed = subprocess.run(
            ["git", "cat-file", "blob", blob], cwd=root, check=True, capture_output=True
        ).stdout
        expected = hashlib.sha256(committed).hexdigest()
        if file_sha256(path) != expected:
            raise ValueError(f"coarse v4 source differs from HEAD: {relative}")
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
    v3_qualification_root: Path,
    v3_review_root: Path,
    human_ledger_path: Path,
    correction_path: Path,
    config_path: Path,
    destination: Path,
) -> dict[str, Any]:
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    source_revision = _source_revision()
    config = load_coarse_v4_config(config_path)
    v3_manifest = _load_object(v3_qualification_root / "manifest.json")
    review_manifest = _load_object(v3_review_root / "manifest.json")
    if (
        file_sha256(workstation_bundle_path)
        != config["source"]["workstation_bundle_sha256"]
        or v3_manifest.get("manifest_sha256")
        != config["source"]["v3_qualification_manifest_sha256"]
        or review_manifest.get("manifest_sha256")
        != config["source"]["v3_review_packet_manifest_sha256"]
        or file_sha256(human_ledger_path) != config["source"]["v3_human_ledger_sha256"]
        or file_sha256(correction_path)
        != config["source"]["v3_post_seal_corrections_sha256"]
    ):
        raise ValueError("coarse v4 frozen source drift")
    workstation = _load_object(workstation_bundle_path)
    qualification = build_v4_qualification(
        workstation_bundle=workstation,
        review_root=v3_review_root,
        human_ledger_path=human_ledger_path,
        correction_path=correction_path,
        config=config,
    )
    price_path = (config_path.parent / config["provider"]["price_snapshot"]).resolve()
    prices = _load_object(price_path)
    plan = cost_plan_v4(qualification["requests"], config, prices)
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
            temporary / "replica-bindings.json", qualification["replica_bindings"]
        )
        atomic_write_json(
            temporary / "segmentation-repair-audit.json", qualification["repair_audit"]
        )
        atomic_write_json(
            temporary / "full-corpus-segmentation-audit.json",
            qualification["full_corpus_segmentation_audit"],
        )
        atomic_write_json(temporary / "cost-plan.json", plan)
        atomic_write_bytes(temporary / "protocol-config.json", config_path.read_bytes())
        atomic_write_bytes(temporary / "price-snapshot.json", price_path.read_bytes())
        atomic_write_bytes(
            temporary / "v3-human-ledger.jsonl", human_ledger_path.read_bytes()
        )
        atomic_write_bytes(
            temporary / "v3-post-seal-corrections.jsonl", correction_path.read_bytes()
        )
        leaks = {
            r["request_id"]: forbidden_provider_input_leaks_v4(r)
            for r in qualification["requests"]
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
            "compatibility_gate": config["compatibility_gate"],
            "config_id": config["config_id"],
            "protocol_config_relative_path": "protocol-config.json",
            "config_sha256": file_sha256(config_path),
            "source_workstation_bundle": str(workstation_bundle_path.resolve()),
            "source_workstation_bundle_sha256": file_sha256(workstation_bundle_path),
            "source_v3_qualification_root": str(v3_qualification_root.resolve()),
            "source_v3_qualification_manifest_sha256": v3_manifest["manifest_sha256"],
            "source_v3_review_root": str(v3_review_root.resolve()),
            "source_v3_review_packet_manifest_sha256": review_manifest[
                "manifest_sha256"
            ],
            "source_v3_human_ledger_sha256": file_sha256(human_ledger_path),
            "source_v3_post_seal_corrections_sha256": file_sha256(correction_path),
            "counts": {
                "arms": 1,
                "unique_windows": 15,
                "unique_focal_units": 24,
                "physical_requests": 45,
                "replica_requests": 30,
            },
            "request_bindings_in_order": [
                {
                    key: request[key]
                    for key in (
                        "request_id",
                        "arm_id",
                        "replica_index",
                        "window_index",
                        "body_sha256",
                        "repeat_of_request_id",
                    )
                }
                for request in requests
            ],
            "source_revision": source_revision,
            "files": files,
            "network_calls_made": 0,
            "v3_artifacts_mutated": False,
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
    parser.add_argument("--v3-qualification-root", type=Path, required=True)
    parser.add_argument("--v3-review-root", type=Path, required=True)
    parser.add_argument("--human-ledger", type=Path, required=True)
    parser.add_argument("--corrections", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    manifest = build(
        workstation_bundle_path=args.workstation_bundle.resolve(),
        v3_qualification_root=args.v3_qualification_root.resolve(),
        v3_review_root=args.v3_review_root.resolve(),
        human_ledger_path=args.human_ledger.resolve(),
        correction_path=args.corrections.resolve(),
        config_path=args.config.resolve(),
        destination=args.destination.resolve(),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
