from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from scripts.bonafide.build_candidate_union_c2_plan import (
    C2_BUNDLE_SCHEMA,
    MAX_CASES_PER_WAVE,
    _balanced_shards,
    _bundle_manifest_contracts,
    build_candidate_union_c2_plan,
)
from scripts.bonafide.execution_plan import sha256_file

REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_PATH = (
    REPO_ROOT / "scripts/bonafide/manifests/"
    "qwen3_4b_instruct_topk_c2_launch_bundle_v1.json"
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_c2_balanced_shards_are_bounded_and_deterministic() -> None:
    cases = [
        {
            "case_id": f"case-{index:03d}",
            "estimated_reference_edge_count": (index * 7919) % 100_000,
            "frozen_union_candidate_edge_counts": [
                (index * 3571) % 40_000 + 1,
                (index * 6271) % 40_000 + 1,
            ],
        }
        for index in range(168)
    ]

    first = _balanced_shards(cases, shard_count=11)
    second = _balanced_shards(list(reversed(cases)), shard_count=11)

    assert first == second
    assert len(first) == 11
    assert sum(len(shard) for shard in first) == 168
    assert max(len(shard) for shard in first) <= MAX_CASES_PER_WAVE
    assert (
        max(
            sum(sum(case["frozen_union_candidate_edge_counts"]) for case in shard)
            for shard in first
        )
        / min(
            sum(sum(case["frozen_union_candidate_edge_counts"]) for case in shard)
            for shard in first
        )
        < 1.1
    )


def test_c2_bundle_contract_binds_all_manifests_and_rejects_cohort_drift(
    tmp_path: Path,
) -> None:
    bundle = _load_json(BUNDLE_PATH)
    copied = copy.deepcopy(bundle)
    for index, record in enumerate(copied["manifests"]):
        manifest = _load_json(Path(record["path"]))
        if index == 0:
            manifest["cohort"]["cohort_id"] = "drifted-cohort"
        path = tmp_path / f"candidate-{index}.json"
        path.write_text(
            json.dumps(manifest, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        record["path"] = str(path)
        record["sha256"] = sha256_file(path)

    with pytest.raises(ValueError, match="disagree on source or cohort"):
        _bundle_manifest_contracts(copied)


def test_c2_builder_rejects_invalid_config_before_loading_references() -> None:
    bundle = {"schema_version": C2_BUNDLE_SCHEMA}
    with pytest.raises(ValueError, match="run config"):
        build_candidate_union_c2_plan(
            bundle,
            {},
            {},
            {},
            bundle_path=BUNDLE_PATH,
            bundle_sha256="0" * 64,
            selection_path=BUNDLE_PATH,
            selection_sha256="0" * 64,
            candidate_zero_manifest_path=BUNDLE_PATH,
            candidate_zero_manifest_sha256="0" * 64,
            config_path=BUNDLE_PATH,
            config_sha256="0" * 64,
            reference_root=REPO_ROOT,
        )
