from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import pytest
from circuits.analysis.bonafide import coarse_sampling_post_campaign_v2 as sampling_v2
from circuits.analysis.bonafide.canonical import canonical_sha256, file_sha256


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def _minimal_manifest(**overrides: object) -> dict[str, object]:
    manifest: dict[str, object] = {
        "schema_version": sampling_v2.ANALYSIS_SCHEMA,
        "status": "frozen_candidate_designs_not_selected_for_tracing",
        "inventory_sha256": "0" * 64,
        "execution_source_revision": {
            "git_commit": "a" * 40,
            "git_tree": "b" * 40,
            "execution_source_subset_matches_head": True,
            "files": [],
        },
        "candidate_design_status": "frozen",
        "candidate_tier_status": "frozen_candidate_only",
        "selected_for_tracing": False,
        "trace_ready": False,
        "trace_policy_selection_status": "pending_audit_and_resource_gate",
        "network_calls_made": 0,
        "parent_v1_mutated": False,
        "design_contract_sha256": "1" * 64,
        "expected_frontiers_sha256": "2" * 64,
        "realized_candidate_tiers_sha256": "3" * 64,
    }
    manifest.update(overrides)
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    return manifest


def _overlapping_test_kernels() -> dict[str, dict[str, dict[int, float]]]:
    kernels: dict[str, dict[str, dict[int, float]]] = {}
    for group_index in range(10):
        kernels[f"g{group_index}"] = {
            mechanism: {group_index * 2: 0.5, group_index * 2 + 1: 0.5}
            for mechanism in sampling_v2.MECHANISMS
        }
    return kernels


def _equal_test_bases() -> dict[str, dict[str, float]]:
    return {
        mechanism: {f"g{group_index}": 0.1 for group_index in range(10)}
        for mechanism in sampling_v2.MECHANISMS
    }


def test_overlap_frontiers_solve_expected_unique_owner_shares(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shares = {
        "balanced": {
            "process_enrichment": 0.20,
            "evaluation_commitment": 0.20,
            "diversity": 0.20,
            "uncertainty_missing": 0.20,
            "uniform_reserve": 0.20,
        }
    }
    monkeypatch.setattr(sampling_v2, "SHARES", shares)
    monkeypatch.setattr(sampling_v2, "BUDGETS", (2, 3, 4))

    frontiers, candidates, _solutions = sampling_v2.build_overlap_frontiers_v2(
        _overlapping_test_kernels(), _equal_test_bases()
    )

    assert [row["nominal_expected_unique_target_budget"] for row in frontiers] == [
        2,
        3,
        4,
    ]
    prior_rates: dict[str, float] = defaultdict(float)
    for frontier in frontiers:
        budget = frontier["nominal_expected_unique_target_budget"]
        assert math.isclose(
            frontier["expected_unique_target_positions"],
            budget,
            rel_tol=0,
            abs_tol=1e-8,
        )
        assert frontier["candidate_design_status"] == "frozen"
        assert frontier["selected_for_tracing"] is False
        assert frontier["trace_ready"] is False
        assert (
            frontier["trace_policy_selection_status"]
            == "pending_audit_and_resource_gate"
        )
        for mechanism in sampling_v2.OWNERSHIP_ORDER:
            diagnostic = frontier["mechanism_diagnostics"][mechanism]
            assert math.isclose(
                diagnostic["expected_first_owner_unique_positions"],
                budget * shares["balanced"][mechanism],
                rel_tol=0,
                abs_tol=1e-8,
            )
            assert frontier["poisson_rates"][mechanism] >= prior_rates[mechanism]
            prior_rates[mechanism] = frontier["poisson_rates"][mechanism]
        assert (
            frontier["expected_raw_arrivals"]
            >= frontier["expected_unique_target_positions"]
        )
        assert frontier["expected_within_route_collisions"] >= 0
        assert frontier["expected_cross_route_collisions"] >= 0

        designs = [
            design
            for rows in candidates.values()
            for design in rows
            if design["nominal_expected_unique_target_budget"] == budget
        ]
        assert math.isclose(
            sum(design["marginal_inclusion_probability"] for design in designs),
            budget,
            rel_tol=0,
            abs_tol=1e-8,
        )
        assert all(
            math.isclose(
                design["inverse_probability_weight"],
                1.0 / design["marginal_inclusion_probability"],
            )
            for design in designs
        )


def test_reviewed_mechanism_capacity_contract_is_explicit() -> None:
    assert sampling_v2.EXPECTED_FRAME == {
        "psus": 94_479,
        "atoms": 94_546,
        "positions": 842_007,
    }
    assert sampling_v2.EXPECTED_ELIGIBILITY == {
        "process_enrichment": {
            "psus": 20_330,
            "atoms": 20_373,
            "positions": 304_951,
        },
        "evaluation_commitment": {
            "psus": 17_441,
            "atoms": 17_459,
            "positions": 171_041,
        },
        "uncertainty_missing": {
            "psus": 30_842,
            "atoms": 30_908,
            "positions": 284_695,
        },
        "diversity": {
            "psus": 74_979,
            "atoms": 75_046,
            "positions": 820_236,
        },
        "uniform_reserve": sampling_v2.EXPECTED_FRAME,
    }


def test_position_kernels_preserve_group_atom_position_design() -> None:
    frame = [
        {
            "psu_id": "g0",
            "response_id": "r0",
            "eligible_mechanisms": list(sampling_v2.MECHANISMS),
            "atoms": [
                {
                    "unit_id": "u-short",
                    "token_span": [0, 1],
                    "eligible_mechanisms": list(sampling_v2.MECHANISMS),
                },
                {
                    "unit_id": "u-long",
                    "token_span": [1, 4],
                    "eligible_mechanisms": list(sampling_v2.MECHANISMS),
                },
            ],
        }
    ]
    units = {
        "u-short": {
            "unit_id": "u-short",
            "core_character_span": [0, 1],
            "token_span": [0, 1],
        },
        "u-long": {
            "unit_id": "u-long",
            "core_character_span": [2, 5],
            "token_span": [1, 4],
        },
    }
    documents = {
        "r0": {
            "response_id": "r0",
            "text": "1 abc",
            "tokenization": {"tokens": [[1, 0, 1], [2, 2, 3], [3, 3, 4], [4, 4, 5]]},
        }
    }

    kernels = sampling_v2.build_position_kernels_v2(frame, units, documents)["g0"]

    assert kernels["uniform_reserve"] == pytest.approx(
        {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
    )
    assert kernels["diversity"] == pytest.approx({0: 0.5, 1: 1 / 6, 2: 1 / 6, 3: 1 / 6})
    assert kernels["evaluation_commitment"] == pytest.approx(
        {
            0: 0.5,
            1: 0.5 * (0.30 + 0.40 / 3),
            2: 0.5 * (0.40 / 3),
            3: 0.5 * (0.30 + 0.40 / 3),
        }
    )
    assert kernels["uncertainty_missing"] == pytest.approx(
        {
            0: 0.5,
            1: 0.5 * (0.25 + 0.50 / 3),
            2: 0.5 * (0.50 / 3),
            3: 0.5 * (0.25 + 0.50 / 3),
        }
    )
    # The one-token first atom is an observable numeric anchor. Both atoms
    # still receive equal atom mass before their within-atom mixtures apply.
    assert kernels["process_enrichment"] == pytest.approx(
        {
            0: 0.5,
            1: 0.5 * (0.20 + 0.65 / 3),
            2: 0.5 * (0.65 / 3),
            3: 0.5 * (0.15 + 0.65 / 3),
        }
    )


def test_hierarchical_bases_make_uniform_route_uniform_per_position() -> None:
    frame = [
        {
            "psu_id": "g0",
            "position_count": 1,
            "source_key": "source-a",
            "prompt_sha256": "prompt-1",
            "response_id": "response-1",
            "response_relative_third": 0,
            "eligible_mechanisms": list(sampling_v2.MECHANISMS),
        },
        {
            "psu_id": "g1",
            "position_count": 3,
            "source_key": "source-a",
            "prompt_sha256": "prompt-1",
            "response_id": "response-1",
            "response_relative_third": 0,
            "eligible_mechanisms": list(sampling_v2.MECHANISMS),
        },
        {
            "psu_id": "g2",
            "position_count": 2,
            "source_key": "source-a",
            "prompt_sha256": "prompt-2",
            "response_id": "response-2",
            "response_relative_third": 1,
            "eligible_mechanisms": list(sampling_v2.MECHANISMS),
        },
        {
            "psu_id": "g3",
            "position_count": 4,
            "source_key": "source-b",
            "prompt_sha256": "prompt-3",
            "response_id": "response-3",
            "response_relative_third": 2,
            "eligible_mechanisms": list(sampling_v2.MECHANISMS),
        },
    ]

    bases = sampling_v2.build_hierarchical_group_bases_v2(frame)

    assert bases["uniform_reserve"] == {
        "g0": 1.0,
        "g1": 3.0,
        "g2": 2.0,
        "g3": 4.0,
    }
    for mechanism in sampling_v2.MECHANISMS:
        if mechanism != "uniform_reserve":
            assert math.isclose(sum(bases[mechanism].values()), 1.0)
    assert bases["process_enrichment"] == pytest.approx(
        {"g0": 0.125, "g1": 0.125, "g2": 0.25, "g3": 0.5}
    )

    uniform_kernels = {
        "g0": {0: 1.0},
        "g1": {1: 1 / 3, 2: 1 / 3, 3: 1 / 3},
        "g2": {4: 0.5, 5: 0.5},
        "g3": {6: 0.25, 7: 0.25, 8: 0.25, 9: 0.25},
    }
    intensities = {
        position: bases["uniform_reserve"][group] * conditional
        for group, kernel in uniform_kernels.items()
        for position, conditional in kernel.items()
    }
    assert set(intensities.values()) == {1.0}


def test_realized_target_ids_are_nested_candidate_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sampling_v2,
        "SHARES",
        {
            "balanced": {
                "process_enrichment": 0.20,
                "evaluation_commitment": 0.20,
                "diversity": 0.20,
                "uncertainty_missing": 0.20,
                "uniform_reserve": 0.20,
            }
        },
    )
    monkeypatch.setattr(sampling_v2, "BUDGETS", (2, 3, 4))
    kernels = _overlapping_test_kernels()
    _frontiers, _candidates, solutions = sampling_v2.build_overlap_frontiers_v2(
        kernels, _equal_test_bases()
    )

    rows, summary = sampling_v2.build_realized_tiers_v2(
        kernels, _equal_test_bases(), solutions
    )

    target_ids = {
        budget: {
            row["target_id"]
            for row in rows
            if row["nominal_expected_unique_target_budget"] == budget
        }
        for budget in (2, 3, 4)
    }
    assert target_ids[2] <= target_ids[3] <= target_ids[4]
    assert summary["target_identity_nesting_verified"] is True
    assert summary["candidate_design_status"] == "frozen"
    assert summary["candidate_tier_status"] == "frozen_candidate_only"
    assert summary["selected_for_tracing"] is False
    assert summary["trace_ready"] is False
    assert all(row["selected_for_tracing"] is False for row in rows)
    assert all(0 < row["marginal_inclusion_probability"] <= 1 for row in rows)
    assert all(
        math.isclose(
            row["inverse_probability_weight"],
            1 / row["marginal_inclusion_probability"],
        )
        for row in rows
    )
    for tier in summary["tiers"]:
        per_mechanism = tier["per_mechanism"]
        assert set(per_mechanism) == set(sampling_v2.MECHANISMS)
        assert (
            sum(values["realized_raw_arrivals"] for values in per_mechanism.values())
            == tier["realized_raw_arrivals"]
        )
        assert (
            sum(
                values["realized_route_unique_positions"]
                for values in per_mechanism.values()
            )
            == tier["realized_route_unique_positions"]
        )
        assert (
            sum(
                values["realized_first_owner_unique_positions"]
                for values in per_mechanism.values()
            )
            == tier["realized_unique_target_positions"]
        )
        for mechanism, values in per_mechanism.items():
            expected = (
                tier["nominal_expected_unique_target_budget"]
                * sampling_v2.SHARES[tier["policy"]][mechanism]
            )
            assert values["expected_first_owner_unique_positions"] == expected
            assert values["first_owner_absolute_deviation"] == (
                values["realized_first_owner_unique_positions"] - expected
            )


def test_copied_execution_source_recomputes_git_blob_identity(
    tmp_path: Path,
) -> None:
    copied = tmp_path / "execution-source/runtime.py"
    copied.parent.mkdir()
    source_bytes = b"VALUE = 1\n"
    copied.write_bytes(source_bytes)
    blob_id = hashlib.sha1(
        b"blob " + str(len(source_bytes)).encode() + b"\0" + source_bytes
    ).hexdigest()
    revision = {
        "git_commit": "a" * 40,
        "git_tree": "b" * 40,
        "execution_source_subset_matches_head": True,
        "files": [
            {
                "path": "runtime.py",
                "copied_path": "execution-source/runtime.py",
                "git_blob": blob_id,
                "sha256": hashlib.sha256(source_bytes).hexdigest(),
                "bytes": len(source_bytes),
            }
        ],
    }

    sampling_v2._validate_execution_source(tmp_path, revision)

    revision["files"][0]["git_blob"] = "c" * 40
    with pytest.raises(ValueError, match="copied execution source drift"):
        sampling_v2._validate_execution_source(tmp_path, revision)

    revision["files"][0]["git_blob"] = blob_id
    revision["git_tree"] = "not-a-git-tree"
    with pytest.raises(ValueError, match="execution-source subset gate"):
        sampling_v2._validate_execution_source(tmp_path, revision)

    revision["git_tree"] = "b" * 40
    copied.write_bytes(b"VALUE = 2\n")
    with pytest.raises(ValueError, match="copied execution source drift"):
        sampling_v2._validate_execution_source(tmp_path, revision)


def test_public_loader_rejects_manifest_that_promotes_tracing(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    _write_json(
        root / "manifest.json",
        _minimal_manifest(selected_for_tracing=True, trace_ready=True),
    )

    with pytest.raises(ValueError, match="candidate-only manifest drift"):
        sampling_v2.load_frozen_post_campaign_sampling_v2(root)


def test_public_loader_rejects_duplicate_inventory_paths(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    duplicate = {"path": "duplicate.json", "bytes": 0, "sha256": "0" * 64}
    inventory: dict[str, object] = {
        "schema_version": sampling_v2.INVENTORY_SCHEMA,
        "files": [duplicate, duplicate],
    }
    inventory["inventory_sha256"] = canonical_sha256(inventory)
    manifest = _minimal_manifest(inventory_sha256=inventory["inventory_sha256"])
    _write_json(root / "manifest.json", manifest)
    _write_json(root / "evidence-inventory.json", inventory)

    with pytest.raises(ValueError, match="duplicate inventory path"):
        sampling_v2.load_frozen_post_campaign_sampling_v2(root)


def test_public_loader_rejects_static_design_contract_drift(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    contract: dict[str, object] = {
        "schema_version": "adag.process-witness.coarse-sampling-design-contract.v2",
        "mechanisms_plan_order": list(sampling_v2.MECHANISMS),
        "first_owner_precedence": list(sampling_v2.OWNERSHIP_ORDER),
        "shares": sampling_v2.SHARES,
        "budgets": list(sampling_v2.BUDGETS),
        "kernel_stream_sha256": "4" * 64,
        "group_base_stream_sha256": "5" * 64,
        "group_to_atom_to_position": True,
        "uniform_each_position_probability_equal": True,
        "observable_process_anchors_only": True,
        "halo_status": "incorrectly_enabled",
        "candidate_design_status": "frozen",
        "selected_for_tracing": False,
        "trace_ready": False,
    }
    _write_json(root / "design-contract.json", contract)
    binding = {
        "path": "design-contract.json",
        "bytes": (root / "design-contract.json").stat().st_size,
        "sha256": file_sha256(root / "design-contract.json"),
    }
    inventory: dict[str, object] = {
        "schema_version": sampling_v2.INVENTORY_SCHEMA,
        "files": [binding],
    }
    inventory["inventory_sha256"] = canonical_sha256(inventory)
    manifest = _minimal_manifest(
        inventory_sha256=inventory["inventory_sha256"],
        design_contract_sha256=binding["sha256"],
    )
    _write_json(root / "manifest.json", manifest)
    _write_json(root / "evidence-inventory.json", inventory)

    with pytest.raises(ValueError, match="design contract drift"):
        sampling_v2.load_frozen_post_campaign_sampling_v2(root)


def test_public_loader_rejects_descendant_directory_symlink(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "uninventoried-directory-link").symlink_to(
        outside, target_is_directory=True
    )

    with pytest.raises(ValueError, match="descendant symlink"):
        sampling_v2.load_frozen_post_campaign_sampling_v2(root)


def test_public_loader_rejects_symlinked_artifact_root(tmp_path: Path) -> None:
    real_root = tmp_path / "real-artifact"
    real_root.mkdir()
    linked_root = tmp_path / "linked-artifact"
    linked_root.symlink_to(real_root, target_is_directory=True)

    with pytest.raises(ValueError, match="root may not be a symlink"):
        sampling_v2.load_frozen_post_campaign_sampling_v2(linked_root)


def test_public_build_does_not_replace_preexisting_destination(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "artifact"
    destination.mkdir()
    marker = destination / "owned-by-user"
    marker.write_text("preserve me\n")

    with pytest.raises(FileExistsError, match="destination exists"):
        sampling_v2.build_post_campaign_sampling_v2(
            parent_v1_root=tmp_path / "missing-parent",
            destination=destination,
            cohort_root=tmp_path / "missing-cohort",
        )

    assert marker.read_text() == "preserve me\n"
