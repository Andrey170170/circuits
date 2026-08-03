"""Deterministic inventory of frozen BonaFide width-one trace artifacts."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, cast

from circuits.analysis.bonafide.canonical import (
    canonical_sha256,
    file_sha256,
    load_json_object,
    write_hashed_json,
)
from circuits.analysis.bonafide.partition import CorpusRole, validate_partition_contract

INVENTORY_SCHEMA = "adag.bonafide.atlas-inventory.v1"
SELECTION_SCHEMA = "bonafide-trace-benchmark/v1"
EXECUTION_PLAN_SCHEMA = "bonafide-trace-execution-plan/v1"

ValidationLevel = Literal["integrity", "full"]


class InventoryStatus(StrEnum):
    DISCOVERY = "discovery"
    HOLDOUT = "holdout"
    EXCLUDED_PATHOLOGICAL = "excluded_pathological"
    MISSING = "missing"
    CORRUPT = "corrupt"


@dataclass(frozen=True)
class ExpectedTarget:
    source_artifact_id: str
    source_wave_id: str
    base_question_id: str
    example_id: str
    corpus_role: CorpusRole
    cluster_fit_eligible: bool
    response_position: int
    target_token_id: int
    target_selection: Mapping[str, Any]
    selection_reasons: Sequence[Mapping[str, Any]]
    condition: Mapping[str, Any]
    source_item: Mapping[str, Any]

    @property
    def sort_key(self) -> tuple[str, str, int, str]:
        return (
            self.base_question_id,
            self.example_id,
            self.response_position,
            self.source_artifact_id,
        )


@dataclass(frozen=True)
class PhysicalArtifact:
    path: Path
    manifest: Mapping[str, Any] | None
    read_error: str | None

    @property
    def source_artifact_id(self) -> str | None:
        if self.manifest is None:
            return None
        value = self.manifest.get("source_artifact_id")
        return value if isinstance(value, str) else None


def _integer(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return cast(Mapping[str, Any], value)


def _sequence(value: object, field: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    return value


def _validate_execution_plan(
    plan: Mapping[str, Any],
    *,
    selection_path: Path,
    selection: Mapping[str, Any],
) -> set[str]:
    if plan.get("schema_version") != EXECUTION_PLAN_SCHEMA:
        raise ValueError("unsupported final-trace execution-plan schema")
    recorded_hash = plan.get("plan_sha256")
    core = dict(plan)
    core.pop("plan_sha256", None)
    if recorded_hash != canonical_sha256(core):
        raise ValueError("final-trace execution-plan canonical hash mismatch")

    sources = _mapping(plan.get("sources"), "execution plan sources")
    final_source = _mapping(
        sources.get("final_trace_manifest"),
        "execution plan final_trace_manifest source",
    )
    if final_source.get("sha256") != file_sha256(selection_path):
        raise ValueError(
            "execution plan records a different selection-manifest file hash"
        )
    if final_source.get("canonical_sha256") != canonical_sha256(selection):
        raise ValueError(
            "execution plan records a different selection-manifest canonical hash"
        )

    extremes = _mapping(plan.get("extremes"), "execution plan extremes")
    pathological = _sequence(
        extremes.get("manual_pathological"),
        "execution plan manual_pathological",
    )
    excluded: set[str] = set()
    for item in pathological:
        record = _mapping(item, "manual pathological record")
        source_id = _string(
            record.get("source_artifact_id"),
            "manual pathological source_artifact_id",
        )
        if source_id in excluded:
            raise ValueError("duplicate pathological source_artifact_id")
        excluded.add(source_id)
    return excluded


def _flatten_expected_targets(
    selection: Mapping[str, Any],
) -> tuple[list[ExpectedTarget], dict[str, Any]]:
    if selection.get("schema_version") != SELECTION_SCHEMA:
        raise ValueError("unsupported frozen final-trace selection schema")
    tokenizer = _mapping(selection.get("tokenizer"), "selection tokenizer")
    tokenizer_identity = {
        "model_id": _string(tokenizer.get("model_id"), "tokenizer model_id"),
        "revision": _string(tokenizer.get("revision"), "tokenizer revision"),
        "file_manifest_aggregate_sha256": _string(
            _mapping(
                tokenizer.get("file_manifest"),
                "tokenizer file_manifest",
            ).get("aggregate_sha256"),
            "tokenizer file-manifest aggregate_sha256",
        ),
        "chat_template_sha256": _string(
            tokenizer.get("chat_template_sha256"),
            "tokenizer chat_template_sha256",
        ),
    }

    expected: list[ExpectedTarget] = []
    seen_source_ids: set[str] = set()
    waves = _sequence(selection.get("waves"), "selection waves")
    for wave_value in waves:
        wave = _mapping(wave_value, "selection wave")
        wave_id = _string(wave.get("wave_id"), "wave_id")
        example_id = _string(wave.get("example_id"), "wave example_id")
        fit_eligible = wave.get("cluster_fit_eligible")
        if not isinstance(fit_eligible, bool):
            raise ValueError("wave cluster_fit_eligible must be boolean")
        role = validate_partition_contract(
            _string(wave.get("corpus_role"), "wave corpus_role"),
            cluster_fit_eligible=fit_eligible,
        )
        items = _sequence(wave.get("items"), "wave items")
        for item_value in items:
            item = _mapping(item_value, "selection item")
            source_id = _string(item.get("artifact_id"), "selection artifact_id")
            if source_id in seen_source_ids:
                raise ValueError(f"duplicate selection artifact_id: {source_id}")
            seen_source_ids.add(source_id)
            example = _mapping(item.get("example"), "selection item example")
            if example.get("example_id") != example_id:
                raise ValueError("wave/item example_id mismatch")
            base_question_id = _string(
                example.get("base_question_id"),
                "selection item base_question_id",
            )
            target_selection = _mapping(
                item.get("target_selection"),
                "selection target_selection",
            )
            if _integer(target_selection.get("width"), "target width") != 1:
                raise ValueError("inventory accepts only frozen width-one targets")
            positions = _sequence(
                target_selection.get("response_token_positions"),
                "target response_token_positions",
            )
            if len(positions) != 1:
                raise ValueError(
                    "width-one target requires exactly one response position"
                )
            position = _integer(positions[0], "target response position")
            final_selection = _mapping(
                target_selection.get("final_selection"),
                "target final_selection",
            )
            if final_selection.get("corpus_role") != role.value:
                raise ValueError("wave/item corpus role mismatch")
            reasons = _sequence(
                final_selection.get("selection_reasons"),
                "target selection_reasons",
            )
            diversity_value = example.get("diversity", {})
            diversity = (
                dict(diversity_value) if isinstance(diversity_value, Mapping) else {}
            )
            condition = {
                "src_types": list(example.get("src_types", [])),
                "hint_types": list(example.get("hint_types", [])),
                "hint_datasets": list(example.get("hint_datasets", [])),
                "diversity": diversity,
            }
            expected.append(
                ExpectedTarget(
                    source_artifact_id=source_id,
                    source_wave_id=wave_id,
                    base_question_id=base_question_id,
                    example_id=example_id,
                    corpus_role=role,
                    cluster_fit_eligible=fit_eligible,
                    response_position=position,
                    target_token_id=_integer(
                        target_selection.get("final_target_token_id"),
                        "final_target_token_id",
                    ),
                    target_selection=target_selection,
                    selection_reasons=tuple(
                        _mapping(reason, "selection reason") for reason in reasons
                    ),
                    condition=condition,
                    source_item=item,
                )
            )
    return sorted(expected, key=lambda target: target.sort_key), tokenizer_identity


def _candidate_artifact_directories(root: Path) -> list[Path]:
    if not root.is_dir():
        raise ValueError(f"compact artifact root is not a directory: {root}")
    candidates = {
        manifest.parent.resolve()
        for manifest in root.rglob("manifest.json")
        if manifest.is_file()
    }
    candidates.update(
        path.resolve()
        for path in root.rglob("trace-*")
        if path.is_dir() and not path.name.startswith(".")
    )
    return sorted(candidates)


def _read_physical_artifact(path: Path) -> PhysicalArtifact:
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        return PhysicalArtifact(path, None, "manifest.json is missing")
    try:
        return PhysicalArtifact(path, load_json_object(manifest_path), None)
    except ValueError as error:
        return PhysicalArtifact(path, None, str(error))


def _validate_scientific_match(
    actual: PhysicalArtifact,
    expected: ExpectedTarget,
    *,
    tokenizer_identity: Mapping[str, Any],
    validation_level: ValidationLevel,
) -> Mapping[str, Any]:
    from circuits.tracing.artifact import (
        load_compact_trace,
        validate_compact_trace_integrity,
    )

    manifest = validate_compact_trace_integrity(actual.path)
    if validation_level == "full":
        load_compact_trace(actual.path)
    elif validation_level != "integrity":
        raise ValueError(f"unsupported validation level: {validation_level!r}")

    exact_fields = {
        "source_artifact_id": expected.source_artifact_id,
        "benchmark_wave_id": expected.source_wave_id,
        "model_id": tokenizer_identity["model_id"],
        "model_revision": tokenizer_identity["revision"],
        "target_count": 1,
        "scientifically_reusable": True,
        "numerically_valid": True,
        "benchmark_only": False,
    }
    for field, expected_value in exact_fields.items():
        if manifest.get(field) != expected_value:
            raise ValueError(
                f"compact manifest {field} mismatch: "
                f"{manifest.get(field)!r} != {expected_value!r}"
            )
    if manifest.get("source_target_selection") != expected.target_selection:
        raise ValueError("compact manifest source_target_selection mismatch")
    example = _mapping(manifest.get("bonafide_example"), "bonafide_example")
    if example.get("example_id") != expected.example_id:
        raise ValueError("compact manifest example_id mismatch")
    if example.get("base_question_id") != expected.base_question_id:
        raise ValueError("compact manifest base_question_id mismatch")
    identity = _mapping(manifest.get("artifact_identity"), "artifact_identity")
    if identity.get("source_work_item_sha256") != canonical_sha256(
        expected.source_item
    ):
        raise ValueError("compact manifest source work-item hash mismatch")
    provenance_values = _sequence(
        manifest.get("target_provenance"),
        "compact target_provenance",
    )
    if len(provenance_values) != 1:
        raise ValueError("compact target_provenance must contain exactly one target")
    provenance = _mapping(provenance_values[0], "compact target provenance")
    if provenance.get("response_token_position") != expected.response_position:
        raise ValueError("compact target response position mismatch")
    if provenance.get("token_id") != expected.target_token_id:
        raise ValueError("compact target token ID mismatch")
    runtime_id = _string(manifest.get("artifact_id"), "compact artifact_id")
    if actual.path.name != runtime_id:
        raise ValueError("compact artifact directory/runtime artifact_id mismatch")
    _string(identity.get("sha256"), "trace configuration identity")
    if identity.get("source_artifact_id") not in {
        None,
        expected.source_artifact_id,
    }:
        raise ValueError("artifact identity source_artifact_id mismatch")
    code_revision = _mapping(manifest.get("code_revision"), "code_revision")
    identity_code_revision = identity.get("code_revision")
    if identity_code_revision is not None and identity_code_revision != code_revision:
        raise ValueError("top-level and identity code revisions disagree")
    return manifest


def _base_record(
    expected: ExpectedTarget,
    *,
    selection_sha256: str,
    execution_plan_sha256: str,
) -> dict[str, Any]:
    return {
        "atlas_trace_index": None,
        "source_artifact_id": expected.source_artifact_id,
        "source_wave_id": expected.source_wave_id,
        "base_question_id": expected.base_question_id,
        "example_id": expected.example_id,
        "response_id": expected.example_id,
        "condition": dict(expected.condition),
        "corpus_role": expected.corpus_role.value,
        "cluster_fit_eligible": expected.cluster_fit_eligible,
        "response_position": expected.response_position,
        "target_token_id": expected.target_token_id,
        "selection_reasons": list(expected.selection_reasons),
        "source_work_item_sha256": canonical_sha256(expected.source_item),
        "source_selection_manifest_sha256": selection_sha256,
        "source_execution_plan_sha256": execution_plan_sha256,
        "trace_unit_id": None,
        "artifact_id": None,
        "artifact_path": None,
        "artifact_manifest_sha256": None,
        "artifact_payload_sha256": None,
        "prediction_position": None,
        "target_token_text": None,
        "target_logit": None,
        "target_probability": None,
        "model_id": None,
        "model_revision": None,
        "trace_configuration_identity": None,
        "code_revision": None,
        "tracing_source_tree_sha256": None,
        "error": None,
    }


def _complete_record(
    record: dict[str, Any],
    *,
    expected: ExpectedTarget,
    artifact: PhysicalArtifact,
    manifest: Mapping[str, Any],
) -> None:
    provenance = _mapping(
        _sequence(manifest["target_provenance"], "target_provenance")[0],
        "target provenance",
    )
    identity = _mapping(manifest["artifact_identity"], "artifact_identity")
    code_revision = _mapping(manifest.get("code_revision"), "code_revision")
    runtime_id = _string(manifest.get("artifact_id"), "artifact_id")
    record.update(
        {
            "status": (
                InventoryStatus.HOLDOUT.value
                if expected.corpus_role is CorpusRole.BROAD_CONFIRMATORY_HOLDOUT
                else InventoryStatus.DISCOVERY.value
            ),
            "trace_unit_id": runtime_id,
            "artifact_id": runtime_id,
            "artifact_path": str(artifact.path),
            "artifact_manifest_sha256": file_sha256(artifact.path / "manifest.json"),
            "artifact_payload_sha256": manifest.get("data_sha256"),
            "prediction_position": provenance.get("prediction_token_position"),
            "target_token_text": provenance.get("token_text"),
            "target_logit": provenance.get("logit"),
            "target_probability": provenance.get("probability"),
            "model_id": manifest.get("model_id"),
            "model_revision": manifest.get("model_revision"),
            "trace_configuration_identity": identity.get("sha256"),
            "code_revision": dict(code_revision),
            "tracing_source_tree_sha256": code_revision.get("source_tree_sha256"),
        }
    )


def build_inventory(
    *,
    selection_path: Path,
    execution_plan_path: Path,
    artifact_root: Path,
    validation_level: ValidationLevel = "full",
) -> dict[str, Any]:
    """Build an inventory without mutating source artifacts."""

    selection_path = selection_path.resolve()
    execution_plan_path = execution_plan_path.resolve()
    artifact_root = artifact_root.resolve()
    selection = load_json_object(selection_path)
    plan = load_json_object(execution_plan_path)
    expected, tokenizer_identity = _flatten_expected_targets(selection)
    excluded = _validate_execution_plan(
        plan,
        selection_path=selection_path,
        selection=selection,
    )
    expected_by_id = {target.source_artifact_id: target for target in expected}
    unknown_excluded = excluded - expected_by_id.keys()
    if unknown_excluded:
        raise ValueError(
            f"execution plan excludes unknown targets: {sorted(unknown_excluded)}"
        )

    physical = [
        _read_physical_artifact(path)
        for path in _candidate_artifact_directories(artifact_root)
    ]
    by_source: dict[str, list[PhysicalArtifact]] = defaultdict(list)
    unexpected: list[dict[str, Any]] = []
    for artifact in physical:
        source_id = artifact.source_artifact_id
        if source_id is None:
            unexpected.append(
                {
                    "artifact_path": str(artifact.path),
                    "artifact_id": None,
                    "source_artifact_id": None,
                    "reason": artifact.read_error
                    or "manifest has no source_artifact_id",
                }
            )
        elif source_id not in expected_by_id:
            unexpected.append(
                {
                    "artifact_path": str(artifact.path),
                    "artifact_id": (
                        artifact.manifest.get("artifact_id")
                        if artifact.manifest
                        else None
                    ),
                    "source_artifact_id": source_id,
                    "reason": "source_artifact_id is absent from frozen selection",
                }
            )
        elif source_id in excluded:
            unexpected.append(
                {
                    "artifact_path": str(artifact.path),
                    "artifact_id": (
                        artifact.manifest.get("artifact_id")
                        if artifact.manifest
                        else None
                    ),
                    "source_artifact_id": source_id,
                    "reason": "artifact exists for excluded pathological target",
                }
            )
        else:
            by_source[source_id].append(artifact)

    selection_sha256 = file_sha256(selection_path)
    plan_sha256 = file_sha256(execution_plan_path)
    records: list[dict[str, Any]] = []
    atlas_index = 0
    for target in expected:
        record = _base_record(
            target,
            selection_sha256=selection_sha256,
            execution_plan_sha256=plan_sha256,
        )
        if target.source_artifact_id in excluded:
            record["status"] = InventoryStatus.EXCLUDED_PATHOLOGICAL.value
            records.append(record)
            continue
        candidates = by_source.get(target.source_artifact_id, [])
        if not candidates:
            record["status"] = InventoryStatus.MISSING.value
            records.append(record)
            continue
        if len(candidates) > 1:
            record["status"] = InventoryStatus.CORRUPT.value
            record["error"] = "multiple physical artifacts resolve to one frozen target"
            unexpected.extend(
                {
                    "artifact_path": str(duplicate.path),
                    "artifact_id": (
                        duplicate.manifest.get("artifact_id")
                        if duplicate.manifest
                        else None
                    ),
                    "source_artifact_id": target.source_artifact_id,
                    "reason": "duplicate physical artifact for frozen target",
                }
                for duplicate in candidates[1:]
            )
            records.append(record)
            continue
        artifact = candidates[0]
        try:
            manifest = _validate_scientific_match(
                artifact,
                target,
                tokenizer_identity=tokenizer_identity,
                validation_level=validation_level,
            )
            _complete_record(
                record,
                expected=target,
                artifact=artifact,
                manifest=manifest,
            )
            record["atlas_trace_index"] = atlas_index
            atlas_index += 1
        except (OSError, TypeError, ValueError) as error:
            record["status"] = InventoryStatus.CORRUPT.value
            record["artifact_path"] = str(artifact.path)
            record["error"] = str(error)
        records.append(record)

    status_counts = Counter(record["status"] for record in records)
    discovery_records = [
        record
        for record in records
        if record["corpus_role"]
        in {
            CorpusRole.DENSE_DISCOVERY.value,
            CorpusRole.BROAD_DISCOVERY.value,
        }
    ]
    holdout_records = [
        record
        for record in records
        if record["corpus_role"] == CorpusRole.BROAD_CONFIRMATORY_HOLDOUT.value
    ]
    summary = {
        "planned": len(records),
        "completed": status_counts[InventoryStatus.DISCOVERY.value]
        + status_counts[InventoryStatus.HOLDOUT.value],
        "discovery_planned": len(discovery_records),
        "discovery_completed": status_counts[InventoryStatus.DISCOVERY.value],
        "holdout_planned": len(holdout_records),
        "holdout_completed": status_counts[InventoryStatus.HOLDOUT.value],
        "excluded_pathological": status_counts[
            InventoryStatus.EXCLUDED_PATHOLOGICAL.value
        ],
        "missing": status_counts[InventoryStatus.MISSING.value],
        "corrupt": status_counts[InventoryStatus.CORRUPT.value],
        "unexpected": len(unexpected),
    }
    inventory: dict[str, Any] = {
        "schema_version": INVENTORY_SCHEMA,
        "validation_level": validation_level,
        "sources": {
            "selection_manifest": {
                "path": str(selection_path),
                "sha256": selection_sha256,
                "canonical_sha256": canonical_sha256(selection),
            },
            "execution_plan": {
                "path": str(execution_plan_path),
                "sha256": plan_sha256,
                "plan_sha256": plan.get("plan_sha256"),
            },
            "artifact_root": {"path": str(artifact_root)},
        },
        "tokenizer_identity": tokenizer_identity,
        "summary": summary,
        "records": records,
        "unexpected_artifacts": sorted(
            unexpected,
            key=lambda item: (
                str(item["source_artifact_id"]),
                str(item["artifact_path"]),
            ),
        ),
    }
    inventory["inventory_sha256"] = canonical_sha256(inventory)
    return inventory


def write_inventory(path: Path, inventory: Mapping[str, Any]) -> None:
    write_hashed_json(
        path,
        inventory,
        hash_field="inventory_sha256",
    )
