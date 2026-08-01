"""Validated streaming access to compacted BonaFide feature stores."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray
from scipy.sparse import csr_matrix, load_npz, save_npz

from circuits.analysis.bonafide.canonical import (
    canonical_sha256,
    file_sha256,
    load_json_object,
)
from circuits.analysis.bonafide.clustering import (
    DEFAULT_EPSILON,
    PAIR_EVIDENCE_SCHEMA,
    PairEvidence,
    PairEvidenceAccumulator,
    TargetProfileBlock,
    WeightingMode,
)
from circuits.analysis.bonafide.compaction import (
    COMPACTED_FEATURE_SCHEMA,
    _validate_existing_compaction,
)


@dataclass(frozen=True)
class BasisSupport:
    target_counts: NDArray[np.int64]
    response_counts: NDArray[np.int64]
    family_counts: NDArray[np.int64]
    boundary_mask: NDArray[np.bool_]


@dataclass(frozen=True)
class PairEvidenceBuild:
    evidence: PairEvidence
    basis_support: BasisSupport
    feature_manifest: Mapping[str, Any]
    feature_store_root: Path
    basis_rows: tuple[Mapping[str, Any], ...]
    target_selection: Mapping[str, Any] = field(default_factory=lambda: {"mode": "all"})


BASIS_SUPPORT_SCHEMA = pa.schema(
    [
        pa.field("signed_basis_index", pa.int64(), nullable=False),
        pa.field("target_count", pa.int64(), nullable=False),
        pa.field("response_count", pa.int64(), nullable=False),
        pa.field("family_count", pa.int64(), nullable=False),
        pa.field("boundary", pa.bool_(), nullable=False),
    ]
)

PAIR_MATRIX_FILES = {
    "weighted_similarity_sum": "weighted-similarity-sum.npz",
    "support_weight_sum": "support-weight-sum.npz",
    "overlap_count": "target-overlap-count.npz",
    "response_overlap_count": "response-overlap-count.npz",
    "family_overlap_count": "family-overlap-count.npz",
}


def csr_content_sha256(matrix: csr_matrix) -> str:
    """Hash canonical CSR content independently of NPZ container metadata."""

    canonical = matrix.copy().tocsr()
    canonical.sum_duplicates()
    canonical.sort_indices()
    digest = hashlib.sha256()
    for value in (
        str(canonical.shape),
        canonical.data.dtype.str,
        canonical.indices.dtype.str,
        canonical.indptr.dtype.str,
    ):
        encoded = value.encode("ascii")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    for array in (canonical.indptr, canonical.indices, canonical.data):
        content = np.ascontiguousarray(array).tobytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _basis_tuple(record: Mapping[str, Any]) -> tuple[str, str, int, int, str]:
    return (
        str(record["model_id"]),
        str(record["model_revision"]),
        int(record["layer"]),
        int(record["neuron_index"]),
        str(record["polarity"]),
    )


class FeatureStoreReader:
    """Fail-closed reader that yields one target-local profile block at a time."""

    def __init__(self, compacted_root: Path) -> None:
        self.compacted_root = compacted_root.resolve()
        raw_manifest = load_json_object(self.compacted_root / "manifest.json")
        plan_sha256 = raw_manifest.get("plan_sha256")
        if not isinstance(plan_sha256, str):
            raise ValueError("feature-store manifest lacks plan_sha256")
        self.manifest = _validate_existing_compaction(
            self.compacted_root,
            plan_sha256=plan_sha256,
        )
        if self.manifest.get("schema_version") != COMPACTED_FEATURE_SCHEMA:
            raise ValueError("clustering requires a compacted dense feature store")
        if self.manifest.get("lane") != "dense_features":
            raise ValueError("clustering input lane must be dense_features")

        raw_basis_rows = pq.read_table(
            self.compacted_root / "basis-index.parquet"
        ).to_pylist()
        raw_basis_rows.sort(key=lambda row: int(row["signed_basis_index"]))
        observed_indices = [int(row["signed_basis_index"]) for row in raw_basis_rows]
        if observed_indices != list(range(len(raw_basis_rows))):
            raise ValueError("signed basis index is not contiguous and canonical")
        if len(raw_basis_rows) != int(self.manifest["signed_basis_count"]):
            raise ValueError("signed basis count disagrees with feature manifest")
        self.basis_rows = tuple(raw_basis_rows)
        self._basis_to_index = {
            _basis_tuple(row): int(row["signed_basis_index"]) for row in self.basis_rows
        }
        if len(self._basis_to_index) != len(self.basis_rows):
            raise ValueError("feature store contains duplicate signed basis identities")

        target_rows = pq.read_table(
            self.compacted_root / "target-index.parquet"
        ).to_pylist()
        self._target_by_trace: dict[str, Mapping[str, Any]] = {}
        for row in target_rows:
            trace_unit_id = str(row["trace_unit_id"])
            if trace_unit_id in self._target_by_trace:
                raise ValueError("feature store contains duplicate target identity")
            if not bool(row["cluster_fit_eligible"]):
                raise ValueError("feature store violates cluster-fit holdout firewall")
            if str(row["corpus_role"]) != "dense_discovery":
                raise ValueError("feature store contains a non-discovery fit target")
            fit_weight = float(row["fit_weight"])
            if not math.isfinite(fit_weight) or fit_weight <= 0:
                raise ValueError("target fit weight must be finite and positive")
            self._target_by_trace[trace_unit_id] = row
        if len(self._target_by_trace) != int(self.manifest["target_count"]):
            raise ValueError("target count disagrees with feature manifest")

    @property
    def basis_count(self) -> int:
        return len(self.basis_rows)

    @property
    def target_count(self) -> int:
        return len(self._target_by_trace)

    @property
    def target_rows(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(
            sorted(
                self._target_by_trace.values(),
                key=lambda row: int(row["atlas_trace_index"]),
            )
        )

    def iter_blocks(
        self,
    ) -> Iterator[tuple[TargetProfileBlock, Mapping[str, Any]]]:
        seen_trace_ids: set[str] = set()
        columns = [
            "model_id",
            "model_revision",
            "layer",
            "neuron_index",
            "polarity",
            "trace_unit_id",
            "response_id",
            "base_question_id",
            "fit_weight",
            "attribution_profile",
            "attribution_support",
        ]
        output_root = self.compacted_root.parent
        for shard in self.manifest["shards"]:
            shard_path = output_root / str(shard["path"])
            stats_rows = pq.read_table(
                shard_path / "target-stats.parquet",
                columns=[
                    "trace_unit_id",
                    "attribution_profile_column_count",
                ],
            ).to_pylist()
            expected_width_by_trace = {
                str(row["trace_unit_id"]): int(row["attribution_profile_column_count"])
                for row in stats_rows
            }
            if len(expected_width_by_trace) != len(stats_rows):
                raise ValueError("target stats contain duplicate target identity")
            table = pq.read_table(
                shard_path / "basis-observations.parquet",
                columns=columns,
            )
            trace_ids = table["trace_unit_id"].to_pylist()
            group_starts = [0]
            for row_index in range(1, len(trace_ids)):
                if trace_ids[row_index] != trace_ids[row_index - 1]:
                    group_starts.append(row_index)
            group_starts.append(len(trace_ids))
            shard_trace_ids: set[str] = set()
            for start, stop in zip(
                group_starts[:-1],
                group_starts[1:],
                strict=True,
            ):
                trace_unit_id = str(trace_ids[start])
                if trace_unit_id in shard_trace_ids:
                    raise ValueError(
                        "feature observations for one target are not contiguous"
                    )
                shard_trace_ids.add(trace_unit_id)
                if trace_unit_id in seen_trace_ids:
                    raise ValueError("duplicate target observations across shards")
                seen_trace_ids.add(trace_unit_id)
                target = self._target_by_trace.get(trace_unit_id)
                if target is None:
                    raise ValueError("feature observation references unknown target")

                group = table.slice(start, stop - start)
                basis_records = group.select(
                    [
                        "model_id",
                        "model_revision",
                        "layer",
                        "neuron_index",
                        "polarity",
                    ]
                ).to_pylist()
                try:
                    basis_indices = np.asarray(
                        [
                            self._basis_to_index[_basis_tuple(record)]
                            for record in basis_records
                        ],
                        dtype=np.int64,
                    )
                except KeyError as error:
                    raise ValueError(
                        "feature observation basis is absent from canonical index"
                    ) from error
                response_ids = set(group["response_id"].to_pylist())
                family_ids = set(group["base_question_id"].to_pylist())
                weights = set(float(value) for value in group["fit_weight"].to_pylist())
                if response_ids != {target["response_id"]}:
                    raise ValueError("target response identity drift in feature rows")
                if family_ids != {target["base_question_id"]}:
                    raise ValueError("target family identity drift in feature rows")
                if len(weights) != 1 or not math.isclose(
                    weights.pop(),
                    float(target["fit_weight"]),
                    abs_tol=1e-15,
                ):
                    raise ValueError("target fit weight drift in feature rows")
                profiles = group["attribution_profile"].to_pylist()
                supports = group["attribution_support"].to_pylist()
                widths = {len(profile) for profile in profiles}
                if widths != {expected_width_by_trace.get(trace_unit_id)}:
                    raise ValueError("target attribution profile width drift")
                block = TargetProfileBlock(
                    trace_unit_id=trace_unit_id,
                    response_id=str(target["response_id"]),
                    base_question_id=str(target["base_question_id"]),
                    basis_indices=basis_indices,
                    values=np.asarray(profiles, dtype=np.float32),
                    support=np.asarray(supports, dtype=np.bool_),
                    fit_weight=float(target["fit_weight"]),
                )
                block.validate()
                yield block, target
        if seen_trace_ids != set(self._target_by_trace):
            raise ValueError("feature shards do not cover the canonical target index")


def build_pair_evidence_from_feature_store(
    compacted_root: Path,
    *,
    weighting: WeightingMode = "hierarchical",
    epsilon: float = DEFAULT_EPSILON,
    included_family_ids: frozenset[str] | None = None,
    excluded_family_ids: frozenset[str] | None = None,
) -> PairEvidenceBuild:
    """Build exact pair evidence and recurrence support from a feature store."""

    if included_family_ids is not None and excluded_family_ids is not None:
        raise ValueError("family inclusion and exclusion are mutually exclusive")
    if included_family_ids is not None and not included_family_ids:
        raise ValueError("included_family_ids cannot be empty")
    if excluded_family_ids is not None and not excluded_family_ids:
        raise ValueError("excluded_family_ids cannot be empty")
    reader = FeatureStoreReader(compacted_root)
    accumulator = PairEvidenceAccumulator(
        basis_count=reader.basis_count,
        weighting=weighting,
        epsilon=epsilon,
    )
    target_counts = np.zeros(reader.basis_count, dtype=np.int64)
    response_sets: list[set[str]] = [set() for _ in range(reader.basis_count)]
    family_sets: list[set[str]] = [set() for _ in range(reader.basis_count)]
    selected_trace_ids: list[str] = []
    selected_response_ids: set[str] = set()
    selected_family_ids: set[str] = set()
    for block, target in reader.iter_blocks():
        family_id = str(target["base_question_id"])
        if included_family_ids is not None and family_id not in included_family_ids:
            continue
        if excluded_family_ids is not None and family_id in excluded_family_ids:
            continue
        accumulator.add(block)
        selected_trace_ids.append(block.trace_unit_id)
        target_counts[block.basis_indices] += 1
        response_id = str(target["response_id"])
        selected_response_ids.add(response_id)
        selected_family_ids.add(family_id)
        for basis_index in block.basis_indices:
            response_sets[int(basis_index)].add(response_id)
            family_sets[int(basis_index)].add(family_id)

    layers = np.asarray(
        [int(row["layer"]) for row in reader.basis_rows],
        dtype=np.int64,
    )
    maximum_layer = int(layers.max())
    boundary_mask = (layers < 0) | (layers == maximum_layer)
    if included_family_ids is not None:
        selection = {
            "mode": "include_families",
            "included_family_ids": sorted(included_family_ids),
        }
    elif excluded_family_ids is not None:
        selection = {
            "mode": "exclude_families",
            "excluded_family_ids": sorted(excluded_family_ids),
        }
    else:
        selection = {"mode": "all"}
    selection = {
        **selection,
        "selected_target_count": len(selected_trace_ids),
        "selected_response_count": len(selected_response_ids),
        "selected_family_count": len(selected_family_ids),
        "selected_trace_unit_ids_sha256": canonical_sha256(sorted(selected_trace_ids)),
    }
    return PairEvidenceBuild(
        evidence=accumulator.finalize(),
        basis_support=BasisSupport(
            target_counts=target_counts,
            response_counts=np.asarray(
                [len(values) for values in response_sets],
                dtype=np.int64,
            ),
            family_counts=np.asarray(
                [len(values) for values in family_sets],
                dtype=np.int64,
            ),
            boundary_mask=boundary_mask,
        ),
        feature_manifest=reader.manifest,
        feature_store_root=reader.compacted_root,
        basis_rows=reader.basis_rows,
        target_selection=selection,
    )


def _validate_pair_evidence_manifest(output_root: Path) -> dict[str, Any]:
    manifest = load_json_object(output_root / "manifest.json")
    core = dict(manifest)
    recorded_hash = core.pop("manifest_sha256", None)
    if recorded_hash != canonical_sha256(core):
        raise ValueError("pair-evidence manifest hash mismatch")
    if manifest.get("schema_version") != PAIR_EVIDENCE_SCHEMA:
        raise ValueError("unsupported pair-evidence schema")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ValueError("pair-evidence file inventory is invalid")
    for item in files:
        if not isinstance(item, Mapping):
            raise ValueError("pair-evidence file record is invalid")
        path = output_root / str(item["path"])
        if path.stat().st_size != int(item["size_bytes"]):
            raise ValueError(f"pair-evidence file size drift: {path}")
        if file_sha256(path) != item["sha256"]:
            raise ValueError(f"pair-evidence file hash drift: {path}")
    return manifest


def write_pair_evidence_build(
    output_root: Path,
    build: PairEvidenceBuild,
    *,
    code_revision: Mapping[str, Any],
    environment: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically persist pair evidence without overwriting an existing state."""

    output_root = output_root.resolve()
    if output_root.exists():
        return _validate_pair_evidence_manifest(output_root)
    build.evidence.validate()
    temporary = output_root.parent / f".{output_root.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir(parents=True)
    try:
        matrix_records: dict[str, dict[str, Any]] = {}
        for field, filename in PAIR_MATRIX_FILES.items():
            matrix = getattr(build.evidence, field)
            path = temporary / filename
            save_npz(path, matrix, compressed=True)
            matrix_records[field] = {
                "path": filename,
                "shape": list(matrix.shape),
                "dtype": matrix.dtype.str,
                "nnz": int(matrix.nnz),
                "content_sha256": csr_content_sha256(matrix),
            }
        support_path = temporary / "basis-support.parquet"
        support = build.basis_support
        pq.write_table(
            pa.Table.from_pylist(
                [
                    {
                        "signed_basis_index": basis_index,
                        "target_count": int(support.target_counts[basis_index]),
                        "response_count": int(support.response_counts[basis_index]),
                        "family_count": int(support.family_counts[basis_index]),
                        "boundary": bool(support.boundary_mask[basis_index]),
                    }
                    for basis_index in range(build.evidence.basis_count)
                ],
                schema=BASIS_SUPPORT_SCHEMA,
            ),
            support_path,
            compression="zstd",
            use_dictionary=True,
            write_statistics=True,
        )
        paths = sorted(temporary.iterdir())
        files = [
            {
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in paths
            if path.is_file()
        ]
        manifest: dict[str, Any] = {
            "schema_version": PAIR_EVIDENCE_SCHEMA,
            "source_feature_store": {
                "path": str(build.feature_store_root),
                "manifest_sha256": build.feature_manifest["manifest_sha256"],
                "plan_sha256": build.feature_manifest["plan_sha256"],
                "schema_version": build.feature_manifest["schema_version"],
            },
            "basis_count": build.evidence.basis_count,
            "target_count": build.evidence.target_count,
            "target_selection": dict(build.target_selection),
            "weighting": build.evidence.weighting,
            "epsilon": build.evidence.epsilon,
            "matrix_records": matrix_records,
            "code_revision": dict(code_revision),
            "environment": dict(environment),
            "descriptions_generated": False,
            "scientific_cluster_state": False,
            "files": files,
        }
        manifest["manifest_sha256"] = canonical_sha256(manifest)
        with (temporary / "manifest.json").open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        output_root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, output_root)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def load_pair_evidence(output_root: Path) -> tuple[PairEvidence, BasisSupport]:
    """Load and content-validate a persisted pair-evidence state."""

    output_root = output_root.resolve()
    manifest = _validate_pair_evidence_manifest(output_root)
    source = manifest.get("source_feature_store")
    if not isinstance(source, Mapping):
        raise ValueError("pair-evidence source feature store is invalid")
    source_reader = FeatureStoreReader(Path(str(source["path"])))
    if source_reader.manifest.get("manifest_sha256") != source.get("manifest_sha256"):
        raise ValueError("pair-evidence source feature store drift")
    matrices: dict[str, csr_matrix] = {}
    matrix_records = manifest.get("matrix_records")
    if not isinstance(matrix_records, Mapping):
        raise ValueError("pair-evidence matrix inventory is invalid")
    for field, filename in PAIR_MATRIX_FILES.items():
        matrix = load_npz(output_root / filename).tocsr()
        record = matrix_records.get(field)
        if not isinstance(record, Mapping):
            raise ValueError(f"pair-evidence matrix record is missing: {field}")
        if list(matrix.shape) != record.get("shape"):
            raise ValueError(f"pair-evidence matrix shape drift: {field}")
        if int(matrix.nnz) != int(record["nnz"]):
            raise ValueError(f"pair-evidence matrix nnz drift: {field}")
        if matrix.dtype.str != record.get("dtype"):
            raise ValueError(f"pair-evidence matrix dtype drift: {field}")
        if csr_content_sha256(matrix) != record.get("content_sha256"):
            raise ValueError(f"pair-evidence matrix content drift: {field}")
        matrices[field] = matrix
    evidence = PairEvidence(
        weighted_similarity_sum=matrices["weighted_similarity_sum"],
        support_weight_sum=matrices["support_weight_sum"],
        overlap_count=matrices["overlap_count"],
        response_overlap_count=matrices["response_overlap_count"],
        family_overlap_count=matrices["family_overlap_count"],
        target_count=int(manifest["target_count"]),
        weighting=cast(WeightingMode, str(manifest["weighting"])),
        epsilon=float(manifest["epsilon"]),
    )
    evidence.validate()
    rows = pq.read_table(output_root / "basis-support.parquet").to_pylist()
    rows.sort(key=lambda row: int(row["signed_basis_index"]))
    if [int(row["signed_basis_index"]) for row in rows] != list(
        range(evidence.basis_count)
    ):
        raise ValueError("pair-evidence basis support index is invalid")
    support = BasisSupport(
        target_counts=np.asarray(
            [row["target_count"] for row in rows],
            dtype=np.int64,
        ),
        response_counts=np.asarray(
            [row["response_count"] for row in rows],
            dtype=np.int64,
        ),
        family_counts=np.asarray(
            [row["family_count"] for row in rows],
            dtype=np.int64,
        ),
        boundary_mask=np.asarray(
            [row["boundary"] for row in rows],
            dtype=np.bool_,
        ),
    )
    return evidence, support
