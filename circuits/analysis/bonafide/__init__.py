"""Strict downstream contracts for the exploratory BonaFide analysis."""

from circuits.analysis.bonafide.identity import (
    BASIS_KEY_SCHEMA,
    OCCURRENCE_KEY_SCHEMA,
    OccurrenceKey,
    SignedBasisKey,
    basis_key_from_raw_node,
    occurrence_key_from_raw_node,
)
from circuits.analysis.bonafide.features import (
    FEATURE_SCHEMA,
    build_profile_observations,
    cluster_fully_supported_profiles,
)
from circuits.analysis.bonafide.index import ATLAS_INDEX_SCHEMA, build_atlas_index
from circuits.analysis.bonafide.partition import (
    AnalysisTarget,
    CorpusRole,
    hierarchical_fit_weights,
    validate_partition_contract,
)

__all__ = [
    "BASIS_KEY_SCHEMA",
    "FEATURE_SCHEMA",
    "OCCURRENCE_KEY_SCHEMA",
    "AnalysisTarget",
    "ATLAS_INDEX_SCHEMA",
    "CorpusRole",
    "OccurrenceKey",
    "SignedBasisKey",
    "basis_key_from_raw_node",
    "build_atlas_index",
    "build_profile_observations",
    "cluster_fully_supported_profiles",
    "hierarchical_fit_weights",
    "occurrence_key_from_raw_node",
    "validate_partition_contract",
]
