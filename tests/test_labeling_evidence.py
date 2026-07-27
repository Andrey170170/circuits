from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from circuits.descriptions.types import ActivationRecord
from circuits.labeling.evidence import candidate_messages, select_cluster_ids
from circuits.labeling.profiles import retokenize_for_simulator


def _row() -> dict:
    return {
        "cluster_id": 4,
        "member_basis_count": 9,
        "multiplex_summary": {"support_family_count": 3},
        "prototype_signed_bases": [
            {
                "layer": 1,
                "neuron_index": 2,
                "polarity": "+",
                "internal_affinity_strength": 3.5,
            }
        ],
        "top_recurrent_cluster_edges": [],
        "balanced_target_exemplars": [
            {
                "trace_unit_id": "trace-1",
                "family_partition": "generation",
                "response_position": 3,
                "target_token_text": " x",
                "condition": {"diversity": {"hint_types": ["metadata"]}},
                "cluster_projection": {"absolute_attribution_mass": 0.2},
                "prompt": "prompt",
                "response": "response",
            }
        ],
    }


def test_candidate_prompt_is_deterministic_and_partition_safe() -> None:
    first, first_hash = candidate_messages(
        _row(), highlighted_sequences={"trace-1": "a<mark>b</mark>"}
    )
    second, second_hash = candidate_messages(
        _row(), highlighted_sequences={"trace-1": "a<mark>b</mark>"}
    )
    assert first == second
    assert first_hash == second_hash
    rendered = first[1].content
    assert "a<mark>b</mark>" in rendered
    assert "selection_scoring" not in rendered
    assert "audit" not in rendered


def test_evenly_spaced_cluster_selection_uses_ready_clusters() -> None:
    state = SimpleNamespace(
        name="primary", ready_cluster_ids=[0, 1, 2, 4, 7, 9]
    )
    assert select_cluster_ids(state, limit=3) == [0, 2, 9]


class _CharacterTokenizer:
    def __call__(self, text, add_special_tokens, return_offsets_mapping):
        assert not add_special_tokens
        return {
            "input_ids": list(range(len(text))),
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }

    def convert_ids_to_tokens(self, token_ids):
        return [str(value) for value in token_ids]


def test_retokenization_preserves_character_aligned_values() -> None:
    record = ActivationRecord(
        tokens=["ab", "c"], token_ids=[1, 2], activations=[2.0, -1.0]
    )
    mapped, diagnostics = retokenize_for_simulator(record, _CharacterTokenizer())
    assert mapped.activations == [2.0, 2.0, -1.0]
    assert diagnostics["coverage_fraction"] == 1.0
