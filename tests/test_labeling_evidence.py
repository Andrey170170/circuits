from __future__ import annotations

from types import SimpleNamespace

from circuits.descriptions.types import ActivationRecord
from circuits.labeling.evidence import (
    candidate_messages,
    render_persisted_partition_witnesses,
    select_cluster_ids,
    summary_messages,
)
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
            },
            {
                "trace_unit_id": "trace-2",
                "family_partition": "selection_scoring",
                "response_position": 4,
                "target_token_text": " y",
                "condition": {"diversity": {"hint_types": ["metadata"]}},
                "cluster_projection": {"absolute_attribution_mass": 0.3},
                "prompt": "selection prompt",
                "response": "selection response",
            },
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


def test_width_one_prompts_separate_background_and_use_exact_allowed_witnesses() -> (
    None
):
    candidate, _ = candidate_messages(
        _row(),
        highlighted_sequences={"trace-1": "GENERATION_EXACT"},
        prompt_policy="width_one_v2",
    )
    candidate_text = "\n".join(message.content for message in candidate)
    assert "single-target, width-one" in candidate_text
    assert "top-k target comparison" in candidate_text
    assert "corpus-bounded association" in candidate_text
    assert "localized_evidence" in candidate_text

    profile_payload = {
        "partitions": {
            "generation": [
                {
                    "trace_unit_id": "trace-1",
                    "family_partition": "generation",
                    "record": {
                        "tokens": ["GEN", " exact"],
                        "token_ids": [1, 2],
                        "activations": [2.0, 0.1],
                    },
                }
            ],
            "selection_scoring": [
                {
                    "trace_unit_id": "trace-2",
                    "family_partition": "selection_scoring",
                    "record": {
                        "tokens": ["SELECT", " exact"],
                        "token_ids": [3, 4],
                        "activations": [3.0, 0.2],
                    },
                }
            ],
            "audit": [{"trace_unit_id": "AUDIT_MUST_NOT_APPEAR"}],
        }
    }
    witnesses = {
        partition: render_persisted_partition_witnesses(
            _row(), profile_payload, partition=partition
        )
        for partition in ("generation", "selection_scoring")
    }
    summary, _ = summary_messages(
        _row(),
        scored_candidates=[
            {
                "text": "candidate",
                "correlation": 0.2,
                "candidate": {
                    "description": "candidate",
                    "localized_evidence": "STRUCTURED_LOCALIZED_EVIDENCE",
                    "background_or_confound": "shared context",
                    "limitations": "width one",
                },
            }
        ],
        prompt_policy="width_one_v2",
        highlighted_witnesses=witnesses,
    )
    summary_text = "\n".join(message.content for message in summary)
    assert "GEN" in summary_text
    assert "SELECT" in summary_text
    assert "AUDIT_MUST_NOT_APPEAR" not in summary_text
    assert "STRUCTURED_LOCALIZED_EVIDENCE" in summary_text
    assert "provisional_label" in summary_text


def test_hybrid_candidate_prompts_expose_top_five_but_never_audit() -> None:
    row = _row()
    row["balanced_target_exemplars"][0]["candidate_union_summary"] = {
        "candidate_width": 5,
        "signed_contribution_sum": [1.0, -0.5, 0.25, 0.0, -0.75],
        "axis_scope": "target_local",
    }
    candidate, _ = candidate_messages(
        row,
        highlighted_sequences={"trace-1": "HYBRID_GENERATION_EXACT"},
        prompt_policy="hybrid_candidate_v1",
    )
    candidate_text = "\n".join(message.content for message in candidate)
    assert "top five" in candidate_text
    assert "candidate_union_summary" in candidate_text
    assert "no non-degenerate contribution" not in candidate_text
    assert "HYBRID_GENERATION_EXACT" in candidate_text
    assert "selection response" not in candidate_text

    profile_payload = {
        "partitions": {
            "generation": [
                {
                    "trace_unit_id": "trace-1",
                    "family_partition": "generation",
                    "record": {
                        "tokens": ["GENERATION"],
                        "token_ids": [1],
                        "activations": [2.0],
                    },
                }
            ],
            "selection_scoring": [
                {
                    "trace_unit_id": "trace-2",
                    "family_partition": "selection_scoring",
                    "record": {
                        "tokens": ["SELECTION"],
                        "token_ids": [2],
                        "activations": [1.0],
                    },
                }
            ],
            "audit": [{"trace_unit_id": "AUDIT_FORBIDDEN"}],
        }
    }
    witnesses = {
        partition: render_persisted_partition_witnesses(
            row, profile_payload, partition=partition
        )
        for partition in ("generation", "selection_scoring")
    }
    summary, _ = summary_messages(
        row,
        scored_candidates=[{"text": "candidate", "correlation": 0.5}],
        prompt_policy="hybrid_candidate_v1",
        highlighted_witnesses=witnesses,
    )
    summary_text = "\n".join(message.content for message in summary)
    assert "GENERATION" in summary_text
    assert "SELECTION" in summary_text
    assert "AUDIT_FORBIDDEN" not in summary_text
    assert "top-five" in summary_text


def test_legacy_prompt_policy_is_the_default() -> None:
    implicit = candidate_messages(_row(), highlighted_sequences={"trace-1": "same"})
    explicit = candidate_messages(
        _row(), highlighted_sequences={"trace-1": "same"}, prompt_policy="legacy_v1"
    )
    assert implicit == explicit


def test_evenly_spaced_cluster_selection_uses_ready_clusters() -> None:
    state = SimpleNamespace(name="primary", ready_cluster_ids=[0, 1, 2, 4, 7, 9])
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
