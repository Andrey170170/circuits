from __future__ import annotations

import json
from types import SimpleNamespace

from scripts.bonafide.prepare_t5_corpus import (
    build_source_manifest,
    select_response_draws,
    trace_compatible_screen_row,
)


def _screen_module():
    return SimpleNamespace(
        _ALLOWED_EOS_KINDS={"implicit_default_eos"},
        PREDICATE_FIELDS=(
            "natural_eos",
            "response_length_224_768",
            "total_length_at_most_1024",
            "nonempty_raw_response",
            "nonempty_reasoning_and_final_answer",
            "no_obvious_degeneration",
        ),
        RULE_CONFIG={"schema": "frozen-screen/v1"},
        RULE_CONFIG_SHA256="a" * 64,
        screen_row=lambda _row: {
            "screening_reasoning": "reasoning",
            "screening_final_answer": "answer",
            "stop_reason_kind": "implicit_default_eos",
        },
        degeneration_metrics=lambda ids: {
            "immediate_repeat_detected": False,
            "immediate_repeat_block_length": None,
            "immediate_repeat_start_token_index": None,
            "unique_4gram_count": max(0, len(ids) - 3),
            "total_4gram_count": max(0, len(ids) - 3),
            "unique_4gram_ratio": 1.0,
            "unique_4gram_threshold_applied": True,
            "no_obvious_degeneration": True,
        },
    )


def _row(*, draw: int = 0, eligible_length: int = 224) -> dict:
    response_ids = list(range(eligible_length)) + [999]
    repro = {
        "prompt_token_ids": [1, 2, 3],
        "completion_token_ids": response_ids,
        "stop_reason_kind": "implicit_default_eos",
        "effective_default_stop_token_ids": [999],
        "sampled_token_logprobs": [-0.1] * len(response_ids),
    }
    return {
        "id": f"request-{draw}",
        "source_prompt_id": "prompt-1",
        "question_family_id": "family-1",
        "campaign_id": "campaign-1",
        "campaign_role": "primary_new_prompt",
        "draw_index": str(draw),
        "generation_seed": str(draw + 7),
        "completion_id": f"completion-{draw}",
        "question": "question",
        "prompt": "prompt",
        "model_raw_response": "reasoning\nanswer",
        "source_question_ids_json": '["source-1"]',
        "src_type": "hinting",
        "hint_dataset": "dataset",
        "hint_type": "metadata",
        "prompted_hint": "hint",
        "correct_answer": "correct",
        "hinted_answer": "hinted",
        "target_model": "Qwen/Qwen3-4B-Instruct-2507",
        "model_revision": "revision",
        "assistant_prefix_token_count": "3",
        "reproducibility_info": json.dumps(repro),
    }


def test_trace_compatible_screen_excludes_terminal_assistant_suffix() -> None:
    screened = trace_compatible_screen_row(_row(), _screen_module())

    assert screened["generation_completion_token_count"] == 225
    assert screened["response_token_count"] == 224
    assert screened["trace_response_token_ids"][-1] == 223
    assert screened["terminal_assistant_suffix_token_id"] == 999
    assert screened["screening_eligible"] is True


def test_trace_compatible_screen_uses_exact_retokenized_response_ids() -> None:
    row = _row()
    tokenized = SimpleNamespace(
        assistant_prefix_ids=[1, 2, 3],
        response_ids=[*range(223), 42],
        assistant_suffix_ids=[999],
    )

    screened = trace_compatible_screen_row(row, _screen_module(), tokenized=tokenized)

    assert screened["trace_response_token_ids"] == tokenized.response_ids
    assert screened["trace_tokenization_matches_generation"] is False
    assert screened["trace_response_token_logprobs"] is None
    assert screened["screening_eligible"] is True


def test_response_draw_selection_uses_fallback_and_keeps_role_separate() -> None:
    draw0 = {**_row(draw=0), "screening_eligible": False}
    draw1 = {**_row(draw=1), "screening_eligible": True}

    selected = select_response_draws(
        [draw0, draw1], role_map={"primary_new_prompt": "primary_discovery"}
    )

    assert selected == [draw1]
    assert draw1["trace_corpus_role"] == "primary_discovery"
    assert draw1["selection_status"] == "selected_draw1_fallback"


def test_source_manifest_selects_twenty_independent_stratified_targets() -> None:
    record = trace_compatible_screen_row(_row(eligible_length=224), _screen_module())
    record.update(
        {
            **_row(eligible_length=224),
            "trace_corpus_role": "primary_discovery",
            "selection_status": "selected_draw0_default",
        }
    )
    # Restore the derived fields overwritten by the raw fixture update.
    record = {
        **record,
        **trace_compatible_screen_row(_row(eligible_length=224), _screen_module()),
        "trace_corpus_role": "primary_discovery",
        "selection_status": "selected_draw0_default",
    }
    profile = {
        "profile_id": "profile-v1",
        "campaign_id": "campaign-1",
        "target_selection": {
            "policy_id": "stratified-random-20.v1",
            "targets_per_response": 20,
            "seed": "seed-v1",
        },
    }
    tokenizer = {
        "model_id": "Qwen/Qwen3-4B-Instruct-2507",
        "revision": "revision",
        "chat_template_sha256": "b" * 64,
    }

    manifest, selected = build_source_manifest(
        [record], profile=profile, tokenizer=tokenizer
    )

    items = manifest["waves"][0]["items"]
    positions = [
        item["target_selection"]["response_token_positions"][0] for item in items
    ]
    assert len(items) == len(set(positions)) == 20
    assert min(positions) >= 0
    assert max(positions) < 224
    assert len(selected) == 1
