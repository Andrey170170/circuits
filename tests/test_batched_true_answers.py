"""Regression tests for per-example answer targets across circuit batches."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace
from typing import cast

import pytest


def _import_target_module(
    monkeypatch: pytest.MonkeyPatch, module_name: str
) -> ModuleType:
    if module_name == "circuits.analysis.process_circuits":
        fake_jvp = ModuleType("circuits.core.jvp")
        fake_jvp.ADAGConfig = object
        fake_jvp.get_all_pairs_cl_ja_effects_with_attributions = lambda **_kwargs: None
        monkeypatch.setitem(sys.modules, "circuits.core.jvp", fake_jvp)

        fake_utils = ModuleType("circuits.core.utils")
        fake_utils.Edge = object
        fake_utils.Node = object
        monkeypatch.setitem(sys.modules, "circuits.core.utils", fake_utils)

        fake_wikipedia = ModuleType("user_modeling.datasets.wikipedia")
        fake_wikipedia.get_wikipedia_dataset_by_split = lambda *_args, **_kwargs: None
        monkeypatch.setitem(
            sys.modules, "user_modeling.datasets.wikipedia", fake_wikipedia
        )

        fake_chat_input = ModuleType("util.chat_input")
        fake_chat_input.ChatInput = object
        fake_chat_input.IdsInput = object
        monkeypatch.setitem(sys.modules, "util.chat_input", fake_chat_input)

        fake_subject = ModuleType("util.subject")
        fake_subject.Subject = object
        fake_subject.llama31_8B_instruct_config = object()
        monkeypatch.setitem(sys.modules, "util.subject", fake_subject)

    module = importlib.import_module(module_name)
    if module_name == "circuits.analysis.process_circuits":
        # Avoid leaving a module bound to lightweight dependency stubs in the
        # shared test process after monkeypatch restores those dependencies.
        sys.modules.pop(module_name, None)
    return module


@pytest.mark.parametrize(
    ("module_name", "includes_focus_probs"),
    [
        ("circuits.tracing.trace", True),
        ("circuits.analysis.process_circuits", False),
    ],
)
def test_compute_circuits_slices_true_answers_per_batch(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    includes_focus_probs: bool,
) -> None:
    module = _import_target_module(monkeypatch, module_name)
    token_ids = {"alpha": 11, "beta": 22, "gamma": 33}
    seen_answers: list[list[list[str] | None]] = []
    seen_focus_tokens: list[list[list[int]]] = []

    def fake_prepare_cis(
        _model: object,
        _tokenizer: object,
        questions: list[str],
        _seed_responses: list[str],
        **kwargs: object,
    ) -> tuple[object, ...]:
        true_answers = cast(list[list[str] | None], kwargs["true_answers"])
        seen_answers.append(true_answers)
        focus_tokens = [
            [token_ids[answers[0]]] for answers in true_answers if answers is not None
        ]
        cis = [[index] for index in range(len(questions))]
        attention_masks = [[1] for _ in questions]
        starts = [0 for _ in questions]
        if includes_focus_probs:
            focus_probs = [[0.0] for _ in questions]
            return (
                cis,
                attention_masks,
                focus_tokens,
                focus_probs,
                [0],
                starts,
            )
        return cis, attention_masks, focus_tokens, [0], starts

    def fake_trace_batch(**kwargs: object) -> tuple[list[object], list[object]]:
        focus_tokens = cast(list[list[int]], kwargs["focus_logits"])
        seen_focus_tokens.append(focus_tokens)
        batch_size = len(cast(list[object], kwargs["cis"]))
        return [object() for _ in range(batch_size)], [
            object() for _ in range(batch_size)
        ]

    monkeypatch.setattr(module, "prepare_cis", fake_prepare_cis)
    monkeypatch.setattr(
        module,
        "get_all_pairs_cl_ja_effects_with_attributions",
        fake_trace_batch,
    )

    module.compute_circuits(
        object(),
        object(),
        ["question-0", "question-1", "question-2"],
        config=SimpleNamespace(verbose=False),
        seed_responses=["seed-0", "seed-1", "seed-2"],
        bs=2,
        true_answers=[["alpha"], ["beta"], ["gamma"]],
    )

    assert seen_answers == [
        [["alpha"], ["beta"]],
        [["gamma"]],
    ]
    assert seen_focus_tokens == [
        [[11], [22]],
        [[33]],
    ]


@pytest.mark.parametrize(
    ("module_name", "prepare_result"),
    [
        ("circuits.tracing.trace", ([], [1], [0.0])),
        (
            "circuits.analysis.process_circuits",
            (SimpleNamespace(input_ids=[1]), [1]),
        ),
    ],
)
def test_prepare_cis_rejects_misaligned_true_answers(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    prepare_result: tuple[object, ...],
) -> None:
    module = _import_target_module(monkeypatch, module_name)
    monkeypatch.setattr(module, "prepare_ci", lambda *_args, **_kwargs: prepare_result)

    with pytest.raises(ValueError, match=r"zip\(\) argument 3 is shorter"):
        module.prepare_cis(
            object(),
            SimpleNamespace(pad_token_id=0),
            ["question-0", "question-1"],
            ["seed-0", "seed-1"],
            true_answers=[["alpha"]],
        )
