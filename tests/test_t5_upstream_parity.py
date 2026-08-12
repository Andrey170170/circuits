"""CPU parity coverage for strict upstream-style top-five tracing (T5)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import ClassVar

import pandas as pd
import torch
from circuits.tracing.clja import ADAGConfig
from circuits.tracing.trace import (
    TOPK_CONTRIBUTION_SCHEMA_ID,
    prepare_ci,
    trace_teacher_forced_candidates,
)
from torch import nn


class _ParityTokenizer:
    """Tiny chat tokenizer with an exact assistant-prefix boundary."""

    chat_template = "fake-t5-parity-template-v1"
    eos_token_id = 99
    pad_token_id = 0

    _response_ids: ClassVar[dict[str, int]] = {"a": 20, "b": 21}

    def apply_chat_template(
        self,
        messages,
        *,
        add_generation_prompt,
        chat_template,
    ):
        assert chat_template == self.chat_template
        prefix = [1, 2, 3]
        if add_generation_prompt:
            assert messages[-1]["role"] == "user"
            return prefix
        assert messages[-1]["role"] == "assistant"
        content = messages[-1]["content"]
        return [
            *prefix,
            *(self._response_ids[character] for character in content),
            self.eos_token_id,
        ]

    def decode(self, token_ids):
        return f"tok-{token_ids[0]}"


class _ParityLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.down_proj = nn.Linear(1, 1, bias=False)
        self.mlp.down_proj.weight.data.fill_(1.0)


class _ParityModel(nn.Module):
    """One-dimensional causal stand-in with unique, stable top-five logits."""

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(128, 1)
        self.model.embed_tokens.weight.data.fill_(1.0)
        self.model.layers = nn.ModuleList([_ParityLayer()])
        self.lm_head = nn.Linear(1, 32, bias=False)
        self.lm_head.weight.data.copy_(
            torch.arange(32, dtype=torch.float32).unsqueeze(1)
        )
        self.config = SimpleNamespace(
            _name_or_path="fake/t5-parity-model",
            num_hidden_layers=1,
            hidden_size=1,
        )

    def forward(self, input_ids=None, *, inputs_embeds=None, attention_mask=None):
        del attention_mask
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
        hidden = self.model.layers[0].mlp.down_proj(inputs_embeds)
        return SimpleNamespace(logits=self.lm_head(hidden))


def test_teacher_forced_t5_matches_upstream_top5_sum_and_scalar_graph(
    monkeypatch,
) -> None:
    """T5 is the legacy top-five logit sum at one teacher-forced position.

    The candidate axis retains five raw contribution coordinates, but its
    all-one objective must produce the same scalar pruning/graph attribution as
    the upstream ``focus_logits`` path.
    """

    import circuits.tracing.clja as clja_module
    import circuits.tracing.trace as trace_module

    model = _ParityModel()
    tokenizer = _ParityTokenizer()
    config = ADAGConfig(
        device="cpu",
        disable_stop_grad=True,
        skip_attr_contrib=True,
        percentage_threshold=None,
    )

    legacy_ci, legacy_top5, _ = prepare_ci(
        model,
        tokenizer,
        "question",
        "a",
        k=5,
        system_prompt="system",
    )
    prediction_position = len(legacy_ci) - 1

    real_initial_attribution = clja_module._get_grad_attributions_from_logits
    attribution_calls = []

    def capture_initial_attribution(*args, **kwargs):
        result = real_initial_attribution(*args, **kwargs)
        attribution_calls.append(
            {
                "focus_positions": list(args[3]),
                "focus_logits": [list(row) for row in kwargs["focus_logits"]],
                "objective_weights": kwargs["objective_weights"],
                "mlp_attribution": result[0].clone(),
                "embed_attribution": result[1].clone(),
                "goal": result[2].clone(),
            }
        )
        return result

    def retain_one_neuron(**kwargs):
        layer_count, _, token_count, neuron_count, _ = kwargs[
            "mlp_final_attributions"
        ].shape
        mask = torch.zeros((layer_count, token_count, neuron_count), dtype=torch.bool)
        mask[0, prediction_position, 0] = True
        return mask

    graph_calls = []

    def capture_graph(*_args, **kwargs):
        graph_calls.append(
            {
                "focus_positions": list(kwargs["focus_positions"]),
                "focus_logits": [list(row) for row in kwargs["focus_logits"]],
                "tgt_tokens": list(kwargs["tgt_tokens"]),
                "objective_weights": kwargs["candidate_objective_weights"],
            }
        )
        return [], []

    monkeypatch.setattr(
        clja_module,
        "_get_grad_attributions_from_logits",
        capture_initial_attribution,
    )
    monkeypatch.setattr(
        clja_module,
        "_get_global_important_neurons_mask",
        retain_one_neuron,
    )
    monkeypatch.setattr(clja_module, "_get_cl_ja_based_edges", capture_graph)
    monkeypatch.setattr(
        trace_module,
        "convert_circuit_to_dataframes",
        lambda *_args, **_kwargs: (pd.DataFrame(), pd.DataFrame()),
    )

    # This is the graph objective used by the original compute_circuits(k=5)
    # route after prepare_ci selected its next-token candidates.
    clja_module.get_all_pairs_cl_ja_effects_with_attributions(
        model=model,
        tokenizer=tokenizer,
        cis=[legacy_ci],
        config=config,
        attention_masks=[[1] * len(legacy_ci)],
        focus_logits=[legacy_top5],
        src_tokens=list(range(len(legacy_ci))),
        tgt_tokens=[prediction_position] * 5,
    )

    candidate = trace_teacher_forced_candidates(
        model,
        tokenizer,
        "question",
        "ab",
        target_response_position=1,
        config=config,
        candidate_policy_id="model_top5",
        candidate_count=5,
        joint_objective_id="raw_logit_sum",
        system_prompt="system",
    )

    selected_ids = [item.token_id for item in candidate.candidate_selection.candidates]
    assert selected_ids == legacy_top5 == [31, 30, 29, 28, 27]
    assert candidate.joint_objective.candidate_weights == (1.0,) * 5
    assert candidate.joint_objective.percentage_threshold_reference == (
        "signed_joint_objective"
    )

    legacy_attr, t5_attr = attribution_calls
    assert (
        legacy_attr["focus_positions"]
        == t5_attr["focus_positions"]
        == ([prediction_position] * 5)
    )
    assert legacy_attr["focus_logits"] == t5_attr["focus_logits"] == [legacy_top5]
    assert legacy_attr["objective_weights"] is None
    assert t5_attr["objective_weights"] == (1.0,) * 5
    torch.testing.assert_close(legacy_attr["goal"], t5_attr["goal"])
    torch.testing.assert_close(
        legacy_attr["mlp_attribution"],
        t5_attr["mlp_attribution"][..., : len(legacy_ci), :],
    )
    torch.testing.assert_close(
        legacy_attr["embed_attribution"],
        t5_attr["embed_attribution"][..., : len(legacy_ci)],
    )

    legacy_graph, t5_graph = graph_calls
    assert legacy_graph["focus_positions"] == t5_graph["focus_positions"]
    assert legacy_graph["focus_logits"] == t5_graph["focus_logits"]
    assert legacy_graph["tgt_tokens"] == t5_graph["tgt_tokens"]
    assert legacy_graph["objective_weights"] is None
    assert t5_graph["objective_weights"] == (1.0,) * 5
    assert candidate.candidate_contribution_schema == {
        "schema_id": TOPK_CONTRIBUTION_SCHEMA_ID,
        "axis": "candidate_index",
        "width": 5,
        "semantics": "gradient_times_activation_for_each_raw_candidate_logit",
        "scalar_graph_attribution_semantics": "named_joint_objective",
    }
