"""CPU-only tests for frozen-response tracing and compact artifacts."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from circuits.tracing.artifact import (
    DATA_FILENAME,
    load_compact_trace,
    save_compact_trace,
    validate_compact_trace_data,
)
from circuits.tracing.clja import ADAGConfig
from circuits.tracing.instrumentation import TraceInstrumentation
from circuits.tracing.trace import (
    CircuitData,
    prepare_teacher_forced_input,
    trace_teacher_forced_response,
)


class FakeChatTokenizer:
    name_or_path = "fake/chat-model"
    chat_template = "fake-template-v1"
    pad_token_id = 0

    @staticmethod
    def _prefix(messages):
        system_marker = 8 if messages and messages[0]["role"] == "system" else 7
        return [1, system_marker, 12, 13]

    def apply_chat_template(self, messages, *, add_generation_prompt, chat_template):
        assert chat_template == self.chat_template
        prefix = self._prefix(messages)
        if add_generation_prompt:
            assert messages[-1]["role"] == "user"
            return prefix
        assert messages[-1]["role"] == "assistant"
        content = messages[-1]["content"]
        return prefix + [30 + ord(character) % 50 for character in content] + [99]

    def decode(self, token_ids):
        return f"tok-{token_ids[0]}"


class FakeModel:
    def __init__(self):
        self._parameter = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(_name_or_path="fake/model")
        self.forward_calls = 0

    def parameters(self):
        yield self._parameter

    def __call__(self, *, input_ids, attention_mask):
        self.forward_calls += 1
        assert input_ids.shape == attention_mask.shape
        vocab_logits = torch.arange(128, dtype=torch.float32)
        logits = vocab_logits.repeat(input_ids.shape[0], input_ids.shape[1], 1)
        return SimpleNamespace(logits=logits)


def test_prepare_teacher_forced_input_uses_exact_template_boundary_and_truncates():
    prepared = prepare_teacher_forced_input(
        FakeChatTokenizer(),
        "question",
        "abcd",
        [1, 3],
        system_prompt="system",
    )

    assert prepared.assistant_prefix_token_count == 4
    assert prepared.response_token_count == 4
    assert prepared.included_response_token_count == 4
    assert prepared.input_ids == [1, 8, 12, 13, 77, 78, 79, 30]
    assert prepared.target_token_ids == [78, 30]
    assert prepared.target_prediction_positions == [4, 6]
    assert prepared.attention_mask == [1] * 8


@pytest.mark.parametrize(
    ("positions", "error"),
    [
        ([], "at least one"),
        ([1, 1], "unique"),
        ([2, 1], "sorted"),
        ([-1], "negative"),
        ([4], "outside"),
    ],
)
def test_prepare_teacher_forced_input_validates_response_positions(positions, error):
    with pytest.raises((TypeError, ValueError), match=error):
        prepare_teacher_forced_input(FakeChatTokenizer(), "q", "abcd", positions)


def test_trace_teacher_forced_response_wires_single_target_to_clja(monkeypatch):
    import circuits.tracing.trace as trace_module

    captured = {}

    def fake_convert(nodes, edges, labels, starts, **kwargs):
        assert nodes == [[captured_nodes]]
        assert edges == [[captured_edges]]
        assert labels == ["row-1"]
        assert starts == [[0]]
        return pd.DataFrame({"layer": [0]}), pd.DataFrame({"layer": ["0->1"]})

    captured_nodes = object()
    captured_edges = object()

    def fake_clja_with_known_values(**kwargs):
        captured.update(kwargs)
        return [captured_nodes], [captured_edges]

    monkeypatch.setattr(
        trace_module,
        "get_all_pairs_cl_ja_effects_with_attributions",
        fake_clja_with_known_values,
    )
    monkeypatch.setattr(trace_module, "convert_circuit_to_dataframes", fake_convert)

    model = FakeModel()
    instrumentation = TraceInstrumentation(device="cpu")
    data = trace_teacher_forced_response(
        model,
        FakeChatTokenizer(),
        "question",
        "abcd",
        [2],
        ADAGConfig(device="cpu"),
        label="row-1",
        instrumentation=instrumentation,
    )

    assert model.forward_calls == 1
    assert captured["cis"] == [[1, 7, 12, 13, 77, 78, 79]]
    assert captured["attention_masks"] == [[1] * 7]
    assert captured["focus_logits"] == [[79]]
    assert captured["tgt_tokens"] == [5]
    assert captured["src_tokens"] == list(range(6))
    assert captured["instrumentation"] is instrumentation
    assert data.target_logits == [[79]]
    assert data.target_logit_values == [[79.0]]
    assert data.target_provenance[0]["response_token_position"] == 2
    assert data.target_provenance[0]["absolute_token_position"] == 6
    assert data.target_provenance[0]["prediction_token_position"] == 5
    assert data.benchmark_only is False
    snapshot = data.trace_metadata["instrumentation"]
    assert set(("prepare_input", "target_scoring", "clja_total", "dataframe_conversion")) <= set(
        snapshot["stages"]
    )
    assert snapshot["counters"]["raw_node_count"] == 1
    assert snapshot["counters"]["raw_edge_count"] == 1
    assert snapshot["counters"]["final_dataframe_node_count"] == 1
    assert snapshot["counters"]["final_dataframe_edge_count"] == 1


def test_multi_target_trace_requires_benchmark_only(monkeypatch):
    model = FakeModel()
    with pytest.raises(ValueError, match="benchmark_only=True"):
        trace_teacher_forced_response(
            model,
            FakeChatTokenizer(),
            "question",
            "abcd",
            [0, 1],
            ADAGConfig(device="cpu"),
        )
    assert model.forward_calls == 0


def test_instrumentation_does_not_change_teacher_forced_trace_outputs(monkeypatch):
    import circuits.tracing.trace as trace_module

    node = object()
    edge = object()
    monkeypatch.setattr(
        trace_module,
        "get_all_pairs_cl_ja_effects_with_attributions",
        lambda **_kwargs: ([node], [edge]),
    )
    monkeypatch.setattr(
        trace_module,
        "convert_circuit_to_dataframes",
        lambda *_args, **_kwargs: (
            pd.DataFrame({"layer": [0], "attribution": [0.5]}),
            pd.DataFrame({"layer": ["0->1"], "weight": [0.25]}),
        ),
    )
    kwargs = {
        "tokenizer": FakeChatTokenizer(),
        "prompt": "question",
        "response": "abcd",
        "target_response_positions": [2],
        "config": ADAGConfig(device="cpu"),
    }

    plain = trace_teacher_forced_response(model=FakeModel(), **kwargs)
    instrumented = trace_teacher_forced_response(
        model=FakeModel(),
        instrumentation=TraceInstrumentation(device="cpu"),
        **kwargs,
    )

    pd.testing.assert_frame_equal(plain.df_node, instrumented.df_node)
    pd.testing.assert_frame_equal(plain.df_edge, instrumented.df_edge)
    assert plain.cis == instrumented.cis
    assert plain.attention_masks == instrumented.attention_masks
    assert plain.target_logits == instrumented.target_logits
    assert plain.target_logit_probs == instrumented.target_logit_probs
    assert plain.target_logit_values == instrumented.target_logit_values
    assert plain.target_provenance == instrumented.target_provenance
    assert "instrumentation" not in plain.trace_metadata
    assert {
        key: value
        for key, value in instrumented.trace_metadata.items()
        if key != "instrumentation"
    } == plain.trace_metadata


def _circuit_data(*, target_count=1, benchmark_only=False):
    token_ids = list(range(40, 40 + target_count))
    provenance = [
        {
            "response_token_position": index,
            "absolute_token_position": 5 + index,
            "prediction_token_position": 4 + index,
            "token_id": token_id,
            "token_text": f"tok-{token_id}",
            "logit": 1.0,
            "probability": 0.5,
        }
        for index, token_id in enumerate(token_ids)
    ]
    return CircuitData(
        df_node=pd.DataFrame(
            {
                "layer": [0],
                "attribution": [0.25],
                "activation": [1.5],
                "attr_map": [[0.1, 0.2]],
            }
        ),
        df_edge=pd.DataFrame(
            {"layer": ["0->1"], "attribution": [0.125], "weight": [-0.5]}
        ),
        cis=[[1, 2, 3]],
        attention_masks=[[1, 1, 1]],
        labels=["row-1"],
        target_logits=[token_ids],
        target_logit_probs=[[0.5] * target_count],
        target_logit_values=[[1.0] * target_count],
        target_provenance=provenance,
        trace_metadata={"trace_mode": "teacher_forced_response"},
        benchmark_only=benchmark_only,
        k=target_count,
        config=ADAGConfig(device="cpu"),
        model_id="fake/model",
    )


def test_compact_trace_round_trip_and_atomic_destination(tmp_path):
    destination = tmp_path / "trace-unit"
    save_compact_trace(
        destination,
        _circuit_data(),
        metrics={"elapsed_seconds": 3.5},
        manifest={"dataset_row_id": "row-1"},
    )

    loaded = load_compact_trace(destination)
    assert loaded.metrics == {"elapsed_seconds": 3.5}
    assert loaded.manifest["dataset_row_id"] == "row-1"
    assert loaded.manifest["target_count"] == 1
    assert loaded.manifest["numerically_valid"] is True
    assert loaded.manifest["scientifically_reusable"] is True
    assert loaded.circuit_data.df_node.iloc[0].attr_map == [0.1, 0.2]
    assert not list(tmp_path.glob(".trace-unit.tmp-*"))
    with pytest.raises(FileExistsError):
        save_compact_trace(destination, _circuit_data())


def test_compact_trace_rejects_unmarked_multi_target_and_marks_benchmark(tmp_path):
    with pytest.raises(ValueError, match="benchmark_only"):
        save_compact_trace(tmp_path / "invalid", _circuit_data(target_count=2))

    destination = tmp_path / "benchmark"
    save_compact_trace(
        destination,
        _circuit_data(target_count=2, benchmark_only=True),
    )
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["benchmark_only"] is True
    assert manifest["scientifically_reusable"] is False


def test_compact_trace_detects_payload_corruption(tmp_path):
    destination = tmp_path / "trace-unit"
    save_compact_trace(destination, _circuit_data())
    with (destination / DATA_FILENAME).open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(ValueError, match="(size|checksum) mismatch"):
        load_compact_trace(destination)


@pytest.mark.parametrize(
    ("frame_name", "column", "bad_value"),
    [
        ("df_node", "attribution", float("nan")),
        ("df_node", "activation", float("inf")),
        ("df_edge", "attribution", float("-inf")),
        ("df_edge", "weight", "not-a-number"),
    ],
)
def test_compact_trace_rejects_invalid_scalar_numerics(
    tmp_path, frame_name, column, bad_value
):
    data = _circuit_data()
    if isinstance(bad_value, str):
        getattr(data, frame_name)[column] = getattr(data, frame_name)[column].astype(
            object
        )
    getattr(data, frame_name).loc[0, column] = bad_value

    with pytest.raises(ValueError, match=rf"{frame_name}\.{column}"):
        save_compact_trace(tmp_path / "invalid", data)

    assert not (tmp_path / "invalid").exists()


def test_compact_trace_preserves_structurally_empty_graphs(tmp_path):
    data = _circuit_data()
    data.df_node = data.df_node.iloc[0:0].copy()
    data.df_edge = data.df_edge.iloc[0:0].copy()

    assert validate_compact_trace_data(data) == 1
    destination = tmp_path / "empty-graph"
    save_compact_trace(destination, data)
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["numerically_valid"] is True
    assert manifest["node_count"] == 0
    assert manifest["edge_count"] == 0


def test_circuit_data_merge_remains_compatible_with_old_pickles():
    shards = [_circuit_data(), _circuit_data()]
    for shard in shards:
        shard.df_node["label"] = "row-1___0"
        shard.df_edge["label"] = "row-1___0"
        del shard.target_logit_values
        del shard.target_provenance
        del shard.trace_metadata
        del shard.benchmark_only

    merged = CircuitData.merge(shards)

    assert merged.target_logit_values == []
    assert merged.target_provenance == []
    assert merged.benchmark_only is False
    assert merged.trace_metadata == {"merged_shard_metadata": [{}, {}]}
