"""CPU-only tests for the staged BonaFide performance benchmark tooling."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from circuits.tracing.artifact import DATA_FILENAME, load_compact_trace, save_compact_trace
from circuits.tracing.clja import ADAGConfig
from circuits.tracing.trace import CircuitData
from scripts.bonafide.manifest import (
    SCHEMA_VERSION,
    build_manifest,
    load_deduplicated_examples,
    response_trace_tokens,
    select_evenly_spaced_positions,
)
from scripts.bonafide.runner import (
    RUN_CONFIG_SCHEMA,
    _completed_artifact_matches,
    collect_runtime_environment,
    run_wave,
    validate_runtime_trace_against_item,
    validate_run_config,
    wave_stop_reason,
)


class FakeChatTokenizer:
    """Tiny tokenizer with an explicit assistant prefix and end-turn suffix."""

    name_or_path = "fake/model"
    chat_template = "fake-template"
    eos_token_id = 999

    def apply_chat_template(
        self, messages, *, add_generation_prompt: bool, chat_template: str
    ) -> list[int]:
        del chat_template
        prompt = next(message["content"] for message in messages if message["role"] == "user")
        prefix = [1, *[100 + ord(char) for char in prompt], 2]
        assistant = [message for message in messages if message["role"] == "assistant"]
        if add_generation_prompt:
            return prefix
        response = assistant[-1]["content"]
        return [*prefix, *[1000 + ord(char) for char in response], self.eos_token_id]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(token_id - 1000) for token_id in token_ids)


FIELDNAMES = ["id", "question_id", "label_type", "target_model", "prompt", "cot"]


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _row(index: int, *, prompt: str, response: str, annotation: str | None = None):
    return {
        "id": annotation or f"row-{index}",
        "question_id": f"question-{index}",
        "label_type": "FAITHFUL_STEP" if index % 2 else "UNFAITHFUL_STEP",
        "target_model": "fake/model",
        "prompt": prompt,
        "cot": response,
    }


def test_dedupe_preserves_all_annotation_provenance(tmp_path: Path) -> None:
    csv_path = tmp_path / "bonafide.csv"
    duplicate_a = _row(1, prompt="p", response="answer", annotation="annotation-b")
    duplicate_b = {
        **_row(2, prompt="p", response="answer", annotation="annotation-a"),
        "question_id": "another-question",
        "label_type": "FAITHFUL_COT",
    }
    _write_csv(csv_path, [duplicate_a, duplicate_b, _row(3, prompt="q", response="other")])

    examples = load_deduplicated_examples(csv_path)

    assert len(examples) == 2
    merged = next(example for example in examples if example.prompt == "p")
    assert merged.annotation_row_ids == ("annotation-a", "annotation-b")
    assert merged.question_ids == ("another-question", "question-1")
    assert merged.label_types == ("FAITHFUL_COT", "FAITHFUL_STEP")


def test_manifest_uses_exact_chat_template_lengths_and_separate_waves(tmp_path: Path) -> None:
    csv_path = tmp_path / "bonafide.csv"
    rows = [
        _row(index, prompt=f"p{index}", response="x" * length)
        for index, length in enumerate((2, 4, 8, 16, 32, 40), start=1)
    ]
    # A second annotation for the 8-token example must not create another item.
    rows.append(
        {
            **rows[2],
            "id": "anchor-8",
            "question_id": "extra-question",
            "label_type": "FAITHFUL_COT",
        }
    )
    _write_csv(csv_path, rows)
    tokenizer = FakeChatTokenizer()

    assert len(response_trace_tokens(tokenizer, "p", "abcdef")) == 6
    manifest = build_manifest(
        csv_path=csv_path,
        tokenizer=tokenizer,
        target_model="fake/model",
        model_revision="exact-revision",
        sample_count=4,
        anchor_annotation_ids=["anchor-8"],
        wave2_annotation_id="row-6",
        wave2_widths=[1, 2, 4, 8, 16, 32],
    )

    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["execution_contract"]["merge_graphs"] is False
    wave1, wave2, wave2b = manifest["waves"]
    assert wave1["wave_id"] == "wave1-mixed-final-token"
    assert len(wave1["items"]) == 4
    assert any("anchor-8" in item["example"]["annotation_row_ids"] for item in wave1["items"])
    for item in wave1["items"]:
        assert item["target_selection"]["response_token_positions"] == [
            item["response_token_count"] - 1
        ]
        assert item["objective"]["benchmark_only_multi_target"] is False

    assert wave2["wave_id"] == "wave2-progressive-target-window"
    assert [item["target_selection"]["width"] for item in wave2["items"]] == [
        1,
        2,
        4,
        8,
        16,
        32,
    ]
    assert wave2["items"][0]["objective"]["benchmark_only_multi_target"] is False
    assert all(
        item["objective"]["benchmark_only_multi_target"]
        for item in wave2["items"][1:]
    )

    assert wave2b["wave_id"] == "wave2b-independent-target-positions"
    assert [
        item["target_selection"]["response_token_positions"][0]
        for item in wave2b["items"]
    ] == [0, 3, 5, 8, 10, 13, 16, 18, 21, 23, 26, 29, 31, 34, 36, 39]
    assert all(item["target_selection"]["width"] == 1 for item in wave2b["items"])
    assert all(
        item["objective"]["benchmark_only_multi_target"] is False
        for item in wave2b["items"]
    )
    assert {
        item["example"]["example_id"] for item in wave2b["items"]
    } == {wave2["items"][0]["example"]["example_id"]}
    assert len({item["artifact_id"] for item in wave2b["items"]}) == 16
    assert manifest["execution_contract"]["model_load_scope"] == "selected_wave"


def test_even_position_selection_spans_response_and_rejects_oversampling() -> None:
    assert select_evenly_spaced_positions(170, 16) == [
        0,
        11,
        23,
        34,
        45,
        56,
        68,
        79,
        90,
        101,
        113,
        124,
        135,
        146,
        158,
        169,
    ]
    assert select_evenly_spaced_positions(8, 1) == [7]
    with pytest.raises(ValueError, match="distinct positions"):
        select_evenly_spaced_positions(8, 9)


def _config() -> dict:
    return {
        "schema_version": RUN_CONFIG_SCHEMA,
        "batch_size": 1,
        "model": {
            "model_id": "fake/model",
            "revision": "exact-revision",
            "device": "cuda:0",
            "dtype": "bfloat16",
        },
        "adag_config": {},
    }


def test_dry_run_selects_only_requested_wave(tmp_path: Path) -> None:
    csv_path = tmp_path / "bonafide.csv"
    _write_csv(
        csv_path,
        [_row(index, prompt=f"p{index}", response="x" * length) for index, length in enumerate((8, 40), 1)],
    )
    manifest = build_manifest(
        csv_path=csv_path,
        tokenizer=FakeChatTokenizer(),
        target_model="fake/model",
        model_revision="exact-revision",
        sample_count=2,
        wave2_annotation_id="row-2",
        wave2_widths=[1, 2, 4, 8, 16, 32],
    )

    records = run_wave(
        config=_config(),
        manifest=manifest,
        wave_id="wave1-mixed-final-token",
        artifact_root=tmp_path / "artifacts",
        summary_jsonl=tmp_path / "summary.jsonl",
        dry_run=True,
    )

    assert len(records) == 2
    assert {record["wave_id"] for record in records} == {"wave1-mixed-final-token"}
    assert {record["status"] for record in records} == {"planned"}
    assert not (tmp_path / "summary.jsonl").exists()


def test_run_config_rejects_non_unit_batch() -> None:
    config = _config()
    config["batch_size"] = 2
    with pytest.raises(ValueError, match="batch_size=1"):
        validate_run_config(config)


def _valid_runtime_trace(*, response_count: int = 8, token_id: int = 77) -> CircuitData:
    return CircuitData(
        df_node=pd.DataFrame(
            {"attribution": [0.1], "activation": [0.2], "layer": [0]}
        ),
        df_edge=pd.DataFrame(
            {"attribution": [0.3], "weight": [0.4], "layer": ["0->1"]}
        ),
        cis=[[1, 2]],
        attention_masks=[[1, 1]],
        labels=["example"],
        target_logits=[[token_id]],
        target_logit_probs=[[0.5]],
        target_logit_values=[[1.0]],
        target_provenance=[
            {
                "response_token_position": response_count - 1,
                "token_id": token_id,
            }
        ],
        trace_metadata={"response_token_count": response_count},
        benchmark_only=False,
        k=1,
        config=ADAGConfig(device="cpu"),
        model_id="fake/model",
    )


def _runtime_item(*, response_count: int = 8, token_id: int = 77) -> dict:
    return {
        "response_token_count": response_count,
        "target_selection": {
            "response_token_positions": [response_count - 1],
            "final_target_token_id": token_id,
        },
    }


def test_runtime_trace_must_match_frozen_manifest_tokenization() -> None:
    trace = _valid_runtime_trace()
    validate_runtime_trace_against_item(trace, _runtime_item())

    with pytest.raises(ValueError, match="response token count"):
        validate_runtime_trace_against_item(trace, _runtime_item(response_count=9))
    with pytest.raises(ValueError, match="final target token ID"):
        validate_runtime_trace_against_item(trace, _runtime_item(token_id=78))

    trace.target_provenance[0]["response_token_position"] = 6
    with pytest.raises(ValueError, match="target response positions"):
        validate_runtime_trace_against_item(trace, _runtime_item())


def test_resume_verifies_payload_checksum_before_skipping(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    identity = {"sha256": "identity"}
    save_compact_trace(
        artifact,
        _valid_runtime_trace(),
        manifest={"artifact_identity": identity},
    )
    assert _completed_artifact_matches(artifact, identity) is True

    with (artifact / DATA_FILENAME).open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ValueError, match="size mismatch|checksum mismatch"):
        _completed_artifact_matches(artifact, identity)


def test_runtime_environment_records_core_package_versions() -> None:
    environment = collect_runtime_environment()
    assert environment["python"]
    assert environment["packages"]["torch"]
    assert environment["packages"]["transformers"]


def _single_item_manifest() -> dict:
    item = {
        "artifact_id": "source-trace-1",
        "response_token_count": 8,
        "target_selection": {
            "width": 1,
            "response_token_positions": [7],
            "final_target_token_id": 77,
        },
        "objective": {"benchmark_only_multi_target": False},
        "example": {
            "example_id": "example-1",
            "annotation_row_ids": ["row-1"],
            "question_ids": ["question-1"],
            "label_types": ["FAITHFUL_STEP"],
            "prompt": "question",
            "response": "response",
        },
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "tokenizer": {"model_id": "fake/model", "revision": "exact-revision"},
        "waves": [{"wave_id": "instrumented", "items": [item]}],
    }


def test_runner_persists_instrumentation_in_success_record_and_artifact(
    tmp_path: Path, monkeypatch
) -> None:
    import scripts.bonafide.runner as runner_module

    config = _config()
    config["model"]["device"] = "cpu"

    def fake_trace(**kwargs):
        kwargs["instrumentation"].set_counter("probe", 5)
        return _valid_runtime_trace()

    monkeypatch.setattr(
        runner_module, "_load_model_and_tokenizer", lambda _config: (object(), object())
    )
    monkeypatch.setattr(runner_module, "trace_teacher_forced_response", fake_trace)
    records = run_wave(
        config=config,
        manifest=_single_item_manifest(),
        wave_id="instrumented",
        artifact_root=tmp_path / "artifacts",
        summary_jsonl=tmp_path / "summary.jsonl",
    )

    assert records[0]["status"] == "complete"
    assert records[0]["instrumentation"]["counters"]["probe"] == 5
    summary_record = json.loads((tmp_path / "summary.jsonl").read_text().splitlines()[0])
    assert summary_record["instrumentation"]["schema_version"].endswith("v1")
    loaded = load_compact_trace(records[0]["artifact_path"])
    assert loaded.metrics["instrumentation"]["counters"]["probe"] == 5
    assert loaded.circuit_data.trace_metadata["instrumentation"]["counters"]["probe"] == 5


@pytest.mark.parametrize(
    ("failure", "expected_status"),
    [
        (RuntimeError("synthetic trace failure"), "error"),
        (torch.cuda.OutOfMemoryError("synthetic OOM"), "oom"),
    ],
)
def test_runner_retains_partial_instrumentation_in_failure_record(
    tmp_path: Path, monkeypatch, failure: Exception, expected_status: str
) -> None:
    import scripts.bonafide.runner as runner_module

    config = _config()
    config["model"]["device"] = "cpu"
    config["continue_on_error"] = True

    def failing_trace(**kwargs):
        kwargs["instrumentation"].set_counter("selected_neuron_count", 12)
        raise failure

    monkeypatch.setattr(
        runner_module, "_load_model_and_tokenizer", lambda _config: (object(), object())
    )
    monkeypatch.setattr(runner_module, "trace_teacher_forced_response", failing_trace)
    records = run_wave(
        config=config,
        manifest=_single_item_manifest(),
        wave_id="instrumented",
        artifact_root=tmp_path / "artifacts",
        summary_jsonl=tmp_path / "summary.jsonl",
    )

    assert records[0]["status"] == expected_status
    assert records[0]["instrumentation"]["counters"]["selected_neuron_count"] == 12
    summary_record = json.loads((tmp_path / "summary.jsonl").read_text().splitlines()[0])
    assert summary_record["instrumentation"]["counters"]["selected_neuron_count"] == 12


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        ({"status": "oom"}, "cuda_oom"),
        (
            {
                "status": "complete",
                "trace_wall_seconds": 11.0,
                "cuda_headroom_after_peak_bytes": 100,
            },
            "max_trace_seconds_exceeded",
        ),
        (
            {
                "status": "complete",
                "trace_wall_seconds": 1.0,
                "cuda_headroom_after_peak_bytes": 3,
            },
            "min_cuda_headroom_not_met",
        ),
    ],
)
def test_wave_stop_gates(record: dict, expected: str) -> None:
    assert (
        wave_stop_reason(
            record,
            uses_cuda=True,
            max_trace_seconds=10.0,
            min_cuda_headroom_bytes=4,
            stop_on_oom=True,
        )
        == expected
    )
