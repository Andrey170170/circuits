"""CPU-only tests for the staged BonaFide performance benchmark tooling."""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import pandas as pd
import pytest
import torch
from circuits.tracing.artifact import (
    DATA_FILENAME,
    load_compact_trace,
    save_compact_trace,
)
from circuits.tracing.clja import ADAGConfig
from circuits.tracing.trace import CircuitData
from scripts.bonafide.cuda_headroom import assess_cuda_headroom
from scripts.bonafide.manifest import (
    SCHEMA_VERSION,
    build_manifest,
    load_deduplicated_examples,
    response_trace_tokens,
    select_evenly_spaced_positions,
    select_stratified_random_positions,
)
from scripts.bonafide.runner import (
    RUN_CONFIG_SCHEMA,
    _completed_artifact_matches,
    collect_runtime_environment,
    normalized_instrumentation,
    normalized_trace_warmup,
    run_wave,
    runtime_artifact_identity,
    trace_warmup_applies,
    validate_run_config,
    validate_runtime_trace_against_item,
    validate_target_selection,
    validate_wave_sampling_design,
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
        prompt = next(
            message["content"] for message in messages if message["role"] == "user"
        )
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
    _write_csv(
        csv_path, [duplicate_a, duplicate_b, _row(3, prompt="q", response="other")]
    )

    examples = load_deduplicated_examples(csv_path)

    assert len(examples) == 2
    merged = next(example for example in examples if example.prompt == "p")
    assert merged.annotation_row_ids == ("annotation-a", "annotation-b")
    assert merged.question_ids == ("another-question", "question-1")
    assert merged.label_types == ("FAITHFUL_COT", "FAITHFUL_STEP")


def test_manifest_uses_exact_chat_template_lengths_and_separate_waves(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "bonafide.csv"
    rows = [
        _row(index, prompt=f"p{index}", response="x" * length)
        for index, length in enumerate((8, 10, 12, 16, 32, 40), start=1)
    ]
    # A second annotation for the 12-token example must not create another item.
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
    wave1, wave2, wave2b, *wave2c_waves = manifest["waves"]
    assert wave1["wave_id"] == "wave1-mixed-final-token"
    assert len(wave1["items"]) == 4
    assert any(
        "anchor-8" in item["example"]["annotation_row_ids"] for item in wave1["items"]
    )
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
        item["objective"]["benchmark_only_multi_target"] for item in wave2["items"][1:]
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
    assert {item["example"]["example_id"] for item in wave2b["items"]} == {
        wave2["items"][0]["example"]["example_id"]
    }
    assert len({item["artifact_id"] for item in wave2b["items"]}) == 16
    assert manifest["execution_contract"]["model_load_scope"] == "selected_wave"

    assert len(wave2c_waves) == 3
    assert all(
        wave["wave_id"].startswith("wave2c-stratified-independent-")
        for wave in wave2c_waves
    )
    assert all(len(wave["items"]) == 8 for wave in wave2c_waves)
    assert {wave["items"][0]["example"]["example_id"] for wave in wave2c_waves} == {
        item["example"]["example_id"] for item in wave1["items"]
    } - {wave2["items"][0]["example"]["example_id"]}
    for wave in wave2c_waves:
        validate_wave_sampling_design(wave, manifest)
        assert len({item["example"]["example_id"] for item in wave["items"]}) == 1
        assert len({item["artifact_id"] for item in wave["items"]}) == 8
        for stratum_index, item in enumerate(wave["items"]):
            sampling = item["target_selection"]["sampling"]
            position = item["target_selection"]["response_token_positions"][0]
            assert item["target_selection"]["width"] == 1
            assert item["objective"]["benchmark_only_multi_target"] is False
            assert sampling["stratum_index"] == stratum_index
            assert (
                sampling["stratum_start"]
                <= position
                < sampling["stratum_end_exclusive"]
            )
            assert sampling["stratum_size"] == (
                sampling["stratum_end_exclusive"] - sampling["stratum_start"]
            )
            assert sampling["selection_probability"] == pytest.approx(
                1 / sampling["stratum_size"]
            )
            assert sampling["projection_weight"] == sampling["stratum_size"]


def test_manifest_requires_wave2_reference_to_be_in_wave1(tmp_path: Path) -> None:
    csv_path = tmp_path / "bonafide.csv"
    _write_csv(
        csv_path,
        [
            _row(1, prompt="short", response="x" * 8),
            _row(2, prompt="middle", response="x" * 16),
            _row(3, prompt="long", response="x" * 40),
        ],
    )

    with pytest.raises(ValueError, match="Wave 2 reference must belong to Wave 1"):
        build_manifest(
            csv_path=csv_path,
            tokenizer=FakeChatTokenizer(),
            target_model="fake/model",
            model_revision="exact-revision",
            sample_count=2,
            wave2_annotation_id="row-2",
            wave2_widths=[1, 2, 4, 8, 16],
        )


def test_sample_seed_changes_source_artifact_identity_at_same_positions(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "bonafide.csv"
    _write_csv(
        csv_path,
        [
            _row(1, prompt="sampled", response="x" * 8),
            _row(2, prompt="reference", response="x" * 40),
        ],
    )
    kwargs = {
        "csv_path": csv_path,
        "tokenizer": FakeChatTokenizer(),
        "target_model": "fake/model",
        "model_revision": "exact-revision",
        "sample_count": 2,
        "wave2_annotation_id": "row-2",
        "wave2_widths": [1, 2, 4, 8, 16, 32],
    }
    seed_a = build_manifest(**kwargs, wave2c_seed="seed-a")["waves"][3]["items"]
    seed_b = build_manifest(**kwargs, wave2c_seed="seed-b")["waves"][3]["items"]

    assert [
        item["target_selection"]["response_token_positions"] for item in seed_a
    ] == [item["target_selection"]["response_token_positions"] for item in seed_b]
    assert [item["artifact_id"] for item in seed_a] != [
        item["artifact_id"] for item in seed_b
    ]


def test_wave_sampling_design_rejects_incomplete_or_disagreeing_items(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "bonafide.csv"
    _write_csv(
        csv_path,
        [
            _row(1, prompt="sampled", response="x" * 8),
            _row(2, prompt="reference", response="x" * 40),
        ],
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
    wave = manifest["waves"][3]
    validate_wave_sampling_design(wave, manifest)

    broken = copy.deepcopy(wave)
    broken["items"].pop()
    with pytest.raises(ValueError, match="one item for every stratum"):
        validate_wave_sampling_design(broken, manifest)

    broken = copy.deepcopy(wave)
    broken["items"][1]["target_selection"] = copy.deepcopy(
        broken["items"][0]["target_selection"]
    )
    with pytest.raises(ValueError, match="unique and complete"):
        validate_wave_sampling_design(broken, manifest)

    for field in ("seed", "sampler", "design"):
        broken = copy.deepcopy(wave)
        broken["items"][0]["target_selection"]["sampling"][field] = "disagrees"
        with pytest.raises(ValueError, match=f"{field} disagrees"):
            validate_wave_sampling_design(broken, manifest)

    broken = copy.deepcopy(wave)
    broken["sampling_design"]["response_token_population_size"] = 9
    with pytest.raises(ValueError, match="population size disagrees"):
        validate_wave_sampling_design(broken, manifest)

    broken = copy.deepcopy(wave)
    broken["sampling_design"]["excluded_reference_example_id"] = "wrong-reference"
    with pytest.raises(ValueError, match="disagrees with Wave 2"):
        validate_wave_sampling_design(broken, manifest)


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


def test_stratified_random_selection_is_stable_and_records_sampling_design() -> None:
    selections = select_stratified_random_positions(
        170,
        8,
        seed="fixed-seed",
        example_id="bf-example",
    )

    assert [selection["response_token_position"] for selection in selections] == [
        4,
        25,
        45,
        71,
        98,
        118,
        127,
        164,
    ]
    assert selections == select_stratified_random_positions(
        170,
        8,
        seed="fixed-seed",
        example_id="bf-example",
    )
    assert selections != select_stratified_random_positions(
        170,
        8,
        seed="another-seed",
        example_id="bf-example",
    )
    assert [
        (item["stratum_start"], item["stratum_end_exclusive"]) for item in selections
    ] == [
        (0, 21),
        (21, 42),
        (42, 63),
        (63, 85),
        (85, 106),
        (106, 127),
        (127, 148),
        (148, 170),
    ]
    assert sum(item["projection_weight"] for item in selections) == 170
    assert all(
        item["selection_probability"] == pytest.approx(1 / item["stratum_size"])
        for item in selections
    )

    with pytest.raises(ValueError, match="non-empty strata"):
        select_stratified_random_positions(7, 8, seed="s", example_id="e")


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
        [
            _row(index, prompt=f"p{index}", response="x" * length)
            for index, length in enumerate((8, 40), 1)
        ],
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


def test_run_config_validates_cuda_memory_instrumentation_policy() -> None:
    config = _config()
    config["instrumentation"] = {"cuda_memory_telemetry": True}
    with pytest.raises(ValueError, match="supported only by the top-k runner"):
        validate_run_config(config)
    validate_run_config(config, allow_instrumentation=True)

    assert normalized_instrumentation(config) == {
        "cuda_memory_telemetry": True,
        "cuda_allocator_snapshot_telemetry": False,
    }

    config["instrumentation"]["cuda_allocator_snapshot_telemetry"] = True
    validate_run_config(config, allow_instrumentation=True)
    assert normalized_instrumentation(config) == {
        "cuda_memory_telemetry": True,
        "cuda_allocator_snapshot_telemetry": True,
    }

    config["instrumentation"]["cuda_memory_telemetry"] = False
    with pytest.raises(ValueError, match="requires cuda_memory_telemetry=true"):
        validate_run_config(config, allow_instrumentation=True)

    config["instrumentation"] = {"cuda_memory_telemetry": "yes"}
    with pytest.raises(ValueError, match="cuda_memory_telemetry must be boolean"):
        validate_run_config(config, allow_instrumentation=True)

    config["instrumentation"] = {
        "cuda_memory_telemetry": True,
        "cuda_allocator_snapshot_telemetry": "yes",
    }
    with pytest.raises(
        ValueError, match="cuda_allocator_snapshot_telemetry must be boolean"
    ):
        validate_run_config(config, allow_instrumentation=True)

    config["instrumentation"] = {"cuda_memory_telemetry": True, "typo": False}
    with pytest.raises(ValueError, match="unsupported fields: typo"):
        validate_run_config(config, allow_instrumentation=True)


def test_run_config_requires_allocator_telemetry_for_allocator_headroom() -> None:
    config = _config()
    config["wave_limits"] = {"cuda_headroom_policy": "allocator_dense_joint_v1"}
    with pytest.raises(ValueError, match="requires explicit"):
        validate_run_config(config, allow_instrumentation=True)

    config["wave_limits"]["cuda_headroom_action"] = "warn"
    with pytest.raises(ValueError, match="requires an instrumentation object"):
        validate_run_config(config, allow_instrumentation=True)

    config["instrumentation"] = {
        "cuda_memory_telemetry": True,
        "cuda_allocator_snapshot_telemetry": True,
    }
    with pytest.raises(ValueError, match="cuda_dense_joint_pressure_telemetry=true"):
        validate_run_config(config, allow_instrumentation=True)

    config["instrumentation"]["cuda_dense_joint_pressure_telemetry"] = True
    validate_run_config(config, allow_instrumentation=True)

    config["instrumentation"]["cuda_dense_joint_pressure_telemetry"] = "yes"
    with pytest.raises(
        ValueError, match="cuda_dense_joint_pressure_telemetry must be boolean"
    ):
        validate_run_config(config, allow_instrumentation=True)


def test_run_config_normalizes_and_validates_trace_warmup() -> None:
    assert normalized_trace_warmup(_config()) == {
        "enabled": False,
        "mode": "first_wave_item_full_trace_discard",
        "wave_id_prefixes": [],
    }
    config = _config()
    config["trace_warmup"] = {
        "enabled": True,
        "mode": "first_wave_item_full_trace_discard",
        "wave_id_prefixes": ["wave2c-"],
    }
    validate_run_config(config)
    policy = normalized_trace_warmup(config)
    assert policy["enabled"] is True
    assert trace_warmup_applies(policy, "wave2c-stratified-independent-01-example")
    assert not trace_warmup_applies(policy, "wave1-mixed-final-token")
    assert not trace_warmup_applies(policy, "wave2-progressive-target-window")
    assert not trace_warmup_applies(policy, "wave2b-independent-target-positions")

    config["trace_warmup"]["mode"] = "partial_forward"
    with pytest.raises(ValueError, match=r"trace_warmup\.mode"):
        validate_run_config(config)


def _valid_runtime_trace(*, response_count: int = 8, token_id: int = 77) -> CircuitData:
    return CircuitData(
        df_node=pd.DataFrame({"attribution": [0.1], "activation": [0.2], "layer": [0]}),
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
    with pytest.raises(ValueError, match=r"size mismatch|checksum mismatch"):
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
            "sampling": {
                "seed": "seed-a",
                "design": "one_uniform_position_per_contiguous_response_stratum",
                "sampler": "sha256-rejection-v1",
                "response_token_position": 7,
                "stratum_index": 0,
                "stratum_count": 1,
                "stratum_start": 0,
                "stratum_end_exclusive": 8,
                "stratum_size": 8,
                "selection_probability": 0.125,
                "projection_weight": 8,
            },
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
        "waves": [
            {
                "wave_id": "instrumented",
                "sampling_design": {
                    "design": "one_uniform_position_per_contiguous_response_stratum",
                    "seed": "seed-a",
                    "sampler": "sha256-rejection-v1",
                    "response_token_population_size": 8,
                    "stratum_count": 1,
                    "excluded_reference_example_id": "reference-example",
                },
                "items": [item],
            }
        ],
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("response_token_position", 6, "position does not match"),
        ("stratum_start", 1, "bounds are inconsistent"),
        ("stratum_size", 7, "stratum_size is inconsistent"),
        ("selection_probability", 0.5, "selection_probability is inconsistent"),
        ("projection_weight", 7, "projection_weight is inconsistent"),
    ],
)
def test_runner_rejects_inconsistent_sampling_metadata(
    field: str, value: object, message: str
) -> None:
    item = copy.deepcopy(_single_item_manifest()["waves"][0]["items"][0])
    item["target_selection"]["sampling"][field] = value
    with pytest.raises(ValueError, match=message):
        validate_target_selection(item)


def test_runtime_identity_explicitly_binds_sampling_and_warmup() -> None:
    item = copy.deepcopy(_single_item_manifest()["waves"][0]["items"][0])
    config = _config()
    artifact_id, identity = runtime_artifact_identity(
        item,
        config,
        {},
        {},
        wave_id="instrumented",
        warmup_source_item=None,
    )

    assert artifact_id.startswith("trace-")
    assert identity["source_target_selection"] == item["target_selection"]
    assert identity["trace_warmup"]["enabled"] is False

    config["trace_warmup"] = {
        "enabled": True,
        "mode": "first_wave_item_full_trace_discard",
        "wave_id_prefixes": ["instrumented"],
    }
    _, warm_identity = runtime_artifact_identity(
        item,
        config,
        {},
        {},
        wave_id="instrumented",
        warmup_source_item=item,
    )
    assert warm_identity["sha256"] != identity["sha256"]
    assert warm_identity["trace_warmup"]["source_artifact_id"] == "source-trace-1"

    config["trace_warmup"]["wave_id_prefixes"] = ["wave2c-"]
    _, wave1_identity = runtime_artifact_identity(
        item,
        config,
        {},
        {},
        wave_id="wave1-mixed-final-token",
        warmup_source_item=None,
    )
    assert wave1_identity["trace_warmup"]["applies_to_wave"] is False
    assert "source_artifact_id" not in wave1_identity["trace_warmup"]


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
    summary_record = json.loads(
        (tmp_path / "summary.jsonl").read_text().splitlines()[0]
    )
    assert summary_record["instrumentation"]["schema_version"].endswith("v1")
    assert summary_record["target_sampling"]["stratum_index"] == 0
    assert summary_record["target_sampling"]["selection_probability"] == 0.125
    loaded = load_compact_trace(records[0]["artifact_path"])
    assert loaded.metrics["instrumentation"]["counters"]["probe"] == 5
    assert (
        loaded.circuit_data.trace_metadata["instrumentation"]["counters"]["probe"] == 5
    )
    assert (
        loaded.manifest["source_target_selection"]["sampling"]["projection_weight"] == 8
    )


def test_runner_discards_full_warmup_then_measures_every_planned_item(
    tmp_path: Path, monkeypatch
) -> None:
    import scripts.bonafide.runner as runner_module

    config = _config()
    config["model"]["device"] = "cpu"
    config["trace_warmup"] = {
        "enabled": True,
        "mode": "first_wave_item_full_trace_discard",
        "wave_id_prefixes": ["instrumented"],
    }
    manifest = _single_item_manifest()
    first = manifest["waves"][0]["items"][0]
    first["target_selection"]["response_token_positions"] = [3]
    first["target_selection"]["sampling"].update(
        {
            "response_token_position": 3,
            "stratum_index": 0,
            "stratum_count": 2,
            "stratum_start": 0,
            "stratum_end_exclusive": 4,
            "stratum_size": 4,
            "selection_probability": 0.25,
            "projection_weight": 4,
        }
    )
    manifest["waves"][0]["sampling_design"]["stratum_count"] = 2
    second = copy.deepcopy(manifest["waves"][0]["items"][0])
    second["artifact_id"] = "source-trace-2"
    second["target_selection"]["response_token_positions"] = [7]
    second["target_selection"]["sampling"].update(
        {
            "response_token_position": 7,
            "stratum_index": 1,
            "stratum_start": 4,
            "stratum_end_exclusive": 8,
        }
    )
    manifest["waves"][0]["items"].append(second)
    calls: list[tuple[str, int]] = []

    def fake_trace(**kwargs):
        calls.append((kwargs["label"], kwargs["target_response_positions"][0]))
        trace = _valid_runtime_trace()
        trace.target_provenance[0]["response_token_position"] = kwargs[
            "target_response_positions"
        ][0]
        return trace

    monkeypatch.setattr(
        runner_module, "_load_model_and_tokenizer", lambda _config: (object(), object())
    )
    monkeypatch.setattr(runner_module, "trace_teacher_forced_response", fake_trace)
    records = run_wave(
        config=config,
        manifest=manifest,
        wave_id="instrumented",
        artifact_root=tmp_path / "artifacts",
        summary_jsonl=tmp_path / "summary.jsonl",
    )

    assert calls == [("example-1", 3), ("example-1", 3), ("example-1", 7)]
    assert [record["status"] for record in records] == [
        "warmup_complete",
        "complete",
        "complete",
    ]
    assert records[0]["warmup"]["source_artifact_id"] == "source-trace-1"
    assert records[0]["warmup"]["wall_seconds"] >= 0
    assert records[1]["trace_warmup"]["status"] == "complete"
    summary = [
        json.loads(line)
        for line in (tmp_path / "summary.jsonl").read_text().splitlines()
    ]
    assert summary[0]["record_type"] == "discarded_trace_warmup"
    assert summary[1]["trace_warmup"]["source_artifact_id"] == "source-trace-1"
    loaded = load_compact_trace(records[1]["artifact_path"])
    assert loaded.manifest["trace_warmup"]["status"] == "complete"


def test_filtered_wave_uses_fixed_first_wave_item_as_warmup_source(
    tmp_path: Path, monkeypatch
) -> None:
    import scripts.bonafide.runner as runner_module

    csv_path = tmp_path / "bonafide.csv"
    _write_csv(
        csv_path,
        [
            _row(1, prompt="sampled", response="x" * 8),
            _row(2, prompt="reference", response="x" * 40),
        ],
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
    wave = manifest["waves"][3]
    fixed_source = wave["items"][0]
    selected = wave["items"][1]
    config = _config()
    config["model"]["device"] = "cpu"
    config["trace_warmup"] = {
        "enabled": True,
        "mode": "first_wave_item_full_trace_discard",
        "wave_id_prefixes": ["wave2c-"],
    }
    calls: list[int] = []

    def fake_trace(**kwargs):
        position = kwargs["target_response_positions"][0]
        calls.append(position)
        trace = _valid_runtime_trace(token_id=1120)
        trace.target_provenance[0]["response_token_position"] = position
        return trace

    monkeypatch.setattr(
        runner_module, "_load_model_and_tokenizer", lambda _config: (object(), object())
    )
    monkeypatch.setattr(runner_module, "trace_teacher_forced_response", fake_trace)
    records = run_wave(
        config=config,
        manifest=manifest,
        wave_id=wave["wave_id"],
        only_artifact_id=selected["artifact_id"],
        artifact_root=tmp_path / "artifacts",
        summary_jsonl=tmp_path / "summary.jsonl",
    )

    assert calls == [
        fixed_source["target_selection"]["response_token_positions"][0],
        selected["target_selection"]["response_token_positions"][0],
    ]
    assert records[0]["warmup"]["source_artifact_id"] == fixed_source["artifact_id"]
    identity = load_compact_trace(records[1]["artifact_path"]).manifest[
        "artifact_identity"
    ]
    assert identity["trace_warmup"]["source_artifact_id"] == fixed_source["artifact_id"]
    assert (
        identity["trace_warmup"]["source_target_selection"]
        == fixed_source["target_selection"]
    )


@pytest.mark.parametrize(
    ("failure", "status"),
    [
        (RuntimeError("warmup failed"), "warmup_error"),
        (torch.cuda.OutOfMemoryError("warmup OOM"), "warmup_oom"),
    ],
)
def test_failed_warmup_records_failure_and_writes_no_measured_artifact(
    tmp_path: Path, monkeypatch, failure: Exception, status: str
) -> None:
    import scripts.bonafide.runner as runner_module

    config = _config()
    config["model"]["device"] = "cpu"
    config["trace_warmup"] = {
        "enabled": True,
        "mode": "first_wave_item_full_trace_discard",
        "wave_id_prefixes": ["instrumented"],
    }

    def failing_trace(**kwargs):
        raise failure

    monkeypatch.setattr(
        runner_module, "_load_model_and_tokenizer", lambda _config: (object(), object())
    )
    monkeypatch.setattr(runner_module, "trace_teacher_forced_response", failing_trace)
    artifact_root = tmp_path / "artifacts"
    with pytest.raises(type(failure), match="warmup"):
        run_wave(
            config=config,
            manifest=_single_item_manifest(),
            wave_id="instrumented",
            artifact_root=artifact_root,
            summary_jsonl=tmp_path / "summary.jsonl",
        )

    summary = json.loads((tmp_path / "summary.jsonl").read_text().splitlines()[0])
    assert summary["status"] == status
    assert summary["warmup"]["source_artifact_id"] == "source-trace-1"
    assert not list(artifact_root.rglob("manifest.json"))


def test_warmup_cleanup_failure_does_not_suppress_recorded_original_error(
    tmp_path: Path, monkeypatch
) -> None:
    import scripts.bonafide.runner as runner_module

    config = _config()
    config["model"]["device"] = "cpu"
    config["trace_warmup"] = {
        "enabled": True,
        "mode": "first_wave_item_full_trace_discard",
        "wave_id_prefixes": ["instrumented"],
    }

    def failing_trace(**kwargs):
        raise RuntimeError("original warmup failure")

    def failing_cleanup():
        raise ValueError("cleanup failure")

    monkeypatch.setattr(
        runner_module, "_load_model_and_tokenizer", lambda _config: (object(), object())
    )
    monkeypatch.setattr(runner_module, "trace_teacher_forced_response", failing_trace)
    monkeypatch.setattr(runner_module.gc, "collect", failing_cleanup)
    with pytest.raises(RuntimeError, match="original warmup failure"):
        run_wave(
            config=config,
            manifest=_single_item_manifest(),
            wave_id="instrumented",
            artifact_root=tmp_path / "artifacts",
            summary_jsonl=tmp_path / "summary.jsonl",
        )

    summary = json.loads((tmp_path / "summary.jsonl").read_text().splitlines()[0])
    assert summary["status"] == "warmup_error"
    assert summary["error_type"] == "RuntimeError"
    assert summary["cleanup_error_type"] == "ValueError"


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
    summary_record = json.loads(
        (tmp_path / "summary.jsonl").read_text().splitlines()[0]
    )
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


def test_wave_stop_gate_prefers_validated_headroom_receipt() -> None:
    receipt = assess_cuda_headroom(
        policy="peak_reserved_v1",
        threshold_bytes=4,
        action="warn",
        device_total_bytes=100,
        peak_reserved_bytes=97,
        instrumentation={},
    )
    record = {
        "status": "complete",
        "trace_wall_seconds": 1.0,
        "cuda_headroom_after_peak_bytes": 3,
        "cuda_headroom_gate": receipt,
    }

    assert (
        wave_stop_reason(
            record,
            uses_cuda=True,
            max_trace_seconds=10.0,
            min_cuda_headroom_bytes=4,
            stop_on_oom=True,
            cuda_headroom_policy="peak_reserved_v1",
            cuda_headroom_action="warn",
        )
        is None
    )


def test_wave_stop_allocator_policy_requires_receipt() -> None:
    with pytest.raises(ValueError, match="requires a gate receipt"):
        wave_stop_reason(
            {
                "status": "complete",
                "trace_wall_seconds": 1.0,
                "cuda_headroom_after_peak_bytes": 100,
            },
            uses_cuda=True,
            max_trace_seconds=10.0,
            min_cuda_headroom_bytes=4,
            stop_on_oom=True,
            cuda_headroom_policy="allocator_dense_joint_v1",
            cuda_headroom_action="warn",
        )
