"""CPU-only tests for the graph-free ADAG probe path and JSON artifacts."""

from __future__ import annotations

import json
import inspect
import subprocess
from types import SimpleNamespace

import pytest
import torch

from circuits.tracing.clja import (
    ADAGConfig,
    CLJAProbeSelection,
    get_all_pairs_cl_ja_effects_with_attributions,
    _selected_probe_occurrences,
)
from circuits.tracing.instrumentation import TraceInstrumentation
from circuits.tracing.probe_artifact import (
    METRICS_FILENAME,
    PROBE_FILENAME,
    load_probe_artifact,
    save_probe_artifact,
    validate_probe_artifact_integrity,
)
from circuits.tracing.trace import (
    PROBE_FEATURE_BASIS_SCHEMA_VERSION,
    PROBE_OCCURRENCE_SCHEMA_VERSION,
    PROBE_SCHEMA_VERSION,
    TeacherForcedProbeResult,
    probe_teacher_forced_response,
    validate_teacher_forced_probe_result,
    _stable_json_hash,
)
from tests.test_teacher_forced_trace import FakeChatTokenizer, FakeModel
from tests.test_bonafide_benchmark import _config, _single_item_manifest


def _predictor_without_clock(snapshot: dict) -> dict:
    predictors = dict(snapshot["early_predictors"])
    predictors.pop("early_predictors_ready_seconds")
    return predictors


def test_low_level_probe_matches_full_selection_and_skips_graph_work(monkeypatch) -> None:
    import circuits.tracing.clja as clja_module

    layers, tokens, neurons = 3, 4, 5
    initial_attr = torch.zeros((layers, 1, tokens, neurons))
    initial_attr[0, 0, 1, 2] = 0.25
    initial_attr[2, 0, 3, 4] = -0.75
    mask = torch.zeros((layers, tokens, neurons), dtype=torch.bool)
    mask[0, 1, 2] = True
    mask[2, 3, 4] = True

    monkeypatch.setattr(
        clja_module,
        "_get_grad_attributions_from_logits",
        lambda *_args, **_kwargs: (
            initial_attr.clone(),
            torch.zeros((1, tokens)),
            torch.tensor(1.0),
            torch.zeros((layers, 1, tokens, neurons)),
            torch.zeros((1, tokens)),
        ),
    )
    monkeypatch.setattr(
        clja_module,
        "_get_global_important_neurons_mask",
        lambda **_kwargs: mask.clone(),
    )
    graph_calls = 0

    def fake_graph(*_args, **_kwargs):
        nonlocal graph_calls
        graph_calls += 1
        return [], []

    monkeypatch.setattr(clja_module, "_get_cl_ja_based_edges", fake_graph)
    model = SimpleNamespace(
        config=SimpleNamespace(num_hidden_layers=layers, _name_or_path="fake/model"),
        to=lambda _device: model,
    )
    common = dict(
        model=model,
        tokenizer=FakeChatTokenizer(),
        cis=[[1, 2, 3, 4]],
        config=ADAGConfig(
            device="cpu", disable_stop_grad=True, skip_attr_contrib=True
        ),
        src_tokens=[0, 1, 2],
        tgt_tokens=[2],
        keep_tokens=[0, 1, 2, 3],
        attention_masks=[[1, 1, 1, 1]],
        focus_positions=[2],
        focus_logits=[4],
    )
    probe_instrumentation = TraceInstrumentation(device="cpu")
    selection = get_all_pairs_cl_ja_effects_with_attributions(
        **common, instrumentation=probe_instrumentation, probe_only=True
    )
    assert isinstance(selection, CLJAProbeSelection)
    assert selection.selected_occurrences == [
        {"layer": 0, "token_position": 1, "neuron": 2, "attribution": 0.25},
        {"layer": 2, "token_position": 3, "neuron": 4, "attribution": -0.75},
    ]
    assert graph_calls == 0
    assert "graph_expansion" not in probe_instrumentation.snapshot()["stages"]

    full_instrumentation = TraceInstrumentation(device="cpu")
    assert get_all_pairs_cl_ja_effects_with_attributions(
        **common, instrumentation=full_instrumentation
    ) == ([], [])
    assert graph_calls == 1
    assert _predictor_without_clock(
        probe_instrumentation.snapshot()
    ) == _predictor_without_clock(full_instrumentation.snapshot())


def _install_stop_grad_probe_fakes(monkeypatch, *, attribution_error=None):
    import circuits.tracing.clja as clja_module

    model = SimpleNamespace(
        config=SimpleNamespace(num_hidden_layers=1, _name_or_path="fake/model"),
        wrapped=False,
    )
    model.to = lambda _device: model
    cleanup_states = []

    def fake_revert(value):
        value.wrapped = False
        cleanup_states.append(value.wrapped)
        return value

    def fake_stop(value, **_kwargs):
        value.wrapped = True
        return value

    def fake_attribution(*_args, **_kwargs):
        if attribution_error is not None:
            raise attribution_error
        return (
            torch.tensor([[[[0.0, 0.5], [0.0, 0.0]]]]),
            torch.zeros((1, 2)),
            torch.tensor(1.0),
            torch.zeros((1, 1, 2, 2)),
            torch.zeros((1, 2)),
        )

    mask = torch.zeros((1, 2, 2), dtype=torch.bool)
    mask[0, 0, 1] = True
    monkeypatch.setattr(clja_module, "revert_stop_nonlinear_grad", fake_revert)
    monkeypatch.setattr(clja_module, "stop_nonlinear_grad", fake_stop)
    monkeypatch.setattr(
        clja_module, "_get_grad_attributions_from_logits", fake_attribution
    )
    monkeypatch.setattr(
        clja_module,
        "_get_global_important_neurons_mask",
        lambda **_kwargs: mask,
    )
    kwargs = dict(
        model=model,
        tokenizer=FakeChatTokenizer(),
        cis=[[1, 2]],
        config=ADAGConfig(device="cpu"),
        src_tokens=[0],
        tgt_tokens=[0],
        keep_tokens=[0],
        attention_masks=[[1, 1]],
        focus_positions=[0],
        focus_logits=[2],
        probe_only=True,
    )
    return model, cleanup_states, kwargs


def test_probe_stop_grad_cleanup_runs_on_success(monkeypatch) -> None:
    model, cleanup_states, kwargs = _install_stop_grad_probe_fakes(monkeypatch)
    result = get_all_pairs_cl_ja_effects_with_attributions(**kwargs)
    assert isinstance(result, CLJAProbeSelection)
    assert model.wrapped is False
    # One pre-setup normalization and one guaranteed probe cleanup.
    assert cleanup_states == [False, False]


def test_probe_stop_grad_cleanup_runs_after_attribution_failure(monkeypatch) -> None:
    model, cleanup_states, kwargs = _install_stop_grad_probe_fakes(
        monkeypatch, attribution_error=RuntimeError("attribution failed")
    )
    with pytest.raises(RuntimeError, match="attribution failed"):
        get_all_pairs_cl_ja_effects_with_attributions(**kwargs)
    assert model.wrapped is False
    assert cleanup_states == [False, False]


def test_probe_occurrence_export_is_vectorized() -> None:
    source = inspect.getsource(_selected_probe_occurrences)
    assert ".item(" not in source
    assert source.count(".cpu()") == 1


def test_probe_rejects_return_only_important_neurons_before_setup() -> None:
    model = SimpleNamespace(
        to=lambda _device: pytest.fail("invalid probe reached model setup")
    )
    with pytest.raises(ValueError, match="cannot be enabled together"):
        get_all_pairs_cl_ja_effects_with_attributions(
            model=model,
            tokenizer=FakeChatTokenizer(),
            cis=[[1]],
            config=ADAGConfig(
                device="cpu", return_only_important_neurons=True
            ),
            src_tokens=[0],
            tgt_tokens=[0],
            probe_only=True,
        )


def _fake_selection(recorder: TraceInstrumentation) -> CLJAProbeSelection:
    recorder.set_early_predictors(
        {
            "selected_neuron_count": 3,
            "planned_active_layer_pair_count": 1,
            "candidate_mlp_edge_count": 1,
        }
    )
    return CLJAProbeSelection(
        selected_occurrences=[
            {"layer": 0, "token_position": 1, "neuron": 2, "attribution": 0.25},
            {"layer": 0, "token_position": 3, "neuron": 2, "attribution": 0.125},
            {"layer": 2, "token_position": 3, "neuron": 4, "attribution": -0.75},
        ],
        effective_start_layer=-1,
        effective_end_layer=3,
    )


def test_public_probe_is_single_target_json_and_never_converts_graph(monkeypatch) -> None:
    import circuits.tracing.trace as trace_module

    captured = {}

    def fake_clja(**kwargs):
        captured.update(kwargs)
        return _fake_selection(kwargs["instrumentation"])

    monkeypatch.setattr(
        trace_module, "get_all_pairs_cl_ja_effects_with_attributions", fake_clja
    )
    monkeypatch.setattr(
        trace_module,
        "convert_circuit_to_dataframes",
        lambda *_args, **_kwargs: pytest.fail("probe attempted dataframe conversion"),
    )
    model = FakeModel()
    result = probe_teacher_forced_response(
        model,
        FakeChatTokenizer(),
        "question",
        "abcd",
        [2],
        ADAGConfig(device="cpu"),
        model_revision="exact-test-revision",
    )

    assert result.schema_version == PROBE_SCHEMA_VERSION
    assert captured["probe_only"] is True
    assert captured["focus_logits"] == [[79]]
    assert result.target_provenance["response_token_position"] == 2
    assert result.target_provenance["token_id"] == 79
    assert result.target_provenance["logit"] == 79.0
    assert result.occurrence_signature["occurrence_ids"] == [
        [0, 1, 2],
        [0, 3, 2],
        [2, 3, 4],
    ]
    assert result.feature_basis_signature["feature_ids"] == [[0, 2], [2, 4]]
    assert len(result.occurrence_signature["sha256"]) == 64
    assert result.model_identity["revision"] == "exact-test-revision"
    assert len(result.trace_metadata["input_sha256"]) == 64
    assert len(result.trace_metadata["adag_config_sha256"]) == 64
    assert result.instrumentation["counters"]["probe_graph_work_skipped"] is True
    assert "dataframe_conversion" not in result.instrumentation["stages"]
    json.dumps(result.to_dict(), allow_nan=False)
    validate_teacher_forced_probe_result(result)


def test_public_probe_rejects_multi_target_before_model_work() -> None:
    model = FakeModel()
    with pytest.raises(ValueError, match="exactly one"):
        probe_teacher_forced_response(
            model,
            FakeChatTokenizer(),
            "question",
            "abcd",
            [1, 2],
            ADAGConfig(device="cpu"),
            model_revision="exact-test-revision",
        )
    assert model.forward_calls == 0


def test_public_probe_requires_declared_model_revision_before_model_work() -> None:
    model = FakeModel()
    with pytest.raises(ValueError, match="requires model_revision"):
        probe_teacher_forced_response(
            model,
            FakeChatTokenizer(),
            "question",
            "abcd",
            [2],
            ADAGConfig(device="cpu"),
        )
    assert model.forward_calls == 0


def _valid_probe() -> TeacherForcedProbeResult:
    recorder = TraceInstrumentation(device="cpu")
    selection = _fake_selection(recorder)
    occurrences = selection.selected_occurrences
    occurrence_ids = [
        [entry["layer"], entry["token_position"], entry["neuron"]]
        for entry in occurrences
    ]
    feature_ids = sorted({(entry["layer"], entry["neuron"]) for entry in occurrences})
    feature_ids_json = [[layer, neuron] for layer, neuron in feature_ids]
    adag_config = {"device": "cpu"}
    model_identity = {
        "model_id": "fake/model",
        "revision": "exact-test-revision",
        "model_type": "fake",
        "hash_semantics": "declared_source_revision_and_model_config_v1",
        "model_config_sha256": "b" * 64,
    }
    model_identity["sha256"] = _stable_json_hash(model_identity)

    return TeacherForcedProbeResult(
        target_provenance={
            "response_token_position": 2,
            "absolute_token_position": 6,
            "prediction_token_position": 5,
            "token_id": 79,
            "token_text": "tok-79",
            "logit": 79.0,
            "probability": 0.5,
        },
        selected_occurrences=occurrences,
        occurrence_signature={
            "schema_version": PROBE_OCCURRENCE_SCHEMA_VERSION,
            "ordering": "layer_token_neuron_ascending",
            "occurrence_ids": occurrence_ids,
            "sha256": _stable_json_hash(occurrence_ids),
        },
        feature_basis_signature={
            "schema_version": PROBE_FEATURE_BASIS_SCHEMA_VERSION,
            "ordering": "layer_neuron_ascending_unique",
            "feature_ids": feature_ids_json,
            "sha256": _stable_json_hash(feature_ids_json),
        },
        instrumentation=recorder.snapshot(),
        trace_metadata={
            "trace_mode": "teacher_forced_probe",
            "prompt_sha256": "1" * 64,
            "response_sha256": "2" * 64,
            "system_prompt_sha256": None,
            "text_bundle_sha256": "3" * 64,
            "input_sha256": "4" * 64,
            "adag_config_sha256": _stable_json_hash(adag_config),
            "chat_template_sha256": "5" * 64,
            "assistant_prefix_token_count": 4,
            "response_token_count": 4,
            "included_response_token_count": 3,
            "input_token_count": 7,
            "effective_start_layer": -1,
            "effective_end_layer": 3,
        },
        model_identity=model_identity,
        adag_config=adag_config,
    )


def test_probe_artifact_roundtrip_and_corruption_detection(tmp_path) -> None:
    destination = tmp_path / "probe-unit"
    save_probe_artifact(
        destination,
        _valid_probe(),
        metrics={"probe_wall_seconds": 0.5},
        manifest={"artifact_id": "probe-1"},
    )
    manifest = validate_probe_artifact_integrity(destination)
    assert manifest["artifact_id"] == "probe-1"
    loaded = load_probe_artifact(destination)
    assert loaded.probe["schema_version"] == PROBE_SCHEMA_VERSION
    assert loaded.metrics["probe_wall_seconds"] == 0.5

    with (destination / PROBE_FILENAME).open("a", encoding="utf-8") as handle:
        handle.write(" ")
    with pytest.raises(ValueError, match="size mismatch|checksum mismatch"):
        validate_probe_artifact_integrity(destination)


def test_probe_artifact_detects_metrics_corruption(tmp_path) -> None:
    destination = tmp_path / "probe-unit"
    save_probe_artifact(
        destination,
        _valid_probe(),
        metrics={"probe_wall_seconds": 0.5},
    )
    with (destination / METRICS_FILENAME).open("a", encoding="utf-8") as handle:
        handle.write(" ")
    with pytest.raises(ValueError, match="metrics size mismatch|metrics checksum mismatch"):
        validate_probe_artifact_integrity(destination)


def test_probe_validation_rejects_basis_or_post_selection_stage() -> None:
    probe = _valid_probe().to_dict()
    probe["occurrence_signature"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_teacher_forced_probe_result(probe)

    probe = _valid_probe().to_dict()
    probe["feature_basis_signature"]["feature_ids"] = [[0, 2]]
    with pytest.raises(ValueError, match="feature basis disagrees"):
        validate_teacher_forced_probe_result(probe)

    probe = _valid_probe().to_dict()
    probe["instrumentation"]["stages"]["graph_expansion"] = {
        "wall_seconds": 1.0,
        "calls": 1,
        "failed_calls": 0,
    }
    with pytest.raises(ValueError, match="forbidden post-selection"):
        validate_teacher_forced_probe_result(probe)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda probe: probe["target_provenance"].__setitem__(
                "probability", 1.01
            ),
            "probability must be in",
        ),
        (
            lambda probe: probe["target_provenance"].__setitem__(
                "response_token_position", True
            ),
            "non-negative integer",
        ),
        (
            lambda probe: probe["trace_metadata"].__setitem__(
                "input_sha256", "not-a-hash"
            ),
            "lowercase SHA-256",
        ),
        (
            lambda probe: probe["trace_metadata"].__setitem__(
                "included_response_token_count", 4
            ),
            "response-position metadata",
        ),
        (
            lambda probe: probe["adag_config"].__setitem__("device", "cuda:0"),
            "ADAG config checksum",
        ),
        (
            lambda probe: probe["model_identity"].__setitem__("revision", ""),
            "model_identity.revision",
        ),
    ],
)
def test_probe_validation_rejects_invalid_promised_metadata(mutation, message) -> None:
    probe = _valid_probe().to_dict()
    mutation(probe)
    with pytest.raises(ValueError, match=message):
        validate_teacher_forced_probe_result(probe)


def test_probe_wave_dry_run_uses_distinct_identity_without_loading_model(
    tmp_path, monkeypatch
) -> None:
    import scripts.bonafide.probe_runner as probe_runner

    monkeypatch.setattr(
        probe_runner,
        "_load_model_and_tokenizer",
        lambda _config: pytest.fail("dry-run loaded a model"),
    )
    records = probe_runner.run_probe_wave(
        config=_config(),
        manifest=_single_item_manifest(),
        wave_id="instrumented",
        artifact_root=tmp_path / "probes",
        summary_jsonl=tmp_path / "summary.jsonl",
        dry_run=True,
    )
    assert records[0]["status"] == "planned"
    assert records[0]["artifact_id"].startswith("probe-")
    assert records[0]["mode"] == "teacher_forced_probe"
    assert not (tmp_path / "summary.jsonl").exists()


def test_probe_runner_binds_revision_and_preserves_error_provenance(
    tmp_path, monkeypatch
) -> None:
    import scripts.bonafide.probe_runner as probe_runner

    config = _config()
    config["model"]["device"] = "cpu"
    config["continue_on_error"] = True
    captured = {}

    monkeypatch.setattr(
        probe_runner, "_load_model_and_tokenizer", lambda _config: (object(), object())
    )

    def failing_probe(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("probe failed")

    monkeypatch.setattr(probe_runner, "probe_teacher_forced_response", failing_probe)
    records = probe_runner.run_probe_wave(
        config=config,
        manifest=_single_item_manifest(),
        wave_id="instrumented",
        artifact_root=tmp_path / "probes",
        summary_jsonl=tmp_path / "summary.jsonl",
    )

    record = records[0]
    assert captured["model_revision"] == config["model"]["revision"]
    assert record["status"] == "error"
    assert record["artifact_identity_sha256"] == record["artifact_identity"]["sha256"]
    assert record["source_target_selection"] == _single_item_manifest()["waves"][0][
        "items"
    ][0]["target_selection"]
    assert record["target_response_positions"] == [7]
    assert record["model_revision"] == config["model"]["revision"]
    summary = json.loads((tmp_path / "summary.jsonl").read_text().splitlines()[0])
    assert summary["artifact_identity"] == record["artifact_identity"]


def test_slurm_launcher_defaults_to_trace_and_dispatches_probe() -> None:
    script = (
        __import__("pathlib").Path(__file__).parents[1]
        / "scripts"
        / "bonafide"
        / "benchmark_tracing.sbatch"
    )
    subprocess.run(["bash", "-n", str(script)], check=True)
    source = script.read_text()
    assert 'EXECUTION_MODE="${EXECUTION_MODE:-trace}"' in source
    assert 'RUNNER_MODULE="scripts.bonafide.runner"' in source
    assert 'RUNNER_MODULE="scripts.bonafide.probe_runner"' in source
    assert '"$PYTHON_BIN" -m "$RUNNER_MODULE"' in source
