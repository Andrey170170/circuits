from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest
from circuits.analysis.bonafide.canonical import canonical_sha256
from circuits.analysis.bonafide.process_witness_resource_calibration_v1 import (
    build_resource_calibration_v1,
    load_frozen_resource_calibration_v1,
)

SYSTEM_PROMPT = "system"
MECHANISMS = (
    "process_enrichment",
    "evaluation_commitment",
    "diversity",
    "uncertainty_missing",
    "uniform_reserve",
)


class FakeThinkingTokenizer:
    name_or_path = "fake-thinking"
    chat_template = "fake-thinking-template"

    def apply_chat_template(
        self,
        messages,
        *,
        add_generation_prompt: bool,
        tokenize: bool = False,
        enable_thinking: bool = True,
        chat_template: str | None = None,
    ):
        assert add_generation_prompt is True
        assert tokenize is False
        assert enable_thinking is True
        assert chat_template == self.chat_template
        prompt = messages[-1]["content"]
        return f"PREFIX:{prompt}:"

    def __call__(self, text: str, **_kwargs):
        return {"input_ids": [ord(char) for char in text]}


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _write_gzip_jsonl(path: Path, rows: list[dict]) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def _source_fixture(tmp_path: Path) -> tuple[Path, FakeThinkingTokenizer]:
    root = tmp_path / "sampling-v2"
    parent = tmp_path / "parent-v1"
    root.mkdir()
    tokenizer = FakeThinkingTokenizer()
    requested = (40, 50, 60, 70, 80)
    documents = []
    context_rows = []
    candidates = []
    for index, context in enumerate(requested):
        response_id = f"response-{index}"
        prompt = f"prompt-{index}"
        response = "x" * 120
        prefix_text = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=True,
            chat_template=tokenizer.chat_template,
        )
        prefix_ids = tokenizer(prefix_text)["input_ids"]
        response_ids = tokenizer(response)["input_ids"]
        token_index = context - len(prefix_ids) - 1
        assert token_index >= 0
        documents.append(
            {
                "response_id": response_id,
                "task_context": {"prompt": prompt},
                "text": response,
                "tokenization": {
                    "token_count": len(response_ids),
                    "input_ids_sha256": canonical_sha256(response_ids),
                    "tokens": [
                        [token_id, position, position + 1]
                        for position, token_id in enumerate(response_ids)
                    ],
                },
            }
        )
        context_rows.append(
            {
                "response_id": response_id,
                "assistant_prefix_token_count": len(prefix_ids),
                "assistant_prefix_ids_sha256": canonical_sha256(prefix_ids),
                "assistant_prefix_token_ids": prefix_ids,
                "source_kind": "generation_reproducibility_prompt_token_ids",
            }
        )
        target_id = f"target-{index}"
        base = {
            "target_id": target_id,
            "response_id": response_id,
            "psu_id": f"psu-{index}",
            "unit_id": f"unit-{index}",
            "token_index": token_index,
            "rendered_total_context_token_count": context,
            "arrival_mechanisms": ["uniform_reserve"],
            "nominal_expected_unique_target_budget": 40_000,
            "policy": "balanced",
            "first_owner_mechanism": MECHANISMS[index],
        }
        candidates.append(base)
    _write_json(
        root / "manifest.json",
        {
            "schema_version": "adag.process-witness.coarse-post-campaign-sampling.v2",
            "manifest_sha256": "5d2a49a14123ed819ab404c3da8b4633eab55d8e30cf6996c7e9544c3bfc7089",
            "inventory_sha256": "d6ded745d84c2b59129f32beefed5ea2ba7e31c9d485eaa5bea8c4ebebf5e94c",
            "parent_v1_root": str(parent),
        },
    )
    _write_json(
        root / "context-source-binding.json",
        {
            "literal_census": {"responses": 3, "frame_positions": 45},
            "step0_model_id": "Qwen/Qwen3-4B-Thinking-2507",
            "step0_model_revision": "768f209d9ea81521153ed38c47d515654e938aea",
        },
    )
    _write_gzip_jsonl(root / "context-count-evidence.jsonl.gz", context_rows)
    _write_gzip_jsonl(root / "realized-candidate-tiers.jsonl.gz", candidates)
    _write_json(
        parent / "source-evidence/bundle/workstation-bundle.json",
        {"documents": documents},
    )
    return root, tokenizer


def test_builder_freezes_deterministic_label_blind_actual_context_ladder(
    tmp_path: Path, monkeypatch
) -> None:
    root, tokenizer = _source_fixture(tmp_path)
    destination = tmp_path / "calibration-v1"
    monkeypatch.setattr(
        "circuits.analysis.bonafide.process_witness_resource_calibration_v1.load_frozen_post_campaign_sampling_v2",
        lambda _root: {"manifest": json.loads((_root / "manifest.json").read_text())},
    )

    manifest = build_resource_calibration_v1(
        sampling_v2_root=root,
        destination=destination,
        tokenizer=tokenizer,
        system_prompt=SYSTEM_PROMPT,
        context_bins=(("test-context-bin", 35, 85),),
        expected_sampling_manifest_sha256="5d2a49a14123ed819ab404c3da8b4633eab55d8e30cf6996c7e9544c3bfc7089",
        expected_sampling_inventory_sha256="d6ded745d84c2b59129f32beefed5ea2ba7e31c9d485eaa5bea8c4ebebf5e94c",
        expected_generation_census=None,
    )

    assert manifest["selected_contexts_by_wave"] == {
        "test-context-bin": [40, 50, 60, 70, 80]
    }
    trace_manifest = json.loads((destination / "trace-manifest.json").read_text())
    items = trace_manifest["waves"][0]["items"]
    assert [
        item["resource_calibration"]["actual_context_token_count"] for item in items
    ] == [40, 50, 60, 70, 80]
    assert len({item["example"]["example_id"] for item in items}) == 5
    assert all("label_types" not in item["example"] for item in items)
    assert all("coarse_label" not in item["resource_calibration"] for item in items)
    assert items[0]["resource_calibration"]["policy_memberships"] == [
        {"policy": "balanced", "budget": 40_000},
    ]
    source_manifest = json.loads(
        (destination / "width1-source-manifest.json").read_text()
    )
    assert (
        hashlib.sha256(
            (destination / "width1-source-manifest.json").read_bytes()
        ).hexdigest()
        == trace_manifest["source"]["width1_manifest_sha256"]
    )
    assert source_manifest["waves"] == trace_manifest["waves"]

    loaded = load_frozen_resource_calibration_v1(
        destination, tokenizer=tokenizer, system_prompt=SYSTEM_PROMPT
    )
    assert loaded["manifest"] == manifest


def test_loader_rejects_frozen_target_token_drift(tmp_path: Path, monkeypatch) -> None:
    root, tokenizer = _source_fixture(tmp_path)
    destination = tmp_path / "calibration-v1"
    monkeypatch.setattr(
        "circuits.analysis.bonafide.process_witness_resource_calibration_v1.load_frozen_post_campaign_sampling_v2",
        lambda _root: {"manifest": json.loads((_root / "manifest.json").read_text())},
    )
    build_resource_calibration_v1(
        sampling_v2_root=root,
        destination=destination,
        tokenizer=tokenizer,
        system_prompt=SYSTEM_PROMPT,
        context_bins=(("test-context-bin", 35, 85),),
        expected_sampling_manifest_sha256="5d2a49a14123ed819ab404c3da8b4633eab55d8e30cf6996c7e9544c3bfc7089",
        expected_sampling_inventory_sha256="d6ded745d84c2b59129f32beefed5ea2ba7e31c9d485eaa5bea8c4ebebf5e94c",
        expected_generation_census=None,
    )
    trace_path = destination / "trace-manifest.json"
    trace_path.chmod(0o644)
    trace = json.loads(trace_path.read_text())
    trace["waves"][0]["items"][0]["target_selection"]["final_target_token_id"] += 1
    _write_json(trace_path, trace)

    with pytest.raises(ValueError, match="drift"):
        load_frozen_resource_calibration_v1(
            destination, tokenizer=tokenizer, system_prompt=SYSTEM_PROMPT
        )
