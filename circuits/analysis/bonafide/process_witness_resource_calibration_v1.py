"""Freeze and validate a label-blind strict-T5 resource calibration ladder."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import subprocess
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts.bonafide.build_process_witness_annotations import SYSTEM_PROMPT

from circuits.analysis.bonafide.canonical import canonical_sha256, file_sha256
from circuits.analysis.bonafide.coarse_sampling_post_campaign_v1 import (
    _publish_no_replace,
)
from circuits.analysis.bonafide.coarse_sampling_post_campaign_v2 import (
    load_frozen_post_campaign_sampling_v2,
)
from circuits.tracing.trace import (
    get_chat_template,
    tokenize_historical_thinking_continuation,
)

SCHEMA_VERSION = "adag.process-witness.resource-calibration.v1"
INVENTORY_SCHEMA_VERSION = "adag.process-witness.resource-calibration-inventory.v1"
TRACE_PHASE = "process_witness_resource_calibration_v1"
TRACE_WAVE_ID = "process-witness-resource-calibration-v1"
MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
MODEL_REVISION = "768f209d9ea81521153ed38c47d515654e938aea"
EXPECTED_SAMPLING_MANIFEST_SHA256 = (
    "5d2a49a14123ed819ab404c3da8b4633eab55d8e30cf6996c7e9544c3bfc7089"
)
EXPECTED_SAMPLING_INVENTORY_SHA256 = (
    "d6ded745d84c2b59129f32beefed5ea2ba7e31c9d485eaa5bea8c4ebebf5e94c"
)
DEFAULT_CONTEXT_BINS = (
    ("context-le-1268", 0, 1268),
    ("context-1269-2500", 1269, 2500),
    ("context-2501-4000", 2501, 4000),
    ("context-4001-6000", 4001, 6000),
    ("context-6001-8000", 6001, 8000),
    ("context-gt-8000", 8001, 10767),
)
CONTEXT_QUANTILES = (0.1, 0.3, 0.5, 0.7, 0.9)
CALIBRATION_MECHANISMS = (
    "process_enrichment",
    "evaluation_commitment",
    "diversity",
    "uncertainty_missing",
    "uniform_reserve",
)
CLAIM_BOUNDARY = (
    "This label-blind method-development ladder measures strict T5 tracing "
    "runtime and resource feasibility only. It does not select a sampling "
    "policy or trace corpus, inspect graph semantics, test ADAG adequacy, "
    "identify a motif or witness, or support a faithfulness claim."
)
EXECUTION_SOURCE_PATHS = (
    "circuits/analysis/bonafide/process_witness_resource_calibration_v1.py",
    "circuits/tracing/artifact.py",
    "circuits/tracing/clja.py",
    "circuits/tracing/trace.py",
    "scripts/bonafide/build_process_witness_resource_calibration_v1.py",
    "scripts/bonafide/process_witness_resource_calibration_v1.sbatch",
    "scripts/bonafide/runner.py",
    "scripts/bonafide/topk_manifest.py",
    "scripts/bonafide/topk_runner.py",
    "pyproject.toml",
    "uv.lock",
)


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _iter_gzip_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"expected JSONL object: {path}")
                yield value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            dict(value),
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    dict(row),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _artifact_id(target_id: str, context: int) -> str:
    digest = hashlib.sha256(f"{target_id}|{context}".encode()).hexdigest()
    return f"trace-source-{digest[:24]}"


def _response_tokens(document: Mapping[str, Any]) -> list[int]:
    tokenization = document.get("tokenization")
    if not isinstance(tokenization, Mapping):
        raise ValueError("workstation document lacks tokenization")
    tokens = tokenization.get("tokens")
    if not isinstance(tokens, list) or not tokens:
        raise ValueError("workstation document has no response token stream")
    ids: list[int] = []
    for token in tokens:
        if (
            not isinstance(token, list)
            or len(token) < 1
            or isinstance(token[0], bool)
            or not isinstance(token[0], int)
        ):
            raise ValueError("workstation response token stream drift")
        ids.append(int(token[0]))
    if tokenization.get("token_count") != len(ids):
        raise ValueError("workstation response token census drift")
    if tokenization.get("input_ids_sha256") != canonical_sha256(ids):
        raise ValueError("workstation response token hash drift")
    return ids


def _exact_runtime_tokenization(
    *,
    tokenizer: Any,
    system_prompt: str,
    document: Mapping[str, Any],
    context_evidence: Mapping[str, Any],
) -> tuple[list[int], list[int]] | None:
    task_context = document.get("task_context")
    if not isinstance(task_context, Mapping):
        raise ValueError("workstation document lacks task context")
    prompt = task_context.get("prompt")
    response = document.get("text")
    if (
        not isinstance(prompt, str)
        or not prompt
        or not isinstance(response, str)
        or not response
    ):
        raise ValueError("workstation prompt/response drift")
    frozen_response_ids = _response_tokens(document)
    tokenized = tokenize_historical_thinking_continuation(
        tokenizer,
        prompt,
        response,
        system_prompt=system_prompt,
    )
    prefix_ids = tokenized.assistant_prefix_ids
    response_ids = tokenized.response_ids
    if (
        len(prefix_ids) != context_evidence.get("assistant_prefix_token_count")
        or canonical_sha256(prefix_ids)
        != context_evidence.get("assistant_prefix_ids_sha256")
        or response_ids != frozen_response_ids
    ):
        return None
    embedded = context_evidence.get("assistant_prefix_token_ids")
    if embedded is not None and list(map(int, embedded)) != prefix_ids:
        return None
    return prefix_ids, response_ids


def _candidate_union(
    rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_target: dict[str, dict[str, Any]] = {}
    memberships: dict[str, set[tuple[str, int]]] = defaultdict(set)
    for raw in rows:
        row = dict(raw)
        target_id = str(row.get("target_id", ""))
        if not target_id:
            raise ValueError("realized candidate target identity drift")
        raw_budget = row.get("nominal_expected_unique_target_budget")
        if isinstance(raw_budget, bool) or not isinstance(raw_budget, int):
            raise ValueError("realized candidate budget membership drift")
        membership = (str(row.get("policy", "")), raw_budget)
        if not membership[0]:
            raise ValueError("realized candidate policy membership drift")
        comparable = {
            field: row.get(field)
            for field in (
                "target_id",
                "response_id",
                "psu_id",
                "unit_id",
                "token_index",
                "rendered_total_context_token_count",
            )
        }
        previous = by_target.setdefault(target_id, comparable)
        if any(previous.get(field) != value for field, value in comparable.items()):
            raise ValueError("realized candidate identity drift across tiers")
        memberships[target_id].add(membership)
        if membership == ("balanced", 40_000):
            owner = row.get("first_owner_mechanism")
            if owner not in CALIBRATION_MECHANISMS:
                raise ValueError("balanced/40k candidate owner mechanism drift")
            existing_owner = previous.get("balanced_40k_first_owner_mechanism")
            if existing_owner is not None and existing_owner != owner:
                raise ValueError("balanced/40k target owner mechanism drift")
            previous["balanced_40k_first_owner_mechanism"] = owner
    result = []
    for target_id, row in by_target.items():
        result.append(
            {
                **row,
                "policy_memberships": [
                    {"policy": policy, "budget": budget}
                    for policy, budget in sorted(memberships[target_id])
                ],
            }
        )
    return result


def _select_waves(
    *,
    candidates: Sequence[Mapping[str, Any]],
    exact_response_ids: set[str],
    prompt_block_by_response: Mapping[str, str],
    context_bins: Sequence[tuple[str, int, int]],
) -> list[dict[str, Any]]:
    waves: list[dict[str, Any]] = []
    used_responses: set[str] = set()
    used_prompt_blocks: set[str] = set()
    for bin_index, (wave_id, lower, upper) in enumerate(context_bins):
        if not wave_id or lower < 0 or upper < lower:
            raise ValueError("calibration context-bin contract drift")
        wave_selected: list[dict[str, Any]] = []
        wave_prompt_blocks: set[str] = set()
        for mechanism_index, mechanism in enumerate(CALIBRATION_MECHANISMS):
            quantile_index = (mechanism_index + bin_index) % len(CONTEXT_QUANTILES)
            quantile = CONTEXT_QUANTILES[quantile_index]
            anchor = round(lower + quantile * (upper - lower))
            base = [
                row
                for row in candidates
                if str(row["response_id"]) in exact_response_ids
                and str(row["response_id"]) not in used_responses
                and row.get("balanced_40k_first_owner_mechanism") == mechanism
                and lower <= int(row["rendered_total_context_token_count"]) <= upper
            ]
            if not base:
                raise ValueError(
                    f"no exact-token balanced/40k {mechanism} candidate in {wave_id}"
                )
            distinct_global = [
                row
                for row in base
                if prompt_block_by_response[str(row["response_id"])]
                not in used_prompt_blocks
            ]
            distinct_wave = [
                row
                for row in base
                if prompt_block_by_response[str(row["response_id"])]
                not in wave_prompt_blocks
            ]
            pool = distinct_global or distinct_wave or base
            chosen = min(
                pool,
                key=lambda row: (
                    abs(int(row["rendered_total_context_token_count"]) - anchor),
                    hashlib.sha256(
                        f"{wave_id}|{mechanism}|{anchor}|{row['target_id']}".encode()
                    ).hexdigest(),
                ),
            )
            value = dict(chosen)
            value.update(
                {
                    "wave_id": wave_id,
                    "context_bin_lower": lower,
                    "context_bin_upper": upper,
                    "assigned_mechanism": mechanism,
                    "assigned_quantile": quantile,
                    "requested_context_token_count": anchor,
                    "prompt_block_sha256": prompt_block_by_response[
                        str(chosen["response_id"])
                    ],
                }
            )
            wave_selected.append(value)
            response_id = str(value["response_id"])
            prompt_block = prompt_block_by_response[response_id]
            used_responses.add(response_id)
            used_prompt_blocks.add(prompt_block)
            wave_prompt_blocks.add(prompt_block)
        wave_selected.sort(
            key=lambda row: (
                int(row["rendered_total_context_token_count"]),
                str(row["target_id"]),
            )
        )
        waves.append(
            {
                "wave_id": wave_id,
                "bin_index": bin_index,
                "lower": lower,
                "upper": upper,
                "selected": wave_selected,
                "distinct_prompt_block_count": len(wave_prompt_blocks),
            }
        )
    return waves


def _run_config() -> dict[str, Any]:
    return {
        "schema_version": "bonafide-trace-run-config/v1",
        "artifact_root": "results/bonafide/process-witness-resource-calibration-v1",
        "batch_size": 1,
        "continue_on_error": False,
        "trace_warmup": {
            "enabled": False,
            "mode": "first_wave_item_full_trace_discard",
            "wave_id_prefixes": [],
        },
        "wave_limits": {
            "max_trace_seconds": 1800,
            "min_cuda_headroom_bytes": 8589934592,
            "stop_on_oom": True,
        },
        "model": {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "device": "cuda:0",
            "dtype": "bfloat16",
            "local_files_only": True,
            "local_snapshot_path": (
                "${HF_HUB_CACHE}/models--Qwen--Qwen3-4B-Thinking-2507/"
                f"snapshots/{MODEL_REVISION}"
            ),
            "from_pretrained_kwargs": {},
        },
        "adag_config": {
            "verbose": False,
            "parent_threshold": None,
            "edge_threshold": 0.01,
            "node_attribution_threshold": None,
            "topk": None,
            "batch_aggregation": "any",
            "topk_neurons": None,
            "percentage_threshold": 0.005,
            "use_relp_grad": True,
            "disable_half_rule": False,
            "disable_stop_grad": False,
            "ablation_mode": "zero",
            "use_stop_grad_on_mlps": True,
            "return_nodes_only": False,
            "focus_last_residual": False,
            "skip_attr_contrib": False,
            "center_logits": False,
            "ig_steps": None,
            "ig_mode": "ig-inputs",
            "return_only_important_neurons": False,
            "apply_blacklist": True,
        },
    }


def _inventory(root: Path) -> dict[str, Any]:
    files = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in {"manifest.json", "inventory.json"}
    ]
    core = {"schema_version": INVENTORY_SCHEMA_VERSION, "files": files}
    return {**core, "inventory_sha256": canonical_sha256(core)}


def _readonly_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def _execution_source_revision(repo_root: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    files = []
    for relative in EXECUTION_SOURCE_PATHS:
        path = repo_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"calibration execution source is absent: {path}")
        files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return {
        "repo_root": str(repo_root.resolve()),
        "git_commit": git("rev-parse", "HEAD"),
        "git_tree": git("rev-parse", "HEAD^{tree}"),
        "binding_scope": "strict_t5_calibration_builder_loader_and_runtime_closure",
        "files": files,
    }


def build_resource_calibration_v1(
    *,
    sampling_v2_root: Path,
    destination: Path,
    tokenizer: Any,
    system_prompt: str = SYSTEM_PROMPT,
    context_bins: Sequence[tuple[str, int, int]] = DEFAULT_CONTEXT_BINS,
    expected_sampling_manifest_sha256: str = EXPECTED_SAMPLING_MANIFEST_SHA256,
    expected_sampling_inventory_sha256: str = EXPECTED_SAMPLING_INVENTORY_SHA256,
    expected_generation_census: tuple[int, int, int] | None = (186, 177, 9),
) -> dict[str, Any]:
    """Build one immutable, deterministic, label-blind actual-candidate ladder."""

    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"resource calibration destination exists: {destination}")
    source = sampling_v2_root.resolve()
    validated = load_frozen_post_campaign_sampling_v2(source)
    source_manifest = validated["manifest"]
    if (
        source_manifest.get("manifest_sha256") != expected_sampling_manifest_sha256
        or source_manifest.get("inventory_sha256") != expected_sampling_inventory_sha256
    ):
        raise ValueError("sampling-v2 manifest/inventory source drift")
    bins = tuple(
        (str(name), int(lower), int(upper)) for name, lower, upper in context_bins
    )
    if not bins:
        raise ValueError("resource calibration requires context bins")

    parent_root = Path(str(source_manifest["parent_v1_root"])).resolve()
    workstation = _load_object(
        parent_root / "source-evidence/bundle/workstation-bundle.json"
    )
    documents = {str(row["response_id"]): row for row in workstation["documents"]}
    context_rows = {
        str(row["response_id"]): row
        for row in _iter_gzip_jsonl(source / "context-count-evidence.jsonl.gz")
    }
    if set(documents) != set(context_rows):
        raise ValueError("sampling-v2 context/workstation response census drift")
    exact_tokenizations: dict[str, tuple[list[int], list[int]]] = {}
    excluded_response_ids = []
    non_generation_response_ids = []
    for response_id in sorted(documents):
        if context_rows[response_id].get("source_kind") != (
            "generation_reproducibility_prompt_token_ids"
        ):
            non_generation_response_ids.append(response_id)
            continue
        tokenized = _exact_runtime_tokenization(
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            document=documents[response_id],
            context_evidence=context_rows[response_id],
        )
        if tokenized is None:
            excluded_response_ids.append(response_id)
        else:
            exact_tokenizations[response_id] = tokenized
    if expected_generation_census is not None:
        observed_generation_census = (
            len(documents) - len(non_generation_response_ids),
            len(exact_tokenizations),
            len(excluded_response_ids),
        )
        if observed_generation_census != expected_generation_census:
            raise ValueError(
                "runtime exact-generation tokenization census drift: "
                f"{observed_generation_census!r} != {expected_generation_census!r}"
            )
        if len(non_generation_response_ids) != 2:
            raise ValueError("runtime non-generation response census drift")

    candidates = _candidate_union(
        _iter_gzip_jsonl(source / "realized-candidate-tiers.jsonl.gz")
    )
    prompt_block_by_response = {
        response_id: _hash_text(str(document["task_context"]["prompt"]))
        for response_id, document in documents.items()
    }
    selected_waves = _select_waves(
        candidates=candidates,
        exact_response_ids=set(exact_tokenizations),
        prompt_block_by_response=prompt_block_by_response,
        context_bins=bins,
    )
    chat_template = get_chat_template(tokenizer)
    if not isinstance(chat_template, str) or not chat_template:
        raise ValueError("Thinking tokenizer chat template is missing")
    chat_template_sha256 = _hash_text(chat_template)
    system_prompt_sha256 = _hash_text(system_prompt)
    trace_waves = []
    selection_rows = []
    for wave in selected_waves:
        items = []
        for ordinal, row in enumerate(wave["selected"]):
            response_id = str(row["response_id"])
            document = documents[response_id]
            prefix_ids, response_ids = exact_tokenizations[response_id]
            token_index = int(row["token_index"])
            context_count = int(row["rendered_total_context_token_count"])
            if (
                not 0 <= token_index < len(response_ids)
                or context_count != len(prefix_ids) + token_index + 1
            ):
                raise ValueError("selected target token/context identity drift")
            target_id = str(row["target_id"])
            item = {
                "artifact_id": _artifact_id(target_id, context_count),
                "example": {
                    "example_id": response_id,
                    "annotation_row_ids": [],
                    "prompt": document["task_context"]["prompt"],
                    "response": document["text"],
                    "system_prompt": system_prompt,
                    "historical_replay_scope": "stored_assistant_serialization",
                    "teacher_forced_serialization_mode": (
                        "historical_thinking_continuation"
                    ),
                    "token_identity": {
                        "schema_version": "adag.teacher-forced-token-identity.v1",
                        "hash_encoding": (
                            "sha256_utf8_canonical_json_integer_array_v1"
                        ),
                        "assistant_prefix_ids_sha256": canonical_sha256(prefix_ids),
                        "response_ids_sha256": canonical_sha256(response_ids),
                        "assistant_prefix_token_count": len(prefix_ids),
                        "response_token_count": len(response_ids),
                    },
                },
                "response_token_count": len(response_ids),
                "target_selection": {
                    "kind": "explicit_response_positions",
                    "response_token_positions": [token_index],
                    "width": 1,
                    "final_target_token_id": response_ids[token_index],
                },
                "objective": {
                    "name": "sum_selected_logits",
                    "benchmark_only_multi_target": False,
                },
                "resource_calibration": {
                    "wave_ordinal": ordinal,
                    "selection_basis": (
                        "label_blind_balanced_40k_first_owner_rotated_context_quantile"
                    ),
                    "requested_context_token_count": int(
                        row["requested_context_token_count"]
                    ),
                    "actual_context_token_count": context_count,
                    "target_id": target_id,
                    "psu_id": row["psu_id"],
                    "unit_id": row["unit_id"],
                    "assigned_first_owner_mechanism": row["assigned_mechanism"],
                    "assigned_quantile": row["assigned_quantile"],
                    "context_bin": {
                        "lower_inclusive": row["context_bin_lower"],
                        "upper_inclusive": row["context_bin_upper"],
                    },
                    "prompt_block_sha256": row["prompt_block_sha256"],
                    "policy_memberships": row["policy_memberships"],
                },
            }
            items.append(item)
            selection_rows.append(
                {
                    "wave_id": wave["wave_id"],
                    **item["resource_calibration"],
                }
            )
        trace_waves.append(
            {
                "wave_id": wave["wave_id"],
                "corpus_role": "resource_method_development",
                "purpose": "increasing-context strict-T5 resource calibration only",
                "resource_context_bin": {
                    "lower_inclusive": wave["lower"],
                    "upper_inclusive": wave["upper"],
                    "distinct_prompt_block_count": wave["distinct_prompt_block_count"],
                },
                "items": items,
            }
        )
    width1_manifest = {
        "schema_version": "bonafide-trace-benchmark/v1",
        "dataset": {
            "path": str(source),
            "sha256": expected_sampling_manifest_sha256,
        },
        "tokenizer": {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "chat_template_sha256": chat_template_sha256,
        },
        "execution_contract": {
            "trace_units_are_independent": True,
            "merge_graphs": False,
            "historical_replay_scope": "stored_assistant_serialization",
        },
        "waves": trace_waves,
    }

    temporary = destination.parent / f".{destination.name}.building-{os.getpid()}"
    if temporary.exists():
        raise FileExistsError(f"resource calibration staging root exists: {temporary}")
    temporary.mkdir(parents=True)
    try:
        execution_source_revision = _execution_source_revision(
            Path(__file__).resolve().parents[3]
        )
        _write_json(temporary / "width1-source-manifest.json", width1_manifest)
        trace_manifest = {
            "schema_version": "bonafide-topk-trace-manifest/v1",
            "phase": TRACE_PHASE,
            "claim_boundary": CLAIM_BOUNDARY,
            "trace_family": {
                "trace_family_id": "bonafide.t5-upstream-summed-top5.v1",
                "candidate_policy_id": "model_top5",
                "candidate_policy_version": "1",
                "candidate_count": 5,
                "joint_objective_id": "raw_logit_sum",
                "joint_objective_version": "1",
            },
            "teacher_forcing_contract": {
                "serialization_mode": "historical_thinking_continuation",
                "system_prompt_sha256": system_prompt_sha256,
                "token_identity_schema_version": (
                    "adag.teacher-forced-token-identity.v1"
                ),
                "hash_encoding": ("sha256_utf8_canonical_json_integer_array_v1"),
            },
            "source": {
                "width1_manifest_path": str(
                    (destination / "width1-source-manifest.json").resolve()
                ),
                "width1_manifest_sha256": file_sha256(
                    temporary / "width1-source-manifest.json"
                ),
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "tokenizer_revision": MODEL_REVISION,
                "chat_template_sha256": chat_template_sha256,
            },
            "waves": trace_waves,
        }
        _write_json(temporary / "trace-manifest.json", trace_manifest)
        _write_json(temporary / "run-config.json", _run_config())
        _write_jsonl(temporary / "selection.jsonl", selection_rows)
        inventory = _inventory(temporary)
        _write_json(temporary / "inventory.json", inventory)
        core = {
            "schema_version": SCHEMA_VERSION,
            "status": "frozen_resource_calibration_not_launched",
            "claim_boundary": CLAIM_BOUNDARY,
            "sampling_v2_root": str(source),
            "sampling_v2_manifest_sha256": expected_sampling_manifest_sha256,
            "sampling_v2_inventory_sha256": expected_sampling_inventory_sha256,
            "source_literal_census": _load_object(
                source / "context-source-binding.json"
            )["literal_census"],
            "runtime_tokenization_census": {
                "source_responses": len(documents),
                "full_generation_responses": len(documents)
                - len(non_generation_response_ids),
                "exact_responses": len(exact_tokenizations),
                "excluded_responses": len(excluded_response_ids),
                "excluded_response_ids": excluded_response_ids,
                "non_generation_responses": len(non_generation_response_ids),
                "non_generation_response_ids": non_generation_response_ids,
            },
            "context_bins": [
                {"wave_id": name, "lower_inclusive": lower, "upper_inclusive": upper}
                for name, lower, upper in bins
            ],
            "selected_contexts_by_wave": {
                wave["wave_id"]: [
                    int(row["rendered_total_context_token_count"])
                    for row in wave["selected"]
                ]
                for wave in selected_waves
            },
            "selected_target_ids": [
                str(row["target_id"])
                for wave in selected_waves
                for row in wave["selected"]
            ],
            "width1_source_manifest_sha256": file_sha256(
                temporary / "width1-source-manifest.json"
            ),
            "trace_manifest_sha256": file_sha256(temporary / "trace-manifest.json"),
            "run_config_sha256": file_sha256(temporary / "run-config.json"),
            "selection_sha256": file_sha256(temporary / "selection.jsonl"),
            "inventory_sha256": inventory["inventory_sha256"],
            "execution_source_revision": execution_source_revision,
            "selected_sampling_policy": None,
            "selected_trace_corpus": None,
            "semantic_graph_inspection_performed": False,
        }
        manifest = {**core, "manifest_sha256": canonical_sha256(core)}
        _write_json(temporary / "manifest.json", manifest)
        _readonly_tree(temporary)
        _publish_no_replace(temporary, destination)
        return manifest
    except BaseException:
        if temporary.exists():
            temporary.chmod(0o755)
            for path in temporary.rglob("*"):
                path.chmod(0o755 if path.is_dir() else 0o644)
            shutil.rmtree(temporary)
        raise


def _reject_symlinks(root: Path) -> None:
    if root.is_symlink():
        raise ValueError("resource calibration root symlink drift")
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(
                f"resource calibration descendant symlink drift: {path.relative_to(root)}"
            )


def _verify_self_hash(value: Mapping[str, Any], field: str, label: str) -> None:
    core = dict(value)
    observed = core.pop(field, None)
    if observed != canonical_sha256(core):
        raise ValueError(f"{label} self-hash drift")


def _load_default_tokenizer(config: Mapping[str, Any]) -> Any:
    from transformers import AutoTokenizer

    model = config["model"]
    source = Path(os.path.expandvars(str(model["local_snapshot_path"])))
    if not source.is_dir():
        raise FileNotFoundError(
            f"frozen Thinking tokenizer snapshot is absent: {source}"
        )
    return AutoTokenizer.from_pretrained(source, local_files_only=True)


def load_frozen_resource_calibration_v1(
    root: Path,
    *,
    tokenizer: Any | None = None,
    system_prompt: str = SYSTEM_PROMPT,
) -> dict[str, Any]:
    """Strictly reload a calibration root and reject source or target drift."""

    _reject_symlinks(root)
    root = root.resolve()
    manifest = _load_object(root / "manifest.json")
    _verify_self_hash(manifest, "manifest_sha256", "resource calibration manifest")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != "frozen_resource_calibration_not_launched"
        or manifest.get("claim_boundary") != CLAIM_BOUNDARY
        or manifest.get("selected_sampling_policy") is not None
        or manifest.get("selected_trace_corpus") is not None
        or manifest.get("semantic_graph_inspection_performed") is not False
    ):
        raise ValueError("resource calibration claim/status drift")
    inventory = _load_object(root / "inventory.json")
    _verify_self_hash(inventory, "inventory_sha256", "resource calibration inventory")
    if inventory.get("schema_version") != INVENTORY_SCHEMA_VERSION or inventory.get(
        "inventory_sha256"
    ) != manifest.get("inventory_sha256"):
        raise ValueError("resource calibration inventory binding drift")
    declared_rows = inventory.get("files")
    if not isinstance(declared_rows, list):
        raise ValueError("resource calibration inventory files drift")
    declared = {str(row.get("path")): row for row in declared_rows}
    if len(declared) != len(declared_rows):
        raise ValueError("resource calibration duplicate inventory path drift")
    observed = {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file() and path.name not in {"manifest.json", "inventory.json"}
    }
    if set(declared) != set(observed):
        raise ValueError("resource calibration file membership drift")
    for relative, path in observed.items():
        binding = declared[relative]
        if (path.stat().st_size, file_sha256(path)) != (
            int(binding["bytes"]),
            str(binding["sha256"]),
        ):
            raise ValueError(f"resource calibration file drift: {relative}")
    if root.stat().st_mode & 0o777 != 0o555:
        raise ValueError("resource calibration root mode drift")
    for path in root.rglob("*"):
        expected_mode = 0o555 if path.is_dir() else 0o444
        if path.stat().st_mode & 0o777 != expected_mode:
            raise ValueError(
                f"resource calibration mode drift: {path.relative_to(root)}"
            )

    sampling_root = Path(str(manifest["sampling_v2_root"])).resolve()
    source = load_frozen_post_campaign_sampling_v2(sampling_root)
    source_manifest = source["manifest"]
    if source_manifest.get("manifest_sha256") != manifest.get(
        "sampling_v2_manifest_sha256"
    ) or source_manifest.get("inventory_sha256") != manifest.get(
        "sampling_v2_inventory_sha256"
    ):
        raise ValueError("resource calibration sampling-v2 source drift")
    context_binding = _load_object(sampling_root / "context-source-binding.json")
    if context_binding.get("literal_census") != manifest.get("source_literal_census"):
        raise ValueError("resource calibration source census drift")
    execution_source = manifest.get("execution_source_revision")
    if not isinstance(execution_source, Mapping):
        raise ValueError("resource calibration execution source drift")
    repo_root = Path(str(execution_source.get("repo_root", ""))).resolve()
    if _execution_source_revision(repo_root) != execution_source:
        raise ValueError("resource calibration execution source drift")

    run_config = _load_object(root / "run-config.json")
    if run_config != _run_config() or file_sha256(
        root / "run-config.json"
    ) != manifest.get("run_config_sha256"):
        raise ValueError("resource calibration run-config drift")
    if tokenizer is None:
        tokenizer = _load_default_tokenizer(run_config)
    width1 = _load_object(root / "width1-source-manifest.json")
    trace = _load_object(root / "trace-manifest.json")
    if (
        file_sha256(root / "width1-source-manifest.json")
        != manifest.get("width1_source_manifest_sha256")
        or file_sha256(root / "trace-manifest.json")
        != manifest.get("trace_manifest_sha256")
        or trace.get("waves") != width1.get("waves")
        or trace.get("phase") != TRACE_PHASE
        or trace.get("claim_boundary") != CLAIM_BOUNDARY
        or Path(str(trace["source"]["width1_manifest_path"])).resolve()
        != (root / "width1-source-manifest.json")
        or trace["source"].get("width1_manifest_sha256")
        != file_sha256(root / "width1-source-manifest.json")
    ):
        raise ValueError("resource calibration trace/source manifest drift")
    from scripts.bonafide.topk_manifest import validate_topk_manifest

    validate_topk_manifest(trace)

    parent_root = Path(str(source_manifest["parent_v1_root"])).resolve()
    workstation = _load_object(
        parent_root / "source-evidence/bundle/workstation-bundle.json"
    )
    documents = {str(row["response_id"]): row for row in workstation["documents"]}
    context_rows = {
        str(row["response_id"]): row
        for row in _iter_gzip_jsonl(sampling_root / "context-count-evidence.jsonl.gz")
    }
    exact: dict[str, tuple[list[int], list[int]]] = {}
    excluded = []
    non_generation = []
    for response_id in sorted(documents):
        if context_rows[response_id].get("source_kind") != (
            "generation_reproducibility_prompt_token_ids"
        ):
            non_generation.append(response_id)
            continue
        value = _exact_runtime_tokenization(
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            document=documents[response_id],
            context_evidence=context_rows[response_id],
        )
        if value is None:
            excluded.append(response_id)
        else:
            exact[response_id] = value
    expected_census = {
        "source_responses": len(documents),
        "full_generation_responses": len(documents) - len(non_generation),
        "exact_responses": len(exact),
        "excluded_responses": len(excluded),
        "excluded_response_ids": excluded,
        "non_generation_responses": len(non_generation),
        "non_generation_response_ids": non_generation,
    }
    if expected_census != manifest.get("runtime_tokenization_census"):
        raise ValueError("resource calibration runtime tokenization census drift")

    candidates = _candidate_union(
        _iter_gzip_jsonl(sampling_root / "realized-candidate-tiers.jsonl.gz")
    )
    bins = tuple(
        (
            str(row["wave_id"]),
            int(row["lower_inclusive"]),
            int(row["upper_inclusive"]),
        )
        for row in manifest["context_bins"]
    )
    prompt_blocks = {
        response_id: _hash_text(str(document["task_context"]["prompt"]))
        for response_id, document in documents.items()
    }
    selected = _select_waves(
        candidates=candidates,
        exact_response_ids=set(exact),
        prompt_block_by_response=prompt_blocks,
        context_bins=bins,
    )
    expected_ids = [
        str(row["target_id"]) for wave in selected for row in wave["selected"]
    ]
    if expected_ids != manifest.get("selected_target_ids"):
        raise ValueError("resource calibration target selection drift")
    trace_waves = trace.get("waves")
    if not isinstance(trace_waves, list) or len(trace_waves) != len(selected):
        raise ValueError("resource calibration wave census drift")
    for expected_wave, frozen_wave_raw in zip(selected, trace_waves, strict=True):
        if not isinstance(frozen_wave_raw, Mapping):
            raise ValueError("resource calibration wave object drift")
        frozen_wave = dict(frozen_wave_raw)
        if frozen_wave.get("wave_id") != expected_wave["wave_id"]:
            raise ValueError("resource calibration wave identity drift")
        items = frozen_wave.get("items")
        if not isinstance(items, list) or len(items) != len(expected_wave["selected"]):
            raise ValueError("resource calibration target census drift")
        expected_sorted = expected_wave["selected"]
        for expected_row, item_raw in zip(expected_sorted, items, strict=True):
            if not isinstance(item_raw, Mapping):
                raise ValueError("resource calibration target object drift")
            item = dict(item_raw)
            response_id = str(expected_row["response_id"])
            prefix_ids, response_ids = exact[response_id]
            position = int(expected_row["token_index"])
            resource = item.get("resource_calibration")
            if not isinstance(resource, Mapping):
                raise ValueError("resource calibration target provenance drift")
            if (
                item.get("artifact_id")
                != _artifact_id(
                    str(expected_row["target_id"]),
                    int(expected_row["rendered_total_context_token_count"]),
                )
                or item.get("response_token_count") != len(response_ids)
                or item.get("target_selection")
                != {
                    "kind": "explicit_response_positions",
                    "response_token_positions": [position],
                    "width": 1,
                    "final_target_token_id": response_ids[position],
                }
                or item.get("example", {}).get("prompt")
                != documents[response_id]["task_context"]["prompt"]
                or item.get("example", {}).get("response")
                != documents[response_id]["text"]
                or item.get("example", {})
                .get("token_identity", {})
                .get("assistant_prefix_ids_sha256")
                != canonical_sha256(prefix_ids)
                or item.get("example", {})
                .get("token_identity", {})
                .get("response_ids_sha256")
                != canonical_sha256(response_ids)
                or resource.get("target_id") != expected_row["target_id"]
                or resource.get("actual_context_token_count")
                != expected_row["rendered_total_context_token_count"]
            ):
                raise ValueError("resource calibration target/token/context drift")
    return {
        "manifest": manifest,
        "trace_manifest": trace,
        "run_config": run_config,
        "selected_waves": selected,
    }
