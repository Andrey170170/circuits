"""Frozen, provider-free human review packets for coarse qualification output."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from circuits.analysis.bonafide.canonical import canonical_sha256, file_sha256
from circuits.analysis.bonafide.coarse_sampling_annotation import (
    COARSE_TAGS,
    validate_decisions,
)
from circuits.analysis.bonafide.coarse_sampling_openai_run import (
    EVENT_SCHEMA,
    INTENT_SCHEMA,
    RUN_SCHEMA,
    load_offline_qualification,
)
from circuits.labeling.api import openai_usage
from circuits.labeling.io import (
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_jsonl,
)
from circuits.labeling.pricing import estimate_cost, load_price_snapshot

REVIEW_ROW_SCHEMA = "adag.process-witness.coarse-human-review-row.v1"
REVIEW_PACKET_SCHEMA = "adag.process-witness.coarse-human-review-packet.v1"
HUMAN_EXPORT_SCHEMA = "adag.process-witness.coarse-human-review-decision.v1"
COST_CORRECTION_SCHEMA = "adag.process-witness.coarse-cost-correction-audit.v1"
UI_VERSION = "process-witness-coarse-review-ui-v2-blind-first"

_BOUND_SOURCE_FILES = (
    "circuits/analysis/bonafide/coarse_sampling_review.py",
    "circuits/analysis/bonafide/coarse_sampling_openai_run.py",
    "circuits/analysis/bonafide/coarse_sampling_annotation.py",
    "circuits/analysis/bonafide/canonical.py",
    "circuits/labeling/api.py",
    "circuits/labeling/io.py",
    "circuits/labeling/pricing.py",
    "circuits/labeling/schema.py",
    "scripts/bonafide/build_process_witness_coarse_review_packet.py",
)


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unreadable JSON object: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"JSONL line {number} is not an object: {path}")
            rows.append(row)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unreadable JSONL: {path}") from error
    return rows


def _verify_self_hash(value: Mapping[str, Any], field: str, label: str) -> None:
    payload = dict(value)
    recorded = payload.pop(field, None)
    if not isinstance(recorded, str) or recorded != canonical_sha256(payload):
        raise ValueError(f"{label} self-hash drift")


def _exact_child_names(root: Path, expected: set[str], label: str) -> None:
    try:
        observed = {path.name for path in root.iterdir()}
    except OSError as error:
        raise ValueError(f"unreadable {label}: {root}") from error
    if observed != expected:
        raise ValueError(f"{label} membership drift")


def _nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"invalid raw receipt usage field: {label}")
    return value


def _corrected_receipt_cost(
    *, raw: Mapping[str, Any], request: Mapping[str, Any], prices: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_usage = raw.get("usage")
    if not isinstance(raw_usage, Mapping):
        raise ValueError(f"raw receipt usage is missing: {request['request_id']}")
    input_tokens = _nonnegative_integer(raw_usage.get("input_tokens"), "input_tokens")
    output_tokens = _nonnegative_integer(
        raw_usage.get("output_tokens"), "output_tokens"
    )
    input_details = raw_usage.get("input_tokens_details")
    output_details = raw_usage.get("output_tokens_details")
    if not isinstance(input_details, Mapping) or not isinstance(
        output_details, Mapping
    ):
        raise ValueError(
            f"raw receipt usage details are missing: {request['request_id']}"
        )
    cached = _nonnegative_integer(input_details.get("cached_tokens"), "cached_tokens")
    cache_write = _nonnegative_integer(
        input_details.get("cache_write_tokens"), "cache_write_tokens"
    )
    _nonnegative_integer(output_details.get("reasoning_tokens"), "reasoning_tokens")
    if cached + cache_write > input_tokens:
        raise ValueError(
            f"raw receipt input buckets exceed total: {request['request_id']}"
        )
    if (
        "total_tokens" in raw_usage
        and _nonnegative_integer(raw_usage["total_tokens"], "total_tokens")
        != input_tokens + output_tokens
    ):
        raise ValueError(f"raw receipt total token drift: {request['request_id']}")
    usage = openai_usage(raw_usage)
    if (
        usage.input_tokens != input_tokens
        or usage.output_tokens != output_tokens
        or usage.uncached_input_tokens is None
        or usage.uncached_input_tokens + cached + cache_write != input_tokens
    ):
        raise ValueError(f"normalized raw receipt usage drift: {request['request_id']}")
    cost = estimate_cost(
        prices,
        provider="openai",
        model=str(request["provider_body"]["model"]),
        transport="live",
        usage=usage,
    )
    if not cost.complete or cost.total_cost is None:
        raise ValueError(f"raw receipt cost is incomplete: {request['request_id']}")
    return usage.model_dump(mode="json"), cost.model_dump(mode="json")


def _deep_validate_completed_run(
    *, run_root: Path, qualification_root: Path, qualification: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate every receipt, decision, and cross-artifact binding in a run."""

    _exact_child_names(
        run_root,
        {
            "events.jsonl",
            "intents",
            "raw",
            "records",
            "run-intent.json",
            "run-manifest.json",
        },
        "completed run",
    )
    manifest_path = run_root / "run-manifest.json"
    manifest = _load_object(manifest_path)
    _verify_self_hash(manifest, "run_manifest_sha256", "run manifest")
    if (
        manifest.get("schema_version") != RUN_SCHEMA
        or manifest.get("status") != "complete"
    ):
        raise ValueError("run is not a completed coarse qualification")
    qualification_manifest = qualification["manifest"]
    if (
        Path(manifest.get("qualification_root", "")).resolve()
        != qualification_root.resolve()
        or manifest.get("qualification_manifest_sha256")
        != qualification_manifest["manifest_sha256"]
        or manifest.get("cost_plan_sha256")
        != qualification["cost_plan"]["cost_plan_sha256"]
    ):
        raise ValueError("run-to-qualification binding drift")

    run_intent = _load_object(run_root / "run-intent.json")
    _verify_self_hash(run_intent, "run_intent_sha256", "run intent")
    if run_intent.get("schema_version") != RUN_SCHEMA:
        raise ValueError("run intent schema drift")
    for key, value in run_intent.items():
        if key not in {"status", "run_intent_sha256"} and manifest.get(key) != value:
            raise ValueError(f"run intent/manifest drift: {key}")

    events_path = run_root / "events.jsonl"
    if file_sha256(events_path) != manifest.get("events_jsonl_sha256"):
        raise ValueError("run events hash drift")
    events = _read_jsonl(events_path)
    requests = qualification["requests"]
    request_ids = [request["request_id"] for request in requests]
    if (
        manifest.get("request_ids_in_order") != request_ids
        or run_intent.get("request_ids_in_order") != request_ids
        or len(events) != len(requests)
        or manifest.get("event_count") != len(events)
        or manifest.get("success_count") != len(events)
        or manifest.get("invalid_output_count") != 0
        or manifest.get("provider_incomplete_count") != 0
    ):
        raise ValueError("completed run cardinality or success-count drift")

    expected_names = {f"{request_id}.json" for request_id in request_ids}
    for directory in ("intents", "raw", "records"):
        _exact_child_names(run_root / directory, expected_names, f"run {directory}")

    bindings = manifest.get("record_bindings_in_order")
    if not isinstance(bindings, list) or len(bindings) != len(events):
        raise ValueError("run record bindings drift")
    known_cost = 0.0
    corrected_cost = 0.0
    validated: list[dict[str, Any]] = []
    corrections: list[dict[str, Any]] = []
    price_path = Path(qualification["cost_plan"]["price_snapshot_path"])
    price_sha256 = qualification["cost_plan"]["price_snapshot_sha256"]
    if file_sha256(price_path) != price_sha256:
        raise ValueError("cost-correction price snapshot drift")
    prices = load_price_snapshot(price_path)
    for request, event, binding in zip(requests, events, bindings, strict=True):
        request_id = request["request_id"]
        _verify_self_hash(event, "event_sha256", f"event {request_id}")
        if (
            event.get("schema_version") != EVENT_SCHEMA
            or event.get("status") != "success"
            or event.get("request_id") != request_id
            or event.get("body_sha256") != request["body_sha256"]
            or event.get("repeat_of_request_id") != request["repeat_of_request_id"]
            or binding
            != {"request_id": request_id, "event_sha256": event["event_sha256"]}
        ):
            raise ValueError(f"event/request binding drift: {request_id}")
        record = _load_object(run_root / "records" / f"{request_id}.json")
        if record != event:
            raise ValueError(f"event/record content drift: {request_id}")
        intent = _load_object(run_root / "intents" / f"{request_id}.json")
        _verify_self_hash(intent, "intent_sha256", f"attempt intent {request_id}")
        if (
            intent.get("schema_version") != INTENT_SCHEMA
            or intent.get("request_id") != request_id
            or intent.get("body_sha256") != request["body_sha256"]
            or intent.get("repeat_of_request_id") != request["repeat_of_request_id"]
            or intent.get("run_intent_sha256") != run_intent["run_intent_sha256"]
            or event.get("intent_sha256") != intent["intent_sha256"]
        ):
            raise ValueError(f"attempt intent binding drift: {request_id}")
        raw_path = run_root / "raw" / f"{request_id}.json"
        if (
            event.get("raw_response_path") != f"raw/{request_id}.json"
            or event.get("raw_response_sha256") != file_sha256(raw_path)
            or event.get("raw_text_sha256")
            != hashlib.sha256(str(event.get("raw_text", "")).encode()).hexdigest()
        ):
            raise ValueError(f"raw provider receipt binding drift: {request_id}")
        raw = _load_object(raw_path)
        corrected_usage, corrected_estimate = _corrected_receipt_cost(
            raw=raw, request=request, prices=prices
        )
        decisions = validate_decisions(
            {"decisions": event.get("decisions")},
            focal_unit_ids=request["focal_unit_ids"],
        )
        cost = event.get("cost")
        if not isinstance(cost, Mapping) or cost.get("complete") is not True:
            raise ValueError(f"incomplete priced usage in completed run: {request_id}")
        recorded_request_cost = float(cost["total_cost"])
        corrected_request_cost = float(corrected_estimate["total_cost"])
        known_cost += recorded_request_cost
        corrected_cost += corrected_request_cost
        if not math.isclose(
            float(event.get("cumulative_cost_usd", -1)), known_cost, abs_tol=1e-15
        ):
            raise ValueError(f"cumulative cost drift: {request_id}")
        validated.append({**event, "decisions": decisions})
        corrections.append(
            {
                "request_id": request_id,
                "raw_response_sha256": event["raw_response_sha256"],
                "recorded_usage": event["usage"],
                "corrected_usage_from_raw_receipt": corrected_usage,
                "recorded_cost": dict(cost),
                "corrected_cost": corrected_estimate,
                "cost_delta_usd": corrected_request_cost - recorded_request_cost,
            }
        )
    if (
        manifest.get("cost_complete") is not True
        or not math.isclose(
            float(manifest.get("known_priced_cost_usd", -1)), known_cost, abs_tol=1e-15
        )
        or not math.isclose(
            float(manifest.get("actual_total_cost_usd", -1)), known_cost, abs_tol=1e-15
        )
    ):
        raise ValueError("completed run total-cost drift")
    correction_audit = {
        "schema_version": COST_CORRECTION_SCHEMA,
        "status": "offline_correction_preserving_original_run",
        "original_run_mutated": False,
        "run_manifest_sha256": manifest["run_manifest_sha256"],
        "run_events_jsonl_sha256": file_sha256(events_path),
        "price_snapshot_path": str(price_path.resolve()),
        "price_snapshot_sha256": price_sha256,
        "price_snapshot_id": prices["snapshot_id"],
        "request_count": len(corrections),
        "original_recorded_total_cost_usd": known_cost,
        "corrected_total_cost_usd": corrected_cost,
        "cost_delta_usd": corrected_cost - known_cost,
        "requests": corrections,
    }
    correction_audit["cost_correction_audit_sha256"] = canonical_sha256(
        correction_audit
    )
    return {
        "manifest": manifest,
        "manifest_file_sha256": file_sha256(manifest_path),
        "events": validated,
        "events_file_sha256": file_sha256(events_path),
        "cost_correction_audit": correction_audit,
    }


def _validate_review_template(
    rows: Sequence[Mapping[str, Any]], requests: Sequence[Mapping[str, Any]]
) -> None:
    primary = [
        request for request in requests if request["repeat_of_request_id"] is None
    ]
    expected_ids = [
        unit_id for request in primary for unit_id in request["focal_unit_ids"]
    ]
    observed_ids = [row.get("unit_id") for row in rows]
    if len(rows) != 72 or len(set(observed_ids)) != 72 or observed_ids != expected_ids:
        raise ValueError("human review template focal-unit order or cardinality drift")
    row_by_id = {row["unit_id"]: row for row in rows}
    for request in primary:
        for unit_id in request["focal_unit_ids"]:
            row = row_by_id[unit_id]
            bounded = row.get("bounded_response_units")
            if (
                row.get("schema_version")
                != "adag.process-witness.coarse-human-review-template.v1"
                or row.get("response_id") != request["response_id"]
                or row.get("prompt_sha256") != request["prompt_sha256"]
                or row.get("window_index") != request["window_index"]
                or not isinstance(row.get("task_prompt"), str)
                or not isinstance(row.get("text"), str)
                or not isinstance(bounded, list)
                or [item.get("unit_id") for item in bounded]
                != request["context_unit_ids"]
                or [
                    item.get("unit_id")
                    for item in bounded
                    if item.get("role") == "target"
                ]
                != request["focal_unit_ids"]
            ):
                raise ValueError(f"review template/request drift: {unit_id}")


def merge_review_rows(
    *, qualification_root: Path, run_root: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return 72 unique focal rows after deep validation of both sources."""

    qualification = load_offline_qualification(qualification_root)
    run = _deep_validate_completed_run(
        run_root=run_root,
        qualification_root=qualification_root,
        qualification=qualification,
    )
    templates = _read_jsonl(qualification_root / "human-review-template.jsonl")
    _validate_review_template(templates, qualification["requests"])

    requests = qualification["requests"]
    events_by_request = {event["request_id"]: event for event in run["events"]}
    repeats_by_primary = {
        request["repeat_of_request_id"]: request
        for request in requests
        if request["repeat_of_request_id"] is not None
    }
    request_by_unit: dict[str, Mapping[str, Any]] = {}
    for request in requests:
        if request["repeat_of_request_id"] is None:
            for unit_id in request["focal_unit_ids"]:
                if unit_id in request_by_unit:
                    raise ValueError(
                        f"focal unit belongs to multiple requests: {unit_id}"
                    )
                request_by_unit[unit_id] = request

    identity = {
        "qualification_manifest_sha256": qualification["manifest"]["manifest_sha256"],
        "run_manifest_sha256": run["manifest"]["run_manifest_sha256"],
        "events_jsonl_sha256": run["events_file_sha256"],
        "cost_correction_audit_sha256": run["cost_correction_audit"][
            "cost_correction_audit_sha256"
        ],
    }
    packet_id = f"process-witness-coarse-review-v1-{canonical_sha256(identity)[:16]}"
    merged: list[dict[str, Any]] = []
    for index, template in enumerate(templates):
        unit_id = template["unit_id"]
        request = request_by_unit[unit_id]
        event = events_by_request[request["request_id"]]
        decision_by_id = {value["unit_id"]: value for value in event["decisions"]}
        primary_decision = decision_by_id[unit_id]
        repeat_request = repeats_by_primary.get(request["request_id"])
        repeat_model = None
        if repeat_request is not None:
            repeat_event = events_by_request[repeat_request["request_id"]]
            repeat_decision = {
                value["unit_id"]: value for value in repeat_event["decisions"]
            }[unit_id]
            repeat_model = {
                "request_id": repeat_request["request_id"],
                "model_resolved": repeat_event["model_resolved"],
                **repeat_decision,
            }
        bounded = [
            {**item, "is_reviewed_focal": item["unit_id"] == unit_id}
            for item in template["bounded_response_units"]
        ]
        row = {
            **template,
            "schema_version": REVIEW_ROW_SCHEMA,
            "packet_id": packet_id,
            "review_index": index,
            "bounded_response_units": bounded,
            "machine_primary": {
                "request_id": request["request_id"],
                "model_resolved": event["model_resolved"],
                **primary_decision,
            },
            "machine_repeat": repeat_model,
            "repeat_tag_agreement": (
                None
                if repeat_model is None
                else primary_decision["tag"] == repeat_model["tag"]
            ),
            "repeat_confidence_agreement": (
                None
                if repeat_model is None
                else primary_decision["confidence"] == repeat_model["confidence"]
            ),
            "human_tag": None,
            "human_notes": "",
        }
        row.pop("notes", None)
        row["row_sha256"] = canonical_sha256(row)
        merged.append(row)
    return merged, {
        "packet_id": packet_id,
        "qualification": qualification,
        "run": run,
    }


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source_revision() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    if Path(_git(root, "rev-parse", "--show-toplevel")) != root:
        raise ValueError("coarse review builder repository root drift")
    if _git(root, "status", "--porcelain=v1", "--untracked-files=no"):
        raise ValueError("coarse review build requires a clean tracked worktree")
    commit = _git(root, "rev-parse", "HEAD")
    files = []
    for relative in _BOUND_SOURCE_FILES:
        if _git(root, "ls-files", "--error-unmatch", "--", relative) != relative:
            raise ValueError(f"coarse review source is untracked: {relative}")
        path = root / relative
        blob = _git(root, "rev-parse", f"{commit}:{relative}")
        committed = subprocess.run(
            ["git", "cat-file", "blob", blob],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
        expected = hashlib.sha256(committed).hexdigest()
        if file_sha256(path) != expected:
            raise ValueError(f"coarse review source differs from HEAD: {relative}")
        files.append({"path": relative, "git_blob": blob, "sha256": expected})
    return {
        "repo_root": str(root),
        "git_commit": commit,
        "git_tree": _git(root, "rev-parse", "HEAD^{tree}"),
        "tracked_worktree_clean": True,
        "files": files,
    }


def _readonly_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Coarse qualification review</title>
<style>
:root{color-scheme:light;--ink:#182220;--muted:#66716e;--line:#d7deda;--paper:#f6f7f4;--card:#fff;--accent:#126a55;--target:#fff0bf;--context:#f1f3f1}
*{box-sizing:border-box}body{margin:0;font:15px/1.45 system-ui,sans-serif;background:var(--paper);color:var(--ink)}
header{position:sticky;top:0;z-index:2;background:#18372f;color:#fff;padding:12px 20px;display:flex;gap:14px;align-items:center;flex-wrap:wrap}
header strong{margin-right:auto}.progress{font-variant-numeric:tabular-nums}.controls{display:flex;gap:7px;align-items:center}button,select,textarea,input{font:inherit}button,select{padding:7px 10px;border:1px solid #aab5b1;border-radius:6px;background:#fff}button{cursor:pointer}main{max-width:1280px;margin:18px auto;padding:0 18px 40px;display:grid;grid-template-columns:minmax(0,1fr) 350px;gap:16px}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px}.meta{color:var(--muted);font-size:13px}.prompt{white-space:pre-wrap;max-height:260px;overflow:auto;background:#f5f7f6;border-radius:7px;padding:12px}.units{display:flex;flex-direction:column;gap:7px}.unit{white-space:pre-wrap;padding:9px 11px;border-left:4px solid #b8c2be;background:var(--context);border-radius:4px}.unit.target{border-color:#d79a12;background:#fff8df}.unit.focus{outline:3px solid #126a55;background:var(--target)}.unit small{display:block;color:var(--muted);margin-bottom:4px}.model{border:1px solid var(--line);border-radius:7px;padding:10px;margin:8px 0}.tag{font-weight:700;color:var(--accent)}.warn{color:#a24418;font-weight:650}.review{position:sticky;top:82px;height:max-content}textarea{width:100%;min-height:140px;padding:9px;border:1px solid #aab5b1;border-radius:6px}label{display:block;font-weight:650;margin:14px 0 5px}.claim{font-size:12px;color:var(--muted);margin-top:14px}@media(max-width:850px){main{grid-template-columns:1fr}.review{position:static}}
</style></head><body>
<header><strong>Coarse qualification review</strong><span id="packet"></span><span class="progress" id="progress"></span><div class="controls"><button id="prev">Previous</button><input id="jump" type="number" min="1" style="width:74px"><button id="next">Next</button><button id="export">Export JSONL</button></div></header>
<main><section><div class="card"><div class="meta" id="meta"></div><h3>Task prompt</h3><div class="prompt" id="prompt"></div><h3>Bounded response context</h3><div class="units" id="units"></div></div></section>
<aside class="card review"><div id="blind"><h3>Blind human judgment</h3><p class="meta">Choose and lock a tag before model labels are revealed. The locked judgment cannot be edited.</p><label for="human">Human tag</label><select id="human"></select><label for="notes">Notes</label><textarea id="notes" placeholder="Optional pre-reveal review notes"></textarea><button id="lock">Lock judgment and reveal labels</button><p class="warn" id="lockError"></p></div><div id="revealed" hidden><h3>Model labels (revealed)</h3><div id="models"></div><label for="correction">Optional post-reveal correction</label><select id="correction"></select><label for="correctionNotes">Correction notes</label><textarea id="correctionNotes" placeholder="This is stored separately; it never changes the blind judgment."></textarea><button id="saveCorrection">Save post-reveal correction</button><p class="meta" id="revealMeta"></p></div><p class="claim">These coarse tags are sampling metadata only, not adequacy or motif labels.</p></aside></main>
<script>
const rows=JSON.parse(new TextDecoder().decode(Uint8Array.from(atob("__ROWS_BASE64__"),c=>c.charCodeAt(0))));
const tags=__TAGS_JSON__,packetId="__PACKET_ID__",uiVersion="__UI_VERSION__",uiTemplateSha256="__UI_TEMPLATE_SHA256__";
const key=`coarse-review:${packetId}:${uiVersion}:${uiTemplateSha256}`;
const emptyState=()=>({schema_version:"adag.process-witness.coarse-human-review-browser-state.v2",packet_id:packetId,ui_version:uiVersion,ui_template_sha256:uiTemplateSha256,records:{}});
let saved=emptyState();try{const candidate=JSON.parse(localStorage.getItem(key)||"null");if(candidate&&candidate.packet_id===packetId&&candidate.ui_version===uiVersion&&candidate.ui_template_sha256===uiTemplateSha256)saved=candidate}catch(e){}let index=0;
const $=id=>document.getElementById(id), human=$("human"),notes=$("notes"),correction=$("correction"),correctionNotes=$("correctionNotes");
const options='<option value="">Select a tag</option>'+tags.map(t=>`<option value="${t}">${t}</option>`).join('');human.innerHTML=options;correction.innerHTML=options;
function text(el,value){el.textContent=value==null?'':String(value)}
function record(r){return saved.records[r.unit_id]||(saved.records[r.unit_id]={events:[]})}function eventOf(rec,type){return rec.events.find(e=>e.event_type===type)}function corrections(rec){return rec.events.filter(e=>e.event_type==='post_reveal_correction_recorded')}
function persist(){localStorage.setItem(key,JSON.stringify(saved));renderProgress()}
function modelCard(title,m){const d=document.createElement('div');d.className='model';if(!m){d.textContent=title+': no repeat';return d}const b=(m.boundary_concerns||[]).join(', ')||'none';d.innerHTML=`<strong></strong><div class="tag"></div><div class="meta"></div><div></div>`;text(d.children[0],title);text(d.children[1],m.tag);text(d.children[2],`confidence: ${m.confidence} · model: ${m.model_resolved}`);text(d.children[3],`boundary: ${b}${m.boundary_note?' · '+m.boundary_note:''}`);return d}
function renderProgress(){const n=rows.filter(r=>eventOf(record(r),'pre_reveal_judgment_locked')).length;text($("progress"),`${index+1} / ${rows.length} · ${n} locked`)}
function render(){const r=rows[index],rec=record(r),locked=eventOf(rec,'pre_reveal_judgment_locked'),reveal=eventOf(rec,'model_labels_revealed'),latest=corrections(rec).at(-1);text($("packet"),packetId);text($("meta"),`${r.source_type_stratum} · ${r.position_stratum} · tokens ${r.token_span.join(':')} · ${r.unit_id}`);text($("prompt"),r.task_prompt);const units=$("units");units.replaceChildren();for(const u of r.bounded_response_units){const d=document.createElement('div');d.className='unit '+(u.role==='target'?'target ':'')+(u.is_reviewed_focal?'focus':'');const small=document.createElement('small');text(small,(u.is_reviewed_focal?'REVIEW THIS UNIT · ':'')+u.role.toUpperCase()+' · '+u.unit_id);const body=document.createElement('span');text(body,u.text);d.append(small,body);units.append(d)}$("blind").hidden=!!locked;$("revealed").hidden=!locked;if(locked){const models=$("models");models.replaceChildren(modelCard('Primary',r.machine_primary),modelCard('Repeat',r.machine_repeat));if(r.machine_repeat&&r.repeat_tag_agreement===false){const w=document.createElement('p');w.className='warn';text(w,'Repeat tag disagreement');models.append(w)}text($("revealMeta"),`Blind tag locked ${locked.at}; labels revealed ${reveal?.at||'unknown'}. Blind tag: ${locked.human_tag}.`);correction.value=latest?.human_tag||'';correctionNotes.value=latest?.human_notes||''}else{human.value='';notes.value='';text($("lockError"),'')}$("jump").value=index+1;renderProgress()}
$("lock").onclick=()=>{const r=rows[index],rec=record(r);if(eventOf(rec,'pre_reveal_judgment_locked'))return;if(!human.value){text($("lockError"),'Select a tag before locking.');return}const lockedAt=new Date().toISOString();rec.events.push({event_type:'pre_reveal_judgment_locked',at:lockedAt,human_tag:human.value,human_notes:notes.value});rec.events.push({event_type:'model_labels_revealed',at:new Date().toISOString()});persist();render()};
$("saveCorrection").onclick=()=>{const r=rows[index],rec=record(r);if(!eventOf(rec,'pre_reveal_judgment_locked')||!correction.value)return;rec.events.push({event_type:'post_reveal_correction_recorded',at:new Date().toISOString(),human_tag:correction.value,human_notes:correctionNotes.value});persist();render()};
$("prev").onclick=()=>{index=Math.max(0,index-1);render()};$("next").onclick=()=>{index=Math.min(rows.length-1,index+1);render()};$("jump").onchange=e=>{index=Math.max(0,Math.min(rows.length-1,Number(e.target.value||1)-1));render()};
$("export").onclick=()=>{const now=new Date().toISOString(),out=rows.map(r=>{const rec=record(r),locked=eventOf(rec,'pre_reveal_judgment_locked'),reveal=eventOf(rec,'model_labels_revealed');return{schema_version:"adag.process-witness.coarse-human-review-decision.v1",packet_id:packetId,ui_version:uiVersion,ui_template_sha256:uiTemplateSha256,row_sha256:r.row_sha256,unit_id:r.unit_id,pre_reveal_human_tag:locked?.human_tag||null,pre_reveal_human_notes:locked?.human_notes||'',pre_reveal_locked_at:locked?.at||null,model_labels_revealed_at:reveal?.at||null,post_reveal_corrections:corrections(rec),event_history:rec.events,exported_at:now}});const blob=new Blob([out.map(x=>JSON.stringify(x)).join('\n')+'\n'],{type:'application/x-ndjson'}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=packetId+'-decisions.jsonl';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};render();
</script></body></html>"""


def render_review_html(rows: Sequence[Mapping[str, Any]], packet_id: str) -> bytes:
    payload = json.dumps(rows, ensure_ascii=False, allow_nan=False).encode("utf-8")
    encoded = base64.b64encode(payload).decode("ascii")
    html = (
        _HTML.replace("__ROWS_BASE64__", encoded)
        .replace("__TAGS_JSON__", json.dumps(list(COARSE_TAGS)))
        .replace("__PACKET_ID__", packet_id)
        .replace("__UI_VERSION__", UI_VERSION)
        .replace("__UI_TEMPLATE_SHA256__", hashlib.sha256(_HTML.encode()).hexdigest())
    )
    return html.encode("utf-8")


def build_review_packet(
    *, qualification_root: Path, run_root: Path, destination: Path
) -> dict[str, Any]:
    """Build and freeze a self-contained, network-free human review packet."""

    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    rows, sources = merge_review_rows(
        qualification_root=qualification_root.resolve(), run_root=run_root.resolve()
    )
    source_revision = _source_revision()
    packet_id = sources["packet_id"]
    temporary = destination.parent / f".{destination.name}.building-{uuid.uuid4().hex}"
    temporary.mkdir(parents=True)
    try:
        rows_path = temporary / "review-rows.jsonl"
        html_path = temporary / "review.html"
        audit_path = temporary / "cost-correction-audit.json"
        atomic_write_jsonl(rows_path, rows)
        atomic_write_bytes(html_path, render_review_html(rows, packet_id))
        qualification = sources["qualification"]
        run = sources["run"]
        atomic_write_json(audit_path, run["cost_correction_audit"])
        files = [
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in (audit_path, html_path, rows_path)
        ]
        ui_template_sha256 = hashlib.sha256(_HTML.encode()).hexdigest()
        price_path = Path(qualification["cost_plan"]["price_snapshot_path"])
        manifest = {
            "schema_version": REVIEW_PACKET_SCHEMA,
            "status": "frozen_offline_human_review_packet",
            "packet_id": packet_id,
            "ui_version": UI_VERSION,
            "ui_template_sha256": ui_template_sha256,
            "claim_boundary": qualification["manifest"]["claim_boundary"],
            "qualification_claim_boundary": qualification["manifest"][
                "qualification_claim_boundary"
            ],
            "network_calls_made": 0,
            "environment": {
                "hostname": platform.node(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            },
            "counts": {
                "review_rows": len(rows),
                "rows_with_repeat": sum(
                    row["machine_repeat"] is not None for row in rows
                ),
                "repeat_tag_disagreements": sum(
                    row["repeat_tag_agreement"] is False for row in rows
                ),
            },
            "qualification_root": str(qualification_root.resolve()),
            "qualification_manifest_sha256": qualification["manifest"][
                "manifest_sha256"
            ],
            "qualification_manifest_file_sha256": file_sha256(
                qualification_root / "manifest.json"
            ),
            "human_review_template_sha256": file_sha256(
                qualification_root / "human-review-template.jsonl"
            ),
            "run_root": str(run_root.resolve()),
            "run_manifest_sha256": run["manifest"]["run_manifest_sha256"],
            "run_manifest_file_sha256": run["manifest_file_sha256"],
            "run_events_jsonl_sha256": run["events_file_sha256"],
            "cost_correction_audit_sha256": run["cost_correction_audit"][
                "cost_correction_audit_sha256"
            ],
            "cost_correction_audit_file_sha256": file_sha256(audit_path),
            "original_recorded_total_cost_usd": run["cost_correction_audit"][
                "original_recorded_total_cost_usd"
            ],
            "corrected_total_cost_usd": run["cost_correction_audit"][
                "corrected_total_cost_usd"
            ],
            "cost_delta_usd": run["cost_correction_audit"]["cost_delta_usd"],
            "price_snapshot_path": str(price_path.resolve()),
            "price_snapshot_sha256": qualification["cost_plan"][
                "price_snapshot_sha256"
            ],
            "price_snapshot_id": run["cost_correction_audit"]["price_snapshot_id"],
            "ui_sha256": file_sha256(html_path),
            "source_revision": source_revision,
            "files": files,
        }
        manifest["manifest_sha256"] = canonical_sha256(manifest)
        atomic_write_json(temporary / "manifest.json", manifest)
        os.replace(temporary, destination)
        _readonly_tree(destination)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
