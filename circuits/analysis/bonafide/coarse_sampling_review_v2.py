"""Blind-first, full-response human review for the two-arm qualification."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import platform
import shutil
import subprocess
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from circuits.analysis.bonafide.canonical import canonical_sha256, file_sha256
from circuits.analysis.bonafide.coarse_sampling_annotation import COARSE_TAGS
from circuits.analysis.bonafide.coarse_sampling_annotation_v2 import (
    ARM_FULL_UNIT,
    ARM_TARGET_ONLY,
)
from circuits.analysis.bonafide.coarse_sampling_comparison_v2 import (
    load_comparison_bundle,
    load_completed_comparison_inputs,
)
from circuits.labeling.io import (
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_jsonl,
    read_jsonl,
)

DOCUMENT_SCHEMA = "adag.process-witness.coarse-human-review-document.v2"
ITEM_SCHEMA = "adag.process-witness.coarse-human-review-item.v2"
PACKET_SCHEMA = "adag.process-witness.coarse-human-review-packet.v2"
EXPORT_SCHEMA = "adag.process-witness.coarse-human-review-decision.v2"
UI_VERSION = "process-witness-coarse-review-ui-v3-full-response-two-arm-blind"

_BOUND_SOURCE_FILES = (
    "circuits/analysis/bonafide/coarse_sampling_review_v2.py",
    "circuits/analysis/bonafide/coarse_sampling_comparison_v2.py",
    "circuits/analysis/bonafide/coarse_sampling_annotation_v2.py",
    "circuits/analysis/bonafide/canonical.py",
    "circuits/labeling/io.py",
    "scripts/bonafide/build_process_witness_coarse_review_packet_v2.py",
)


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unreadable JSON object: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _decision(event: Mapping[str, Any], unit_id: str) -> dict[str, Any]:
    matches = [value for value in event["decisions"] if value["unit_id"] == unit_id]
    if len(matches) != 1:
        raise ValueError(f"review decision coverage drift: {unit_id}")
    return dict(matches[0])


def _revealed_decision(
    *, key: str, label: str, event: Mapping[str, Any] | None, unit_id: str
) -> dict[str, Any]:
    if event is None:
        return {"decision_key": key, "label": label, "available": False}
    return {
        "decision_key": key,
        "label": label,
        "available": True,
        "request_id": event["request_id"],
        "model_resolved": event.get("model_resolved"),
        "decision": _decision(event, unit_id),
    }


def _validate_doc_and_units(
    document: Mapping[str, Any], units: Sequence[Mapping[str, Any]]
) -> None:
    text = document["text"]
    prompt = document["task_context"]["prompt"]
    tokenization = document["tokenization"]
    if (
        hashlib.sha256(text.encode()).hexdigest() != document["text_sha256"]
        or hashlib.sha256(prompt.encode()).hexdigest() != document["prompt_sha256"]
        or tokenization.get("token_count") != len(tokenization.get("tokens", []))
    ):
        raise ValueError("review source document text/token identity drift")
    previous_end = 0
    for index, unit in enumerate(units):
        core = unit["core_character_span"]
        token_span = unit["token_span"]
        if (
            unit.get("response_id") != document["response_id"]
            or unit.get("prompt_sha256") != document["prompt_sha256"]
            or unit.get("input_ids_sha256") != tokenization["input_ids_sha256"]
            or unit.get("offset_mapping_sha256")
            != tokenization["offset_mapping_sha256"]
            or unit.get("sequence_index") != index
            or not 0 <= core[0] < core[1] <= len(text)
            or core[0] < previous_end
            or text[core[0] : core[1]] != unit["text"]
            or unit.get("text_sha256") != document["text_sha256"]
            or not 0 <= token_span[0] < token_span[1] <= tokenization["token_count"]
        ):
            raise ValueError(
                f"review coarse-unit identity drift: {unit.get('unit_id')}"
            )
        previous_end = core[1]


def assemble_review_payload(
    *, qualification_root: Path, run_root: Path, comparison_root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Create deduplicated documents and 72 blind review items."""

    inputs = load_completed_comparison_inputs(
        run_root=run_root, qualification_root=qualification_root
    )
    comparison = load_comparison_bundle(comparison_root)
    qualification = inputs["qualification"]
    q_manifest = qualification["manifest"]
    c_manifest = comparison["manifest"]
    if (
        Path(c_manifest["source_run_root"]).resolve() != run_root.resolve()
        or Path(c_manifest["source_qualification_root"]).resolve()
        != qualification_root.resolve()
        or c_manifest["source_bindings"] != comparison["report"]["source_bindings"]
        or c_manifest["source_bindings"]["qualification_manifest_sha256"]
        != q_manifest["manifest_sha256"]
        or c_manifest["source_bindings"]["collection_manifest_sha256"]
        != inputs["collection"]["collection_manifest_sha256"]
    ):
        raise ValueError("review comparison/source binding drift")

    workstation_path = Path(q_manifest["source_workstation_bundle"])
    if file_sha256(workstation_path) != q_manifest["source_workstation_bundle_sha256"]:
        raise ValueError("review workstation bundle hash drift")
    workstation = _load_object(workstation_path)
    workstation_docs = {
        value["response_id"]: value for value in workstation.get("documents", [])
    }
    v1_root = Path(q_manifest["source_v1_qualification_root"])
    units_path = v1_root / "units.jsonl"
    if file_sha256(units_path) != qualification["config"]["source"]["v1_units_sha256"]:
        raise ValueError("review source coarse-unit file drift")
    selected_response_ids = {
        window["response_id"] for window in qualification["windows"]
    }
    units_by_response: dict[str, list[dict[str, Any]]] = {
        response_id: [] for response_id in selected_response_ids
    }
    for unit in read_jsonl(units_path):
        if unit["response_id"] in units_by_response:
            units_by_response[unit["response_id"]].append(unit)
    focal_by_id = {value["unit_id"]: value for value in qualification["focal_units"]}

    documents = []
    doc_by_response: dict[str, dict[str, Any]] = {}
    for window in qualification["windows"]:
        response_id = window["response_id"]
        source = workstation_docs.get(response_id)
        if not isinstance(source, Mapping):
            raise ValueError(f"review workstation document missing: {response_id}")
        source_units = sorted(
            units_by_response[response_id], key=lambda value: value["sequence_index"]
        )
        _validate_doc_and_units(source, source_units)
        if source["prompt_sha256"] != window["prompt_sha256"] or any(
            focal_by_id[unit_id]
            != next(value for value in source_units if value["unit_id"] == unit_id)
            for unit_id in window["focal_unit_ids"]
        ):
            raise ValueError("review window/focal document binding drift")
        document = {
            "schema_version": DOCUMENT_SCHEMA,
            "document_id": response_id,
            "response_id": response_id,
            "prompt_sha256": source["prompt_sha256"],
            "task_prompt": source["task_context"]["prompt"],
            "response_text_sha256": source["text_sha256"],
            "response_text": source["text"],
            "response_character_count": len(source["text"]),
            "tokenization": {
                key: source["tokenization"][key]
                for key in (
                    "identity_status",
                    "token_count",
                    "input_ids_sha256",
                    "offset_mapping_sha256",
                )
            },
            "coarse_units": [
                {
                    key: unit[key]
                    for key in (
                        "unit_id",
                        "sequence_index",
                        "unit_kind",
                        "token_span",
                        "core_character_span",
                        "covering_character_span",
                    )
                }
                for unit in source_units
            ],
        }
        document["document_sha256"] = canonical_sha256(document)
        documents.append(document)
        doc_by_response[response_id] = document
    if len(documents) != 12 or len(doc_by_response) != 12:
        raise ValueError("review document census drift")

    requests = qualification["requests"]
    v2_event_by_request = {value["request_id"]: value for value in inputs["events"]}
    v1_events = qualification["v1_comparison_baseline"]["events"]
    v1_event_by_request = {value["request_id"]: value for value in v1_events}
    v1_repeat_by_primary = {
        value["repeat_of_request_id"]: value
        for value in v1_events
        if value["repeat_of_request_id"] is not None
    }
    primary_by_arm_window = {
        (value["arm_id"], value["window_index"]): value
        for value in requests
        if value["repeat_of_request_id"] is None
    }
    repeat_by_primary = {
        value["repeat_of_request_id"]: value
        for value in requests
        if value["repeat_of_request_id"] is not None
    }
    identity = {
        "qualification_manifest_sha256": q_manifest["manifest_sha256"],
        "collection_manifest_sha256": inputs["collection"][
            "collection_manifest_sha256"
        ],
        "comparison_manifest_sha256": c_manifest["manifest_sha256"],
        "comparison_report_sha256": comparison["report"]["report_sha256"],
        "workstation_bundle_sha256": q_manifest["source_workstation_bundle_sha256"],
        "v1_units_sha256": qualification["config"]["source"]["v1_units_sha256"],
    }
    packet_id = f"process-witness-coarse-review-v2-{canonical_sha256(identity)[:16]}"

    items = []
    for window in qualification["windows"]:
        window_index = window["window_index"]
        target_primary = primary_by_arm_window[(ARM_TARGET_ONLY, window_index)]
        full_primary = primary_by_arm_window[(ARM_FULL_UNIT, window_index)]
        if (
            target_primary["focal_unit_ids"] != window["focal_unit_ids"]
            or full_primary["focal_unit_ids"] != window["focal_unit_ids"]
        ):
            raise ValueError("review arm/window focal binding drift")
        v1_primary = v1_event_by_request[target_primary["source_v1_request_id"]]
        v1_repeat = v1_repeat_by_primary.get(v1_primary["request_id"])
        target_repeat_request = repeat_by_primary.get(target_primary["request_id"])
        full_repeat_request = repeat_by_primary.get(full_primary["request_id"])
        target_repeat = (
            None
            if target_repeat_request is None
            else v2_event_by_request[target_repeat_request["request_id"]]
        )
        full_repeat = (
            None
            if full_repeat_request is None
            else v2_event_by_request[full_repeat_request["request_id"]]
        )
        for group_position, unit_id in enumerate(window["focal_unit_ids"], start=1):
            unit = focal_by_id[unit_id]
            reveal = [
                _revealed_decision(
                    key="v1_primary",
                    label="v1 bounded context · primary",
                    event=v1_primary,
                    unit_id=unit_id,
                ),
                _revealed_decision(
                    key="v1_repeat",
                    label="v1 bounded context · repeat",
                    event=v1_repeat,
                    unit_id=unit_id,
                ),
                _revealed_decision(
                    key="target_only_primary",
                    label="v2 target-only markup · primary",
                    event=v2_event_by_request[target_primary["request_id"]],
                    unit_id=unit_id,
                ),
                _revealed_decision(
                    key="target_only_repeat",
                    label="v2 target-only markup · repeat",
                    event=target_repeat,
                    unit_id=unit_id,
                ),
                _revealed_decision(
                    key="full_unit_primary",
                    label="v2 full-unit markup · primary",
                    event=v2_event_by_request[full_primary["request_id"]],
                    unit_id=unit_id,
                ),
                _revealed_decision(
                    key="full_unit_repeat",
                    label="v2 full-unit markup · repeat",
                    event=full_repeat,
                    unit_id=unit_id,
                ),
            ]
            item = {
                "schema_version": ITEM_SCHEMA,
                "packet_id": packet_id,
                "review_index": len(items),
                "item_id": f"pwcoarsereviewv2-{unit_id.removeprefix('pwcoarseunit-')}",
                "document_id": window["response_id"],
                "document_sha256": doc_by_response[window["response_id"]][
                    "document_sha256"
                ],
                "window_index": window_index,
                "source_type_stratum": window["source_type_stratum"],
                "position_stratum": window["position_stratum"],
                "unit_id": unit_id,
                "unit_text": unit["text"],
                "token_span": unit["token_span"],
                "core_character_span": unit["core_character_span"],
                "covering_character_span": unit["covering_character_span"],
                "group_position": group_position,
                "group_size": 6,
                "group_focal_unit_ids": window["focal_unit_ids"],
                "revealed_decisions": reveal,
                "human_tag": None,
                "human_notes": "",
            }
            item["item_sha256"] = canonical_sha256(item)
            items.append(item)
    if len(items) != 72 or len({value["unit_id"] for value in items}) != 72:
        raise ValueError("review item census drift")
    return (
        documents,
        items,
        {
            "packet_id": packet_id,
            "identity": identity,
            "inputs": inputs,
            "comparison": comparison,
            "workstation_path": workstation_path,
            "v1_units_path": units_path,
        },
    )


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
        raise ValueError("coarse v2 review builder repository root drift")
    if _git(root, "status", "--porcelain=v1", "--untracked-files=no"):
        raise ValueError("coarse v2 review build requires a clean tracked worktree")
    commit = _git(root, "rev-parse", "HEAD")
    files = []
    for relative in _BOUND_SOURCE_FILES:
        if _git(root, "ls-files", "--error-unmatch", "--", relative) != relative:
            raise ValueError(f"coarse v2 review source is untracked: {relative}")
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
            raise ValueError(f"coarse v2 review source differs from HEAD: {relative}")
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
<title>Two-arm coarse qualification blind review</title>
<style>
:root{color-scheme:light;--ink:#17211f;--muted:#66716e;--line:#d7deda;--paper:#f4f6f3;--card:#fff;--accent:#126a55;--group:#ffedb0;--focus:#9ee4cf;--boundary:#dce5e1;--warn:#a24418}
*{box-sizing:border-box}body{margin:0;font:15px/1.5 system-ui,sans-serif;background:var(--paper);color:var(--ink)}button,input,select,textarea{font:inherit}button,select,input{border:1px solid #aab5b1;border-radius:6px;background:#fff;padding:7px 9px}button{cursor:pointer}button:disabled{opacity:.5;cursor:not-allowed}
header{position:sticky;top:0;z-index:5;background:#18372f;color:#fff;padding:10px 16px;display:flex;gap:8px;align-items:center;flex-wrap:wrap}header strong{margin-right:auto}.toolbar{display:flex;gap:7px;align-items:center;flex-wrap:wrap}.progress{font-variant-numeric:tabular-nums}.search{width:220px}
main{max-width:1500px;margin:16px auto;padding:0 16px 36px;display:grid;grid-template-columns:minmax(0,1fr) 390px;gap:15px}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:15px}.meta{color:var(--muted);font-size:13px}.prompt,.response{white-space:pre-wrap;overflow:auto;background:#f7f8f6;border:1px solid #e1e5e2;border-radius:7px;padding:12px}.prompt{max-height:230px}.response{max-height:62vh;line-height:1.65}.group{background:var(--group);border-radius:2px}.focus{background:var(--focus);outline:2px solid var(--accent);scroll-margin:160px 0}.boundary{box-shadow:inset 0 -1px var(--boundary)}.boundary:hover{outline:1px solid #8fa29b}.legend{display:flex;gap:16px;margin:8px 0}.swatch{display:inline-block;width:14px;height:14px;border-radius:3px;vertical-align:-2px;margin-right:5px}.swatch.group{background:var(--group)}.swatch.focus{background:var(--focus);outline:1px solid var(--accent)}
.review{position:sticky;top:112px;height:max-content;max-height:calc(100vh - 130px);overflow:auto}label{display:block;font-weight:650;margin:12px 0 5px}textarea{width:100%;min-height:110px;border:1px solid #aab5b1;border-radius:6px;padding:8px}.model{border:1px solid var(--line);border-radius:7px;padding:9px;margin:8px 0}.model .tag{font-weight:750;color:var(--accent)}.model.missing{color:var(--muted);background:#f7f8f6}.warn{color:var(--warn);font-weight:650}.claim{font-size:12px;color:var(--muted);margin-top:13px}.importStatus{font-size:12px;color:var(--muted)}@media(max-width:920px){main{grid-template-columns:1fr}.review{position:static;max-height:none}}
</style></head><body>
<header><strong>Two-arm coarse qualification · blind review</strong><span id="packet"></span><span class="progress" id="progress"></span><div class="toolbar"><button id="prev">Previous</button><input id="jump" type="number" min="1" style="width:72px"><button id="next">Next</button><input id="search" class="search" type="search" placeholder="Search prompt, response, unit ID"><select id="filter"><option value="all">All items</option><option value="unreviewed">Unreviewed</option><option value="locked">Locked</option></select><label style="margin:0;font-weight:400"><input id="boundaries" type="checkbox"> Show all coarse-unit boundaries</label><button id="importButton">Import JSONL</button><input id="importFile" type="file" accept=".jsonl,.ndjson,application/x-ndjson" hidden><button id="export">Export JSONL</button></div></header>
<main><section><div class="card"><div class="meta" id="meta"></div><div class="legend"><span><i class="swatch group"></i>other focal units in group</span><span><i class="swatch focus"></i>current review unit</span></div><h3>Complete task prompt</h3><div class="prompt" id="prompt"></div><h3>Complete exact response</h3><div class="response" id="response"></div></div></section>
<aside class="card review"><div id="blind"><h3>Blind human judgment</h3><p class="meta">Model decisions, repeat availability, and disagreements remain hidden until this judgment is locked.</p><label for="human">Human tag</label><select id="human"></select><label for="notes">Pre-reveal notes</label><textarea id="notes" placeholder="Optional notes made without seeing model labels"></textarea><button id="lock">Lock judgment and reveal all decisions</button><p class="warn" id="lockError"></p></div><div id="revealed" hidden><h3>Decisions revealed after lock</h3><div id="lockedHuman"></div><div id="models"></div><label for="correction">Optional post-reveal correction</label><select id="correction"></select><label for="correctionNotes">Correction notes</label><textarea id="correctionNotes" placeholder="Stored separately; never changes the blind judgment"></textarea><button id="saveCorrection">Save separate correction</button><p class="meta" id="revealMeta"></p></div><p id="importStatus" class="importStatus"></p><p class="claim">Review concerns coarse trace-sampling labels only. It does not judge correctness, faithfulness, motifs, or internal computation.</p></aside></main>
<script>
const payload=JSON.parse(new TextDecoder().decode(Uint8Array.from(atob("__PAYLOAD_BASE64__"),c=>c.charCodeAt(0))));
const documents=new Map(payload.documents.map(d=>[d.document_id,d])),items=payload.items,tags=__TAGS_JSON__,packetId="__PACKET_ID__",uiVersion="__UI_VERSION__",uiTemplateSha256="__UI_TEMPLATE_SHA256__";
const key=`coarse-review-v2:${packetId}:${uiVersion}:${uiTemplateSha256}`,exportSchema="adag.process-witness.coarse-human-review-decision.v2",stateSchema="adag.process-witness.coarse-human-review-browser-state.v3";
const emptyState=()=>({schema_version:stateSchema,packet_id:packetId,ui_version:uiVersion,ui_template_sha256:uiTemplateSha256,records:{}});let saved=emptyState();try{const x=JSON.parse(localStorage.getItem(key)||"null");if(x&&x.schema_version===stateSchema&&x.packet_id===packetId&&x.ui_version===uiVersion&&x.ui_template_sha256===uiTemplateSha256)saved=x}catch(e){}
let index=0,visible=[...items.keys()];const $=id=>document.getElementById(id),human=$("human"),notes=$("notes"),correction=$("correction"),correctionNotes=$("correctionNotes");const options='<option value="">Select a tag</option>'+tags.map(t=>`<option value="${t}">${t}</option>`).join('');human.innerHTML=options;correction.innerHTML=options;
function text(el,value){el.textContent=value==null?'':String(value)}function record(item){return saved.records[item.item_id]||(saved.records[item.item_id]={events:[]})}function eventOf(rec,type){return rec.events.find(e=>e.event_type===type)}function corrections(rec){return rec.events.filter(e=>e.event_type==='post_reveal_correction_recorded')}function persist(){localStorage.setItem(key,JSON.stringify(saved));renderProgress()}
function current(){return items[visible[index]]}function cp(textValue){return Array.from(textValue)}function sliceCodePoints(points,start,end){return points.slice(start,end).join('')}
function renderResponse(doc,item){const box=$("response"),points=cp(doc.response_text),group=new Set(item.group_focal_unit_ids),source=$("boundaries").checked?doc.coarse_units:doc.coarse_units.filter(u=>group.has(u.unit_id));box.replaceChildren();let cursor=0,focus=null;for(const u of source){const [start,end]=u.core_character_span;if(start<cursor)throw new Error('overlapping source spans');if(start>cursor)box.append(document.createTextNode(sliceCodePoints(points,cursor,start)));const span=document.createElement('span');span.textContent=sliceCodePoints(points,start,end);span.className=($("boundaries").checked?'boundary ':'')+(group.has(u.unit_id)?'group ':'')+(u.unit_id===item.unit_id?'focus':'');span.title=`${u.unit_id} · tokens ${u.token_span[0]}:${u.token_span[1]} · chars ${start}:${end}`;box.append(span);if(u.unit_id===item.unit_id)focus=span;cursor=end}if(cursor<points.length)box.append(document.createTextNode(sliceCodePoints(points,cursor,points.length)));requestAnimationFrame(()=>focus?.scrollIntoView({block:'center',inline:'nearest'}))}
function modelCard(model){const d=document.createElement('div');d.className='model'+(model.available?'':' missing');const title=document.createElement('strong');text(title,model.label);d.append(title);if(!model.available){const m=document.createElement('div');text(m,'No repeat in the frozen cohort');d.append(m);return d}const decision=model.decision,tag=document.createElement('div'),meta=document.createElement('div'),boundary=document.createElement('div');tag.className='tag';meta.className='meta';text(tag,decision.tag);text(meta,`confidence: ${decision.confidence} · request: ${model.request_id}`);text(boundary,`boundary: ${(decision.boundary_concerns||[]).join(', ')||'none'}${decision.boundary_note?' · '+decision.boundary_note:''}`);d.append(tag,meta,boundary);return d}
function applyFilter(){const query=$("search").value.trim().toLocaleLowerCase(),mode=$("filter").value;visible=items.map((_,i)=>i).filter(i=>{const item=items[i],doc=documents.get(item.document_id),locked=!!eventOf(record(item),'pre_reveal_judgment_locked');const status=mode==='all'||(mode==='locked'&&locked)||(mode==='unreviewed'&&!locked);const hay=[item.unit_id,item.unit_text,item.document_id,item.source_type_stratum,item.position_stratum,doc.task_prompt,doc.response_text].join('\n').toLocaleLowerCase();return status&&(!query||hay.includes(query))});index=Math.max(0,Math.min(index,Math.max(0,visible.length-1)));render()}
function renderProgress(){const locked=items.filter(item=>eventOf(record(item),'pre_reveal_judgment_locked')).length;text($("progress"),visible.length?`${index+1} / ${visible.length} visible · ${locked} / ${items.length} locked`:`0 visible · ${locked} / ${items.length} locked`)}
function render(){text($("packet"),packetId);if(!visible.length){text($("meta"),'No items match the current search/filter.');$("prompt").replaceChildren();$("response").replaceChildren();$("blind").hidden=true;$("revealed").hidden=true;renderProgress();return}const item=current(),doc=documents.get(item.document_id),rec=record(item),locked=eventOf(rec,'pre_reveal_judgment_locked'),reveal=eventOf(rec,'model_labels_revealed'),latest=corrections(rec).at(-1);text($("meta"),`${item.source_type_stratum} · ${item.position_stratum} · group ${item.group_position}/${item.group_size} · tokens [${item.token_span.join(', ')}) · core chars [${item.core_character_span.join(', ')}) · covering chars [${item.covering_character_span.join(', ')}) · response ${doc.tokenization.token_count} tokens / ${doc.response_character_count} code points · ${item.unit_id}`);text($("prompt"),doc.task_prompt);renderResponse(doc,item);$("blind").hidden=!!locked;$("revealed").hidden=!locked;if(locked){const models=$("models");models.replaceChildren(...item.revealed_decisions.map(modelCard));text($("lockedHuman"),`Locked human tag: ${locked.human_tag}${locked.human_notes?' · '+locked.human_notes:''}`);text($("revealMeta"),`Blind judgment locked ${locked.at}; decisions revealed ${reveal?.at||'unknown'}.`);correction.value=latest?.human_tag||'';correctionNotes.value=latest?.human_notes||''}else{human.value='';notes.value='';text($("lockError"),'')}$("jump").value=index+1;$("jump").max=visible.length;renderProgress()}
$("lock").onclick=()=>{const item=current(),rec=record(item);if(eventOf(rec,'pre_reveal_judgment_locked'))return;if(!human.value){text($("lockError"),'Select a tag before locking.');return}rec.events.push({event_type:'pre_reveal_judgment_locked',at:new Date().toISOString(),human_tag:human.value,human_notes:notes.value});rec.events.push({event_type:'model_labels_revealed',at:new Date().toISOString()});persist();applyFilter()};
$("saveCorrection").onclick=()=>{const item=current(),rec=record(item);if(!eventOf(rec,'pre_reveal_judgment_locked')||!correction.value)return;rec.events.push({event_type:'post_reveal_correction_recorded',at:new Date().toISOString(),human_tag:correction.value,human_notes:correctionNotes.value});persist();render()};
$("prev").onclick=()=>{index=Math.max(0,index-1);render()};$("next").onclick=()=>{index=Math.min(visible.length-1,index+1);render()};$("jump").onchange=e=>{index=Math.max(0,Math.min(visible.length-1,Number(e.target.value||1)-1));render()};$("search").oninput=applyFilter;$("filter").onchange=applyFilter;$("boundaries").onchange=render;
function exportRows(){const now=new Date().toISOString();return items.map(item=>{const rec=record(item),locked=eventOf(rec,'pre_reveal_judgment_locked'),reveal=eventOf(rec,'model_labels_revealed');return{schema_version:exportSchema,packet_id:packetId,ui_version:uiVersion,ui_template_sha256:uiTemplateSha256,item_id:item.item_id,item_sha256:item.item_sha256,document_id:item.document_id,document_sha256:item.document_sha256,unit_id:item.unit_id,pre_reveal_human_tag:locked?.human_tag||null,pre_reveal_human_notes:locked?.human_notes||'',pre_reveal_locked_at:locked?.at||null,model_decisions_revealed_at:reveal?.at||null,post_reveal_corrections:corrections(rec),event_history:rec.events,exported_at:now}})}
$("export").onclick=()=>{const out=exportRows(),blob=new Blob([out.map(x=>JSON.stringify(x)).join('\n')+'\n'],{type:'application/x-ndjson'}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=packetId+'-decisions.jsonl';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};
function validateImportedEvents(events){if(!Array.isArray(events))throw new Error('event_history must be an array');const allowed=new Set(['pre_reveal_judgment_locked','model_labels_revealed','post_reveal_correction_recorded']);if(events.some(e=>!e||!allowed.has(e.event_type)||typeof e.at!=='string'))throw new Error('unknown or malformed event');const locks=events.filter(e=>e.event_type==='pre_reveal_judgment_locked'),reveals=events.filter(e=>e.event_type==='model_labels_revealed'),corrections=events.filter(e=>e.event_type==='post_reveal_correction_recorded');if(locks.length>1||reveals.length>1||locks.some(e=>!tags.includes(e.human_tag))||corrections.some(e=>!tags.includes(e.human_tag))||locks.length!==reveals.length)throw new Error('invalid event history');if(locks.length){const lockIndex=events.indexOf(locks[0]),revealIndex=events.indexOf(reveals[0]);if(lockIndex!==0||revealIndex!==1||corrections.some(e=>events.indexOf(e)<=revealIndex))throw new Error('invalid blind/reveal event order')}else if(events.length)throw new Error('events without locked judgment')}
$("importButton").onclick=()=>$("importFile").click();$("importFile").onchange=async e=>{try{const content=await e.target.files[0].text(),rows=content.split(/\r?\n/).filter(Boolean).map(line=>JSON.parse(line)),byId=new Map(items.map(item=>[item.item_id,item]));if(rows.length!==items.length)throw new Error('import must contain exactly one row per review item');const incoming={};for(const row of rows){const item=byId.get(row.item_id);if(!item||row.schema_version!==exportSchema||row.packet_id!==packetId||row.ui_version!==uiVersion||row.ui_template_sha256!==uiTemplateSha256||row.item_sha256!==item.item_sha256||row.document_sha256!==item.document_sha256||row.unit_id!==item.unit_id||incoming[row.item_id])throw new Error('import identity mismatch');validateImportedEvents(row.event_history);const existing=saved.records[row.item_id]?.events||[],oldLock=existing.find(x=>x.event_type==='pre_reveal_judgment_locked'),newLock=row.event_history.find(x=>x.event_type==='pre_reveal_judgment_locked');if(oldLock&&JSON.stringify(oldLock)!==JSON.stringify(newLock))throw new Error('import conflicts with an already locked blind judgment');incoming[row.item_id]={events:row.event_history}}saved={...emptyState(),records:incoming};persist();applyFilter();text($("importStatus"),`Imported ${rows.length} review rows.`)}catch(error){text($("importStatus"),'Import rejected: '+error.message)}finally{e.target.value=''}};render();
</script></body></html>"""


def render_review_html(
    documents: Sequence[Mapping[str, Any]],
    items: Sequence[Mapping[str, Any]],
    packet_id: str,
) -> bytes:
    payload = json.dumps(
        {"documents": documents, "items": items},
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode()
    encoded = base64.b64encode(payload).decode("ascii")
    html = (
        _HTML.replace("__PAYLOAD_BASE64__", encoded)
        .replace("__TAGS_JSON__", json.dumps(list(COARSE_TAGS)))
        .replace("__PACKET_ID__", packet_id)
        .replace("__UI_VERSION__", UI_VERSION)
        .replace("__UI_TEMPLATE_SHA256__", hashlib.sha256(_HTML.encode()).hexdigest())
    )
    return html.encode()


def build_review_packet_v2(
    *,
    qualification_root: Path,
    run_root: Path,
    comparison_root: Path,
    destination: Path,
) -> dict[str, Any]:
    """Build and freeze a new full-response two-arm review packet."""

    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    documents, items, sources = assemble_review_payload(
        qualification_root=qualification_root.resolve(),
        run_root=run_root.resolve(),
        comparison_root=comparison_root.resolve(),
    )
    source_revision = _source_revision()
    packet_id = sources["packet_id"]
    temporary = destination.parent / f".{destination.name}.building-{uuid.uuid4().hex}"
    temporary.mkdir(parents=True)
    try:
        docs_path = temporary / "review-documents.jsonl"
        items_path = temporary / "review-items.jsonl"
        html_path = temporary / "review.html"
        atomic_write_jsonl(docs_path, documents)
        atomic_write_jsonl(items_path, items)
        atomic_write_bytes(html_path, render_review_html(documents, items, packet_id))
        files = [
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in (docs_path, html_path, items_path)
        ]
        inputs = sources["inputs"]
        comparison = sources["comparison"]
        manifest = {
            "schema_version": PACKET_SCHEMA,
            "status": "frozen_offline_full_response_blind_review_packet",
            "packet_id": packet_id,
            "ui_version": UI_VERSION,
            "ui_template_sha256": hashlib.sha256(_HTML.encode()).hexdigest(),
            "network_calls_made": 0,
            "claim_boundary": (
                "Human judgments are blind-first coarse-label reviews only. Revealed "
                "model decisions are not correctness, faithfulness, motif, or internal-"
                "computation ground truth."
            ),
            "environment": {
                "hostname": platform.node(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            },
            "counts": {
                "documents": len(documents),
                "review_items": len(items),
                "focal_units_per_group": 6,
                "items_with_six_available_decisions": sum(
                    all(value["available"] for value in item["revealed_decisions"])
                    for item in items
                ),
            },
            "qualification_root": str(qualification_root.resolve()),
            "qualification_manifest_sha256": inputs["qualification"]["manifest"][
                "manifest_sha256"
            ],
            "qualification_manifest_file_sha256": file_sha256(
                qualification_root / "manifest.json"
            ),
            "run_root": str(run_root.resolve()),
            "run_intent_sha256": inputs["run_intent"]["run_intent_sha256"],
            "collection_manifest_sha256": inputs["collection"][
                "collection_manifest_sha256"
            ],
            "collection_manifest_file_sha256": file_sha256(
                run_root / "collection-manifest.json"
            ),
            "run_events_jsonl_sha256": inputs["collection"]["events_jsonl_sha256"],
            "comparison_root": str(comparison_root.resolve()),
            "comparison_manifest_sha256": comparison["manifest"]["manifest_sha256"],
            "comparison_manifest_file_sha256": file_sha256(
                comparison_root / "manifest.json"
            ),
            "comparison_report_sha256": comparison["report"]["report_sha256"],
            "workstation_bundle": str(sources["workstation_path"].resolve()),
            "workstation_bundle_sha256": file_sha256(sources["workstation_path"]),
            "v1_units_path": str(sources["v1_units_path"].resolve()),
            "v1_units_sha256": file_sha256(sources["v1_units_path"]),
            "document_bindings_in_order": [
                {
                    "document_id": value["document_id"],
                    "document_sha256": value["document_sha256"],
                    "prompt_sha256": value["prompt_sha256"],
                    "response_text_sha256": value["response_text_sha256"],
                    "input_ids_sha256": value["tokenization"]["input_ids_sha256"],
                    "offset_mapping_sha256": value["tokenization"][
                        "offset_mapping_sha256"
                    ],
                }
                for value in documents
            ],
            "item_bindings_in_order": [
                {
                    "item_id": value["item_id"],
                    "item_sha256": value["item_sha256"],
                    "document_id": value["document_id"],
                    "unit_id": value["unit_id"],
                    "token_span": value["token_span"],
                    "core_character_span": value["core_character_span"],
                }
                for value in items
            ],
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
