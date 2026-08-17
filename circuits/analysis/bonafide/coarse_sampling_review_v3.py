"""Globally sealed blind human-review packet for coarse qualification v3."""

from __future__ import annotations

import json
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from circuits.analysis.bonafide.canonical import canonical_sha256, file_sha256
from circuits.analysis.bonafide.coarse_sampling_annotation import (
    BOUNDARY_CONCERNS,
    COARSE_TAGS,
)
from circuits.labeling.io import read_jsonl

PACKET_SCHEMA = "adag.process-witness.coarse-human-review-packet.v3"
ITEM_SCHEMA = "adag.process-witness.coarse-human-review-item.v3"
EXPORT_SCHEMA = "adag.process-witness.coarse-human-review-decision.v3"
UI_VERSION = "process-witness-coarse-review-ui-v4-global-seal"


def load_review_packet(root: Path) -> dict[str, Any]:
    """Validate the frozen blind packet without loading any model output."""

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    payload = dict(manifest)
    observed = payload.pop("manifest_sha256", None)
    if observed != canonical_sha256(payload):
        raise ValueError("coarse v3 review manifest self-hash drift")
    if (
        manifest.get("schema_version") != PACKET_SCHEMA
        or manifest.get("status") != "frozen_blind_global_seal_review_packet"
        or manifest.get("counts") != {"response_blocks": 24, "items": 144}
    ):
        raise ValueError("coarse v3 review manifest contract drift")
    required = {"documents.jsonl", "items.jsonl", "packet.json", "review.html"}
    if {item["path"] for item in manifest.get("files", [])} != required:
        raise ValueError("coarse v3 review file membership drift")
    for item in manifest["files"]:
        path = root / item["path"]
        if (
            not path.is_file()
            or path.stat().st_size != item["bytes"]
            or file_sha256(path) != item["sha256"]
        ):
            raise ValueError("coarse v3 review file drift")
    packet = json.loads((root / "packet.json").read_text(encoding="utf-8"))
    items = read_jsonl(root / "items.jsonl")
    documents = read_jsonl(root / "documents.jsonl")
    identity = {
        key: packet[key]
        for key in (
            "schema_version",
            "qualification_manifest_sha256",
            "ui_version",
            "item_ids_in_order",
        )
    }
    if (
        packet.get("packet_binding_sha256") != canonical_sha256(identity)
        or packet.get("packet_id")
        != f"process-witness-coarse-review-v3-{canonical_sha256(identity)[:16]}"
        or packet.get("item_ids_in_order") != [item["item_id"] for item in items]
        or len(items) != 144
        or len(documents) != 24
        or len({item["unit_id"] for item in items}) != 144
        or "reveal" in packet
    ):
        raise ValueError("coarse v3 review packet identity drift")
    return {
        "manifest": manifest,
        "packet": packet,
        "items": items,
        "documents": documents,
    }


def build_review_payload(
    *,
    qualification: Mapping[str, Any],
    workstation_bundle: Mapping[str, Any],
) -> dict[str, Any]:
    """Randomize response blocks without placing model outputs in the packet."""

    documents = {
        document["response_id"]: document
        for document in workstation_bundle.get("documents", [])
    }
    units = {unit["unit_id"]: unit for unit in qualification["focal_units"]}
    windows = list(qualification["windows"])
    rng = random.Random(
        qualification["config"]["human_review"]["response_block_randomization_seed"]
    )
    rng.shuffle(windows)
    review_documents = []
    items = []
    for block_index, window in enumerate(windows):
        document = documents[window["response_id"]]
        review_documents.append(
            {
                "response_id": document["response_id"],
                "prompt": document["task_context"]["prompt"],
                "prompt_sha256": document["prompt_sha256"],
                "response": document["text"],
                "response_sha256": document["text_sha256"],
                "focal_unit_ids": window["focal_unit_ids"],
            }
        )
        group_units = sorted(
            (units[unit_id] for unit_id in window["focal_unit_ids"]),
            key=lambda unit: unit["sequence_index"],
        )
        for within_block_index, unit in enumerate(group_units):
            identity = {
                "schema_version": ITEM_SCHEMA,
                "qualification_manifest_sha256": qualification["manifest"][
                    "manifest_sha256"
                ],
                "response_id": document["response_id"],
                "unit_id": unit["unit_id"],
                "randomized_response_block_index": block_index,
                "within_response_index": within_block_index,
            }
            item_id = f"pwcoarsereviewv3-{canonical_sha256(identity)[:32]}"
            items.append(
                {
                    **identity,
                    "item_id": item_id,
                    "unit_text": unit["text"],
                    "core_character_span": unit["core_character_span"],
                    "group_spans": [
                        {
                            "unit_id": other["unit_id"],
                            "core_character_span": other["core_character_span"],
                        }
                        for other in group_units
                    ],
                }
            )
    if (
        len(review_documents) != 24
        or len(items) != 144
        or len({item["unit_id"] for item in items}) != 144
        or len({document["response_id"] for document in review_documents}) != 24
    ):
        raise ValueError("coarse v3 review payload cardinality drift")
    packet_identity = {
        "schema_version": PACKET_SCHEMA,
        "qualification_manifest_sha256": qualification["manifest"]["manifest_sha256"],
        "ui_version": UI_VERSION,
        "item_ids_in_order": [item["item_id"] for item in items],
    }
    packet_binding_sha256 = canonical_sha256(packet_identity)
    return {
        "packet": {
            **packet_identity,
            "packet_id": f"process-witness-coarse-review-v3-{packet_binding_sha256[:16]}",
            "packet_binding_sha256": packet_binding_sha256,
            "status": "blind_unsealed_no_model_outputs_present",
            "counts": {"response_blocks": 24, "items": 144},
            "global_seal_required_before_model_reveal": True,
            "model_votes_are_not_human_ground_truth": True,
        },
        "documents": review_documents,
        "items": items,
    }


def render_review_html(payload: Mapping[str, Any]) -> str:
    """Render a self-contained review UI with one irreversible global reveal."""

    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    tags = json.dumps(list(COARSE_TAGS))
    concerns = json.dumps(list(BOUNDARY_CONCERNS))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Coarse v3 globally sealed blind review</title>
<style>
:root{{--bg:#f5f3ed;--panel:#fff;--ink:#17231f;--muted:#66736e;--line:#d8dedb;--accent:#087765;--group:#fff0bd;--focus:#ffb84d}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,sans-serif}}
header{{position:sticky;top:0;z-index:3;background:#18352d;color:#fff;padding:12px 18px;display:flex;gap:12px;align-items:center;flex-wrap:wrap}}
header button,header input{{font:inherit;padding:7px 10px}}header .spacer{{flex:1}}main{{max-width:1500px;margin:auto;padding:16px;display:grid;grid-template-columns:minmax(0,1fr) 330px;gap:16px}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px}}pre{{white-space:pre-wrap;word-break:break-word;font:15px/1.55 ui-monospace,monospace;max-height:36vh;overflow:auto;background:#f7f9f8;padding:14px;border-radius:9px}}
#response{{max-height:52vh}}.group{{background:var(--group);border-radius:3px}}.focus{{background:var(--focus);outline:2px solid #8d5400}}label{{display:block;margin:9px 0}}select,textarea{{width:100%;font:inherit;padding:8px}}textarea{{min-height:90px}}.checks label{{display:flex;gap:8px;margin:5px 0}}button{{cursor:pointer}}button:disabled{{cursor:not-allowed;opacity:.5}}#reveal{{border-color:#b78300;background:#fffaf0}}.hidden{{display:none}}.vote{{border-left:4px solid var(--accent);padding:8px 10px;margin:8px 0;background:#f0f8f5}}.warn{{color:#8b3d00}}@media(max-width:850px){{main{{grid-template-columns:1fr}}header{{position:static}}}}
</style></head><body>
<header><strong id="packet"></strong><button id="prev">Previous</button><button id="next">Next</button><input id="jump" type="number" min="1"><span id="progress"></span><span class="spacer"></span><button id="progressExport">Export progress</button><label>Import <input id="import" type="file" accept=".json,.jsonl"></label><button id="decisionsExport" disabled>Export sealed JSONL</button></header>
<main><section><div class="card"><h2>Task prompt</h2><pre id="prompt"></pre></div><div class="card"><h2>Complete response</h2><pre id="response"></pre></div></section>
<aside><div class="card" id="blind"><h2>Blind judgment</h2><label>Primary label<select id="primary"><option value="">Choose…</option></select></label><fieldset class="checks"><legend>Optional defensible alternatives</legend><div id="alternatives"></div></fieldset><fieldset class="checks"><legend>Boundary concerns</legend><div id="boundaries"></div></fieldset><label>Note<textarea id="note"></textarea></label><button id="save">Save item</button><p id="saveStatus"></p><hr><p><strong>Global seal:</strong> no model outputs are present in this packet. Seal all 144 primary labels before building a separate reveal artifact.</p><button id="seal" disabled>Seal all 144 blind decisions</button><p class="warn" id="sealStatus"></p></div>
</aside></main>
<script>
const DATA={data},TAGS={tags},CONCERNS={concerns};const packet=DATA.packet,docs=new Map(DATA.documents.map(d=>[d.response_id,d]));
const key=`coarse-review-v3:${{packet.packet_id}}:${{packet.ui_version}}`;let state={{index:0,sealed:false,sealed_at:null,decisions:{{}}}};
function validDecision(d,item){{return d&&d.unit_id===item.unit_id&&TAGS.includes(d.primary_label)&&Array.isArray(d.defensible_alternatives)&&d.defensible_alternatives.every(v=>TAGS.includes(v)&&v!==d.primary_label)&&Array.isArray(d.boundary_concerns)&&d.boundary_concerns.every(v=>CONCERNS.includes(v))&&typeof d.note==='string'}}
function validUnsealedState(s){{if(!s||s.sealed||!Number.isInteger(s.index)||s.index<0||s.index>=144||typeof s.decisions!=='object')return false;for(const [id,d] of Object.entries(s.decisions)){{const item=DATA.items.find(x=>x.item_id===id);if(!item||!validDecision(d,item))return false}}return true}}
try{{const prior=JSON.parse(localStorage.getItem(key));if(prior&&prior.packet_id===packet.packet_id&&validUnsealedState(prior.state))state=prior.state}}catch(e){{}}
const el=id=>document.getElementById(id);el('packet').textContent=packet.packet_id;for(const tag of TAGS){{const o=document.createElement('option');o.value=tag;o.textContent=tag;el('primary').append(o)}}
function checks(root,values,name){{root.textContent='';for(const value of values){{const l=document.createElement('label'),i=document.createElement('input');i.type='checkbox';i.value=value;i.name=name;l.append(i,document.createTextNode(value));root.append(l)}}}}
checks(el('alternatives'),TAGS,'alt');checks(el('boundaries'),CONCERNS,'boundary');
function persist(){{localStorage.setItem(key,JSON.stringify({{packet_id:packet.packet_id,state}}))}}
function renderResponse(item,doc){{const root=el('response');root.textContent='';const spans=[...item.group_spans].sort((a,b)=>a.core_character_span[0]-b.core_character_span[0]);const chars=Array.from(doc.response);let cursor=0;for(const s of spans){{const [a,b]=s.core_character_span;root.append(document.createTextNode(chars.slice(cursor,a).join('')));const mark=document.createElement('span');mark.className='group'+(s.unit_id===item.unit_id?' focus':'');mark.textContent=chars.slice(a,b).join('');root.append(mark);cursor=b}}root.append(document.createTextNode(chars.slice(cursor).join('')));setTimeout(()=>root.querySelector('.focus')?.scrollIntoView({{block:'center'}}),0)}}
function render(){{const item=DATA.items[state.index],doc=docs.get(item.response_id),decision=state.decisions[item.item_id]||{{primary_label:'',defensible_alternatives:[],boundary_concerns:[],note:''}};el('prompt').textContent=doc.prompt;renderResponse(item,doc);el('progress').textContent=`${{state.index+1}} / 144 · saved ${{Object.keys(state.decisions).length}}`;el('jump').value=state.index+1;el('primary').value=decision.primary_label;for(const i of document.querySelectorAll('input[name=alt]')){{i.checked=decision.defensible_alternatives.includes(i.value);i.disabled=state.sealed||i.value===decision.primary_label}}for(const i of document.querySelectorAll('input[name=boundary]')){{i.checked=decision.boundary_concerns.includes(i.value);i.disabled=state.sealed}}el('note').value=decision.note;for(const i of [el('primary'),el('note'),el('save')])i.disabled=state.sealed;const complete=Object.keys(state.decisions).length===144&&DATA.items.every(item=>validDecision(state.decisions[item.item_id],item));el('seal').disabled=state.sealed||!complete;el('decisionsExport').disabled=!state.sealed;}}
el('primary').onchange=()=>{{for(const i of document.querySelectorAll('input[name=alt]')){{if(i.value===el('primary').value)i.checked=false;i.disabled=i.value===el('primary').value}}}};el('save').onclick=()=>{{const item=DATA.items[state.index],primary=el('primary').value;if(!primary){{el('saveStatus').textContent='Choose a primary label.';return}}state.decisions[item.item_id]={{unit_id:item.unit_id,primary_label:primary,defensible_alternatives:[...document.querySelectorAll('input[name=alt]:checked')].map(i=>i.value).filter(v=>v!==primary),boundary_concerns:[...document.querySelectorAll('input[name=boundary]:checked')].map(i=>i.value),note:el('note').value}};persist();el('saveStatus').textContent='Saved.';render()}};
el('prev').onclick=()=>{{state.index=(state.index+143)%144;persist();render()}};el('next').onclick=()=>{{state.index=(state.index+1)%144;persist();render()}};el('jump').onchange=()=>{{state.index=Math.max(0,Math.min(143,Number(el('jump').value)-1));persist();render()}};
function download(name,text,type){{const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([text],{{type}}));a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}}
function exportSealed(){{if(!state.sealed)return;const rows=DATA.items.map(item=>JSON.stringify({{schema_version:'{EXPORT_SCHEMA}',packet_id:packet.packet_id,packet_binding_sha256:packet.packet_binding_sha256,item_id:item.item_id,unit_id:item.unit_id,...state.decisions[item.item_id],globally_sealed:true,global_seal_id:state.global_seal_id,global_sealed_at:state.sealed_at}}));download(packet.packet_id+'-decisions.jsonl',rows.join('\n')+'\n','application/x-ndjson')}}
el('seal').onclick=()=>{{if(Object.keys(state.decisions).length!==144||!DATA.items.every(item=>validDecision(state.decisions[item.item_id],item)))return;if(!confirm('Seal all 144 blind decisions? Judgments can no longer be edited.'))return;state.sealed=true;state.sealed_at=new Date().toISOString();state.global_seal_id=crypto.randomUUID();localStorage.removeItem(key);render();exportSealed();el('sealStatus').textContent='Sealed ledger downloaded. Keep that file for the later reveal build.'}};
el('progressExport').onclick=()=>download(packet.packet_id+'-progress.json',JSON.stringify({{packet_id:packet.packet_id,state}},null,2),'application/json');
el('decisionsExport').onclick=exportSealed;
el('import').onchange=async e=>{{const text=await e.target.files[0].text();let value;try{{value=JSON.parse(text)}}catch(err){{alert('Progress import must be the JSON progress file, not a sealed JSONL ledger.');return}}if(value.packet_id!==packet.packet_id||!validUnsealedState(value.state)){{alert('Invalid, sealed, or mismatched progress state');return}}state=value.state;persist();render()}};render();
</script></body></html>"""
