"""Globally sealed blind human-review packet for coarse qualification v4."""

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

PACKET_SCHEMA = "adag.process-witness.coarse-human-review-packet.v4"
ITEM_SCHEMA = "adag.process-witness.coarse-human-review-item.v4"
EXPORT_SCHEMA = "adag.process-witness.coarse-human-review-decision.v4"
UI_VERSION = "process-witness-coarse-review-ui-v5-v2-layout-definitions"

BOUNDARY_DEFINITIONS = {
    "split_needed": "This unit combines distinct roles and should be split for a clean label.",
    "merge_previous": "Its role depends on text that belongs in the preceding unit.",
    "merge_next": "Its role depends on text that belongs in the following unit.",
    "meaning_unclear": "Even with full context, the unit's visible role remains unclear.",
}


def load_review_packet(root: Path) -> dict[str, Any]:
    """Validate the frozen blind packet without loading any model output."""

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    payload = dict(manifest)
    observed = payload.pop("manifest_sha256", None)
    if observed != canonical_sha256(payload):
        raise ValueError("coarse v4 review manifest self-hash drift")
    if (
        manifest.get("schema_version") != PACKET_SCHEMA
        or manifest.get("status") != "frozen_blind_global_seal_review_packet"
        or manifest.get("counts") != {"response_blocks": 15, "items": 24}
    ):
        raise ValueError("coarse v4 review manifest contract drift")
    required = {"documents.jsonl", "items.jsonl", "packet.json", "review.html"}
    if {item["path"] for item in manifest.get("files", [])} != required:
        raise ValueError("coarse v4 review file membership drift")
    for item in manifest["files"]:
        path = root / item["path"]
        if (
            not path.is_file()
            or path.stat().st_size != item["bytes"]
            or file_sha256(path) != item["sha256"]
        ):
            raise ValueError("coarse v4 review file drift")
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
            "tag_definitions",
            "boundary_definitions",
            "decision_precedence",
        )
    }
    if (
        packet.get("packet_binding_sha256") != canonical_sha256(identity)
        or packet.get("packet_id")
        != f"process-witness-coarse-review-v4-{canonical_sha256(identity)[:16]}"
        or packet.get("item_ids_in_order") != [item["item_id"] for item in items]
        or len(items) != 24
        or len(documents) != 15
        or len({item["unit_id"] for item in items}) != 24
        or "reveal" in packet
    ):
        raise ValueError("coarse v4 review packet identity drift")
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
            item_id = f"pwcoarsereviewv4-{canonical_sha256(identity)[:32]}"
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
        len(review_documents) != 15
        or len(items) != 24
        or len({item["unit_id"] for item in items}) != 24
        or len({document["response_id"] for document in review_documents}) != 15
    ):
        raise ValueError("coarse v4 review payload cardinality drift")
    packet_identity = {
        "schema_version": PACKET_SCHEMA,
        "qualification_manifest_sha256": qualification["manifest"]["manifest_sha256"],
        "ui_version": UI_VERSION,
        "item_ids_in_order": [item["item_id"] for item in items],
        "tag_definitions": dict(qualification["config"]["tags"]),
        "boundary_definitions": BOUNDARY_DEFINITIONS,
        "decision_precedence": list(qualification["config"]["decision_precedence"]),
    }
    if set(packet_identity["tag_definitions"]) != set(COARSE_TAGS):
        raise ValueError("coarse v4 review tag definitions drift")
    if set(packet_identity["boundary_definitions"]) != set(BOUNDARY_CONCERNS):
        raise ValueError("coarse v4 review boundary definitions drift")
    if len(packet_identity["decision_precedence"]) != len(COARSE_TAGS) or set(
        packet_identity["decision_precedence"]
    ) != set(COARSE_TAGS):
        raise ValueError("coarse v4 review decision precedence drift")
    packet_binding_sha256 = canonical_sha256(packet_identity)
    return {
        "packet": {
            **packet_identity,
            "packet_id": f"process-witness-coarse-review-v4-{packet_binding_sha256[:16]}",
            "packet_binding_sha256": packet_binding_sha256,
            "status": "blind_unsealed_no_model_outputs_present",
            "counts": {"response_blocks": 15, "items": 24},
            "global_seal_required_before_model_reveal": True,
            "model_votes_are_not_human_ground_truth": True,
        },
        "documents": review_documents,
        "items": items,
    }


def render_review_html(payload: Mapping[str, Any]) -> str:
    """Render a self-contained v2-style UI with one irreversible global seal."""

    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    tags = json.dumps(list(COARSE_TAGS))
    concerns = json.dumps(list(BOUNDARY_CONCERNS))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Coarse v4 globally sealed blind review</title>
<style>
:root{{color-scheme:light;--ink:#17211f;--muted:#66716e;--line:#d7deda;--paper:#f4f6f3;--card:#fff;--accent:#126a55;--group:#ffedb0;--focus:#9ee4cf;--boundary:#dce5e1;--warn:#a24418}}
*{{box-sizing:border-box}}html,body{{margin:0;min-height:100%;font:15px/1.5 system-ui,sans-serif;background:var(--paper);color:var(--ink)}}button,input,select,textarea{{font:inherit}}button,select,input{{border:1px solid #aab5b1;border-radius:6px;background:#fff;padding:7px 9px}}button{{cursor:pointer}}button:disabled{{opacity:.5;cursor:not-allowed}}
header{{position:sticky;top:0;z-index:5;background:#18372f;color:#fff;padding:10px 16px;display:flex;gap:8px;align-items:center;flex-wrap:wrap}}header strong{{margin-right:auto}}.toolbar{{display:flex;gap:7px;align-items:center;flex-wrap:wrap}}.toolbar select{{width:auto}}.progress{{font-variant-numeric:tabular-nums}}.search{{width:220px}}
main{{max-width:1800px;margin:16px auto;padding:0 16px;display:grid;grid-template-columns:minmax(0,1fr) 320px 380px;gap:15px}}.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:15px}}.meta{{color:var(--muted);font-size:13px}}.prompt,.response{{white-space:pre-wrap;word-break:break-word;overflow:auto;background:#f7f8f6;border:1px solid #e1e5e2;border-radius:7px;padding:12px}}.prompt{{max-height:230px}}.response{{max-height:62vh;line-height:1.65}}.group{{background:var(--group);border-radius:2px}}.focus{{background:var(--focus);outline:2px solid var(--accent);scroll-margin:160px 0}}.legend{{display:flex;gap:16px;margin:8px 0}}.swatch{{display:inline-block;width:14px;height:14px;border-radius:3px;vertical-align:-2px;margin-right:5px}}.swatch.group{{background:var(--group)}}.swatch.focus{{background:var(--focus);outline:1px solid var(--accent)}}
.side{{position:sticky;align-self:start;top:var(--header-offset,112px);height:max-content;max-height:calc(100vh - var(--header-offset,112px) - 18px);overflow:auto}}label{{display:block;font-weight:650;margin:12px 0 5px}}textarea{{width:100%;min-height:100px;border:1px solid #aab5b1;border-radius:6px;padding:8px}}select{{width:100%}}.checks label{{display:flex;gap:8px;margin:5px 0;font-weight:400}}.actions{{display:flex;gap:8px;flex-wrap:wrap;margin-top:12px}}.warn{{color:var(--warn);font-weight:650}}.claim{{font-size:12px;color:var(--muted);margin-top:13px}}.definitions dl{{margin:0}}.definitions dt{{font-weight:750;color:var(--accent);margin-top:12px}}.definitions dd{{margin:3px 0 0;color:#35413d}}.definitions h4{{margin:18px 0 5px}}@media(max-width:1120px){{main{{grid-template-columns:minmax(0,1fr) 280px 300px;gap:10px;padding-left:8px;padding-right:8px}}.card{{padding:12px}}}}@media(max-width:820px){{main{{grid-template-columns:1fr}}.side,.definitions{{position:static;max-height:none;grid-column:auto}}header{{position:static}}}}
</style></head><body>
<header><strong>Refined coarse qualification · globally blind review</strong><span id="packet"></span><span class="progress" id="progress"></span><div class="toolbar"><button id="prev">Previous</button><input id="jump" type="number" min="1" style="width:72px"><button id="next">Next</button><input id="search" class="search" type="search" placeholder="Search prompt, response, unit ID"><select id="filter"><option value="all">All items</option><option value="unreviewed">Unreviewed</option><option value="reviewed">Reviewed</option></select><button id="progressExport">Export progress</button><button id="importButton">Import progress</button><input id="import" type="file" accept=".json,application/json" hidden><button id="decisionsExport" disabled>Export sealed JSONL</button></div></header>
<main><section><div class="card"><div class="meta" id="meta"></div><div class="legend"><span><i class="swatch group"></i>other focal units in this six-unit group</span><span><i class="swatch focus"></i>current review unit</span></div><h3>Complete task prompt</h3><div class="prompt" id="prompt"></div><h3>Complete exact response</h3><div class="response" id="response"></div></div></section>
<aside class="card side" id="blind"><h3>Blind human judgment</h3><p class="meta">Model outputs are absent from this page. The page is populated automatically; Import progress is only for resuming an earlier export. Save all 24 judgments before the global seal.</p><label>Primary label<select id="primary"><option value="">Choose…</option></select></label><fieldset class="checks"><legend>Optional defensible alternatives</legend><div id="alternatives"></div></fieldset><fieldset class="checks"><legend>Boundary concerns</legend><div id="boundaries"></div></fieldset><label>Note<textarea id="note" placeholder="Optional rationale or ambiguity note"></textarea></label><div class="actions"><button id="save">Save item</button><button id="saveNext">Save &amp; next</button></div><p id="saveStatus" class="meta"></p><hr><p><strong>Global seal:</strong> irreversible after all 24 primary labels are saved.</p><button id="seal" disabled>Seal all 24 blind decisions</button><p class="warn" id="sealStatus"></p><p class="claim">These are coarse trace-sampling labels, not correctness, faithfulness, motif, or computation judgments.</p></aside>
<aside class="card side definitions"><h3>Label reference</h3><p class="meta" id="labelReferenceIntro"></p><dl id="tagDefinitions"></dl><h4>Boundary concerns</h4><dl id="boundaryDefinitions"></dl></aside></main>
<script>
const DATA={data},TAGS={tags},CONCERNS={concerns};const packet=DATA.packet,docs=new Map(DATA.documents.map(d=>[d.response_id,d]));
const key=`coarse-review-v4:${{packet.packet_id}}:${{packet.ui_version}}`;let state={{index:0,sealed:false,sealed_at:null,decisions:{{}}}},visible=[...DATA.items.keys()];
function validDecision(d,item){{return d&&d.unit_id===item.unit_id&&TAGS.includes(d.primary_label)&&Array.isArray(d.defensible_alternatives)&&d.defensible_alternatives.every(v=>TAGS.includes(v)&&v!==d.primary_label)&&Array.isArray(d.boundary_concerns)&&d.boundary_concerns.every(v=>CONCERNS.includes(v))&&typeof d.note==='string'}}
function decisionsValid(s){{if(!s||!Number.isInteger(s.index)||s.index<0||s.index>=24||typeof s.decisions!=='object')return false;for(const [id,d] of Object.entries(s.decisions)){{const item=DATA.items.find(x=>x.item_id===id);if(!item||!validDecision(d,item))return false}}return true}}
function completeDecisions(s){{return decisionsValid(s)&&Object.keys(s.decisions).length===24&&DATA.items.every(item=>validDecision(s.decisions[item.item_id],item))}}
function validUnsealedState(s){{return decisionsValid(s)&&!s.sealed}}
function validSealedState(s){{return completeDecisions(s)&&s.sealed===true&&typeof s.global_seal_id==='string'&&s.global_seal_id.length>0&&typeof s.sealed_at==='string'&&s.sealed_at.length>0}}
try{{const prior=JSON.parse(localStorage.getItem(key));if(prior&&prior.packet_id===packet.packet_id&&(validUnsealedState(prior.state)||validSealedState(prior.state)))state=prior.state}}catch(e){{}}
const el=id=>document.getElementById(id);el('packet').textContent=packet.packet_id;for(const tag of TAGS){{const o=document.createElement('option');o.value=tag;o.textContent=tag;el('primary').append(o)}}
function checks(root,values,name){{root.textContent='';for(const value of values){{const l=document.createElement('label'),i=document.createElement('input');i.type='checkbox';i.value=value;i.name=name;l.append(i,document.createTextNode(value));root.append(l)}}}}
checks(el('alternatives'),TAGS,'alt');checks(el('boundaries'),CONCERNS,'boundary');
function definitions(root,values){{root.textContent='';for(const [name,description] of Object.entries(values)){{const term=document.createElement('dt'),detail=document.createElement('dd');term.textContent=name;detail.textContent=description;root.append(term,detail)}}}}
definitions(el('tagDefinitions'),packet.tag_definitions);definitions(el('boundaryDefinitions'),packet.boundary_definitions);
el('labelReferenceIntro').textContent=`Choose the unit's primary visible trajectory effect. Apply precedence: ${{packet.decision_precedence.join(' → ')}}. Use uncertain only when the tie or boundary remains genuine.`;
function persist(){{localStorage.setItem(key,JSON.stringify({{packet_id:packet.packet_id,state}}))}}
function updateHeaderOffset(){{document.documentElement.style.setProperty('--header-offset',`${{document.querySelector('header').offsetHeight+12}}px`)}}
window.addEventListener('resize',updateHeaderOffset);new ResizeObserver(updateHeaderOffset).observe(document.querySelector('header'));updateHeaderOffset();
function renderResponse(item,doc){{const root=el('response');root.textContent='';const spans=[...item.group_spans].sort((a,b)=>a.core_character_span[0]-b.core_character_span[0]);const chars=Array.from(doc.response);let cursor=0;for(const s of spans){{const [a,b]=s.core_character_span;root.append(document.createTextNode(chars.slice(cursor,a).join('')));const mark=document.createElement('span');mark.className='group'+(s.unit_id===item.unit_id?' focus':'');mark.textContent=chars.slice(a,b).join('');root.append(mark);cursor=b}}root.append(document.createTextNode(chars.slice(cursor).join('')));setTimeout(()=>{{const focus=root.querySelector('.focus');if(focus){{const rootRect=root.getBoundingClientRect(),focusRect=focus.getBoundingClientRect();root.scrollTop=Math.max(0,root.scrollTop+focusRect.top-rootRect.top-(root.clientHeight-focusRect.height)/2)}}}},0)}}
function current(){{return visible.includes(state.index)?DATA.items[state.index]:null}}
function visiblePosition(){{return visible.indexOf(state.index)}}
function refreshVisible(preferredPosition=0){{const query=el('search').value.trim().toLocaleLowerCase(),mode=el('filter').value;visible=DATA.items.map((_,i)=>i).filter(i=>{{const item=DATA.items[i],doc=docs.get(item.response_id),reviewed=!!state.decisions[item.item_id],status=mode==='all'||(mode==='reviewed'&&reviewed)||(mode==='unreviewed'&&!reviewed),hay=[item.unit_id,item.unit_text,doc.prompt,doc.response].join('\\n').toLocaleLowerCase();return status&&(!query||hay.includes(query))}});if(visible.length&&!visible.includes(state.index))state.index=visible[Math.min(Math.max(0,preferredPosition),visible.length-1)]}}
function applyFilter(){{const position=Math.max(0,visiblePosition());refreshVisible(position);persist();render()}}
function render(){{el('saveStatus').textContent='';const position=visiblePosition();el('prev').disabled=state.sealed||position<=0;el('next').disabled=state.sealed||position<0||position>=visible.length-1;el('jump').disabled=state.sealed||!visible.length;el('importButton').disabled=state.sealed;el('progressExport').disabled=state.sealed;if(!visible.length){{el('meta').textContent='No items match the current search/filter.';el('prompt').textContent='';el('response').textContent='';el('progress').textContent=`0 visible · saved ${{Object.keys(state.decisions).length}} / 24`;for(const control of [el('primary'),el('note'),el('save'),el('saveNext')])control.disabled=true;return}}const item=current(),doc=docs.get(item.response_id),decision=state.decisions[item.item_id]||{{primary_label:'',defensible_alternatives:[],boundary_concerns:[],note:''}};el('meta').textContent=`response block ${{item.randomized_response_block_index+1}} / 15 · unit ${{item.within_response_index+1}} / ${{item.group_spans.length}} · chars [${{item.core_character_span.join(', ')}}) · ${{item.unit_id}}`;el('prompt').textContent=doc.prompt;renderResponse(item,doc);el('progress').textContent=`${{position+1}} / ${{visible.length}} visible · saved ${{Object.keys(state.decisions).length}} / 24`;el('jump').value=position+1;el('jump').max=visible.length;el('primary').value=decision.primary_label;for(const i of document.querySelectorAll('input[name=alt]')){{i.checked=decision.defensible_alternatives.includes(i.value);i.disabled=state.sealed||i.value===decision.primary_label}}for(const i of document.querySelectorAll('input[name=boundary]')){{i.checked=decision.boundary_concerns.includes(i.value);i.disabled=state.sealed}}el('note').value=decision.note;for(const i of [el('primary'),el('note'),el('save'),el('saveNext')])i.disabled=state.sealed;el('seal').disabled=state.sealed||!completeDecisions(state);el('decisionsExport').disabled=!state.sealed;el('sealStatus').textContent=state.sealed?'This browser profile is sealed. You may re-export the same immutable ledger.':'';}}
el('primary').onchange=()=>{{for(const i of document.querySelectorAll('input[name=alt]')){{if(i.value===el('primary').value)i.checked=false;i.disabled=i.value===el('primary').value}}}};
function saveCurrent(advance){{const item=current(),primary=el('primary').value;if(!item)return false;if(!primary){{el('saveStatus').textContent='Choose a primary label.';return false}}const originalIndex=state.index,position=Math.max(0,visiblePosition());state.decisions[item.item_id]={{unit_id:item.unit_id,primary_label:primary,defensible_alternatives:[...document.querySelectorAll('input[name=alt]:checked')].map(i=>i.value).filter(v=>v!==primary),boundary_concerns:[...document.querySelectorAll('input[name=boundary]:checked')].map(i=>i.value),note:el('note').value}};refreshVisible(position);if(advance&&visible.includes(originalIndex)){{const nextPosition=visible.indexOf(originalIndex)+1;if(nextPosition<visible.length)state.index=visible[nextPosition]}}persist();render();el('saveStatus').textContent='Saved.';return true}}
el('save').onclick=()=>saveCurrent(false);el('saveNext').onclick=()=>saveCurrent(true);
el('prev').onclick=()=>{{const position=visiblePosition();if(position>0)state.index=visible[position-1];persist();render()}};el('next').onclick=()=>{{const position=visiblePosition();if(position>=0&&position<visible.length-1)state.index=visible[position+1];persist();render()}};el('jump').onchange=()=>{{if(!visible.length)return;const position=Math.max(0,Math.min(visible.length-1,Number(el('jump').value)-1));state.index=visible[position];persist();render()}};el('search').oninput=applyFilter;el('filter').onchange=applyFilter;
function download(name,text,type){{const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([text],{{type}}));a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}}
function exportSealed(){{if(!state.sealed)return;const rows=DATA.items.map(item=>JSON.stringify({{schema_version:'{EXPORT_SCHEMA}',packet_id:packet.packet_id,packet_binding_sha256:packet.packet_binding_sha256,item_id:item.item_id,unit_id:item.unit_id,...state.decisions[item.item_id],globally_sealed:true,global_seal_id:state.global_seal_id,global_sealed_at:state.sealed_at}}));download(packet.packet_id+'-decisions.jsonl',rows.join('\\n')+'\\n','application/x-ndjson')}}
el('seal').onclick=()=>{{if(!completeDecisions(state))return;if(!confirm('Seal all 24 blind decisions? Judgments can no longer be edited in this browser profile.'))return;state.sealed=true;state.sealed_at=new Date().toISOString();state.global_seal_id=crypto.randomUUID();persist();render();exportSealed();el('sealStatus').textContent='Sealed ledger downloaded and the sealed state retained locally. Keep that file for the later reveal build.'}};
el('progressExport').onclick=()=>download(packet.packet_id+'-progress.json',JSON.stringify({{packet_id:packet.packet_id,state}},null,2),'application/json');
el('decisionsExport').onclick=exportSealed;
el('importButton').onclick=()=>el('import').click();el('import').onchange=async e=>{{const file=e.target.files?.[0];if(!file)return;const text=await file.text();let value;try{{value=JSON.parse(text)}}catch(err){{alert('Progress import must be the JSON progress file, not a sealed JSONL ledger.');e.target.value='';return}}if(value.packet_id!==packet.packet_id||!validUnsealedState(value.state)){{alert('Invalid, sealed, or mismatched progress state');e.target.value='';return}}state=value.state;el('search').value='';el('filter').value='all';visible=[...DATA.items.keys()];persist();render();e.target.value=''}};render();
</script></body></html>"""
