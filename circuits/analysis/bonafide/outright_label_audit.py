"""Build a compact, self-contained audit page for outright source labels.

This is a presentation-only derivative of the provenance-bound v2 review packet.
It deliberately removes tokenizer and tracing-target data while retaining the
source completion, answers, and annotations needed to audit label correctness.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
import os
import re
import shutil
import uuid
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from circuits.analysis.bonafide.canonical import canonical_json, canonical_sha256
from circuits.analysis.bonafide.outright_target_review import (
    SCHEMA_VERSION as TARGET_REVIEW_SCHEMA_VERSION,
)

SCHEMA_VERSION = "adag.raw-graph-observatory.label-audit.v1"
DEFAULT_QWEN_MODEL = "Qwen/Qwen3-4B-Thinking-2507"
_PAYLOAD_PATTERN = re.compile(rb'atob\("([A-Za-z0-9+/=]+)"\)')


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _atomic_write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"destination already exists: {path}")
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        with temporary.open("xb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_write_json(path: Path, value: object) -> None:
    _atomic_write_bytes(
        path,
        (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(),
    )


def load_target_review_packet(source: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load and verify the v2 review manifest and embedded canonical payload."""

    manifest_path = source / "manifest.json"
    page_path = source / "review.html"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        page = page_path.read_bytes()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"unreadable source review packet: {source}") from error
    if manifest.get("schema_version") != TARGET_REVIEW_SCHEMA_VERSION:
        raise ValueError("source review schema drift")
    expected_page = manifest.get("files", {}).get("review.html", {})
    if expected_page.get("bytes") != len(page) or expected_page.get(
        "sha256"
    ) != _sha256(page):
        raise ValueError("source review page hash or size drift")
    match = _PAYLOAD_PATTERN.search(page)
    if match is None:
        raise ValueError("source review page lacks an embedded payload")
    try:
        payload_bytes = gzip.decompress(base64.b64decode(match.group(1), validate=True))
        payload = json.loads(payload_bytes)
    except (ValueError, OSError, json.JSONDecodeError) as error:
        raise ValueError("source review payload is invalid") from error
    if _sha256(payload_bytes) != manifest.get("payload_sha256"):
        raise ValueError("source review payload hash drift")
    if canonical_json(payload) != payload_bytes:
        raise ValueError("source review payload is not canonical JSON")
    if payload.get("schemaVersion") != TARGET_REVIEW_SCHEMA_VERSION:
        raise ValueError("embedded source review schema drift")
    return manifest, payload


def project_label_audit_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Drop tracing-only fields and retain completion-level label-audit evidence."""

    if payload.get("schemaVersion") != TARGET_REVIEW_SCHEMA_VERSION:
        raise ValueError("source payload schema drift")
    completions = []
    for item in payload.get("completions", []):
        annotations = [
            {
                key: annotation[key]
                for key in (
                    "sourceRowIndex",
                    "sourceRowId",
                    "labelType",
                    "sentenceText",
                    "sentenceSpan",
                    "extract",
                    "extractSpan",
                    "labelingReason",
                )
            }
            for annotation in item["annotations"]
        ]
        statistics = item["statistics"]
        completions.append(
            {
                "completionId": item["completionId"],
                "taskId": item["taskId"],
                "model": item["model"],
                "question": item["question"],
                "prompt": item["prompt"],
                "reasoning": item["reasoning"],
                "modelAnswer": item["modelAnswer"],
                "correctAnswer": item["correctAnswer"],
                "sourceType": item["sourceType"],
                "broadLabel": item["broadLabel"],
                "exactLabelTypes": item["exactLabelTypes"],
                "hasUnfaithful": item["hasUnfaithful"],
                "statistics": {
                    key: statistics[key]
                    for key in ("responseTokens", "characters", "words", "lines")
                },
                "annotations": annotations,
            }
        )
    if not completions:
        raise ValueError("source payload contains no completions")
    completions.sort(
        key=lambda item: (
            not item["hasUnfaithful"],
            item["model"].casefold(),
            item["question"].casefold(),
            item["completionId"],
        )
    )
    unfaithful = [item for item in completions if item["hasUnfaithful"]]
    if not unfaithful:
        raise ValueError("source payload contains no unfaithful labels to audit")
    model_counts = Counter(item["model"] for item in completions)
    flagged_counts = Counter(item["model"] for item in unfaithful)
    models = [
        {
            "model": model,
            "completionCount": model_counts[model],
            "flaggedCount": flagged_counts[model],
        }
        for model in sorted(model_counts, key=str.casefold)
    ]
    default_model = (
        DEFAULT_QWEN_MODEL
        if flagged_counts[DEFAULT_QWEN_MODEL]
        else unfaithful[0]["model"]
    )
    source_meta = payload.get("meta", {})
    return {
        "schemaVersion": SCHEMA_VERSION,
        "meta": {
            "title": "BonaFide Source-Label Audit",
            "sourceName": source_meta.get("sourceName"),
            "sourceSha256": source_meta.get("sourceSha256"),
            "sourceReviewPayloadSha256": _sha256(canonical_json(payload)),
            "claimBoundary": (
                "This page displays source annotations for human audit. "
                "A source UNFAITHFUL label is a claim to inspect, not a validated verdict."
            ),
        },
        "defaults": {"model": default_model, "scope": "contains-unfaithful"},
        "counts": {
            "completions": len(completions),
            "containsUnfaithful": len(unfaithful),
            "defaultModelContainsUnfaithful": flagged_counts[default_model],
        },
        "models": models,
        "completions": completions,
    }


_HTML_HEAD = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="light"><title>BonaFide Source-Label Audit</title>
<style>
:root{--ink:#172033;--muted:#687187;--rule:#dce1e9;--soft:#f6f7f9;--blue:#315ec8;--blue-soft:#edf2ff;--red:#a53d2d;--red-soft:#fff0eb;--green:#26703b;--green-soft:#eaf7ed;--ui:Inter,ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;--prose:Georgia,"Times New Roman",serif}
*{box-sizing:border-box}html,body{height:100%;margin:0;color:var(--ink);background:#fff}body{font:14px/1.45 var(--ui);overflow:hidden}button,input,select{font:inherit;color:inherit}button,select,input{background:#fff;border:1px solid #c7ceda;border-radius:6px}button{cursor:pointer;padding:8px 11px}button:hover{border-color:#788399}button:focus-visible,input:focus-visible,select:focus-visible{outline:3px solid #b7c7f4;outline-offset:1px}.topbar{min-height:76px;border-bottom:1px solid var(--rule);display:flex;align-items:center;gap:22px;padding:12px 20px}.title{font-size:20px;margin:0}.subtitle{font-size:11px;color:var(--muted);margin-top:3px}.filters{margin-left:auto;display:flex;gap:8px;align-items:center;flex-wrap:wrap}.filters select,.filters input{height:36px;padding:0 9px}.filters input{width:210px}.shell{height:calc(100vh - 76px);display:grid;grid-template-columns:350px minmax(0,1fr)}.index{border-right:1px solid var(--rule);overflow:auto}.summary{position:sticky;top:0;z-index:2;background:#fff;border-bottom:1px solid var(--rule);padding:11px 14px;color:var(--muted);font-size:11px}.completion{display:block;width:100%;border:0;border-bottom:1px solid var(--rule);border-left:3px solid transparent;border-radius:0;text-align:left;padding:13px 14px;background:#fff}.completion:hover{background:var(--soft)}.completion.active{border-left-color:var(--blue);background:var(--blue-soft)}.completion-model{font-size:11px;font-weight:750}.completion-question{font:14px/1.35 var(--prose);margin-top:5px}.completion-meta{font-size:10px;color:var(--red);font-weight:700;margin-top:7px}.reader{overflow:auto}.reader-inner{max-width:1000px;margin:0 auto;padding:24px 30px 80px}.reader-header{display:flex;gap:16px;align-items:start;border-bottom:1px solid var(--rule);padding-bottom:16px}.reader-heading{min-width:0}.reader-question{font:21px/1.3 var(--prose);margin:0 0 6px}.byline{font-size:11px;color:var(--muted);overflow-wrap:anywhere}.badge{margin-left:auto;background:var(--red-soft);color:var(--red);border:1px solid #e9b9ad;border-radius:999px;padding:5px 9px;font-size:10px;font-weight:800;text-transform:uppercase;letter-spacing:.05em;white-space:nowrap}.nav{display:flex;gap:7px;margin:14px 0}.panel{border:1px solid var(--rule);border-radius:8px;padding:14px 16px;margin:14px 0}.panel h3,.section h3{font-size:10px;text-transform:uppercase;letter-spacing:.09em;margin:0 0 10px}.claim{background:var(--red-soft);border-color:#e9b9ad}.claim h3{color:var(--red)}.evidence{background:var(--green-soft);border-color:#b9ddc2}.evidence h3{color:var(--green)}.annotation{border-top:1px solid rgba(80,90,110,.16);padding:9px 0}.annotation:first-of-type{border-top:0;padding-top:0}.annotation:last-child{padding-bottom:0}.annotation-label{font-size:10px;font-weight:800}.annotation-extract{font:15px/1.5 var(--prose);margin:4px 0}.annotation-reason{font-size:12px;color:#4f586b;white-space:pre-wrap}.section{border-top:1px solid var(--rule);padding:18px 0}.section:first-of-type{border-top:0}.prose{font:16px/1.65 var(--prose);white-space:pre-wrap;overflow-wrap:anywhere}.reasoning mark{border-radius:2px;padding:0 1px}.reasoning mark.faithful{background:#d6efdc;box-shadow:inset 0 -2px var(--green)}.reasoning mark.unfaithful{background:#f9d9d1;box-shadow:inset 0 -2px var(--red)}.reasoning mark.mixed{background:#f0dfec;box-shadow:inset 0 -2px #875073}.legend{font-size:11px;color:var(--muted);margin-bottom:10px}.swatch{display:inline-block;width:12px;height:12px;border-radius:2px;vertical-align:-2px;margin:0 4px 0 12px}.swatch:first-child{margin-left:0}.swatch.faithful{background:#d6efdc;border-bottom:2px solid var(--green)}.swatch.unfaithful{background:#f9d9d1;border-bottom:2px solid var(--red)}.answers{display:grid;grid-template-columns:1fr 1fr;gap:20px}.answer-label{font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin-bottom:5px}.answer{font:15px/1.5 var(--prose);white-space:pre-wrap}.stats{display:flex;gap:6px;flex-wrap:wrap;margin-top:9px}.stat{font-size:10px;color:var(--muted);border:1px solid var(--rule);border-radius:999px;padding:3px 7px}details{margin-top:8px}summary{cursor:pointer;font-weight:700}.empty{padding:30px;color:var(--muted)}
@media(max-width:820px){body{overflow:auto}.topbar{align-items:start;flex-direction:column}.filters{margin:0;width:100%}.filters select,.filters input{flex:0 0 100%;width:100%;min-width:0}.shell{height:auto;display:block}.index{border-right:0;border-bottom:1px solid var(--rule);max-height:310px}.reader{overflow:visible}.reader-inner{padding:18px 15px 60px}.reader-header{flex-wrap:wrap}.badge{margin-left:0}.answers{grid-template-columns:1fr}}
</style></head><body>
<header class="topbar"><div><h1 class="title">BonaFide source-label audit</h1><div class="subtitle" id="source-note">Loading verified source annotations…</div></div><div class="filters"><select data-filter="model" aria-label="Model"><option value="">All models</option></select><select data-filter="scope" aria-label="Label scope"><option value="contains-unfaithful">Mixed / contains UNFAITHFUL</option><option value="all">All labeled completions</option></select><input data-filter="search" type="search" placeholder="Search question or response" aria-label="Search"></div></header>
<main class="shell"><aside class="index"><div class="summary" id="result-summary"></div><div id="completion-list"></div></aside><article class="reader"><div class="reader-inner" id="reader-inner"></div></article></main>
<script>
"""


_HTML_SCRIPT = r"""
async function decodePayload(){const bytes=Uint8Array.from(atob("__PAYLOAD__"),c=>c.charCodeAt(0));const stream=new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));const raw=new Uint8Array(await new Response(stream).arrayBuffer());const digest=new Uint8Array(await crypto.subtle.digest("SHA-256",raw));const hex=[...digest].map(v=>v.toString(16).padStart(2,"0")).join("");if(hex!=="__PAYLOAD_SHA__")throw new Error("embedded payload hash mismatch");return JSON.parse(new TextDecoder().decode(raw))}
function el(tag,className,text){const node=document.createElement(tag);if(className)node.className=className;if(text!==undefined)node.textContent=String(text);return node}
function shortModel(value){return value.split("/").pop()}
function polarity(label){return label.toUpperCase().startsWith("UNFAITHFUL")?"unfaithful":"faithful"}
function annotationBlock(annotation){const row=el("div","annotation");row.append(el("div","annotation-label",annotation.labelType));if(annotation.extract)row.append(el("div","annotation-extract","“"+annotation.extract+"”"));row.append(el("div","annotation-reason",annotation.labelingReason||"No source rationale supplied."));return row}
function appendHighlighted(container,text,annotations){const spans=annotations.filter(a=>a.extractSpan[0]>=0&&a.extractSpan[1]>a.extractSpan[0]&&a.extractSpan[1]<=text.length);const points=new Set([0,text.length]);spans.forEach(a=>{points.add(a.extractSpan[0]);points.add(a.extractSpan[1])});const sorted=[...points].sort((a,b)=>a-b);for(let i=0;i<sorted.length-1;i++){const start=sorted[i],end=sorted[i+1],part=text.slice(start,end),active=spans.filter(a=>a.extractSpan[0]<=start&&a.extractSpan[1]>=end);if(!active.length){container.append(document.createTextNode(part));continue}const signs=new Set(active.map(a=>polarity(a.labelType)));const mark=el("mark",signs.size>1?"mixed":[...signs][0],part);mark.title=active.map(a=>a.labelType+": "+a.labelingReason).join("\n");container.append(mark)}}
async function main(){const payload=await decodePayload();const model=document.querySelector('[data-filter="model"]'),scope=document.querySelector('[data-filter="scope"]'),search=document.querySelector('[data-filter="search"]'),list=document.getElementById("completion-list"),summary=document.getElementById("result-summary"),reader=document.getElementById("reader-inner");payload.models.forEach(item=>{const option=el("option","",shortModel(item.model)+" · "+item.flaggedCount+" flagged");option.value=item.model;model.append(option)});model.value=payload.defaults.model;scope.value=payload.defaults.scope;const state={active:""};
function visible(){const needle=search.value.trim().toLocaleLowerCase();return payload.completions.filter(item=>(!model.value||item.model===model.value)&&(scope.value!=="contains-unfaithful"||item.hasUnfaithful)&&(!needle||[item.question,item.prompt,item.reasoning,item.modelAnswer,item.correctAnswer,...item.annotations.map(a=>a.labelingReason)].join("\n").toLocaleLowerCase().includes(needle)))}
function setActive(id){state.active=id;render()}
function renderList(items){list.replaceChildren();summary.textContent=items.length+" completion"+(items.length===1?"":"s")+" shown";if(!items.length){list.append(el("p","empty","No completions match these filters."));return}items.forEach(item=>{const button=el("button","completion"+(item.completionId===state.active?" active":""));button.type="button";button.dataset.completionId=item.completionId;button.append(el("div","completion-model",shortModel(item.model)),el("div","completion-question",item.question),el("div","completion-meta",item.exactLabelTypes.join(" + ")));button.addEventListener("click",()=>setActive(item.completionId));list.append(button)})}
function panel(title,className,annotations,emptyText){const node=el("section","panel "+className);node.append(el("h3","",title));if(!annotations.length)node.append(el("div","annotation-reason",emptyText));else annotations.forEach(annotation=>node.append(annotationBlock(annotation)));return node}
function renderReader(items){reader.replaceChildren();if(!items.length){reader.append(el("p","empty","Choose different filters to inspect a completion."));return}let index=items.findIndex(item=>item.completionId===state.active);if(index<0){index=0;state.active=items[0].completionId}const item=items[index],unfaithful=item.annotations.filter(a=>polarity(a.labelType)==="unfaithful"),faithful=item.annotations.filter(a=>polarity(a.labelType)==="faithful");const header=el("header","reader-header"),heading=el("div","reader-heading");heading.append(el("h2","reader-question",item.question),el("div","byline",item.model+" · "+item.sourceType+" · "+item.completionId));header.append(heading,el("div","badge","Source label: "+item.broadLabel));reader.append(header);const nav=el("div","nav"),previous=el("button","","Previous"),next=el("button","","Next");previous.disabled=index===0;next.disabled=index===items.length-1;previous.addEventListener("click",()=>setActive(items[index-1].completionId));next.addEventListener("click",()=>setActive(items[index+1].completionId));nav.append(previous,next);reader.append(nav);reader.append(panel("Claim to audit: why the source says unfaithful","claim",unfaithful,"No UNFAITHFUL annotation is attached."));reader.append(panel("Evidence the source separately marked faithful","evidence",faithful,"No faithful extracts are attached."));const reasoning=el("section","section");reasoning.append(el("h3","","Full model reasoning"));const legend=el("div","legend");legend.append(el("span","swatch faithful"),document.createTextNode("faithful extract"),el("span","swatch unfaithful"),document.createTextNode("unfaithful extract"));reasoning.append(legend);const prose=el("div","prose reasoning");appendHighlighted(prose,item.reasoning,item.annotations);reasoning.append(prose);const stats=el("div","stats");[["Response tokens",item.statistics.responseTokens],["Words",item.statistics.words],["Lines",item.statistics.lines]].forEach(pair=>stats.append(el("span","stat",pair[0]+" "+pair[1])));reasoning.append(stats);reader.append(reasoning);const answers=el("section","section answers");[["Model answer",item.modelAnswer],["Correct answer",item.correctAnswer]].forEach(pair=>{const cell=el("div");cell.append(el("div","answer-label",pair[0]),el("div","answer",pair[1]||"—"));answers.append(cell)});reader.append(answers);const prompt=el("details","section"),promptSummary=el("summary","","Full prompt");prompt.append(promptSummary,el("div","prose",item.prompt));reader.append(prompt)}
function render(){const items=visible();if(items.length&&!items.some(item=>item.completionId===state.active))state.active=items[0].completionId;renderList(items);renderReader(items)}
[model,scope,search].forEach(node=>node.addEventListener(node===search?"input":"change",render));document.getElementById("source-note").textContent=payload.counts.defaultModelContainsUnfaithful+" disputed Qwen completions shown by default · "+payload.counts.containsUnfaithful+" source-unfaithful completions total · source "+payload.meta.sourceSha256.slice(0,12);render()}
main().catch(error=>{const reader=document.getElementById("reader-inner");reader.replaceChildren(el("p","empty","Could not load audit data: "+error.message));console.error(error)});
</script></body></html>
"""


def render_label_audit_html(payload: Mapping[str, Any]) -> str:
    """Render a safe standalone audit page with a compressed inert payload."""

    payload_bytes = canonical_json(payload)
    encoded = base64.b64encode(gzip.compress(payload_bytes, mtime=0)).decode("ascii")
    return _HTML_HEAD + _HTML_SCRIPT.replace("__PAYLOAD__", encoded).replace(
        "__PAYLOAD_SHA__", _sha256(payload_bytes)
    )


def build_label_audit_packet(*, source: Path, destination: Path) -> dict[str, Any]:
    """Build the presentation-only label audit and its binding manifest."""

    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    source_manifest, source_payload = load_target_review_packet(source)
    payload = project_label_audit_payload(source_payload)
    payload_bytes = canonical_json(payload)
    html = render_label_audit_html(payload).encode("utf-8")
    temporary = destination.parent / f".{destination.name}.building-{uuid.uuid4().hex}"
    temporary.mkdir(parents=True)
    try:
        _atomic_write_bytes(temporary / "review.html", html)
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": "human_source_label_audit",
            "source_review": {
                "path": str(source),
                "schema_version": source_manifest["schema_version"],
                "manifest_sha256": source_manifest["manifest_sha256"],
                "page_sha256": source_manifest["page_sha256"],
                "payload_sha256": source_manifest["payload_sha256"],
            },
            "scope": {
                "default_model": payload["defaults"]["model"],
                "default_filter": payload["defaults"]["scope"],
                "all_source_completions_available": True,
            },
            "counts": payload["counts"],
            "payload_sha256": _sha256(payload_bytes),
            "page_sha256": _sha256(html),
            "files": {"review.html": {"sha256": _sha256(html), "bytes": len(html)}},
            "claim_boundaries": [
                "This is a human audit view of source labels, not an evaluation.",
                "Source labels and rationales are displayed verbatim and are not treated as validated verdicts.",
                "The page contains no tracing-target selection, token identities, model inference, scoring, or neuron interpretation.",
            ],
        }
        manifest["manifest_sha256"] = canonical_sha256(manifest)
        _atomic_write_json(temporary / "manifest.json", manifest)
        os.replace(temporary, destination)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
