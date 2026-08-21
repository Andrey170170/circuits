"""Build an offline review packet for labeled outright completions.

The packet deliberately stops at human selection.  It does not score, rank, or
interpret completions, and it removes excluded model families before the
browser payload is serialized.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import json
import os
import shutil
import uuid
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from circuits.analysis.bonafide.canonical import (
    canonical_json,
    canonical_sha256,
    file_sha256,
)

SCHEMA_VERSION = "adag.raw-graph-observatory.outright-review.v1"
EXPECTED_SOURCE_SHA256 = (
    "5833b500c378bbdcc7103340987749efda10b5944897168e10aed2be4538e13e"
)
OUTRIGHT_SOURCE_TYPES = frozenset({"complex", "graph"})
EXCLUDED_MODEL_SUBSTRING = "qwen"
EXPECTED_COLUMNS = (
    "id",
    "question_id",
    "label_type",
    "sentence_text",
    "sentence_span_start",
    "sentence_span_end",
    "extract",
    "extract_span_start",
    "extract_span_end",
    "labeling_reason",
    "target_model",
    "question",
    "prompt",
    "cot",
    "model_answer",
    "correct_answer",
    "hinted_answer",
    "src_type",
    "hint_dataset",
    "hint_type",
    "prompted_hint",
)


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
        (
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8"),
    )


def _content_id(kind: str, value: object) -> str:
    return f"{kind}_{canonical_sha256(value)}"


def _parse_span(
    row: Mapping[str, str],
    *,
    row_number: int,
    prefix: str,
    text_field: str,
) -> list[int]:
    try:
        start = int(row[f"{prefix}_span_start"])
        end = int(row[f"{prefix}_span_end"])
    except (TypeError, ValueError) as error:
        raise ValueError(f"row {row_number}: invalid {prefix} span") from error
    expected = row[text_field]
    if (start, end) == (0, -1) and expected == "":
        return [start, end]
    cot = row["cot"]
    if not 0 <= start <= end <= len(cot):
        raise ValueError(
            f"row {row_number}: {prefix} span [{start}, {end}] outside cot"
        )
    actual = cot[start:end]
    if actual != expected:
        raise ValueError(
            f"row {row_number}: {prefix} span text drift; "
            f"expected {expected!r}, found {actual!r}"
        )
    return [start, end]


def read_source_rows(
    source_path: Path,
    *,
    expected_source_sha256: str | None = EXPECTED_SOURCE_SHA256,
) -> tuple[list[dict[str, str]], str]:
    """Read and strictly validate the authoritative annotation CSV."""

    source_sha256 = file_sha256(source_path)
    if (
        expected_source_sha256 is not None
        and source_sha256 != expected_source_sha256
    ):
        raise ValueError(
            "source SHA256 drift: "
            f"expected {expected_source_sha256}, found {source_sha256}"
        )
    try:
        with source_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != EXPECTED_COLUMNS:
                raise ValueError(
                    "source schema drift: expected exact columns "
                    f"{EXPECTED_COLUMNS!r}, found {tuple(reader.fieldnames or ())!r}"
                )
            rows = [dict(row) for row in reader]
    except UnicodeDecodeError as error:
        raise ValueError(f"source is not UTF-8: {source_path}") from error
    if not rows:
        raise ValueError("source contains no annotation rows")
    if any(set(row) != set(EXPECTED_COLUMNS) for row in rows):
        raise ValueError("source row schema drift")
    return rows, source_sha256


def _label_flags(label_types: Iterable[str]) -> tuple[bool, bool]:
    labels = tuple(label_types)
    faithful = any(value.upper().startswith("FAITHFUL") for value in labels)
    unfaithful = any(value.upper().startswith("UNFAITHFUL") for value in labels)
    unknown = [
        value
        for value in labels
        if not value.upper().startswith(("FAITHFUL", "UNFAITHFUL"))
    ]
    if unknown:
        raise ValueError(f"unknown label polarity: {sorted(set(unknown))!r}")
    return faithful, unfaithful


def assemble_review_payload(
    rows: Sequence[Mapping[str, str]],
    *,
    source_sha256: str,
    source_name: str = "BonaFide.csv",
) -> dict[str, Any]:
    """Filter, deduplicate, and serialize completion-level review data."""

    outright_rows: list[tuple[int, Mapping[str, str]]] = []
    excluded_rows = 0
    excluded_completions: set[tuple[str, str, str]] = set()
    for row_number, row in enumerate(rows, start=2):
        if tuple(row.keys()) != EXPECTED_COLUMNS:
            raise ValueError(f"row {row_number}: source schema drift")
        source_type = row["src_type"].strip().lower()
        if source_type not in OUTRIGHT_SOURCE_TYPES:
            continue
        # Validate every outright row, including rows removed by policy.  This
        # keeps the packet fail-closed rather than allowing excluded bad spans
        # to hide source drift.
        _parse_span(
            row,
            row_number=row_number,
            prefix="sentence",
            text_field="sentence_text",
        )
        _parse_span(
            row,
            row_number=row_number,
            prefix="extract",
            text_field="extract",
        )
        if EXCLUDED_MODEL_SUBSTRING in row["target_model"].lower():
            excluded_rows += 1
            excluded_completions.add(
                (row["target_model"], row["prompt"], row["cot"])
            )
            continue
        outright_rows.append((row_number, row))

    grouped: dict[tuple[str, str, str], list[tuple[int, Mapping[str, str]]]] = (
        defaultdict(list)
    )
    for row_number, row in outright_rows:
        grouped[(row["target_model"], row["prompt"], row["cot"])].append(
            (row_number, row)
        )

    completions: list[dict[str, Any]] = []
    tasks_by_id: dict[str, dict[str, Any]] = {}
    for completion_key, source_rows in grouped.items():
        first = source_rows[0][1]
        invariant_fields = (
            "question",
            "model_answer",
            "correct_answer",
            "src_type",
            "hint_dataset",
            "hint_type",
            "prompted_hint",
            "hinted_answer",
        )
        for field in invariant_fields:
            values = {row[field] for _, row in source_rows}
            if len(values) != 1:
                raise ValueError(
                    f"completion metadata drift for {field}: {sorted(values)!r}"
                )

        task_identity = {
            "source_type": first["src_type"].strip().lower(),
            "question": first["question"],
            "correct_answer": first["correct_answer"],
        }
        task_id = _content_id("task", task_identity)
        completion_id = _content_id(
            "completion",
            {
                "target_model": completion_key[0],
                "prompt": completion_key[1],
                "cot": completion_key[2],
            },
        )
        annotations = []
        for row_number, row in source_rows:
            annotations.append(
                {
                    "sourceRowIndex": row_number,
                    "sourceRowId": row["id"],
                    "questionId": row["question_id"],
                    "labelType": row["label_type"],
                    "sentenceText": row["sentence_text"],
                    "sentenceSpan": _parse_span(
                        row,
                        row_number=row_number,
                        prefix="sentence",
                        text_field="sentence_text",
                    ),
                    "extract": row["extract"],
                    "extractSpan": _parse_span(
                        row,
                        row_number=row_number,
                        prefix="extract",
                        text_field="extract",
                    ),
                    "labelingReason": row["labeling_reason"],
                }
            )
        exact_label_types = sorted({item["labelType"] for item in annotations})
        has_faithful, has_unfaithful = _label_flags(exact_label_types)
        broad_label = (
            "mixed"
            if has_faithful and has_unfaithful
            else "contains-unfaithful"
            if has_unfaithful
            else "faithful-only"
        )
        question_ids = sorted(
            {item["questionId"] for item in annotations if item["questionId"]}
        )
        completion = {
            "completionId": completion_id,
            "taskId": task_id,
            "model": first["target_model"],
            "questionIds": question_ids,
            "question": first["question"],
            "prompt": first["prompt"],
            "reasoning": first["cot"],
            "modelAnswer": first["model_answer"],
            "correctAnswer": first["correct_answer"],
            "sourceType": first["src_type"].strip().lower(),
            "hintedAnswer": first["hinted_answer"],
            "hintDataset": first["hint_dataset"],
            "hintType": first["hint_type"],
            "promptedHint": first["prompted_hint"],
            "broadLabel": broad_label,
            "hasUnfaithful": has_unfaithful,
            "exactLabelTypes": exact_label_types,
            "annotations": annotations,
        }
        completions.append(completion)
        tasks_by_id.setdefault(
            task_id,
            {
                "taskId": task_id,
                "sourceType": completion["sourceType"],
                "question": completion["question"],
                "correctAnswer": completion["correctAnswer"],
            },
        )

    completions.sort(
        key=lambda item: (
            item["question"].casefold(),
            item["sourceType"],
            item["model"].casefold(),
            item["completionId"],
        )
    )
    completion_counts_by_task = Counter(item["taskId"] for item in completions)
    tasks = [
        {**task, "completionCount": completion_counts_by_task[task["taskId"]]}
        for task in tasks_by_id.values()
    ]
    tasks.sort(key=lambda item: (item["question"].casefold(), item["taskId"]))

    model_names = sorted({item["model"] for item in completions}, key=str.casefold)
    models = [
        {
            "model": model,
            "completionCount": sum(item["model"] == model for item in completions),
        }
        for model in model_names
    ]
    label_types = sorted(
        {label for item in completions for label in item["exactLabelTypes"]}
    )

    def dimension_counts(field: str) -> dict[str, dict[str, int]]:
        values = sorted({str(item[field]) for item in completions}, key=str.casefold)
        result: dict[str, dict[str, int]] = {}
        for value in values:
            selected = [item for item in completions if str(item[field]) == value]
            result[value] = {
                "completions": len(selected),
                "source_rows": sum(len(item["annotations"]) for item in selected),
                "tasks": len({item["taskId"] for item in selected}),
            }
        return result

    counts_by_label = {
        label: {
            "source_rows": sum(
                annotation["labelType"] == label
                for item in completions
                for annotation in item["annotations"]
            ),
            "completions": sum(label in item["exactLabelTypes"] for item in completions),
        }
        for label in label_types
    }
    counts = {
        "source_rows": len(outright_rows),
        "completions": len(completions),
        "tasks": len(tasks),
        "models": len(models),
        "excluded_source_rows": excluded_rows,
        "excluded_completions": len(excluded_completions),
        "by_model": dimension_counts("model"),
        "by_source_type": dimension_counts("sourceType"),
        "by_label_type": counts_by_label,
        "by_broad_label": {
            "faithful-only": sum(
                item["broadLabel"] == "faithful-only" for item in completions
            ),
            "contains-unfaithful": sum(
                item["hasUnfaithful"] for item in completions
            ),
            "mixed": sum(item["broadLabel"] == "mixed" for item in completions),
        },
    }
    return {
        "schemaVersion": SCHEMA_VERSION,
        "meta": {
            "title": "Outright Completion Review",
            "sourceName": source_name,
            "sourceSha256": source_sha256,
            "outrightDefinition": "src_type is exactly complex or graph",
            "exclusionPolicy": "remove rows when lowercase target_model contains qwen",
            "deduplicationKey": ["target_model", "prompt", "cot"],
            "claimBoundary": (
                "Human candidate browsing only. Labels are source annotations; "
                "selection is not a score, trace result, neuron interpretation, "
                "or faithfulness validation."
            ),
        },
        "counts": counts,
        "models": models,
        "tasks": tasks,
        "labelTypes": label_types,
        "completions": completions,
    }


_HTML_HEAD = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light">
<title>Outright Completion Review</title>
<style>
:root{--white:#fff;--ink:#11172b;--muted:#697184;--rule:#d9dde6;--soft:#f6f7fb;--indigo:#405de6;--indigo-soft:#eef1ff;--green:#286a3a;--green-soft:#e6f4e8;--rust:#a23d24;--rust-soft:#fbe8e2;--ui:Inter,ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;--prose:Georgia,"Times New Roman",serif}
*{box-sizing:border-box}html,body{height:100%;margin:0;background:var(--white);color:var(--ink)}body{font-family:var(--ui);font-size:14px;overflow:hidden}button,input,select{font:inherit;color:inherit}button,select,input[type=search]{background:#fff;border:1px solid #cbd0dc;border-radius:5px;min-height:34px}button{cursor:pointer;padding:0 12px}button:hover{border-color:#8e98af}button:focus-visible,input:focus-visible,select:focus-visible,[tabindex]:focus-visible{outline:3px solid #b8c3ff;outline-offset:1px}.topbar{height:64px;border-bottom:1px solid var(--rule);display:flex;align-items:center;gap:20px;padding:0 22px}.title{font-weight:720;font-size:19px;letter-spacing:-.02em;margin:0}.subtitle{color:var(--muted);font-size:12px}.top-actions{margin-left:auto;display:flex;gap:9px;align-items:center}.check{display:flex;align-items:center;gap:7px;white-space:nowrap}.primary{border-color:#aebaff;color:#314dcc}.shell{height:calc(100vh - 64px);display:grid;grid-template-columns:230px 370px minmax(0,1fr)}.rail,.index{border-right:1px solid var(--rule);min-width:0;overflow:auto}.rail{padding:20px 0;display:flex;flex-direction:column}.section-title{font-size:11px;line-height:1.2;text-transform:uppercase;letter-spacing:.1em;color:var(--muted);font-weight:750;margin:0 18px 10px}.model-list{display:flex;flex-direction:column}.model{border:0;border-left:3px solid transparent;border-radius:0;min-height:47px;padding:8px 16px;text-align:left;display:grid;grid-template-columns:minmax(0,1fr) auto;gap:8px;align-items:center}.model:hover{background:var(--soft)}.model.active{background:var(--indigo-soft);border-left-color:var(--indigo);color:#243fae}.model-name{overflow-wrap:anywhere;font-size:12px;font-weight:650}.count{font-variant-numeric:tabular-nums;color:var(--muted);font-size:11px}.rail-note{margin:auto 18px 0;padding-top:20px;color:var(--muted);font-size:11px;line-height:1.5}.index{display:flex;flex-direction:column}.filters{padding:17px;border-bottom:1px solid var(--rule);display:grid;grid-template-columns:1fr 1fr;gap:8px;position:sticky;top:0;background:#fff;z-index:2}.filters .wide{grid-column:1/-1}.filters select,.filters input{width:100%;min-width:0;padding:0 9px}.result-summary{grid-column:1/-1;color:var(--muted);font-size:11px;display:flex;justify-content:space-between}.completion-list{padding-bottom:30px}.task-group{border-bottom:1px solid var(--rule)}.task-heading{margin:0;padding:13px 16px 8px;font-family:var(--prose);font-size:13px;line-height:1.35}.task-meta{font-family:var(--ui);display:block;color:var(--muted);font-size:10px;margin-top:4px;text-transform:uppercase;letter-spacing:.06em}.completion-row{width:100%;border:0;border-radius:0;border-left:3px solid transparent;min-height:58px;text-align:left;padding:9px 14px;display:grid;grid-template-columns:18px minmax(0,1fr);gap:8px}.completion-row:hover{background:var(--soft)}.completion-row.active{background:var(--indigo-soft);border-left-color:var(--indigo)}.completion-row input{pointer-events:none;margin:3px 0}.row-model{font-size:11px;font-weight:700;overflow-wrap:anywhere}.row-meta{color:var(--muted);font-size:10px;margin-top:5px;display:flex;gap:6px;align-items:center}.dot{width:6px;height:6px;border-radius:50%;display:inline-block}.dot.faithful{background:var(--green)}.dot.unfaithful,.dot.mixed{background:var(--rust)}.empty{padding:32px 18px;color:var(--muted);font-family:var(--prose)}.reader{overflow:auto;min-width:0}.reader-inner{max-width:940px;margin:0 auto;padding:22px 30px 80px}.reader-header{display:flex;gap:16px;align-items:start;border-bottom:1px solid var(--rule);padding-bottom:16px}.reader-header .check{margin-top:4px}.reader-heading{min-width:0}.reader-task{font-family:var(--prose);font-size:18px;line-height:1.35;margin:0 0 6px}.reader-byline{color:var(--muted);font-size:11px;overflow-wrap:anywhere}.status{margin-left:auto;font-size:11px;font-weight:750;text-transform:uppercase;letter-spacing:.07em}.status.faithful{color:var(--green)}.status.unfaithful,.status.mixed{color:var(--rust)}.doc-section{border-bottom:1px solid var(--rule);padding:19px 0}.doc-section h2{font:750 11px/1.2 var(--ui);text-transform:uppercase;letter-spacing:.1em;color:var(--muted);margin:0 0 10px}.prose{font:16px/1.68 var(--prose);white-space:pre-wrap;overflow-wrap:anywhere;margin:0}.answer-grid{display:grid;grid-template-columns:1fr 1fr;gap:28px}.answer{font:16px/1.5 var(--prose);white-space:pre-wrap}.annotation-list{display:flex;flex-direction:column}.annotation{display:grid;grid-template-columns:132px minmax(0,1fr);gap:16px;padding:14px 0;border-top:1px solid #eceef3}.annotation:first-child{border-top:0}.annotation-key{font-size:10px;color:var(--muted);line-height:1.5;overflow-wrap:anywhere}.annotation-type{font-weight:750}.annotation-type.faithful{color:var(--green)}.annotation-type.unfaithful{color:var(--rust)}.annotation-body{min-width:0}.annotation-extract{font:15px/1.5 var(--prose);margin:0 0 7px}.annotation-sentence{font:13px/1.5 var(--prose);color:#41485a;margin:0 0 6px}.annotation-reason{font-size:12px;line-height:1.5;color:#545d70;margin:0}mark{padding:.08em .12em;border-radius:2px;color:inherit}mark.faithful{background:var(--green-soft);box-shadow:inset 0 -2px #8cc99a}mark.unfaithful{background:var(--rust-soft);box-shadow:inset 0 -2px #d9917d}mark.mixed{background:#f4e6ef;box-shadow:inset 0 -2px #aa7197}.keyboard{font-size:10px;color:var(--muted);margin-top:12px}.hidden{display:none!important}@media(max-width:1050px){.shell{grid-template-columns:180px 310px minmax(0,1fr)}.reader-inner{padding-left:22px;padding-right:22px}.subtitle{display:none}}@media(max-width:780px){body{overflow:auto}.topbar{height:auto;min-height:60px;flex-wrap:wrap;padding:12px 15px}.top-actions{width:100%;margin:0;overflow:auto;padding-bottom:2px}.shell{height:auto;display:block}.rail,.index,.reader{overflow:visible;border-right:0}.rail{border-bottom:1px solid var(--rule);padding:12px 0}.section-title{margin-bottom:6px}.model-list{flex-direction:row;overflow:auto;padding:0 12px}.model{flex:0 0 180px;border-left:0;border-bottom:3px solid transparent}.model.active{border-bottom-color:var(--indigo)}.rail-note{display:none}.index{border-bottom:1px solid var(--rule)}.filters{position:static}.completion-list{max-height:360px;overflow:auto}.reader-inner{padding:20px 16px 70px}.answer-grid{grid-template-columns:1fr;gap:16px}.annotation{grid-template-columns:1fr}.reader-header{flex-wrap:wrap}.status{margin-left:0;width:100%}}
</style>
</head>
<body>
<header class="topbar">
  <div><h1 class="title">Outright Completion Review</h1><div class="subtitle" id="source-note"></div></div>
  <div class="top-actions">
    <label class="check"><input id="selection-only" type="checkbox">Selected only</label>
    <button id="export" class="primary" type="button">Export JSON</button>
    <button id="previous" type="button" aria-label="Previous completion">Previous</button>
    <button id="next" type="button" aria-label="Next completion">Next</button>
  </div>
</header>
<main class="shell">
  <aside class="rail" aria-label="Models"><h2 class="section-title">Model</h2><nav id="model-list" class="model-list"></nav><p id="rail-note" class="rail-note"></p></aside>
  <section class="index" aria-label="Task and completion index">
    <div class="filters">
      <select id="task-filter" class="wide" data-filter="task" aria-label="Exact task"><option value="">All exact tasks</option></select>
      <select id="source-filter" data-filter="source-type" aria-label="Source type"><option value="">All source types</option><option value="complex">Complex</option><option value="graph">Graph</option></select>
      <select id="broad-filter" data-filter="broad-label" aria-label="Broad label"><option value="">All broad labels</option><option value="faithful-only">Faithful only</option><option value="contains-unfaithful">Contains unfaithful</option><option value="mixed">Mixed</option></select>
      <select id="label-filter" class="wide" data-filter="exact-label-type" aria-label="Exact label type"><option value="">All exact label types</option></select>
      <input id="search" class="wide" data-filter="search" type="search" placeholder="Search prompt, reasoning, answers…" aria-label="Search completions">
      <div class="result-summary"><span id="result-count"></span><button id="reset" type="button">Reset filters</button></div>
    </div>
    <div id="completion-list" class="completion-list"></div>
  </section>
  <article id="reader" class="reader" aria-live="polite"><div id="reader-inner" class="reader-inner"></div></article>
</main>
'''

_HTML_SCRIPT = r'''
<script>
"use strict";
const payload = JSON.parse(new TextDecoder().decode(Uint8Array.from(atob("__PAYLOAD__"), c => c.charCodeAt(0))));
const storageKey = "outright-review-selection:" + payload.meta.sourceSha256;
const state = {model:"",task:"",source:"",broad:"",label:"",search:"",selectionOnly:false,active:"",selected:new Set()};
try { const saved=JSON.parse(localStorage.getItem(storageKey)||"[]"); if(Array.isArray(saved)) saved.forEach(id=>state.selected.add(id)); } catch(_error) {}
const $ = id => document.getElementById(id);
const nodes = {models:$("model-list"),list:$("completion-list"),reader:$("reader-inner"),task:$("task-filter"),source:$("source-filter"),broad:$("broad-filter"),label:$("label-filter"),search:$("search"),only:$("selection-only"),count:$("result-count")};
function el(tag,className,text){const node=document.createElement(tag);if(className)node.className=className;if(text!==undefined)node.textContent=String(text);return node}
function shortModel(value){const pieces=value.split("/");return pieces[pieces.length-1]}
function polarity(label){return label.toUpperCase().startsWith("UNFAITHFUL")?"unfaithful":"faithful"}
function persist(){localStorage.setItem(storageKey,JSON.stringify([...state.selected].sort()))}
function addOption(select,value,text){const option=el("option","",text);option.value=value;select.append(option)}
payload.tasks.forEach(task=>addOption(nodes.task,task.taskId,task.question+" · "+task.sourceType));
payload.labelTypes.forEach(label=>addOption(nodes.label,label,label));
function matches(item){
  if(state.model&&item.model!==state.model)return false;
  if(state.task&&item.taskId!==state.task)return false;
  if(state.source&&item.sourceType!==state.source)return false;
  if(state.broad==="faithful-only"&&item.broadLabel!=="faithful-only")return false;
  if(state.broad==="contains-unfaithful"&&!item.hasUnfaithful)return false;
  if(state.broad==="mixed"&&item.broadLabel!=="mixed")return false;
  if(state.label&&!item.exactLabelTypes.includes(state.label))return false;
  if(state.selectionOnly&&!state.selected.has(item.completionId))return false;
  if(state.search){const hay=[item.model,item.question,item.prompt,item.reasoning,item.modelAnswer,item.correctAnswer,...item.annotations.flatMap(a=>[a.labelType,a.extract,a.sentenceText,a.labelingReason])].join("\n").toLocaleLowerCase();if(!hay.includes(state.search))return false}
  return true;
}
function visible(){return payload.completions.filter(matches)}
function setActive(id){state.active=id;renderList();renderReader();requestAnimationFrame(()=>{const row=document.querySelector('[data-completion-id="'+CSS.escape(id)+'"]');if(row)row.scrollIntoView({block:"nearest"})})}
function toggle(id){state.selected.has(id)?state.selected.delete(id):state.selected.add(id);persist();renderAll()}
function renderModels(){nodes.models.replaceChildren();const values=[{model:"",completionCount:payload.counts.completions},...payload.models];values.forEach(value=>{const button=el("button","model"+(state.model===value.model?" active":""));button.type="button";button.setAttribute("aria-pressed",String(state.model===value.model));const name=el("span","model-name",value.model?shortModel(value.model):"All models");const count=el("span","count",value.completionCount);button.append(name,count);button.addEventListener("click",()=>{state.model=value.model;renderAll()});nodes.models.append(button)})}
function renderList(){const items=visible();nodes.count.textContent=items.length+" of "+payload.counts.completions+" completions";nodes.list.replaceChildren();if(!items.length){nodes.list.append(el("p","empty","No completions match these filters."));return}if(!items.some(item=>item.completionId===state.active))state.active=items[0].completionId;const groups=new Map();items.forEach(item=>{if(!groups.has(item.taskId))groups.set(item.taskId,[]);groups.get(item.taskId).push(item)});groups.forEach(group=>{const wrap=el("section","task-group");const heading=el("h3","task-heading",group[0].question);const meta=el("span","task-meta",group[0].sourceType+" · "+group.length+" completion"+(group.length===1?"":"s"));heading.append(meta);wrap.append(heading);group.forEach(item=>{const button=el("button","completion-row"+(state.active===item.completionId?" active":""));button.type="button";button.dataset.completionId=item.completionId;button.setAttribute("aria-pressed",String(state.active===item.completionId));const check=document.createElement("input");check.type="checkbox";check.checked=state.selected.has(item.completionId);check.tabIndex=-1;check.setAttribute("aria-hidden","true");const body=el("span");body.append(el("span","row-model",shortModel(item.model)));const meta=el("span","row-meta");meta.append(el("span","dot "+(item.hasUnfaithful?"unfaithful":"faithful")));meta.append(document.createTextNode(item.broadLabel+" · "+item.annotations.length+" annotation"+(item.annotations.length===1?"":"s")));body.append(meta);button.append(check,body);button.addEventListener("click",()=>setActive(item.completionId));wrap.append(button)});nodes.list.append(wrap)})}
function appendHighlighted(container,text,annotations){const valid=annotations.filter(a=>a.extractSpan[0]>=0&&a.extractSpan[1]>=a.extractSpan[0]&&a.extractSpan[1]<=text.length&&a.extractSpan[1]>a.extractSpan[0]);const points=new Set([0,text.length]);valid.forEach(a=>{points.add(a.extractSpan[0]);points.add(a.extractSpan[1])});const sorted=[...points].sort((a,b)=>a-b);for(let i=0;i<sorted.length-1;i++){const start=sorted[i],end=sorted[i+1],part=text.slice(start,end);const active=valid.filter(a=>a.extractSpan[0]<=start&&a.extractSpan[1]>=end);if(!active.length){container.append(document.createTextNode(part));continue}const polarities=new Set(active.map(a=>polarity(a.labelType)));const mark=el("mark",polarities.size>1?"mixed":[...polarities][0],part);mark.title=active.map(a=>a.labelType+": "+a.labelingReason).join("\n");container.append(mark)}}
function section(title){const wrap=el("section","doc-section");wrap.append(el("h2","",title));return wrap}
function renderReader(){nodes.reader.replaceChildren();const item=payload.completions.find(value=>value.completionId===state.active);if(!item||!matches(item)){nodes.reader.append(el("p","empty","Choose a completion to inspect."));return}const header=el("header","reader-header");const check=el("label","check");const input=document.createElement("input");input.type="checkbox";input.checked=state.selected.has(item.completionId);input.setAttribute("aria-label","Select this completion");input.addEventListener("change",()=>toggle(item.completionId));check.append(input);const heading=el("div","reader-heading");heading.append(el("h2","reader-task",item.question),el("div","reader-byline",item.model+" · "+item.sourceType+" · "+item.completionId));const status=el("div","status "+(item.hasUnfaithful?"mixed":"faithful"),item.broadLabel);header.append(check,heading,status);nodes.reader.append(header);
  const prompt=section("Full prompt");prompt.append(el("p","prose",item.prompt));nodes.reader.append(prompt);
  const reasoning=section("Reasoning");const prose=el("p","prose");appendHighlighted(prose,item.reasoning,item.annotations);reasoning.append(prose);nodes.reader.append(reasoning);
  const answers=section("Answers");const grid=el("div","answer-grid");const model=el("div");model.append(el("h2","","Model answer"),el("div","answer",item.modelAnswer||"—"));const correct=el("div");correct.append(el("h2","","Correct answer"),el("div","answer",item.correctAnswer||"—"));grid.append(model,correct);answers.append(grid);nodes.reader.append(answers);
  const annotationSection=section("Annotations · "+item.annotations.length);const list=el("div","annotation-list");item.annotations.forEach(a=>{const row=el("article","annotation");const key=el("div","annotation-key");key.append(el("div","annotation-type "+polarity(a.labelType),a.labelType),el("div","",a.sourceRowId),el("div","","row "+a.sourceRowIndex),el("div","","sentence ["+a.sentenceSpan.join(", ")+"]"),el("div","","extract ["+a.extractSpan.join(", ")+"]"));const body=el("div","annotation-body");body.append(el("p","annotation-extract",a.extract||"Whole-completion annotation"));if(a.sentenceText)body.append(el("p","annotation-sentence",a.sentenceText));body.append(el("p","annotation-reason",a.labelingReason||"No reason supplied."));row.append(key,body);list.append(row)});annotationSection.append(list,el("p","keyboard","Keyboard: J / K next or previous · S select or deselect"));nodes.reader.append(annotationSection)}
function renderAll(){renderModels();renderList();renderReader()}
function move(delta){const items=visible();if(!items.length)return;let index=items.findIndex(item=>item.completionId===state.active);index=index<0?0:(index+delta+items.length)%items.length;setActive(items[index].completionId)}
nodes.task.addEventListener("change",e=>{state.task=e.target.value;renderAll()});nodes.source.addEventListener("change",e=>{state.source=e.target.value;renderAll()});nodes.broad.addEventListener("change",e=>{state.broad=e.target.value;renderAll()});nodes.label.addEventListener("change",e=>{state.label=e.target.value;renderAll()});nodes.search.addEventListener("input",e=>{state.search=e.target.value.toLocaleLowerCase().trim();renderAll()});nodes.only.addEventListener("change",e=>{state.selectionOnly=e.target.checked;renderAll()});$("previous").addEventListener("click",()=>move(-1));$("next").addEventListener("click",()=>move(1));
$("reset").addEventListener("click",()=>{Object.assign(state,{model:"",task:"",source:"",broad:"",label:"",search:"",selectionOnly:false});nodes.task.value="";nodes.source.value="";nodes.broad.value="";nodes.label.value="";nodes.search.value="";nodes.only.checked=false;renderAll()});
$("export").addEventListener("click",()=>{const selections=payload.completions.filter(item=>state.selected.has(item.completionId));const exported={schemaVersion:"adag.raw-graph-observatory.outright-selection.v1",provenance:{sourceName:payload.meta.sourceName,sourceSha256:payload.meta.sourceSha256,reviewPayloadSha256:"__PAYLOAD_SHA__",exclusionPolicy:payload.meta.exclusionPolicy,claimBoundary:payload.meta.claimBoundary},selectedCompletionIds:selections.map(item=>item.completionId),selections:selections.map(item=>({completionId:item.completionId,taskId:item.taskId,model:item.model,question:item.question,sourceType:item.sourceType,broadLabel:item.broadLabel,exactLabelTypes:item.exactLabelTypes,sourceRowIds:item.annotations.map(a=>a.sourceRowId)}))};const blob=new Blob([JSON.stringify(exported,null,2)+"\n"],{type:"application/json"});const link=document.createElement("a");link.href=URL.createObjectURL(blob);link.download="outright-completion-selection.json";link.click();setTimeout(()=>URL.revokeObjectURL(link.href),0)});
document.addEventListener("keydown",event=>{if(event.ctrlKey||event.metaKey||event.altKey||/^(INPUT|SELECT|TEXTAREA|BUTTON)$/.test(event.target.tagName))return;if(event.key.toLowerCase()==="j"){event.preventDefault();move(1)}else if(event.key.toLowerCase()==="k"){event.preventDefault();move(-1)}else if(event.key.toLowerCase()==="s"&&state.active){event.preventDefault();toggle(state.active)}});
$("source-note").textContent=payload.counts.completions+" completions · "+payload.counts.tasks+" tasks · source "+payload.meta.sourceSha256.slice(0,12);
$("rail-note").textContent=payload.meta.claimBoundary;
renderAll();
</script>
</body>
</html>
'''


def render_review_html(payload: Mapping[str, Any]) -> str:
    """Render a self-contained page whose data never enters executable HTML."""

    payload_bytes = canonical_json(payload)
    encoded = base64.b64encode(payload_bytes).decode("ascii")
    payload_sha256 = hashlib.sha256(payload_bytes).hexdigest()
    return (
        _HTML_HEAD
        + _HTML_SCRIPT.replace("__PAYLOAD__", encoded).replace(
            "__PAYLOAD_SHA__", payload_sha256
        )
    )


def build_review_packet(
    *,
    source_path: Path,
    destination: Path,
    expected_source_sha256: str | None = EXPECTED_SOURCE_SHA256,
) -> dict[str, Any]:
    """Build review.html and its binding manifest without partial output."""

    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    rows, source_sha256 = read_source_rows(
        source_path, expected_source_sha256=expected_source_sha256
    )
    payload = assemble_review_payload(
        rows, source_sha256=source_sha256, source_name=source_path.name
    )
    payload_bytes = canonical_json(payload)
    html = render_review_html(payload).encode("utf-8")
    temporary = destination.parent / f".{destination.name}.building-{uuid.uuid4().hex}"
    temporary.mkdir(parents=True)
    try:
        _atomic_write_bytes(temporary / "review.html", html)
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": "initial_exploratory_candidate_review",
            "source": {
                "name": source_path.name,
                "sha256": source_sha256,
                "authoritative_sha256": expected_source_sha256,
            },
            "scope": {
                "outright_source_types": sorted(OUTRIGHT_SOURCE_TYPES),
                "deduplication_key": ["target_model", "prompt", "cot"],
                "excluded_model_policy": (
                    "lowercase target_model containing qwen is removed before payload assembly"
                ),
            },
            "counts": payload["counts"],
            "payload_sha256": hashlib.sha256(payload_bytes).hexdigest(),
            "page_sha256": hashlib.sha256(html).hexdigest(),
            "files": {
                "review.html": {
                    "sha256": hashlib.sha256(html).hexdigest(),
                    "bytes": len(html),
                }
            },
            "claim_boundaries": [
                "This is a human browsing and selection packet, not an evaluation.",
                "Source labels are displayed verbatim and are not independently validated here.",
                "Selection does not establish faithfulness, causal mechanism, or neuron meaning.",
                "No graph tracing, ADAG clustering, neuron labeling, scoring, or ranking occurs.",
            ],
        }
        manifest["manifest_sha256"] = canonical_sha256(manifest)
        _atomic_write_json(temporary / "manifest.json", manifest)
        os.replace(temporary, destination)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
