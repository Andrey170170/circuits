"""Non-blind qualitative commentary packet for the frozen coarse audit draw."""

from __future__ import annotations

import ast
import hashlib
import json
import shutil
import stat
import subprocess
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from circuits.analysis.bonafide.canonical import canonical_sha256, file_sha256
from circuits.analysis.bonafide.coarse_sampling_post_campaign_v1 import (
    _publish_no_replace,
    _writable_tree,
    load_frozen_post_campaign_analysis,
)
from circuits.analysis.bonafide.coarse_sampling_post_campaign_v2 import (
    load_frozen_post_campaign_sampling_v2,
)
from circuits.labeling.io import (
    atomic_write_bytes,
    atomic_write_json,
    read_jsonl,
)

PACKET_SCHEMA = "adag.process-witness.coarse-commentary-audit-packet.v1"
ITEM_SCHEMA = "adag.process-witness.coarse-commentary-audit-item.v1"
EXPORT_SCHEMA = "adag.process-witness.coarse-commentary-audit-decision.v1"
UI_VERSION = "process-witness-coarse-commentary-audit-ui-v1"
STATUS = "frozen_nonblind_qualitative_commentary_packet"
CLAIM_BOUNDARY = (
    "This non-blind packet supports qualitative commentary on frozen coarse "
    "model proposals. It is not an accuracy estimate, acceptance or pass gate, "
    "semantic truth ledger, trace selection, ADAG adequacy result, motif or "
    "witness result, or faithfulness judgment. Review completion is optional. "
    "V2 supplement pool memberships only enrich cases already selected by the "
    "frozen v1 audit; those pool frames were not separately sampled and add no "
    "cases."
)

TAG_DEFINITIONS = {
    "active_task_work": (
        "The unit performs an operation that creates new task state or evidence, "
        "such as calculation, transformation, retrieval, traversal, lookup, "
        "comparison, selection, counting, or state update."
    ),
    "evaluation_or_revision": (
        "The unit primarily assesses, rejects, validates, corrects, or replaces "
        "an already available candidate, result, or state. Merely planning to "
        "check is not evaluation."
    ),
    "intermediate_commitment": (
        "The unit primarily reports or settles a derived non-final result or state "
        "without performing its derivation or evaluation in that same unit."
    ),
    "final_answer": "The unit commits or serializes the terminal answer.",
    "other_semantic_text": (
        "The unit plans future work, explains, restates, quotes, or comments without "
        "creating new task state or evidence."
    ),
    "surface_or_control": (
        "The unit carries no proposition by itself and consists only of formatting, "
        "punctuation, whitespace/control, tags, or other structural material."
    ),
    "uncertain": (
        "Two classifications remain defensible after applying the trajectory-effect "
        "rules, or the supplied unit boundary prevents a meaningful decision. Do "
        "not force a label."
    ),
}
DECISION_PRECEDENCE = (
    "final_answer",
    "evaluation_or_revision",
    "active_task_work",
    "intermediate_commitment",
    "other_semantic_text",
    "surface_or_control",
    "uncertain",
)
BOUNDARY_DEFINITIONS = {
    "split_needed": "The unit combines distinct roles and should be split.",
    "merge_previous": "The unit depends on text belonging to the preceding unit.",
    "merge_next": "The unit depends on text belonging to the following unit.",
    "meaning_unclear": "The visible role remains unclear even with full context.",
}
DISPOSITIONS = (
    "looks_fine",
    "discuss",
    "likely_correction",
    "unclear",
)
EXPECTED_FILES = {"documents.jsonl", "items.jsonl", "packet.json", "review.html"}
PRODUCTION_SOURCE = {
    "analysis_manifest_sha256": (
        "610e8765095551bbaadea643ca372e03254416f286bbfb8475b7df14ab501a0b"
    ),
    "sampling_manifest_sha256": (
        "5d2a49a14123ed819ab404c3da8b4633eab55d8e30cf6996c7e9544c3bfc7089"
    ),
    "blind_audit_sha256": (
        "2efaff2a6c9ad466d5eff3f5a385934b54bd2aa788380dcdeb6c20771d58ab4f"
    ),
    "audit_reveal_sha256": (
        "e831d52fae3e457d5f0650a847b531ea5c59438a379f3dfb6a42ea6c85df525b"
    ),
    "audit_plan_sha256": (
        "0623fb841703ec6df89953270a086e70d3ac3305b176af70dee6ca1fcd550f1b"
    ),
    "response_contexts_sha256": (
        "dafd6607eb99e350b0d8650a5e4dbf229cbbe4086d73d914517bba616fa9d94f"
    ),
    "audit_supplement_pools_sha256": (
        "7532b3ace8824778d32debc3e6ae164eaf157037b69bcb0c0c5611834c892cef"
    ),
    "audit_supplement_plan_sha256": (
        "bb7bec9fc72bf34cecc864ef28efd42726757f9798daf7d2d1abc02cbfd8a544"
    ),
}
PRODUCTION_COUNTS = {
    "audit_draws": 541,
    "documents": 188,
    "items": 608,
    "single_atom_draws": 490,
    "multi_atom_draws": 51,
    "complete_proposals": 596,
    "insufficient_proposals": 12,
    "physical_votes": 1809,
    "vote_origins": {
        "conservative_exact_id_salvage": 75,
        "provider_schema_valid": 1734,
    },
}
EXECUTION_ENTRY_PATHS = (
    "circuits/analysis/bonafide/coarse_sampling_commentary_audit_v1.py",
    "scripts/bonafide/build_process_witness_coarse_commentary_audit_v1.py",
)
EXECUTION_DATA_PATHS = (
    "scripts/bonafide/configs/process_witness_coarse_production_v1.json",
    "pyproject.toml",
    "uv.lock",
)
ARTIFACT_BINDING_PLACEHOLDER = "__PWCOARSECOMMENTARYARTIFACTBINDINGSHA256__"


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _verify_self_hash(value: Mapping[str, Any], field: str, label: str) -> None:
    payload = dict(value)
    observed = payload.pop(field, None)
    if observed != canonical_sha256(payload):
        raise ValueError(f"{label} self-hash drift")


def _reject_symlinks(root: Path) -> None:
    if root.is_symlink() or any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("commentary audit packet may not contain symlinks")


def _readonly_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def _module_path(repo: Path, module: str) -> Path | None:
    relative = Path(*module.split("."))
    direct = repo / relative.with_suffix(".py")
    if direct.is_file():
        return direct
    package = repo / relative / "__init__.py"
    return package if package.is_file() else None


def _package_initializers(repo: Path, module: str) -> tuple[Path, ...]:
    parts = module.split(".")
    initializers = []
    for length in range(1, len(parts) + 1):
        candidate = repo.joinpath(*parts[:length], "__init__.py")
        if candidate.is_file():
            initializers.append(candidate)
    return tuple(initializers)


def _runtime_source_paths(repo: Path | None = None) -> tuple[str, ...]:
    """Return the static in-repository import closure for packet execution."""

    repo = repo or Path(__file__).resolve().parents[3]
    pending = [repo / relative for relative in EXECUTION_ENTRY_PATHS]
    observed: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in observed:
            continue
        if not path.is_file():
            raise ValueError(f"commentary audit execution source absent: {path}")
        observed.add(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.add(node.module)
                modules.update(f"{node.module}.{alias.name}" for alias in node.names)
        for module in modules:
            if not module.startswith("circuits"):
                continue
            pending.extend(
                initializer
                for initializer in _package_initializers(repo, module)
                if initializer not in observed
            )
            dependency = _module_path(repo, module)
            if dependency is not None and dependency not in observed:
                pending.append(dependency)
    observed.update(repo / relative for relative in EXECUTION_DATA_PATHS)
    return tuple(sorted(path.relative_to(repo).as_posix() for path in observed))


def _execution_source_revision() -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[3]
    source_paths = _runtime_source_paths(repo)
    status = subprocess.run(
        [
            "git",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            *source_paths,
        ],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise ValueError("commentary audit execution source is dirty or untracked")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    files = []
    for relative in source_paths:
        path = repo / relative
        blob = subprocess.run(
            ["git", "rev-parse", f"HEAD:{relative}"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
                "git_blob_sha1": blob,
            }
        )
    return {
        "repository_commit": commit,
        "repository_tree": tree,
        "tracked_source_clean": True,
        "files": files,
    }


def _fine_majority(proposal: Mapping[str, Any]) -> str | None:
    votes = list(map(str, proposal.get("fine_votes", [])))
    if not votes:
        return None
    ordered = Counter(votes).most_common()
    if len(ordered) > 1 and ordered[0][1] == ordered[1][1]:
        return None
    return ordered[0][0]


def _compact_proposal(proposal: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "proposal_status": proposal["proposal_status"],
        "source": proposal["source"],
        "assignment_route": proposal["assignment_route"],
        "fine_majority": _fine_majority(proposal),
        "fine_agreement_pattern": proposal["fine_agreement_pattern"],
        "fine_vote_histogram": proposal["fine_vote_histogram"],
        "broad_majority": proposal["broad_majority"],
        "broad_agreement_pattern": proposal["broad_agreement_pattern"],
        "broad_vote_histogram": proposal["broad_vote_histogram"],
        "replica_coverage": proposal["replica_coverage"],
        "missing_replica_indices": proposal["missing_replica_indices"],
        "physical_votes": [
            {
                "replica_index": vote["replica_index"],
                "tag": vote["tag"],
                "confidence": vote["confidence"],
                "boundary_concerns": vote["boundary_concerns"],
                "boundary_note": vote["boundary_note"],
                "vote_origin": vote["vote_origin"],
            }
            for vote in proposal["physical_votes"]
        ],
        "fragment_of": proposal.get("fragment_of"),
        "sequence_index": proposal["sequence_index"],
    }


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (
            json.dumps(
                dict(row),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        for row in rows
    )


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _artifact_binding_identity(
    *,
    packet_without_artifact_binding: Mapping[str, Any],
    documents_bytes: bytes,
    items_bytes: bytes,
    normalized_html_bytes: bytes,
    execution_source_revision: Mapping[str, Any],
) -> dict[str, str]:
    return {
        "packet_without_artifact_binding_sha256": canonical_sha256(
            packet_without_artifact_binding
        ),
        "documents_jsonl_sha256": _bytes_sha256(documents_bytes),
        "items_jsonl_sha256": _bytes_sha256(items_bytes),
        "normalized_review_html_sha256": _bytes_sha256(normalized_html_bytes),
        "execution_source_revision_sha256": canonical_sha256(execution_source_revision),
    }


def _validate_item_semantics(
    item: Mapping[str, Any],
    document: Mapping[str, Any],
    source_bindings: Mapping[str, Any],
) -> None:
    identity = {
        "schema_version": ITEM_SCHEMA,
        "blind_audit_sha256": source_bindings["blind_audit_sha256"],
        "audit_id": item.get("audit_id"),
        "unit_id": item.get("unit_id"),
    }
    expected_id = f"pwcoarsecommentv1-{canonical_sha256(identity)[:32]}"
    if (
        item.get("schema_version") != ITEM_SCHEMA
        or item.get("blind_audit_sha256") != source_bindings["blind_audit_sha256"]
        or item.get("item_id") != expected_id
    ):
        raise ValueError("commentary audit item identity drift")
    response = list(str(document.get("full_response")))
    span = item.get("core_character_span")
    covering = item.get("covering_character_span")
    token = item.get("token_span")
    if (
        not isinstance(span, list)
        or len(span) != 2
        or not all(isinstance(value, int) for value in span)
        or not (0 <= span[0] <= span[1] <= len(response))
        or not isinstance(covering, list)
        or len(covering) != 2
        or not all(isinstance(value, int) for value in covering)
        or not (0 <= covering[0] <= span[0] <= span[1] <= covering[1] <= len(response))
        or not isinstance(token, list)
        or len(token) != 2
        or not all(isinstance(value, int) for value in token)
        or not (0 <= token[0] <= token[1])
        or "".join(response[span[0] : span[1]]) != item.get("text")
    ):
        raise ValueError("commentary audit item span/text drift")
    draw_spans = item.get("draw_target_spans")
    if not isinstance(draw_spans, list) or len(draw_spans) != item.get(
        "targets_in_draw"
    ):
        raise ValueError("commentary audit draw spans drift")
    current = 0
    current_present = False
    for row in sorted(draw_spans, key=lambda value: value["core_character_span"]):
        other = row.get("core_character_span")
        if (
            not isinstance(other, list)
            or len(other) != 2
            or not all(isinstance(value, int) for value in other)
            or not (current <= other[0] <= other[1] <= len(response))
        ):
            raise ValueError("commentary audit draw spans drift")
        current = other[1]
        current_present |= row.get("unit_id") == item.get("unit_id") and other == span
    if not current_present:
        raise ValueError("commentary audit draw spans drift")


def _source_bindings(
    *, analysis_root: Path, sampling_root: Path
) -> tuple[dict[str, str], dict[str, Any], dict[str, Any]]:
    analysis = load_frozen_post_campaign_analysis(analysis_root)
    sampling = load_frozen_post_campaign_sampling_v2(
        sampling_root, parent_v1_root=analysis_root
    )
    analysis_manifest = analysis["manifest"]
    sampling_manifest = sampling["manifest"]
    bindings = {
        "analysis_manifest_sha256": analysis_manifest["manifest_sha256"],
        "sampling_manifest_sha256": sampling_manifest["manifest_sha256"],
        "blind_audit_sha256": file_sha256(analysis_root / "blind-audit.jsonl"),
        "audit_reveal_sha256": file_sha256(analysis_root / "audit-reveal.jsonl"),
        "audit_plan_sha256": file_sha256(analysis_root / "audit-plan.json"),
        "response_contexts_sha256": file_sha256(
            analysis_root / "source-evidence/response-contexts.jsonl"
        ),
        "audit_supplement_pools_sha256": file_sha256(
            sampling_root / "audit-supplement-pools.jsonl"
        ),
        "audit_supplement_plan_sha256": file_sha256(
            sampling_root / "audit-supplement-plan.json"
        ),
    }
    return bindings, analysis, sampling


def _assemble_payload(
    *,
    analysis_root: Path,
    sampling_root: Path,
    source_bindings: Mapping[str, str],
) -> dict[str, Any]:
    blind = read_jsonl(analysis_root / "blind-audit.jsonl")
    reveal = read_jsonl(analysis_root / "audit-reveal.jsonl")
    if [row.get("audit_id") for row in blind] != [
        row.get("audit_id") for row in reveal
    ]:
        raise ValueError("blind/reveal audit order or membership drift")
    supplements = read_jsonl(sampling_root / "audit-supplement-pools.jsonl")
    pools_by_psu: dict[str, list[str]] = {}
    for row in supplements:
        psu_id = str(row["psu_id"])
        if psu_id in pools_by_psu:
            raise ValueError("duplicate supplement PSU membership")
        pools_by_psu[psu_id] = list(map(str, row["pool_ids"]))

    documents_by_id: dict[str, dict[str, Any]] = {}
    items: list[dict[str, Any]] = []
    complete = 0
    vote_origins: Counter[str] = Counter()
    for draw_index, (blind_row, reveal_row) in enumerate(
        zip(blind, reveal, strict=True)
    ):
        targets = blind_row["targets"]
        proposals = reveal_row["proposals"]
        if len(targets) != len(proposals) or [row["unit_id"] for row in targets] != [
            row["unit_id"] for row in proposals
        ]:
            raise ValueError("audit target/proposal alignment drift")
        response_id = str(blind_row["response_id"])
        prompt = str(blind_row["task_prompt"])
        response = str(blind_row["full_response"])
        document = {
            "response_id": response_id,
            "task_prompt": prompt,
            "task_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "full_response": response,
            "full_response_sha256": hashlib.sha256(response.encode()).hexdigest(),
        }
        prior = documents_by_id.setdefault(response_id, document)
        if prior != document:
            raise ValueError("audit response context drift")
        draw_spans = [
            {
                "unit_id": row["unit_id"],
                "core_character_span": row["core_character_span"],
            }
            for row in targets
        ]
        for target_index, (target, proposal) in enumerate(
            zip(targets, proposals, strict=True)
        ):
            identity = {
                "schema_version": ITEM_SCHEMA,
                "blind_audit_sha256": source_bindings["blind_audit_sha256"],
                "audit_id": blind_row["audit_id"],
                "unit_id": target["unit_id"],
            }
            item_id = f"pwcoarsecommentv1-{canonical_sha256(identity)[:32]}"
            compact = _compact_proposal(proposal)
            complete += compact["proposal_status"] == "complete"
            vote_origins.update(
                str(vote["vote_origin"]) for vote in compact["physical_votes"]
            )
            items.append(
                {
                    **identity,
                    "item_id": item_id,
                    "draw_index": draw_index,
                    "target_index_within_draw": target_index,
                    "targets_in_draw": len(targets),
                    "response_id": response_id,
                    "psu_id": reveal_row["psu_id"],
                    "route": reveal_row["route"],
                    "strata": list(reveal_row["strata"]),
                    "probability_base_inclusion_probability": reveal_row[
                        "probability_base_inclusion_probability"
                    ],
                    "supplement_pool_ids": pools_by_psu.get(
                        str(reveal_row["psu_id"]), []
                    ),
                    "text": target["text"],
                    "token_span": target["token_span"],
                    "core_character_span": target["core_character_span"],
                    "covering_character_span": target["covering_character_span"],
                    "draw_target_spans": draw_spans,
                    "model_proposal": compact,
                }
            )
    if len({row["unit_id"] for row in items}) != len(items):
        raise ValueError("audit commentary unit identity reused")
    if len({row["psu_id"] for row in reveal}) != len(reveal):
        raise ValueError("audit commentary PSU identity reused")
    counts = {
        "audit_draws": len(blind),
        "documents": len(documents_by_id),
        "items": len(items),
        "single_atom_draws": sum(len(row["targets"]) == 1 for row in blind),
        "multi_atom_draws": sum(len(row["targets"]) > 1 for row in blind),
        "complete_proposals": complete,
        "insufficient_proposals": len(items) - complete,
        "physical_votes": sum(vote_origins.values()),
        "vote_origins": dict(sorted(vote_origins.items())),
    }
    packet_identity = {
        "schema_version": PACKET_SCHEMA,
        "source_bindings": dict(source_bindings),
        "ui_version": UI_VERSION,
        "item_ids_in_order": [row["item_id"] for row in items],
        "tag_definitions": TAG_DEFINITIONS,
        "decision_precedence": list(DECISION_PRECEDENCE),
        "boundary_definitions": BOUNDARY_DEFINITIONS,
        "dispositions": list(DISPOSITIONS),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    packet_binding = canonical_sha256(packet_identity)
    packet = {
        **packet_identity,
        "packet_binding_sha256": packet_binding,
        "packet_id": f"process-witness-coarse-commentary-audit-v1-{packet_binding[:16]}",
        "status": STATUS,
        "counts": counts,
        "qualitative_only": True,
        "completion_required": False,
        "model_proposals_visible": True,
        "export_schema": EXPORT_SCHEMA,
    }
    return {
        "packet": packet,
        "documents": list(documents_by_id.values()),
        "items": items,
    }


def render_commentary_audit_html(payload: Mapping[str, Any]) -> str:
    """Render a self-contained, non-blind commentary UI."""

    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    tags = json.dumps(list(TAG_DEFINITIONS))
    concerns = json.dumps(list(BOUNDARY_DEFINITIONS))
    dispositions = json.dumps(list(DISPOSITIONS))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Coarse proposal commentary audit</title>
<style>
:root{{--ink:#17211f;--muted:#66716e;--line:#d7deda;--paper:#f4f6f3;--card:#fff;--accent:#126a55;--focus:#9ee4cf;--group:#ffedb0;--warn:#9b461b}}*{{box-sizing:border-box}}html,body{{margin:0;min-height:100%;font:15px/1.5 system-ui,sans-serif;background:var(--paper);color:var(--ink)}}button,input,select,textarea{{font:inherit}}button,select,input,textarea{{border:1px solid #aab5b1;border-radius:6px;background:#fff;padding:7px 9px}}button{{cursor:pointer}}header{{position:sticky;top:0;z-index:5;background:#18372f;color:#fff;padding:10px 16px;display:flex;gap:8px;align-items:center;flex-wrap:wrap}}header strong{{margin-right:auto}}.toolbar{{display:flex;gap:7px;align-items:center;flex-wrap:wrap}}main{{max-width:1900px;margin:14px auto;padding:0 14px;display:grid;grid-template-columns:minmax(0,1fr) 390px 330px;gap:14px}}.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}}.side{{position:sticky;top:var(--header-offset,100px);align-self:start;max-height:calc(100vh - var(--header-offset,100px) - 16px);overflow:auto}}.meta,.claim{{color:var(--muted);font-size:13px}}.prompt,.response{{white-space:pre-wrap;word-break:break-word;overflow:auto;background:#f7f8f6;border:1px solid #e1e5e2;border-radius:7px;padding:12px}}.prompt{{max-height:210px}}.response{{max-height:60vh;line-height:1.65}}.group{{background:var(--group)}}.focus{{background:var(--focus);outline:2px solid var(--accent);scroll-margin:150px 0}}.proposal{{border-left:4px solid var(--accent);padding-left:12px}}.votes{{display:grid;gap:8px}}.vote{{background:#f7f8f6;border:1px solid var(--line);border-radius:7px;padding:8px}}.chip{{display:inline-block;background:#e4ece8;border-radius:999px;padding:2px 7px;margin:2px;font-size:12px}}.warn{{color:var(--warn)}}label{{display:block;font-weight:650;margin:10px 0 4px}}textarea{{width:100%;min-height:130px}}select{{max-width:250px}}.checks label{{display:flex;gap:7px;font-weight:400;margin:5px 0}}.reviewed{{display:flex;gap:8px;align-items:center;font-weight:750}}.actions{{display:flex;gap:8px;flex-wrap:wrap;margin-top:12px}}dl{{margin:0}}dt{{font-weight:750;color:var(--accent);margin-top:10px}}dd{{margin:2px 0 0}}@media(max-width:1050px){{main{{grid-template-columns:1fr}}.side{{position:static;max-height:none}}header{{position:static}}}}
</style></head><body>
<header><strong>Frozen coarse proposals · qualitative commentary</strong><span id="packet"></span><span id="progress"></span><div class="toolbar"><button id="prev">Previous</button><input id="jump" type="number" min="1" style="width:72px"><button id="next">Next</button><select id="reviewFilter"><option value="all">All</option><option value="unreviewed">Unreviewed</option><option value="reviewed">Reviewed</option></select><select id="priorityFilter"><option value="all">All strata/pools</option></select><select id="labelFilter"><option value="all">All proposal labels</option></select><input id="search" type="search" placeholder="Search text or ID"><button id="export">Export commentary JSONL</button></div></header>
<main><section><div class="card"><div class="meta" id="meta"></div><h3>Complete task prompt</h3><div class="prompt" id="prompt"></div><h3>Complete exact response</h3><div class="response" id="response"></div></div></section>
<aside class="card side"><h3>Model proposal</h3><div class="proposal" id="proposal"></div><h4>Replica votes</h4><div class="votes" id="votes"></div><hr><h3>Your commentary</h3><p class="meta">Nothing here is treated as a correctness score. Mark reviewed only when you have looked at this atom; every field is optional.</p><label class="reviewed"><input id="reviewed" type="checkbox">Reviewed</label><label>Disposition<select id="disposition"><option value="">No disposition</option></select></label><label>Suggested primary label<select id="suggested"><option value="">No correction</option></select></label><fieldset class="checks"><legend>Suggested boundary correction</legend><div id="boundaries"></div></fieldset><label>Comment<textarea id="comment" placeholder="Observation, question, rationale, or proposed correction"></textarea></label><div class="actions"><button id="save">Save</button><button id="saveNext">Save &amp; next</button></div><p class="meta" id="saveStatus"></p><p class="claim">{CLAIM_BOUNDARY}</p></aside>
<aside class="card side"><h3>Label reference</h3><p class="meta" id="precedence"></p><dl id="definitions"></dl><h3>Boundary reference</h3><dl id="boundaryDefinitions"></dl></aside></main>
<script>
const DATA={data},TAGS={tags},CONCERNS={concerns},DISPOSITIONS={dispositions};const packet=DATA.packet,docs=new Map(DATA.documents.map(d=>[d.response_id,d])),el=id=>document.getElementById(id);const key=`coarse-commentary:${{packet.packet_id}}:${{packet.ui_version}}:${{packet.artifact_binding_sha256}}`;let state={{index:0,decisions:{{}}}},visible=[...DATA.items.keys()];
try{{const saved=JSON.parse(localStorage.getItem(key));if(saved?.packet_id===packet.packet_id&&saved.state?.decisions)state=saved.state}}catch(e){{}}
function persist(){{localStorage.setItem(key,JSON.stringify({{packet_id:packet.packet_id,state}}))}}function addOptions(root,values){{for(const value of values){{const option=document.createElement('option');option.value=value;option.textContent=value;root.append(option)}}}}addOptions(el('disposition'),DISPOSITIONS);addOptions(el('suggested'),TAGS);addOptions(el('labelFilter'),TAGS);
const priorities=[...new Set(DATA.items.flatMap(item=>[...item.strata,...item.supplement_pool_ids]))].sort();addOptions(el('priorityFilter'),priorities);function checks(root,values){{for(const value of values){{const label=document.createElement('label'),input=document.createElement('input');input.type='checkbox';input.value=value;label.append(input,document.createTextNode(value));root.append(label)}}}}checks(el('boundaries'),CONCERNS);
function definitions(root,values){{for(const [term,description] of Object.entries(values)){{const dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=term;dd.textContent=description;root.append(dt,dd)}}}}definitions(el('definitions'),packet.tag_definitions);definitions(el('boundaryDefinitions'),packet.boundary_definitions);el('precedence').textContent=`When a composite unit contains multiple trajectory effects, apply the frozen precedence: ${{packet.decision_precedence.join(' → ')}}.`;el('packet').textContent=packet.packet_id;
function current(){{return visible.includes(state.index)?DATA.items[state.index]:null}}function visiblePosition(){{return visible.indexOf(state.index)}}function decisionFor(item){{return state.decisions[item.item_id]||{{reviewed:false,disposition:'',suggested_primary_label:'',suggested_boundary_concerns:[],comment:''}}}}function refreshVisible(){{const review=el('reviewFilter').value,priority=el('priorityFilter').value,label=el('labelFilter').value,query=el('search').value.trim().toLowerCase(),old=Math.max(0,visiblePosition());visible=DATA.items.map((_,i)=>i).filter(i=>{{const item=DATA.items[i],d=decisionFor(item),reviewOK=review==='all'||(review==='reviewed'&&d.reviewed)||(review==='unreviewed'&&!d.reviewed),priorityOK=priority==='all'||item.strata.includes(priority)||item.supplement_pool_ids.includes(priority),labelOK=label==='all'||item.model_proposal.fine_majority===label,hay=[item.item_id,item.audit_id,item.psu_id,item.unit_id,item.text,docs.get(item.response_id).task_prompt,docs.get(item.response_id).full_response].join('\\n').toLowerCase();return reviewOK&&priorityOK&&labelOK&&(!query||hay.includes(query))}});if(visible.length&&!visible.includes(state.index))state.index=visible[Math.min(old,visible.length-1)]}}
function renderResponse(item,doc){{const root=el('response');root.textContent='';const spans=[...item.draw_target_spans].sort((a,b)=>a.core_character_span[0]-b.core_character_span[0]),chars=Array.from(doc.full_response);let cursor=0;for(const span of spans){{const [a,b]=span.core_character_span;root.append(document.createTextNode(chars.slice(cursor,a).join('')));const mark=document.createElement('span');mark.className='group'+(span.unit_id===item.unit_id?' focus':'');mark.textContent=chars.slice(a,b).join('');root.append(mark);cursor=b}}root.append(document.createTextNode(chars.slice(cursor).join('')));setTimeout(()=>root.querySelector('.focus')?.scrollIntoView({{block:'center'}}),0)}}
function renderProposal(item){{const p=item.model_proposal,root=el('proposal');root.textContent='';for(const [name,value] of [['fine majority',p.fine_majority??'none'],['fine agreement',p.fine_agreement_pattern],['broad majority',p.broad_majority],['status',p.proposal_status],['route',item.route],['PSU',item.psu_id],['PSU-level probability-base inclusion',item.probability_base_inclusion_probability??'diagnostic only'],['replicas',`${{p.replica_coverage}}; missing ${{p.missing_replica_indices.join(', ')||'none'}}`]]){{const div=document.createElement('div'),strong=document.createElement('strong');strong.textContent=name+': ';div.append(strong,document.createTextNode(String(value)));root.append(div)}}for(const value of [...item.strata,...item.supplement_pool_ids]){{const chip=document.createElement('span');chip.className='chip';chip.textContent=value;root.append(chip)}}const votes=el('votes');votes.textContent='';if(!p.physical_votes.length)votes.textContent='No exact provider votes available.';for(const vote of p.physical_votes){{const div=document.createElement('div');div.className='vote';div.textContent=`replica ${{vote.replica_index}} · ${{vote.tag}} · ${{vote.confidence}} · origin ${{vote.vote_origin}}`;if(vote.boundary_concerns.length)div.append(document.createElement('br'),document.createTextNode(`boundary: ${{vote.boundary_concerns.join(', ')}}`));if(vote.boundary_note)div.append(document.createElement('br'),document.createTextNode(vote.boundary_note));votes.append(div)}}}}
function render(){{const pos=visiblePosition();el('prev').disabled=pos<=0;el('next').disabled=pos<0||pos>=visible.length-1;el('progress').textContent=`${{visible.length?pos+1:0}} / ${{visible.length}} visible · ${{Object.values(state.decisions).filter(d=>d.reviewed).length}} / ${{DATA.items.length}} reviewed`;if(!visible.length){{el('meta').textContent='No items match the current filters.';el('prompt').textContent='';el('response').textContent='';return}}const item=current(),doc=docs.get(item.response_id),d=decisionFor(item);el('jump').value=pos+1;el('jump').max=visible.length;el('meta').textContent=`draw ${{item.draw_index+1}} / ${{packet.counts.audit_draws}} · atom ${{item.target_index_within_draw+1}} / ${{item.targets_in_draw}} · ${{item.unit_id}} · tokens [${{item.token_span.join(', ')}})`;el('prompt').textContent=doc.task_prompt;renderResponse(item,doc);renderProposal(item);el('reviewed').checked=!!d.reviewed;el('disposition').value=d.disposition||'';el('suggested').value=d.suggested_primary_label||'';el('comment').value=d.comment||'';for(const input of el('boundaries').querySelectorAll('input'))input.checked=(d.suggested_boundary_concerns||[]).includes(input.value);el('saveStatus').textContent=''}}
function captureCurrent(){{const item=current();if(!item)return null;const decision={{reviewed:el('reviewed').checked,disposition:el('disposition').value,suggested_primary_label:el('suggested').value,suggested_boundary_concerns:[...el('boundaries').querySelectorAll('input:checked')].map(i=>i.value),comment:el('comment').value}};state.decisions[item.item_id]=decision;persist();return item}}function save(advance){{const originalItem=captureCurrent();if(!originalItem)return;const old=visiblePosition(),original=state.index;refreshVisible();if(advance&&visible.length){{const next=visible.includes(original)?old+1:old;state.index=visible[Math.min(Math.max(0,next),visible.length-1)]}}persist();render();el('saveStatus').textContent='Saved locally.'}}el('save').onclick=()=>save(false);el('saveNext').onclick=()=>save(true);el('prev').onclick=()=>{{const p=visiblePosition();if(p>0)state.index=visible[p-1];persist();render()}};el('next').onclick=()=>{{const p=visiblePosition();if(p>=0&&p<visible.length-1)state.index=visible[p+1];persist();render()}};el('jump').onchange=()=>{{if(!visible.length)return;state.index=visible[Math.max(0,Math.min(visible.length-1,Number(el('jump').value)-1))];persist();render()}};for(const id of ['reviewFilter','priorityFilter','labelFilter','search'])el(id).oninput=()=>{{refreshVisible();persist();render()}};
function download(name,text){{const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([text],{{type:'application/x-ndjson'}}));a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}}el('export').onclick=()=>{{captureCurrent();const exportedAt=new Date().toISOString(),rows=DATA.items.map(item=>{{const decision=decisionFor(item);return JSON.stringify({{schema_version:'{EXPORT_SCHEMA}',packet_id:packet.packet_id,packet_binding_sha256:packet.packet_binding_sha256,artifact_binding_sha256:packet.artifact_binding_sha256,item_id:item.item_id,audit_id:item.audit_id,psu_id:item.psu_id,response_id:item.response_id,unit_id:item.unit_id,reviewed:!!decision.reviewed,disposition:decision.disposition||'',suggested_primary_label:decision.suggested_primary_label||'',suggested_boundary_concerns:decision.suggested_boundary_concerns||[],comment:decision.comment||'',exported_at:exportedAt,qualitative_only:true}})}});download(packet.packet_id+'-commentary.jsonl',rows.join('\\n')+'\\n')}};
function headerOffset(){{document.documentElement.style.setProperty('--header-offset',`${{document.querySelector('header').offsetHeight+12}}px`)}}window.addEventListener('resize',headerOffset);headerOffset();refreshVisible();render();
</script></body></html>"""


def build_commentary_audit_packet(
    *, analysis_root: Path, sampling_root: Path, destination: Path
) -> dict[str, Any]:
    """Build an immutable commentary packet without mutating either parent."""

    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    analysis_root = analysis_root.resolve()
    sampling_root = sampling_root.resolve()
    source_bindings, _, _ = _source_bindings(
        analysis_root=analysis_root, sampling_root=sampling_root
    )
    execution_source_revision = _execution_source_revision()
    payload = _assemble_payload(
        analysis_root=analysis_root,
        sampling_root=sampling_root,
        source_bindings=source_bindings,
    )
    production_matches = {
        key: source_bindings.get(key) == value
        for key, value in PRODUCTION_SOURCE.items()
    }
    if any(production_matches.values()) and not all(production_matches.values()):
        raise ValueError("commentary audit partial production source binding drift")
    if (
        all(production_matches.values())
        and payload["packet"]["counts"] != PRODUCTION_COUNTS
    ):
        raise ValueError("commentary audit production census drift")
    execution_source_revision_sha256 = canonical_sha256(execution_source_revision)
    payload["packet"]["execution_source_revision_sha256"] = (
        execution_source_revision_sha256
    )
    documents_bytes = _jsonl_bytes(payload["documents"])
    items_bytes = _jsonl_bytes(payload["items"])
    packet_without_artifact_binding = dict(payload["packet"])
    payload["packet"]["artifact_binding_sha256"] = ARTIFACT_BINDING_PLACEHOLDER
    normalized_html = render_commentary_audit_html(payload).encode("utf-8")
    artifact_identity = _artifact_binding_identity(
        packet_without_artifact_binding=packet_without_artifact_binding,
        documents_bytes=documents_bytes,
        items_bytes=items_bytes,
        normalized_html_bytes=normalized_html,
        execution_source_revision=execution_source_revision,
    )
    artifact_binding = canonical_sha256(artifact_identity)
    payload["packet"]["artifact_binding_sha256"] = artifact_binding
    html = render_commentary_audit_html(payload).encode("utf-8")
    if (
        html.replace(
            artifact_binding.encode("ascii"),
            ARTIFACT_BINDING_PLACEHOLDER.encode("ascii"),
        )
        != normalized_html
        or ARTIFACT_BINDING_PLACEHOLDER.encode("ascii") in html
    ):
        raise ValueError("commentary audit HTML artifact binding drift")
    temporary = destination.parent / f".{destination.name}.building-{uuid.uuid4().hex}"
    temporary.mkdir(parents=True)
    try:
        atomic_write_json(temporary / "packet.json", payload["packet"])
        atomic_write_bytes(temporary / "documents.jsonl", documents_bytes)
        atomic_write_bytes(temporary / "items.jsonl", items_bytes)
        atomic_write_bytes(temporary / "review.html", html)
        files = [
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in sorted(temporary.iterdir())
        ]
        manifest = {
            "schema_version": PACKET_SCHEMA,
            "status": STATUS,
            "packet_id": payload["packet"]["packet_id"],
            "packet_binding_sha256": payload["packet"]["packet_binding_sha256"],
            "artifact_binding_sha256": artifact_binding,
            "artifact_binding_identity": artifact_identity,
            "source_bindings": source_bindings,
            "execution_source_revision": execution_source_revision,
            "counts": payload["packet"]["counts"],
            "claim_boundary": CLAIM_BOUNDARY,
            "files": files,
        }
        manifest["manifest_sha256"] = canonical_sha256(manifest)
        atomic_write_json(temporary / "manifest.json", manifest)
        _readonly_tree(temporary)
        load_commentary_audit_packet(temporary)
        _publish_no_replace(temporary, destination)
        load_commentary_audit_packet(destination)
        return manifest
    except BaseException:
        if temporary.exists():
            _writable_tree(temporary)
            shutil.rmtree(temporary)
        raise


def load_commentary_audit_packet(root: Path) -> dict[str, Any]:
    """Strictly validate the standalone packet without loading parent artifacts."""

    _reject_symlinks(root)
    if stat.S_IMODE(root.stat().st_mode) != 0o555:
        raise ValueError("commentary audit root mode drift")
    if stat.S_IMODE((root / "manifest.json").stat().st_mode) != 0o444:
        raise ValueError("commentary audit manifest mode drift")
    manifest = _load_object(root / "manifest.json")
    _verify_self_hash(manifest, "manifest_sha256", "commentary audit manifest")
    if (
        manifest.get("schema_version") != PACKET_SCHEMA
        or manifest.get("status") != STATUS
        or manifest.get("claim_boundary") != CLAIM_BOUNDARY
    ):
        raise ValueError("commentary audit manifest contract drift")
    declared = {str(row.get("path")): row for row in manifest.get("files", [])}
    children = {
        path.name: path for path in root.iterdir() if path.name != "manifest.json"
    }
    observed = {name: path for name, path in children.items() if path.is_file()}
    if set(declared) != EXPECTED_FILES or set(observed) != EXPECTED_FILES:
        raise ValueError("commentary audit file membership drift")
    if set(children) != EXPECTED_FILES:
        raise ValueError("commentary audit root membership drift")
    for relative, binding in declared.items():
        path = observed[relative]
        if (
            stat.S_IMODE(path.stat().st_mode) != 0o444
            or path.stat().st_size != binding.get("bytes")
            or file_sha256(path) != binding.get("sha256")
        ):
            raise ValueError("commentary audit file drift")
    packet = _load_object(root / "packet.json")
    documents = read_jsonl(root / "documents.jsonl")
    items = read_jsonl(root / "items.jsonl")
    identity_keys = (
        "schema_version",
        "source_bindings",
        "ui_version",
        "item_ids_in_order",
        "tag_definitions",
        "decision_precedence",
        "boundary_definitions",
        "dispositions",
        "claim_boundary",
    )
    identity = {key: packet.get(key) for key in identity_keys}
    binding = canonical_sha256(identity)
    execution = manifest.get("execution_source_revision")
    execution_files = (
        execution.get("files", []) if isinstance(execution, Mapping) else []
    )
    execution_paths = [row.get("path") for row in execution_files]
    artifact_binding = packet.get("artifact_binding_sha256")
    if not isinstance(artifact_binding, str) or len(artifact_binding) != 64:
        raise ValueError("commentary audit artifact binding drift")
    html_bytes = (root / "review.html").read_bytes()
    artifact_token = artifact_binding.encode("ascii")
    normalized_html = html_bytes.replace(
        artifact_token, ARTIFACT_BINDING_PLACEHOLDER.encode("ascii")
    )
    packet_without_artifact_binding = dict(packet)
    packet_without_artifact_binding.pop("artifact_binding_sha256", None)
    artifact_identity = _artifact_binding_identity(
        packet_without_artifact_binding=packet_without_artifact_binding,
        documents_bytes=(root / "documents.jsonl").read_bytes(),
        items_bytes=(root / "items.jsonl").read_bytes(),
        normalized_html_bytes=normalized_html,
        execution_source_revision=execution if isinstance(execution, Mapping) else {},
    )
    if (
        packet.get("packet_binding_sha256") != binding
        or packet.get("packet_id")
        != f"process-witness-coarse-commentary-audit-v1-{binding[:16]}"
        or packet.get("status") != STATUS
        or packet.get("qualitative_only") is not True
        or packet.get("completion_required") is not False
        or packet.get("model_proposals_visible") is not True
        or packet.get("export_schema") != EXPORT_SCHEMA
        or packet.get("tag_definitions") != TAG_DEFINITIONS
        or packet.get("decision_precedence") != list(DECISION_PRECEDENCE)
        or packet.get("boundary_definitions") != BOUNDARY_DEFINITIONS
        or packet.get("dispositions") != list(DISPOSITIONS)
        or packet.get("item_ids_in_order") != [row.get("item_id") for row in items]
        or len({row.get("item_id") for row in items}) != len(items)
        or manifest.get("packet_binding_sha256") != binding
        or manifest.get("packet_id") != packet.get("packet_id")
        or manifest.get("source_bindings") != packet.get("source_bindings")
        or packet.get("execution_source_revision_sha256")
        != canonical_sha256(execution if isinstance(execution, Mapping) else {})
        or manifest.get("artifact_binding_sha256") != artifact_binding
        or manifest.get("artifact_binding_identity") != artifact_identity
        or canonical_sha256(artifact_identity) != artifact_binding
        or artifact_token not in html_bytes
        or ARTIFACT_BINDING_PLACEHOLDER.encode("ascii") in html_bytes
        or not isinstance(execution, Mapping)
        or execution.get("tracked_source_clean") is not True
        or not isinstance(execution.get("repository_commit"), str)
        or len(execution.get("repository_commit")) != 40
        or not isinstance(execution.get("repository_tree"), str)
        or len(execution.get("repository_tree")) != 40
        or len(execution_paths) != len(set(execution_paths))
        or not set(EXECUTION_ENTRY_PATHS).issubset(execution_paths)
    ):
        raise ValueError("commentary audit packet identity drift")
    document_ids = [str(row.get("response_id")) for row in documents]
    document_by_id = dict(zip(document_ids, documents, strict=True))
    if len(document_by_id) != len(documents):
        raise ValueError("commentary audit duplicate document")
    for document in documents:
        if hashlib.sha256(
            str(document.get("task_prompt")).encode()
        ).hexdigest() != document.get("task_prompt_sha256") or hashlib.sha256(
            str(document.get("full_response")).encode()
        ).hexdigest() != document.get("full_response_sha256"):
            raise ValueError("commentary audit document hash drift")
    complete = 0
    vote_origins: Counter[str] = Counter()
    draw_sizes: Counter[int] = Counter()
    seen_draws: set[str] = set()
    seen_psus: set[str] = set()
    unit_ids: set[str] = set()
    items_by_draw: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        if (
            item.get("schema_version") != ITEM_SCHEMA
            or item.get("response_id") not in document_by_id
            or not isinstance(item.get("model_proposal"), Mapping)
            or not isinstance(item.get("strata"), list)
            or not isinstance(item.get("supplement_pool_ids"), list)
        ):
            raise ValueError("commentary audit item contract drift")
        _validate_item_semantics(
            item, document_by_id[str(item["response_id"])], packet["source_bindings"]
        )
        audit_id = str(item["audit_id"])
        items_by_draw.setdefault(audit_id, []).append(item)
        if audit_id not in seen_draws:
            seen_draws.add(audit_id)
            psu_id = str(item["psu_id"])
            if psu_id in seen_psus:
                raise ValueError("commentary audit PSU identity reused")
            seen_psus.add(psu_id)
            draw_sizes[int(item["targets_in_draw"])] += 1
        unit_id = str(item["unit_id"])
        if unit_id in unit_ids:
            raise ValueError("commentary audit unit identity reused")
        unit_ids.add(unit_id)
        complete += item["model_proposal"].get("proposal_status") == "complete"
        vote_origins.update(
            str(vote["vote_origin"])
            for vote in item["model_proposal"].get("physical_votes", [])
        )
    for audit_id, draw_items in items_by_draw.items():
        first = draw_items[0]
        expected_size = int(first["targets_in_draw"])
        stable_fields = (
            "draw_index",
            "response_id",
            "psu_id",
            "route",
            "strata",
            "probability_base_inclusion_probability",
            "draw_target_spans",
        )
        if (
            len(draw_items) != expected_size
            or {int(row["target_index_within_draw"]) for row in draw_items}
            != set(range(expected_size))
            or any(
                row["targets_in_draw"] != expected_size
                or any(row[field] != first[field] for field in stable_fields)
                for row in draw_items
            )
        ):
            raise ValueError(f"commentary audit draw contract drift: {audit_id}")
    counts = {
        "audit_draws": len(seen_draws),
        "documents": len(documents),
        "items": len(items),
        "single_atom_draws": draw_sizes[1],
        "multi_atom_draws": sum(
            count for size, count in draw_sizes.items() if size > 1
        ),
        "complete_proposals": complete,
        "insufficient_proposals": len(items) - complete,
        "physical_votes": sum(vote_origins.values()),
        "vote_origins": dict(sorted(vote_origins.items())),
    }
    if packet.get("counts") != counts or manifest.get("counts") != counts:
        raise ValueError("commentary audit census drift")
    source = packet["source_bindings"]
    if all(source.get(key) == value for key, value in PRODUCTION_SOURCE.items()):
        if counts != PRODUCTION_COUNTS:
            raise ValueError("commentary audit production census drift")
    elif any(source.get(key) == value for key, value in PRODUCTION_SOURCE.items()):
        raise ValueError("commentary audit partial production source binding drift")
    return {
        "manifest": manifest,
        "packet": packet,
        "documents": documents,
        "items": items,
    }
