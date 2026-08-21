from __future__ import annotations

import base64
import csv
import hashlib
import json
import re
from pathlib import Path

import pytest
from circuits.analysis.bonafide.outright_review import (
    EXPECTED_COLUMNS,
    assemble_review_payload,
    build_review_packet,
    read_source_rows,
    render_review_html,
)


def _row(
    *,
    row_id: str,
    model: str,
    prompt: str,
    cot: str,
    extract: str,
    label: str = "FAITHFUL_STEP",
    question: str = "What is two plus two?",
    source_type: str = "complex",
) -> dict[str, str]:
    start = cot.index(extract) if extract else 0
    end = start + len(extract) if extract else -1
    values = dict.fromkeys(EXPECTED_COLUMNS, "")
    values.update(
        {
            "id": row_id,
            "question_id": "task-source-id",
            "label_type": label,
            "sentence_text": extract,
            "sentence_span_start": str(start),
            "sentence_span_end": str(end),
            "extract": extract,
            "extract_span_start": str(start),
            "extract_span_end": str(end),
            "labeling_reason": "fixture reason",
            "target_model": model,
            "question": question,
            "prompt": prompt,
            "cot": cot,
            "model_answer": "4",
            "correct_answer": "4",
            "src_type": source_type,
        }
    )
    return values


def _rows() -> list[dict[str, str]]:
    unsafe = "alpha </script><script>alert('no')</script> omega"
    return [
        _row(
            row_id="a-faithful",
            model="org/Model-A",
            prompt="prompt A",
            cot=unsafe,
            extract="alpha",
        ),
        _row(
            row_id="a-unfaithful",
            model="org/Model-A",
            prompt="prompt A",
            cot=unsafe,
            extract="omega",
            label="UNFAITHFUL_COT",
        ),
        _row(
            row_id="b-faithful",
            model="org/Model-A",
            prompt="prompt B",
            cot="a second faithful completion",
            extract="faithful",
            question="Count to four.",
        ),
        _row(
            row_id="c-faithful",
            model="org/Model-B",
            prompt="prompt C",
            cot="graph reasoning",
            extract="graph",
            source_type="graph",
        ),
        _row(
            row_id="excluded",
            model="org/QwEn-hidden",
            prompt="prompt excluded",
            cot="excluded reasoning",
            extract="excluded",
        ),
        _row(
            row_id="not-outright",
            model="org/Model-C",
            prompt="prompt ignored",
            cot="ignored reasoning",
            extract="ignored",
            source_type="hinted",
        ),
    ]


def _write_csv(path: Path, rows: list[dict[str, str]]) -> str:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXPECTED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_payload_counts_dedup_annotations_and_exclusion() -> None:
    payload = assemble_review_payload(_rows(), source_sha256="f" * 64)
    assert payload["counts"]["source_rows"] == 4
    assert payload["counts"]["completions"] == 3
    assert payload["counts"]["models"] == 2
    assert payload["counts"]["excluded_source_rows"] == 1
    assert payload["counts"]["by_source_type"]["complex"]["completions"] == 2
    assert payload["counts"]["by_source_type"]["graph"]["completions"] == 1
    assert payload["counts"]["by_label_type"]["UNFAITHFUL_COT"] == {
        "source_rows": 1,
        "completions": 1,
    }
    mixed = next(item for item in payload["completions"] if item["broadLabel"] == "mixed")
    assert [item["sourceRowId"] for item in mixed["annotations"]] == [
        "a-faithful",
        "a-unfaithful",
    ]
    assert not any("qwen" in item["model"].lower() for item in payload["completions"])
    assert not any("qwen" in item["model"].lower() for item in payload["models"])


def test_content_identities_are_stable_under_row_reordering() -> None:
    first = assemble_review_payload(_rows(), source_sha256="f" * 64)
    second = assemble_review_payload(list(reversed(_rows())), source_sha256="f" * 64)
    assert {item["completionId"] for item in first["completions"]} == {
        item["completionId"] for item in second["completions"]
    }
    assert {item["taskId"] for item in first["tasks"]} == {
        item["taskId"] for item in second["tasks"]
    }


def test_html_is_base64_embedded_safe_dom_and_has_all_filters() -> None:
    payload = assemble_review_payload(_rows(), source_sha256="f" * 64)
    html = render_review_html(payload)
    assert "</script><script>alert('no')</script>" not in html
    assert ".innerHTML" not in html
    assert "textContent" in html
    for marker in (
        'data-filter="task"',
        'data-filter="source-type"',
        'data-filter="broad-label"',
        'data-filter="exact-label-type"',
        'data-filter="search"',
        'id="model-list"',
        'id="selection-only"',
        "localStorage",
        "Export JSON",
    ):
        assert marker in html
    match = re.search(r'atob\("([A-Za-z0-9+/=]+)"\)', html)
    assert match is not None
    decoded = json.loads(base64.b64decode(match.group(1)))
    assert decoded == payload


def test_source_hash_schema_and_span_drift_fail_closed(tmp_path: Path) -> None:
    source = tmp_path / "fixture.csv"
    digest = _write_csv(source, _rows())
    loaded, actual = read_source_rows(source, expected_source_sha256=digest)
    assert len(loaded) == 6
    assert actual == digest
    with pytest.raises(ValueError, match="source SHA256 drift"):
        read_source_rows(source, expected_source_sha256="0" * 64)

    broken = _rows()
    broken[0]["extract_span_start"] = "2"
    with pytest.raises(ValueError, match="extract span text drift"):
        assemble_review_payload(broken, source_sha256=digest)

    bad_schema = tmp_path / "bad-schema.csv"
    with bad_schema.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXPECTED_COLUMNS[:-1])
        writer.writeheader()
    with pytest.raises(ValueError, match="source schema drift"):
        read_source_rows(bad_schema, expected_source_sha256=None)


def test_build_is_deterministic_and_manifest_binds_page(tmp_path: Path) -> None:
    source = tmp_path / "fixture.csv"
    digest = _write_csv(source, _rows())
    first = tmp_path / "packet-a"
    second = tmp_path / "packet-b"
    manifest_a = build_review_packet(
        source_path=source,
        destination=first,
        expected_source_sha256=digest,
    )
    manifest_b = build_review_packet(
        source_path=source,
        destination=second,
        expected_source_sha256=digest,
    )
    assert (first / "review.html").read_bytes() == (second / "review.html").read_bytes()
    assert manifest_a == manifest_b
    assert manifest_a["page_sha256"] == hashlib.sha256(
        (first / "review.html").read_bytes()
    ).hexdigest()
    assert manifest_a["files"]["review.html"]["sha256"] == manifest_a["page_sha256"]
