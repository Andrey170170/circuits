from __future__ import annotations

from circuits.tracing.artifact import (
    save_compact_trace,
    save_topk_compact_trace,
)
from scripts.bonafide.topk_parity import (
    run_parity_plan,
    save_parity_report,
)
from tests.test_topk_parity import _legacy_and_candidate


def _saved_pair(tmp_path):
    legacy, candidate = _legacy_and_candidate()
    width1_path = tmp_path / "width1"
    topk_path = tmp_path / "topk"
    save_compact_trace(
        width1_path,
        legacy,
        manifest={
            "artifact_id": "width1-runtime-artifact",
            "source_artifact_id": "width1-source-selection",
        },
    )
    save_topk_compact_trace(
        topk_path,
        candidate,
        manifest={
            "artifact_id": "topk-artifact",
            "source_width1_artifact_id": "width1-source-selection",
        },
    )
    return {
        "schema_version": "bonafide-topk-k1-parity-plan/v1",
        "pairs": [
            {
                "pair_id": "pair-1",
                "width1_artifact_path": str(width1_path),
                "topk_artifact_path": str(topk_path),
            }
        ],
    }


def test_saved_k1_parity_plan_passes_and_records_payload_hashes(tmp_path) -> None:
    report = run_parity_plan(_saved_pair(tmp_path))

    assert report["all_passed"] is True
    assert report["pair_count"] == 1
    assert (
        report["results"][0]["width1_runtime_artifact_id"] == "width1-runtime-artifact"
    )
    assert len(report["results"][0]["width1_payload_sha256"]) == 64
    assert len(report["results"][0]["topk_payload_sha256"]) == 64


def test_parity_report_is_atomic_and_never_overwritten(tmp_path) -> None:
    report = run_parity_plan(_saved_pair(tmp_path))
    output = tmp_path / "reports" / "parity.json"

    save_parity_report(output, report)

    assert output.is_file()
    assert not list(output.parent.glob(".parity.json.tmp-*"))
    try:
        save_parity_report(output, report)
    except FileExistsError:
        pass
    else:
        raise AssertionError("parity report was unexpectedly overwritten")
