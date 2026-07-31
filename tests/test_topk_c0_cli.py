from __future__ import annotations

from circuits.tracing.artifact import save_topk_compact_trace
from scripts.bonafide.topk_c0_compare import run_c0_plan, save_c0_report
from tests.test_topk_topology_comparison import _joint_and_references


def test_saved_c0_plan_compares_joint_and_independent_artifacts(tmp_path) -> None:
    joint, references, _ = _joint_and_references()
    joint_path = tmp_path / "joint"
    save_topk_compact_trace(joint_path, joint, manifest={"artifact_id": "joint"})
    reference_paths = []
    for index, reference in enumerate(references):
        path = tmp_path / f"reference-{index}"
        save_topk_compact_trace(
            path, reference, manifest={"artifact_id": f"reference-{index}"}
        )
        reference_paths.append(str(path))
    plan = {
        "schema_version": "bonafide-topk-c0-comparison-plan/v1",
        "cases": [
            {
                "case_id": "c0-case-1",
                "joint_artifact_path": str(joint_path),
                "independent_artifact_paths": reference_paths,
            }
        ],
    }

    report = run_c0_plan(plan)

    assert report["case_count"] == 1
    comparison = report["results"][0]["comparison"]
    assert comparison["union_edge_recall"] == 5 / 6
    assert len(report["results"][0]["independent_artifacts"]) == 5
    output = tmp_path / "report.json"
    save_c0_report(output, report)
    assert output.is_file()
