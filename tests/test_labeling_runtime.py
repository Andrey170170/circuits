from __future__ import annotations

from pathlib import Path

from circuits.labeling.runtime import resolve_local_snapshot


def test_resolve_local_snapshot_uses_exact_revision(
    monkeypatch, tmp_path: Path
) -> None:
    snapshot = (
        tmp_path
        / "models--Qwen--Example"
        / "snapshots"
        / "012345"
    )
    snapshot.mkdir(parents=True)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    assert resolve_local_snapshot("Qwen/Example", "012345") == snapshot
