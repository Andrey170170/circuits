#!/usr/bin/env python3
"""Build the compact PI-facing outright source-label audit page."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from circuits.analysis.bonafide.outright_label_audit import build_label_audit_packet

DEFAULT_SOURCE = Path("experiments/raw_graph_observatory/outright-task-review-v2")
DEFAULT_DESTINATION = Path("experiments/raw_graph_observatory/outright-label-audit-v1")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a simplified audit page from the verified v2 review packet."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    args = parser.parse_args()
    manifest = build_label_audit_packet(
        source=args.source.resolve(), destination=args.destination.resolve()
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
