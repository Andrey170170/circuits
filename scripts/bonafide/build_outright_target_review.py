#!/usr/bin/env python3
"""Build the v2 all-model outright trace-target review packet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from circuits.analysis.bonafide.outright_review import EXPECTED_SOURCE_SHA256
from circuits.analysis.bonafide.outright_target_review import (
    DEFAULT_REGISTRY_PATH,
    build_target_review_packet,
)

DEFAULT_SOURCE = Path(
    "/uufs/chpc.utah.edu/common/home/u1653998/projects/circuits/BonaFide.csv"
)
DEFAULT_DESTINATION = Path(
    "experiments/raw_graph_observatory/outright-task-review-v2"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the static all-model exact-token target review page."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--tokenizer-registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    args = parser.parse_args()
    manifest = build_target_review_packet(
        source_path=args.source.resolve(),
        destination=args.destination.resolve(),
        registry_path=args.tokenizer_registry.resolve(),
        expected_source_sha256=EXPECTED_SOURCE_SHA256,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
