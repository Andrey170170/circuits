#!/usr/bin/env python3
"""Execute the frozen direct-Responses coarse qualification exactly once."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from circuits.analysis.bonafide.coarse_sampling_openai_run import (
    run_direct_qualification,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qualification-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-authorized-cost-usd", type=float, required=True)
    parser.add_argument("--authorization-note", required=True)
    args = parser.parse_args()
    manifest = run_direct_qualification(
        qualification_root=args.qualification_root.resolve(),
        output_root=args.output.resolve(),
        maximum_authorized_cost_usd=args.maximum_authorized_cost_usd,
        authorization_note=args.authorization_note,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
