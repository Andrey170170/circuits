"""CLI for graph-local occurrence labeling runs."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from circuits.graph_labeling.runtime import (
    execute,
    export_overlay,
    ingest_results,
    prepare,
    status,
)
from circuits.observatory.external_labels import install_label_set


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m circuits.graph_labeling")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--spec", type=Path, required=True)
    prepare_parser.add_argument("--run-root", type=Path, required=True)

    execute_parser = commands.add_parser("execute")
    execute_parser.add_argument("--run-root", type=Path, required=True)
    execute_parser.add_argument("--method-id", required=True)
    execute_parser.add_argument("--execution", type=Path, required=True)

    ingest_parser = commands.add_parser("ingest-results")
    ingest_parser.add_argument("--run-root", type=Path, required=True)
    ingest_parser.add_argument("--method-id", required=True)
    ingest_parser.add_argument("--results-jsonl", type=Path, required=True)

    export_parser = commands.add_parser("export-overlay")
    export_parser.add_argument("--run-root", type=Path, required=True)
    export_parser.add_argument("--label-set-id", required=True)
    export_parser.add_argument("--site-root", type=Path, required=True)
    export_parser.add_argument("--destination", type=Path, required=True)

    install_parser = commands.add_parser("install-overlay")
    install_parser.add_argument("--source-site", type=Path, required=True)
    install_parser.add_argument("--label-set", type=Path, required=True)
    install_parser.add_argument("--destination-site", type=Path, required=True)

    status_parser = commands.add_parser("status")
    status_parser.add_argument("--run-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        value = prepare(args.spec, args.run_root).model_dump(mode="json")
    elif args.command == "execute":
        value = execute(args.run_root, args.method_id, args.execution).model_dump(
            mode="json"
        )
    elif args.command == "export-overlay":
        value = export_overlay(
            args.run_root,
            args.label_set_id,
            args.site_root,
            args.destination,
        ).model_dump(mode="json")
    elif args.command == "ingest-results":
        value = ingest_results(
            args.run_root, args.method_id, args.results_jsonl
        ).model_dump(mode="json")
    elif args.command == "install-overlay":
        value = install_label_set(
            args.source_site, args.label_set, args.destination_site
        )
    else:
        value = status(args.run_root)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
