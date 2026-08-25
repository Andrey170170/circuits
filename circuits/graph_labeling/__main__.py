"""CLI for graph-local occurrence labeling runs."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from circuits.graph_labeling.openai_batch import (
    abandon_openai_attempt,
    collect_openai_batch,
    openai_batch_status,
    prepare_openai_batch,
    recover_openai_batch,
    recover_openai_upload,
    submit_openai_batch,
)
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

    batch_prepare = commands.add_parser("prepare-openai-batch")
    batch_prepare.add_argument("--run-root", type=Path, required=True)
    batch_prepare.add_argument("--method-id", required=True)
    batch_prepare.add_argument("--max-cost-usd", type=float, required=True)

    batch_submit = commands.add_parser("submit-openai-batch")
    batch_submit.add_argument("--run-root", type=Path, required=True)
    batch_submit.add_argument("--method-id", required=True)
    batch_submit.add_argument("--max-cost-usd", type=float, required=True)

    batch_status = commands.add_parser("openai-batch-status")
    batch_status.add_argument("--run-root", type=Path, required=True)
    batch_status.add_argument("--method-id", required=True)

    batch_recover = commands.add_parser("recover-openai-batch")
    batch_recover.add_argument("--run-root", type=Path, required=True)
    batch_recover.add_argument("--method-id", required=True)
    batch_recover.add_argument("--batch-id", required=True)

    upload_recover = commands.add_parser("recover-openai-upload")
    upload_recover.add_argument("--run-root", type=Path, required=True)
    upload_recover.add_argument("--method-id", required=True)
    upload_recover.add_argument("--input-file-id", required=True)

    batch_abandon = commands.add_parser("abandon-openai-attempt")
    batch_abandon.add_argument("--run-root", type=Path, required=True)
    batch_abandon.add_argument("--method-id", required=True)
    batch_abandon.add_argument("--reason", required=True)

    batch_collect = commands.add_parser("collect-openai-batch")
    batch_collect.add_argument("--run-root", type=Path, required=True)
    batch_collect.add_argument("--method-id", required=True)
    batch_collect.add_argument("--finalize", action="store_true")
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
    elif args.command == "prepare-openai-batch":
        value = prepare_openai_batch(
            args.run_root, args.method_id, max_cost_usd=args.max_cost_usd
        )
    elif args.command == "submit-openai-batch":
        value = submit_openai_batch(
            args.run_root, args.method_id, max_cost_usd=args.max_cost_usd
        )
    elif args.command == "openai-batch-status":
        value = openai_batch_status(args.run_root, args.method_id)
    elif args.command == "recover-openai-batch":
        value = recover_openai_batch(args.run_root, args.method_id, args.batch_id)
    elif args.command == "recover-openai-upload":
        value = recover_openai_upload(args.run_root, args.method_id, args.input_file_id)
    elif args.command == "abandon-openai-attempt":
        value = abandon_openai_attempt(
            args.run_root, args.method_id, reason=args.reason
        )
    elif args.command == "collect-openai-batch":
        value = collect_openai_batch(
            args.run_root, args.method_id, finalize=args.finalize
        )
    else:
        value = status(args.run_root)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
