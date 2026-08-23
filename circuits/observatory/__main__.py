"""Command-line entry point for the Trace Observatory."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m circuits.observatory")
    commands = parser.add_subparsers(dest="command", required=True)

    sync = commands.add_parser(
        "sync", help="project trusted compact traces to safe JSON"
    )
    sync.add_argument("--trace-root", required=True)
    sync.add_argument("--site-root", required=True)
    sync.add_argument("--state-root", required=True)
    sync.add_argument("--tokenizer-path")
    sync.add_argument(
        "--allow-numeric-tokens",
        action="store_true",
        help="explicitly allow [token_id] labels if no matching offline tokenizer exists",
    )
    sync.add_argument(
        "--replace",
        action="store_true",
        help="atomically replace the bundle while preserving a timestamped backup",
    )

    serve = commands.add_parser("serve", help="serve safe JSON and packaged assets")
    serve.add_argument("--site-root", required=True)
    serve.add_argument("--state-root", required=True)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8032)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "sync":
        # Importing the trusted pickle/tokenizer stack is isolated to sync.
        from circuits.observatory.bundle import sync_bundle

        result = sync_bundle(
            trace_root=args.trace_root,
            site_root=args.site_root,
            state_root=args.state_root,
            tokenizer_path=args.tokenizer_path,
            allow_numeric_tokens=args.allow_numeric_tokens,
            replace=args.replace,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "serve":
        from circuits.observatory.server import serve

        serve(
            site_root=args.site_root,
            state_root=args.state_root,
            host=args.host,
            port=args.port,
        )
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
