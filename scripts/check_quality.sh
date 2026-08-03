#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"
source scripts/chpc_env.sh
export UV_CONFIG_FILE="${UV_CONFIG_FILE:-/dev/null}"

uv lock --check
uv run --frozen --no-sync ruff check .
uv run --frozen --no-sync ruff format --check \
    circuits/analysis/bonafide \
    circuits/frontend \
    circuits/labeling \
    circuits/tracing \
    scripts/bonafide \
    tests
uv run --frozen --no-sync ty check
