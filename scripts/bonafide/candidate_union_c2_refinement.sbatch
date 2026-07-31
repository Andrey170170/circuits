#!/bin/bash
#SBATCH --job-name=bonafide-c2-union
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --clusters=notchpeak
#SBATCH --partition=marasovic-gpu-np
#SBATCH --qos=marasovic-gpu-np
#SBATCH --account=marasovic-gpu-np
#SBATCH --gres=gpu:a100:1
#SBATCH --signal=B:USR1@300

# Rescore one frozen C2 shard on the exact pass-one node/edge union.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"

: "${PLAN:?Set PLAN to the absolute C2 candidate-union plan path}"
: "${PLAN_SHA256:?Set PLAN_SHA256 to its frozen digest}"
: "${WAVE:?Set WAVE to exactly one C2 candidate-union wave ID}"
: "${EXPECTED_GIT_COMMIT:?Set EXPECTED_GIT_COMMIT to the frozen launch commit}"
: "${UV_PROJECT_ENVIRONMENT:?Set a lane-specific absolute uv environment}"
if [[ "$PLAN" != /* ]] || [[ ! -f "$PLAN" ]]; then
    echo "PLAN must be an existing absolute path." >&2
    exit 2
fi
if [[ ! "$PLAN_SHA256" =~ ^[0-9a-f]{64}$ ]] ||
   [[ "$(sha256sum "$PLAN" | awk '{print $1}')" != "$PLAN_SHA256" ]]; then
    echo "C2 candidate-union plan hash drift." >&2
    exit 2
fi
if [[ ! "$WAVE" =~ ^candidate-union-c2-(dense|broad)-[0-9]{2}$ ]]; then
    echo "WAVE is not an approved C2 candidate-union wave: ${WAVE}." >&2
    exit 2
fi
if [[ "$UV_PROJECT_ENVIRONMENT" != /* ]]; then
    echo "UV_PROJECT_ENVIRONMENT must be absolute." >&2
    exit 2
fi
if [[ ! "$EXPECTED_GIT_COMMIT" =~ ^[0-9a-f]{40}$ ]] ||
   [[ "$(git rev-parse HEAD)" != "$EXPECTED_GIT_COMMIT" ]] ||
   [[ -n "$(git status --porcelain)" ]]; then
    echo "C2 launch requires the exact clean frozen Git commit." >&2
    exit 2
fi
C2_UNION_UV_PROJECT_ENVIRONMENT="$UV_PROJECT_ENVIRONMENT"

module load python/3.12.12 uv/0.11.14
# shellcheck disable=SC1091
source scripts/chpc_env.sh
export UV_PROJECT_ENVIRONMENT="$C2_UNION_UV_PROJECT_ENVIRONMENT"
PYTHON_BIN="${UV_PROJECT_ENVIRONMENT}/bin/python"

PLAN_PHASE="$(
    "$PYTHON_BIN" -c \
        'import json,pathlib,sys; print(json.loads(pathlib.Path(sys.argv[1]).read_text())["phase"])' \
        "$PLAN"
)"
if [[ "$PLAN_PHASE" != "c2_candidate_union_refinement" ]]; then
    echo "This launcher rejects plan phase ${PLAN_PHASE}." >&2
    exit 2
fi

readarray -t EXECUTION_METADATA < <(
    "$PYTHON_BIN" - "$PLAN" <<'PY'
import json
import pathlib
import sys

execution = json.loads(
    pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
)["execution"]
print(execution["config_path"])
print(execution["config_sha256"])
PY
)
CONFIG="${CONFIG:-${EXECUTION_METADATA[0]}}"
if [[ "$CONFIG" != /* ]] || [[ ! -f "$CONFIG" ]] ||
   [[ "$(sha256sum "$CONFIG" | awk '{print $1}')" != "${EXECUTION_METADATA[1]}" ]]; then
    echo "C2 execution config path/hash drift." >&2
    exit 2
fi
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${CIRCUITS_RESULTS_DIR}/bonafide/downstream/candidate-union-c2-v1}"
ATTEMPT_ID="${SLURM_JOB_ID}-r${SLURM_RESTART_COUNT:-0}"
SUMMARY_JSONL="${SUMMARY_JSONL:-${ARTIFACT_ROOT}/execution-summaries/${WAVE}/${ATTEMPT_ID}.jsonl}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "job_id=${SLURM_JOB_ID} node=${SLURMD_NODENAME} wave=${WAVE}"
echo "config=${CONFIG} plan=${PLAN} artifact_root=${ARTIFACT_ROOT}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

GPU_MONITOR=/uufs/chpc.utah.edu/sys/installdir/chpcscripts/gpu/gpu-monitor.sh
monitor_pid=""
runner_pid=""
if [[ -x "$GPU_MONITOR" ]]; then
    (
        cd "${REPO_ROOT}/logs"
        exec "$GPU_MONITOR"
    ) &
    monitor_pid=$!
fi
forward_usr1() {
    if [[ -n "$runner_pid" ]] && kill -0 "$runner_pid" 2>/dev/null; then
        kill -USR1 "$runner_pid"
    fi
}
cleanup() {
    if [[ -n "$monitor_pid" ]]; then
        kill "$monitor_pid" 2>/dev/null || true
        wait "$monitor_pid" 2>/dev/null || true
    fi
}
trap forward_usr1 USR1
trap cleanup EXIT

runner_args=(
    --config "$CONFIG"
    --plan "$PLAN"
    --wave "$WAVE"
    --artifact-root "$ARTIFACT_ROOT"
    --summary-jsonl "$SUMMARY_JSONL"
)
if [[ -n "${ONLY_CASE_ID:-}" ]]; then
    runner_args+=(--only-case-id "$ONLY_CASE_ID")
fi
if [[ "${PRINT_RECORDS:-0}" == "1" ]]; then
    runner_args+=(--print-records)
fi

"$PYTHON_BIN" -m scripts.bonafide.candidate_union_runner "${runner_args[@]}" &
runner_pid=$!
set +e
wait "$runner_pid"
runner_status=$?
set -e
exit "$runner_status"
