#!/bin/bash
#SBATCH --job-name=bonafide-step0-t5
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --clusters=notchpeak
#SBATCH --partition=marasovic-gpu-np
#SBATCH --qos=marasovic-gpu-np
#SBATCH --account=marasovic-gpu-np
#SBATCH --gres=gpu:a100:1
#SBATCH --signal=B:USR1@300
#SBATCH --no-requeue

# Execute only the frozen historical Step-0 strict-T5 smoke. Both executable
# inputs must be explicit absolute paths so a submission cannot silently pick
# up another model or manifest from the submit directory.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ ! -f "${REPO_ROOT}/scripts/chpc_env.sh" ]]; then
    echo "Submit from the frozen circuits worktree; got ${REPO_ROOT}." >&2
    exit 2
fi
cd "$REPO_ROOT"

: "${CONFIG:?Set CONFIG to an existing absolute frozen run-config path}"
: "${MANIFEST:?Set MANIFEST to an existing absolute frozen T5 manifest path}"
: "${WAVE:?Set WAVE to exactly one Step-0 T5 smoke wave ID}"
: "${UV_PROJECT_ENVIRONMENT:?Set a lane-specific absolute uv environment}"
for frozen_path in "$CONFIG" "$MANIFEST"; do
    if [[ "$frozen_path" != /* ]] || [[ ! -f "$frozen_path" ]]; then
        echo "CONFIG and MANIFEST must be existing absolute paths: ${frozen_path}." >&2
        exit 2
    fi
done
if [[ ! "$WAVE" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "WAVE must be a filesystem-safe manifest wave ID; got ${WAVE}." >&2
    exit 2
fi
if [[ "$UV_PROJECT_ENVIRONMENT" != /* ]]; then
    echo "UV_PROJECT_ENVIRONMENT must be absolute." >&2
    exit 2
fi
T5_UV_PROJECT_ENVIRONMENT="$UV_PROJECT_ENVIRONMENT"

module load python/3.12.12 uv/0.11.14
# shellcheck disable=SC1091
source scripts/chpc_env.sh
export UV_PROJECT_ENVIRONMENT="$T5_UV_PROJECT_ENVIRONMENT"

PYTHON_BIN="${UV_PROJECT_ENVIRONMENT}/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Frozen lane environment is missing at ${PYTHON_BIN}." >&2
    exit 2
fi
MANIFEST_PHASE="$(
    "$PYTHON_BIN" -c \
        'import json, pathlib, sys; print(json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))["phase"])' \
        "$MANIFEST"
)"
if [[ "$MANIFEST_PHASE" != "step0_t5_smoke" ]]; then
    echo "This launcher rejects non-Step-0-T5 phase ${MANIFEST_PHASE}." >&2
    exit 2
fi

ARTIFACT_ROOT="${ARTIFACT_ROOT:-${CIRCUITS_RESULTS_DIR}/bonafide/process-witness/step0-t5-smoke-v1}"
ATTEMPT_ID="${SLURM_JOB_ID}-r${SLURM_RESTART_COUNT:-0}"
SUMMARY_JSONL="${SUMMARY_JSONL:-${ARTIFACT_ROOT}/execution-summaries/${WAVE}/${ATTEMPT_ID}.jsonl}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "job_id=${SLURM_JOB_ID} node=${SLURMD_NODENAME} wave=${WAVE}"
echo "config=${CONFIG} manifest=${MANIFEST} artifact_root=${ARTIFACT_ROOT}"
echo "uv_environment=${UV_PROJECT_ENVIRONMENT}"
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
    echo "Forwarding Slurm USR1 warning to T5 runner pid=${runner_pid}." >&2
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
    --manifest "$MANIFEST"
    --wave "$WAVE"
    --artifact-root "$ARTIFACT_ROOT"
    --summary-jsonl "$SUMMARY_JSONL"
)
if [[ -n "${ONLY_ARTIFACT_ID:-}" ]]; then
    runner_args+=(--only-artifact-id "$ONLY_ARTIFACT_ID")
fi
if [[ "${PRINT_RECORDS:-0}" == "1" ]]; then
    runner_args+=(--print-records)
fi

"$PYTHON_BIN" -m scripts.bonafide.t5_runner "${runner_args[@]}" &
runner_pid=$!
set +e
wait "$runner_pid"
runner_status=$?
set -e
exit "$runner_status"
