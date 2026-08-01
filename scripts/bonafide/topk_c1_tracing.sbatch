#!/bin/bash
#SBATCH --job-name=bonafide-topk-c1
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

# Execute one frozen C1 independent-candidate wave. This launcher does not
# authorize C2 or a matched full-corpus run.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ ! -f "${REPO_ROOT}/scripts/chpc_env.sh" ]]; then
    echo "Submit this job from the frozen circuits worktree; got ${REPO_ROOT}." >&2
    exit 2
fi
cd "$REPO_ROOT"

: "${MANIFEST:?Set MANIFEST to an absolute C1 top-k manifest path}"
: "${WAVE:?Set WAVE to exactly one C1 top-k wave ID}"
: "${UV_PROJECT_ENVIRONMENT:?Set a lane-specific absolute uv environment}"
if [[ "$MANIFEST" != /* ]] || [[ ! -f "$MANIFEST" ]]; then
    echo "MANIFEST must be an existing absolute path; got ${MANIFEST}." >&2
    exit 2
fi
if [[ ! "$WAVE" =~ ^c1-independent-candidate-[0-5]-(dense|broad)$ ]]; then
    echo "WAVE is not an approved C1 independent wave: ${WAVE}." >&2
    exit 2
fi
if [[ "$UV_PROJECT_ENVIRONMENT" != /* ]]; then
    echo "UV_PROJECT_ENVIRONMENT must be absolute." >&2
    exit 2
fi
TOPK_UV_PROJECT_ENVIRONMENT="$UV_PROJECT_ENVIRONMENT"

module load python/3.12.12 uv/0.11.14
# shellcheck disable=SC1091
source scripts/chpc_env.sh
export UV_PROJECT_ENVIRONMENT="$TOPK_UV_PROJECT_ENVIRONMENT"

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
if [[ "$MANIFEST_PHASE" != "c1_policy_resource" ]]; then
    echo "This launcher rejects non-C1 phase ${MANIFEST_PHASE}." >&2
    exit 2
fi
TRACE_FAMILY_ID="$(
    "$PYTHON_BIN" -c \
        'import json, pathlib, sys; print(json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))["trace_family"]["trace_family_id"])' \
        "$MANIFEST"
)"
if [[ ! "$TRACE_FAMILY_ID" =~ ^bonafide[.]c1[.]independent-candidate-[0-5][.]v1$ ]]; then
    echo "This launcher rejects trace family ${TRACE_FAMILY_ID}." >&2
    exit 2
fi

CONFIG="${CONFIG:-${REPO_ROOT}/scripts/bonafide/configs/qwen3_4b_instruct.json}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${CIRCUITS_RESULTS_DIR}/bonafide/downstream/top5-c1-v1}"
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
    echo "Forwarding Slurm USR1 warning to top-k runner pid=${runner_pid}." >&2
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

"$PYTHON_BIN" -m scripts.bonafide.topk_runner "${runner_args[@]}" &
runner_pid=$!
set +e
wait "$runner_pid"
runner_status=$?
set -e
exit "$runner_status"
