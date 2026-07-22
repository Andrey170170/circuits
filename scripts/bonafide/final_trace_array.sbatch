#!/bin/bash
#SBATCH --job-name=bonafide-final-traces
#SBATCH --output=logs/slurm-%x-%A_%a.out
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
#SBATCH --array=0-11%4
#SBATCH --signal=B:USR1@300

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ ! -f "${REPO_ROOT}/scripts/chpc_env.sh" ]]; then
    echo "Submit this job from the circuits repository root; got ${REPO_ROOT}." >&2
    exit 2
fi
cd "$REPO_ROOT"

: "${EXECUTION_PLAN:?Set EXECUTION_PLAN to an absolute execution-plan path}"
if [[ "$EXECUTION_PLAN" != /* ]] || [[ ! -f "$EXECUTION_PLAN" ]]; then
    echo "EXECUTION_PLAN must be an existing absolute path; got ${EXECUTION_PLAN}." >&2
    exit 2
fi
: "${SLURM_ARRAY_TASK_ID:?This launcher must run as a Slurm array task}"

module load python/3.12.12 uv/0.11.14
# shellcheck disable=SC1091
source scripts/chpc_env.sh

PYTHON_BIN="${UV_PROJECT_ENVIRONMENT}/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python environment is missing at $PYTHON_BIN; run uv sync --frozen first." >&2
    exit 2
fi

readarray -t plan_fields < <(
    "$PYTHON_BIN" -c \
        'import json,sys; p=json.load(open(sys.argv[1])); print(p["sources"]["final_trace_manifest"]["path"]); print(p["sources"]["trace_run_config"]["path"]); print(p["plan_sha256"]); print(p["tasks"][int(sys.argv[2])]["task_kind"]); print(p["sharding"]["routine_shard_count"])' \
        "$EXECUTION_PLAN" "$SLURM_ARRAY_TASK_ID"
)
MANIFEST="${plan_fields[0]}"
CONFIG="${CONFIG:-${plan_fields[1]}}"
PLAN_SHA256="${plan_fields[2]}"
TASK_KIND="${plan_fields[3]}"
ROUTINE_SHARD_COUNT="${plan_fields[4]}"

if [[ "${ALLOW_EXTREMES:-0}" != "1" ]]; then
    if [[ "$TASK_KIND" != "routine" ]]; then
        echo "The default launcher accepts routine tasks only; got ${TASK_KIND}." >&2
        echo "Set ALLOW_EXTREMES=1 only for a separately reviewed extreme submission." >&2
        exit 2
    fi
    expected_max=$((ROUTINE_SHARD_COUNT - 1))
    if [[ "${SLURM_ARRAY_TASK_MIN}" != "0" ]] || \
       [[ "${SLURM_ARRAY_TASK_MAX}" != "$expected_max" ]] || \
       [[ "${SLURM_ARRAY_TASK_COUNT}" != "$ROUTINE_SHARD_COUNT" ]] || \
       [[ "${SLURM_ARRAY_TASK_STEP}" != "1" ]]; then
        echo "Default array shape must be 0-${expected_max} with step 1 and ${ROUTINE_SHARD_COUNT} tasks." >&2
        exit 2
    fi
fi
if [[ "$TASK_KIND" == "pathological_manual" ]] && [[ "${ALLOW_PATHOLOGICAL:-0}" != "1" ]]; then
    echo "Task ${SLURM_ARRAY_TASK_ID} is pathological_manual; set ALLOW_PATHOLOGICAL=1 explicitly." >&2
    exit 2
fi

ARTIFACT_ROOT="${ARTIFACT_ROOT:-${CIRCUITS_RESULTS_DIR}/bonafide/final-traces}"
ATTEMPT_ID="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}-r${SLURM_RESTART_COUNT:-0}"
SUMMARY_JSONL="${SUMMARY_JSONL:-${ARTIFACT_ROOT}/execution-summaries/${PLAN_SHA256}/task-$(printf '%02d' "$SLURM_ARRAY_TASK_ID")/${ATTEMPT_ID}.jsonl}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "job_id=${SLURM_JOB_ID} array_task=${SLURM_ARRAY_TASK_ID} node=${SLURMD_NODENAME} task_kind=${TASK_KIND}"
echo "plan=${EXECUTION_PLAN} plan_sha256=${PLAN_SHA256} artifact_root=${ARTIFACT_ROOT}"
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
    echo "Forwarding Slurm USR1 warning to tracing runner pid=${runner_pid}." >&2
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

"$PYTHON_BIN" -m scripts.bonafide.runner \
    --config "$CONFIG" \
    --manifest "$MANIFEST" \
    --execution-plan "$EXECUTION_PLAN" \
    --execution-task-index "$SLURM_ARRAY_TASK_ID" \
    --artifact-root "$ARTIFACT_ROOT" \
    --summary-jsonl "$SUMMARY_JSONL" &
runner_pid=$!
set +e
wait "$runner_pid"
runner_status=$?
if [[ "$runner_status" -ge 128 ]] && kill -0 "$runner_pid" 2>/dev/null; then
    wait "$runner_pid"
    runner_status=$?
fi
set -e
exit "$runner_status"
