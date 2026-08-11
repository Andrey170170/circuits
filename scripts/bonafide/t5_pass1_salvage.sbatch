#!/bin/bash
#SBATCH --job-name=bonafide-t5-salvage
#SBATCH --output=logs/slurm-%x-%A_%a.out
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

# Run one task from a frozen T5 pass-one salvage plan. The orchestration code
# validates the plan, then starts the original frozen trace runner separately
# for each artifact so one bad trace cannot suppress later retry items.

set -euo pipefail

ORCHESTRATION_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ORCHESTRATION_ROOT"

: "${SALVAGE_PLAN:?Set SALVAGE_PLAN to an absolute salvage-plan path}"
: "${SALVAGE_PLAN_SHA256:?Set SALVAGE_PLAN_SHA256 to its exact file digest}"
: "${EXPECTED_ORCHESTRATION_COMMIT:?Set the clean orchestration commit}"
: "${UV_PROJECT_ENVIRONMENT:?Set the frozen pass-one uv environment}"
TASK_INDEX="${TASK_INDEX:-${SLURM_ARRAY_TASK_ID:-}}"

if [[ "$SALVAGE_PLAN" != /* ]] || [[ ! -f "$SALVAGE_PLAN" ]]; then
    echo "SALVAGE_PLAN must be an existing absolute file." >&2
    exit 2
fi
if [[ ! "$SALVAGE_PLAN_SHA256" =~ ^[0-9a-f]{64}$ ]] ||
   [[ "$(sha256sum "$SALVAGE_PLAN" | awk '{print $1}')" != "$SALVAGE_PLAN_SHA256" ]]; then
    echo "T5 salvage-plan hash drift." >&2
    exit 2
fi
if [[ ! "$TASK_INDEX" =~ ^[0-9]+$ ]]; then
    echo "TASK_INDEX or SLURM_ARRAY_TASK_ID must be a nonnegative integer." >&2
    exit 2
fi
if [[ ! "$EXPECTED_ORCHESTRATION_COMMIT" =~ ^[0-9a-f]{40}$ ]] ||
   [[ "$(git rev-parse HEAD)" != "$EXPECTED_ORCHESTRATION_COMMIT" ]] ||
   [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    echo "T5 salvage requires its exact clean orchestration commit." >&2
    exit 2
fi
if [[ "$UV_PROJECT_ENVIRONMENT" != /* ]]; then
    echo "UV_PROJECT_ENVIRONMENT must be absolute." >&2
    exit 2
fi
SALVAGE_UV_PROJECT_ENVIRONMENT="$UV_PROJECT_ENVIRONMENT"

module load python/3.12.12 uv/0.11.14
# shellcheck disable=SC1091
source scripts/chpc_env.sh
export UV_PROJECT_ENVIRONMENT="$SALVAGE_UV_PROJECT_ENVIRONMENT"
PYTHON_BIN="${UV_PROJECT_ENVIRONMENT}/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Frozen pass-one Python is unavailable: ${PYTHON_BIN}" >&2
    exit 2
fi

ATTEMPT_ID="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${TASK_INDEX}-r${SLURM_RESTART_COUNT:-0}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "job_id=${SLURM_JOB_ID} task=${TASK_INDEX} salvage_plan=${SALVAGE_PLAN}"
echo "orchestration_root=${ORCHESTRATION_ROOT} uv_environment=${UV_PROJECT_ENVIRONMENT}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

GPU_MONITOR=/uufs/chpc.utah.edu/sys/installdir/chpcscripts/gpu/gpu-monitor.sh
monitor_pid=""
executor_pid=""
if [[ -x "$GPU_MONITOR" ]]; then
    (
        cd "${ORCHESTRATION_ROOT}/logs"
        exec "$GPU_MONITOR"
    ) &
    monitor_pid=$!
fi
forward_usr1() {
    if [[ -n "$executor_pid" ]] && kill -0 "$executor_pid" 2>/dev/null; then
        kill -USR1 "$executor_pid" 2>/dev/null || true
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

"$PYTHON_BIN" -m scripts.bonafide.run_t5_pass1_salvage \
    --plan "$SALVAGE_PLAN" \
    --plan-sha256 "$SALVAGE_PLAN_SHA256" \
    --task-index "$TASK_INDEX" \
    --attempt-id "$ATTEMPT_ID" &
executor_pid=$!
set +e
wait "$executor_pid"
executor_status=$?
set -e
exit "$executor_status"
