#!/bin/bash
#SBATCH --job-name=bonafide-t5-pass1
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
#SBATCH --signal=B:USR1@300

# Execute one immutable pass-one wave from a prepared T5 corpus bundle.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"

: "${BUNDLE:?Set BUNDLE to the absolute T5 pass-one bundle path}"
: "${BUNDLE_SHA256:?Set BUNDLE_SHA256 to the frozen bundle digest}"
: "${EXPECTED_GIT_COMMIT:?Set EXPECTED_GIT_COMMIT to the launch commit}"
: "${UV_PROJECT_ENVIRONMENT:?Set a lane-specific absolute uv environment}"
TASK_INDEX="${TASK_INDEX:-${SLURM_ARRAY_TASK_ID:-}}"
if [[ "$BUNDLE" != /* ]] || [[ ! -f "$BUNDLE" ]]; then
    echo "BUNDLE must be an existing absolute path." >&2
    exit 2
fi
if [[ ! "$BUNDLE_SHA256" =~ ^[0-9a-f]{64}$ ]] ||
   [[ "$(sha256sum "$BUNDLE" | awk '{print $1}')" != "$BUNDLE_SHA256" ]]; then
    echo "T5 pass-one bundle hash drift." >&2
    exit 2
fi
if [[ ! "$TASK_INDEX" =~ ^[0-9]+$ ]]; then
    echo "TASK_INDEX or SLURM_ARRAY_TASK_ID must be a nonnegative integer." >&2
    exit 2
fi
if [[ ! "$EXPECTED_GIT_COMMIT" =~ ^[0-9a-f]{40}$ ]] ||
   [[ "$(git rev-parse HEAD)" != "$EXPECTED_GIT_COMMIT" ]] ||
   [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    echo "T5 pass one requires the exact clean frozen Git commit." >&2
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

readarray -t TASK < <(
    "$PYTHON_BIN" - "$BUNDLE" "$TASK_INDEX" <<'PY'
import json
import pathlib
import sys

bundle = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
index = int(sys.argv[2])
tasks = bundle["tasks"]
if index >= len(tasks) or tasks[index]["task_index"] != index:
    raise SystemExit(f"T5 task index is outside the frozen bundle: {index}")
task = tasks[index]
print(task["manifest_path"])
print(task["manifest_sha256"])
print(task["wave_id"])
print(task["candidate_index"])
PY
)
MANIFEST="${TASK[0]}"
MANIFEST_SHA256="${TASK[1]}"
WAVE="${TASK[2]}"
CANDIDATE_INDEX="${TASK[3]}"
if [[ ! -f "$MANIFEST" ]] ||
   [[ "$(sha256sum "$MANIFEST" | awk '{print $1}')" != "$MANIFEST_SHA256" ]]; then
    echo "T5 pass-one manifest file/hash drift." >&2
    exit 2
fi

CONFIG="${CONFIG:-${REPO_ROOT}/scripts/bonafide/configs/qwen3_4b_instruct.json}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${CIRCUITS_RESULTS_DIR}/bonafide/downstream/t5-corpus-v1/pass1}"
ATTEMPT_ID="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${TASK_INDEX}-r${SLURM_RESTART_COUNT:-0}"
SUMMARY_JSONL="${SUMMARY_JSONL:-${ARTIFACT_ROOT}/execution-summaries/${WAVE}/${ATTEMPT_ID}.jsonl}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "job_id=${SLURM_JOB_ID} task=${TASK_INDEX} candidate=${CANDIDATE_INDEX} wave=${WAVE}"
echo "config=${CONFIG} manifest=${MANIFEST} artifact_root=${ARTIFACT_ROOT}"
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
