#!/bin/bash
#SBATCH --job-name=bonafide-topk-c2
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --clusters=notchpeak
#SBATCH --partition=marasovic-gpu-np
#SBATCH --qos=marasovic-gpu-np
#SBATCH --account=marasovic-gpu-np
#SBATCH --gres=gpu:a100:1
#SBATCH --signal=B:USR1@300

# Execute exactly one frozen C2 independent-candidate wave.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"

: "${MANIFEST:?Set MANIFEST to an absolute C2 manifest path}"
: "${MANIFEST_SHA256:?Set MANIFEST_SHA256 to its frozen digest}"
: "${WAVE:?Set WAVE to exactly one C2 wave ID}"
: "${UV_PROJECT_ENVIRONMENT:?Set a lane-specific absolute uv environment}"
if [[ "$MANIFEST" != /* ]] || [[ ! -f "$MANIFEST" ]]; then
    echo "MANIFEST must be an existing absolute path." >&2
    exit 2
fi
if [[ ! "$MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]] ||
   [[ "$(sha256sum "$MANIFEST" | awk '{print $1}')" != "$MANIFEST_SHA256" ]]; then
    echo "C2 manifest hash drift." >&2
    exit 2
fi
if [[ ! "$WAVE" =~ ^c2-independent-candidate-[0-5]-(dense|broad)-[0-9]{2}$ ]]; then
    echo "WAVE is not an approved C2 independent wave: ${WAVE}." >&2
    exit 2
fi
if [[ "$UV_PROJECT_ENVIRONMENT" != /* ]]; then
    echo "UV_PROJECT_ENVIRONMENT must be absolute." >&2
    exit 2
fi
C2_UV_PROJECT_ENVIRONMENT="$UV_PROJECT_ENVIRONMENT"

module load python/3.12.12 uv/0.11.14
# shellcheck disable=SC1091
source scripts/chpc_env.sh
export UV_PROJECT_ENVIRONMENT="$C2_UV_PROJECT_ENVIRONMENT"
PYTHON_BIN="${UV_PROJECT_ENVIRONMENT}/bin/python"

readarray -t MANIFEST_METADATA < <(
    "$PYTHON_BIN" - "$MANIFEST" <<'PY'
import json
import pathlib
import sys

manifest = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
print(manifest["phase"])
print(manifest["trace_family"]["trace_family_id"])
PY
)
if [[ "${MANIFEST_METADATA[0]}" != "c2_scientific_utility" ]] ||
   [[ ! "${MANIFEST_METADATA[1]}" =~ ^bonafide[.]c2[.]independent-candidate-[0-5][.]v1$ ]]; then
    echo "This launcher rejects the supplied phase or trace family." >&2
    exit 2
fi

CONFIG="${CONFIG:-${REPO_ROOT}/scripts/bonafide/configs/qwen3_4b_instruct.json}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${CIRCUITS_RESULTS_DIR}/bonafide/downstream/top5-c2-v1}"
ATTEMPT_ID="${SLURM_JOB_ID}-r${SLURM_RESTART_COUNT:-0}"
SUMMARY_JSONL="${SUMMARY_JSONL:-${ARTIFACT_ROOT}/execution-summaries/${WAVE}/${ATTEMPT_ID}.jsonl}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "job_id=${SLURM_JOB_ID} node=${SLURMD_NODENAME} wave=${WAVE}"
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

"$PYTHON_BIN" -m scripts.bonafide.topk_runner \
    --config "$CONFIG" \
    --manifest "$MANIFEST" \
    --wave "$WAVE" \
    --artifact-root "$ARTIFACT_ROOT" \
    --summary-jsonl "$SUMMARY_JSONL" &
runner_pid=$!
set +e
wait "$runner_pid"
runner_status=$?
set -e
exit "$runner_status"
