#!/bin/bash
#SBATCH --job-name=bonafide-adag-bench
#SBATCH --output=slurm-%x-%j.out
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

# Run one explicitly selected benchmark wave on a lab-owned A100. Owner jobs
# may preempt guest jobs currently borrowing notch369/notch370. For example:
#
#   sbatch --export=ALL,MANIFEST=/path/to/manifest.json,WAVE=wave1-mixed-final-token \
#     scripts/bonafide/benchmark_tracing.sbatch
#
# Optional exports: EXECUTION_MODE=trace|probe, CONFIG, ARTIFACT_ROOT,
# SUMMARY_JSONL, ONLY_ARTIFACT_ID. The default remains full tracing.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ ! -f "${REPO_ROOT}/scripts/chpc_env.sh" ]]; then
    echo "Submit this job from the circuits repository root; got ${REPO_ROOT}." >&2
    exit 2
fi
cd "$REPO_ROOT"

module load python/3.12.12 uv/0.11.14
# shellcheck disable=SC1091
source scripts/chpc_env.sh

: "${MANIFEST:?Set MANIFEST to an absolute benchmark-manifest path}"
: "${WAVE:?Set WAVE to exactly one wave ID from the manifest}"

CONFIG="${CONFIG:-${REPO_ROOT}/scripts/bonafide/configs/qwen3_4b_instruct.json}"
EXECUTION_MODE="${EXECUTION_MODE:-trace}"
case "$EXECUTION_MODE" in
    trace)
        RUNNER_MODULE="scripts.bonafide.runner"
        ARTIFACT_ROOT="${ARTIFACT_ROOT:-${CIRCUITS_RESULTS_DIR}/bonafide/performance}"
        SUMMARY_JSONL="${SUMMARY_JSONL:-${ARTIFACT_ROOT}/benchmark-summary.jsonl}"
        ;;
    probe)
        RUNNER_MODULE="scripts.bonafide.probe_runner"
        ARTIFACT_ROOT="${ARTIFACT_ROOT:-${CIRCUITS_RESULTS_DIR}/bonafide/probes}"
        SUMMARY_JSONL="${SUMMARY_JSONL:-${ARTIFACT_ROOT}/probe-summary.jsonl}"
        ;;
    *)
        echo "EXECUTION_MODE must be trace or probe; got ${EXECUTION_MODE}." >&2
        exit 2
        ;;
esac
PYTHON_BIN="${UV_PROJECT_ENVIRONMENT}/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python environment is missing at $PYTHON_BIN; run uv sync --frozen first." >&2
    exit 2
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "job_id=${SLURM_JOB_ID} node=${SLURMD_NODENAME} mode=${EXECUTION_MODE} wave=${WAVE}"
echo "config=${CONFIG} manifest=${MANIFEST} artifact_root=${ARTIFACT_ROOT}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

GPU_MONITOR=/uufs/chpc.utah.edu/sys/installdir/chpcscripts/gpu/gpu-monitor.sh
monitor_pid=""
if [[ -x "$GPU_MONITOR" ]]; then
    "$GPU_MONITOR" &
    monitor_pid=$!
fi
cleanup() {
    if [[ -n "$monitor_pid" ]]; then
        kill "$monitor_pid" 2>/dev/null || true
        wait "$monitor_pid" 2>/dev/null || true
    fi
}
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

"$PYTHON_BIN" -m "$RUNNER_MODULE" "${runner_args[@]}"
