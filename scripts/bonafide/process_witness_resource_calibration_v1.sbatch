#!/bin/bash
#SBATCH --job-name=pw-resource-cal-v1
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

# Execute exactly one increasing-context strict-T5 resource-calibration wave.
# Submit the six waves as afterok dependencies so a failed resource tier blocks
# every higher tier. This launcher does not select a trace corpus or inspect
# graph semantics.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"

: "${CALIBRATION_ROOT:?Set CALIBRATION_ROOT to the immutable calibration root}"
: "${CALIBRATION_MANIFEST_FILE_SHA256:?Set the frozen manifest.json file SHA-256}"
: "${EXPECTED_GIT_COMMIT:?Set EXPECTED_GIT_COMMIT to the frozen launch commit}"
: "${WAVE:?Set WAVE to exactly one calibration wave ID}"
: "${UV_PROJECT_ENVIRONMENT:?Set a lane-specific absolute uv environment}"
if [[ "$CALIBRATION_ROOT" != /* ]] || [[ ! -d "$CALIBRATION_ROOT" ]]; then
    echo "CALIBRATION_ROOT must be an existing absolute directory." >&2
    exit 2
fi
if [[ ! "$CALIBRATION_MANIFEST_FILE_SHA256" =~ ^[0-9a-f]{64}$ ]] ||
   [[ "$(sha256sum "$CALIBRATION_ROOT/manifest.json" | awk '{print $1}')" != "$CALIBRATION_MANIFEST_FILE_SHA256" ]]; then
    echo "Resource-calibration manifest file hash drift." >&2
    exit 2
fi
if [[ ! "$EXPECTED_GIT_COMMIT" =~ ^[0-9a-f]{40}$ ]] ||
   [[ "$(git rev-parse HEAD)" != "$EXPECTED_GIT_COMMIT" ]] ||
   [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    echo "Resource calibration requires the exact clean frozen Git commit." >&2
    exit 2
fi
if [[ ! "$WAVE" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "WAVE must be a filesystem-safe calibration wave ID." >&2
    exit 2
fi
if [[ "$UV_PROJECT_ENVIRONMENT" != /* ]]; then
    echo "UV_PROJECT_ENVIRONMENT must be absolute." >&2
    exit 2
fi
CALIBRATION_UV_PROJECT_ENVIRONMENT="$UV_PROJECT_ENVIRONMENT"

module load python/3.12.12 uv/0.11.14
# shellcheck disable=SC1091
source scripts/chpc_env.sh
export UV_PROJECT_ENVIRONMENT="$CALIBRATION_UV_PROJECT_ENVIRONMENT"
PYTHON_BIN="${UV_PROJECT_ENVIRONMENT}/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Frozen lane Python environment is missing at ${PYTHON_BIN}." >&2
    exit 2
fi

"$PYTHON_BIN" -m scripts.bonafide.build_process_witness_resource_calibration_v1 \
    validate --root "$CALIBRATION_ROOT"

MANIFEST="$CALIBRATION_ROOT/trace-manifest.json"
CONFIG="$CALIBRATION_ROOT/run-config.json"
"$PYTHON_BIN" -c \
    'import json,pathlib,sys; m=json.loads(pathlib.Path(sys.argv[1]).read_text()); w=sys.argv[2]; matches=[x for x in m["waves"] if x["wave_id"]==w]; assert len(matches)==1 and len(matches[0]["items"])==5' \
    "$MANIFEST" "$WAVE"

ARTIFACT_ROOT="${ARTIFACT_ROOT:-${CIRCUITS_RESULTS_DIR}/process_witness/resource-calibration-v1/traces}"
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
    echo "Forwarding Slurm USR1 warning to strict-T5 runner pid=${runner_pid}." >&2
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
if [[ "$runner_status" -ne 0 ]]; then
    exit "$runner_status"
fi

"$PYTHON_BIN" -c \
    'import json,pathlib,sys
manifest=json.loads(pathlib.Path(sys.argv[1]).read_text())
wave=next(w for w in manifest["waves"] if w["wave_id"]==sys.argv[2])
expected={item["artifact_id"] for item in wave["items"]}
rows=[json.loads(line) for line in pathlib.Path(sys.argv[3]).read_text().splitlines() if line.strip()]
bad=[row for row in rows if row.get("status") not in {"complete","skipped_complete"}]
seen={row.get("source_width1_artifact_id") for row in rows if row.get("status") in {"complete","skipped_complete"}}
if bad or seen != expected or len(rows) != len(expected):
    raise SystemExit("calibration wave stopped, failed, or has incomplete accounting")' \
    "$MANIFEST" "$WAVE" "$SUMMARY_JSONL"
