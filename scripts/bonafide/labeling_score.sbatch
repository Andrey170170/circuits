#!/bin/bash
#SBATCH --job-name=bonafide-label-score
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

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ ! -f "${REPO_ROOT}/scripts/chpc_env.sh" ]]; then
    echo "Submit from a circuits repository root; got ${REPO_ROOT}." >&2
    exit 2
fi
cd "$REPO_ROOT"
REPO_ROOT="$(pwd -P)"

: "${RUN_ROOT:?Set RUN_ROOT to one prepared labeling run root}"
: "${LABELING_ENV:?Set LABELING_ENV to the frozen circuits labeling environment}"
: "${PHASE:?Set PHASE to candidate_selection or summary_audit}"
if [[ "$RUN_ROOT" != /* ]] || [[ ! -f "$RUN_ROOT/manifest.json" ]]; then
    echo "RUN_ROOT must be an absolute prepared labeling run." >&2
    exit 2
fi
if [[ "$LABELING_ENV" != /* ]] || [[ ! -x "$LABELING_ENV/bin/python" ]]; then
    echo "LABELING_ENV must be an absolute Python environment." >&2
    exit 2
fi
if [[ "$PHASE" != "candidate_selection" ]] && [[ "$PHASE" != "summary_audit" ]]; then
    echo "PHASE must be candidate_selection or summary_audit." >&2
    exit 2
fi
SHARED_ENV="/scratch/general/vast/${USER}/circuits/envs/circuits-py312"
if [[ "$LABELING_ENV" == "$SHARED_ENV" ]] && [[ "${ALLOW_SHARED_ENV:-0}" != "1" ]]; then
    echo "Production scoring refuses the shared mutable environment." >&2
    exit 2
fi

export UV_PROJECT_ENVIRONMENT="$LABELING_ENV"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
module load python/3.12.12 uv/0.11.14
# shellcheck disable=SC1091
source scripts/chpc_env.sh

"$LABELING_ENV/bin/python" scripts/bonafide/labeling_pipeline.py score-local \
    --run-root "$RUN_ROOT" \
    --phase "$PHASE"
