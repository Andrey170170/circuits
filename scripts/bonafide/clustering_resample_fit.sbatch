#!/bin/bash
#SBATCH --job-name=bonafide-cluster-resample-fit
#SBATCH --output=logs/slurm-%x-%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --clusters=granite
#SBATCH --partition=coe-class-grn
#SBATCH --qos=coe-students-grn
#SBATCH --account=marasovic

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"
: "${RESAMPLE_PLAN:?Set RESAMPLE_PLAN to an absolute resample plan}"
: "${CLUSTER_ENV:?Set CLUSTER_ENV to the frozen clustering environment}"
: "${SLURM_ARRAY_TASK_ID:?This launcher must run as an array task}"
if [[ "$RESAMPLE_PLAN" != /* ]] || [[ ! -f "$RESAMPLE_PLAN" ]]; then
    echo "RESAMPLE_PLAN must be an existing absolute file." >&2
    exit 2
fi
if [[ "$CLUSTER_ENV" != /* ]] || [[ ! -x "${CLUSTER_ENV}/bin/python" ]]; then
    echo "CLUSTER_ENV must be an existing absolute Python environment." >&2
    exit 2
fi
export UV_PROJECT_ENVIRONMENT="$CLUSTER_ENV"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
module load python/3.12.12 uv/0.11.14
# shellcheck disable=SC1091
source scripts/chpc_env.sh
"${UV_PROJECT_ENVIRONMENT}/bin/python" \
    -m scripts.bonafide.clustering_resample_fit \
    --plan "$RESAMPLE_PLAN" \
    --task-index "$SLURM_ARRAY_TASK_ID"
