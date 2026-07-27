#!/bin/bash
#SBATCH --job-name=bonafide-cluster-finalize
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --clusters=granite
#SBATCH --partition=coe-class-grn
#SBATCH --qos=coe-students-grn
#SBATCH --account=marasovic

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"
: "${SOURCE_PLAN:?Set SOURCE_PLAN to the completed clustering plan}"
: "${STRUCTURAL_REPORT:?Set STRUCTURAL_REPORT to the structural report}"
: "${PROJECTION_ROOT:?Set PROJECTION_ROOT to the multiplex projection}"
: "${RESAMPLE_PLAN:?Set RESAMPLE_PLAN to the resampling plan}"
: "${RESAMPLE_REPORT:?Set RESAMPLE_REPORT to the resampling report}"
: "${SELECTED_ROOT:?Set SELECTED_ROOT to the selected-state output directory}"
: "${CLUSTER_ENV:?Set CLUSTER_ENV to the frozen environment}"
for path in \
    "$SOURCE_PLAN" \
    "$STRUCTURAL_REPORT" \
    "$PROJECTION_ROOT/manifest.json" \
    "$RESAMPLE_PLAN" \
    "$RESAMPLE_REPORT" \
    "$CLUSTER_ENV/bin/python"
do
    if [[ "$path" != /* ]] || [[ ! -e "$path" ]]; then
        echo "Required input must be an existing absolute path: $path" >&2
        exit 2
    fi
done
if [[ "$SELECTED_ROOT" != /* ]] || [[ -e "$SELECTED_ROOT" ]]; then
    echo "SELECTED_ROOT must be an absent absolute path: $SELECTED_ROOT" >&2
    exit 2
fi
export UV_PROJECT_ENVIRONMENT="$CLUSTER_ENV"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
module load python/3.12.12 uv/0.11.14
# shellcheck disable=SC1091
source scripts/chpc_env.sh
"$CLUSTER_ENV/bin/python" -m scripts.bonafide.clustering_finalize \
    --source-plan "$SOURCE_PLAN" \
    --structural-report "$STRUCTURAL_REPORT" \
    --projection-root "$PROJECTION_ROOT" \
    --resample-plan "$RESAMPLE_PLAN" \
    --resample-report "$RESAMPLE_REPORT" \
    --output-root "$SELECTED_ROOT"
