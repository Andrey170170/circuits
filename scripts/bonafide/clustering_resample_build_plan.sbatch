#!/bin/bash
#SBATCH --job-name=bonafide-cluster-resample-plan
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --clusters=granite
#SBATCH --partition=coe-class-grn
#SBATCH --qos=coe-students-grn
#SBATCH --account=marasovic

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"
: "${SOURCE_PLAN:?Set SOURCE_PLAN to the completed clustering plan}"
: "${STRUCTURAL_REPORT:?Set STRUCTURAL_REPORT to the structural report}"
: "${PROJECTION_ROOT:?Set PROJECTION_ROOT to the projection directory}"
: "${RESAMPLE_ROOT:?Set RESAMPLE_ROOT to the output directory}"
: "${RESAMPLE_PLAN:?Set RESAMPLE_PLAN to the output plan path}"
: "${CLUSTER_ENV:?Set CLUSTER_ENV to the frozen environment}"
export UV_PROJECT_ENVIRONMENT="$CLUSTER_ENV"
module load python/3.12.12 uv/0.11.14
# shellcheck disable=SC1091
source scripts/chpc_env.sh
"$CLUSTER_ENV/bin/python" \
    -m scripts.bonafide.clustering_resample_build_plan \
    --source-plan "$SOURCE_PLAN" \
    --structural-report "$STRUCTURAL_REPORT" \
    --projection-manifest "$PROJECTION_ROOT/manifest.json" \
    --output-root "$RESAMPLE_ROOT" \
    --plan "$RESAMPLE_PLAN"
