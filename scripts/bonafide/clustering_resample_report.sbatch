#!/bin/bash
#SBATCH --job-name=bonafide-cluster-resample-report
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
: "${RESAMPLE_PLAN:?Set RESAMPLE_PLAN to the frozen resample plan}"
: "${RESAMPLE_REPORT:?Set RESAMPLE_REPORT to the output JSON path}"
: "${CLUSTER_ENV:?Set CLUSTER_ENV to the frozen environment}"
export UV_PROJECT_ENVIRONMENT="$CLUSTER_ENV"
module load python/3.12.12 uv/0.11.14
# shellcheck disable=SC1091
source scripts/chpc_env.sh
"$CLUSTER_ENV/bin/python" \
    -m scripts.bonafide.clustering_resample_report \
    --plan "$RESAMPLE_PLAN" \
    --output "$RESAMPLE_REPORT"
