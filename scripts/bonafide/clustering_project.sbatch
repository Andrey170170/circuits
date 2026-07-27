#!/bin/bash
#SBATCH --job-name=bonafide-cluster-project
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
: "${MULTIPLEX_ROOT:?Set MULTIPLEX_ROOT to the compacted multiplex}"
: "${PROJECTION_ROOT:?Set PROJECTION_ROOT to the output directory}"
: "${CLUSTER_ENV:?Set CLUSTER_ENV to the frozen environment}"
for path in "$SOURCE_PLAN" "$MULTIPLEX_ROOT/manifest.json" "$CLUSTER_ENV/bin/python"; do
    if [[ "$path" != /* ]] || [[ ! -e "$path" ]]; then
        echo "Required input must be an existing absolute path: $path" >&2
        exit 2
    fi
done
export UV_PROJECT_ENVIRONMENT="$CLUSTER_ENV"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
module load python/3.12.12 uv/0.11.14
# shellcheck disable=SC1091
source scripts/chpc_env.sh
"$CLUSTER_ENV/bin/python" -m scripts.bonafide.clustering_project \
    --source-plan "$SOURCE_PLAN" \
    --structural-report "$STRUCTURAL_REPORT" \
    --multiplex-root "$MULTIPLEX_ROOT" \
    --output-root "$PROJECTION_ROOT"
