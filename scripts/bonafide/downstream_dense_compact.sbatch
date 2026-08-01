#!/bin/bash
#SBATCH --job-name=bonafide-dense-compact
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --clusters=granite
#SBATCH --partition=coe-class-grn
#SBATCH --qos=coe-students-grn
#SBATCH --account=marasovic

# Submit only after the matching response array succeeds:
#
#   sbatch --dependency=afterok:<array-job-id> \
#     --export=ALL,PLAN=/abs/plan.json,DOWNSTREAM_ENV=/scratch/.../<lane-env> \
#     scripts/bonafide/downstream_dense_compact.sbatch

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ ! -f "${REPO_ROOT}/scripts/chpc_env.sh" ]]; then
    echo "Submit from a circuits repository root; got ${REPO_ROOT}." >&2
    exit 2
fi
cd "$REPO_ROOT"
REPO_ROOT="$(pwd -P)"

: "${PLAN:?Set PLAN to an absolute downstream build-plan path}"
: "${DOWNSTREAM_ENV:?Set DOWNSTREAM_ENV to the frozen lane environment}"
if [[ "$PLAN" != /* ]] || [[ ! -f "$PLAN" ]]; then
    echo "PLAN must be an existing absolute path; got ${PLAN}." >&2
    exit 2
fi
if [[ "$DOWNSTREAM_ENV" != /* ]] || [[ ! -x "${DOWNSTREAM_ENV}/bin/python" ]]; then
    echo "DOWNSTREAM_ENV must be an existing absolute Python environment." >&2
    exit 2
fi
SHARED_MUTABLE_ENV="/scratch/general/vast/${USER}/circuits/envs/circuits-py312"
if [[ "$DOWNSTREAM_ENV" == "$SHARED_MUTABLE_ENV" ]]; then
    echo "Production lane cannot use the shared mutable project environment." >&2
    exit 2
fi

export UV_PROJECT_ENVIRONMENT="$DOWNSTREAM_ENV"
module load python/3.12.12 uv/0.11.14
# shellcheck disable=SC1091
source scripts/chpc_env.sh
PYTHON_BIN="${UV_PROJECT_ENVIRONMENT}/bin/python"

readarray -t plan_fields < <(
    "$PYTHON_BIN" -c \
        'import json,sys; p=json.load(open(sys.argv[1])); print(p["plan_sha256"]); print(p["lane"]); print(str(p["development"]).lower()); print(len(p["tasks"])); print(p["repo_root"]); print(p["output_root"])' \
        "$PLAN"
)
PLAN_SHA256="${plan_fields[0]}"
LANE="${plan_fields[1]}"
DEVELOPMENT="${plan_fields[2]}"
TASK_COUNT="${plan_fields[3]}"
PLAN_REPO_ROOT="${plan_fields[4]}"
OUTPUT_ROOT="${plan_fields[5]}"

if [[ "$DEVELOPMENT" != "false" ]]; then
    echo "Production compactor refuses a development build plan." >&2
    exit 2
fi
if [[ "$TASK_COUNT" != "11" ]] || [[ "$PLAN_REPO_ROOT" != "$REPO_ROOT" ]]; then
    echo "Plan is not the frozen 11-response plan for this worktree." >&2
    exit 2
fi

"$PYTHON_BIN" -m scripts.bonafide.downstream_build_plan validate \
    --plan "$PLAN" \
    --verify-code

echo "job_id=${SLURM_JOB_ID} node=${SLURMD_NODENAME}"
echo "lane=${LANE} plan_sha256=${PLAN_SHA256} output_root=${OUTPUT_ROOT}"

"$PYTHON_BIN" -m scripts.bonafide.downstream_compact --plan "$PLAN"
