#!/bin/bash
#SBATCH --job-name=bonafide-dense-joint
#SBATCH --output=logs/slurm-%x-%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --clusters=granite
#SBATCH --partition=coe-class-grn
#SBATCH --qos=coe-students-grn
#SBATCH --account=marasovic
#SBATCH --array=0-10%4

# Submit from the clean immutable lane worktree:
#
#   sbatch --export=ALL,FEATURE_PLAN=/abs/features-plan.json,\
#MULTIPLEX_PLAN=/abs/multiplex-plan.json,DOWNSTREAM_ENV=/scratch/.../<env> \
#     scripts/bonafide/downstream_dense_joint_array.sbatch

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ ! -f "${REPO_ROOT}/scripts/chpc_env.sh" ]]; then
    echo "Submit from a circuits repository root; got ${REPO_ROOT}." >&2
    exit 2
fi
cd "$REPO_ROOT"
REPO_ROOT="$(pwd -P)"

: "${FEATURE_PLAN:?Set FEATURE_PLAN to an absolute feature build-plan path}"
: "${MULTIPLEX_PLAN:?Set MULTIPLEX_PLAN to an absolute multiplex build-plan path}"
: "${DOWNSTREAM_ENV:?Set DOWNSTREAM_ENV to the frozen joint-lane environment}"
: "${SLURM_ARRAY_TASK_ID:?This launcher must run as an array task}"
for plan_path in "$FEATURE_PLAN" "$MULTIPLEX_PLAN"; do
    if [[ "$plan_path" != /* ]] || [[ ! -f "$plan_path" ]]; then
        echo "Both plans must be existing absolute paths; got ${plan_path}." >&2
        exit 2
    fi
done
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

readarray -t feature_fields < <(
    "$PYTHON_BIN" -c \
        'import json,sys; p=json.load(open(sys.argv[1])); print(p["plan_sha256"]); print(p["lane"]); print(str(p["development"]).lower()); print(len(p["tasks"])); print(p["repo_root"]); print(p["output_root"])' \
        "$FEATURE_PLAN"
)
readarray -t multiplex_fields < <(
    "$PYTHON_BIN" -c \
        'import json,sys; p=json.load(open(sys.argv[1])); print(p["plan_sha256"]); print(p["lane"]); print(str(p["development"]).lower()); print(len(p["tasks"])); print(p["repo_root"]); print(p["output_root"])' \
        "$MULTIPLEX_PLAN"
)

if [[ "${feature_fields[1]}" != "dense_features" ]] || \
   [[ "${multiplex_fields[1]}" != "dense_multiplex" ]]; then
    echo "Joint launcher requires dense_features and dense_multiplex plans." >&2
    exit 2
fi
if [[ "${feature_fields[2]}" != "false" ]] || \
   [[ "${multiplex_fields[2]}" != "false" ]]; then
    echo "Production joint array refuses development plans." >&2
    exit 2
fi
if [[ "${feature_fields[3]}" != "11" ]] || \
   [[ "${multiplex_fields[3]}" != "11" ]]; then
    echo "Frozen dense joint array requires exactly 11 response tasks." >&2
    exit 2
fi
if [[ "${feature_fields[4]}" != "$REPO_ROOT" ]] || \
   [[ "${multiplex_fields[4]}" != "$REPO_ROOT" ]]; then
    echo "Plan repo_root does not match the submitted immutable worktree." >&2
    exit 2
fi
if [[ "${SLURM_ARRAY_TASK_MIN}" != "0" ]] || \
   [[ "${SLURM_ARRAY_TASK_MAX}" != "10" ]] || \
   [[ "${SLURM_ARRAY_TASK_COUNT}" != "11" ]] || \
   [[ "${SLURM_ARRAY_TASK_STEP}" != "1" ]]; then
    echo "Frozen dense joint array shape must be 0-10 with step 1." >&2
    exit 2
fi

"$PYTHON_BIN" -m scripts.bonafide.downstream_build_plan validate \
    --plan "$FEATURE_PLAN" \
    --verify-code
"$PYTHON_BIN" -m scripts.bonafide.downstream_build_plan validate \
    --plan "$MULTIPLEX_PLAN" \
    --verify-code

echo "job_id=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} node=${SLURMD_NODENAME}"
echo "feature_plan_sha256=${feature_fields[0]} output=${feature_fields[5]}"
echo "multiplex_plan_sha256=${multiplex_fields[0]} output=${multiplex_fields[5]}"

"$PYTHON_BIN" -m scripts.bonafide.downstream_build_joint \
    --feature-plan "$FEATURE_PLAN" \
    --multiplex-plan "$MULTIPLEX_PLAN" \
    --task-index "$SLURM_ARRAY_TASK_ID"
