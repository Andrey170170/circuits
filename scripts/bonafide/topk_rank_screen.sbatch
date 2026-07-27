#!/bin/bash
#SBATCH --job-name=bonafide-topk-rank
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --clusters=notchpeak
#SBATCH --partition=marasovic-gpu-np
#SBATCH --qos=marasovic-gpu-np
#SBATCH --account=marasovic-gpu-np
#SBATCH --gres=gpu:a100:1

# Discovery-only selection evidence. This job scores observed ranks but does
# not construct or save scientific trace artifacts.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"

: "${SOURCE_MANIFEST:?Set SOURCE_MANIFEST to an absolute width-one manifest}"
: "${OUTPUT:?Set OUTPUT to an absolute new rank-screen JSON path}"
: "${UV_PROJECT_ENVIRONMENT:?Set a lane-specific absolute uv environment}"
if [[ "$SOURCE_MANIFEST" != /* ]] || [[ ! -f "$SOURCE_MANIFEST" ]]; then
    echo "SOURCE_MANIFEST must be an existing absolute path." >&2
    exit 2
fi
if [[ "$OUTPUT" != /* ]] || [[ -e "$OUTPUT" ]]; then
    echo "OUTPUT must be an absolute path that does not exist." >&2
    exit 2
fi
if [[ "$UV_PROJECT_ENVIRONMENT" != /* ]]; then
    echo "UV_PROJECT_ENVIRONMENT must be absolute." >&2
    exit 2
fi
RANK_UV_PROJECT_ENVIRONMENT="$UV_PROJECT_ENVIRONMENT"

module load python/3.12.12 uv/0.11.14
# shellcheck disable=SC1091
source scripts/chpc_env.sh
export UV_PROJECT_ENVIRONMENT="$RANK_UV_PROJECT_ENVIRONMENT"

PYTHON_BIN="${UV_PROJECT_ENVIRONMENT}/bin/python"
CONFIG="${CONFIG:-${REPO_ROOT}/scripts/bonafide/configs/qwen3_4b_instruct.json}"
MAX_ITEMS="${MAX_ITEMS:-32}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "job_id=${SLURM_JOB_ID} node=${SLURMD_NODENAME} max_items=${MAX_ITEMS}"
echo "config=${CONFIG} source_manifest=${SOURCE_MANIFEST} output=${OUTPUT}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

"$PYTHON_BIN" -m scripts.bonafide.topk_rank_screen \
    --config "$CONFIG" \
    --source-manifest "$SOURCE_MANIFEST" \
    --max-items "$MAX_ITEMS" \
    --output "$OUTPUT"
