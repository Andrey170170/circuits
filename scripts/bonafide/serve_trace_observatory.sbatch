#!/bin/bash
#SBATCH --job-name=trace-observatory
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=04:00:00
#SBATCH --clusters=granite
#SBATCH --partition=coe-class-grn
#SBATCH --qos=coe-students-grn
#SBATCH --account=marasovic

# Run the CPU-only Trace Observatory as a scheduler-owned service. Optional
# exports: SITE_ROOT, STATE_ROOT, HOST, PORT. Submit from the repository root.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ ! -f "${REPO_ROOT}/scripts/chpc_env.sh" ]]; then
    echo "Submit this job from the circuits repository root; got ${REPO_ROOT}." >&2
    exit 2
fi
cd "$REPO_ROOT"

module load python/3.12.12 uv/0.11.14
# shellcheck disable=SC1091
source scripts/chpc_env.sh

SITE_ROOT="${SITE_ROOT:-/scratch/general/vast/${USER}/circuits/results/bonafide/raw-graph-observatory-viewer-v1}"
STATE_ROOT="${STATE_ROOT:-/uufs/chpc.utah.edu/common/home/${USER}/projects/circuits-observatory-state}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8032}"
PYTHON_BIN="${UV_PROJECT_ENVIRONMENT}/bin/python"

for required_path in "$SITE_ROOT" "$STATE_ROOT" "$PYTHON_BIN"; do
    if [[ "$required_path" != /* ]] || [[ ! -e "$required_path" ]]; then
        echo "Required path must be absolute and exist: $required_path" >&2
        exit 2
    fi
done
if [[ "$HOST" != "127.0.0.1" ]]; then
    echo "HOST must remain 127.0.0.1; got $HOST." >&2
    exit 2
fi
if [[ ! "$PORT" =~ ^[0-9]+$ ]] || ((PORT < 1024 || PORT > 65535)); then
    echo "PORT must be an integer from 1024 through 65535; got $PORT." >&2
    exit 2
fi

echo "job_id=${SLURM_JOB_ID} node=${SLURMD_NODENAME} host=${HOST} port=${PORT}"
echo "site_root=${SITE_ROOT} state_root=${STATE_ROOT}"

exec "$PYTHON_BIN" -m circuits.observatory serve \
    --site-root "$SITE_ROOT" \
    --state-root "$STATE_ROOT" \
    --host "$HOST" \
    --port "$PORT"
