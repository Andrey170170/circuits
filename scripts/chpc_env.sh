#!/bin/bash

# Shared CHPC defaults for local development and project-owned jobs.
# Source this file from the repository root before running uv or Hugging Face tools.

CIRCUITS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -f "${CIRCUITS_ROOT}/.env" ]]; then
    case $- in
        *a*) CIRCUITS_ALLEXPORT_WAS_SET=1 ;;
        *) CIRCUITS_ALLEXPORT_WAS_SET=0 ;;
    esac
    set -a
    # shellcheck disable=SC1091
    source "${CIRCUITS_ROOT}/.env"
    if [[ "$CIRCUITS_ALLEXPORT_WAS_SET" -eq 0 ]]; then
        set +a
    fi
    unset CIRCUITS_ALLEXPORT_WAS_SET
fi

CIRCUITS_CACHE_ROOT="${CIRCUITS_CACHE_ROOT:-/scratch/general/vast/${USER}/circuits}"
SHARED_HF_CACHE_ROOT="${SHARED_HF_CACHE_ROOT:-/scratch/general/vast/${USER}/nlp_research_project/huggingface}"
SHARED_UV_CACHE_ROOT="${SHARED_UV_CACHE_ROOT:-/scratch/general/vast/${USER}/nlp_research_project/uv-cache}"

if [[ -n "${SLURM_JOB_ID:-}" && -d "/scratch/local/${USER}/${SLURM_JOB_ID}" ]]; then
    CIRCUITS_DEFAULT_TMPDIR="/scratch/local/${USER}/${SLURM_JOB_ID}/circuits-tmp"
else
    CIRCUITS_DEFAULT_TMPDIR="${CIRCUITS_CACHE_ROOT}/tmp"
fi

export UV_CACHE_DIR="${UV_CACHE_DIR:-${SHARED_UV_CACHE_ROOT}}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-${CIRCUITS_CACHE_ROOT}/envs/circuits-py312}"
export HF_HOME="${HF_HOME:-${SHARED_HF_CACHE_ROOT}}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}}"
export CIRCUITS_RESULTS_DIR="${CIRCUITS_RESULTS_DIR:-${CIRCUITS_CACHE_ROOT}/results}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${CIRCUITS_CACHE_ROOT}/matplotlib}"
export TMPDIR="${TMPDIR:-${CIRCUITS_DEFAULT_TMPDIR}}"

mkdir -p \
    "$UV_CACHE_DIR" \
    "$UV_PROJECT_ENVIRONMENT" \
    "$HF_HOME" \
    "$CIRCUITS_RESULTS_DIR" \
    "$MPLCONFIGDIR" \
    "$TMPDIR"

export CIRCUITS_ROOT
unset CIRCUITS_DEFAULT_TMPDIR
