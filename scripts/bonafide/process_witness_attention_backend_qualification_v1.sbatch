#!/bin/bash
#SBATCH --job-name=pw-attn-backend-qv1
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --clusters=notchpeak
#SBATCH --partition=marasovic-gpu-np
#SBATCH --qos=marasovic-gpu-np
#SBATCH --account=marasovic-gpu-np
#SBATCH --gres=gpu:a100:1
#SBATCH --signal=B:USR1@300
#SBATCH --no-requeue

# Execute exactly one existing qualification-v4 strict-T5 source item with
# explicit tracing strategies. Each strategy/item attempt is a fresh,
# provenance-bound artifact; this launcher never resumes or overwrites.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ ! -f "${REPO_ROOT}/scripts/chpc_env.sh" ]]; then
    echo "Submit from the frozen circuits worktree; got ${REPO_ROOT}." >&2
    exit 2
fi
cd "$REPO_ROOT"

: "${CONFIG:?Set CONFIG to an absolute tracing qualification run config}"
: "${CONFIG_FILE_SHA256:?Set CONFIG_FILE_SHA256 to the frozen config file SHA-256}"
: "${MANIFEST:?Set MANIFEST to the absolute qualification-v4 trace manifest}"
: "${MANIFEST_FILE_SHA256:?Set MANIFEST_FILE_SHA256 to its frozen file SHA-256}"
: "${WIDTH1_SOURCE_MANIFEST_FILE_SHA256:?Set the frozen width-one source manifest file SHA-256}"
: "${ARTIFACT_ROOT:?Set ARTIFACT_ROOT to an absolute fresh qualification output root}"
: "${EXPECTED_GIT_COMMIT:?Set EXPECTED_GIT_COMMIT to the exact launch commit}"
: "${EXPECTED_ATTENTION_BACKEND:?Set EXPECTED_ATTENTION_BACKEND to the explicit public backend name}"
: "${EXPECTED_STOP_GRADIENT_CONTRIBUTION_EXECUTION:?Set EXPECTED_STOP_GRADIENT_CONTRIBUTION_EXECUTION to the explicit contribution strategy}"
: "${EXPECTED_STOP_GRADIENT_CONTRIBUTION_TARGET_LANE_CHUNK_SIZE:?Set EXPECTED_STOP_GRADIENT_CONTRIBUTION_TARGET_LANE_CHUNK_SIZE to none or a positive integer}"
: "${WAVE:?Set WAVE to exactly one qualification-v4 wave ID}"
: "${ONLY_ARTIFACT_ID:?Set ONLY_ARTIFACT_ID to exactly one item in WAVE}"
: "${UV_PROJECT_ENVIRONMENT:?Set a lane-specific absolute uv environment}"
EXPECTED_CUDA_ALLOCATOR_POLICY="${EXPECTED_CUDA_ALLOCATOR_POLICY:-}"
EXPECTED_CUDA_HEADROOM_POLICY="${EXPECTED_CUDA_HEADROOM_POLICY:-peak_reserved_v1}"
EXPECTED_CUDA_HEADROOM_ACTION="${EXPECTED_CUDA_HEADROOM_ACTION:-stop}"
EXPECTED_EMBEDDING_EDGE_MATERIALIZATION="${EXPECTED_EMBEDDING_EDGE_MATERIALIZATION:-}"
EXPECTED_CROSS_LAYER_JACOBIAN_EXECUTION="${EXPECTED_CROSS_LAYER_JACOBIAN_EXECUTION:-}"
EXPECTED_SELECTED_NEURON_CONTRIBUTION_TARGET_LANE_CHUNK_SIZE="${EXPECTED_SELECTED_NEURON_CONTRIBUTION_TARGET_LANE_CHUNK_SIZE:-legacy_unbound}"
EXPECTED_SELECTED_EMBED_CONTRIBUTION_TARGET_LANE_CHUNK_SIZE="${EXPECTED_SELECTED_EMBED_CONTRIBUTION_TARGET_LANE_CHUNK_SIZE:-legacy_unbound}"
EXPECTED_STOP_GRADIENT_EMBED_CONTRIBUTION_TARGET_LANE_CHUNK_SIZE="${EXPECTED_STOP_GRADIENT_EMBED_CONTRIBUTION_TARGET_LANE_CHUNK_SIZE:-legacy_unbound}"
EXPECTED_SELECTED_ATTRIBUTION_NEURON_LANE_CHUNK_SIZE="${EXPECTED_SELECTED_ATTRIBUTION_NEURON_LANE_CHUNK_SIZE:-legacy_unbound}"
EXPECTED_STOP_GRADIENT_SELECTED_ATTRIBUTION_FORWARD_EXECUTION="${EXPECTED_STOP_GRADIENT_SELECTED_ATTRIBUTION_FORWARD_EXECUTION:-legacy_unbound}"
EXPECTED_STOP_GRADIENT_SELECTED_ATTRIBUTION_STORAGE="${EXPECTED_STOP_GRADIENT_SELECTED_ATTRIBUTION_STORAGE:-legacy_unbound}"
EXPECTED_SELECTED_TARGET_LOGIT_EXECUTION="${EXPECTED_SELECTED_TARGET_LOGIT_EXECUTION:-legacy_unbound}"
EXPECTED_POST_SELECTION_STATE_STORAGE="${EXPECTED_POST_SELECTION_STATE_STORAGE:-legacy_unbound}"

for path_name in CONFIG MANIFEST ARTIFACT_ROOT UV_PROJECT_ENVIRONMENT; do
    path_value="${!path_name}"
    if [[ "$path_value" != /* ]]; then
        echo "${path_name} must be absolute; got ${path_value}." >&2
        exit 2
    fi
done
if [[ ! -f "$CONFIG" ]] || [[ ! -f "$MANIFEST" ]]; then
    echo "CONFIG and MANIFEST must be existing regular files." >&2
    exit 2
fi
if [[ ! "$WAVE" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] ||
   [[ ! "$ONLY_ARTIFACT_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "WAVE and ONLY_ARTIFACT_ID must be filesystem-safe identifiers." >&2
    exit 2
fi
case "$EXPECTED_ATTENTION_BACKEND" in
    legacy_eager_unmasked_v1|eager_causal_v1|flash_sdpa_causal_v1) ;;
    *)
        echo "Unsupported EXPECTED_ATTENTION_BACKEND=${EXPECTED_ATTENTION_BACKEND}." >&2
        exit 2
        ;;
esac
case "$EXPECTED_STOP_GRADIENT_CONTRIBUTION_EXECUTION" in
    full_graph_v1|source_leaf_v1|sparse_source_leaf_v1) ;;
    *)
        echo "Unsupported EXPECTED_STOP_GRADIENT_CONTRIBUTION_EXECUTION=${EXPECTED_STOP_GRADIENT_CONTRIBUTION_EXECUTION}." >&2
        exit 2
        ;;
esac
if [[ "$EXPECTED_STOP_GRADIENT_CONTRIBUTION_TARGET_LANE_CHUNK_SIZE" != "none" ]] &&
   [[ ! "$EXPECTED_STOP_GRADIENT_CONTRIBUTION_TARGET_LANE_CHUNK_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo "EXPECTED_STOP_GRADIENT_CONTRIBUTION_TARGET_LANE_CHUNK_SIZE must be none or a positive integer." >&2
    exit 2
fi
if [[ "$EXPECTED_SELECTED_NEURON_CONTRIBUTION_TARGET_LANE_CHUNK_SIZE" != "legacy_unbound" ]] &&
   [[ "$EXPECTED_SELECTED_NEURON_CONTRIBUTION_TARGET_LANE_CHUNK_SIZE" != "none" ]] &&
   [[ ! "$EXPECTED_SELECTED_NEURON_CONTRIBUTION_TARGET_LANE_CHUNK_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo "EXPECTED_SELECTED_NEURON_CONTRIBUTION_TARGET_LANE_CHUNK_SIZE must be legacy_unbound, none, or a positive integer." >&2
    exit 2
fi
if [[ "$EXPECTED_SELECTED_EMBED_CONTRIBUTION_TARGET_LANE_CHUNK_SIZE" != "legacy_unbound" ]] &&
   [[ "$EXPECTED_SELECTED_EMBED_CONTRIBUTION_TARGET_LANE_CHUNK_SIZE" != "none" ]] &&
   [[ ! "$EXPECTED_SELECTED_EMBED_CONTRIBUTION_TARGET_LANE_CHUNK_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo "EXPECTED_SELECTED_EMBED_CONTRIBUTION_TARGET_LANE_CHUNK_SIZE must be legacy_unbound, none, or a positive integer." >&2
    exit 2
fi
if [[ "$EXPECTED_STOP_GRADIENT_EMBED_CONTRIBUTION_TARGET_LANE_CHUNK_SIZE" != "legacy_unbound" ]] &&
   [[ "$EXPECTED_STOP_GRADIENT_EMBED_CONTRIBUTION_TARGET_LANE_CHUNK_SIZE" != "none" ]] &&
   [[ ! "$EXPECTED_STOP_GRADIENT_EMBED_CONTRIBUTION_TARGET_LANE_CHUNK_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo "EXPECTED_STOP_GRADIENT_EMBED_CONTRIBUTION_TARGET_LANE_CHUNK_SIZE must be legacy_unbound, none, or a positive integer." >&2
    exit 2
fi
if [[ "$EXPECTED_SELECTED_ATTRIBUTION_NEURON_LANE_CHUNK_SIZE" != "legacy_unbound" ]] &&
   [[ "$EXPECTED_SELECTED_ATTRIBUTION_NEURON_LANE_CHUNK_SIZE" != "none" ]] &&
   [[ ! "$EXPECTED_SELECTED_ATTRIBUTION_NEURON_LANE_CHUNK_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo "EXPECTED_SELECTED_ATTRIBUTION_NEURON_LANE_CHUNK_SIZE must be legacy_unbound, none, or a positive integer." >&2
    exit 2
fi
case "$EXPECTED_STOP_GRADIENT_SELECTED_ATTRIBUTION_FORWARD_EXECUTION" in
    legacy_unbound|full_model_v1|prefix_stop_v1) ;;
    *)
        echo "Unsupported EXPECTED_STOP_GRADIENT_SELECTED_ATTRIBUTION_FORWARD_EXECUTION=${EXPECTED_STOP_GRADIENT_SELECTED_ATTRIBUTION_FORWARD_EXECUTION}." >&2
        exit 2
        ;;
esac
case "$EXPECTED_STOP_GRADIENT_SELECTED_ATTRIBUTION_STORAGE" in
    legacy_unbound|graph_retaining_v1|terminal_detached_v1) ;;
    *)
        echo "Unsupported EXPECTED_STOP_GRADIENT_SELECTED_ATTRIBUTION_STORAGE=${EXPECTED_STOP_GRADIENT_SELECTED_ATTRIBUTION_STORAGE}." >&2
        exit 2
        ;;
esac
case "$EXPECTED_SELECTED_TARGET_LOGIT_EXECUTION" in
    legacy_unbound|full_logits_v1|selected_position_logits_v1) ;;
    *)
        echo "Unsupported EXPECTED_SELECTED_TARGET_LOGIT_EXECUTION=${EXPECTED_SELECTED_TARGET_LOGIT_EXECUTION}." >&2
        exit 2
        ;;
esac
case "$EXPECTED_POST_SELECTION_STATE_STORAGE" in
    legacy_unbound|dense_v1|compact_cpu_v1) ;;
    *)
        echo "Unsupported EXPECTED_POST_SELECTION_STATE_STORAGE=${EXPECTED_POST_SELECTION_STATE_STORAGE}." >&2
        exit 2
        ;;
esac
case "$EXPECTED_EMBEDDING_EDGE_MATERIALIZATION" in
    ""|scalar_v1|vectorized_v1) ;;
    *)
        echo "Unsupported EXPECTED_EMBEDDING_EDGE_MATERIALIZATION=${EXPECTED_EMBEDDING_EDGE_MATERIALIZATION}." >&2
        exit 2
        ;;
esac
case "$EXPECTED_CROSS_LAYER_JACOBIAN_EXECUTION" in
    ""|full_model_v1|cached_range_v1) ;;
    *)
        echo "Unsupported EXPECTED_CROSS_LAYER_JACOBIAN_EXECUTION=${EXPECTED_CROSS_LAYER_JACOBIAN_EXECUTION}." >&2
        exit 2
        ;;
esac
case "$EXPECTED_CUDA_HEADROOM_POLICY" in
    peak_reserved_v1|allocator_dense_joint_v1) ;;
    *)
        echo "Unsupported EXPECTED_CUDA_HEADROOM_POLICY=${EXPECTED_CUDA_HEADROOM_POLICY}." >&2
        exit 2
        ;;
esac
case "$EXPECTED_CUDA_HEADROOM_ACTION" in
    warn|stop) ;;
    *)
        echo "Unsupported EXPECTED_CUDA_HEADROOM_ACTION=${EXPECTED_CUDA_HEADROOM_ACTION}." >&2
        exit 2
        ;;
esac

apply_cuda_allocator_policy() {
    case "$EXPECTED_CUDA_ALLOCATOR_POLICY" in
        "") ;;
        default_v1) unset PYTORCH_CUDA_ALLOC_CONF ;;
        expandable_segments_v1)
            export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
            ;;
        *)
            echo "Unsupported EXPECTED_CUDA_ALLOCATOR_POLICY=${EXPECTED_CUDA_ALLOCATOR_POLICY}." >&2
            exit 2
            ;;
    esac
}
# Apply before this launcher starts Python. Reapply after environment setup so
# the Python runtime cannot inherit a value that disagrees with the policy ID.
apply_cuda_allocator_policy

verify_file_sha256() {
    local path="$1"
    local expected="$2"
    local label="$3"
    if [[ ! "$expected" =~ ^[0-9a-f]{64}$ ]] ||
       [[ "$(sha256sum "$path" | awk '{print $1}')" != "$expected" ]]; then
        echo "${label} file SHA-256 drift." >&2
        exit 2
    fi
}
verify_file_sha256 "$CONFIG" "$CONFIG_FILE_SHA256" "Run config"
verify_file_sha256 "$MANIFEST" "$MANIFEST_FILE_SHA256" "Trace manifest"

if [[ ! "$EXPECTED_GIT_COMMIT" =~ ^[0-9a-f]{40}$ ]] ||
   [[ "$(git rev-parse HEAD)" != "$EXPECTED_GIT_COMMIT" ]] ||
   [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
    echo "Tracing qualification requires the exact clean Git commit." >&2
    exit 2
fi

QUALIFICATION_UV_PROJECT_ENVIRONMENT="$UV_PROJECT_ENVIRONMENT"
module load python/3.12.12 uv/0.11.14
# shellcheck disable=SC1091
source scripts/chpc_env.sh
export UV_PROJECT_ENVIRONMENT="$QUALIFICATION_UV_PROJECT_ENVIRONMENT"
apply_cuda_allocator_policy
PYTHON_BIN="${UV_PROJECT_ENVIRONMENT}/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Frozen lane Python environment is missing at ${PYTHON_BIN}." >&2
    exit 2
fi

EXPECTED_GPU_NAME="${EXPECTED_GPU_NAME:-NVIDIA A100 80GB PCIe}"
mapfile -t gpu_names < <(
    nvidia-smi --query-gpu=name --format=csv,noheader,nounits
)
if [[ "${#gpu_names[@]}" -ne 1 ]] || [[ "${gpu_names[0]}" != "$EXPECTED_GPU_NAME" ]]; then
    printf 'Expected exactly one GPU named %s; observed: %s\n' \
        "$EXPECTED_GPU_NAME" "${gpu_names[*]:-none}" >&2
    exit 2
fi

ATTEMPT_ID="${SLURM_JOB_ID}-r${SLURM_RESTART_COUNT:-0}"
SUMMARY_JSONL="${SUMMARY_JSONL:-${ARTIFACT_ROOT}/execution-summaries/${EXPECTED_ATTENTION_BACKEND}/${WAVE}/${ONLY_ARTIFACT_ID}/${ATTEMPT_ID}.jsonl}"
if [[ "$SUMMARY_JSONL" != /* ]]; then
    echo "SUMMARY_JSONL must be absolute." >&2
    exit 2
fi

# This preflight validates the frozen inputs and computes the exact output path
# with the same identity functions used by topk_runner. Runtime JSON goes to the
# scheduler log; it is also recorded by topk_runner in the artifact and summary.
EXPECTED_OUTPUT_DIR="$(
    "$PYTHON_BIN" -c \
        'import hashlib,json,os,pathlib,sys
from scripts.bonafide.runner import collect_code_revision, collect_runtime_environment
from scripts.bonafide.cuda_allocator_policy import bind_cuda_allocator_runtime_receipt, declared_cuda_allocator_policy, validate_cuda_allocator_environment
from scripts.bonafide.topk_runner import topk_runtime_artifact_identity

config_path=pathlib.Path(sys.argv[1])
manifest_path=pathlib.Path(sys.argv[2])
artifact_root=pathlib.Path(sys.argv[3])
wave_id=sys.argv[4]
source_id=sys.argv[5]
expected_backend=sys.argv[6]
expected_contribution_execution=sys.argv[7]
expected_chunk_size_raw=sys.argv[8]
expected_chunk_size=None if expected_chunk_size_raw == "none" else int(expected_chunk_size_raw)
expected_source_sha=sys.argv[9]
repo_root=pathlib.Path(sys.argv[10])
expected_allocator_policy=sys.argv[11] or None
expected_embedding_edge_materialization=sys.argv[12] or None
expected_cross_layer_jacobian_execution=sys.argv[13] or None
expected_selected_chunk_size_raw=sys.argv[14]
expected_selected_chunk_size=(
    None
    if expected_selected_chunk_size_raw in {"legacy_unbound", "none"}
    else int(expected_selected_chunk_size_raw)
)
expected_selected_embed_chunk_size_raw=sys.argv[15]
expected_selected_embed_chunk_size=(
    None
    if expected_selected_embed_chunk_size_raw in {"legacy_unbound", "none"}
    else int(expected_selected_embed_chunk_size_raw)
)
expected_stop_gradient_embed_chunk_size_raw=sys.argv[16]
expected_stop_gradient_embed_chunk_size=(
    None
    if expected_stop_gradient_embed_chunk_size_raw in {"legacy_unbound", "none"}
    else int(expected_stop_gradient_embed_chunk_size_raw)
)
expected_selected_attribution_chunk_size_raw=sys.argv[17]
expected_selected_attribution_chunk_size=(
    None
    if expected_selected_attribution_chunk_size_raw in {"legacy_unbound", "none"}
    else int(expected_selected_attribution_chunk_size_raw)
)
expected_selected_forward_execution=sys.argv[18]
expected_selected_storage=sys.argv[19]
expected_selected_target_logit_execution=sys.argv[20]
expected_post_selection_state_storage=sys.argv[21]
expected_cuda_headroom_policy=sys.argv[22]
expected_cuda_headroom_action=sys.argv[23]
config=json.loads(config_path.read_text(encoding="utf-8"))
manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
adag=config.get("adag_config")
if not isinstance(adag, dict) or "stop_gradient_attention_backend" not in adag:
    raise SystemExit("run config must explicitly declare adag_config.stop_gradient_attention_backend")
if adag["stop_gradient_attention_backend"] != expected_backend:
    raise SystemExit("run config backend disagrees with EXPECTED_ATTENTION_BACKEND")
if adag.get("stop_gradient_contribution_execution") != expected_contribution_execution:
    raise SystemExit("run config contribution execution disagrees with EXPECTED_STOP_GRADIENT_CONTRIBUTION_EXECUTION")
chunk_size_field="stop_gradient_contribution_target_lane_chunk_size"
if chunk_size_field not in adag or adag[chunk_size_field] != expected_chunk_size:
    raise SystemExit("run config target-lane chunk size disagrees with EXPECTED_STOP_GRADIENT_CONTRIBUTION_TARGET_LANE_CHUNK_SIZE")
selected_chunk_size_field="selected_neuron_contribution_target_lane_chunk_size"
selected_chunk_size_is_bound=expected_selected_chunk_size_raw != "legacy_unbound"
if selected_chunk_size_is_bound:
    if selected_chunk_size_field not in adag or adag[selected_chunk_size_field] != expected_selected_chunk_size:
        raise SystemExit("run config selected-neuron target-lane chunk size disagrees with EXPECTED_SELECTED_NEURON_CONTRIBUTION_TARGET_LANE_CHUNK_SIZE")
elif selected_chunk_size_field in adag:
    raise SystemExit("legacy-unbound qualification config unexpectedly declares selected-neuron target-lane chunk size")
selected_embed_chunk_size_field="selected_embed_contribution_target_lane_chunk_size"
selected_embed_chunk_size_is_bound=expected_selected_embed_chunk_size_raw != "legacy_unbound"
if selected_embed_chunk_size_is_bound:
    if selected_embed_chunk_size_field not in adag or adag[selected_embed_chunk_size_field] != expected_selected_embed_chunk_size:
        raise SystemExit("run config selected-embed target-lane chunk size disagrees with EXPECTED_SELECTED_EMBED_CONTRIBUTION_TARGET_LANE_CHUNK_SIZE")
elif selected_embed_chunk_size_field in adag:
    raise SystemExit("legacy-unbound qualification config unexpectedly declares selected-embed target-lane chunk size")
stop_gradient_embed_chunk_size_field="stop_gradient_embed_contribution_target_lane_chunk_size"
stop_gradient_embed_chunk_size_is_bound=expected_stop_gradient_embed_chunk_size_raw != "legacy_unbound"
if stop_gradient_embed_chunk_size_is_bound:
    if stop_gradient_embed_chunk_size_field not in adag or adag[stop_gradient_embed_chunk_size_field] != expected_stop_gradient_embed_chunk_size:
        raise SystemExit("run config stop-gradient-embed target-lane chunk size disagrees with EXPECTED_STOP_GRADIENT_EMBED_CONTRIBUTION_TARGET_LANE_CHUNK_SIZE")
elif stop_gradient_embed_chunk_size_field in adag:
    raise SystemExit("legacy-unbound qualification config unexpectedly declares stop-gradient-embed target-lane chunk size")
selected_attribution_chunk_size_field="selected_attribution_neuron_lane_chunk_size"
selected_attribution_chunk_size_is_bound=(
    expected_selected_attribution_chunk_size_raw != "legacy_unbound"
)
if selected_attribution_chunk_size_is_bound:
    if selected_attribution_chunk_size_field not in adag or adag[selected_attribution_chunk_size_field] != expected_selected_attribution_chunk_size:
        raise SystemExit("run config selected-attribution neuron-lane chunk size disagrees with EXPECTED_SELECTED_ATTRIBUTION_NEURON_LANE_CHUNK_SIZE")
elif selected_attribution_chunk_size_field in adag:
    raise SystemExit("legacy-unbound qualification config unexpectedly declares selected-attribution neuron-lane chunk size")
selected_forward_field="stop_gradient_selected_attribution_forward_execution"
selected_forward_is_bound=expected_selected_forward_execution != "legacy_unbound"
if selected_forward_is_bound:
    if selected_forward_field not in adag or adag[selected_forward_field] != expected_selected_forward_execution:
        raise SystemExit("run config stop-gradient selected-attribution forward execution disagrees with EXPECTED_STOP_GRADIENT_SELECTED_ATTRIBUTION_FORWARD_EXECUTION")
elif selected_forward_field in adag:
    raise SystemExit("legacy-unbound qualification config unexpectedly declares stop-gradient selected-attribution forward execution")
selected_storage_field="stop_gradient_selected_attribution_storage"
selected_storage_is_bound=expected_selected_storage != "legacy_unbound"
if selected_storage_is_bound:
    if selected_storage_field not in adag or adag[selected_storage_field] != expected_selected_storage:
        raise SystemExit("run config stop-gradient selected-attribution storage disagrees with EXPECTED_STOP_GRADIENT_SELECTED_ATTRIBUTION_STORAGE")
elif selected_storage_field in adag:
    raise SystemExit("legacy-unbound qualification config unexpectedly declares stop-gradient selected-attribution storage")
selected_target_logit_field="selected_target_logit_execution"
selected_target_logit_is_bound=(
    expected_selected_target_logit_execution != "legacy_unbound"
)
if selected_target_logit_is_bound:
    if selected_target_logit_field not in adag or adag[selected_target_logit_field] != expected_selected_target_logit_execution:
        raise SystemExit("run config selected target-logit execution disagrees with EXPECTED_SELECTED_TARGET_LOGIT_EXECUTION")
elif selected_target_logit_field in adag:
    raise SystemExit("legacy-unbound qualification config unexpectedly declares selected target-logit execution")
post_selection_state_storage_field="post_selection_state_storage"
post_selection_state_storage_is_bound=(
    expected_post_selection_state_storage != "legacy_unbound"
)
if post_selection_state_storage_is_bound:
    if post_selection_state_storage_field not in adag or adag[post_selection_state_storage_field] != expected_post_selection_state_storage:
        raise SystemExit("run config post-selection state storage disagrees with EXPECTED_POST_SELECTION_STATE_STORAGE")
elif post_selection_state_storage_field in adag:
    raise SystemExit("legacy-unbound qualification config unexpectedly declares post-selection state storage")
declared_embedding_edge_materialization=adag.get("embedding_edge_materialization")
if declared_embedding_edge_materialization != expected_embedding_edge_materialization:
    if expected_embedding_edge_materialization is not None or "embedding_edge_materialization" in adag:
        raise SystemExit("run config embedding-edge materialization disagrees with EXPECTED_EMBEDDING_EDGE_MATERIALIZATION")
declared_cross_layer_jacobian_execution=adag.get("cross_layer_jacobian_execution")
if declared_cross_layer_jacobian_execution != expected_cross_layer_jacobian_execution:
    if expected_cross_layer_jacobian_execution is not None or "cross_layer_jacobian_execution" in adag:
        raise SystemExit("run config cross-layer Jacobian execution disagrees with EXPECTED_CROSS_LAYER_JACOBIAN_EXECUTION")
declared_allocator_policy=declared_cuda_allocator_policy(config)
if declared_allocator_policy != expected_allocator_policy:
    raise SystemExit("run config allocator policy disagrees with EXPECTED_CUDA_ALLOCATOR_POLICY")
if config.get("trace_warmup", {}).get("enabled") is not False:
    raise SystemExit("tracing qualification requires trace_warmup.enabled=false")
limits=config.get("wave_limits")
expected_limits={"max_trace_seconds": 1800, "min_cuda_headroom_bytes": 8589934592, "stop_on_oom": True}
if isinstance(limits, dict) and "cuda_headroom_policy" in limits:
    expected_limits["cuda_headroom_policy"]=expected_cuda_headroom_policy
if isinstance(limits, dict) and "cuda_headroom_action" in limits:
    expected_limits["cuda_headroom_action"]=expected_cuda_headroom_action
declared_cuda_headroom_policy=(
    limits.get("cuda_headroom_policy", "peak_reserved_v1")
    if isinstance(limits, dict)
    else None
)
if declared_cuda_headroom_policy != expected_cuda_headroom_policy:
    raise SystemExit("run config CUDA headroom policy disagrees with EXPECTED_CUDA_HEADROOM_POLICY")
declared_cuda_headroom_action=(
    limits.get("cuda_headroom_action", "stop") if isinstance(limits, dict) else None
)
if declared_cuda_headroom_action != expected_cuda_headroom_action:
    raise SystemExit("run config CUDA headroom action disagrees with EXPECTED_CUDA_HEADROOM_ACTION")
if limits != expected_limits:
    raise SystemExit("tracing qualification resource gates drifted")
if manifest.get("phase") != "process_witness_resource_calibration_v1":
    raise SystemExit("launcher only accepts the qualification-v4 resource-calibration phase")
waves=[wave for wave in manifest.get("waves", []) if wave.get("wave_id")==wave_id]
if len(waves) != 1:
    raise SystemExit("WAVE is not present exactly once in MANIFEST")
items=[item for item in waves[0].get("items", []) if item.get("artifact_id")==source_id]
if len(items) != 1:
    raise SystemExit("ONLY_ARTIFACT_ID is not present exactly once in WAVE")
source=manifest.get("source", {})
source_path=pathlib.Path(os.path.expandvars(str(source.get("width1_manifest_path", ""))))
if not source_path.is_absolute() or not source_path.is_file():
    raise SystemExit("manifest width-one source must be an existing absolute file")
actual_source_sha=hashlib.sha256(source_path.read_bytes()).hexdigest()
if not (len(expected_source_sha)==64 and actual_source_sha==expected_source_sha==source.get("width1_manifest_sha256")):
    raise SystemExit("width-one source manifest file SHA-256 drift")
canonical=lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
validate_cuda_allocator_environment(config)
runtime=bind_cuda_allocator_runtime_receipt(config, collect_runtime_environment())
artifact_id,_identity=topk_runtime_artifact_identity(
    items[0], config=config, trace_family=manifest["trace_family"],
    code_revision=collect_code_revision(repo_root), runtime_environment=runtime,
    source_manifest_sha256=actual_source_sha,
    topk_manifest_sha256=hashlib.sha256(canonical(manifest)).hexdigest(),
    wave_id=wave_id,
)
print(json.dumps({"qualification_runtime_environment": runtime}, sort_keys=True), file=sys.stderr)
print(artifact_root/manifest["trace_family"]["trace_family_id"]/wave_id/artifact_id)' \
        "$CONFIG" "$MANIFEST" "$ARTIFACT_ROOT" "$WAVE" \
        "$ONLY_ARTIFACT_ID" "$EXPECTED_ATTENTION_BACKEND" \
        "$EXPECTED_STOP_GRADIENT_CONTRIBUTION_EXECUTION" \
        "$EXPECTED_STOP_GRADIENT_CONTRIBUTION_TARGET_LANE_CHUNK_SIZE" \
        "$WIDTH1_SOURCE_MANIFEST_FILE_SHA256" "$REPO_ROOT" \
        "$EXPECTED_CUDA_ALLOCATOR_POLICY" \
        "$EXPECTED_EMBEDDING_EDGE_MATERIALIZATION" \
        "$EXPECTED_CROSS_LAYER_JACOBIAN_EXECUTION" \
        "$EXPECTED_SELECTED_NEURON_CONTRIBUTION_TARGET_LANE_CHUNK_SIZE" \
        "$EXPECTED_SELECTED_EMBED_CONTRIBUTION_TARGET_LANE_CHUNK_SIZE" \
        "$EXPECTED_STOP_GRADIENT_EMBED_CONTRIBUTION_TARGET_LANE_CHUNK_SIZE" \
        "$EXPECTED_SELECTED_ATTRIBUTION_NEURON_LANE_CHUNK_SIZE" \
        "$EXPECTED_STOP_GRADIENT_SELECTED_ATTRIBUTION_FORWARD_EXECUTION" \
        "$EXPECTED_STOP_GRADIENT_SELECTED_ATTRIBUTION_STORAGE" \
        "$EXPECTED_SELECTED_TARGET_LOGIT_EXECUTION" \
        "$EXPECTED_POST_SELECTION_STATE_STORAGE" \
        "$EXPECTED_CUDA_HEADROOM_POLICY" \
        "$EXPECTED_CUDA_HEADROOM_ACTION"
)"

for target in "$EXPECTED_OUTPUT_DIR" "$SUMMARY_JSONL"; do
    if [[ -e "$target" ]] || [[ -L "$target" ]]; then
        echo "Refusing existing qualification output or summary target: ${target}" >&2
        exit 2
    fi
done

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "job_id=${SLURM_JOB_ID} node=${SLURMD_NODENAME} wave=${WAVE} source=${ONLY_ARTIFACT_ID}"
echo "backend=${EXPECTED_ATTENTION_BACKEND} config=${CONFIG} manifest=${MANIFEST}"
echo "stop_gradient_contribution_execution=${EXPECTED_STOP_GRADIENT_CONTRIBUTION_EXECUTION}"
echo "stop_gradient_contribution_target_lane_chunk_size=${EXPECTED_STOP_GRADIENT_CONTRIBUTION_TARGET_LANE_CHUNK_SIZE}"
echo "selected_neuron_contribution_target_lane_chunk_size=${EXPECTED_SELECTED_NEURON_CONTRIBUTION_TARGET_LANE_CHUNK_SIZE}"
echo "selected_embed_contribution_target_lane_chunk_size=${EXPECTED_SELECTED_EMBED_CONTRIBUTION_TARGET_LANE_CHUNK_SIZE}"
echo "stop_gradient_embed_contribution_target_lane_chunk_size=${EXPECTED_STOP_GRADIENT_EMBED_CONTRIBUTION_TARGET_LANE_CHUNK_SIZE}"
echo "selected_attribution_neuron_lane_chunk_size=${EXPECTED_SELECTED_ATTRIBUTION_NEURON_LANE_CHUNK_SIZE}"
echo "stop_gradient_selected_attribution_forward_execution=${EXPECTED_STOP_GRADIENT_SELECTED_ATTRIBUTION_FORWARD_EXECUTION}"
echo "stop_gradient_selected_attribution_storage=${EXPECTED_STOP_GRADIENT_SELECTED_ATTRIBUTION_STORAGE}"
echo "selected_target_logit_execution=${EXPECTED_SELECTED_TARGET_LOGIT_EXECUTION}"
echo "post_selection_state_storage=${EXPECTED_POST_SELECTION_STATE_STORAGE}"
echo "cuda_allocator_policy=${EXPECTED_CUDA_ALLOCATOR_POLICY:-legacy_unbound} pytorch_cuda_alloc_conf=${PYTORCH_CUDA_ALLOC_CONF-unset}"
echo "cuda_headroom_policy=${EXPECTED_CUDA_HEADROOM_POLICY}"
echo "cuda_headroom_action=${EXPECTED_CUDA_HEADROOM_ACTION}"
echo "embedding_edge_materialization=${EXPECTED_EMBEDDING_EDGE_MATERIALIZATION:-legacy_unbound}"
echo "cross_layer_jacobian_execution=${EXPECTED_CROSS_LAYER_JACOBIAN_EXECUTION:-legacy_unbound}"
echo "artifact_root=${ARTIFACT_ROOT} expected_output=${EXPECTED_OUTPUT_DIR}"
echo "summary_jsonl=${SUMMARY_JSONL} uv_environment=${UV_PROJECT_ENVIRONMENT}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

GPU_MONITOR=/uufs/chpc.utah.edu/sys/installdir/chpcscripts/gpu/gpu-monitor.sh
monitor_pid=""
runner_pid=""
if [[ -x "$GPU_MONITOR" ]]; then
    (
        cd "${REPO_ROOT}/logs"
        exec "$GPU_MONITOR"
    ) &
    monitor_pid=$!
fi
forward_usr1() {
    echo "Forwarding Slurm USR1 warning to top-k runner pid=${runner_pid}." >&2
    if [[ -n "$runner_pid" ]] && kill -0 "$runner_pid" 2>/dev/null; then
        kill -USR1 "$runner_pid"
    fi
}
cleanup() {
    if [[ -n "$monitor_pid" ]]; then
        kill "$monitor_pid" 2>/dev/null || true
        wait "$monitor_pid" 2>/dev/null || true
    fi
}
trap forward_usr1 USR1
trap cleanup EXIT

"$PYTHON_BIN" -m scripts.bonafide.topk_runner \
    --config "$CONFIG" \
    --manifest "$MANIFEST" \
    --wave "$WAVE" \
    --only-artifact-id "$ONLY_ARTIFACT_ID" \
    --artifact-root "$ARTIFACT_ROOT" \
    --summary-jsonl "$SUMMARY_JSONL" &
runner_pid=$!
set +e
wait "$runner_pid"
runner_status=$?
set -e
if [[ "$runner_status" -ne 0 ]]; then
    exit "$runner_status"
fi

"$PYTHON_BIN" -c \
    'import json,pathlib,sys
from scripts.bonafide.cuda_allocator_policy import ALLOCATOR_ENVIRONMENT_VARIABLE, pytorch_cuda_alloc_conf_for_policy
from scripts.bonafide.cuda_headroom import assess_cuda_headroom, headroom_action_requires_stop, headroom_gate_passed
from circuits.tracing.backend_qualification import post_selection_state_expected_values_identity, validate_post_selection_state_storage_runtime
summary=pathlib.Path(sys.argv[1])
expected_source=sys.argv[2]
expected_output=str(pathlib.Path(sys.argv[3]))
expected_backend=sys.argv[4]
expected_contribution_execution=sys.argv[5]
expected_chunk_size_raw=sys.argv[6]
expected_chunk_size=None if expected_chunk_size_raw == "none" else int(expected_chunk_size_raw)
expected_gpu=sys.argv[7]
expected_allocator_policy=sys.argv[8] or None
expected_embedding_edge_materialization=sys.argv[9] or None
expected_cross_layer_jacobian_execution=sys.argv[10] or None
expected_selected_chunk_size_raw=sys.argv[11]
expected_selected_chunk_size=(
    None
    if expected_selected_chunk_size_raw in {"legacy_unbound", "none"}
    else int(expected_selected_chunk_size_raw)
)
expected_selected_embed_chunk_size_raw=sys.argv[12]
expected_selected_embed_chunk_size=(
    None
    if expected_selected_embed_chunk_size_raw in {"legacy_unbound", "none"}
    else int(expected_selected_embed_chunk_size_raw)
)
expected_stop_gradient_embed_chunk_size_raw=sys.argv[13]
expected_stop_gradient_embed_chunk_size=(
    None
    if expected_stop_gradient_embed_chunk_size_raw in {"legacy_unbound", "none"}
    else int(expected_stop_gradient_embed_chunk_size_raw)
)
expected_selected_attribution_chunk_size_raw=sys.argv[14]
expected_selected_attribution_chunk_size=(
    None
    if expected_selected_attribution_chunk_size_raw in {"legacy_unbound", "none"}
    else int(expected_selected_attribution_chunk_size_raw)
)
expected_selected_forward_execution=sys.argv[15]
expected_selected_storage=sys.argv[16]
expected_selected_target_logit_execution=sys.argv[17]
expected_post_selection_state_storage=sys.argv[18]
expected_cuda_headroom_policy=sys.argv[19]
expected_cuda_headroom_action=sys.argv[20]
rows=[json.loads(line) for line in summary.read_text(encoding="utf-8").splitlines() if line.strip()]
if len(rows) != 1:
    raise SystemExit("qualification summary must contain exactly one record")
row=rows[0]
if row.get("status") != "complete" or row.get("source_width1_artifact_id") != expected_source:
    raise SystemExit("qualification item did not complete exactly once")
if str(pathlib.Path(row.get("artifact_path", ""))) != expected_output:
    raise SystemExit("qualification output path disagrees with the preflight identity")
runtime=row.get("runtime_environment", {}).get("gpu_runtime", {})
devices=runtime.get("devices", [])
if len(devices) != 1 or devices[0].get("name") != expected_gpu or not runtime.get("driver_versions"):
    raise SystemExit("qualification summary lacks the exact GPU/driver runtime receipt")
device_total_bytes=devices[0].get("total_memory_bytes")
if type(device_total_bytes) is not int or device_total_bytes <= 0:
    raise SystemExit("qualification summary lacks a valid GPU total-memory receipt")
artifact_manifest=json.loads((pathlib.Path(expected_output)/"manifest.json").read_text(encoding="utf-8"))
backend=artifact_manifest.get("artifact_identity", {}).get("adag_config", {}).get("stop_gradient_attention_backend")
if backend != expected_backend:
    raise SystemExit("saved artifact backend identity disagrees with requested backend")
identity=artifact_manifest.get("artifact_identity", {})
expected_cuda_headroom_contract={
    "policy": expected_cuda_headroom_policy,
    "min_cuda_headroom_bytes": 8589934592,
    "action": expected_cuda_headroom_action,
    "sampling_version": "boundary_cuda_metrics_v1",
}
if expected_cuda_headroom_policy == "allocator_dense_joint_v1":
    if identity.get("cuda_headroom_gate") != expected_cuda_headroom_contract:
        raise SystemExit("saved artifact CUDA headroom gate identity disagrees with requested contract")
elif "cuda_headroom_gate" in identity:
    raise SystemExit("legacy artifact unexpectedly binds a CUDA headroom identity field")
headroom_receipt=row.get("cuda_headroom_gate")
row_instrumentation=row.get("instrumentation")
if not isinstance(row_instrumentation, dict):
    raise SystemExit("qualification summary lacks instrumentation for CUDA headroom recomputation")
try:
    expected_headroom_receipt=assess_cuda_headroom(
        policy=expected_cuda_headroom_policy,
        threshold_bytes=8589934592,
        action=expected_cuda_headroom_action,
        device_total_bytes=device_total_bytes,
        peak_reserved_bytes=row.get("cuda_peak_reserved_bytes"),
        instrumentation=row_instrumentation,
    )
except (TypeError, ValueError) as error:
    raise SystemExit(f"CUDA headroom gate recomputation failed: {error}") from error
if headroom_receipt != expected_headroom_receipt:
    raise SystemExit("saved CUDA headroom gate receipt disagrees with exact recomputation")
try:
    headroom_passed=headroom_gate_passed(
        headroom_receipt,
        expected_policy=expected_cuda_headroom_policy,
        expected_threshold_bytes=8589934592,
        expected_action=expected_cuda_headroom_action,
    )
except (TypeError, ValueError) as error:
    raise SystemExit(f"saved CUDA headroom gate receipt is invalid: {error}") from error
try:
    headroom_should_stop=headroom_action_requires_stop(
        headroom_receipt,
        expected_policy=expected_cuda_headroom_policy,
        expected_threshold_bytes=8589934592,
        expected_action=expected_cuda_headroom_action,
    )
except (TypeError, ValueError) as error:
    raise SystemExit(f"saved CUDA headroom action decision is invalid: {error}") from error
if headroom_should_stop:
    raise SystemExit("saved CUDA headroom gate receipt did not pass the qualification threshold")
classification=headroom_receipt.get("classification")
warning=row.get("cuda_headroom_warning")
if classification == "comfortable":
    if warning is not None:
        raise SystemExit("comfortable CUDA headroom receipt unexpectedly emitted a warning")
elif expected_cuda_headroom_action == "warn":
    if warning != headroom_receipt.get("warning") or not isinstance(warning, dict):
        raise SystemExit("advisory CUDA headroom receipt lacks its durable warning")
elif not headroom_passed:
    raise SystemExit("stopping CUDA headroom receipt did not pass the threshold")
artifact_adag=identity.get("adag_config", {})
observed_contribution_execution=artifact_adag.get("stop_gradient_contribution_execution")
if observed_contribution_execution != expected_contribution_execution:
    raise SystemExit("saved artifact contribution execution identity disagrees with requested strategy")
chunk_size_field="stop_gradient_contribution_target_lane_chunk_size"
observed_chunk_size=artifact_adag.get(chunk_size_field)
if chunk_size_field not in artifact_adag or observed_chunk_size != expected_chunk_size:
    raise SystemExit("saved artifact target-lane chunk size identity disagrees with requested strategy")
selected_chunk_size_field="selected_neuron_contribution_target_lane_chunk_size"
selected_chunk_size_is_bound=expected_selected_chunk_size_raw != "legacy_unbound"
observed_selected_chunk_size=artifact_adag.get(selected_chunk_size_field)
if selected_chunk_size_is_bound:
    if selected_chunk_size_field not in artifact_adag or observed_selected_chunk_size != expected_selected_chunk_size:
        raise SystemExit("saved artifact selected-neuron target-lane chunk size identity disagrees with requested strategy")
elif selected_chunk_size_field in artifact_adag:
    raise SystemExit("legacy-unbound artifact unexpectedly declares selected-neuron target-lane chunk size")
selected_embed_chunk_size_field="selected_embed_contribution_target_lane_chunk_size"
selected_embed_chunk_size_is_bound=expected_selected_embed_chunk_size_raw != "legacy_unbound"
observed_selected_embed_chunk_size=artifact_adag.get(selected_embed_chunk_size_field)
if selected_embed_chunk_size_is_bound:
    if selected_embed_chunk_size_field not in artifact_adag or observed_selected_embed_chunk_size != expected_selected_embed_chunk_size:
        raise SystemExit("saved artifact selected-embed target-lane chunk size identity disagrees with requested strategy")
elif selected_embed_chunk_size_field in artifact_adag:
    raise SystemExit("legacy-unbound artifact unexpectedly declares selected-embed target-lane chunk size")
stop_gradient_embed_chunk_size_field="stop_gradient_embed_contribution_target_lane_chunk_size"
stop_gradient_embed_chunk_size_is_bound=expected_stop_gradient_embed_chunk_size_raw != "legacy_unbound"
observed_stop_gradient_embed_chunk_size=artifact_adag.get(stop_gradient_embed_chunk_size_field)
if stop_gradient_embed_chunk_size_is_bound:
    if stop_gradient_embed_chunk_size_field not in artifact_adag or observed_stop_gradient_embed_chunk_size != expected_stop_gradient_embed_chunk_size:
        raise SystemExit("saved artifact stop-gradient-embed target-lane chunk size identity disagrees with requested strategy")
elif stop_gradient_embed_chunk_size_field in artifact_adag:
    raise SystemExit("legacy-unbound artifact unexpectedly declares stop-gradient-embed target-lane chunk size")
selected_attribution_chunk_size_field="selected_attribution_neuron_lane_chunk_size"
selected_attribution_chunk_size_is_bound=(
    expected_selected_attribution_chunk_size_raw != "legacy_unbound"
)
observed_selected_attribution_chunk_size=artifact_adag.get(selected_attribution_chunk_size_field)
if selected_attribution_chunk_size_is_bound:
    if selected_attribution_chunk_size_field not in artifact_adag or observed_selected_attribution_chunk_size != expected_selected_attribution_chunk_size:
        raise SystemExit("saved artifact selected-attribution neuron-lane chunk size identity disagrees with requested strategy")
elif selected_attribution_chunk_size_field in artifact_adag:
    raise SystemExit("legacy-unbound artifact unexpectedly declares selected-attribution neuron-lane chunk size")
selected_forward_field="stop_gradient_selected_attribution_forward_execution"
selected_forward_is_bound=expected_selected_forward_execution != "legacy_unbound"
observed_selected_forward_execution=artifact_adag.get(selected_forward_field)
if selected_forward_is_bound:
    if selected_forward_field not in artifact_adag or observed_selected_forward_execution != expected_selected_forward_execution:
        raise SystemExit("saved artifact stop-gradient selected-attribution forward execution identity disagrees with requested strategy")
elif selected_forward_field in artifact_adag:
    raise SystemExit("legacy-unbound artifact unexpectedly declares stop-gradient selected-attribution forward execution")
if selected_forward_is_bound:
    counters=row.get("instrumentation", {}).get("counters", {})
    execution_count=counters.get("stop_gradient_selected_attribution_forward_execution_count")
    strategy_count=counters.get(f"stop_gradient_selected_attribution_{expected_selected_forward_execution}_execution_count")
    if not isinstance(execution_count, int) or execution_count <= 0 or strategy_count != execution_count:
        raise SystemExit("saved artifact lacks exact stop-gradient selected-attribution forward execution receipts")
    down_projection_count=counters.get("stop_gradient_selected_attribution_down_projection_materialized_count")
    decoder_suffix_count=counters.get("stop_gradient_selected_attribution_decoder_suffix_materialized_count")
    logits_count=counters.get("stop_gradient_selected_attribution_logits_materialized_count")
    if expected_selected_forward_execution == "full_model_v1":
        receipts_valid=(
            down_projection_count == execution_count
            and logits_count == execution_count
            and isinstance(decoder_suffix_count, int)
            and 0 <= decoder_suffix_count <= execution_count
        )
    else:
        receipts_valid=(down_projection_count == decoder_suffix_count == logits_count == 0)
    if not receipts_valid:
        raise SystemExit("saved artifact materialization receipts disagree with requested selected-attribution forward execution")
selected_storage_field="stop_gradient_selected_attribution_storage"
selected_storage_is_bound=expected_selected_storage != "legacy_unbound"
observed_selected_storage=artifact_adag.get(selected_storage_field)
if selected_storage_is_bound:
    if selected_storage_field not in artifact_adag or observed_selected_storage != expected_selected_storage:
        raise SystemExit("saved artifact stop-gradient selected-attribution storage identity disagrees with requested strategy")
elif selected_storage_field in artifact_adag:
    raise SystemExit("legacy-unbound artifact unexpectedly declares stop-gradient selected-attribution storage")
if selected_storage_is_bound:
    counters=row.get("instrumentation", {}).get("counters", {})
    storage_count=counters.get("stop_gradient_selected_attribution_storage_execution_count")
    strategy_count=counters.get(f"stop_gradient_selected_attribution_{expected_selected_storage}_storage_count")
    projected_graph_count=counters.get("stop_gradient_selected_attribution_projection_graph_retained_count")
    stored_graph_count=counters.get("stop_gradient_selected_attribution_stored_graph_retained_count")
    detached_count=counters.get("stop_gradient_selected_attribution_terminal_detached_count")
    if not isinstance(storage_count, int) or storage_count <= 0 or strategy_count != storage_count or projected_graph_count != storage_count:
        raise SystemExit("saved artifact lacks exact stop-gradient selected-attribution storage receipts")
    expected_stored_graph_count=storage_count if expected_selected_storage == "graph_retaining_v1" else 0
    expected_detached_count=storage_count if expected_selected_storage == "terminal_detached_v1" else 0
    if stored_graph_count != expected_stored_graph_count or detached_count != expected_detached_count:
        raise SystemExit("saved artifact graph-lifetime receipts disagree with requested selected-attribution storage")
selected_target_logit_field="selected_target_logit_execution"
selected_target_logit_is_bound=(
    expected_selected_target_logit_execution != "legacy_unbound"
)
observed_selected_target_logit_execution=artifact_adag.get(selected_target_logit_field)
if selected_target_logit_is_bound:
    if selected_target_logit_field not in artifact_adag or observed_selected_target_logit_execution != expected_selected_target_logit_execution:
        raise SystemExit("saved artifact selected target-logit execution identity disagrees with requested strategy")
elif selected_target_logit_field in artifact_adag:
    raise SystemExit("legacy-unbound artifact unexpectedly declares selected target-logit execution")
if selected_target_logit_is_bound:
    counters=row.get("instrumentation", {}).get("counters", {})
    if artifact_adag.get("center_logits") is not False:
        raise SystemExit("saved artifact selected target-logit qualification requires center_logits=false")
    if "ig_steps" not in artifact_adag:
        raise SystemExit("saved artifact selected target-logit execution lacks explicit ig_steps")
    ig_steps=artifact_adag["ig_steps"]
    if ig_steps is None:
        expected_execution_indexes=[None]
    elif type(ig_steps) is int and ig_steps > 0:
        expected_execution_indexes=list(range(ig_steps + 1))
    else:
        raise SystemExit("saved artifact selected target-logit execution has invalid ig_steps")
    records=row.get("instrumentation", {}).get("execution_records", {}).get("selected_target_logit_execution")
    observed_execution_indexes=(
        [record.get("execution_index") for record in records]
        if isinstance(records, list) and all(isinstance(record, dict) for record in records)
        else None
    )
    execution_count=counters.get("selected_target_logit_execution_count")
    strategy_count=counters.get(f"selected_target_logit_{expected_selected_target_logit_execution}_execution_count")
    full_materialized_count=counters.get("selected_target_logit_full_sequence_logits_materialized_count")
    selected_materialized_count=counters.get("selected_target_logit_selected_position_logits_materialized_count")
    lm_head_position_row_count=counters.get("selected_target_logit_lm_head_position_rows")
    expected_full_count=(execution_count if expected_selected_target_logit_execution == "full_logits_v1" else 0)
    if execution_count != len(expected_execution_indexes) or strategy_count != execution_count:
        raise SystemExit("saved artifact lacks exact selected target-logit execution receipts")
    if observed_execution_indexes != expected_execution_indexes:
        raise SystemExit("saved artifact selected target-logit execution indexes disagree with ig_steps")
    if full_materialized_count != expected_full_count or selected_materialized_count != execution_count:
        raise SystemExit("saved artifact materialization receipts disagree with requested selected target-logit execution")
    full_strategy=expected_selected_target_logit_execution == "full_logits_v1"
    observed_lm_head_rows=0
    full_sequence_baseline_rows=0
    every_candidate_record_reduces=True
    for record in records:
        batch_size=record.get("batch_size")
        sequence_count=record.get("sequence_position_count")
        selected_count=record.get("selected_position_count")
        unique_count=record.get("unique_selected_position_count")
        vocab_size=record.get("vocab_size")
        lm_head_input_shape=record.get("lm_head_input_shape")
        lm_head_output_shape=record.get("lm_head_output_shape")
        selected_logit_shape=record.get("selected_position_logit_shape")
        target_logit_shape=record.get("target_logit_shape")
        expected_head_positions=sequence_count if full_strategy else selected_count
        shapes_valid=(
            type(batch_size) is int
            and batch_size > 0
            and type(sequence_count) is int
            and sequence_count > 0
            and type(selected_count) is int
            and selected_count > 0
            and type(unique_count) is int
            and 0 < unique_count <= min(selected_count, sequence_count)
            and type(vocab_size) is int
            and vocab_size > 0
            and isinstance(lm_head_input_shape, list)
            and len(lm_head_input_shape) == 3
            and lm_head_input_shape[:2] == [batch_size, expected_head_positions]
            and type(lm_head_input_shape[2]) is int
            and lm_head_input_shape[2] > 0
            and lm_head_output_shape == [batch_size, expected_head_positions, vocab_size]
            and selected_logit_shape == [batch_size, selected_count, vocab_size]
            and target_logit_shape == [selected_count, batch_size]
        )
        if not shapes_valid:
            raise SystemExit("saved artifact selected target-logit execution has malformed shape receipts")
        execution_receipts_valid=(
            record.get("execution") == expected_selected_target_logit_execution
            and record.get("causal_lm_forward_completed") is True
            and record.get("selected_position_request_forwarded") is (not full_strategy)
            and record.get("full_sequence_logits_materialized") is full_strategy
            and record.get("selected_position_logits_materialized") is True
            and record.get("center_logits") is False
        )
        if not execution_receipts_valid:
            raise SystemExit("saved artifact selected target-logit execution record disagrees with requested strategy")
        observed_lm_head_rows += batch_size * lm_head_input_shape[1]
        full_sequence_baseline_rows += batch_size * sequence_count
        if not full_strategy:
            every_candidate_record_reduces=(
                every_candidate_record_reduces
                and selected_count < sequence_count
            )
    if lm_head_position_row_count != observed_lm_head_rows:
        raise SystemExit("saved artifact selected target-logit LM-head row counter disagrees with records")
    if not full_strategy and (
        not every_candidate_record_reduces
        or observed_lm_head_rows >= full_sequence_baseline_rows
    ):
        raise SystemExit("saved artifact selected target-logit execution did not reduce LM-head rows")
post_selection_state_storage_field="post_selection_state_storage"
post_selection_state_storage_is_bound=(
    expected_post_selection_state_storage != "legacy_unbound"
)
observed_post_selection_state_storage=artifact_adag.get(post_selection_state_storage_field)
if post_selection_state_storage_is_bound:
    if post_selection_state_storage_field not in artifact_adag or observed_post_selection_state_storage != expected_post_selection_state_storage:
        raise SystemExit("saved artifact post-selection state storage identity disagrees with requested strategy")
elif post_selection_state_storage_field in artifact_adag:
    raise SystemExit("legacy-unbound artifact unexpectedly declares post-selection state storage")
if post_selection_state_storage_is_bound:
    receipt_valid,_post_selection_runtime=validate_post_selection_state_storage_runtime(
        row.get("instrumentation"),
        expected_strategy=expected_post_selection_state_storage,
        **post_selection_state_expected_values_identity(identity),
    )
    if not receipt_valid:
        raise SystemExit("saved artifact post-selection state storage receipt disagrees with requested strategy")
observed_embedding_edge_materialization=artifact_adag.get("embedding_edge_materialization")
if observed_embedding_edge_materialization != expected_embedding_edge_materialization:
    if expected_embedding_edge_materialization is not None or "embedding_edge_materialization" in artifact_adag:
        raise SystemExit("saved artifact embedding-edge materialization identity disagrees with requested strategy")
observed_cross_layer_jacobian_execution=artifact_adag.get("cross_layer_jacobian_execution")
if observed_cross_layer_jacobian_execution != expected_cross_layer_jacobian_execution:
    if expected_cross_layer_jacobian_execution is not None or "cross_layer_jacobian_execution" in artifact_adag:
        raise SystemExit("saved artifact cross-layer Jacobian execution identity disagrees with requested strategy")
if identity.get("cuda_allocator_policy") != expected_allocator_policy:
    if expected_allocator_policy is not None or "cuda_allocator_policy" in identity:
        raise SystemExit("saved artifact allocator identity disagrees with requested policy")
artifact_runtime=artifact_manifest.get("runtime_environment", {})
if artifact_runtime != identity.get("runtime_environment") or artifact_runtime != row.get("runtime_environment"):
    raise SystemExit("saved allocator runtime receipt disagrees across identity, manifest, and summary")
receipt=artifact_runtime.get("cuda_allocator_policy")
if expected_allocator_policy is not None:
    expected_value=pytorch_cuda_alloc_conf_for_policy(expected_allocator_policy)
    expected_receipt={
        "intended_policy_id": expected_allocator_policy,
        "observed_environment": {
            "name": ALLOCATOR_ENVIRONMENT_VARIABLE,
            "value": expected_value,
            "is_set": expected_value is not None,
        },
        "observed_allocator_backend": "native",
    }
    if receipt != expected_receipt:
        raise SystemExit("saved artifact lacks the exact requested allocator runtime receipt")' \
    "$SUMMARY_JSONL" "$ONLY_ARTIFACT_ID" "$EXPECTED_OUTPUT_DIR" \
    "$EXPECTED_ATTENTION_BACKEND" \
    "$EXPECTED_STOP_GRADIENT_CONTRIBUTION_EXECUTION" \
    "$EXPECTED_STOP_GRADIENT_CONTRIBUTION_TARGET_LANE_CHUNK_SIZE" \
    "$EXPECTED_GPU_NAME" \
    "$EXPECTED_CUDA_ALLOCATOR_POLICY" \
    "$EXPECTED_EMBEDDING_EDGE_MATERIALIZATION" \
    "$EXPECTED_CROSS_LAYER_JACOBIAN_EXECUTION" \
    "$EXPECTED_SELECTED_NEURON_CONTRIBUTION_TARGET_LANE_CHUNK_SIZE" \
    "$EXPECTED_SELECTED_EMBED_CONTRIBUTION_TARGET_LANE_CHUNK_SIZE" \
    "$EXPECTED_STOP_GRADIENT_EMBED_CONTRIBUTION_TARGET_LANE_CHUNK_SIZE" \
    "$EXPECTED_SELECTED_ATTRIBUTION_NEURON_LANE_CHUNK_SIZE" \
    "$EXPECTED_STOP_GRADIENT_SELECTED_ATTRIBUTION_FORWARD_EXECUTION" \
    "$EXPECTED_STOP_GRADIENT_SELECTED_ATTRIBUTION_STORAGE" \
    "$EXPECTED_SELECTED_TARGET_LOGIT_EXECUTION" \
    "$EXPECTED_POST_SELECTION_STATE_STORAGE" \
    "$EXPECTED_CUDA_HEADROOM_POLICY" \
    "$EXPECTED_CUDA_HEADROOM_ACTION"
