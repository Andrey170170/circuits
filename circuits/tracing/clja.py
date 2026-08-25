"""
Circuit-level attribution code, using Jacobian vector product (JVP) to compute edge weights and
compute the entire circuit. Calls core.py to get attributions to filter out unimportant neurons
and make edge-weight computation tractable.
"""

import math
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Literal

import torch
from transformers import PreTrainedTokenizer

from circuits.tracing.attribution import (
    _get_global_important_neurons_mask,
    _get_grad_attributions_from_logits,
    _get_ig_attributions_from_logits,
    _get_neuron_attr_and_contrib,
    _get_neuron_attr_and_contrib_ig,
    _get_neuron_attr_and_contrib_with_stop_grad_on_mlps,
)
from circuits.tracing.candidates import (
    CandidateLogitAxis,
    reduce_candidate_contributions,
)
from circuits.tracing.contribution_execution import (
    DEFAULT_STOP_GRADIENT_CONTRIBUTION_EXECUTION,
    StopGradientContributionExecution,
    resolve_selected_embed_contribution_target_lane_chunk_size,
    resolve_selected_neuron_contribution_target_lane_chunk_size,
    resolve_stop_gradient_contribution_execution,
    resolve_stop_gradient_contribution_target_lane_chunk_size,
    resolve_stop_gradient_embed_contribution_target_lane_chunk_size,
)
from circuits.tracing.cross_layer_jacobian_execution import (
    DEFAULT_CROSS_LAYER_JACOBIAN_EXECUTION,
    CrossLayerJacobianExecution,
    CrossLayerJacobianPair,
    CrossLayerJacobianPairResult,
    CrossLayerJacobianPreparation,
    prepare_cross_layer_jacobian_execution,
    resolve_cross_layer_jacobian_execution,
)
from circuits.tracing.embedding_edge_materialization import (
    DEFAULT_EMBEDDING_EDGE_MATERIALIZATION,
    EmbeddingEdgeMaterialization,
    EmbeddingEdgeMaterializationRequest,
    EmbeddingSource,
    EmbeddingTarget,
    materialize_embedding_edges,
    resolve_embedding_edge_materialization,
)
from circuits.tracing.grad import (
    DEFAULT_STOP_GRADIENT_ATTENTION_BACKEND,
    StopGradientAttentionBackend,
    layerwise_revert_stop_nonlinear_grad,
    layerwise_stop_nonlinear_grad,
    remove_forward_hooks,
    resolve_stop_gradient_attention_backend,
    revert_active_stop_nonlinear_grad,
    revert_stop_nonlinear_grad,
    stop_nonlinear_grad,
)
from circuits.tracing.instrumentation import (
    TraceInstrumentation,
    instrumentation_stage,
    record_selection_predictors,
)
from circuits.tracing.utils import Edge, NeuronIdx, Node, collect_neuron_acts

BLACKLISTED_NEURONS: dict[str, list[NeuronIdx]] = {
    "meta-llama/Llama-3.1-8B-Instruct": [
        NeuronIdx(layer=23, token=-1, neuron=306),
        NeuronIdx(layer=20, token=-1, neuron=3972),
        NeuronIdx(layer=18, token=-1, neuron=7417),
        NeuronIdx(layer=16, token=-1, neuron=1241),
        NeuronIdx(layer=13, token=-1, neuron=4208),
        NeuronIdx(layer=11, token=-1, neuron=11321),
        NeuronIdx(layer=10, token=-1, neuron=11570),
        NeuronIdx(layer=9, token=-1, neuron=4255),
        NeuronIdx(layer=7, token=-1, neuron=6673),
        NeuronIdx(layer=6, token=-1, neuron=5866),
        NeuronIdx(layer=5, token=-1, neuron=7012),
        NeuronIdx(layer=2, token=-1, neuron=4786),
    ],
}


def get_blacklisted_neurons(model) -> list[NeuronIdx]:
    """Get blacklisted neurons for the given model, defaulting to empty list."""
    model_id = getattr(model.config, "_name_or_path", "")
    return BLACKLISTED_NEURONS.get(model_id, [])


@dataclass
class ADAGConfig:
    """Configuration for ADAG circuit tracing."""

    # Basic settings
    device: str = "cuda:0"
    verbose: bool = False
    return_only_important_neurons: bool = False
    return_nodes_only: bool = False
    skip_attr_contrib: bool = False

    # Gradient settings
    use_relp_grad: bool = False
    disable_half_rule: bool = False
    disable_stop_grad: bool = False
    use_stop_grad_on_mlps: bool = False
    stop_gradient_attention_backend: StopGradientAttentionBackend = (
        DEFAULT_STOP_GRADIENT_ATTENTION_BACKEND
    )
    ablation_mode: Literal["zero", "mean"] = "zero"
    center_logits: bool = False

    # IG settings
    ig_steps: int | None = None
    ig_mode: Literal["ig-inputs", "conductance"] = "ig-inputs"

    # Edge pruning settings
    node_attribution_threshold: float | None = 1.0
    topk_neurons: int | None = None
    parent_threshold: float | None = None
    edge_threshold: float | None = None
    topk: int | None = None
    percentage_threshold: float | None = None
    batch_aggregation: Literal["mean", "max", "max_abs", "any"] = "mean"
    return_absolute: bool = False
    apply_blacklist: bool = False

    # Layer settings
    start_layer: int | None = None
    end_layer: int | None = None

    # Tracing settings
    focus_last_residual: bool = False
    # Appended to preserve the positional order of historical fields.
    stop_gradient_contribution_execution: StopGradientContributionExecution = (
        DEFAULT_STOP_GRADIENT_CONTRIBUTION_EXECUTION
    )
    embedding_edge_materialization: EmbeddingEdgeMaterialization = (
        DEFAULT_EMBEDDING_EDGE_MATERIALIZATION
    )
    cross_layer_jacobian_execution: CrossLayerJacobianExecution = (
        DEFAULT_CROSS_LAYER_JACOBIAN_EXECUTION
    )
    # Appended to preserve the positional order of historical fields.
    stop_gradient_contribution_target_lane_chunk_size: int | None = None
    # Independent from the stop-gradient contribution execution above.
    selected_neuron_contribution_target_lane_chunk_size: int | None = None
    selected_embed_contribution_target_lane_chunk_size: int | None = None
    stop_gradient_embed_contribution_target_lane_chunk_size: int | None = None

    def __post_init__(self) -> None:
        resolve_stop_gradient_attention_backend(self.stop_gradient_attention_backend)
        resolve_stop_gradient_contribution_execution(
            self.stop_gradient_contribution_execution
        )
        resolve_embedding_edge_materialization(self.embedding_edge_materialization)
        resolve_cross_layer_jacobian_execution(self.cross_layer_jacobian_execution)
        resolve_stop_gradient_contribution_target_lane_chunk_size(
            self.stop_gradient_contribution_target_lane_chunk_size
        )
        resolve_selected_neuron_contribution_target_lane_chunk_size(
            self.selected_neuron_contribution_target_lane_chunk_size
        )
        resolve_selected_embed_contribution_target_lane_chunk_size(
            self.selected_embed_contribution_target_lane_chunk_size
        )
        resolve_stop_gradient_embed_contribution_target_lane_chunk_size(
            self.stop_gradient_embed_contribution_target_lane_chunk_size
        )

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Load artifacts pickled before stop-gradient strategies were explicit."""

        self.__dict__.update(state)
        if "stop_gradient_attention_backend" not in state:
            self.stop_gradient_attention_backend = (
                DEFAULT_STOP_GRADIENT_ATTENTION_BACKEND
            )
        if "stop_gradient_contribution_execution" not in state:
            self.stop_gradient_contribution_execution = (
                DEFAULT_STOP_GRADIENT_CONTRIBUTION_EXECUTION
            )
        if "embedding_edge_materialization" not in state:
            self.embedding_edge_materialization = DEFAULT_EMBEDDING_EDGE_MATERIALIZATION
        if "cross_layer_jacobian_execution" not in state:
            self.cross_layer_jacobian_execution = DEFAULT_CROSS_LAYER_JACOBIAN_EXECUTION
        if "stop_gradient_contribution_target_lane_chunk_size" not in state:
            self.stop_gradient_contribution_target_lane_chunk_size = None
        if "selected_neuron_contribution_target_lane_chunk_size" not in state:
            self.selected_neuron_contribution_target_lane_chunk_size = None
        if "selected_embed_contribution_target_lane_chunk_size" not in state:
            self.selected_embed_contribution_target_lane_chunk_size = None
        if "stop_gradient_embed_contribution_target_lane_chunk_size" not in state:
            self.stop_gradient_embed_contribution_target_lane_chunk_size = None
        resolve_stop_gradient_attention_backend(self.stop_gradient_attention_backend)
        resolve_stop_gradient_contribution_execution(
            self.stop_gradient_contribution_execution
        )
        resolve_embedding_edge_materialization(self.embedding_edge_materialization)
        resolve_cross_layer_jacobian_execution(self.cross_layer_jacobian_execution)
        resolve_stop_gradient_contribution_target_lane_chunk_size(
            self.stop_gradient_contribution_target_lane_chunk_size
        )
        resolve_selected_neuron_contribution_target_lane_chunk_size(
            self.selected_neuron_contribution_target_lane_chunk_size
        )
        resolve_selected_embed_contribution_target_lane_chunk_size(
            self.selected_embed_contribution_target_lane_chunk_size
        )
        resolve_stop_gradient_embed_contribution_target_lane_chunk_size(
            self.stop_gradient_embed_contribution_target_lane_chunk_size
        )


@dataclass(frozen=True)
class CLJAProbeSelection:
    """Compact result available before selected-attribution and graph work.

    ``selected_occurrences`` is ordered by ``(layer, token, neuron)`` and is
    intentionally JSON-shaped.  Attribution values are the initial-logit
    attributions used to construct the important-neuron mask, not later edge
    or contribution values.
    """

    selected_occurrences: list[dict[str, int | float]]
    effective_start_layer: int
    effective_end_layer: int


@dataclass(frozen=True)
class FrozenGraphTopology:
    """Exact candidate-union topology to rescore without selection or pruning."""

    mlp_nodes: frozenset[NeuronIdx]
    edges: frozenset[tuple[NeuronIdx, NeuronIdx]]


def _selected_probe_occurrences(
    mlp_final_attributions: torch.Tensor,
    neuron_cfg: dict[int, list[list[int]]],
    keep_tokens: list[int],
) -> list[dict[str, int | float]]:
    """Export selected occurrence attributions with one device transfer."""

    keep = set(keep_tokens)
    occurrence_ids = [
        (int(layer), int(token), int(neuron))
        for layer in sorted(neuron_cfg)
        for token, neuron in neuron_cfg[layer]
        if int(token) in keep
    ]
    if not occurrence_ids:
        return []
    device = mlp_final_attributions.device
    layer_indices = torch.tensor(
        [item[0] for item in occurrence_ids], device=device, dtype=torch.long
    )
    token_indices = torch.tensor(
        [item[1] for item in occurrence_ids], device=device, dtype=torch.long
    )
    neuron_indices = torch.tensor(
        [item[2] for item in occurrence_ids], device=device, dtype=torch.long
    )
    attribution_values = (
        mlp_final_attributions[layer_indices, 0, token_indices, neuron_indices, 0]
        .detach()
        .float()
        .cpu()
        .tolist()
    )
    return [
        {
            "layer": layer,
            "token_position": token,
            "neuron": neuron,
            "attribution": float(attribution),
        }
        for (layer, token, neuron), attribution in zip(
            occurrence_ids, attribution_values, strict=True
        )
    ]


def _get_all_pairs_cl_ja_effects_with_attributions_impl(
    model,
    tokenizer: PreTrainedTokenizer,
    cis: list[list[int]],
    # config
    config: ADAGConfig,
    # where to trace from and to
    src_tokens: list[int],
    tgt_tokens: list[int],
    keep_tokens: list[int] | None = None,
    attention_masks: list[list[int]] | torch.Tensor | None = None,
    focus_positions: list[int] | None = None,
    focus_logits: list[list[int]] | list[int] | None = None,
    candidate_axis: CandidateLogitAxis | None = None,
    frozen_topology: FrozenGraphTopology | None = None,
    instrumentation: TraceInstrumentation | None = None,
    probe_only: bool = False,
) -> (
    tuple[list[Node], list[Edge]]
    | tuple[torch.Tensor, torch.Tensor]
    | CLJAProbeSelection
):
    """
    Cross Layer Jacobian Attribution (CLJA) for circuit tracing.

    This function follows the exact same procedure as the original CLSO algorithm
    for finding important neurons, but uses jacobian computation for edge weights
    instead of the Cross Layer Second Order (CLSO) lens method.

    Args:
        Same as get_all_pairs_cl_so_effects_with_attributions in core.py

    Returns:
        tuple[list[Node], list[Edge]]: Circuit nodes and edges
        torch.Tensor: Only return the final attributions for the important neurons
    """
    ############
    # SETTINGS #
    ############
    device = config.device
    verbose = config.verbose
    return_only_important_neurons = config.return_only_important_neurons
    return_nodes_only = config.return_nodes_only
    use_relp_grad = config.use_relp_grad
    disable_half_rule = config.disable_half_rule
    disable_stop_grad = config.disable_stop_grad
    use_stop_grad_on_mlps = config.use_stop_grad_on_mlps
    stop_gradient_attention_backend = config.stop_gradient_attention_backend
    stop_gradient_contribution_execution = config.stop_gradient_contribution_execution
    stop_gradient_contribution_target_lane_chunk_size = (
        config.stop_gradient_contribution_target_lane_chunk_size
    )
    selected_neuron_contribution_target_lane_chunk_size = (
        config.selected_neuron_contribution_target_lane_chunk_size
    )
    selected_embed_contribution_target_lane_chunk_size = (
        config.selected_embed_contribution_target_lane_chunk_size
    )
    stop_gradient_embed_contribution_target_lane_chunk_size = (
        config.stop_gradient_embed_contribution_target_lane_chunk_size
    )
    embedding_edge_materialization = config.embedding_edge_materialization
    cross_layer_jacobian_execution = config.cross_layer_jacobian_execution
    if instrumentation is not None:
        instrumentation.set_counter(
            "stop_gradient_attention_backend",
            stop_gradient_attention_backend,
        )
        instrumentation.set_counter(
            "stop_gradient_contribution_execution",
            stop_gradient_contribution_execution,
        )
        instrumentation.set_counter(
            "stop_gradient_contribution_target_lane_chunk_size",
            stop_gradient_contribution_target_lane_chunk_size,
        )
        instrumentation.set_counter(
            "selected_neuron_contribution_target_lane_chunk_size",
            selected_neuron_contribution_target_lane_chunk_size,
        )
        instrumentation.set_counter(
            "selected_embed_contribution_target_lane_chunk_size",
            selected_embed_contribution_target_lane_chunk_size,
        )
        instrumentation.set_counter(
            "stop_gradient_embed_contribution_target_lane_chunk_size",
            stop_gradient_embed_contribution_target_lane_chunk_size,
        )
        instrumentation.set_counter(
            "embedding_edge_materialization",
            embedding_edge_materialization,
        )
        instrumentation.set_counter(
            "cross_layer_jacobian_execution",
            cross_layer_jacobian_execution,
        )
    ablation_mode = config.ablation_mode
    center_logits = config.center_logits
    ig_steps = config.ig_steps
    ig_mode = config.ig_mode
    # edge pruning settings
    node_attribution_threshold = config.node_attribution_threshold
    topk_neurons = config.topk_neurons
    parent_threshold = config.parent_threshold
    edge_threshold = config.edge_threshold
    topk = config.topk
    percentage_threshold = config.percentage_threshold
    batch_aggregation = config.batch_aggregation
    return_absolute = config.return_absolute
    apply_blacklist = config.apply_blacklist
    # more circuit settings
    focus_last_residual = config.focus_last_residual
    start_layer = config.start_layer
    end_layer = config.end_layer
    skip_attr_contrib = config.skip_attr_contrib

    #########
    # SETUP #
    #########

    if disable_stop_grad and (use_relp_grad or use_stop_grad_on_mlps):
        print(
            "warning: stop grad is disabled but some stop grad configurations are used"
        )

    objective_weights: tuple[float, ...] | None = None
    contribution_tgt_tokens = tgt_tokens
    if candidate_axis is not None:
        if focus_positions is not None or focus_logits is not None:
            raise ValueError(
                "candidate_axis cannot be combined with legacy focus positions/logits"
            )
        if len(cis) != len(candidate_axis.token_ids_by_batch):
            raise ValueError("candidate axis batch width must match input batch width")
        if tgt_tokens != [candidate_axis.prediction_position]:
            raise ValueError(
                "candidate traces require one shared scientific target position"
            )
        focus_positions = [
            candidate_axis.prediction_position
        ] * candidate_axis.candidate_count
        focus_logits = [
            list(token_ids) for token_ids in candidate_axis.token_ids_by_batch
        ]
        contribution_tgt_tokens = list(focus_positions)
        objective_weights = candidate_axis.objective_weights
    elif focus_positions is None:
        focus_positions = tgt_tokens

    if focus_logits is None and not focus_last_residual:
        try:
            focus_logits = [cis[0][pos_idx + 1] for pos_idx in focus_positions]
        except Exception as e:
            print(e)
            max_token_expected = max(focus_positions) + 1
            raise ValueError(
                f"failed to get labels for {max_token_expected} tokens."
            ) from e

    if probe_only:
        if len(cis) != 1:
            raise ValueError("probe_only requires exactly one input sequence")
        if len(focus_positions) != 1:
            raise ValueError("probe_only requires exactly one target position")

    if keep_tokens is None:
        keep_tokens = list(range(max(tgt_tokens) + 1))

    # start and end layer
    start_layer = -1 if start_layer is None else start_layer
    end_layer = model.config.num_hidden_layers if end_layer is None else end_layer

    # get input ids
    input_ids = torch.tensor(cis, device=device)
    if isinstance(attention_masks, torch.Tensor):
        attn_mask_final = attention_masks.to(device)
    else:
        attn_mask_final = torch.tensor(attention_masks, device=device)

    ########
    # CORE #
    ########

    with instrumentation_stage(instrumentation, "model_stop_grad_setup"):
        # ensure model is on the correct device
        model = model.to(device)

        # core HF model has stop gradient replacement model
        if not disable_stop_grad:
            with suppress(Exception):
                _ = revert_stop_nonlinear_grad(model)
            model = stop_nonlinear_grad(
                model,
                use_relp_grad=use_relp_grad,
                use_half_rule=not disable_half_rule,
                attention_backend=stop_gradient_attention_backend,
            )

    # get attributions (same as original)
    with instrumentation_stage(instrumentation, "initial_attribution"):
        if ig_steps is None:
            (
                mlp_final_attributions,
                embed_final_attributions,
                goal_value,
                mlp_final_acts,
                embed_final_acts,
            ) = _get_grad_attributions_from_logits(
                model,
                input_ids,
                keep_tokens,
                focus_positions,
                focus_logits=focus_logits,
                focus_last_residual=focus_last_residual,
                attention_masks=attn_mask_final,
                ablation_mode=ablation_mode,
                disable_stop_grad=disable_stop_grad,
                center_logits=center_logits,
                objective_weights=objective_weights,
                verbose=verbose,
            )
        else:
            (
                mlp_final_attributions,
                embed_final_attributions,
                goal_value,
                mlp_final_acts,
                embed_final_acts,
            ) = _get_ig_attributions_from_logits(
                model,
                input_ids,
                keep_tokens,
                focus_positions,
                focus_logits=focus_logits,
                focus_last_residual=focus_last_residual,
                attention_masks=attn_mask_final,
                disable_stop_grad=disable_stop_grad,
                center_logits=center_logits,
                objective_weights=objective_weights,
                verbose=verbose,
                ig_steps=ig_steps,
            )

    mlp_final_attributions = mlp_final_attributions.unsqueeze(
        -1
    )  # shape: (L, B, T, D_ff, 1)
    embed_final_attributions = embed_final_attributions.unsqueeze(
        -1
    )  # shape: (B, T, 1)
    if verbose:
        print("collected attributions for mlp", mlp_final_attributions.shape)
        print("collected attributions for embed", embed_final_attributions.shape)

    if apply_blacklist:
        for idx in get_blacklisted_neurons(model):
            layer, neuron = idx.layer, idx.neuron
            mlp_final_attributions[layer, :, :, neuron, :] = 0

    # compute per-batch item absolute attribution thresholds
    absolute_attribution_threshold = None
    if percentage_threshold is not None:
        threshold_goal = (
            goal_value.abs()
            if candidate_axis is not None
            and candidate_axis.use_absolute_goal_for_percentage_threshold
            else goal_value
        )
        absolute_attribution_threshold = threshold_goal * percentage_threshold

    # Before calculating anything, either select important neurons normally or
    # rescore the exact MLP-node union frozen by independent candidate traces.
    with instrumentation_stage(instrumentation, "important_mask_selection"):
        if frozen_topology is None:
            global_important_neurons_mask = _get_global_important_neurons_mask(
                keep_tokens=keep_tokens,
                start_layer=start_layer,
                end_layer=end_layer,
                mlp_final_attributions=mlp_final_attributions,
                node_attribution_threshold=node_attribution_threshold,
                topk_neurons=topk_neurons,
                absolute_attribution_threshold=absolute_attribution_threshold,
                batch_aggregation=batch_aggregation,
                verbose=verbose,
            )
        else:
            global_important_neurons_mask = torch.zeros(
                (
                    mlp_final_attributions.shape[0],
                    mlp_final_attributions.shape[2],
                    mlp_final_attributions.shape[3],
                ),
                device=mlp_final_attributions.device,
                dtype=torch.bool,
            )
            keep_token_set = set(keep_tokens)
            for node in frozen_topology.mlp_nodes:
                if not 0 <= node.layer < model.config.num_hidden_layers:
                    raise ValueError(f"frozen MLP node has invalid layer: {node}")
                if node.token not in keep_token_set:
                    raise ValueError(f"frozen MLP node has invalid token: {node}")
                if not 0 <= node.neuron < mlp_final_attributions.shape[3]:
                    raise ValueError(f"frozen MLP node has invalid neuron: {node}")
                global_important_neurons_mask[node.layer, node.token, node.neuron] = (
                    True
                )
    if verbose:
        print("global important neurons mask", global_important_neurons_mask.shape)
        print("TOTAL NEURONS", global_important_neurons_mask.sum().item())
        print(f"GOAL VALUE {goal_value.item():.5f}")
        print(f"EMBED SUM {embed_final_attributions.sum().item():.5f}")
        for layer in range(len(model.model.layers)):
            print(
                f"LAYER {layer} ATTR {mlp_final_attributions[layer].sum().item():.5f}"
            )
        if percentage_threshold is not None:
            print(f"PERCENTAGE THRESHOLD {percentage_threshold:.2%}")
        if absolute_attribution_threshold is not None:
            print(
                f"ABSOLUTE ATTRIBUTION THRESHOLD {absolute_attribution_threshold.item():.5f}"
            )

    # get important neurons for each layer (same as original)
    neuron_cfg: dict[int, list[list[int]]] = {
        layer: global_important_neurons_mask[layer].nonzero(as_tuple=False).tolist()
        for layer in range(max(start_layer, 0), end_layer)
    }
    attribution_chunk_size = 50 if ig_steps is None else 20
    record_selection_predictors(
        instrumentation,
        neuron_cfg,
        keep_tokens=keep_tokens,
        start_layer=start_layer,
        end_layer=end_layer,
        selected_attribution_chunk_size=attribution_chunk_size,
        use_stop_grad_on_mlps=(not disable_stop_grad and use_stop_grad_on_mlps),
        ig_steps=ig_steps,
    )
    last_non_zero_layer = 0
    for layer, neurons in neuron_cfg.items():
        if verbose:
            print(f"Layer {layer} has {neurons} important neurons")
        if len(neurons) > 0:
            last_non_zero_layer = max(last_non_zero_layer, layer)
    end_layer = min(end_layer, last_non_zero_layer + 1)

    if probe_only:
        return CLJAProbeSelection(
            selected_occurrences=_selected_probe_occurrences(
                mlp_final_attributions, neuron_cfg, keep_tokens
            ),
            effective_start_layer=start_layer,
            effective_end_layer=end_layer,
        )

    # if we only want to return the important neurons, we can do that now
    if return_only_important_neurons:
        model = revert_stop_nonlinear_grad(model)
        return (
            mlp_final_attributions,
            embed_final_attributions,
            mlp_final_acts,
            embed_final_acts,
        )

    # get attributions and contributions for important neurons (same as original)
    if not skip_attr_contrib:
        with instrumentation_stage(
            instrumentation, "selected_attribution_contribution"
        ):
            if ig_steps is None:
                attr, contrib, embed_grad_contrib, neuron_tags = (
                    _get_neuron_attr_and_contrib(
                        model,
                        neuron_cfg,
                        input_ids,
                        src_tokens,
                        contribution_tgt_tokens,
                        focus_positions,
                        focus_logits,
                        attn_mask_final,
                        disable_stop_grad=disable_stop_grad,
                        center_logits=center_logits,
                        neuron_chunk_size=50,
                        verbose=verbose,
                        instrumentation=instrumentation,
                        contribution_target_lane_chunk_size=(
                            selected_neuron_contribution_target_lane_chunk_size
                        ),
                        embed_contribution_target_lane_chunk_size=(
                            selected_embed_contribution_target_lane_chunk_size
                        ),
                    )
                )
            else:
                attr, contrib, embed_grad_contrib, neuron_tags = (
                    _get_neuron_attr_and_contrib_ig(
                        model,
                        neuron_cfg,
                        input_ids,
                        src_tokens,
                        contribution_tgt_tokens,
                        focus_positions,
                        focus_logits,
                        attn_mask_final,
                        disable_stop_grad=disable_stop_grad,
                        center_logits=center_logits,
                        ig_steps=ig_steps,
                        ig_mode=ig_mode,
                        neuron_chunk_size=20,  # Smaller chunk size for IG
                        verbose=verbose,
                        instrumentation=instrumentation,
                        contribution_target_lane_chunk_size=(
                            selected_neuron_contribution_target_lane_chunk_size
                        ),
                        embed_contribution_target_lane_chunk_size=(
                            selected_embed_contribution_target_lane_chunk_size
                        ),
                    )
                )

        if verbose:
            print(
                "collecting attributions for", attr.shape
            )  # shape: (neurons, batch, src)
            print(
                "collecting contributions for", contrib.shape
            )  # shape: (neurons, batch, tgt)
            print(
                "collecting embed contributions for", embed_grad_contrib.shape
            )  # shape: (src, batch, tgt)

    # store neuron attributions and contributions (keep on CPU to save GPU memory)
    neuron_attr_map: dict[NeuronIdx, torch.Tensor] = {}
    neuron_contrib_map: dict[NeuronIdx, torch.Tensor] = {}
    if not skip_attr_contrib:
        for neuron_count, neuron_idx in enumerate(neuron_tags):
            neuron_attr_map[neuron_idx] = attr[neuron_count].cpu()
            neuron_contrib_map[neuron_idx] = contrib[neuron_count].cpu()
        for src_token in src_tokens:
            neuron_contrib_map[NeuronIdx(layer=-1, token=src_token, neuron=0)] = (
                embed_grad_contrib[src_token].cpu()
            )

        del attr, contrib, embed_grad_contrib

    # we repopulate attr and contri maps if stop grad is on to get direct edge weights
    if not disable_stop_grad and use_stop_grad_on_mlps:
        with instrumentation_stage(
            instrumentation, "stop_grad_mlp_attribution_contribution"
        ):
            (
                attr_with_stop_grad_on_mlps,
                contrib_with_stop_grad_on_mlps,
                embed_grad_contrib_with_stop_grad_on_mlps,
                neuron_tags_with_stop_grad_on_mlps,
            ) = _get_neuron_attr_and_contrib_with_stop_grad_on_mlps(
                model,
                neuron_cfg,
                input_ids,
                src_tokens,
                contribution_tgt_tokens,
                focus_positions,
                focus_logits,
                attn_mask_final,
                use_relp_grad=use_relp_grad,
                center_logits=center_logits,
                neuron_chunk_size=10,
                verbose=verbose,
                instrumentation=instrumentation,
                attention_backend=stop_gradient_attention_backend,
                contribution_execution=stop_gradient_contribution_execution,
                contribution_target_lane_chunk_size=(
                    stop_gradient_contribution_target_lane_chunk_size
                ),
                embed_contribution_target_lane_chunk_size=(
                    stop_gradient_embed_contribution_target_lane_chunk_size
                ),
            )
        # store neuron attributions and contributions (keep on CPU to save GPU memory)
        neuron_attr_map_with_stop_grad_on_mlps: dict[NeuronIdx, torch.Tensor] = {}
        neuron_contrib_map_with_stop_grad_on_mlps: dict[NeuronIdx, torch.Tensor] = {}
        for neuron_count, neuron_idx in enumerate(neuron_tags_with_stop_grad_on_mlps):
            neuron_attr_map_with_stop_grad_on_mlps[neuron_idx] = (
                attr_with_stop_grad_on_mlps[neuron_count].cpu()
            )
            neuron_contrib_map_with_stop_grad_on_mlps[neuron_idx] = (
                contrib_with_stop_grad_on_mlps[neuron_count].cpu()
            )
        for src_token in src_tokens:
            neuron_contrib_map_with_stop_grad_on_mlps[
                NeuronIdx(layer=-1, token=src_token, neuron=0)
            ] = embed_grad_contrib_with_stop_grad_on_mlps[src_token].cpu()

        del attr_with_stop_grad_on_mlps, contrib_with_stop_grad_on_mlps
        del embed_grad_contrib_with_stop_grad_on_mlps
    else:
        neuron_attr_map_with_stop_grad_on_mlps = neuron_attr_map
        neuron_contrib_map_with_stop_grad_on_mlps = neuron_contrib_map
        # revert back the replacement model to the original HF model
        if not disable_stop_grad:
            model = revert_stop_nonlinear_grad(model)

    if verbose:
        print(f"Global important neurons mask: {global_important_neurons_mask.sum()}")
        print(
            f"Global important (layer, token): "
            f"{len(global_important_neurons_mask.sum(dim=-1).nonzero())}"
        )

    # cross-layer jacobian edge tracing
    with instrumentation_stage(instrumentation, "graph_expansion"):
        nodes, edges = _get_cl_ja_based_edges(
            model,
            tokenizer,
            cis,
            mlp_final_attributions,
            embed_final_attributions,
            global_important_neurons_mask,
            neuron_attr_map,
            neuron_contrib_map,
            device,
            verbose,
            parent_threshold=parent_threshold,
            edge_threshold=edge_threshold,
            topk=topk,
            keep_tokens=keep_tokens,
            src_tokens=src_tokens,  # the token positions to trace from
            tgt_tokens=contribution_tgt_tokens,
            focus_positions=focus_positions,  # the logits to include
            focus_logits=focus_logits,  # the token vocab ids to trace logits
            start_layer=start_layer,
            end_layer=end_layer,
            attention_masks=attention_masks,
            ig_steps=ig_steps,
            ig_mode=ig_mode,
            return_nodes_only=return_nodes_only,
            return_absolute=return_absolute,
            # stop grad params
            use_relp_grad=use_relp_grad,
            disable_stop_grad=disable_stop_grad,
            use_stop_grad_on_mlps=use_stop_grad_on_mlps,
            stop_gradient_attention_backend=stop_gradient_attention_backend,
            neuron_attr_map_with_stop_grad_on_mlps=neuron_attr_map_with_stop_grad_on_mlps,
            neuron_contrib_map_with_stop_grad_on_mlps=neuron_contrib_map_with_stop_grad_on_mlps,
            candidate_objective_weights=objective_weights,
            frozen_edges=(
                frozen_topology.edges if frozen_topology is not None else None
            ),
            embedding_edge_materialization=embedding_edge_materialization,
            cross_layer_jacobian_execution=cross_layer_jacobian_execution,
            instrumentation=instrumentation,
        )

    # final return
    return nodes, edges


def get_all_pairs_cl_ja_effects_with_attributions(
    model,
    tokenizer: PreTrainedTokenizer,
    cis: list[list[int]],
    config: ADAGConfig,
    src_tokens: list[int],
    tgt_tokens: list[int],
    keep_tokens: list[int] | None = None,
    attention_masks: list[list[int]] | torch.Tensor | None = None,
    focus_positions: list[int] | None = None,
    focus_logits: list[list[int]] | list[int] | None = None,
    candidate_axis: CandidateLogitAxis | None = None,
    frozen_topology: FrozenGraphTopology | None = None,
    instrumentation: TraceInstrumentation | None = None,
    probe_only: bool = False,
) -> (
    tuple[list[Node], list[Edge]]
    | tuple[torch.Tensor, torch.Tensor]
    | CLJAProbeSelection
):
    """Run CLJA with exception-safe stop-gradient state cleanup."""

    if probe_only and config.return_only_important_neurons:
        raise ValueError(
            "probe_only and return_only_important_neurons cannot be enabled together"
        )
    tracing_failed = False
    try:
        return _get_all_pairs_cl_ja_effects_with_attributions_impl(
            model=model,
            tokenizer=tokenizer,
            cis=cis,
            config=config,
            src_tokens=src_tokens,
            tgt_tokens=tgt_tokens,
            keep_tokens=keep_tokens,
            attention_masks=attention_masks,
            focus_positions=focus_positions,
            focus_logits=focus_logits,
            candidate_axis=candidate_axis,
            frozen_topology=frozen_topology,
            instrumentation=instrumentation,
            probe_only=probe_only,
        )
    except BaseException:
        tracing_failed = True
        raise
    finally:
        if not config.disable_stop_grad:
            try:
                if probe_only:
                    revert_stop_nonlinear_grad(model)
                else:
                    revert_active_stop_nonlinear_grad(model)
            except BaseException:
                # Preserve the scientific tracing failure. After a successful
                # trace, cleanup failure remains fatal.
                if not tracing_failed:
                    raise


def _get_cl_ja_based_edges(
    model,
    tokenizer: PreTrainedTokenizer,
    cis: list[list[int]],
    mlp_final_attributions: torch.Tensor,
    embed_final_attributions: torch.Tensor,
    global_important_neurons_mask: torch.Tensor,
    neuron_attr_map,
    neuron_contrib_map,
    device: str = "cuda:0",
    verbose: bool = False,
    parent_threshold: float | None = None,
    edge_threshold: float | None = None,
    topk: int | None = None,
    # we use these lists to trace circuit effects
    keep_tokens: list[int] | None = None,  # the token positions to trace circuits
    src_tokens: list[int] | None = None,  # the token positions to trace from
    tgt_tokens: list[int] | None = None,  # the token positions to trace to
    focus_positions: (
        list[int] | None
    ) = None,  # the token positions to consider the logits effect
    focus_logits: list[int] | None = None,  # the token vocab ids to trace logits
    return_cross_edges_only: bool = True,  # only return cross tgt -> src token edges only
    return_absolute: bool = False,
    # we can skip early/late layers by setting these two
    start_layer: int | None = None,
    end_layer: int | None = None,
    keep_fo: bool = True,
    use_absolute: bool = False,
    batch_aggregation: Literal["mean", "max"] = "mean",
    include_error_node: bool = False,
    include_error_edges: bool = False,
    return_nodes_only: bool = False,
    max_buffers: int | None = None,
    return_all_edge_weights: bool = True,
    attention_masks: list[list[int]] | None = None,
    # stop grad params
    use_relp_grad: bool = False,
    disable_stop_grad: bool = False,
    use_stop_grad_on_mlps: bool = False,
    stop_gradient_attention_backend: StopGradientAttentionBackend = (
        DEFAULT_STOP_GRADIENT_ATTENTION_BACKEND
    ),
    neuron_attr_map_with_stop_grad_on_mlps=None,
    neuron_contrib_map_with_stop_grad_on_mlps=None,
    candidate_objective_weights: tuple[float, ...] | None = None,
    frozen_edges: frozenset[tuple[NeuronIdx, NeuronIdx]] | None = None,
    embedding_edge_materialization: EmbeddingEdgeMaterialization = (
        DEFAULT_EMBEDDING_EDGE_MATERIALIZATION
    ),
    cross_layer_jacobian_execution: CrossLayerJacobianExecution = (
        DEFAULT_CROSS_LAYER_JACOBIAN_EXECUTION
    ),
    # IG params
    ig_steps: int | None = None,
    ig_mode: Literal["ig-inputs", "conductance"] = "ig-inputs",
    instrumentation: TraceInstrumentation | None = None,
) -> tuple[Any, Any]:
    """
    Compute circuit nodes and edges using Cross Layer Jacobian Attribution (CLJA).

    This function computes edge weights between important neurons using jacobian computation
    with stop gradient on nonlinear components, as an alternative to CLSO lens methods.
    """
    nodes: list[Node] = []
    edges: list[Edge] = []

    def retain_edge(
        source: NeuronIdx,
        target: NeuronIdx,
        weight: torch.Tensor,
    ) -> bool:
        if frozen_edges is not None:
            return (source, target) in frozen_edges
        if edge_threshold is not None and weight.abs().max() < edge_threshold:
            return False
        return not (
            parent_threshold is not None and weight.abs().max() < parent_threshold
        )

    # caching activations
    if verbose:
        print("Collecting acts... ", end="", flush=True)
    collect_layers = list(range(model.config.num_hidden_layers))
    with instrumentation_stage(instrumentation, "activation_collection"):
        (neurons_LBTI, resids_LBTD, tokens, output_norm_const_BTf11D) = (
            collect_neuron_acts(
                model,
                tokenizer,
                cis,
                attention_masks,
                collect_layers=collect_layers,
                keep_tokens=keep_tokens,
                device=device,
                verbose=verbose,
            )
        )

    # key constants
    L = model.config.num_hidden_layers
    D = model.config.hidden_size
    B = neurons_LBTI[0].size(0)
    neurons_LBTI[0].size(1)
    objective_weight_tensor: torch.Tensor | None = None
    if candidate_objective_weights is not None:
        if len(candidate_objective_weights) != len(focus_positions):
            raise ValueError(
                "candidate objective weight width must match candidate count"
            )
        objective_weight_tensor = torch.tensor(
            candidate_objective_weights,
            device=device,
            dtype=neurons_LBTI[0].dtype,
        )

    def joint_target_attribution(
        target_attribution: torch.Tensor,
    ) -> torch.Tensor:
        """Reduce a raw candidate contribution vector to the joint objective."""

        if objective_weight_tensor is None:
            return target_attribution
        return reduce_candidate_contributions(
            target_attribution,
            objective_weight_tensor,
        )

    # creating final logits nodes
    logit_nodes_before = len(nodes)
    logit_edges_before = len(edges)
    logit_graph_measurement = (
        instrumentation.measurement_start("logit_graph_materialization")
        if instrumentation is not None
        else None
    )
    if verbose:
        print("Creating final logits nodes and all incoming edges...")
    for target_idx, target_token_pos_idx in enumerate(focus_positions):
        focus_logits_target = torch.tensor(focus_logits)[:, target_idx]
        resid_final_BD = resids_LBTD[L - 1][:, target_token_pos_idx].view(
            B, D
        ) * output_norm_const_BTf11D[:, target_token_pos_idx, 0].view(B, D)
        logits_BV = torch.einsum(
            "bd,vd->bv", resid_final_BD, model.lm_head.weight[focus_logits_target]
        ) * torch.eye(B, device=device)

        # add attr and contrib map
        for batch_idx in range(len(focus_logits_target)):
            logit_id = (
                L,
                target_token_pos_idx,
                focus_logits_target[batch_idx].item(),
            )
            if logit_id not in neuron_attr_map:
                neuron_attr_map[logit_id] = torch.zeros(
                    (B, len(src_tokens)), device=device
                )
            if logit_id not in neuron_contrib_map:
                neuron_contrib_map[logit_id] = torch.zeros(
                    (B, len(tgt_tokens)), device=device
                )

            # contrib = logit of this tgt token
            neuron_contrib_map[logit_id][batch_idx, target_idx] += logits_BV[
                batch_idx, batch_idx
            ]

            # attr = sum of contribs from src tokens
            for src_token in src_tokens:
                src_id = (-1, src_token, 0)
                contrib_from_src = neuron_contrib_map.get(
                    src_id, torch.zeros((B, len(tgt_tokens)))
                )[batch_idx, target_idx]
                neuron_attr_map[logit_id][batch_idx, src_token] += contrib_from_src

        # to add logit nodes
        neuron_indices_map = {
            i: focus_logits_target[i].item() for i in range(len(focus_logits_target))
        }
        for idx, neuron_idx in neuron_indices_map.items():
            final_attribution = torch.stack(
                [torch.diagflat(logits_BV[b]) for b in range(logits_BV.shape[0])]
            )[:, idx, :]
            if objective_weight_tensor is not None:
                final_attribution = (
                    final_attribution * objective_weight_tensor[target_idx]
                )
            nodes.append(
                Node(
                    layer=L,
                    token=target_token_pos_idx,
                    neuron=neuron_idx,
                    activation=logits_BV[:, idx].float().cpu(),
                    final_attribution=final_attribution.float().cpu(),
                    attr_map=neuron_attr_map.get(
                        (L, target_token_pos_idx, neuron_idx), None
                    ),
                    contrib_map=neuron_contrib_map.get(
                        (L, target_token_pos_idx, neuron_idx), None
                    ),
                )
            )

            if return_nodes_only:
                continue

            # creating edges pointing to the logit node (this uses stop grad if specified)
            for (
                source_key,
                source_contrib,
            ) in neuron_contrib_map_with_stop_grad_on_mlps.items():
                # skip incoming from last layer
                if return_nodes_only or source_key[0] == L:
                    continue
                target_key = NeuronIdx(
                    layer=L, token=target_token_pos_idx, neuron=neuron_idx
                )

                # Move source_contrib to device if needed
                if source_contrib.device.type == "cpu":
                    source_contrib = source_contrib.to(device)

                edge_weight = torch.zeros(len(tokens), device=device)
                edge_weight[idx] = source_contrib[idx, target_idx]

                eps = logits_BV[idx, idx].abs().mean() * 1e-6
                edge_weight = edge_weight / (logits_BV[idx, idx] + eps)

                source_key = NeuronIdx(
                    layer=source_key[0],
                    token=source_key[1],
                    neuron=(
                        tokens[idx][source_key[1]]
                        if source_key[0] == -1
                        else source_key[2]
                    ),
                )

                if not retain_edge(source_key, target_key, edge_weight):
                    continue

                edges.append(
                    Edge(
                        src=source_key,
                        tgt=target_key,
                        weight=edge_weight.detach().float().cpu(),
                        final_attribution=(edge_weight[:, None] * final_attribution)
                        .detach()
                        .float()
                        .cpu(),
                    )
                )

    if instrumentation is not None and logit_graph_measurement is not None:
        instrumentation.measurement_finish(logit_graph_measurement)
        instrumentation.set_counter("logit_node_delta", len(nodes) - logit_nodes_before)
        instrumentation.set_counter("logit_edge_delta", len(edges) - logit_edges_before)

    # creating MLP neuron nodes
    mlp_nodes_before = len(nodes)
    mlp_edges_before = len(edges)
    mlp_nodes_measurement = (
        instrumentation.measurement_start("mlp_node_materialization")
        if instrumentation is not None
        else None
    )
    for layer in range(max(start_layer, 0), end_layer):
        important_positions = global_important_neurons_mask[layer].nonzero(
            as_tuple=False
        )
        for pos_neuron in important_positions:
            token_pos, neuron_idx = pos_neuron.tolist()
            if token_pos in keep_tokens:
                neuron_key = NeuronIdx(layer=layer, token=token_pos, neuron=neuron_idx)
                activation = neurons_LBTI[layer][
                    :, token_pos, neuron_idx
                ]  # shape: (batch,)
                # get attribution scores (already on CPU from earlier)
                attr_map = neuron_attr_map.get(neuron_key, None)
                contrib_map = neuron_contrib_map.get(neuron_key, None)
                final_attribution = mlp_final_attributions[
                    layer, :, token_pos, neuron_idx, :
                ]
                nodes.append(
                    Node(
                        layer=layer,
                        token=token_pos,
                        neuron=neuron_idx,
                        activation=activation.float().cpu(),
                        final_attribution=final_attribution.float().cpu(),
                        attr_map=attr_map if attr_map is not None else None,
                        contrib_map=contrib_map if contrib_map is not None else None,
                    )
                )

    if instrumentation is not None and mlp_nodes_measurement is not None:
        instrumentation.measurement_finish(mlp_nodes_measurement)
        instrumentation.set_counter("mlp_node_delta", len(nodes) - mlp_nodes_before)
        instrumentation.set_counter("mlp_edge_delta", len(edges) - mlp_edges_before)

    # Clean up after creating MLP nodes
    del mlp_final_attributions
    torch.cuda.empty_cache()

    # creating embedding nodes and ordered edge sources
    embedding_nodes_before = len(nodes)
    embedding_edges_before = len(edges)
    embedding_graph_measurement = (
        instrumentation.measurement_start("embedding_graph_materialization")
        if instrumentation is not None
        else None
    )
    if verbose:
        print("Creating embedding nodes and all outgoing edges...")
    embedding_sources: list[EmbeddingSource] = []
    for src_token in src_tokens:
        final_attributions = embed_final_attributions[
            :, src_token, :
        ]  # (batch, logits)
        for token_type in {tokens[t][src_token] for t in range(len(tokens))}:
            relevant_idxs = [
                batch_idx
                for batch_idx in range(len(tokens))
                if tokens[batch_idx][src_token] == token_type
            ]
            mask = torch.zeros(len(tokens), device=device)
            mask[relevant_idxs] = 1
            mask = mask.to(torch.bool)
            final_attribution = torch.where(mask[:, None], final_attributions, 0)
            attr_map = torch.zeros(
                (len(tokens), min(len(keep_tokens), len(tokens[0]))), device=device
            )
            attr_map[relevant_idxs, src_token] = 1
            contrib_map = neuron_contrib_map.get((-1, src_token, 0), None)
            activations = torch.ones(len(tokens), 1, device=device)

            # Move contrib_map to device if needed for computation, then back to CPU
            if contrib_map is not None:
                if contrib_map.device.type == "cpu":
                    contrib_map = contrib_map.to(device)
                contrib_map_final = (contrib_map * mask[:, None]).cpu()
            else:
                contrib_map_final = None

            nodes.append(
                Node(
                    layer=-1,
                    token=src_token,
                    neuron=token_type,
                    activation=torch.where(mask, activations[:, 0], 0).float().cpu(),
                    final_attribution=final_attribution.float().cpu(),
                    attr_map=attr_map.cpu(),
                    contrib_map=contrib_map_final,
                )
            )
            embedding_sources.append(
                EmbeddingSource(
                    key=NeuronIdx(layer=-1, token=src_token, neuron=token_type),
                    batch_mask=mask,
                )
            )

    embedding_targets = (
        []
        if return_nodes_only
        else [
            EmbeddingTarget(
                key=NeuronIdx(layer=key[0], token=key[1], neuron=key[2]),
                attribution_by_source=target_attr,
                activation=neurons_LBTI[key[0]][:, key[1], key[2]],
                final_attribution=neuron_contrib_map.get(key, None),
            )
            for key, target_attr in neuron_attr_map_with_stop_grad_on_mlps.items()
            if key[0] != -1 and key[0] != L
        ]
    )
    embedding_edges = materialize_embedding_edges(
        embedding_edge_materialization,
        EmbeddingEdgeMaterializationRequest(
            sources=embedding_sources,
            targets=embedding_targets,
            device=device,
            edge_threshold=edge_threshold,
            parent_threshold=parent_threshold,
            objective_weights=objective_weight_tensor,
            frozen_edges=frozen_edges,
            return_nodes_only=return_nodes_only,
        ),
    )
    edges.extend(embedding_edges)

    if instrumentation is not None and embedding_graph_measurement is not None:
        instrumentation.measurement_finish(embedding_graph_measurement)
        instrumentation.set_counter(
            "embedding_node_delta", len(nodes) - embedding_nodes_before
        )
        instrumentation.set_counter(
            "embedding_edge_delta", len(edges) - embedding_edges_before
        )
        instrumentation.set_counter(
            "embedding_candidate_edge_count",
            len(embedding_sources) * len(embedding_targets),
        )
        instrumentation.set_counter(
            "embedding_retained_edge_count", len(embedding_edges)
        )

    # Clean up activation tensors after creating all embedding edges
    del neurons_LBTI, resids_LBTD, output_norm_const_BTf11D
    torch.cuda.empty_cache()

    if return_nodes_only:
        return nodes, edges

    # compute edges with raw stop grad HF models
    if verbose:
        print("Computing CLJA-based edges...")

    # get input ids
    input_ids = torch.tensor(cis, device=device)
    if isinstance(attention_masks, torch.Tensor):
        attn_mask_final = attention_masks.to(device)
    else:
        attn_mask_final = torch.tensor(attention_masks, device=device)

    # for any layer pair, we compute the jacobian-based edge weights
    cross_layer_nodes_before = len(nodes)
    cross_layer_edges_before = len(edges)
    cross_layer_measurement = (
        instrumentation.measurement_start("cross_layer_graph_expansion")
        if instrumentation is not None
        else None
    )
    if instrumentation is not None:
        for counter_name in (
            "cross_layer_pair_count",
            "cross_layer_candidate_edge_count",
            "cross_layer_target_chunks_per_pass",
            "cross_layer_target_chunk_executions",
            "cross_layer_retained_edge_count",
        ):
            instrumentation.set_counter(counter_name, 0)
    active_cross_layer_layers = tuple(
        layer
        for layer in range(start_layer + 1, end_layer)
        if global_important_neurons_mask[layer].any()
    )
    if cross_layer_jacobian_execution == "cached_range_v1" and ig_steps is not None:
        raise ValueError(
            "cached_range_v1 does not support integrated-gradients cross-layer tracing"
        )
    jacobian_executor = None
    if ig_steps is None:
        jacobian_executor = prepare_cross_layer_jacobian_execution(
            CrossLayerJacobianPreparation(
                model=model,
                input_ids=input_ids,
                attention_mask=attn_mask_final,
                source_layers=active_cross_layer_layers[:-1],
                use_relp_grad=use_relp_grad,
                disable_stop_grad=disable_stop_grad,
                use_stop_grad_on_mlps=use_stop_grad_on_mlps,
                device=device,
                attention_backend=stop_gradient_attention_backend,
                instrumentation=instrumentation,
            ),
            execution=cross_layer_jacobian_execution,
        )
    elif instrumentation is not None:
        # The legacy IG path performs its own repeated full-model work. These
        # non-IG execution counters are explicitly unavailable, never false zeroes.
        for counter_name in (
            "cross_layer_preparation_forward_count",
            "cross_layer_preparation_cache_bytes",
            "cross_layer_full_decoder_layer_executions",
            "cross_layer_replay_decoder_layer_entries",
            "cross_layer_vjp_chunk_executions",
        ):
            instrumentation.set_counter(counter_name, None)
    for tgt_layer in range(end_layer - 1, start_layer + 1, -1):
        # if there is no important neurons in the target layer, skip
        if not global_important_neurons_mask[tgt_layer].any():
            continue
        # layers before the target layer only
        for src_layer in range(tgt_layer - 1, start_layer, -1):
            # Skip if no important neurons in source layer (or embeddings)
            if not global_important_neurons_mask[src_layer].any():
                continue

            frozen_pair_edges = None
            if frozen_edges is not None:
                frozen_pair_edges = {
                    edge
                    for edge in frozen_edges
                    if edge[0].layer == src_layer and edge[1].layer == tgt_layer
                }
                if not frozen_pair_edges:
                    continue

            # get the fixed neuron lists
            src_positions = global_important_neurons_mask[src_layer].nonzero(
                as_tuple=False
            )
            src_neuron_list = [
                (pos[0].item(), pos[1].item())
                for pos in src_positions
                if pos[0].item() in keep_tokens
                and (
                    frozen_pair_edges is None
                    or NeuronIdx(src_layer, pos[0].item(), pos[1].item())
                    in {edge[0] for edge in frozen_pair_edges}
                )
            ]
            tgt_positions = global_important_neurons_mask[tgt_layer].nonzero(
                as_tuple=False
            )
            tgt_neuron_list = [
                (pos[0].item(), pos[1].item())
                for pos in tgt_positions
                if pos[0].item() in keep_tokens
                and (
                    frozen_pair_edges is None
                    or NeuronIdx(tgt_layer, pos[0].item(), pos[1].item())
                    in {edge[1] for edge in frozen_pair_edges}
                )
            ]

            if verbose:
                print(f"Compute edge weights {tgt_layer} -> {src_layer}")

            # for other layer pairs, calculate the jacobian
            target_chunk_size = 50 if ig_steps is None else 20
            candidate_edges = len(src_neuron_list) * len(tgt_neuron_list)
            target_chunks = (
                math.ceil(len(tgt_neuron_list) / target_chunk_size)
                if tgt_neuron_list
                else 0
            )
            jacobian_pass_count = ig_steps + 1 if ig_steps is not None else 1
            jacobian_measurement = (
                instrumentation.measurement_start("layer_pair_jacobian")
                if instrumentation is not None
                else None
            )
            exact_receipts = None
            if ig_steps is None:
                if jacobian_executor is None:
                    raise RuntimeError("non-IG Jacobian executor was not prepared")
                pair_result = jacobian_executor.compute_pair(
                    CrossLayerJacobianPair(
                        src_layer=src_layer,
                        tgt_layer=tgt_layer,
                        src_neurons=tuple(src_neuron_list),
                        tgt_neurons=tuple(tgt_neuron_list),
                        tgt_chunk_size=50,
                    )
                )
                if not isinstance(pair_result, CrossLayerJacobianPairResult):
                    raise TypeError("non-IG Jacobian execution returned IG components")
                relative_attribution = pair_result.relative_attribution
                exact_receipts = pair_result.receipts.ordered()
            else:
                relative_attribution = _compute_cl_ja_layer_jacobian_ig(
                    model,
                    input_ids,
                    attn_mask_final,
                    src_layer,
                    tgt_layer,
                    src_neuron_list,
                    tgt_neuron_list,
                    keep_tokens,
                    src_tokens if src_layer == -1 else None,
                    device,
                    ig_steps=ig_steps,
                    ig_mode=ig_mode,
                    tgt_chunk_size=20,  # Smaller chunk size for IG to save memory
                    verbose=verbose,
                )  # shape: (batch, n_src, n_tgt)
            jacobian_shape = [int(size) for size in relative_attribution.shape]
            if instrumentation is not None and jacobian_measurement is not None:
                instrumentation.measurement_finish(jacobian_measurement)
            jacobian_seconds = (
                jacobian_measurement.wall_seconds
                if jacobian_measurement is not None
                else None
            )
            jacobian_cuda_memory = (
                jacobian_measurement.cuda_memory
                if jacobian_measurement is not None
                else None
            )

            # adding edges from every src neuron to every tgt neuron
            pair_edges_before = len(edges)
            materialization_measurement = (
                instrumentation.measurement_start("layer_pair_materialization")
                if instrumentation is not None
                else None
            )
            for i, (src_token, src_neuron) in enumerate(src_neuron_list):
                for j, (tgt_token, tgt_neuron) in enumerate(tgt_neuron_list):
                    edge_weight = relative_attribution[:, i, j]  # shape: (batch,)

                    target_attribution = neuron_contrib_map.get(
                        (tgt_layer, tgt_token, tgt_neuron), None
                    )  # shape: (B, logits)

                    # Move target_attribution to device if needed
                    if (
                        target_attribution is not None
                        and target_attribution.device.type == "cpu"
                    ):
                        target_attribution = target_attribution.to(device)
                    if target_attribution is not None:
                        target_attribution = joint_target_attribution(
                            target_attribution
                        )

                    src_key = NeuronIdx(
                        layer=src_layer, token=src_token, neuron=src_neuron
                    )
                    tgt_key = NeuronIdx(
                        layer=tgt_layer, token=tgt_token, neuron=tgt_neuron
                    )
                    if not retain_edge(src_key, tgt_key, edge_weight):
                        continue

                    edges.append(
                        Edge(
                            src=src_key,
                            tgt=tgt_key,
                            weight=edge_weight.detach().float().cpu(),
                            final_attribution=(
                                edge_weight[:, None] * target_attribution
                            )
                            .detach()
                            .float()
                            .cpu(),
                        )
                    )

            if instrumentation is not None and materialization_measurement is not None:
                instrumentation.measurement_finish(materialization_measurement)
            materialization_seconds = (
                materialization_measurement.wall_seconds
                if materialization_measurement is not None
                else None
            )
            retained_edges = len(edges) - pair_edges_before
            if instrumentation is not None:
                pair_telemetry = (
                    {
                        "jacobian_shape": jacobian_shape,
                        "jacobian_cuda_memory": jacobian_cuda_memory,
                    }
                    if instrumentation.cuda_memory_telemetry
                    else {}
                )
                instrumentation.record_layer_pair(
                    src_layer,
                    tgt_layer,
                    src_count=len(src_neuron_list),
                    tgt_count=len(tgt_neuron_list),
                    candidate_edges=candidate_edges,
                    target_chunk_size=target_chunk_size,
                    target_chunks_per_pass=target_chunks,
                    jacobian_pass_count=jacobian_pass_count,
                    target_chunk_executions=target_chunks * jacobian_pass_count,
                    jacobian_seconds=jacobian_seconds,
                    materialization_seconds=materialization_seconds,
                    retained_edges=retained_edges,
                    exact_receipts=exact_receipts,
                    **pair_telemetry,
                )
            del relative_attribution

    if instrumentation is not None and cross_layer_measurement is not None:
        instrumentation.measurement_finish(cross_layer_measurement)
        instrumentation.set_counter(
            "cross_layer_node_delta", len(nodes) - cross_layer_nodes_before
        )
        instrumentation.set_counter(
            "cross_layer_edge_delta", len(edges) - cross_layer_edges_before
        )

    if verbose:
        print(f"# found nodes: {len(nodes)}")
        print(f"# found edges: {len(edges)}")

    return nodes, edges


def _compute_cl_ja_layer_jacobian(
    model,
    input_ids: torch.Tensor,
    attention_masks: torch.Tensor,
    src_layer: int,
    tgt_layer: int,
    src_neuron_list,
    tgt_neuron_list,
    keep_tokens: list[int],
    src_tokens: list[int] | None,
    use_relp_grad: bool,
    disable_stop_grad: bool,
    use_stop_grad_on_mlps: bool,
    device: str,
    attention_backend: StopGradientAttentionBackend = (
        DEFAULT_STOP_GRADIENT_ATTENTION_BACKEND
    ),
    alpha: float | None = None,
    tgt_chunk_size: int = 50,
    verbose: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute CLJA edge weights between source and target layers using batched jacobian.

    Args:
        alpha: Optional scaling factor for integrated gradients. When set, returns
               (gradients, src_acts, tgt_acts) instead of final attributions.

    Returns:
        If alpha is None: relative_attribution tensor of shape (batch, n_src, n_tgt)
        If alpha is not None: tuple of (gradients, src_acts, tgt_acts)
    """

    for layer_idx in range(len(model.model.layers)):
        remove_forward_hooks(model.model.layers[layer_idx].mlp.down_proj)

    # stop grads
    if not disable_stop_grad:
        model = layerwise_stop_nonlinear_grad(
            model,
            src_layer,
            tgt_layer,
            use_relp_grad=use_relp_grad,
            use_stop_grad_on_mlps=use_stop_grad_on_mlps,
            attention_backend=attention_backend,
        )
    model.zero_grad()
    # populate activation cache for getting jacobian
    embeds = model.model.embed_tokens(input_ids).detach().requires_grad_()
    if alpha is not None:
        embeds = embeds * alpha
    activation_cache = {}

    def make_hook(layer_idx):
        def hook(module, input, output):
            activation_cache[layer_idx] = input[0]  # shape: (batch, seq, hidden)

        return hook

    handles = []
    if not disable_stop_grad:
        src_handle = model.model.layers[
            src_layer
        ].mlp.mlp.down_proj.register_forward_hook(make_hook(src_layer))
        tgt_handle = model.model.layers[
            tgt_layer
        ].mlp.mlp.down_proj.register_forward_hook(make_hook(tgt_layer))
    else:
        src_handle = model.model.layers[src_layer].mlp.down_proj.register_forward_hook(
            make_hook(src_layer)
        )
        tgt_handle = model.model.layers[tgt_layer].mlp.down_proj.register_forward_hook(
            make_hook(tgt_layer)
        )
    handles.append(src_handle)
    handles.append(tgt_handle)
    _ = model(inputs_embeds=embeds, attention_mask=attention_masks)

    # get batch size
    batch = input_ids.shape[0]

    # Collect source activations for all neurons (needed later)
    src_acts_full = activation_cache[src_layer]  # (batch, seq, hidden)
    src_acts_list = []
    for src_pos, src_neuron in src_neuron_list:
        src_acts_list.append(src_acts_full[:, src_pos, src_neuron])  # (batch,)
    src_acts = torch.stack(src_acts_list, dim=-1)  # (batch, n_src)

    # Process target neurons in chunks to avoid OOM
    n_tgt = len(tgt_neuron_list)
    tgt_acts_full = activation_cache[tgt_layer]  # (batch, seq, hidden)

    # Collect all target activations
    tgt_acts_list = []
    for token, neuron in tgt_neuron_list:
        tgt_acts_list.append(tgt_acts_full[:, token, neuron])  # (batch,)
    tgt_activations = torch.stack(tgt_acts_list, dim=0)  # (n_tgt, batch)

    # Process in chunks
    src_ja_chunks = []
    for chunk_start in range(0, n_tgt, tgt_chunk_size):
        chunk_end = min(chunk_start + tgt_chunk_size, n_tgt)
        tgt_chunk = tgt_activations[chunk_start:chunk_end]  # (n_chunk, batch)
        n_chunk = tgt_chunk.shape[0]
        is_last_chunk = chunk_end >= n_tgt

        # prepare batched grads for this chunk
        grad_outputs = torch.eye(
            batch * n_chunk, device=device
        )  # (n_chunk * batch, n_chunk * batch)

        src_acts_full.grad = None
        chunk_src_ja = torch.autograd.grad(
            tgt_chunk.flatten(),  # (n_chunk * batch)
            src_acts_full,  # (batch, seq, d)
            grad_outputs=grad_outputs,  # (n_chunk * batch, n_chunk * batch)
            is_grads_batched=True,
            retain_graph=not is_last_chunk,
        )[0]
        # shape: (n_chunk * batch, batch, seq, d)
        # convert back to (n_chunk, batch, batch, seq, d)
        chunk_src_ja = chunk_src_ja.reshape(
            n_chunk, batch, batch, chunk_src_ja.shape[-2], chunk_src_ja.shape[-1]
        )
        # get only identity along batch, batch
        chunk_src_ja = chunk_src_ja.diagonal(dim1=1, dim2=2).permute(0, 3, 1, 2)
        # now (n_chunk, batch, seq, d)

        src_ja_chunks.append(chunk_src_ja)
        del grad_outputs

    # Concatenate all chunks
    src_ja = torch.cat(src_ja_chunks, dim=0)  # (n_tgt, batch, seq, d)
    del src_ja_chunks

    if alpha is not None:
        # For IG: return gradients only, multiply by activations later
        grads = []
        for src_pos, src_neuron in src_neuron_list:
            grads.append(src_ja[:, :, src_pos, src_neuron])  # (n_tgt, batch)
        grads = torch.stack(grads, dim=-1)  # (n_tgt, batch, n_src)
        grads = grads.permute(1, 2, 0).contiguous()  # (batch, n_src, n_tgt)

        del src_ja
        tgt_acts = tgt_activations.detach().permute(1, 0)  # (batch, n_tgt)

        # remove all hooks
        for handle in handles:
            handle.remove()

        # revert stop grads
        if not disable_stop_grad:
            model = layerwise_revert_stop_nonlinear_grad(
                model,
                src_layer,
                tgt_layer,
            )

        return (
            grads,
            src_acts,
            tgt_acts,
        )  # (batch, n_src, n_tgt), (batch, n_src), (batch, n_tgt)

    # For regular attribution: gradient * activation
    jvps = []
    for src_pos, src_neuron in src_neuron_list:
        jvps.append(
            src_ja[:, :, src_pos, src_neuron]
            * src_acts_full[:, src_pos, src_neuron][None, :].detach()
        )  # (n_tgt, batch) * (1, batch) -> (n_tgt, batch)
    jvps = torch.stack(jvps, dim=-1)  # (n_tgt, batch, n_src)
    jvps = jvps.permute(1, 2, 0).contiguous()  # (batch, n_src, n_tgt)
    del src_ja

    # tgt_activations = (n_tgt, batch)
    tgt_activations = tgt_activations.detach().permute(1, 0)  # (batch, n_tgt)
    # Use adaptive epsilon for numerical stability (matching CLSO approach)
    eps = tgt_activations.abs().mean() * 1e-6
    relative_attribution = jvps / (
        tgt_activations[:, None, :] + eps
    )  # (batch, n_src, n_tgt) / (batch, 1, n_tgt)
    del jvps

    # remove all hooks
    for handle in handles:
        handle.remove()

    # revert stop grads
    if not disable_stop_grad:
        model = layerwise_revert_stop_nonlinear_grad(
            model,
            src_layer,
            tgt_layer,
        )

    return relative_attribution  # shape: (batch, n_src, n_tgt)


def _compute_cl_ja_layer_jacobian_ig(
    model,
    input_ids: torch.Tensor,
    attention_masks: torch.Tensor,
    src_layer: int,
    tgt_layer: int,
    src_neuron_list,
    tgt_neuron_list,
    keep_tokens: list[int],
    src_tokens: list[int] | None,
    device: str,
    ig_steps: int = 10,
    ig_mode: Literal["ig-inputs", "conductance"] = "ig-inputs",
    tgt_chunk_size: int = 20,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Compute CLJA edge weights using Integrated Gradients.

    Note: For IG, stop gradients are disabled (disable_stop_grad=True).

    Args:
        ig_steps: Number of steps for integrated gradients.
        ig_mode: Mode for integrated gradients aggregation ("ig-inputs" or "conductance").

    Returns:
        relative_attribution tensor of shape (batch, n_src, n_tgt)
    """
    # Collect step-wise gradients and activations
    grads_steps = []
    src_acts_steps = []
    tgt_acts_steps = []

    for step in range(ig_steps + 1):
        alpha = step / ig_steps

        grads, src_acts, tgt_acts = _compute_cl_ja_layer_jacobian(
            model=model,
            input_ids=input_ids,
            attention_masks=attention_masks,
            src_layer=src_layer,
            tgt_layer=tgt_layer,
            src_neuron_list=src_neuron_list,
            tgt_neuron_list=tgt_neuron_list,
            keep_tokens=keep_tokens,
            src_tokens=src_tokens,
            use_relp_grad=False,  # Not used for IG
            disable_stop_grad=True,  # Always disable stop grad for IG
            use_stop_grad_on_mlps=False,  # Not used for IG
            device=device,
            alpha=alpha,
            tgt_chunk_size=tgt_chunk_size,
            verbose=verbose,
        )

        # Immediately detach and move to CPU to save GPU memory
        grads_steps.append(grads.detach().cpu())
        src_acts_steps.append(src_acts.detach().cpu())
        tgt_acts_steps.append(tgt_acts.detach().cpu())

        del grads, src_acts, tgt_acts

    if ig_mode == "ig-inputs":
        # Riemann sum in IG (ignore step 0)
        # Average gradients across steps (move back to device for computation)
        grads_avg = (
            torch.stack(grads_steps[1:]).mean(dim=0).to(device)
        )  # (batch, n_src, n_tgt)

        # Compute activation differences
        src_acts_diff = (src_acts_steps[-1] - src_acts_steps[0]).to(
            device
        )  # (batch, n_src)
        tgt_acts_final = src_acts_steps[-1].to(device)  # (batch, n_tgt)

        # Apply IG formula: averaged_gradient * src_activation_diff
        # Shape: (batch, n_src, n_tgt) * (batch, n_src, 1) -> (batch, n_src, n_tgt)
        jvps = grads_avg * src_acts_diff[:, :, None]

        # Normalize by target activations (use final activations)
        tgt_acts_final = tgt_acts_steps[-1].to(device)  # (batch, n_tgt)
        eps = tgt_acts_final.abs().mean() * 1e-6
        relative_attribution = jvps / (tgt_acts_final[:, None, :] + eps)

    elif ig_mode == "conductance":
        # Stack all steps (excluding step 0) and move to device
        grads_all = torch.stack(grads_steps[1:]).to(
            device
        )  # (steps, batch, n_src, n_tgt)

        # Compute step-wise differences in activations
        src_acts_all = torch.stack(src_acts_steps).to(device)  # (steps+1, batch, n_src)

        src_acts_diffs = torch.diff(src_acts_all, dim=0)  # (steps, batch, n_src)

        # Apply conductance: sum over steps of (gradient * src_activation_diff)
        # Shape: (steps, batch, n_src, n_tgt) * (steps, batch, n_src, 1) -> sum over steps
        jvps = (grads_all * src_acts_diffs[:, :, :, None]).sum(
            dim=0
        )  # (batch, n_src, n_tgt)

        # Normalize by target activations (use final activations)
        tgt_acts_final = tgt_acts_steps[-1].to(device)  # (batch, n_tgt)
        eps = tgt_acts_final.abs().mean() * 1e-6
        relative_attribution = jvps / (tgt_acts_final[:, None, :] + eps)

    else:
        raise ValueError(f"Invalid IG mode: {ig_mode}")

    return relative_attribution  # shape: (batch, n_src, n_tgt)
