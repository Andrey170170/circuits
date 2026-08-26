"""
Core functionality for computing gradient-based attributions in language models, after applying
modifications to the backward pass in grad.py.
"""

from collections.abc import Iterator, Sequence
from typing import Literal

import torch
from tqdm import tqdm

from circuits.tracing.contribution_execution import (
    DEFAULT_STOP_GRADIENT_CONTRIBUTION_EXECUTION,
    SelectedNeuronContributionSource,
    StopGradientContributionExecution,
    run_selected_embed_contribution_vjp,
    run_selected_neuron_contribution_vjps,
    run_stop_gradient_contribution_forward,
    run_stop_gradient_contribution_vjp,
    run_stop_gradient_embed_contribution_vjp,
)
from circuits.tracing.grad import (
    DEFAULT_STOP_GRADIENT_ATTENTION_BACKEND,
    StopGradientAttentionBackend,
    layerwise_revert_stop_nonlinear_grad,
    layerwise_stop_nonlinear_grad,
    remove_forward_hooks,
    revert_stop_nonlinear_grad,
)
from circuits.tracing.important_neuron_selection import (
    has_any_strictly_positive_value,
)
from circuits.tracing.instrumentation import (
    TraceInstrumentation,
    cuda_memory_instrumentation_stage,
    cuda_memory_observation_stage,
)
from circuits.tracing.utils import NeuronIdx

DEFAULT_SELECTED_ATTRIBUTION_NEURON_LANE_CHUNK_SIZE = 50


def resolve_selected_attribution_neuron_lane_chunk_size(
    chunk_size: int | None,
) -> int:
    """Resolve the ordinary selected-attribution VJP neuron-lane width.

    ``None`` preserves the historical 50-neuron execution.  The resolved
    width controls only the batched VJP over selected neuron activations; it
    does not affect stop-gradient attribution or contribution target lanes.
    """

    if chunk_size is None:
        return DEFAULT_SELECTED_ATTRIBUTION_NEURON_LANE_CHUNK_SIZE
    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or chunk_size <= 0
    ):
        raise ValueError(
            "selected-attribution neuron-lane chunk size must be a "
            "positive integer or None"
        )
    return chunk_size


def _nonempty_neuron_layers(
    neuron_cfg: dict[int, list[list[int]]],
) -> Iterator[tuple[int, list[list[int]]]]:
    """Yield selected layers in insertion order, omitting empty selections."""

    return ((layer, pairs) for layer, pairs in neuron_cfg.items() if pairs)


def _copy_selected_neuron_contributions(
    layer_grad_contrib: torch.Tensor,
    pairs: list[list[int]],
) -> torch.Tensor:
    """Copy selected ``(position, neuron)`` values into compact pair order."""

    if layer_grad_contrib.ndim != 4:
        raise ValueError("layer contribution gradients must have shape (t, b, s, d)")
    if not pairs:
        raise ValueError("cannot gather an empty neuron selection")
    positions = torch.tensor(
        [int(position) for position, _neuron in pairs],
        device=layer_grad_contrib.device,
    )
    neurons = torch.tensor(
        [int(neuron) for _position, neuron in pairs],
        device=layer_grad_contrib.device,
    )
    # Advanced indexing creates compact storage. The permutation only views
    # that compact result, never the dense (target, batch, sequence, MLP) VJP.
    return layer_grad_contrib[:, :, positions, neurons].permute(2, 1, 0)


def _project_selected_attribution_vjp(
    raw_vjp: torch.Tensor,
    embeddings: torch.Tensor,
    source_tokens: Sequence[int],
    *,
    return_gradient_only: bool,
) -> torch.Tensor:
    """Project a terminal selected-attribution VJP into compact source order.

    The returned values are first-order trace data, not inputs to a later
    backward pass. Detaching here prevents the compact projection from keeping
    the dense raw VJP alive through a gradient edge to ``embeddings``.
    """

    if raw_vjp.ndim != 4 or embeddings.ndim != 3:
        raise ValueError(
            "selected-attribution VJP must have shape "
            "(neuron * batch, batch, sequence, hidden)"
        )
    batch = int(embeddings.shape[0])
    if batch <= 0 or tuple(raw_vjp.shape[1:]) != tuple(embeddings.shape):
        raise ValueError(
            "selected-attribution VJP shape does not match embedding shape"
        )
    if raw_vjp.shape[0] % batch != 0:
        raise ValueError(
            "selected-attribution VJP lane count is not divisible by batch"
        )

    neuron_count = raw_vjp.shape[0] // batch
    dense_vjp = (
        raw_vjp.reshape(
            neuron_count,
            batch,
            batch,
            raw_vjp.shape[-2],
            raw_vjp.shape[-1],
        )
        .diagonal(dim1=1, dim2=2)
        .permute(0, 3, 1, 2)
    )
    source_positions = list(source_tokens)
    if return_gradient_only:
        projected = dense_vjp[:, :, source_positions, :]
    else:
        projected = (dense_vjp * embeddings.detach()[None, ...]).sum(-1)
        projected = projected[:, :, source_positions]
    return projected.detach()


def _get_grad_attributions_from_logits(
    model,
    input_ids,
    keep_tokens: list[int],
    focus_positions: list[list[int]] | list[int],
    focus_logits: list[list[int]] | list[int],
    focus_last_residual: bool = False,
    attention_masks: list[list[int]] | None = None,
    ablation_mode: Literal["zero", "mean", "activations"] = "zero",
    disable_stop_grad: bool = False,
    center_logits: bool = False,
    objective_weights: Sequence[float] | None = None,
    alpha: float | None = None,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[NeuronIdx], torch.Tensor]:
    """
    Get the grad attributions for all MLP neurons to the given tokens.

    Args:
        model: The model to get the grad attributions from.
        input_ids: The input id tokens to the model.
        focus_positions: The positions of the tokens to get the grad attributions to.
        focus_logits: The next-token logits of the tokens to get the grad attributions to.
        attention_masks: The attention masks for the input ids.
        ablation_mode: The ablation mode for the grad attributions.

    Returns:
        The grad attributions for all MLP neurons and embed tokens to the given tokens, under the given settings.
        - layer_grad_attr: (L, B, T, D_ff)
        - embed_grad_attr: (B, T)
    """

    for param in model.parameters():
        param.requires_grad = False

    # compute neuron attributions to source tokens and contributions to target tokens
    # clear hooks
    for i in range(len(model.model.layers)):
        if hasattr(model.model.layers[i].mlp, "mlp"):
            remove_forward_hooks(model.model.layers[i].mlp.mlp.down_proj)
        else:
            remove_forward_hooks(model.model.layers[i].mlp.down_proj)

    # differentiable embeds
    embeds = model.model.embed_tokens(input_ids).detach().requires_grad_()
    if alpha is not None:
        embeds = embeds * alpha
    cache: dict[int, torch.Tensor] = {}
    for layer_index in range(len(model.model.layers)):

        def _hook(layer_index):
            def fn(_, input, output):
                cache[layer_index] = input[0]

            return fn

        if not disable_stop_grad:
            model.model.layers[layer_index].mlp.mlp.down_proj.register_forward_hook(
                _hook(layer_index)
            )
        else:
            model.model.layers[layer_index].mlp.down_proj.register_forward_hook(
                _hook(layer_index)
            )

    # forward pass
    out = model(inputs_embeds=embeds, attention_mask=attention_masks)
    if focus_last_residual:
        goal = cache[len(model.model.layers) - 1][
            :, -1
        ].sum()  # last residual stream vector
        goal_value = (
            cache[len(model.model.layers) - 1][:, -1].detach().sum(dim=-1)
        )  # shape: (batch,)
    else:
        logits = out.logits[:, focus_positions]  # shape: (batch, positions, vocab)
        foc_log = torch.tensor(
            focus_logits, device=logits.device
        )  # shape: (batch, positions)
        # Add an extra dimension so shapes line up
        selected_logits = torch.gather(
            logits,  # (B, P, V)
            dim=2,  # select along vocab axis
            index=foc_log.unsqueeze(-1),  # (B, P, 1)
        ).squeeze(-1)  # (B, P)
        # optionally subtract the mean of all logits
        if center_logits:
            selected_logits -= logits.mean(dim=-1)
        if objective_weights is not None:
            if len(objective_weights) != selected_logits.shape[-1]:
                raise ValueError(
                    "objective weight width must match selected-logit width"
                )
            weights = selected_logits.new_tensor(objective_weights)
            selected_logits = selected_logits * weights
        if verbose:
            print("selected_logits", selected_logits)
        goal = (
            selected_logits.sum()
        )  # only grab the associated logits for each batch item
        goal_value = selected_logits.detach().sum(dim=-1)  # shape: (batch,)

    # collect grad attributions — single backward pass for all layers + embeds
    num_layers = len(model.model.layers)
    grad_targets = [cache[i] for i in range(num_layers)] + [embeds]
    if verbose:
        print("goal", goal.shape)
        print("embeds", embeds.shape)
    all_grads = torch.autograd.grad(goal, grad_targets, retain_graph=False)
    layer_grads = all_grads[:num_layers]
    embed_attributions = all_grads[num_layers]

    layer_grad_attr = []
    layer_acts = []
    for layer_index in range(num_layers):
        grad_attr = layer_grads[layer_index]
        if alpha is not None:
            attributions = grad_attr
        elif ablation_mode == "mean":
            # TODO: attention-based mean ablation is needed
            # take mean over batch
            attributions = grad_attr * (
                cache[layer_index] - cache[layer_index].mean(dim=0, keepdim=True)
            )
        elif ablation_mode == "zero":
            attributions = grad_attr * cache[layer_index]
        else:
            raise ValueError(f"Invalid ablation mode: {ablation_mode}")
        layer_grad_attr.append(attributions.detach())
        layer_acts.append(cache[layer_index].detach())

    # embed grad attr — shape: (B, T)
    if alpha is not None:
        embed_grad_attr = embed_attributions.detach()
    elif ablation_mode == "mean":
        embed_attributions = embed_attributions * (
            embeds - embeds.mean(dim=0, keepdim=True)
        )
        embed_grad_attr = embed_attributions.sum(dim=-1).detach()
    elif ablation_mode == "zero":
        embed_attributions = embed_attributions * embeds
        embed_grad_attr = embed_attributions.sum(dim=-1).detach()
    else:
        raise ValueError(f"Invalid ablation mode: {ablation_mode}")

    # the shape of layer_grad_attr is (L, B, T, D_ff)
    # we want to aggregate over the batch dimension
    layer_grad_attr = torch.stack(layer_grad_attr)
    layer_acts = torch.stack(layer_acts)

    # compute neuron attributions to source tokens and contributions to target tokens
    for i in range(len(model.model.layers)):
        if hasattr(model.model.layers[i].mlp, "mlp"):
            remove_forward_hooks(model.model.layers[i].mlp.mlp.down_proj)
        else:
            remove_forward_hooks(model.model.layers[i].mlp.down_proj)

    return layer_grad_attr, embed_grad_attr, goal_value, layer_acts, embeds


def _get_ig_attributions_from_logits(
    model,
    input_ids,
    keep_tokens: list[int],
    focus_positions: list[list[int]] | list[int],
    focus_logits: list[list[int]] | list[int],
    focus_last_residual: bool = False,
    attention_masks: list[list[int]] | None = None,
    disable_stop_grad: bool = False,
    center_logits: bool = False,
    objective_weights: Sequence[float] | None = None,
    ig_steps: int = 10,
    ig_mode: Literal["ig-inputs", "conductance"] = "ig-inputs",
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get the IG attributions for all MLP neurons to the given tokens.
    """
    # get step-wise grads
    mlp_final_attributions = []
    embed_final_attributions = []
    mlp_final_acts = []
    embed_final_acts = []
    goal_value = []
    for step in range(ig_steps + 1):
        alpha = step / ig_steps
        (
            mlp_step_attributions,
            embed_step_attributions,
            goal_value_step,
            mlp_step_acts,
            embed_step_acts,
        ) = _get_grad_attributions_from_logits(
            model,
            input_ids,
            keep_tokens,
            focus_positions,
            focus_logits=focus_logits,
            focus_last_residual=focus_last_residual,
            attention_masks=attention_masks,
            ablation_mode="zero",
            disable_stop_grad=disable_stop_grad,
            center_logits=center_logits,
            objective_weights=objective_weights,
            verbose=verbose,
            alpha=alpha,
        )
        # Immediately detach and move to CPU to save GPU memory
        mlp_final_attributions.append(mlp_step_attributions.detach().cpu())
        embed_final_attributions.append(embed_step_attributions.detach().cpu())
        mlp_final_acts.append(mlp_step_acts.detach().cpu())
        embed_final_acts.append(embed_step_acts.detach().cpu())
        goal_value.append(goal_value_step.detach())

        # Clean up GPU tensors and computation graph
        del (
            mlp_step_attributions,
            embed_step_attributions,
            mlp_step_acts,
            embed_step_acts,
        )
        torch.cuda.empty_cache()

        # Force garbage collection every few steps
        if step % 5 == 0:
            import gc

            gc.collect()
            torch.cuda.empty_cache()

    # Get device from model
    device = input_ids.device

    if ig_mode == "ig-inputs":
        # WE WILL IGNORE STEP 0
        # reimann sum in IG (move back to device for computation)
        mlp_final_attributions = (
            torch.stack(mlp_final_attributions[1:]).mean(dim=0).to(device)
        )
        embed_final_attributions = (
            torch.stack(embed_final_attributions[1:]).mean(dim=0).to(device)
        )

        # multiply by act diff
        mlp_activation_diff = (mlp_final_acts[-1] - mlp_final_acts[0]).to(device)
        embed_activation_diff = (embed_final_acts[-1] - embed_final_acts[0]).to(device)

        # multiply and collapse embed dims (i.e. outside the integral in IG)
        mlp_final_attributions = mlp_final_attributions * mlp_activation_diff
        embed_final_attributions = embed_final_attributions * embed_activation_diff
        embed_final_attributions = embed_final_attributions.sum(dim=-1).detach()
    elif ig_mode == "conductance":
        # Move to device for computation
        mlp_final_attributions = torch.stack(mlp_final_attributions[1:]).to(device)
        embed_final_attributions = torch.stack(embed_final_attributions[1:]).to(device)

        # all difs
        mlp_final_acts = torch.stack(mlp_final_acts).to(device)
        embed_final_acts = torch.stack(embed_final_acts).to(device)
        mlp_activation_diffs = torch.diff(mlp_final_acts, dim=0)
        embed_activation_diffs = torch.diff(embed_final_acts, dim=0)

        # multiply gradients by diffs
        mlp_final_attributions = (mlp_final_attributions * mlp_activation_diffs).sum(
            dim=0
        )
        embed_final_attributions = (
            (embed_final_attributions * embed_activation_diffs)
            .sum(dim=0)
            .sum(dim=-1)
            .detach()
        )
    else:
        raise ValueError(f"Invalid IG mode: {ig_mode}")

    # return
    return (
        mlp_final_attributions,
        embed_final_attributions,
        goal_value[-1],
        mlp_final_acts[-1],
        embed_final_acts[-1],
    )


@torch.no_grad()
def _get_global_important_neurons_mask(
    keep_tokens: list[int],
    start_layer: int,
    end_layer: int,
    mlp_final_attributions: torch.Tensor,
    node_attribution_threshold: float | None = 1.0,
    topk_neurons: int | None = None,
    absolute_attribution_threshold: torch.Tensor | None = None,
    batch_aggregation: Literal["mean", "max", "max_abs", "any"] = "max",
    verbose: bool = False,
) -> torch.Tensor:
    """
    Filter for the most important MLP neurons.

    Args:
        mlp_final_attributions: Tensor with shape allowing indexing as [layer, batch, token_pos, dim, focus_logits].
        node_attribution_threshold: Cumulative absolute attribution threshold.
        topk_neurons: Topk neurons to select.

    Returns:
        A mask of shape [num_layers, num_tokens, num_neurons] indicating which neurons are important.
    """
    # Only one strategy can be used at a time
    if node_attribution_threshold is not None and topk_neurons is not None:
        raise ValueError("Only one strategy can be used at a time")

    # first aggregate over batch
    if batch_aggregation == "mean":
        mlp_final_attributions = torch.mean(mlp_final_attributions, dim=1, keepdim=True)
    elif batch_aggregation == "max":
        mlp_final_attributions = torch.max(
            mlp_final_attributions, dim=1, keepdim=True
        ).values
    elif batch_aggregation == "max_abs":
        mlp_final_attributions = torch.max(
            torch.abs(mlp_final_attributions), dim=1, keepdim=True
        ).values
    elif batch_aggregation == "any":
        pass  # keep batch dim

    # aggregate over the last dimension of the logits
    mlp_final_attributions = mlp_final_attributions.sum(dim=-1)

    # Stack all layers into a single tensor for efficient processing
    # Resulting shape: [batch, num_layers, num_tokens, dim_neurons]
    # batch is 1 if batch_aggregation is not "any"
    abs_attributions = mlp_final_attributions.abs().permute(1, 0, 2, 3)

    # if it is not in the keep tokens, set them to zeros
    token_mask = ~torch.isin(
        torch.arange(abs_attributions.shape[2]), torch.tensor(keep_tokens)
    )
    abs_attributions[:, :, token_mask, :] = 0

    # if layer indices fall outside start_layer and end_layer, set them to zeros
    layer_indices = torch.arange(abs_attributions.shape[1])
    layer_mask = (layer_indices < start_layer) | (layer_indices >= end_layer)
    abs_attributions[:, layer_mask, :, :] = 0

    if verbose:
        print("abs_attributions", abs_attributions.shape)

    # Flatten to find threshold
    # keep batch dim
    flat_abs_attributions = abs_attributions.flatten(start_dim=1)
    if verbose:
        print("flat_abs_attributions", flat_abs_attributions.shape)
    if not has_any_strictly_positive_value(flat_abs_attributions):
        if verbose:
            print("all zero values")
        return torch.zeros_like(abs_attributions)[0]

    # option 1: threshold
    if node_attribution_threshold is not None:
        # Sort without stable=True to save memory
        sorted_values = torch.sort(
            flat_abs_attributions, descending=True, dim=-1
        ).values
        if verbose:
            print("sorted_values", sorted_values.shape)

        total_mass = sorted_values.sum(dim=-1, keepdim=True)
        cumulative_mass = torch.cumsum(sorted_values, dim=-1)
        cumulative_ratio = cumulative_mass / total_mass  # shape: [batch, num_vals]

        # Find threshold value
        thr = torch.full(
            cumulative_ratio.shape[:-1],  # [B]
            float(node_attribution_threshold),
            dtype=cumulative_ratio.dtype,
            device=cumulative_ratio.device,
        ).unsqueeze(-1)
        cutoff_idx = torch.searchsorted(cumulative_ratio, thr) + 1
        cutoff_idx = cutoff_idx.squeeze(-1)
        threshold_value = sorted_values[
            torch.arange(sorted_values.shape[0]), cutoff_idx - 1
        ]

        # Create mask and get coordinates directly
        mask = abs_attributions >= threshold_value[:, None, None, None]

        # Clean up
        del sorted_values, cumulative_mass, cumulative_ratio
        torch.cuda.empty_cache()
    # option 2: get topk neurons - use topk instead of full sort
    elif topk_neurons is not None:
        # Use topk directly instead of full sort to save memory
        topk_values = torch.topk(
            flat_abs_attributions,
            k=min(topk_neurons, flat_abs_attributions.shape[-1]),
            dim=-1,
        ).values
        threshold_value = topk_values[:, -1]  # Last value in topk is the threshold
        if verbose:
            print("threshold_value", threshold_value.shape)
        mask = abs_attributions >= threshold_value[:, None, None, None]

        # Clean up
        del topk_values
        torch.cuda.empty_cache()
    # option 3: get neurons with absolute attribution above a threshold
    elif absolute_attribution_threshold is not None:
        threshold_value = absolute_attribution_threshold
        mask = abs_attributions >= threshold_value[:, None, None, None]
    else:
        raise ValueError(
            "Either node_attribution_threshold, topk_neurons, or absolute_attribution_threshold must be provided"
        )

    if batch_aggregation != "any":
        mask = mask.squeeze()
    else:
        if verbose:
            print("threshold_value", threshold_value)
        mask = mask.any(dim=0)
    if verbose:
        print("mask", mask.shape)

    return mask


def _get_neuron_attr_and_contrib_with_stop_grad_on_mlps(
    model,
    neuron_cfg: dict[int, list[list[int]]],
    input_ids: torch.Tensor,
    src_tokens: list[int],
    tgt_tokens: list[int],
    focus_positions: list[int],
    focus_logits,
    attention_masks,
    use_relp_grad: bool = False,
    center_logits: bool = False,
    neuron_chunk_size: int = 50,
    verbose: bool = False,
    instrumentation: TraceInstrumentation | None = None,
    attention_backend: StopGradientAttentionBackend = (
        DEFAULT_STOP_GRADIENT_ATTENTION_BACKEND
    ),
    contribution_execution: StopGradientContributionExecution = (
        DEFAULT_STOP_GRADIENT_CONTRIBUTION_EXECUTION
    ),
    contribution_target_lane_chunk_size: int | None = None,
    embed_contribution_target_lane_chunk_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[NeuronIdx]]:
    """
    Compute neuron attributions from source tokens and contributions to target tokens
    with stop gradient on MLPs.
    """

    model = revert_stop_nonlinear_grad(model)
    model.zero_grad()
    for i in range(len(model.model.layers)):
        remove_forward_hooks(model.model.layers[i].mlp.down_proj)

    device = input_ids.device

    # process neuron activations layer by layer for memory efficiency
    attr_list = []
    neuron_acts_list = []
    neuron_tags: list[NeuronIdx] = []

    for lid, pairs in tqdm(
        neuron_cfg.items(), desc="Computing neuron attributions", disable=not verbose
    ):
        if not pairs:  # skip empty layers
            continue

        with cuda_memory_observation_stage(
            instrumentation,
            "stop_grad_selected_layer_forward",
            metadata={
                "operation_kind": "model_forward",
                "layer": lid,
                "selected_neuron_count": len(pairs),
                "planned_chunk_count": (len(pairs) + neuron_chunk_size - 1)
                // neuron_chunk_size,
                "neuron_chunk_size": neuron_chunk_size,
                "source_token_count": len(src_tokens),
                "retained_layer_count_before": len(attr_list),
                "retained_attribution_numel_before": sum(
                    tensor.numel() for tensor in attr_list
                ),
                "retained_neuron_activation_numel_before": sum(
                    tensor.numel() for tensor in neuron_acts_list
                ),
            },
        ) as forward_measurement:
            # stop gradient on MLPs
            model = layerwise_stop_nonlinear_grad(
                model,
                -1,
                lid,
                use_relp_grad=use_relp_grad,
                use_stop_grad_on_mlps=True,
                attention_backend=attention_backend,
            )

            cache = {}

            def _hook(lid, layer_cache):
                def fn(_, input, output):
                    layer_cache[lid] = input[0]

                return fn

            # differentiable embeds
            # shape: (batch, seq, d)
            embeds = model.model.embed_tokens(input_ids).detach().requires_grad_()

            if hasattr(model.model.layers[lid].mlp, "mlp"):
                hook_handle = model.model.layers[
                    lid
                ].mlp.mlp.down_proj.register_forward_hook(_hook(lid, cache))
            else:
                hook_handle = model.model.layers[
                    lid
                ].mlp.down_proj.register_forward_hook(_hook(lid, cache))

            # forward pass
            try:
                out = model(inputs_embeds=embeds, attention_mask=attention_masks)
            finally:
                hook_handle.remove()
            logits = out.logits
            if center_logits:
                logits -= logits.mean(dim=-1)

            act = cache[lid]  # (batch, seq, d)
            if forward_measurement is not None:
                forward_measurement.metadata["embedding_shape"] = list(embeds.shape)
                forward_measurement.metadata["logit_shape"] = list(logits.shape)
                forward_measurement.metadata["activation_shape"] = list(act.shape)
                forward_measurement.metadata["capture_hook_removed"] = True
                forward_measurement.metadata["downstream_graph_released"] = True
            # The selected activation depends only on the prefix through this
            # layer. The logits and downstream suffix graph are never consumed
            # by the selected-attribution VJPs.
            del out, logits, cache, hook_handle

        # Process this layer's neurons in chunks
        all_pairs = list(pairs)
        layer_attr_chunks = []
        layer_neuron_acts_chunks = []

        for chunk_start in range(0, len(all_pairs), neuron_chunk_size):
            with cuda_memory_observation_stage(
                instrumentation,
                "stop_grad_selected_chunk_prepare",
                metadata={
                    "operation_kind": "vjp_prepare",
                    "layer": lid,
                    "chunk_start": chunk_start,
                    "planned_chunk_neuron_count": min(
                        neuron_chunk_size, len(all_pairs) - chunk_start
                    ),
                },
            ) as prepare_measurement:
                chunk_pairs = all_pairs[chunk_start : chunk_start + neuron_chunk_size]

                layer_neuron_acts = []
                layer_neuron_tags = []

                for pos, nid in chunk_pairs:
                    layer_neuron_acts.append(act[:, pos, nid])
                    layer_neuron_tags.append(
                        NeuronIdx(layer=lid, token=pos, neuron=nid)
                    )

                if not layer_neuron_acts:  # skip if no neurons
                    continue

                # get all activations
                layer_neuron_acts = torch.stack(layer_neuron_acts)  # (n_chunk, batch)
                batch = layer_neuron_acts.shape[1]
                n_chunk = layer_neuron_acts.shape[0]
                retain_attribution_graph = chunk_start + len(chunk_pairs) < len(
                    all_pairs
                )
                grad_outputs = torch.eye(
                    n_chunk * batch, device=device
                )  # (n_chunk * batch, n_chunk * batch)

                # attribution to inputs for this layer chunk
                embeds.grad = None
                if prepare_measurement is not None:
                    prepare_measurement.metadata["chunk_neuron_count"] = n_chunk
                    prepare_measurement.metadata["lane_count"] = n_chunk * batch
                    prepare_measurement.metadata["retain_graph"] = (
                        retain_attribution_graph
                    )
                    prepare_measurement.metadata["selected_activation_shape"] = list(
                        layer_neuron_acts.shape
                    )
                    prepare_measurement.metadata["grad_outputs_shape"] = list(
                        grad_outputs.shape
                    )
            with cuda_memory_instrumentation_stage(
                instrumentation,
                "stop_grad_selected_attribution_vjp",
                metadata={
                    "operation_kind": "batched_vjp",
                    "layer": lid,
                    "chunk_start": chunk_start,
                    "chunk_neuron_count": n_chunk,
                    "lane_count": n_chunk * batch,
                    "differentiated_output_shape": list(layer_neuron_acts.shape),
                    "differentiated_input_shape": list(embeds.shape),
                    "grad_outputs_shape": list(grad_outputs.shape),
                    "retain_graph": retain_attribution_graph,
                },
            ) as vjp_measurement:
                layer_grad_attr = torch.autograd.grad(
                    layer_neuron_acts.flatten(),  # (n_chunk * batch)
                    embeds,  # (batch, seq, d)
                    grad_outputs=grad_outputs,  # (n_chunk * batch, n_chunk * batch)
                    is_grads_batched=True,
                    retain_graph=retain_attribution_graph,
                )[0]
                if vjp_measurement is not None:
                    vjp_measurement.metadata["vjp_result_shape"] = list(
                        layer_grad_attr.shape
                    )
            with cuda_memory_observation_stage(
                instrumentation,
                "stop_grad_selected_chunk_projection",
                metadata={
                    "operation_kind": "vjp_projection",
                    "layer": lid,
                    "chunk_start": chunk_start,
                    "chunk_neuron_count": n_chunk,
                    "source_token_count": len(src_tokens),
                    "raw_vjp_result_shape": list(layer_grad_attr.shape),
                    "retained_chunk_count_before": len(layer_attr_chunks),
                },
            ) as projection_measurement:
                # shape: (n_chunk * batch, batch, seq, d)
                # convert back to (n_chunk, batch, batch, seq, d)
                layer_grad_attr = layer_grad_attr.reshape(
                    n_chunk,
                    batch,
                    batch,
                    layer_grad_attr.shape[-2],
                    layer_grad_attr.shape[-1],
                )
                # get only identity along batch, batch
                layer_grad_attr = layer_grad_attr.diagonal(dim1=1, dim2=2).permute(
                    0, 3, 1, 2
                )
                # now (n_chunk, batch, seq, d)

                layer_attr = (
                    (
                        layer_grad_attr.to(torch.float32)
                        * embeds[None, ...].to(torch.float32)
                    )
                    .sum(-1)
                    .to(embeds.dtype)
                )  # (n_chunk, batch, seq)
                layer_attr = layer_attr[:, :, src_tokens]  # (n_chunk, batch, src)

                layer_attr_chunks.append(layer_attr)
                layer_neuron_acts_chunks.append(layer_neuron_acts.detach())
                neuron_tags.extend(layer_neuron_tags)
                if projection_measurement is not None:
                    projection_measurement.metadata["projected_shape"] = list(
                        layer_attr.shape
                    )
                    projection_measurement.metadata["retained_chunk_count_after"] = len(
                        layer_attr_chunks
                    )

                # Clean up memory after each chunk
                del layer_grad_attr, layer_attr, grad_outputs
                torch.cuda.empty_cache()

        with cuda_memory_observation_stage(
            instrumentation,
            "stop_grad_selected_layer_release",
            metadata={
                "operation_kind": "layer_finalize_release",
                "layer": lid,
                "selected_neuron_count": len(pairs),
                "retained_chunk_count": len(layer_attr_chunks),
                "retained_layer_count_before": len(attr_list),
            },
        ) as release_measurement:
            # Concatenate chunks for this layer
            if layer_attr_chunks:
                attr_list.append(torch.cat(layer_attr_chunks, dim=0))
                neuron_acts_list.append(torch.cat(layer_neuron_acts_chunks, dim=0))

            # revert stop grad for the next attribution
            model = layerwise_revert_stop_nonlinear_grad(
                model,
                -1,
                lid,
            )
            remove_forward_hooks(model.model.layers[lid].mlp.down_proj)

            if release_measurement is not None:
                release_measurement.metadata["retained_layer_count_after"] = len(
                    attr_list
                )
                if layer_attr_chunks:
                    release_measurement.metadata["retained_attribution_shape"] = list(
                        attr_list[-1].shape
                    )
                release_measurement.metadata["retained_attribution_numel_after"] = sum(
                    tensor.numel() for tensor in attr_list
                )
                release_measurement.metadata[
                    "retained_neuron_activation_numel_after"
                ] = sum(tensor.numel() for tensor in neuron_acts_list)

            # clean up memory
            del embeds, layer_attr_chunks, layer_neuron_acts_chunks
            torch.cuda.empty_cache()

    with cuda_memory_observation_stage(
        instrumentation,
        "stop_grad_selected_phase_finalize",
        metadata={
            "operation_kind": "phase_finalize",
            "retained_layer_count": len(attr_list),
            "selected_neuron_count": len(neuron_tags),
        },
    ) as selected_finalize_measurement:
        # concatenate results from all layers
        attr = torch.cat(attr_list, dim=0)  # (neurons, batch, seq)
        neuron_acts = torch.cat(neuron_acts_list, dim=0)  # (neurons, batch)
        if selected_finalize_measurement is not None:
            selected_finalize_measurement.metadata["attribution_shape"] = list(
                attr.shape
            )
            selected_finalize_measurement.metadata["neuron_activation_shape"] = list(
                neuron_acts.shape
            )

        # clean up layer-wise lists to free memory
        del attr_list, neuron_acts_list
        torch.cuda.empty_cache()

    with cuda_memory_observation_stage(
        instrumentation,
        "stop_grad_embed_forward",
        metadata={
            "operation_kind": "model_forward",
            "source_token_count": len(src_tokens),
            "target_token_count": len(tgt_tokens),
        },
    ) as embed_forward_measurement:
        # contribution to tgt tokens
        # stop gradient on MLPs
        model = layerwise_stop_nonlinear_grad(
            model,
            -1,
            len(model.model.layers),
            use_relp_grad=use_relp_grad,
            use_stop_grad_on_mlps=True,
            attention_backend=attention_backend,
        )

        # differentiable embeds
        # shape: (batch, seq, d)
        embeds = model.model.embed_tokens(input_ids).detach().requires_grad_()

        # forward pass
        out = model(inputs_embeds=embeds, attention_mask=attention_masks)
        logits = out.logits
        if center_logits:
            logits -= logits.mean(dim=-1)

        tgt_nodes = []
        for id, p in enumerate(focus_positions):
            tid = [focus_logit[id] for focus_logit in focus_logits]
            tgt_nodes.append(logits[torch.arange(logits.shape[0]), p, tid])
        tgt_vec = torch.stack(tgt_nodes)  # (t, batch)
        t = tgt_vec.size(0)
        if embed_forward_measurement is not None:
            embed_forward_measurement.metadata["embedding_shape"] = list(embeds.shape)
            embed_forward_measurement.metadata["logit_shape"] = list(logits.shape)
            embed_forward_measurement.metadata["target_shape"] = list(tgt_vec.shape)

    # compute embed grad contribs
    embed_grad_contrib = run_stop_gradient_embed_contribution_vjp(
        embeds,
        tgt_vec,
        src_tokens,
        target_lane_chunk_size=embed_contribution_target_lane_chunk_size,
        instrumentation=instrumentation,
    )
    with cuda_memory_observation_stage(
        instrumentation,
        "stop_grad_embed_projection_release",
        metadata={
            "operation_kind": "vjp_projection_release",
            "source_token_count": len(src_tokens),
            "target_token_count": t,
            "projected_shape": list(embed_grad_contrib.shape),
        },
    ):
        # revert stop grad for the next attribution
        model = layerwise_revert_stop_nonlinear_grad(
            model,
            -1,
            len(model.model.layers),
        )
        del embeds, tgt_vec
        torch.cuda.empty_cache()

    # compute neuron grad contribs
    grad_contrib = []
    for lid, pairs in tqdm(
        _nonempty_neuron_layers(neuron_cfg),
        desc="Computing neuron contributions",
        disable=not verbose,
    ):
        with cuda_memory_observation_stage(
            instrumentation,
            "stop_grad_neuron_layer_forward",
            metadata={
                "operation_kind": "model_forward",
                "layer": lid,
                "selected_neuron_count": len(pairs),
                "target_token_count": len(tgt_tokens),
                "contribution_execution": contribution_execution,
                "retained_layer_count_before": len(grad_contrib),
                "retained_contribution_numel_before": sum(
                    tensor.numel() for tensor in grad_contrib
                ),
            },
        ) as contribution_forward_measurement:
            # stop gradient on MLPs
            model = layerwise_stop_nonlinear_grad(
                model,
                lid,
                len(model.model.layers),
                use_relp_grad=use_relp_grad,
                use_stop_grad_on_mlps=True,
                attention_backend=attention_backend,
            )

            contribution_forward = run_stop_gradient_contribution_forward(
                model,
                input_ids,
                attention_masks,
                layer=lid,
                execution=contribution_execution,
                selected_coordinates=pairs,
                instrumentation=instrumentation,
            )
            logits = contribution_forward.logits
            if center_logits:
                logits -= logits.mean(dim=-1)

            tgt_nodes = []
            for id, p in enumerate(focus_positions):
                tid = [focus_logit[id] for focus_logit in focus_logits]
                tgt_nodes.append(logits[torch.arange(logits.shape[0]), p, tid])
            tgt_vec = torch.stack(tgt_nodes)  # (t, batch)
            if contribution_forward_measurement is not None:
                contribution_forward_measurement.metadata["logit_shape"] = list(
                    logits.shape
                )
                contribution_forward_measurement.metadata["target_shape"] = list(
                    tgt_vec.shape
                )
                contribution_forward_measurement.metadata[
                    "differentiated_source_shape"
                ] = list(contribution_forward.source_activation.shape)
                contribution_forward_measurement.metadata["source_representation"] = (
                    contribution_forward.source_representation
                )
        grad_contrib.append(
            run_stop_gradient_contribution_vjp(
                contribution_forward,
                tgt_vec,
                layer=lid,
                target_lane_chunk_size=contribution_target_lane_chunk_size,
                instrumentation=instrumentation,
            )
        )

        with cuda_memory_observation_stage(
            instrumentation,
            "stop_grad_neuron_layer_release",
            metadata={
                "operation_kind": "layer_release",
                "layer": lid,
                "selected_neuron_count": len(pairs),
                "retained_layer_count": len(grad_contrib),
                "retained_contribution_shape": list(grad_contrib[-1].shape),
            },
        ) as contribution_release_measurement:
            # revert stop grad for the next attribution
            model = layerwise_revert_stop_nonlinear_grad(
                model,
                lid,
                len(model.model.layers),
            )
            del contribution_forward, tgt_vec
            torch.cuda.empty_cache()
            if contribution_release_measurement is not None:
                contribution_release_measurement.metadata[
                    "retained_contribution_numel"
                ] = sum(tensor.numel() for tensor in grad_contrib)

    with cuda_memory_observation_stage(
        instrumentation,
        "stop_grad_neuron_phase_finalize",
        metadata={
            "operation_kind": "phase_finalize",
            "retained_layer_count": len(grad_contrib),
            "selected_neuron_count": len(neuron_tags),
        },
    ) as neuron_finalize_measurement:
        # multiple by acts to get contributions
        grad_contrib = torch.cat(grad_contrib, dim=0)  # (neurons, batch, tgt)
        contrib = (
            grad_contrib * neuron_acts.detach()[:, :, None]
        )  # (neurons, batch, tgt)
        if neuron_finalize_measurement is not None:
            neuron_finalize_measurement.metadata["gradient_contribution_shape"] = list(
                grad_contrib.shape
            )
            neuron_finalize_measurement.metadata["contribution_shape"] = list(
                contrib.shape
            )

        # Clean up
        del grad_contrib
        torch.cuda.empty_cache()

    # assert shapes
    if verbose:
        print("attr.shape", attr.shape)
        print("contrib.shape", contrib.shape)
    assert attr.shape == (len(neuron_tags), batch, len(src_tokens))
    assert contrib.shape == (len(neuron_tags), batch, len(tgt_tokens))

    # detach
    attr = attr.detach()
    contrib = contrib.detach()

    # return
    return attr, contrib, embed_grad_contrib, neuron_tags


def _get_neuron_attr_and_contrib(
    model,
    neuron_cfg: dict[int, list[list[int]]],
    input_ids: torch.Tensor,
    src_tokens: list[int],
    tgt_tokens: list[int],
    focus_positions: list[int],
    focus_logits,
    attention_masks,
    use_relp_grad: bool = False,
    disable_stop_grad: bool = False,
    center_logits: bool = False,
    alpha: float | None = None,
    neuron_chunk_size: int = 50,
    verbose: bool = False,
    instrumentation: TraceInstrumentation | None = None,
    contribution_target_lane_chunk_size: int | None = None,
    embed_contribution_target_lane_chunk_size: int | None = None,
    contribution_execution_index: int | None = None,
) -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[NeuronIdx]]
    | tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        list[NeuronIdx],
        torch.Tensor,
        torch.Tensor,
    ]
):
    """
    Compute neuron attributions from source tokens and contributions to target tokens.

    Args:
        model: The model to compute neuron attributions from.
        neuron_cfg: A dict mapping layer indices to lists of (position, neuron) tuples.
        input_ids: The input id tokens to the model.
        src_tokens: The token positions to trace from.
        tgt_tokens: The token positions to trace to.
        focus_positions: The token positions to consider the logits effect to.
        focus_logits: The token vocab ids to trace logits to.
        attention_masks: The attention masks for the input ids.
        alpha: Optional scaling factor for integrated gradients.

    Returns:
        If alpha is None:
            A tuple of (neuron attributions, neuron contributions, embed_grad_contrib, neuron tags).
        If alpha is not None:
            A tuple of (neuron attributions, neuron contributions, embed_grad_contrib, neuron tags, neuron_acts, embeds).
        - neuron attributions: Tensor with shape [neurons, batch, src].
        - neuron contributions: Tensor with shape [neurons, batch, tgt].
        - embed_grad_contrib: Tensor with shape [src, batch, tgt].
        - neuron tags: List of NeuronIdx objects indicating which neurons are important.
        - neuron_acts: Tensor with shape [neurons, batch] (only when alpha is not None).
        - embeds: Tensor with shape [batch, seq, d] (only when alpha is not None).
    """

    # compute neuron attributions to source tokens and contributions to target tokens
    # clear hooks
    for i in range(len(model.model.layers)):
        if hasattr(model.model.layers[i].mlp, "mlp"):
            remove_forward_hooks(model.model.layers[i].mlp.mlp.down_proj)
        else:
            remove_forward_hooks(model.model.layers[i].mlp.down_proj)

    device = input_ids.device

    # differentiable embeds
    # shape: (batch, seq, d)
    embeds = model.model.embed_tokens(input_ids).detach().requires_grad_()
    if alpha is not None:
        embeds = embeds * alpha

    cache = {}
    for lid in neuron_cfg:

        def _hook(lid):
            def fn(_, input, output):
                cache[lid] = input[0]

            return fn

        if hasattr(model.model.layers[lid].mlp, "mlp"):
            model.model.layers[lid].mlp.mlp.down_proj.register_forward_hook(_hook(lid))
        else:
            model.model.layers[lid].mlp.down_proj.register_forward_hook(_hook(lid))

    # forward pass
    out = model(inputs_embeds=embeds, attention_mask=attention_masks)
    logits = out.logits
    if center_logits:
        logits -= logits.mean(dim=-1)

    # process neuron activations layer by layer for memory efficiency
    attr_list = []
    neuron_acts_list = []
    neuron_tags: list[NeuronIdx] = []

    # Process neurons in chunks to avoid OOM with large identity matrices
    chunk_size = resolve_selected_attribution_neuron_lane_chunk_size(neuron_chunk_size)

    for lid, pairs in tqdm(
        neuron_cfg.items(), desc="Computing neuron attributions", disable=not verbose
    ):
        if not pairs:  # skip empty layers
            continue

        act = cache[lid]  # (batch, seq, d)

        # Process this layer's neurons in chunks
        all_pairs = list(pairs)
        layer_attr_chunks = []
        layer_neuron_acts_chunks = []
        layer_chunk_count = (len(all_pairs) + chunk_size - 1) // chunk_size

        for chunk_start in range(0, len(all_pairs), chunk_size):
            chunk_pairs = all_pairs[chunk_start : chunk_start + chunk_size]
            chunk_index = chunk_start // chunk_size

            layer_neuron_acts = []
            layer_neuron_tags = []

            for pos, nid in chunk_pairs:
                layer_neuron_acts.append(act[:, pos, nid])
                layer_neuron_tags.append(NeuronIdx(layer=lid, token=pos, neuron=nid))

            if not layer_neuron_acts:  # skip if no neurons
                continue

            # get all activations
            layer_neuron_acts = torch.stack(layer_neuron_acts)  # (n_chunk, batch)
            batch = layer_neuron_acts.shape[1]
            n_chunk = layer_neuron_acts.shape[0]
            grad_outputs = torch.eye(
                n_chunk * batch, device=device
            )  # (n_chunk * batch, n_chunk * batch)

            # attribution to inputs for this layer chunk
            embeds.grad = None
            with cuda_memory_instrumentation_stage(
                instrumentation,
                "selected_attribution_vjp",
                metadata={
                    "operation_kind": "batched_vjp",
                    "layer": lid,
                    "chunk_start": chunk_start,
                    "chunk_index": chunk_index,
                    "chunk_count": layer_chunk_count,
                    "neuron_lane_chunk_size_resolved": chunk_size,
                    "chunk_neuron_count": n_chunk,
                    "lane_count": n_chunk * batch,
                    "differentiated_output_shape": list(layer_neuron_acts.shape),
                    "differentiated_input_shape": list(embeds.shape),
                    "grad_outputs_shape": list(grad_outputs.shape),
                },
            ) as vjp_measurement:
                layer_grad_attr = torch.autograd.grad(
                    layer_neuron_acts.flatten(),  # (n_chunk * batch)
                    embeds,  # (batch, seq, d)
                    grad_outputs=grad_outputs,  # (n_chunk * batch, n_chunk * batch)
                    is_grads_batched=True,
                    retain_graph=True,
                )[0]
                if vjp_measurement is not None:
                    vjp_measurement.metadata["vjp_result_shape"] = list(
                        layer_grad_attr.shape
                    )
            with cuda_memory_observation_stage(
                instrumentation,
                "selected_attribution_chunk_projection",
                metadata={
                    "operation_kind": "terminal_projection",
                    "layer": lid,
                    "chunk_start": chunk_start,
                    "chunk_index": chunk_index,
                    "chunk_count": layer_chunk_count,
                    "neuron_lane_chunk_size_resolved": chunk_size,
                    "chunk_neuron_count": n_chunk,
                    "raw_vjp_result_shape": list(layer_grad_attr.shape),
                    "source_token_count": len(src_tokens),
                    "return_gradient_only": alpha is not None,
                    "terminal_projection_detached": True,
                    "retained_chunk_count_before": len(layer_attr_chunks),
                },
            ) as projection_measurement:
                layer_attr = _project_selected_attribution_vjp(
                    layer_grad_attr,
                    embeds,
                    src_tokens,
                    return_gradient_only=alpha is not None,
                )
                layer_attr_chunks.append(layer_attr)
                del layer_grad_attr
                if projection_measurement is not None:
                    projection_measurement.metadata["projected_shape"] = list(
                        layer_attr.shape
                    )
                    projection_measurement.metadata["projected_requires_grad"] = (
                        layer_attr.requires_grad
                    )
                    projection_measurement.metadata["retained_chunk_count_after"] = len(
                        layer_attr_chunks
                    )

            layer_neuron_acts_chunks.append(layer_neuron_acts.detach())
            neuron_tags.extend(layer_neuron_tags)

            # clean up memory after each chunk
            del layer_attr, grad_outputs
            torch.cuda.empty_cache()

        # Concatenate chunks for this layer
        if layer_attr_chunks:
            attr_list.append(torch.cat(layer_attr_chunks, dim=0))
            neuron_acts_list.append(torch.cat(layer_neuron_acts_chunks, dim=0))

    # concatenate results from all layers
    attr = torch.cat(attr_list, dim=0)  # (neurons, batch, seq)
    neuron_acts = torch.cat(neuron_acts_list, dim=0)  # (neurons, batch)

    # clean up layer-wise lists to free memory
    del attr_list, neuron_acts_list
    torch.cuda.empty_cache()

    # contribution to tgt tokens
    tgt_nodes = []
    for id, p in enumerate(focus_positions):
        tid = [focus_logit[id] for focus_logit in focus_logits]
        tgt_nodes.append(logits[torch.arange(logits.shape[0]), p, tid])
    tgt_vec = torch.stack(tgt_nodes)  # (t, batch)
    t = tgt_vec.size(0)
    needs_full_grad_outputs = (
        embed_contribution_target_lane_chunk_size is None
        or contribution_target_lane_chunk_size is None
    )
    grad_outputs = (
        torch.eye(t * batch, device=device) if needs_full_grad_outputs else None
    )
    embed_grad_contrib = run_selected_embed_contribution_vjp(
        embeds,
        tgt_vec,
        src_tokens,
        return_gradient_only=alpha is not None,
        target_lane_chunk_size=embed_contribution_target_lane_chunk_size,
        full_grad_outputs=(
            grad_outputs if embed_contribution_target_lane_chunk_size is None else None
        ),
        instrumentation=instrumentation,
        execution_index=contribution_execution_index,
    )

    torch.cuda.empty_cache()

    ordinary_full_grad_outputs = (
        grad_outputs if contribution_target_lane_chunk_size is None else None
    )
    if ordinary_full_grad_outputs is None:
        # The embedding compatibility path is finished. Do not retain its full
        # identity while the independently chunked neuron VJPs execute.
        del grad_outputs
        grad_outputs = None

    # Compute compact per-layer gradients while bounding the materialized target
    # lanes. All sources share this forward graph, so the execution module owns
    # the retain-graph and raw-VJP lifetime contract.
    contribution_sources = [
        SelectedNeuronContributionSource(
            layer=lid,
            source_activation=cache[lid],
            selected_coordinates=tuple(
                (int(position), int(neuron)) for position, neuron in pairs
            ),
        )
        for lid, pairs in _nonempty_neuron_layers(neuron_cfg)
    ]
    grad_contrib = run_selected_neuron_contribution_vjps(
        contribution_sources,
        tgt_vec,
        target_lane_chunk_size=contribution_target_lane_chunk_size,
        full_grad_outputs=ordinary_full_grad_outputs,
        instrumentation=instrumentation,
        execution_index=contribution_execution_index,
    )

    # multiple by acts to get contributions
    grad_contrib = torch.cat(grad_contrib, dim=0)  # (neurons, batch, tgt)
    if alpha is not None:
        # For IG: return gradient only
        contrib = grad_contrib
    else:
        # For regular contribution: gradient * activation
        contrib = (
            grad_contrib * neuron_acts.detach()[:, :, None]
        )  # (neurons, batch, tgt)

    # Clean up after contribution computation
    del grad_contrib, tgt_vec, grad_outputs, ordinary_full_grad_outputs
    torch.cuda.empty_cache()

    # assert shapes
    if verbose:
        print("attr.shape", attr.shape)
        print("contrib.shape", contrib.shape)

    if alpha is not None:
        # When alpha is set, shapes include the embedding dimension
        # attr: (neurons, batch, src, d), contrib: (neurons, batch, tgt)
        assert attr.shape[0] == len(neuron_tags)
        assert attr.shape[1] == batch
        assert attr.shape[2] == len(src_tokens)
        assert contrib.shape == (len(neuron_tags), batch, len(tgt_tokens))
    else:
        assert attr.shape == (len(neuron_tags), batch, len(src_tokens))
        assert contrib.shape == (len(neuron_tags), batch, len(tgt_tokens))

    # detach
    attr = attr.detach()
    contrib = contrib.detach()

    # return
    if alpha is not None:
        return attr, contrib, embed_grad_contrib, neuron_tags, neuron_acts, embeds
    return attr, contrib, embed_grad_contrib, neuron_tags


def _get_neuron_attr_and_contrib_ig(
    model,
    neuron_cfg: dict[int, list[list[int]]],
    input_ids: torch.Tensor,
    src_tokens: list[int],
    tgt_tokens: list[int],
    focus_positions: list[int],
    focus_logits,
    attention_masks,
    use_relp_grad: bool = False,
    disable_stop_grad: bool = False,
    center_logits: bool = False,
    ig_steps: int = 10,
    ig_mode: Literal["ig-inputs", "conductance"] = "ig-inputs",
    neuron_chunk_size: int = 50,
    verbose: bool = False,
    instrumentation: TraceInstrumentation | None = None,
    contribution_target_lane_chunk_size: int | None = None,
    embed_contribution_target_lane_chunk_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[NeuronIdx]]:
    """
    Compute neuron attributions and contributions using Integrated Gradients.

    Args:
        model: The model to compute neuron attributions from.
        neuron_cfg: A dict mapping layer indices to lists of (position, neuron) tuples.
        input_ids: The input id tokens to the model.
        src_tokens: The token positions to trace from.
        tgt_tokens: The token positions to trace to.
        focus_positions: The token positions to consider the logits effect to.
        focus_logits: The token vocab ids to trace logits to.
        attention_masks: The attention masks for the input ids.
        ig_steps: Number of steps for integrated gradients.
        ig_mode: Mode for integrated gradients aggregation ("ig-inputs" or "conductance").

    Returns:
        A tuple of (neuron attributions, neuron contributions, embed_grad_contrib, neuron tags).
        - neuron attributions: Tensor with shape [neurons, batch, src].
        - neuron contributions: Tensor with shape [neurons, batch, tgt].
        - embed_grad_contrib: Tensor with shape [src, batch, tgt].
        - neuron tags: List of NeuronIdx objects indicating which neurons are important.
    """
    # Collect step-wise attributions and contributions
    attr_steps = []
    contrib_steps = []
    embed_grad_contrib_steps = []
    neuron_acts_steps = []
    embeds_steps = []
    neuron_tags = None

    for step in range(ig_steps + 1):
        alpha = step / ig_steps

        attr, contrib, embed_grad_contrib, tags, neuron_acts, embeds = (
            _get_neuron_attr_and_contrib(
                model=model,
                neuron_cfg=neuron_cfg,
                input_ids=input_ids,
                src_tokens=src_tokens,
                tgt_tokens=tgt_tokens,
                focus_positions=focus_positions,
                focus_logits=focus_logits,
                attention_masks=attention_masks,
                use_relp_grad=use_relp_grad,
                disable_stop_grad=disable_stop_grad,
                center_logits=center_logits,
                alpha=alpha,
                neuron_chunk_size=neuron_chunk_size,
                verbose=verbose,
                instrumentation=instrumentation,
                contribution_target_lane_chunk_size=(
                    contribution_target_lane_chunk_size
                ),
                embed_contribution_target_lane_chunk_size=(
                    embed_contribution_target_lane_chunk_size
                ),
                contribution_execution_index=step,
            )
        )

        if neuron_tags is None:
            neuron_tags = tags

        # Immediately detach and move to CPU to save GPU memory
        attr_steps.append(attr.detach().cpu())
        contrib_steps.append(contrib.detach().cpu())
        embed_grad_contrib_steps.append(embed_grad_contrib.detach().cpu())
        neuron_acts_steps.append(neuron_acts.detach().cpu())
        embeds_steps.append(embeds.detach().cpu())

        # Clean up GPU tensors and computation graph
        del attr, contrib, embed_grad_contrib, neuron_acts, embeds
        torch.cuda.empty_cache()

        # Force garbage collection every few steps
        if step % 5 == 0:
            import gc

            gc.collect()
            torch.cuda.empty_cache()

    # Get device for moving tensors back
    device = input_ids.device

    if ig_mode == "ig-inputs":
        # Riemann sum in IG (ignore step 0)
        # Average gradients across steps (move back to device for computation)
        attr_grad_avg = (
            torch.stack(attr_steps[1:]).mean(dim=0).to(device)
        )  # (neurons, batch, src, d)
        contrib_grad_avg = (
            torch.stack(contrib_steps[1:]).mean(dim=0).to(device)
        )  # (neurons, batch, tgt)
        embed_grad_avg = (
            torch.stack(embed_grad_contrib_steps[1:]).mean(dim=0).to(device)
        )  # (t, batch, src, d)

        # Compute activation differences
        neuron_acts_diff = (neuron_acts_steps[-1] - neuron_acts_steps[0]).to(
            device
        )  # (neurons, batch)
        embeds_diff = (embeds_steps[-1] - embeds_steps[0]).to(device)  # (batch, seq, d)
        embeds_diff_src = embeds_diff[:, src_tokens, :]  # (batch, src, d)

        # Apply IG formula: averaged_gradient * (final_activation - initial_activation)
        # For attr: gradient is w.r.t. neuron acts projected to embeds
        # Shape: (neurons, batch, src, d) * (batch, src, d) -> sum over d -> (neurons, batch, src)
        attr_final = (attr_grad_avg * embeds_diff_src[None, :, :, :]).sum(dim=-1)

        # For contrib: gradient is w.r.t. neuron acts
        # Shape: (neurons, batch, tgt) * (neurons, batch, 1) -> (neurons, batch, tgt)
        contrib_final = contrib_grad_avg * neuron_acts_diff[:, :, None]

        # For embed_grad_contrib: gradient is w.r.t. embeds
        # Shape: (t, batch, src, d) * (batch, src, d) -> sum over d -> (t, batch, src) -> permute -> (src, batch, t)
        embed_grad_contrib_final = (
            embed_grad_avg * embeds_diff_src[None, :, :, :]
        ).sum(dim=-1)
        embed_grad_contrib_final = embed_grad_contrib_final.permute(
            2, 1, 0
        )  # (src, batch, t)

    elif ig_mode == "conductance":
        # Stack all steps (excluding step 0) and move to device
        attr_grads = torch.stack(attr_steps[1:]).to(
            device
        )  # (steps, neurons, batch, src, d)
        contrib_grads = torch.stack(contrib_steps[1:]).to(
            device
        )  # (steps, neurons, batch, tgt)
        embed_grads = torch.stack(embed_grad_contrib_steps[1:]).to(
            device
        )  # (steps, t, batch, src, d)

        # Compute step-wise differences in activations
        neuron_acts_all = torch.stack(neuron_acts_steps).to(
            device
        )  # (steps+1, neurons, batch)
        embeds_all = torch.stack(embeds_steps).to(device)  # (steps+1, batch, seq, d)

        neuron_acts_diffs = torch.diff(
            neuron_acts_all, dim=0
        )  # (steps, neurons, batch)
        embeds_diffs = torch.diff(embeds_all, dim=0)  # (steps, batch, seq, d)
        embeds_diffs_src = embeds_diffs[:, :, src_tokens, :]  # (steps, batch, src, d)

        # Apply conductance: sum over steps of (gradient * activation_diff)
        # For attr: (steps, neurons, batch, src, d) * (steps, batch, src, d) -> sum over d and steps
        attr_final = (
            (attr_grads * embeds_diffs_src[:, None, :, :, :]).sum(dim=-1).sum(dim=0)
        )

        # For contrib: (steps, neurons, batch, tgt) * (steps, neurons, batch, 1)
        contrib_final = (contrib_grads * neuron_acts_diffs[:, :, :, None]).sum(dim=0)

        # For embed_grad_contrib: (steps, t, batch, src, d) * (steps, batch, src, d)
        embed_grad_contrib_final = (
            (embed_grads * embeds_diffs_src[:, None, :, :, :]).sum(dim=-1).sum(dim=0)
        )
        embed_grad_contrib_final = embed_grad_contrib_final.permute(
            2, 1, 0
        )  # (src, batch, t)
    else:
        raise ValueError(f"Invalid IG mode: {ig_mode}")

    return attr_final, contrib_final, embed_grad_contrib_final, neuron_tags
