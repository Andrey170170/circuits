"""
Code for constructing the replacement model with various modifications to the original backward pass
in order to improve gradient-based attribution techniques.

Supports multiple model architectures (Llama, Qwen3) with a shared interface.
"""

import torch
from torch import nn
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from circuits.tracing.grad.llama import (
    LlamaAttention,
    LlamaMLP,
    LlamaRMSNorm,
    repeat_kv,
)
from circuits.tracing.grad.qwen3 import Qwen3Attention, Qwen3MLP, Qwen3RMSNorm

# Map from HF module types to their "kind" for dispatching
_NORM_TYPES: tuple[type[nn.Module], ...] = (LlamaRMSNorm, Qwen3RMSNorm)
_ATTN_TYPES: tuple[type[nn.Module], ...] = (LlamaAttention, Qwen3Attention)
_MLP_TYPES: tuple[type[nn.Module], ...] = (LlamaMLP, Qwen3MLP)
_STOP_GRAD_STATE_ATTRIBUTE = "_adag_stop_gradient_model_state"


def _rms_layernorm_fn(
    x_X1X2D: torch.Tensor,
    estimator_X1D: torch.Tensor,
    norm_w_D: torch.Tensor,
    eps: float,
):
    """
    Normalizes x along the X1/X2 dimensions by computing RMS statistics across the D dimension of
    estimator_X1D, then applying the same normalization to constant to X2D for all X1.

    We cast to float32 for numeric stability.
    """
    device = x_X1X2D.device
    return (
        norm_w_D[None, None, :].to(device)
        * x_X1X2D
        * torch.rsqrt(estimator_X1D.to(device).pow(2).mean(dim=1) + eps)[:, None, None]
    )


def remove_forward_hooks(main_module: nn.Module):
    """Remove all forward and pre-forward hooks from a module and its sub-modules."""
    for _, submodule in main_module.named_modules():
        if hasattr(submodule, "_forward_hooks"):
            hooks = list(submodule._forward_hooks.keys())
            for hook_id in hooks:
                submodule._forward_hooks.pop(hook_id)
        if hasattr(submodule, "_forward_pre_hooks"):
            pre_hooks = list(submodule._forward_pre_hooks.keys())
            for pre_hook_id in pre_hooks:
                submodule._forward_pre_hooks.pop(pre_hook_id)


class StopGradientModule(nn.Module):
    _stop_gradient = True


class StraightThroughRMSNorm(StopGradientModule):
    """
    Wrap an existing RMSNorm so that

      forward  = real RMSNorm value
      backward = identity wrt input  (dout/dx = I)
      weight   is frozen (requires_grad = False)

    Works with any RMSNorm module that has .weight and .variance_epsilon attributes.
    """

    def __init__(self, norm: nn.Module):
        super().__init__()
        self.norm = norm
        self.norm.weight.requires_grad_(False)
        self.weight = self.norm.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        coeff = _rms_layernorm_fn(
            x.new_ones(B * L, 1, 1),
            x.view(B * L, D),
            self.norm.weight,
            self.norm.variance_epsilon,
        ).detach()
        return x * coeff.permute(1, 0, 2).view(B, L, D)


def noqk_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    """Attention forward that detaches attention weights so gradient only flows through OV."""
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_scores = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_scores = attn_scores + causal_mask

    attn_weights = (
        nn.functional.softmax(attn_scores, dim=-1, dtype=torch.float32)
        .to(query.dtype)
        .detach()
    )

    attn_weights = nn.functional.dropout(
        attn_weights, p=dropout, training=module.training
    )
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


ALL_ATTENTION_FUNCTIONS["noqk"] = noqk_attention_forward


class NoQKGradAttention(StopGradientModule):
    """
    Wraps an existing attention module so that the soft-maxed attention
    map gets no gradient. Gradient only flows through the OV path.
    """

    def __init__(self, attn: nn.Module):
        super().__init__()
        self.attn = attn
        self.q_proj = attn.q_proj
        self.k_proj = attn.k_proj
        self.v_proj = attn.v_proj
        self.o_proj = attn.o_proj

    def forward(self, *args, **kwargs):
        attn_output, attn_weights = self.attn(*args, **kwargs)
        return attn_output, attn_weights


class StopGradGateMLP(StopGradientModule):
    """
    Wrap an existing gated MLP so the activation-gate side
      act_fn( gate_proj(x) )
    is detached from the autograd graph.
    """

    def __init__(self, mlp: nn.Module):
        super().__init__()
        self.mlp = mlp
        for p in self.mlp.gate_proj.parameters():
            p.requires_grad_(False)
        self.down_proj = self.mlp.down_proj
        self.act_fn = self.mlp.act_fn
        self.gate_proj = self.mlp.gate_proj
        self.up_proj = self.mlp.up_proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_act = self.mlp.act_fn(self.mlp.gate_proj(x)).detach()
        up_branch = self.mlp.up_proj(x)
        return self.mlp.down_proj(gate_act * up_branch)


class StopGradMLP(StopGradientModule):
    """
    Wrap an existing MLP and stop *all* gradient through it.
    """

    def __init__(self, mlp: nn.Module):
        super().__init__()
        self.mlp = mlp
        for p in self.mlp.parameters():
            p.requires_grad_(False)
        self.down_proj = self.mlp.down_proj
        self.act_fn = self.mlp.act_fn
        self.gate_proj = self.mlp.gate_proj
        self.up_proj = self.mlp.up_proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = self.mlp(x)
        return out.detach()


class ShapleyElementwiseMult(torch.autograd.Function):
    """
    Shapley gradient for elementwise multiplication. This distributes the attribution equally to
    both branches, avoiding double-counting (which normal gradient would do).
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, y: torch.Tensor, use_half_rule: bool = True):
        ctx.save_for_backward(x, y)
        ctx.use_half_rule = use_half_rule
        return x * y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, y = ctx.saved_tensors
        return (
            (0.5 if ctx.use_half_rule else 1.0) * grad_output * y,
            (0.5 if ctx.use_half_rule else 1.0) * grad_output * x,
            None,
        )


class RelPGradMLP(StopGradientModule):
    """
    RelP gradient for a gated MLP. Linearizes the activation gate by detaching it as a constant,
    then uses Shapley halving for the gate*up elementwise multiplication.
    """

    def __init__(self, mlp: nn.Module, use_half_rule: bool = True):
        super().__init__()
        self.mlp = mlp
        self.down_proj = self.mlp.down_proj
        self.gate_proj = self.mlp.gate_proj
        self.up_proj = self.mlp.up_proj
        self.use_half_rule = use_half_rule

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_proj = self.mlp.gate_proj(x)
        # Linearize: treat act_fn(z)/z as a detached constant so gradient flows only
        # through the linear gate_proj. For SiLU this equals sigmoid(z).
        coeff = (self.mlp.act_fn(gate_proj) / (gate_proj + 1e-10)).detach()
        gate_act = gate_proj * coeff
        up_branch = self.mlp.up_proj(x)
        return self.mlp.down_proj(
            ShapleyElementwiseMult.apply(gate_act, up_branch, self.use_half_rule)
        )


def _valid_layer_scope(model, layer_indices) -> tuple[int, ...]:
    layer_count = len(model.model.layers)
    return tuple(
        sorted(
            {
                int(layer_index)
                for layer_index in layer_indices
                if 0 <= int(layer_index) < layer_count
            }
        )
    )


def _layerwise_scope(model, start_layer: int, end_layer: int) -> tuple[int, ...]:
    candidates = [start_layer, end_layer]
    if start_layer < end_layer:
        candidates.extend(range(start_layer + 1, end_layer))
    return _valid_layer_scope(model, candidates)


def _begin_stop_gradient_state(
    model, *, operation: str, scope: tuple[int, ...]
) -> None:
    """Snapshot exact modules and mutable state before any replacement."""

    if hasattr(model, _STOP_GRAD_STATE_ATTRIBUTE):
        raise RuntimeError("stop-gradient model state is already active")
    layer_modules = {
        layer_index: {
            "input_layernorm": model.model.layers[layer_index].input_layernorm,
            "post_attention_layernorm": model.model.layers[
                layer_index
            ].post_attention_layernorm,
            "self_attn": model.model.layers[layer_index].self_attn,
            "mlp": model.model.layers[layer_index].mlp,
        }
        for layer_index in scope
    }
    affected_modules = [model.model.norm]
    for modules in layer_modules.values():
        affected_modules.extend(modules.values())

    parameters = []
    seen_parameters: set[int] = set()
    for module in affected_modules:
        for parameter in module.parameters():
            if id(parameter) in seen_parameters:
                continue
            seen_parameters.add(id(parameter))
            parameters.append((parameter, bool(parameter.requires_grad)))

    attention_configs = []
    seen_configs: set[int] = set()
    for modules in layer_modules.values():
        config = modules["self_attn"].config
        if id(config) in seen_configs:
            continue
        seen_configs.add(id(config))
        attention_configs.append(
            (
                config,
                hasattr(config, "_attn_implementation"),
                getattr(config, "_attn_implementation", None),
            )
        )
    setattr(
        model,
        _STOP_GRAD_STATE_ATTRIBUTE,
        {
            "operation": operation,
            "scope": scope,
            "global_norm": model.model.norm,
            "layer_modules": layer_modules,
            "attention_configs": attention_configs,
            "parameters": parameters,
        },
    )
    try:
        for config, _had_attribute, _original_value in attention_configs:
            config._attn_implementation = "noqk"
    except BaseException:
        _restore_stop_gradient_state(
            model, expected_operation=operation, expected_scope=scope
        )
        raise


def _restore_stop_gradient_state(
    model,
    *,
    expected_operation: str,
    expected_scope: tuple[int, ...],
) -> bool:
    """Restore an exact stop-gradient transaction, or no-op when clean."""

    state = getattr(model, _STOP_GRAD_STATE_ATTRIBUTE, None)
    if state is None:
        return False
    if state["operation"] != expected_operation or state["scope"] != expected_scope:
        raise RuntimeError(
            "stop-gradient revert does not match active operation/scope: "
            f"active={state['operation']}:{state['scope']}, "
            f"requested={expected_operation}:{expected_scope}"
        )

    model.model.norm = state["global_norm"]
    for layer_index, modules in state["layer_modules"].items():
        layer = model.model.layers[layer_index]
        layer.input_layernorm = modules["input_layernorm"]
        layer.post_attention_layernorm = modules["post_attention_layernorm"]
        layer.self_attn = modules["self_attn"]
        layer.mlp = modules["mlp"]
    for config, had_attribute, original_value in state["attention_configs"]:
        if had_attribute:
            config._attn_implementation = original_value
        elif hasattr(config, "_attn_implementation"):
            delattr(config, "_attn_implementation")
    for parameter, original_requires_grad in state["parameters"]:
        parameter.requires_grad_(original_requires_grad)
    delattr(model, _STOP_GRAD_STATE_ATTRIBUTE)
    return True


def stop_nonlinear_grad(
    model,
    use_relp_grad: bool = False,
    use_half_rule: bool = True,
):
    """
    Stop gradient for all non-linear layers in the model.

    - LayerNorms: linearized via StraightThroughRMSNorm (detached coefficients)
    - Attention: QK path detached, gradient flows only through OV
    - MLP: if use_relp_grad, activation gate is detached and Shapley halving applied;
           otherwise, entire gate branch is detached
    """
    scope = _valid_layer_scope(model, range(len(model.model.layers)))
    _begin_stop_gradient_state(model, operation="global", scope=scope)
    try:
        model.model.norm = StraightThroughRMSNorm(model.model.norm)
        for layer in scope:
            model.model.layers[layer].input_layernorm = StraightThroughRMSNorm(
                model.model.layers[layer].input_layernorm
            )
            model.model.layers[layer].post_attention_layernorm = StraightThroughRMSNorm(
                model.model.layers[layer].post_attention_layernorm
            )
            model.model.layers[layer].self_attn = NoQKGradAttention(
                model.model.layers[layer].self_attn
            )
            if use_relp_grad:
                model.model.layers[layer].mlp = RelPGradMLP(
                    model.model.layers[layer].mlp, use_half_rule
                )
            else:
                model.model.layers[layer].mlp = StopGradGateMLP(
                    model.model.layers[layer].mlp
                )
    except BaseException:
        _restore_stop_gradient_state(
            model, expected_operation="global", expected_scope=scope
        )
        raise
    return model


def revert_stop_nonlinear_grad(model):
    """
    Revert stop gradient for all non-linear layers in the model.
    """
    scope = _valid_layer_scope(model, range(len(model.model.layers)))
    _restore_stop_gradient_state(
        model, expected_operation="global", expected_scope=scope
    )
    return model


def layerwise_stop_nonlinear_grad(
    model,
    start_layer: int,
    end_layer: int,
    use_relp_grad: bool = False,
    use_stop_grad_on_mlps: bool = True,
    use_half_rule: bool = True,
):
    scope = _layerwise_scope(model, start_layer, end_layer)
    _begin_stop_gradient_state(model, operation="layerwise", scope=scope)
    boundary_layers = {layer for layer in (start_layer, end_layer) if layer in scope}
    interior_layers = [layer for layer in scope if layer not in boundary_layers]
    try:
        model.model.norm = StraightThroughRMSNorm(model.model.norm)
        # Boundaries use the ordinary gated-MLP rule and are wrapped once even
        # when start_layer == end_layer.
        for layer in sorted(boundary_layers):
            model.model.layers[layer].input_layernorm = StraightThroughRMSNorm(
                model.model.layers[layer].input_layernorm
            )
            model.model.layers[layer].post_attention_layernorm = StraightThroughRMSNorm(
                model.model.layers[layer].post_attention_layernorm
            )
            model.model.layers[layer].self_attn = NoQKGradAttention(
                model.model.layers[layer].self_attn
            )
            if use_relp_grad:
                model.model.layers[layer].mlp = RelPGradMLP(
                    model.model.layers[layer].mlp, use_half_rule
                )
            else:
                model.model.layers[layer].mlp = StopGradGateMLP(
                    model.model.layers[layer].mlp
                )

        for layer in interior_layers:
            model.model.layers[layer].input_layernorm = StraightThroughRMSNorm(
                model.model.layers[layer].input_layernorm
            )
            model.model.layers[layer].post_attention_layernorm = StraightThroughRMSNorm(
                model.model.layers[layer].post_attention_layernorm
            )
            model.model.layers[layer].self_attn = NoQKGradAttention(
                model.model.layers[layer].self_attn
            )
            model.model.layers[layer].mlp = StopGradMLP(model.model.layers[layer].mlp)
    except BaseException:
        _restore_stop_gradient_state(
            model, expected_operation="layerwise", expected_scope=scope
        )
        raise

    return model


def layerwise_revert_stop_nonlinear_grad(
    model,
    start_layer: int,
    end_layer: int,
):
    scope = _layerwise_scope(model, start_layer, end_layer)
    _restore_stop_gradient_state(
        model, expected_operation="layerwise", expected_scope=scope
    )
    return model


# Backward-compatible aliases
StraightThroughLlamaRMSNorm = StraightThroughRMSNorm
stop_nonlinear_grad_for_llama = stop_nonlinear_grad
revert_stop_nonlinear_grad_for_llama = revert_stop_nonlinear_grad
layerwise_stop_nonlinear_grad_for_llama = layerwise_stop_nonlinear_grad
layerwise_revert_stop_nonlinear_grad_for_llama = layerwise_revert_stop_nonlinear_grad
