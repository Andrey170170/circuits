"""Provenance-bound CUDA allocator policies for tracing qualification runs."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, MutableMapping
from typing import Any

POLICY_CONFIG_FIELD = "cuda_allocator_policy"
ALLOCATOR_ENVIRONMENT_VARIABLE = "PYTORCH_CUDA_ALLOC_CONF"

_POLICY_ENVIRONMENT_VALUES: dict[str, str | None] = {
    "default_v1": None,
    "expandable_segments_v1": "expandable_segments:True",
}


def declared_cuda_allocator_policy(config: Mapping[str, Any]) -> str | None:
    """Return and validate the optional stable allocator policy identifier."""

    if POLICY_CONFIG_FIELD not in config:
        return None
    policy_id = config[POLICY_CONFIG_FIELD]
    if not isinstance(policy_id, str) or policy_id not in _POLICY_ENVIRONMENT_VALUES:
        supported = ", ".join(sorted(_POLICY_ENVIRONMENT_VALUES))
        raise ValueError(
            f"run config {POLICY_CONFIG_FIELD} must be one of: {supported}"
        )
    return policy_id


def pytorch_cuda_alloc_conf_for_policy(policy_id: str) -> str | None:
    """Resolve a validated policy identifier to its exact PyTorch environment."""

    if policy_id not in _POLICY_ENVIRONMENT_VALUES:
        supported = ", ".join(sorted(_POLICY_ENVIRONMENT_VALUES))
        raise ValueError(f"CUDA allocator policy must be one of: {supported}")
    return _POLICY_ENVIRONMENT_VALUES[policy_id]


def bind_cuda_allocator_runtime_receipt(
    config: Mapping[str, Any],
    runtime_environment: Mapping[str, Any],
    *,
    environ: Mapping[str, str] | None = None,
    allocator_backend: Callable[[], str] | None = None,
) -> dict[str, Any]:
    """Validate an explicit policy and attach its observed runtime receipt.

    Legacy configs without ``cuda_allocator_policy`` are returned without a
    receipt so their historical artifact identities remain stable.
    """

    policy_id = validate_cuda_allocator_environment(config, environ=environ)
    runtime = dict(runtime_environment)
    if policy_id is None:
        return runtime

    environment = os.environ if environ is None else environ
    observed_value = environment.get(ALLOCATOR_ENVIRONMENT_VARIABLE)
    if allocator_backend is None:
        import torch

        get_backend = torch.cuda.memory.get_allocator_backend
    else:
        get_backend = allocator_backend
    observed_backend = get_backend()
    if not isinstance(observed_backend, str) or not observed_backend:
        raise RuntimeError("PyTorch CUDA allocator backend receipt is unavailable")

    receipt = {
        "intended_policy_id": policy_id,
        "observed_environment": {
            "name": ALLOCATOR_ENVIRONMENT_VARIABLE,
            "value": observed_value,
            "is_set": ALLOCATOR_ENVIRONMENT_VARIABLE in environment,
        },
        "observed_allocator_backend": observed_backend,
    }
    if (
        "cuda_allocator_policy" in runtime
        and runtime["cuda_allocator_policy"] != receipt
    ):
        raise RuntimeError(
            "runtime environment contains a conflicting CUDA allocator receipt"
        )
    runtime["cuda_allocator_policy"] = receipt
    return runtime


def validate_cuda_allocator_environment(
    config: Mapping[str, Any],
    *,
    environ: Mapping[str, str] | None = None,
) -> str | None:
    """Fail before CUDA inspection when the declared policy and env disagree."""

    policy_id = declared_cuda_allocator_policy(config)
    if policy_id is None:
        return None
    environment = os.environ if environ is None else environ
    expected_value = pytorch_cuda_alloc_conf_for_policy(policy_id)
    observed_value = environment.get(ALLOCATOR_ENVIRONMENT_VARIABLE)
    if observed_value != expected_value:
        raise RuntimeError(
            f"{ALLOCATOR_ENVIRONMENT_VARIABLE} disagrees with declared "
            f"{POLICY_CONFIG_FIELD}={policy_id!r}: expected "
            f"{expected_value!r}, observed {observed_value!r}"
        )
    return policy_id


def apply_cuda_allocator_policy_to_environment(
    policy_id: str,
    environ: MutableMapping[str, str] | None = None,
) -> None:
    """Apply an exact policy before a Python process imports PyTorch.

    This Torch-free helper describes the shell launcher's required mutation;
    the launcher performs the equivalent export/unset before starting Python.
    """

    environment = os.environ if environ is None else environ
    value = pytorch_cuda_alloc_conf_for_policy(policy_id)
    if value is None:
        environment.pop(ALLOCATOR_ENVIRONMENT_VARIABLE, None)
    else:
        environment[ALLOCATOR_ENVIRONMENT_VARIABLE] = value
