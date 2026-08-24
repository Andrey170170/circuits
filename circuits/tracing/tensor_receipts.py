"""Exact raw-dtype receipts for compact tracing tensors."""

from __future__ import annotations

import hashlib
import json

import torch


def raw_tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash dtype, shape, and exact compact tensor bytes without value casting."""

    value = tensor.detach().contiguous()
    header = json.dumps(
        {"dtype": str(value.dtype), "shape": list(value.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    # A size-one trailing dimension may remain "contiguous" with a non-unit
    # stride. Flatten first so dtype reinterpretation always has byte-viewable
    # storage without changing element order or values.
    digest.update(value.reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


__all__ = ["raw_tensor_sha256"]
