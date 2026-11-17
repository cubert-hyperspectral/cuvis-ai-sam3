"""The decoder's attention must not depend on the flash kernel alone.

PyTorch's Windows wheels ship no flash scaled-dot-product kernel, so a flash-only context fails
there with "No available kernel" as soon as the SAM 3.1 tracker attends to its memory.
"""

from __future__ import annotations

import contextlib

import pytest
import torch
from torch.nn.attention import SDPBackend

import sam3.model.decoder as decoder

pytestmark = pytest.mark.unit


def test_functional_attention_allows_the_kernels_every_build_has(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested: list[list[SDPBackend]] = []

    def record(backends, *_args, **_kwargs):  # noqa: ANN001, ANN202
        requested.append(list(backends) if isinstance(backends, (list, tuple)) else [backends])
        return contextlib.nullcontext()

    monkeypatch.setattr(decoder, "sdpa_kernel", record)
    q = torch.rand(1, 4, 8)

    out = decoder.functional_attention(q, q, q, dropout=0.0, num_heads=2, rope_k_repeat=False)

    assert out.shape == (1, 4, 8)
    (backends,) = requested
    assert backends[0] == SDPBackend.FLASH_ATTENTION  # still the first choice where it exists
    assert SDPBackend.EFFICIENT_ATTENTION in backends
    assert SDPBackend.MATH in backends
