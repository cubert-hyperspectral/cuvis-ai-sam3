from __future__ import annotations

import pytest
import torch

from cuvis_ai_sam3.node.spectral_signature_extractor import SpectralSignatureExtractor


def test_trimmed_mean_signature_matches_expected_value() -> None:
    node = SpectralSignatureExtractor(trim_fraction=0.34, min_mask_pixels=1)
    cube = torch.tensor(
        [
            [
                [[1.0, 10.0], [2.0, 20.0]],
                [[100.0, 1000.0], [9.0, 90.0]],
            ]
        ],
        dtype=torch.float32,
    )
    mask = torch.tensor([[[1, 1], [1, 0]]], dtype=torch.int32)
    object_ids = torch.tensor([[1]], dtype=torch.int64)

    out = node.forward(cube=cube, mask=mask, object_ids=object_ids)
    signature = out["signatures"][0, 0]
    signature_std = out["signatures_std"][0, 0]

    assert signature.tolist() == pytest.approx([2.0, 20.0], rel=1e-5)
    assert signature_std.tolist() == pytest.approx([0.0, 0.0], abs=1e-6)


def test_zero_norm_pixels_are_filtered() -> None:
    node = SpectralSignatureExtractor(
        trim_fraction=0.0, min_mask_pixels=1, zero_norm_threshold=1e-8
    )
    cube = torch.tensor(
        [
            [
                [[0.0, 0.0], [3.0, 4.0]],
                [[7.0, 8.0], [9.0, 10.0]],
            ]
        ],
        dtype=torch.float32,
    )
    mask = torch.tensor([[[1, 1], [0, 0]]], dtype=torch.int32)
    object_ids = torch.tensor([[1]], dtype=torch.int64)

    out = node.forward(cube=cube, mask=mask, object_ids=object_ids)
    signature = out["signatures"][0, 0]

    assert signature.tolist() == pytest.approx([3.0, 4.0], rel=1e-5)


def test_mask_is_resized_when_resolution_differs() -> None:
    node = SpectralSignatureExtractor(trim_fraction=0.0, min_mask_pixels=1)
    cube = torch.tensor(
        [[[[1.0], [2.0]], [[3.0], [4.0]]]],
        dtype=torch.float32,
    )
    mask = torch.tensor([[[1]]], dtype=torch.int32)

    out = node.forward(cube=cube, mask=mask, object_ids=None)
    assert out["signatures"].shape == (1, 1, 1)
    assert out["signatures"][0, 0, 0].item() == pytest.approx(2.5, rel=1e-5)


def test_insufficient_pixels_returns_zero_signature() -> None:
    node = SpectralSignatureExtractor(trim_fraction=0.0, min_mask_pixels=2)
    cube = torch.tensor(
        [[[[5.0, 6.0], [7.0, 8.0]], [[9.0, 10.0], [11.0, 12.0]]]],
        dtype=torch.float32,
    )
    mask = torch.tensor([[[1, 0], [0, 0]]], dtype=torch.int32)
    object_ids = torch.tensor([[1]], dtype=torch.int64)

    out = node.forward(cube=cube, mask=mask, object_ids=object_ids)
    assert out["signatures"][0, 0].tolist() == pytest.approx([0.0, 0.0], abs=1e-6)
