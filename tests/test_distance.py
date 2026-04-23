"""Numerical tests for the distance helpers used by CLANLoss."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from nids.training.distance import (
    chunked_centroid_sims,
    cosdist,
    cosdist_inference,
    cossim,
    edist,
    esim,
)


def test_cosdist_self_distance_is_zero_on_diagonal():
    a = torch.randn(8, 16)
    d = cosdist(a)
    diag = torch.diag(d)
    assert torch.allclose(diag, torch.zeros_like(diag), atol=1e-5)


def test_cosdist_range_is_0_1():
    a = torch.randn(16, 8)
    b = torch.randn(16, 8)
    d = cosdist(a, b)
    assert (d >= 0).all()
    assert (d <= 1.0 + 1e-5).all()


def test_cosdist_symmetric():
    a = torch.randn(8, 4)
    b = torch.randn(8, 4)
    d_ab = cosdist(a, b)
    d_ba = cosdist(b, a)
    assert torch.allclose(d_ab, d_ba.T, atol=1e-5)


def test_cossim_inverse_of_cosdist():
    a = torch.randn(6, 4)
    b = torch.randn(6, 4)
    assert torch.allclose(cossim(a, b) + cosdist(a, b), torch.ones(6, 6), atol=1e-5)


def test_edist_self_distance_zero_on_diagonal():
    a = torch.randn(8, 4)
    d = edist(a)
    diag = torch.diag(d)
    assert torch.allclose(diag, torch.zeros_like(diag), atol=1e-5)


def test_esim_is_negative_edist():
    a = torch.randn(4, 3)
    b = torch.randn(4, 3)
    assert torch.allclose(esim(a, b), -edist(a, b), atol=1e-6)


def test_cosdist_inference_matches_manual():
    centroid = torch.randn(16)
    x = torch.randn(32, 16)
    out = cosdist_inference(centroid, x)
    manual = -1.0 * torch.nn.functional.cosine_similarity(
        centroid.unsqueeze(0), x, dim=1
    ).squeeze()
    assert torch.allclose(out, manual, atol=1e-6)


def test_chunked_centroid_sims_matches_oneshot():
    embeddings = torch.randn(512, 16)
    centroid = torch.randn(16)
    chunked = chunked_centroid_sims(embeddings, centroid, chunk_size=64)
    oneshot = chunked_centroid_sims(embeddings, centroid, chunk_size=None)
    import numpy as np
    np.testing.assert_allclose(chunked, oneshot, atol=1e-5)
