"""Pyramid round-trip and shape tests, against matlabPyrTools conventions.

``recon_lpyr(build_lpyr(x))`` must be an exact identity (the MATLAB Laplacian
pyramid is perfectly reconstructable), and ``max_pyr_ht`` must match
matlabPyrTools' recursive floor(log2/...) definition.
"""

from __future__ import annotations

import numpy as np
import pytest

from evm.pyramids import (
    BINOM5,
    blur_dn,
    blur_dn_clr,
    build_lpyr,
    laplacian_pyramid_channels,
    max_pyr_ht,
    recon_lpyr,
    reconstruct_from_channels,
)


@pytest.mark.parametrize(
    "imsz,filtsz,expected",
    [
        ((64, 64), 5, 4),
        ((528, 592), 5, 7),  # face.mp4
        ((960, 544), 5, 7),  # baby.mp4
        ((4, 4), 5, 0),
    ],
)
def test_max_pyr_ht(imsz, filtsz, expected) -> None:
    assert max_pyr_ht(imsz, filtsz) == expected


@pytest.mark.parametrize("h,w", [(64, 64), (96, 64), (45, 33)])
def test_lpyr_roundtrip_single_channel(h: int, w: int) -> None:
    """The Laplacian pyramid is an exact identity, even for odd sizes."""
    rng = np.random.default_rng(0)
    img = rng.random((h, w))
    pyr, pind = build_lpyr(img, "auto")
    rec = recon_lpyr(pyr, pind)
    assert rec.shape == img.shape
    assert np.abs(rec - img).max() < 1e-9


def test_lpyr_auto_levels_match_matlab() -> None:
    """Auto height == 1 + max_pyr_ht for the binom5 filter (size 5)."""
    img = np.zeros((64, 64))
    _, pind = build_lpyr(img, "auto")
    assert pind.shape[0] == 1 + max_pyr_ht((64, 64), 5)


def test_lpyr_explicit_height() -> None:
    img = np.random.default_rng(1).random((32, 32))
    pyr, pind = build_lpyr(img, 3)
    assert pind.shape[0] == 3
    rec = recon_lpyr(pyr, pind)
    assert np.abs(rec - img).max() < 1e-9


def test_lpyr_channels_roundtrip() -> None:
    rng = np.random.default_rng(2)
    frame = rng.random((48, 48, 3))
    bands, pind = laplacian_pyramid_channels(frame, "auto")
    rec = reconstruct_from_channels(bands, pind)
    assert rec.shape == frame.shape
    assert np.abs(rec - frame).max() < 1e-9


@pytest.mark.parametrize("nlevs", [1, 2, 4])
def test_blur_dn_halves(nlevs: int) -> None:
    img = np.random.default_rng(3).random((64, 64))
    out = blur_dn(img, nlevs)
    assert out.shape == (64 // (2 ** nlevs), 64 // (2 ** nlevs))


def test_blur_dn_clr_preserves_channels() -> None:
    frame = np.random.default_rng(4).random((64, 64, 3))
    out = blur_dn_clr(frame, 4)
    assert out.shape == (4, 4, 3)
