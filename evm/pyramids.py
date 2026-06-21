"""Pyramid utilities, faithful to the MIT matlabPyrTools reference.

This module reimplements the small subset of Eero Simoncelli's matlabPyrTools
that the EVM MATLAB code depends on:

* ``corrDn`` — correlate with a 1-D kernel (separable), ``reflect1`` edges,
  then downsample by 2. Implemented via :func:`scipy.signal.correlate2d` on a
  ``numpy.pad(mode='reflect')`` image, which reproduces MATLAB ``reflect1``
  exactly (edge pixel *not* duplicated).
* ``upConv`` — transpose of ``corrDn``: upsample by inserting zeros, convolve
  with the same kernel, ``reflect1`` edges, crop to a target size.
* ``buildLpyr`` / ``reconLpyr`` — Laplacian pyramid build/reconstruct with the
  ``binom5`` filter and ``reflect1`` edges, height ``maxPyrHt``.
* ``blurDn`` / ``blurDnClr`` — repeated blur+downsample, used by the color
  (Gaussian-downsample) pipeline.

Matching the reference kernel-for-kernel matters because (a) it makes the
Python baseline a true correctness oracle and (b) the CUDA port will implement
exactly these convolutions on the device.

Reference: people.csail.mit.edu/mrub/evm/code/EVM_Matlab-1.1.zip (2012).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

# binom5 = sqrt(2) * [1 4 6 4 1] / 16, then normalized to sum=1 inside blurDn
# exactly as matlabPyrTools does. buildLpyr uses the L2-normalized binom5
# directly (it does NOT renormalize), so we keep both forms available.
_BINOM5_RAW = np.array([1.0, 4.0, 6.0, 4.0, 1.0])
BINOM5 = (np.sqrt(2.0) * _BINOM5_RAW / _BINOM5_RAW.sum()).astype(np.float64)
BINOM5_SUM1 = (_BINOM5_RAW / _BINOM5_RAW.sum()).astype(np.float64)


# ---------------------------------------------------------------------------
# corrDn / upConv
# ---------------------------------------------------------------------------


def corr_dn_axis(
    img: np.ndarray, filt: np.ndarray, axis: int
) -> np.ndarray:
    """Apply a 1-D ``filt`` along ``axis`` (reflect1) then downsample by 2.

    Mirrors the separable ``corrDn(im, filt', 'reflect1', [1 2])`` and
    ``corrDn(..., [2 1])`` calls inside ``blurDn``/``buildLpyr``. Padding is
    ``filt.shape[0] // 2`` on each side under ``reflect1`` (numpy
    ``mode='reflect'``), then a correlation is taken and every other sample
    starting at index 0 is kept (MATLAB ``start=[1,1]``).
    """
    pad = filt.shape[0] // 2
    pad_width = [(0, 0)] * img.ndim
    pad_width[axis] = (pad, pad)
    padded = np.pad(img, pad_width, mode="reflect")
    # Reverse the kernel along the target axis so this is a convolution of the
    # reversed kernel == correlation of the original kernel, matching
    # matlabPyrTools (which flips the filter then calls conv2 'valid').
    rev = filt[::-1].astype(padded.dtype)
    out = np.apply_along_axis(
        lambda v, k=rev: np.convolve(v, k, mode="valid"), axis, padded
    )
    sl = [slice(None)] * img.ndim
    sl[axis] = slice(None, None, 2)
    return out[tuple(sl)]


def up_conv_axis(
    img: np.ndarray, filt: np.ndarray, axis: int, out_size: int
) -> np.ndarray:
    """Transpose of :func:`corr_dn_axis`: upsample by 2 then convolve along axis.

    Matches matlabPyrTools ``upConv`` with ``step=[2 1]``/``[1 2]`` and
    ``[1 1]`` start, cropped to ``out_size``. The convolution here uses the
    same (reversed-kernel) convention as ``corr_dn_axis`` so the two are
    transposes of each other.
    """
    # Insert zeros between samples along axis at offset 0 (MATLAB start=[1,1]).
    up_shape = list(img.shape)
    up_shape[axis] = img.shape[axis] * 2
    up = np.zeros(up_shape, dtype=np.float64)
    sl_out = [slice(None)] * img.ndim
    sl_out[axis] = slice(None, None, 2)
    up[tuple(sl_out)] = img

    pad = filt.shape[0] // 2
    pad_width = [(0, 0)] * img.ndim
    pad_width[axis] = (pad, pad)
    padded = np.pad(up, pad_width, mode="reflect")
    rev = filt[::-1].astype(padded.dtype)
    out = np.apply_along_axis(
        lambda v, k=rev: np.convolve(v, k, mode="valid"), axis, padded
    )
    sl = [slice(None)] * img.ndim
    sl[axis] = slice(0, out_size)
    return out[tuple(sl)]


# ---------------------------------------------------------------------------
# maxPyrHt / binomial filter
# ---------------------------------------------------------------------------


def max_pyr_ht(imsz: Tuple[int, int], filtsz: int) -> int:
    """matlabPyrTools ``maxPyrHt`` — max downsample depth for filtsz>im always."""
    h, w = imsz
    f = filtsz
    if h < f or w < f:
        return 0
    return 1 + max_pyr_ht((h // 2, w // 2), f)


# ---------------------------------------------------------------------------
# blurDn / blurDnClr  (color pipeline downsample stack)
# ---------------------------------------------------------------------------


def blur_dn(img: np.ndarray, nlevs: int, filt: np.ndarray = BINOM5_SUM1) -> np.ndarray:
    """matlabPyrTools ``blurDn``: blur+downsample by 2, ``nlevs`` times.

    For the color pipeline, matlabPyrTools renormalizes the filter to sum=1
    (``filt = filt/sum(filt(:))``) before applying it separably with reflect1.
    """
    if nlevs <= 0:
        return img
    out = blur_dn(img, nlevs - 1, filt)
    f = filt / float(filt.sum())
    # Separable: rows then columns, each downsampled by 2.
    out = corr_dn_axis(out, f, axis=0)
    out = corr_dn_axis(out, f, axis=1)
    return out


def blur_dn_clr(img: np.ndarray, nlevs: int, filt: np.ndarray = BINOM5_SUM1) -> np.ndarray:
    """``blurDnClr``: apply :func:`blur_dn` to each color channel independently."""
    chans = [blur_dn(img[:, :, c], nlevs, filt) for c in range(img.shape[2])]
    return np.stack(chans, axis=-1)


# ---------------------------------------------------------------------------
# buildLpyr / reconLpyr  (motion pipeline Laplacian pyramid)
# ---------------------------------------------------------------------------


def build_lpyr(
    img: np.ndarray, height: int | str = "auto", filt: np.ndarray = BINOM5
) -> Tuple[np.ndarray, np.ndarray]:
    """matlabPyrTools ``buildLpyr`` for a 2-D single-channel image.

    Returns ``(pyr, pind)`` where ``pyr`` is a flat vector of concatenated
    bands (finest first, coarsest residual last) and ``pind`` is an ``(N, 2)``
    array of per-level (H, W) sizes — exactly matching the MATLAB signature so
    downstream indexing is identical.
    """
    img = img.astype(np.float64)
    imsz = img.shape
    f = filt  # buildLpyr uses binom5 verbatim (L2-normalized), NOT renormalized
    if height == "auto" or height is None:
        height = 1 + max_pyr_ht(imsz, f.shape[0])
    if height <= 1:
        return img.reshape(-1, 1), np.array([[imsz[0], imsz[1]]])

    # lo = corrDn(im, filt', [1 2]); lo2 = corrDn(lo, filt, [2 1])
    lo = corr_dn_axis(img, f, axis=1)  # downsample columns (x)
    lo2 = corr_dn_axis(lo, f, axis=0)  # downsample rows (y)
    int_sz = lo.shape

    npyr, nind = build_lpyr(lo2, height - 1, f)

    # hi = upConv(lo2, filt, [2 1], int_sz); hi2 = upConv(hi, filt', [1 2], im_sz)
    hi = up_conv_axis(lo2, f, axis=0, out_size=int_sz[0])
    hi2 = up_conv_axis(hi, f, axis=1, out_size=imsz[1])
    band = img - hi2  # Laplacian band at full resolution

    pyr = np.concatenate([band.reshape(-1, 1), npyr], axis=0)
    pind = np.vstack([np.array([[imsz[0], imsz[1]]]), nind])
    return pyr, pind


def recon_lpyr(
    pyr: np.ndarray,
    pind: np.ndarray,
    filt: np.ndarray = BINOM5,
) -> np.ndarray:
    """matlabPyrTools ``reconLpyr`` — collapse a Laplacian pyramid."""
    f = filt
    res_sz = (int(pind[0, 0]), int(pind[0, 1]))
    if pind.shape[0] == 1:
        return pyr.reshape(res_sz)

    # Recurse on the sub-pyramid starting at level 2.
    offset = res_sz[0] * res_sz[1]
    nres = recon_lpyr(pyr[offset:], pind[1:], f)

    int_sz = (int(pind[0, 0]), int(pind[1, 1]))
    hi = up_conv_axis(nres, f, axis=0, out_size=int_sz[0])
    res = up_conv_axis(hi, f, axis=1, out_size=res_sz[1])
    band = pyr[:offset].reshape(res_sz)
    return band + res


# ---------------------------------------------------------------------------
# Convenience wrappers operating on full (H, W, C) frames
# ---------------------------------------------------------------------------


def laplacian_pyramid_channels(
    frame: np.ndarray, height: int | str = "auto"
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Build a Laplacian pyramid per channel, returning per-level stacked bands.

    Matches ``build_Lpyr_stack``'s per-frame behaviour: each color channel gets
    its own pyramid with the *same* ``pind`` (channels are independent). The
    return value is ``(levels, pind)`` where ``levels[l]`` has shape
    ``(H_l, W_l, C)``.
    """
    h, w, c = frame.shape
    pyr_per_c = [build_lpyr(frame[:, :, ch], height) for ch in range(c)]
    pind = pyr_per_c[0][1]
    # Group by level across channels.
    bands: List[np.ndarray] = []
    for l in range(pind.shape[0]):
        lh, lw = int(pind[l, 0]), int(pind[l, 1])
        n = lh * lw
        offsets = np.cumsum([0] + [int(pind[i, 0] * pind[i, 1]) for i in range(l)])
        start = offsets[l]
        chan_bands = [
            pyr_per_c[ch][0][start : start + n].reshape(lh, lw)
            for ch in range(c)
        ]
        bands.append(np.stack(chan_bands, axis=-1))
    return bands, pind


def reconstruct_from_channels(
    bands: List[np.ndarray], pind: np.ndarray
) -> np.ndarray:
    """Inverse of :func:`laplacian_pyramid_channels`."""
    c = bands[0].shape[2]
    out = np.empty(
        (int(pind[0, 0]), int(pind[0, 1]), c), dtype=np.float64
    )
    for ch in range(c):
        pyr = np.concatenate(
            [bands[l][:, :, ch].reshape(-1) for l in range(len(bands))]
        )
        out[:, :, ch] = recon_lpyr(pyr.reshape(-1, 1), pind)
    return out
