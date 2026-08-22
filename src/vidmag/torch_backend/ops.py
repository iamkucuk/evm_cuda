"""The PyTorch implementation of :class:`vidmag.backend.Ops`.

Every primitive here is written against the same definitions the NumPy
baseline uses, and is checked against it by the shared conformance suite. Two
details carry the whole port, and both are the places where implementations of
this method are known to diverge quietly:

**The border rule.** The reference mirrors at an edge without repeating the
edge sample — ``reflect1`` in the original MATLAB, ``mode="reflect"`` in NumPy.
PyTorch has a padding mode by that name and it follows the same rule, but it
refuses to pad by more than the array is long, which the coarsest pyramid
levels do need. So the padding here is done by gathering indices computed from
the rule itself. That works at any size and, being the rule rather than an
approximation of it, cannot drift from the reference.

**Correlation, not convolution.** The reference reverses the filter and then
convolves, which is a correlation. PyTorch's convolution functions already
compute correlation, so the filter is passed unreversed. The binomial filter
this project uses is symmetric, so the two agree either way; the distinction is
written down because a future filter might not be.

Arrays are ``torch.Tensor``. Shapes follow the protocol: frames are
``(T, H, W, 3)`` and temporal filters run along axis 0.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ..cpu.pyramids import BINOM5, BINOM5_SUM1
from ..io.video import rgb_to_yiq, yiq_to_rgb
from .runtime import pick_device
from .runtime import torch as _torch

__all__ = ["TorchOps"]

# Derived from the baseline's own conversion rather than retyped: applying it
# to the three basis vectors recovers exactly the matrix it uses, so these
# cannot drift from `vidmag.io.video` the way a copied literal would.
_RGB_TO_YIQ = np.asarray(rgb_to_yiq(np.eye(3, dtype=np.float64)), dtype=np.float64)
_YIQ_TO_RGB = np.asarray(yiq_to_rgb(np.eye(3, dtype=np.float64)), dtype=np.float64)


def _reflect_index(length: int, pad: int, torch: Any, device: Any) -> Any:
    """Indices for `pad` samples either side, mirrored without repeating edges.

    This is `reflect1` from the reference, as an index vector: position ``i``
    outside ``[0, length)`` maps to its mirror, the period being ``2*length-2``
    so the edge sample appears once rather than twice.
    """
    if length == 1:
        return torch.zeros(1 + 2 * pad, dtype=torch.long, device=device)
    period = 2 * length - 2
    i = torch.arange(-pad, length + pad, device=device)
    i = torch.abs(i)  # reflection is symmetric about zero
    i = torch.remainder(i, period)
    return torch.where(i >= length, period - i, i)


class TorchOps:
    """Every primitive the pipelines need, computed with PyTorch."""

    name = "torch"

    def __init__(self, device: str | None = None) -> None:
        self._torch = _torch()
        self.device = pick_device(device)
        # Single precision throughout, matching the other device backends. The
        # reference computes in double; the conformance suite's tolerance is
        # set for that difference.
        self.dtype = self._torch.float32

    # -- transfer -----------------------------------------------------------

    def from_numpy(self, array: np.ndarray) -> Any:
        # Double precision is narrowed to single on the way in. This is not a
        # convenience: Apple's GPU has no float64 at all and refuses the
        # transfer outright. The reference computes in double and every device
        # backend in this project computes in single, so the conformance
        # tolerance already covers the difference. Integer frames are left
        # alone — 8-bit input is 8-bit input.
        array = np.ascontiguousarray(array)
        if array.dtype == np.float64:
            array = array.astype(np.float32)
        return self._torch.from_numpy(array).to(self.device)

    def to_numpy(self, array: Any) -> np.ndarray:
        return array.detach().to("cpu").numpy()

    # -- the two spatial primitives everything else is built from -----------

    def _pad_reflect(self, x: Any, pad: int, dim: int) -> Any:
        torch = self._torch
        idx = _reflect_index(x.shape[dim], pad, torch, x.device)
        return torch.index_select(x, dim, idx)

    def _correlate(self, x: Any, filt: Any, dim: int) -> Any:
        """Correlate along `dim` with a 1-D filter, 'valid' after padding.

        The tensor is moved so the filtered axis is last, flattened to a batch
        of rows, and run through a 1-D convolution — which in PyTorch is a
        correlation, matching the reference.
        """
        torch = self._torch
        x = x.movedim(dim, -1)
        shape = x.shape
        rows = x.reshape(-1, 1, shape[-1])
        out = torch.nn.functional.conv1d(rows, filt.view(1, 1, -1))
        out = out.reshape(*shape[:-1], out.shape[-1])
        return out.movedim(-1, dim)

    def _corr_dn(self, x: Any, filt: Any, dim: int) -> Any:
        """Filter along `dim`, then keep every other sample from index 0."""
        pad = filt.shape[0] // 2
        out = self._correlate(self._pad_reflect(x, pad, dim), filt, dim)
        return out.index_select(
            dim,
            self._torch.arange(0, out.shape[dim], 2, device=out.device),
        )

    def _up_conv(self, x: Any, filt: Any, dim: int, out_size: int) -> Any:
        """Insert a zero between samples, filter, and crop to `out_size`.

        The transpose of :meth:`_corr_dn`, which is what makes building and
        reconstructing a pyramid inverse to one another.
        """
        torch = self._torch
        shape = list(x.shape)
        shape[dim] = x.shape[dim] * 2
        up = torch.zeros(shape, dtype=x.dtype, device=x.device)
        up.index_copy_(dim, torch.arange(0, shape[dim], 2, device=x.device), x)
        pad = filt.shape[0] // 2
        out = self._correlate(self._pad_reflect(up, pad, dim), filt, dim)
        return out.narrow(dim, 0, out_size)

    def _filt(self, values: np.ndarray) -> Any:
        return self._torch.tensor(values, dtype=self.dtype, device=self.device)

    # -- colour -------------------------------------------------------------

    def bgr_u8_to_ntsc(self, frames: Any) -> Any:
        """(T, H, W, 3) 8-bit blue-green-red to NTSC floats in [0, 1]."""
        rgb = frames.flip(-1).to(self.dtype) / 255.0
        m = self._torch.tensor(_RGB_TO_YIQ, dtype=self.dtype, device=frames.device)
        return rgb @ m

    def add_and_quantize(self, frames_ntsc: Any, delta: Any) -> Any:
        torch = self._torch
        m = torch.tensor(_YIQ_TO_RGB, dtype=self.dtype, device=frames_ntsc.device)
        rgb = ((frames_ntsc + delta) @ m).clamp(0.0, 1.0)
        # The reference rounds half away from zero; torch.round is half-to-even,
        # which differs on exact .5 and would show up as one step of the final
        # 8-bit value. Adding 0.5 and truncating reproduces the reference.
        return torch.floor(rgb.flip(-1) * 255.0 + 0.5).to(torch.uint8)

    # -- spatial ------------------------------------------------------------

    def blur_dn(self, frames: Any, levels: int) -> Any:
        """`levels` rounds of blur and halve, on the colour pipeline's filter.

        The reference renormalises the filter to sum to one here — and only
        here — and applies rows before columns.
        """
        f = self._filt(BINOM5_SUM1 / float(BINOM5_SUM1.sum()))
        out = frames.to(self.dtype)
        for _ in range(levels):
            out = self._corr_dn(out, f, dim=1)  # rows
            out = self._corr_dn(out, f, dim=2)  # columns
        return out

    def build_lpyr(self, frames: Any, levels: int) -> list[Any]:
        """One band per scale, finest first, each (T, H_l, W_l, 3).

        The pyramid path uses the filter unnormalised, and takes columns before
        rows — the opposite order from :meth:`blur_dn`. Both follow the
        reference; the asymmetry is in the original.
        """
        f = self._filt(BINOM5)
        cur = frames.to(self.dtype)
        bands = []
        for _ in range(levels - 1):
            lo = self._corr_dn(cur, f, dim=2)  # columns
            lo2 = self._corr_dn(lo, f, dim=1)  # then rows
            hi = self._up_conv(lo2, f, dim=1, out_size=lo.shape[1])
            hi2 = self._up_conv(hi, f, dim=2, out_size=cur.shape[2])
            bands.append(cur - hi2)
            cur = lo2
        bands.append(cur)  # the coarsest level is kept whole
        return bands

    def recon_lpyr(self, bands: Sequence[Any]) -> Any:
        """Collapse the pyramid, the inverse of :meth:`build_lpyr`."""
        f = self._filt(BINOM5)
        bands = list(bands)
        cur = bands[-1]
        for level in range(len(bands) - 2, -1, -1):
            target = bands[level]
            hi = self._up_conv(cur, f, dim=1, out_size=target.shape[1])
            res = self._up_conv(hi, f, dim=2, out_size=target.shape[2])
            cur = target + res
        return cur

    def upsample_bilinear(self, frames: Any, height: int, width: int) -> Any:
        torch = self._torch
        x = frames.to(self.dtype).permute(0, 3, 1, 2)
        # align_corners=False is the convention OpenCV's INTER_LINEAR follows,
        # which is what the reference resizes with.
        out = torch.nn.functional.interpolate(
            x, size=(height, width), mode="bilinear", align_corners=False
        )
        return out.permute(0, 2, 3, 1).contiguous()

    # -- temporal, all along axis 0 -----------------------------------------

    def ideal_bandpass(
        self, series: Any, fl: float, fh: float, sampling_rate: float
    ) -> Any:
        """Keep only frequencies strictly between `fl` and `fh`.

        This deliberately uses the full complex transform and masks against a
        one-sided frequency ramp over *every* bin, then takes the real part.
        That is what the reference does, following the original MATLAB, and it
        is not the same as filtering with a real-input transform: masking the
        ramp discards the upper half of the spectrum rather than treating it as
        the conjugate mirror, and the two give different answers. The obvious
        real-input version was written first here and disagreed with the
        reference by the full amplitude of the signal.
        """
        import warnings

        torch = self._torch
        x = series.to(self.dtype)
        n = x.shape[0]
        freqs = torch.arange(n, device=x.device, dtype=self.dtype) / n * sampling_rate
        keep = (freqs > fl) & (freqs < fh)
        if not bool(keep.any()):
            # Same condition the reference warns about: a band narrower than
            # this clip's frequency resolution selects nothing, the result is
            # zeros, and the magnified output equals the input. Saying so beats
            # returning zeros quietly.
            warnings.warn(
                f"ideal_bandpass selected no frequency bins: the band "
                f"({fl:.4g}, {fh:.4g}) Hz falls between the bins of a "
                f"{n}-frame signal at {sampling_rate:g} fps. The result is all "
                f"zeros, so the magnified output will equal the input.",
                RuntimeWarning,
                stacklevel=2,
            )
        shape = [-1] + [1] * (x.dim() - 1)
        spectrum = torch.fft.fft(x.to(torch.complex64), dim=0)
        spectrum = spectrum * keep.reshape(shape)
        return torch.real(torch.fft.ifft(spectrum, dim=0)).to(self.dtype)

    def butter_bandpass(
        self,
        series: Any,
        fl: float,
        fh: float,
        sampling_rate: float,
        order: int = 1,
    ) -> Any:
        """The difference of two lowpasses, coefficients from the baseline.

        The coefficients come from SciPy through :mod:`vidmag.cpu.filters` rather
        than being recomputed here: they are a property of the filter design,
        not of the device, and one source for them is one place to be wrong.
        """
        from ..cpu import filters as cpu_filters

        nyq = 0.5 * sampling_rate
        b_hi, a_hi = cpu_filters._butter_lowpass_coeffs(order, fh / nyq)
        b_lo, a_lo = cpu_filters._butter_lowpass_coeffs(order, fl / nyq)
        return self._lowpass(series, b_hi, a_hi) - self._lowpass(series, b_lo, a_lo)

    def _lowpass(self, series: Any, b: np.ndarray, a: np.ndarray) -> Any:
        """One first-order recursion along time, matching the reference's."""
        torch = self._torch
        x = series.to(self.dtype)
        out = torch.zeros_like(x)
        prev_y = torch.zeros_like(x[0])
        prev_x = torch.zeros_like(x[0])
        for t in range(x.shape[0]):
            y = float(b[0]) * x[t] + float(b[1]) * prev_x - float(a[1]) * prev_y
            out[t] = y
            prev_y, prev_x = y, x[t]
        return out

    def iir_bandpass(self, series: Any, r1: float, r2: float) -> Any:
        """Two decaying averages subtracted, both starting at the first frame."""
        torch = self._torch
        x = series.to(self.dtype)
        out = torch.zeros_like(x)
        fast = x[0].clone()
        slow = x[0].clone()
        for t in range(1, x.shape[0]):
            fast = (1.0 - r1) * fast + r1 * x[t]
            slow = (1.0 - r2) * slow + r2 * x[t]
            out[t] = fast - slow
        return out

    # -- amplification ------------------------------------------------------

    def apply_gain(
        self, frames: Any, gain_y: float, gain_i: float, gain_q: float
    ) -> Any:
        gains = self._torch.tensor(
            [gain_y, gain_i, gain_q], dtype=self.dtype, device=frames.device
        )
        return frames.to(self.dtype) * gains

    # -- streaming ----------------------------------------------------------

    def iir_step(self, fast: Any, slow: Any, current: Any, r1: float, r2: float) -> Any:
        """Advance both running averages by one frame, in place, as the others do."""
        fast.mul_(1.0 - r1).add_(current * r1)
        slow.mul_(1.0 - r2).add_(current * r2)
        return fast - slow
