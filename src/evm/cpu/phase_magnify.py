"""Phase-based motion magnification.

The 2013 follow-up to the method the rest of this library implements. Both
amplify small movement; they differ in what they amplify.

The original works on a Laplacian pyramid and scales the difference between
neighbouring scales. That difference approximates what a small shift does to an
image, and adding a multiple of it back approximates shifting further. The
approximation is good while the movement is small compared with the detail it
moves, and it is what produces ripples and haloes at edges when it is not.

This works on a complex steerable pyramid, where each part of the image has a
phase as well as an amplitude, and a shift shows up as a change in phase. So
instead of approximating a shift, it changes the phase directly — which *is* a
shift, for the band in question. That is why it holds together at
amplifications where the original tears.

The cost is speed. This transforms every frame into the frequency domain and
back once per scale and direction, where the original does a handful of small
convolutions.
"""

from __future__ import annotations

import numpy as np

from .csp import SteerablePyramid
from .filters import butter_bandpass, iir_bandpass

__all__ = ["phase_magnify"]


def _wrap(phase: np.ndarray) -> np.ndarray:
    """Bring a phase difference into (-pi, pi].

    Phase is an angle, so a change of 1.9 pi is really a change of -0.1 pi in
    the other direction. Without this the largest possible small movement would
    be read as an enormous one, in the wrong direction, wherever the phase
    happened to cross the wrap point.
    """
    return np.mod(phase + np.pi, 2 * np.pi) - np.pi


def _amplitude_weighted_blur(
    phase: np.ndarray, amplitude: np.ndarray, sigma: float
) -> np.ndarray:
    """Smooth the phase spatially, weighting by how strong the signal is.

    Phase is meaningless where there is nothing to have a phase — a flat patch
    of image gives a coefficient near zero whose angle is noise. Averaging
    plainly would let that noise spread into its neighbours. Weighting by
    amplitude lets confident measurements dominate the uncertain ones.

    Set ``sigma`` to zero to skip it, which is what the original paper's
    simplest form does.
    """
    if sigma <= 0:
        return phase

    from scipy.ndimage import gaussian_filter

    weighted = gaussian_filter(phase * amplitude, sigma=(0, sigma, sigma))
    weights = gaussian_filter(amplitude, sigma=(0, sigma, sigma))
    # Where nothing was measured, leave the phase alone rather than dividing by
    # something close to zero.
    return np.where(weights > 1e-9, weighted / np.maximum(weights, 1e-9), phase)


def phase_magnify(
    frames_bgr_u8: np.ndarray,
    fps: float,
    *,
    alpha: float,
    fl: float | None = None,
    fh: float | None = None,
    r1: float | None = None,
    r2: float | None = None,
    scales: int = 3,
    orientations: int = 4,
    sigma: float = 0.0,
    attenuate_other_frequencies: bool = False,
) -> np.ndarray:
    """Amplify motion by changing phase rather than by scaling detail.

    Args:
        frames_bgr_u8: ``(T, H, W, 3)`` uint8, blue-green-red.
        fps: frame rate, which is what makes ``fl`` and ``fh`` mean something.
        alpha: how much to amplify. Phase-based magnification tolerates larger
            values than the original method before it breaks down.
        fl, fh: the band to amplify, in cycles per second. Filtered with a
            Butterworth filter.
        r1, r2: an alternative to ``fl``/``fh``: two decay rates whose running
            averages are subtracted, as the original method's cheapest filter
            does. Give one pair or the other.
        scales: how many scales the pyramid splits into.
        orientations: how many directions per scale.
        sigma: how much to smooth the measured phase spatially, weighted by
            amplitude. Zero skips it. Raise it if the result looks speckled in
            flat areas.
        attenuate_other_frequencies: if set, keep only the amplified band and
            suppress everything else, which isolates the movement instead of
            adding it to the original picture.

    Returns:
        ``(T, H, W, 3)`` uint8, same shape as the input.

    Only brightness is processed. Colour is carried through unchanged: motion
    is a brightness phenomenon at edges, and amplifying colour phase adds
    speckle without adding movement.
    """
    frames = np.asarray(frames_bgr_u8)
    if frames.ndim != 4 or frames.shape[3] != 3:
        raise ValueError(f"frames must be (T, H, W, 3); got {frames.shape}")
    if frames.dtype != np.uint8:
        raise TypeError(f"frames must be uint8, got {frames.dtype}")

    using_band = fl is not None and fh is not None
    using_rates = r1 is not None and r2 is not None
    if using_band == using_rates:
        raise ValueError(
            "give either fl and fh (a band in cycles per second) or r1 and r2 "
            "(two decay rates), not both and not neither"
        )

    count, height, width, _ = frames.shape
    pyramid = SteerablePyramid(height, width, scales=scales, orientations=orientations)

    # Brightness only. The two colour channels are put back untouched at the
    # end, so the result keeps its colour without it being amplified.
    yiq = np.stack([_bgr_to_yiq(frame) for frame in frames], axis=0)
    luma = yiq[..., 0]

    decomposed = [pyramid.decompose(luma[t]) for t in range(count)]

    magnified_bands: list[list[list[np.ndarray]]] = [
        [[None] * orientations for _ in range(scales)]  # type: ignore[list-item]
        for _ in range(count)
    ]

    for scale in range(scales):
        for direction in range(orientations):
            series = np.stack(
                [decomposed[t].bands[scale][direction] for t in range(count)]
            )
            amplitude = np.abs(series)
            phase = np.angle(series)

            # Movement is change in phase, measured against the first frame.
            # Unwrapping step by step keeps a real movement from being read as
            # a huge one in the opposite direction wherever the angle crosses
            # the wrap point.
            relative = np.empty_like(phase)
            relative[0] = 0.0
            running = np.zeros_like(phase[0])
            for t in range(1, count):
                running = running + _wrap(phase[t] - phase[t - 1])
                relative[t] = running

            if fl is not None and fh is not None:
                filtered = butter_bandpass(relative, fl, fh, fps, order=1, axis=0)
            else:
                # The check at the top of this function has already established
                # that exactly one pair was given, so these cannot be absent.
                assert r1 is not None and r2 is not None
                filtered = iir_bandpass(relative, r1, r2, axis=0)

            filtered = _amplitude_weighted_blur(filtered, amplitude, sigma)

            shift = np.exp(1j * alpha * filtered)
            if attenuate_other_frequencies:
                # Keep only what the filter selected: rebuild the coefficient
                # from the filtered phase alone rather than adding to what was
                # already there.
                new_series = amplitude * np.exp(
                    1j * (phase - relative + (1 + alpha) * filtered)
                )
            else:
                new_series = series * shift

            for t in range(count):
                magnified_bands[t][scale][direction] = new_series[t]

    out = np.empty_like(frames)
    for t in range(count):
        rebuilt = pyramid.reconstruct(
            type(decomposed[t])(
                highpass=decomposed[t].highpass,
                bands=magnified_bands[t],
                lowpass=decomposed[t].lowpass,
            )
        )
        frame_yiq = yiq[t].copy()
        frame_yiq[..., 0] = rebuilt
        out[t] = _yiq_to_bgr_u8(frame_yiq)
    return out


def _bgr_to_yiq(frame: np.ndarray) -> np.ndarray:
    from ..io.video import rgb_to_yiq

    rgb = frame[:, :, ::-1].astype(np.float64) / 255.0
    return rgb_to_yiq(rgb)


def _yiq_to_bgr_u8(frame_yiq: np.ndarray) -> np.ndarray:
    from ..io.video import yiq_to_rgb

    rgb = np.clip(yiq_to_rgb(frame_yiq), 0.0, 1.0)
    return np.round(rgb[:, :, ::-1] * 255.0).astype(np.uint8)
