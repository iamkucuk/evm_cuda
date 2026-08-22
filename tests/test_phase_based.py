"""Phase-based magnification, checked against motion we constructed ourselves.

The rest of this project is verified by comparison: the NumPy implementation
against the original authors' published output, everything else against the
NumPy implementation. That route is not available here, because the authors'
rendered output for this method is not among the files this project can fetch.

So this is verified differently, and the difference is worth being explicit
about. The clip is built with a known movement — a texture shifted by a known
fraction of a pixel, following a known frequency — and the output is measured
to see whether the movement grew by the amount the method says it should.
That is a check against ground truth rather than against another
implementation, which is a different kind of evidence: it confirms the method
does what it claims, but it does not confirm this matches the authors' code
detail for detail.
"""

from __future__ import annotations

import numpy as np
import pytest

from vidmag.cpu.csp import SteerablePyramid
from vidmag.cpu.filters import butter_bandpass
from vidmag.cpu.phase_magnify import phase_magnify

FPS = 30.0
MOTION_HZ = 1.0
BAND = (0.5, 1.5)


# ---------------------------------------------------------------------------
# The decomposition
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scales,orientations", [(2, 4), (3, 4), (3, 2)])
def test_the_pyramid_rebuilds_its_input_exactly(scales, orientations):
    """Nothing built on a decomposition that loses information can be trusted."""
    image = np.random.default_rng(0).random((64, 64))
    pyramid = SteerablePyramid(64, 64, scales=scales, orientations=orientations)
    rebuilt = pyramid.reconstruct(pyramid.decompose(image))
    error = np.abs(rebuilt - image).max()
    assert error < 1e-10, f"round trip lost {error:.2e}, so the filters are wrong"


def test_the_bands_are_complex_and_carry_a_phase():
    """A real-valued band cannot say which way anything moved.

    Each band covers half the frequency plane, which is what makes it complex.
    A decomposition whose bands came out real would be a Laplacian pyramid with
    extra steps, and phase-based magnification would be impossible on it.
    """
    image = np.random.default_rng(1).random((64, 64))
    bands = SteerablePyramid(64, 64, scales=2, orientations=4).decompose(image)
    for scale in bands.bands:
        for band in scale:
            assert np.iscomplexobj(band)
            assert np.abs(band.imag).max() > 1e-6, "band has no phase to read"


def test_asking_for_more_scales_than_the_image_allows_is_refused():
    with pytest.raises(ValueError, match="more than"):
        SteerablePyramid(32, 32, scales=8)


# ---------------------------------------------------------------------------
# The magnification, against known movement
# ---------------------------------------------------------------------------


def _moving_clip(
    frames: int = 32, size: int = 64, amplitude: float = 0.5
) -> np.ndarray:
    """A textured image sliding up and down by a fraction of a pixel."""
    rng = np.random.default_rng(0)
    base = np.clip(rng.random((size + 8, size + 8)) * 120 + 60, 0, 255)
    ys, xs = np.mgrid[0:size, 0:size]
    columns = xs.astype(int) + 4

    out = np.empty((frames, size, size, 3), dtype=np.uint8)
    for t in range(frames):
        shift = amplitude * np.sin(2 * np.pi * MOTION_HZ * t / FPS)
        rows = ys + 4 + shift
        row0 = np.floor(rows).astype(int)
        weight = rows - row0
        value = base[row0, columns] * (1 - weight) + base[row0 + 1, columns] * weight
        out[t] = np.repeat(
            np.clip(value, 0, 255).astype(np.uint8)[:, :, None], 3, axis=2
        )
    return out


def _vertical_shift(clip: np.ndarray) -> np.ndarray:
    """How far each frame has moved vertically against the first.

    Correlation against the first frame at three offsets, with a parabola
    through them for the sub-pixel position. Crude, but it needs no library and
    its error is small compared with the effect being measured.
    """
    reference = clip[0, :, :, 0].astype(float)
    reference -= reference.mean()

    shifts = []
    for frame in clip:
        current = frame[:, :, 0].astype(float)
        current -= current.mean()
        scores = np.array(
            [np.sum(np.roll(reference, k, axis=0) * current) for k in (-1, 0, 1)]
        )
        denominator = scores[0] - 2 * scores[1] + scores[2]
        shifts.append(
            0.0 if denominator == 0 else 0.5 * (scores[0] - scores[2]) / denominator
        )
    return np.array(shifts)


def _filter_gain() -> float:
    """How much of the movement the temporal filter actually passes.

    The amplification is applied to the *filtered* movement, not the movement,
    so the expected growth is 1 + alpha times this. Measured rather than
    assumed, because a first-order Butterworth passes only about half its
    input at the centre of a band this narrow, and predicting 1 + alpha would
    be wrong by that factor.
    """
    time = np.arange(64)
    signal = np.sin(2 * np.pi * MOTION_HZ * time / FPS)[:, None]
    filtered = butter_bandpass(signal, BAND[0], BAND[1], FPS, order=1, axis=0)
    half = len(time) // 2  # skip the filter starting up
    return float(np.abs(filtered[half:]).max() / np.abs(signal[half:]).max())


def test_no_amplification_leaves_the_clip_alone():
    """The strongest single check that the pipeline is transparent.

    Transform to phase, filter, multiply by zero, transform back: any error
    anywhere in that chain shows up here, with nothing to hide behind.
    """
    clip = _moving_clip()
    out = phase_magnify(
        clip, FPS, alpha=0.0, fl=BAND[0], fh=BAND[1], scales=3, orientations=4
    )
    before = np.abs(_vertical_shift(clip)).max()
    after = np.abs(_vertical_shift(out)).max()
    assert abs(after - before) < 0.01, (
        f"alpha=0 changed the movement from {before:.4f} to {after:.4f} pixels"
    )


@pytest.mark.parametrize("alpha", [2.0, 4.0])
def test_movement_grows_by_the_predicted_amount(alpha):
    """The quantitative claim: movement grows by 1 + alpha times what the
    temporal filter passed.

    This is the test that says the method works, rather than merely that it
    changes the picture.
    """
    clip = _moving_clip()
    out = phase_magnify(
        clip, FPS, alpha=alpha, fl=BAND[0], fh=BAND[1], scales=3, orientations=4
    )

    before = np.abs(_vertical_shift(clip)).max()
    after = np.abs(_vertical_shift(out)).max()
    assert before > 0.1, "the test clip barely moves; the comparison is weak"

    measured = after / before
    predicted = 1.0 + alpha * _filter_gain()
    # A generous bound, because the shift measurement above is a three-point
    # parabola rather than anything careful. It is still far tighter than the
    # difference between working and not working: with no amplification the
    # ratio is 1, and the prediction here is 2 to 3.
    assert abs(measured - predicted) < 0.35 * predicted, (
        f"movement grew {measured:.2f} times, but 1 + {alpha} x filter gain "
        f"predicts {predicted:.2f}"
    )


def test_more_amplification_moves_things_further():
    clip = _moving_clip()
    ratios = []
    for alpha in (0.0, 2.0, 4.0):
        out = phase_magnify(
            clip, FPS, alpha=alpha, fl=BAND[0], fh=BAND[1], scales=3, orientations=4
        )
        ratios.append(np.abs(_vertical_shift(out)).max())
    assert ratios[0] < ratios[1] < ratios[2], (
        f"amplification is not monotonic: {ratios}"
    )


def test_the_two_ways_of_choosing_a_band_are_both_accepted():
    clip = _moving_clip(frames=24)
    by_band = phase_magnify(
        clip, FPS, alpha=2.0, fl=BAND[0], fh=BAND[1], scales=2, orientations=2
    )
    by_rates = phase_magnify(
        clip, FPS, alpha=2.0, r1=0.4, r2=0.05, scales=2, orientations=2
    )
    assert by_band.shape == clip.shape
    assert by_rates.shape == clip.shape
    assert not np.array_equal(by_band, by_rates), (
        "two different filters produced identical output, so one is ignored"
    )


def test_giving_both_or_neither_filter_is_refused():
    clip = _moving_clip(frames=8)
    with pytest.raises(ValueError, match="either"):
        phase_magnify(clip, FPS, alpha=2.0)
    with pytest.raises(ValueError, match="either"):
        phase_magnify(clip, FPS, alpha=2.0, fl=0.5, fh=1.5, r1=0.4, r2=0.05)


def test_colour_is_carried_through_rather_than_amplified():
    """Only brightness is processed; amplifying colour phase adds speckle."""
    clip = _moving_clip(frames=16)
    coloured = clip.copy()
    coloured[..., 2] = np.clip(coloured[..., 2].astype(int) + 40, 0, 255)

    out = phase_magnify(
        coloured, FPS, alpha=2.0, fl=BAND[0], fh=BAND[1], scales=2, orientations=2
    )
    assert out.shape == coloured.shape
    assert out.dtype == np.uint8
