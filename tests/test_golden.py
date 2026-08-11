"""Golden end-to-end fixtures: all four pipelines, no downloads, no GPU.

``test_against_mit_reference.py`` is the correctness oracle, but it skips unless
``data/*.mp4`` has been downloaded — so on a fresh CI runner nothing checks the
pipelines end to end. These tests close that hole: a deterministic synthetic
clip is committed in ``tests/fixtures/golden_input.npz``, its output through
each of the four public ``magnify_*`` entry points is committed alongside it,
and any change to pipeline behaviour shows up here immediately.

**Why the clip is re-encoded instead of being committed as a video.** The
pipelines take a *path* and decode it themselves (``evm.cpu.magnify._read_frames``
-> ``cv2.VideoCapture``), so a file has to exist. The project's own writer
(:func:`evm.io.video.encode_video`) is H.264 ``yuv420p`` at ``crf 18`` — lossy and
chroma-subsampled, and its exact output depends on the bundled libx264 version,
so the frames the pipelines saw would differ from machine to machine and no
exact golden value would be reproducible. Instead the committed frames are
written with FFV1 (mathematically lossless, present in every ffmpeg build) and
:func:`test_source_video_decodes_bit_exactly` asserts the decode is bit-exact
*before* the golden comparisons run. The pipelines are therefore driven through
their real public entry points, on their real decode path, with input frames
that are provably the committed ones.

Regenerate the fixtures with ``python scripts/dev/make_golden_fixtures.py``.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
import pytest

from evm import (
    magnify_color_gdown_ideal,
    magnify_motion_lpyr_butter,
    magnify_motion_lpyr_ideal,
    magnify_motion_lpyr_iir,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
INPUT_FIXTURE = FIXTURES / "golden_input.npz"

# The single source of truth for what the fixtures contain: name -> (entry
# point, keyword arguments). ``scripts/dev/make_golden_fixtures.py`` imports
# this dict, so the generator and the test can never drift apart.
#
# The band [1.5, 2.5] Hz brackets the clip's 2 Hz oscillation, which at 30 fps
# over the 30 retained frames lands exactly on DFT bin 2 — the ideal (FFT)
# filter therefore passes it cleanly instead of straddling two bins.
# ``alpha=10`` for the color pipeline (rather than the pulse preset's 50) keeps
# the amplified result off the 0/1 clipping rails, so the fixture stays
# sensitive to changes everywhere in the frame.
CASES: dict[str, tuple] = {
    "color_gdown_ideal": (
        magnify_color_gdown_ideal,
        dict(alpha=10.0, level=2, fl=1.5, fh=2.5,
             chrom_attenuation=1.0, sampling_rate=30.0),
    ),
    "motion_lpyr_ideal": (
        magnify_motion_lpyr_ideal,
        dict(alpha=10.0, lambda_c=16.0, fl=1.5, fh=2.5,
             chrom_attenuation=0.1, sampling_rate=30.0),
    ),
    "motion_lpyr_butter": (
        magnify_motion_lpyr_butter,
        dict(alpha=10.0, lambda_c=16.0, fl=1.5, fh=2.5,
             chrom_attenuation=0.1, sampling_rate=30.0),
    ),
    "motion_lpyr_iir": (
        magnify_motion_lpyr_iir,
        dict(alpha=10.0, lambda_c=16.0, r1=0.4, r2=0.05,
             chrom_attenuation=0.1),
    ),
}

# Both sides of the comparison are exactly k/255 in float32, so a byte-identical
# pipeline gives a difference of exactly 0. Anything looser would have to be
# justified by a measurement; do not loosen without one.
ATOL = 1e-6


def write_lossless_video(path: str | Path, frames_bgr: np.ndarray, fps: float) -> None:
    """Write ``(T, H, W, 3)`` uint8 BGR frames as FFV1, which decodes bit-exactly.

    Deliberately *not* :func:`evm.io.video.encode_video` — see the module
    docstring. ``bgr0`` keeps the frames in RGB space (no chroma subsampling,
    no YUV round-trip), which is what makes the decode byte-identical.
    """
    import av

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream(
            "ffv1", rate=Fraction(fps).limit_denominator(1_000_000)
        )
        # add_stream is typed as returning a video, audio or subtitle stream.
        # "ffv1" is a video codec, so this narrows to the one the next lines
        # assume, and says so if that ever stops being true.
        assert isinstance(stream, av.video.stream.VideoStream)
        stream.height, stream.width = frames_bgr.shape[1:3]
        stream.pix_fmt = "bgr0"
        for i in range(frames_bgr.shape[0]):
            frame = av.VideoFrame.from_ndarray(frames_bgr[i], format="bgr24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():  # flush
            container.mux(packet)


def load_input() -> tuple[np.ndarray, float]:
    """Committed synthetic clip: ``(T, H, W, 3)`` uint8 BGR frames + fps."""
    with np.load(INPUT_FIXTURE) as f:
        return f["frames"], float(f["fps"])


def _decode(path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened(), f"could not open {path}"
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return np.stack(frames, axis=0)


@pytest.fixture(scope="module")
def source_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The committed clip, re-encoded losslessly to a file the pipelines can open."""
    frames, fps = load_input()
    path = tmp_path_factory.mktemp("golden") / "synthetic.mkv"
    write_lossless_video(path, frames, fps)
    return path


def test_source_video_decodes_bit_exactly(source_video: Path) -> None:
    """Precondition for every golden comparison below.

    If this fails, the pipelines are being fed something other than the frames
    the fixtures were generated from, and a golden mismatch would be a false
    alarm — so it is asserted separately and loudly.
    """
    frames, fps = load_input()
    decoded = _decode(source_video)
    assert decoded.shape == frames.shape
    assert np.array_equal(decoded, frames), "FFV1 round-trip was not lossless"

    cap = cv2.VideoCapture(str(source_video))
    assert cap.get(cv2.CAP_PROP_FPS) == fps
    cap.release()


@pytest.mark.parametrize("name", sorted(CASES))
def test_golden_output_matches_fixture(
    name: str, source_video: Path, tmp_path: Path
) -> None:
    fn, kwargs = CASES[name]
    out = fn(str(source_video), str(tmp_path / f"{name}.mp4"), **kwargs)

    with np.load(FIXTURES / f"golden_{name}.npz") as f:
        golden = f["output"].astype(np.float32) / 255.0

    assert out.shape == golden.shape
    max_diff = float(np.abs(out - golden).max())
    assert max_diff <= ATOL, (
        f"{name}: max |out - golden| = {max_diff:.6g} (atol {ATOL:g}); "
        f"{int((np.abs(out - golden) > ATOL).sum())} of {out.size} values differ. "
        "Regenerate with scripts/dev/make_golden_fixtures.py only if the change "
        "is intended."
    )


@pytest.mark.parametrize("name", sorted(CASES))
def test_golden_output_is_not_degenerate(name: str) -> None:
    """A fixture over an unamplified clip would pass forever and prove nothing.

    Each pipeline must move the input by far more than the 1/255 quantisation
    step; the smallest measured deviation across the four is ~0.06 (Butterworth).
    """
    frames, _ = load_input()
    with np.load(FIXTURES / f"golden_{name}.npz") as f:
        golden = f["output"].astype(np.float32) / 255.0

    reference = frames[: golden.shape[0]].astype(np.float32) / 255.0
    assert np.abs(golden - reference).max() > 0.02
