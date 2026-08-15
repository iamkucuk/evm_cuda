"""Frame-by-frame magnification must equal whole-clip magnification.

This is the only claim that makes the streaming path worth having. If it drifts
from the batch path, it is a second implementation that would need verifying
separately against the reference, and the argument that this project's output
matches a published method would no longer cover it.

The equality is exact for a good reason rather than by luck: both compute the
same recursion, in the same order, on the same numbers. The streaming version
just keeps the running state between calls instead of holding the whole clip.
"""

from __future__ import annotations

import numpy as np
import pytest

from evm.cpu import magnify as direct
from evm.stream import MotionStream

FPS = 30.0
PARAMS = dict(alpha=10.0, lambda_c=16.0, r1=0.4, r2=0.05, chrom_attenuation=0.1)


def _clip(seed: int = 5, frames: int = 24, size: int = 32) -> np.ndarray:
    """A textured image that really moves, sub-pixel.

    Brightness that changes uniformly is not movement: it lands in the coarsest
    pyramid band, the detail bands see nothing, and the pipeline returns its
    input — which would make every comparison below pass for the wrong reason.
    """
    rng = np.random.default_rng(seed)
    base = rng.integers(60, 190, (size + 4, size + 4, 3)).astype(np.float64)
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float64)

    out = np.empty((frames, size, size, 3), dtype=np.uint8)
    for t in range(frames):
        shift = 0.8 * np.sin(2 * np.pi * 1.0 * t / FPS)
        sy, sx = ys + 2.0 + shift, xs + 2.0
        y0, x0 = np.floor(sy).astype(int), np.floor(sx).astype(int)
        wy = (sy - y0)[..., None]
        out[t] = np.clip(
            base[y0, x0] * (1 - wy) + base[y0 + 1, x0] * wy, 0, 255
        ).astype(np.uint8)
    return out


def test_streaming_reproduces_the_whole_clip_result():
    """The claim the streaming path rests on."""
    clip = _clip()
    expected = direct.motion_lpyr_iir_core(clip, FPS, **PARAMS)

    assert np.abs(expected.astype(np.int16) - clip.astype(np.int16)).max() > 0, (
        "the batch pipeline did nothing on this clip, so this proves nothing"
    )

    stream = MotionStream(clip.shape[1], clip.shape[2], backend="cpu", **PARAMS)
    got = np.stack([stream.push(frame) for frame in clip])

    difference = np.abs(got.astype(np.int16) - expected.astype(np.int16))
    assert difference.max() == 0, (
        f"streaming and batch disagree by up to {difference.max()} steps on "
        f"{int((difference > 0).sum())} values"
    )


def test_the_first_frame_comes_back_unchanged():
    """Both running averages start at the first frame, so nothing is amplified.

    Worth pinning: an implementation that started them at zero instead would
    produce an enormous flash on the first frame, which on a live feed is the
    most visible failure possible.
    """
    clip = _clip()
    stream = MotionStream(clip.shape[1], clip.shape[2], backend="cpu", **PARAMS)
    assert np.array_equal(stream.push(clip[0]), clip[0])


def test_memory_does_not_grow_with_the_length_of_the_feed():
    """State is a fixed number of arrays, however many frames have gone by."""
    clip = _clip(frames=60)
    stream = MotionStream(clip.shape[1], clip.shape[2], backend="cpu", **PARAMS)

    for frame in clip[:5]:
        stream.push(frame)
    after_five = sum(a.nbytes for a in stream._fast + stream._slow)

    for frame in clip[5:]:
        stream.push(frame)
    after_sixty = sum(a.nbytes for a in stream._fast + stream._slow)

    assert after_five == after_sixty, "the retained state grew with the feed"
    assert stream.frames_seen == len(clip)


def test_resetting_starts_the_stream_over():
    clip = _clip()
    stream = MotionStream(clip.shape[1], clip.shape[2], backend="cpu", **PARAMS)

    first_pass = np.stack([stream.push(frame) for frame in clip])
    stream.reset()
    second_pass = np.stack([stream.push(frame) for frame in clip])

    assert np.array_equal(first_pass, second_pass), (
        "a reset stream did not reproduce its own first run"
    )


def test_a_frame_of_the_wrong_size_is_refused():
    """Silently resizing would corrupt the running state."""
    stream = MotionStream(32, 32, backend="cpu", **PARAMS)
    with pytest.raises(ValueError, match="was created for"):
        stream.push(np.zeros((48, 48, 3), dtype=np.uint8))


def test_a_frame_of_the_wrong_type_is_refused():
    stream = MotionStream(32, 32, backend="cpu", **PARAMS)
    with pytest.raises(TypeError, match="uint8"):
        stream.push(np.zeros((32, 32, 3), dtype=np.float32))


def test_the_decay_rates_must_be_the_right_way_round():
    """A faster average minus a slower one; the reverse is not a bandpass."""
    with pytest.raises(ValueError, match="r1 > r2"):
        MotionStream(32, 32, backend="cpu", r1=0.05, r2=0.4)


def test_it_reports_which_backend_it_is_using():
    stream = MotionStream(32, 32, backend="cpu", **PARAMS)
    assert stream.backend_name == "cpu"
    assert "cpu" in repr(stream)


def test_the_default_backend_can_actually_stream():
    """Automatic selection must not choose a backend that cannot stream.

    The whole-clip preference order puts the hand-written NVIDIA backend first,
    and that backend implements the four whole-clip pipelines but not the
    frame-at-a-time operations. Taking that order unfiltered would make the
    default unusable on exactly the machines this project is fastest on.
    """
    import numpy as np

    from evm.backend import registry
    from evm.stream import MotionStream

    stream = MotionStream(32, 48)
    chosen = next(i for i in registry.list_backends() if i.name == stream.backend_name)
    assert chosen.available, f"default chose {chosen.name}, which cannot run here"
    assert chosen.capabilities.streaming, (
        f"default chose {chosen.name}, which does not claim it can stream"
    )
    # And it works, not merely claims to.
    out = stream.push(np.zeros((32, 48, 3), dtype=np.uint8))
    assert out.shape == (32, 48, 3) and out.dtype == np.uint8
