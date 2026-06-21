"""Slow integration tests comparing against MIT's own rendered outputs.

Skipped unless ``data/face.mp4`` / ``data/baby.mp4`` and the MIT reference
result videos are present (downloaded via ``scripts/download_samples.py`` and
``scripts/download_mit_outputs.py``).

These are the ultimate correctness check for the port: they reproduce the
exact calls in ``reproduceResults.m`` and assert that the per-pixel RMSE
against MIT's published outputs is within video-codec re-encoding noise.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evm import (  # noqa: E402
    load_video,
    magnify_color_gdown_ideal,
    magnify_motion_lpyr_iir,
)

DATA = ROOT / "data"
OUT = ROOT / "output"


def _have(*names: str) -> bool:
    return all((DATA / n).exists() for n in names)


@pytest.mark.skipif(
    not _have("face.mp4", "face_mit_ref.mp4"),
    reason="download data/face.mp4 and data/face_mit_ref.mp4 first",
)
def test_face_color_matches_mit(tmp_path: Path) -> None:
    out = magnify_color_gdown_ideal(
        str(DATA / "face.mp4"),
        str(tmp_path / "face.mp4"),
        alpha=50,
        level=4,
        fl=50 / 60,
        fh=60 / 60,
        chrom_attenuation=1.0,
        sampling_rate=30.0,
    )
    ref, _ = load_video(str(DATA / "face_mit_ref.mp4"))
    assert out.shape == ref.shape
    rmse = float(np.sqrt(((out - ref) ** 2).mean()))
    # MIT's outputs are MP4-reencoded; ~0.03 RMSE is the codec noise floor.
    assert rmse < 0.05, f"face RMSE {rmse:.4f} too high vs MIT"


@pytest.mark.skipif(
    not _have("baby.mp4", "baby_mit_ref.mp4"),
    reason="download data/baby.mp4 and data/baby_mit_ref.mp4 first",
)
def test_baby_iir_matches_mit(tmp_path: Path) -> None:
    out = magnify_motion_lpyr_iir(
        str(DATA / "baby.mp4"),
        str(tmp_path / "baby.mp4"),
        alpha=10,
        lambda_c=16,
        r1=0.4,
        r2=0.05,
        chrom_attenuation=0.1,
    )
    ref, _ = load_video(str(DATA / "baby_mit_ref.mp4"))
    assert out.shape == ref.shape
    rmse = float(np.sqrt(((out - ref) ** 2).mean()))
    assert rmse < 0.05, f"baby RMSE {rmse:.4f} too high vs MIT"
