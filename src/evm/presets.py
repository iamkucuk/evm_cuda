"""Named parameter sets for the four magnification pipelines.

One frozen table, :data:`PRESETS`, is the single source of truth for every
"just magnify it" entry point — the facade, the CLI and the docs all read it,
so a preset's numbers exist in exactly one place.

Each row records three things: which pipeline it selects, the parameters, and
what it is for. A fourth field, ``source``, names the place in this repository
the numbers were taken from; nothing is in this table that cannot be traced to
a call the project already makes and checks.

``Preset.pipeline`` is a pipeline *stem*. It resolves two ways, and both are
the same pipeline:

    stem "motion_lpyr_iir"  ->  evm.cpu.magnify.motion_lpyr_iir_core(frames, fps, **params)
                            ->  evm.magnify_motion_lpyr_iir(in_path, out_path, **params)

``fl`` / ``fh`` are in Hz and are interpreted against the clip's own frame
rate. No preset pins ``sampling_rate``: the reference calls pass it explicitly
only because their clips happen to be 30 fps, and hard-coding 30 into a preset
would silently mis-filter every clip that is not (see ``pulse`` below, where
that difference is a no-op on the reference clip and a correctness fix on any
other).
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping, NamedTuple

__all__ = ["Preset", "PRESETS", "get"]


class Preset(NamedTuple):
    """One row of :data:`PRESETS`.

    ``pipeline`` is the stem shared by the array core
    (``<stem>_core`` in :mod:`evm.cpu.magnify`) and the path function
    (``magnify_<stem>``); ``params`` are keyword arguments for either.
    """

    pipeline: str
    params: Mapping[str, float | int]
    description: str
    source: str


PRESETS: Mapping[str, Preset] = MappingProxyType(
    {
        "pulse": Preset(
            pipeline="color_gdown_ideal",
            params=MappingProxyType(
                {
                    "alpha": 50.0,
                    "level": 4,
                    "fl": 50 / 60,
                    "fh": 60 / 60,
                    "chrom_attenuation": 1.0,
                }
            ),
            description=(
                "Human pulse: the colour change blood flow makes in skin, "
                "banded to 50-60 bpm. Faces, wrists, babies."
            ),
            source=(
                "tests/test_against_mit_reference.py::test_face_color_matches_mit "
                "— reproduceResults.m's face.mp4 call, checked against MIT's own "
                "render face-ideal-from-0.83333-to-1-alpha-50-level-4-chromAtn-1.mp4. "
                "That call also passes sampling_rate=30.0; this preset omits it "
                "because face.mp4 is exactly 30.0 fps, so the core's default "
                "(the clip's own fps) reproduces the call byte for byte while "
                "staying correct on clips at other frame rates."
            ),
        ),
        "motion": Preset(
            pipeline="motion_lpyr_iir",
            params=MappingProxyType(
                {
                    "alpha": 10.0,
                    "lambda_c": 16.0,
                    "r1": 0.4,
                    "r2": 0.05,
                    "chrom_attenuation": 0.1,
                }
            ),
            description=(
                "Sub-pixel motion at everyday speeds: an infant's breathing, "
                "a chest rising, a swaying structure. The r1/r2 IIR band is "
                "sampling-rate free, so this preset needs no fps assumption."
            ),
            source=(
                "tests/test_against_mit_reference.py::test_baby_iir_matches_mit "
                "— reproduceResults.m's baby.mp4 call, checked against MIT's own "
                "render baby-iir-r1-0.4-r2-0.05-alpha-10-lambda_c-16-chromAtn-0.1.mp4."
            ),
        ),
        "motion_phase": Preset(
            pipeline="phase",
            params=MappingProxyType(
                {
                    "alpha": 15.0,
                    "fl": 0.5,
                    "fh": 1.5,
                    "scales": 3,
                    "orientations": 4,
                    "sigma": 0.0,
                }
            ),
            description=(
                "The same sub-pixel motion as 'motion', but amplified by "
                "changing phase rather than by scaling image detail. Slower, "
                "and it holds together at amplifications where the other "
                "method tears into ripples at edges. Use it when 'motion' "
                "produces artefacts before it produces a visible movement."
            ),
            source=(
                "Wadhwa, Rubinstein, Durand and Freeman, 'Phase-Based Video "
                "Motion Processing', SIGGRAPH 2013. The parameters here are a "
                "starting point rather than a reproduction of a published "
                "call: unlike the other presets, this one is NOT checked "
                "against the authors' own rendered output, because that output "
                "is not among the files this project can fetch. What is "
                "checked, in tests/test_phase_based.py, is that a clip built "
                "with a known sub-pixel movement comes out moved by the "
                "predicted amount."
            ),
        ),
        "vibration": Preset(
            pipeline="motion_lpyr_ideal",
            params=MappingProxyType(
                {
                    "alpha": 50.0,
                    "lambda_c": 10.0,
                    "fl": 72.0,
                    "fh": 92.0,
                    "chrom_attenuation": 0.0,
                }
            ),
            description=(
                "Mechanical vibration in a narrow band — the guitar low-E "
                "string at 72-92 Hz. REQUIRES A HIGH-SPEED CLIP: those cutoffs "
                "are above Nyquist for anything under ~184 fps, where the ideal "
                "bandpass passes nothing and the output is the input."
            ),
            source=(
                "scripts/run_evm.py module docstring, the guitar.mp4 E-string "
                "example (--mode motion --alpha 50 --lambda-c 10 --fl 72 --fh 92 "
                "--chromatt 0). Weaker provenance than the two above: this "
                "repository downloads guitar.mp4 (scripts/download_samples.py) "
                "but holds no MIT render of it, so nothing here checks the "
                "result — unlike pulse and motion, which the MIT-reference "
                "tests measure."
            ),
        ),
        # No "breathing" preset. The only breathing material in this repository
        # is baby.mp4 (README.md:34, colab/evm_cuda_benchmark.ipynb), and it is
        # rendered with exactly the call above: "breathing" would be a second
        # name for "motion" with identical numbers. Presets earn their place by
        # carrying parameters, not by renaming a row.
    }
)


def get(name: str) -> Preset:
    """Look up a preset by name, naming the alternatives when there is no match."""
    try:
        return PRESETS[name]
    except KeyError:
        raise KeyError(
            f"unknown preset {name!r}; available: {', '.join(sorted(PRESETS))}"
        ) from None
