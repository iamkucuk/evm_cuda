"""CPU (NumPy) reference implementation — the correctness oracle.

The three modules here are the MIT-faithful Python baseline that every other
backend is measured against:

* :mod:`vidmag.cpu.pyramids` — matlabPyrTools ``binom5`` + ``reflect1`` corrDn /
  upConv, and the Laplacian pyramid built on top of them.
* :mod:`vidmag.cpu.filters` — the three temporal filters (ideal FFT, first-order
  Butterworth, direct r1/r2 IIR).
* :mod:`vidmag.cpu.magnify` — the four ``magnify_*`` pipelines, one per MATLAB
  ``amplify_spatial_*`` script.

Everything re-exported here is also re-exported from the package root
(``vidmag``), except :func:`corr_dn_axis` and :func:`up_conv_axis`, which are the
low-level separable primitives used by the CUDA parity tests.
"""

from .pyramids import (
    BINOM5,
    BINOM5_SUM1,
    blur_dn,
    blur_dn_clr,
    build_lpyr,
    corr_dn_axis,
    laplacian_pyramid_channels,
    max_pyr_ht,
    recon_lpyr,
    reconstruct_from_channels,
    up_conv_axis,
)
from .filters import (
    butter_bandpass,
    ideal_bandpass,
    iir_bandpass,
)
from .magnify import (
    DROP_LAST,
    EXAGGERATION_FACTOR,
    figure6_alpha_schedule,
    magnify_color_gdown_ideal,
    magnify_motion_lpyr_butter,
    magnify_motion_lpyr_ideal,
    magnify_motion_lpyr_iir,
    magnify_phase,
)

__all__ = [
    "BINOM5",
    "BINOM5_SUM1",
    "blur_dn",
    "blur_dn_clr",
    "build_lpyr",
    "corr_dn_axis",
    "laplacian_pyramid_channels",
    "max_pyr_ht",
    "recon_lpyr",
    "reconstruct_from_channels",
    "up_conv_axis",
    "butter_bandpass",
    "ideal_bandpass",
    "iir_bandpass",
    "DROP_LAST",
    "EXAGGERATION_FACTOR",
    "figure6_alpha_schedule",
    "magnify_color_gdown_ideal",
    "magnify_motion_lpyr_butter",
    "magnify_motion_lpyr_ideal",
    "magnify_motion_lpyr_iir",
    "magnify_phase",
]
