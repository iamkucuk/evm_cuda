"""Eulerian Video Magnification — Python baseline (MIT reference port).

Four entry points matching the four MATLAB amplification functions in
EVM_Matlab-1.1.zip:

* ``magnify_color_gdown_ideal`` — ``amplify_spatial_Gdown_temporal_ideal``
* ``magnify_motion_lpyr_ideal``  — ``amplify_spatial_lpyr_temporal_ideal``
* ``magnify_motion_lpyr_butter`` — ``amplify_spatial_lpyr_temporal_butter``
* ``magnify_motion_lpyr_iir``    — ``amplify_spatial_lpyr_temporal_iir``

The spatial (binom5 + reflect1 corrDn/upConv) and temporal (ideal / 1st-order
Butterworth / direct r1-r2 IIR) kernels reproduce matlabPyrTools and the
reference bandpassing bit-for-bit, so this package is the correctness oracle
for the forthcoming CUDA port.
"""

from .video import load_video, save_video, VideoInfo, rgb_to_yiq, yiq_to_rgb
from .pyramids import (
    BINOM5,
    BINOM5_SUM1,
    blur_dn,
    blur_dn_clr,
    build_lpyr,
    recon_lpyr,
    laplacian_pyramid_channels,
    reconstruct_from_channels,
    max_pyr_ht,
)
from .filters import (
    ideal_bandpass,
    butter_bandpass,
    iir_bandpass,
)
from .magnify import (
    figure6_alpha_schedule,
    magnify_color_gdown_ideal,
    magnify_motion_lpyr_ideal,
    magnify_motion_lpyr_butter,
    magnify_motion_lpyr_iir,
    DROP_LAST,
    EXAGGERATION_FACTOR,
)

__all__ = [
    "load_video",
    "save_video",
    "VideoInfo",
    "rgb_to_yiq",
    "yiq_to_rgb",
    "BINOM5",
    "BINOM5_SUM1",
    "blur_dn",
    "blur_dn_clr",
    "build_lpyr",
    "recon_lpyr",
    "laplacian_pyramid_channels",
    "reconstruct_from_channels",
    "max_pyr_ht",
    "ideal_bandpass",
    "butter_bandpass",
    "iir_bandpass",
    "figure6_alpha_schedule",
    "magnify_color_gdown_ideal",
    "magnify_motion_lpyr_ideal",
    "magnify_motion_lpyr_butter",
    "magnify_motion_lpyr_iir",
    "DROP_LAST",
    "EXAGGERATION_FACTOR",
]
