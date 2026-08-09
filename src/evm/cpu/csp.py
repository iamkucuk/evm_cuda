"""A complex steerable pyramid.

This is the decomposition the 2013 follow-up to the original method is built
on. Where the Laplacian pyramid used elsewhere in this library splits an image
by scale alone, this splits it by scale *and* direction, and — the part that
matters — gives each part a phase as well as an amplitude.

Phase is what makes it useful. Shift a striped pattern sideways and the
amplitude of the response barely changes, while its phase advances in
proportion to the shift. So movement can be read off as a change in phase, and
amplified by changing phase directly, rather than by scaling the difference
between neighbouring scales and hoping the result resembles a shift. That is
why the method produces fewer ripples and haloes at high amplification.

The filters are built in the frequency domain, where the construction is exact
and simple: a set of radial windows selecting scale, multiplied by a set of
angular windows selecting direction, arranged so that summing the squares of
all of them gives exactly one at every frequency. That last property is what
makes the decomposition invertible.

Reference: Portilla and Simoncelli, "A parametric texture model based on joint
statistics of complex wavelet coefficients", 2000; and Wadhwa and colleagues,
"Phase-Based Video Motion Processing", 2013.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["SteerablePyramid", "PyramidBands"]


@dataclass(frozen=True)
class PyramidBands:
    """What the decomposition produces.

    Attributes:
        highpass: what is finer than the finest oriented band, kept so the
            image can be rebuilt exactly.
        bands: one entry per scale, each a list with one complex array per
            direction. A coefficient's absolute value is how strongly that
            direction and scale is present; its angle is the phase movement is
            read from.
        lowpass: what is coarser than the coarsest oriented band.
    """

    highpass: np.ndarray
    bands: list[list[np.ndarray]]
    lowpass: np.ndarray

    @property
    def scales(self) -> int:
        return len(self.bands)

    @property
    def orientations(self) -> int:
        return len(self.bands[0]) if self.bands else 0


class SteerablePyramid:
    """Builds and inverts a complex steerable pyramid of a fixed size.

    The filters depend only on the image size, the number of scales and the
    number of directions, so they are built once here and reused for every
    frame of a clip. For a video that is the difference between building them
    once and building them several hundred times.

    Args:
        height: image height in pixels.
        width: image width.
        scales: how many scales to split into. More scales reach larger
            movements; the limit is set by the image size.
        orientations: how many directions per scale. Four is the usual choice;
            more gives finer angular selectivity at proportional cost.
    """

    def __init__(
        self, height: int, width: int, *, scales: int = 4, orientations: int = 4
    ) -> None:
        if scales < 1:
            raise ValueError(f"scales must be at least 1, got {scales}")
        if orientations < 1:
            raise ValueError(f"orientations must be at least 1, got {orientations}")

        limit = int(np.floor(np.log2(min(height, width)))) - 2
        if scales > limit:
            raise ValueError(
                f"{scales} scales is more than a {height}x{width} image "
                f"supports; the most it allows is {limit}"
            )

        self.height = int(height)
        self.width = int(width)
        self.scales = int(scales)
        self.orientations = int(orientations)
        self._filters = self._build_filters()

    # -- filter construction -------------------------------------------------

    def _polar_grid(self) -> tuple[np.ndarray, np.ndarray]:
        """Distance from the centre, and angle, for every frequency."""
        rows = np.fft.fftshift(np.fft.fftfreq(self.height)) * 2.0
        cols = np.fft.fftshift(np.fft.fftfreq(self.width)) * 2.0
        y, x = np.meshgrid(rows, cols, indexing="ij")
        radius = np.sqrt(x**2 + y**2)
        radius[self.height // 2, self.width // 2] = radius[
            self.height // 2, max(self.width // 2 - 1, 0)
        ]
        angle = np.arctan2(y, x)
        return radius, angle

    @staticmethod
    def _radial_window(radius: np.ndarray, centre: float) -> np.ndarray:
        """A smooth window selecting one octave around ``centre``.

        Raised-cosine in the logarithm of frequency, which makes each window
        one octave wide and, crucially, makes the squares of a set of them at
        octave spacing sum to one. Without that the decomposition would not be
        invertible.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            log_radius = np.log2(np.where(radius > 0, radius, 1e-12) / centre)
        window = np.cos(np.pi / 2.0 * np.clip(log_radius, -1.0, 1.0))
        window[np.abs(log_radius) >= 1.0] = 0.0
        return window

    def _angular_window(self, angle: np.ndarray, index: int) -> np.ndarray:
        """A window selecting one direction.

        The angular difference wraps modulo pi rather than 2pi, which is what
        makes the squares of the whole set sum to a constant — the property the
        decomposition's invertibility rests on. That constant depends only on
        how many directions there are, so it is divided out here and the set
        then sums to exactly one.
        """
        orientations = self.orientations
        centre = np.pi * index / orientations
        difference = np.mod(angle - centre + np.pi / 2, np.pi) - np.pi / 2
        window = np.cos(difference) ** (orientations - 1)

        # Measured rather than quoted: the constant the squares sum to, for
        # this many directions.
        probe = np.linspace(0, np.pi, 4096, endpoint=False)
        total = np.zeros_like(probe)
        for k in range(orientations):
            offset = (
                np.mod(probe - np.pi * k / orientations + np.pi / 2, np.pi) - np.pi / 2
            )
            total += np.cos(offset) ** (2 * (orientations - 1))
        return window / np.sqrt(float(total[0]))

    def _half_plane(self) -> np.ndarray:
        """One half of the frequency plane, which is what makes bands complex.

        A filter covering both halves responds identically to a pattern and its
        mirror image, so its output is real and carries no phase. Keeping one
        half makes each band an analytic signal, whose angle is the phase that
        movement is read from. The other half is restored during
        reconstruction using the symmetry every real image's spectrum has.
        """
        rows = np.fft.fftshift(np.fft.fftfreq(self.height))
        cols = np.fft.fftshift(np.fft.fftfreq(self.width))
        y, x = np.meshgrid(rows, cols, indexing="ij")
        return ((y > 0) | ((y == 0) & (x >= 0))).astype(np.float64)

    def _build_filters(self) -> dict:
        radius, angle = self._polar_grid()

        # Octave-spaced centres, finest first.
        centres = [0.5 / (2**scale) for scale in range(self.scales)]
        radial = [self._radial_window(radius, c) for c in centres]
        angular = [self._angular_window(angle, k) for k in range(self.orientations)]

        # The angular set sums to one, so the oriented bands together cover
        # exactly the sum of the squared radial windows. Whatever is left is
        # split into what is finer than the finest band and what is coarser
        # than the coarsest, so that everything together covers the plane
        # exactly once.
        covered = np.clip(sum(r**2 for r in radial), 0.0, 1.0)
        residual = np.clip(1.0 - covered, 0.0, None)
        finest = centres[0]
        highpass = np.sqrt(np.where(radius > finest, residual, 0.0))
        lowpass = np.sqrt(np.where(radius <= finest, residual, 0.0))

        return {
            "radial": radial,
            "angular": angular,
            "highpass": highpass,
            "lowpass": lowpass,
            "half": self._half_plane(),
        }

    # -- decomposition and reconstruction ------------------------------------

    def decompose(self, image: np.ndarray) -> PyramidBands:
        """Split one greyscale image into oriented complex bands."""
        if image.shape != (self.height, self.width):
            raise ValueError(
                f"image is {image.shape}, but this pyramid was built for "
                f"{(self.height, self.width)}"
            )
        spectrum = np.fft.fftshift(np.fft.fft2(image.astype(np.float64)))
        f = self._filters

        highpass = np.real(np.fft.ifft2(np.fft.ifftshift(spectrum * f["highpass"])))
        lowpass = np.real(np.fft.ifft2(np.fft.ifftshift(spectrum * f["lowpass"])))

        bands = []
        for radial in f["radial"]:
            per_direction = []
            for angular in f["angular"]:
                filtered = spectrum * radial * angular * f["half"]
                per_direction.append(np.fft.ifft2(np.fft.ifftshift(filtered)))
            bands.append(per_direction)
        return PyramidBands(highpass=highpass, bands=bands, lowpass=lowpass)

    def reconstruct(self, pyramid: PyramidBands) -> np.ndarray:
        """Put the bands back together into an image.

        Exact to floating-point rounding when the bands are unmodified, which
        is what :mod:`tests.test_phase_based` checks first: a decomposition
        that cannot be inverted cannot be used to modify anything.
        """
        f = self._filters
        residuals = (
            np.fft.fftshift(np.fft.fft2(pyramid.highpass)) * f["highpass"]
            + np.fft.fftshift(np.fft.fft2(pyramid.lowpass)) * f["lowpass"]
        )

        oriented = np.zeros_like(residuals)
        for radial, per_direction in zip(f["radial"], pyramid.bands):
            for angular, band in zip(f["angular"], per_direction):
                oriented = oriented + (
                    np.fft.fftshift(np.fft.fft2(band)) * radial * angular * f["half"]
                )

        # The oriented bands only ever covered half the plane. A real image's
        # spectrum is symmetric about the origin — each frequency's mirror is
        # its complex conjugate — so the discarded half is recovered rather
        # than needing to have been stored.
        mirrored = np.conj(_mirror(oriented))
        return np.real(np.fft.ifft2(np.fft.ifftshift(residuals + oriented + mirrored)))


def _mirror(spectrum: np.ndarray) -> np.ndarray:
    """Reflect a centred spectrum through the origin.

    In a centred layout the frequency at (i, j) has its mirror at (-i, -j),
    which after the shift is (H-i mod H, W-j mod W). Rolling by one after
    reversing is what lines those up.
    """
    return np.roll(np.roll(spectrum[::-1, ::-1], 1, axis=0), 1, axis=1)
