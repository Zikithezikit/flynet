"""Receptive field module for spatial filtering.

Provides Gaussian and Manhattan-distance weighted kernels that mimic
retinal / early visual processing receptive fields.
"""

from __future__ import annotations

import numpy as np

try:
    from scipy.signal import fft_convolve as _fft_convolve  # type: ignore[import-untyped]
except ImportError:
    _fft_convolve = None


class ReceptiveField:
    """2D Gaussian-weighted receptive field (on-center or off-center).

    Attributes:
        kernel: The 2D kernel array used for convolution.
    """

    def __init__(
        self,
        size: int = 5,
        sigma: float | None = None,
        on_center: bool = True,
    ) -> None:
        """Initialise the receptive field.

        Args:
            size (int): Kernel side length (must be odd).
            sigma (float | None): Standard deviation of the Gaussian. Defaults to
                ``size / 6``.
            on_center (bool): If ``True`` the centre is positive and surround is
                negative (on-centre). If ``False`` the signs are flipped
                (off-centre / surround-dominant).

        Raises:
            ValueError: If ``size`` is not odd.
        """
        if size % 2 == 0:
            raise ValueError(f"size must be odd, got {size}")
        self._size = size
        if sigma is None:
            sigma = size / 6.0
        self._sigma = sigma
        self._on_center = on_center
        self._kernel = self._build_kernel()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_kernel(self) -> np.ndarray:
        half = self._size // 2
        ax = np.arange(-half, half + 1, dtype=np.float64)
        xx, yy = np.meshgrid(ax, ax, indexing="ij")
        g = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * self._sigma ** 2))
        # Normalise so centre weight is 1
        g /= g.max()
        # Subtract a low-surround to create centre-surround contrast
        surround = 0.25 * g
        kernel = g - surround
        # Zero-mean so that uniform illumination produces zero response
        kernel -= kernel.mean()
        if not self._on_center:
            kernel = -kernel
        return kernel

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def kernel(self) -> np.ndarray:
        """The 2D kernel array."""
        return self._kernel

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Convolve an image with this receptive field kernel.

        Args:
            image (np.ndarray): 2D array representing the input image or feature map.

        Returns:
            np.ndarray: Convolved potential map of the same spatial size as the input.

        Raises:
            ValueError: If ``image`` is not a 2D array.
        """
        image = np.asarray(image, dtype=np.float64)
        if image.ndim == 1:
            image = image.reshape(1, -1)
        if image.ndim != 2:
            raise ValueError(
                f"image must be 2D, got {image.ndim}D with shape {image.shape}"
            )
        return apply_rf(image, self._kernel)


def manhattan_rf(size: int = 5) -> np.ndarray:
    """Create an on-centred receptive field weighted by Manhattan distance.

    Follows the pattern from the Spiking-Neural-Network ``receptive_field.py``:
    ``weight = -0.375 * distance + 1`` for each cell in the kernel.

    Args:
        size (int): Kernel side length (must be odd).

    Returns:
        np.ndarray: 2D array of shape ``(size, size)``.

    Raises:
        ValueError: If ``size`` is not odd.
    """
    if size % 2 == 0:
        raise ValueError(f"size must be odd, got {size}")
    half = size // 2
    kernel = np.zeros((size, size), dtype=np.float64)
    for i in range(size):
        for j in range(size):
            d = abs(half - i) + abs(half - j)
            kernel[i, j] = -0.375 * d + 1.0
    return kernel


def apply_rf(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve an image with a receptive-field kernel.

    Uses ``scipy.signal.fftconvolve`` when available, otherwise falls back
    to a zero-padded sliding-window implementation.

    Args:
        image (np.ndarray): 2D input array.
        kernel (np.ndarray): 2D convolution kernel (should be smaller than *image*).

    Returns:
        np.ndarray: Potential map of the same spatial size as *image*.

    Raises:
        ValueError: If ``image`` or ``kernel`` is not 2D.
    """
    image = np.asarray(image, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)

    if image.ndim != 2 or kernel.ndim != 2:
        raise ValueError(
            f"image and kernel must be 2D, got shapes {image.shape} and {kernel.shape}"
        )

    # Prefer scipy's FFT-based convolution when available
    if _fft_convolve is not None:
        from scipy.signal import fft_convolve

        conv = fft_convolve(image, kernel, mode="same")
        return conv

    # Manual zero-padded sliding window
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant")
    out = np.empty_like(image)
    ih, iw = image.shape

    for i in range(ih):
        for j in range(iw):
            region = padded[i : i + kh, j : j + kw]
            out[i, j] = np.sum(region * kernel)

    return out
