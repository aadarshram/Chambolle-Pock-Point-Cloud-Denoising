"""
Color noise and graph ROF denoising of a color stored at each point.
"""

import numpy as np

from cp_pointcloud import chambolle_pock, knn_graph
from cp_pointcloud.denoise import RELATIVE_CHANGE_TOL

#--------------------------------------------------------------------------------


def add_color_noise(colors, noise_std, rng):
    """
    Add Gaussian noise to each color channel and clip the result to [0, 1].
    """
    signal = _as_colors(colors)
    scale = _as_nonnegative(noise_std, "noise_std")
    _as_generator(rng)
    noisy = signal + rng.normal(scale=scale, size=signal.shape)
    return np.clip(noisy, 0.0, 1.0)


def color_mse(restored, clean):
    """
    Mean square error between matched color arrays of the same shape.
    """
    left = _as_colors(restored)
    right = _as_colors(clean)
    if left.shape != right.shape:
        raise ValueError("restored and clean must have the same shape")
    return float(np.mean((left - right) ** 2))


def denoise_color(
    points, colors, k, lam, n_iter=5000, tol=RELATIVE_CHANGE_TOL, weight="equal", bandwidth=None
):
    """
    Denoise the color channels on the k-nearest-neighbor graph of `points`.
    
    `lam` multiplies the data term of each color channel. `weight` is
    "equal" or "distance", as in `knn_graph`.
    """
    cloud = _as_points(points)
    signal = _as_colors(colors)
    if cloud.shape[0] != signal.shape[0]:
        raise ValueError("points and colors must have the same number of rows")
    sources, targets, weights = knn_graph(cloud, k, weight=weight, bandwidth=bandwidth)
    return chambolle_pock(
        signal, sources, targets, weights, lam, n_iter=n_iter, tol=tol
    )


#--------------------------------------------------------------------------------


def _as_points(points):
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or values.shape[0] < 1:
        raise ValueError("points must have shape (N, 3) with N >= 1")
    if not np.all(np.isfinite(values)):
        raise ValueError("points must contain only finite values")
    return values


def _as_colors(colors):
    values = np.asarray(colors, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or values.shape[0] < 1:
        raise ValueError("colors must have shape (N, 3) with N >= 1")
    if not np.all(np.isfinite(values)):
        raise ValueError("colors must contain only finite values")
    return values


def _as_generator(rng):
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a numpy Generator")


def _as_nonnegative(value, name):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number >= 0") from exc
    if not np.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be a finite number >= 0")
    return number
