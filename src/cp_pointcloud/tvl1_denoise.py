"""
Total-variation denoising of a grayscale image on its pixel grid.

The default model is TV-L1 from Chambolle and Pock (2011):
min_u TV(u) + lam * ||u - f||_1. The grid has one edge from each pixel to its
right neighbor and one to the pixel below, each of weight 1. The vertex norm
of equation (7) then stacks the two forward differences, which is the discrete
isotropic total variation of Chambolle and Pock. A pixel on the last row or
column has no edge in that direction, so that difference is 0 (a Neumann
boundary). Every pixel has at most two outgoing and two incoming edges, so
`operator_norm_bound` gives sqrt(8), the bound L^2 <= 8 of the paper.
"""

import numpy as np

from cp_pointcloud.denoise import RELATIVE_CHANGE_TOL, chambolle_pock, total_variation

#--------------------------------------------------------------------------------


def grid_graph(height, width):
    """
    Forward-difference edges of a height x width pixel grid in row-major order.

    Returns sources, targets, and unit weights, as `knn_graph` does.
    """
    height = _as_positive_int(height, "height")
    width = _as_positive_int(width, "width")
    index = np.arange(height * width).reshape(height, width)
    sources = np.concatenate([index[:, :-1].ravel(), index[:-1, :].ravel()])
    targets = np.concatenate([index[:, 1:].ravel(), index[1:, :].ravel()])
    return sources, targets, np.ones(sources.shape[0])


def denoise_image(
    image, lam, data="l1", n_iter=5000, theta=1.0, tol=RELATIVE_CHANGE_TOL
):
    """
    Denoise a grayscale image with isotropic total variation on its pixel grid.

    `data` is "l1" for the TV-L1 model or "l2" for the ROF model.
    `lam` multiplies the data term.
    """
    observed = _as_image(image, "image")
    sources, targets, weights = grid_graph(*observed.shape)
    restored = chambolle_pock(
        observed.ravel(),
        sources,
        targets,
        weights,
        lam,
        n_iter=n_iter,
        theta=theta,
        tol=tol,
        data=data,
    )
    return restored.reshape(observed.shape)


def tvl1_energy(observed, restored, lam):
    """
    TV(u) + lam * ||u - f||_1 for u = `restored` and f = `observed`.
    """
    noisy = _as_image(observed, "observed")
    candidate = _as_image(restored, "restored")
    if noisy.shape != candidate.shape:
        raise ValueError("observed and restored must have the same shape")
    lam = _as_nonnegative(lam, "lam")
    sources, targets, weights = grid_graph(*candidate.shape)
    variation = total_variation(candidate.ravel(), sources, targets, weights)
    return variation + lam * float(np.sum(np.abs(candidate - noisy)))


#--------------------------------------------------------------------------------


def _as_image(array, name):
    values = np.asarray(array, dtype=float)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError(f"{name} must have shape (H, W) with H >= 1 and W >= 1")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    return values


def _as_positive_int(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer >= 1")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer >= 1") from exc
    if number != value or number < 1:
        raise ValueError(f"{name} must be an integer >= 1")
    return number


def _as_nonnegative(value, name):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number >= 0") from exc
    if not np.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be a finite number >= 0")
    return number
