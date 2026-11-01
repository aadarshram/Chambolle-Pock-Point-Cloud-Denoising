"""
Total-variation denoising of a grayscale image (Chambolle and Pock, 2011).

The image u is an (H, W) array. K = grad is the forward difference with a
Neumann boundary, and it maps u to an array of shape (2, H, W). The total
variation is the sum over pixels of the Euclidean norm of the two differences,
which is the discrete isotropic TV of section 6.1. ||K||^2 <= 8.

The data term is lam * ||u - f||_1 for TV-L1 (the default), or
(lam / 2) * ||u - f||^2 for ROF. The iteration itself is `cp_solver.chambolle_pock`.
"""

import numpy as np

from cp_solver import RELATIVE_CHANGE_TOL, chambolle_pock
from cp_solver.prox import prox_l1_data, prox_quadratic_data

# Section 6.1 of the paper: ||grad||^2 <= 8.
OPERATOR_NORM_BOUND = float(np.sqrt(8.0))

#--------------------------------------------------------------------------------


def gradient(image):
    """
    Forward differences of an (H, W) image, as an array of shape (2, H, W).

    Entry 0 is u[r, c + 1] - u[r, c] and entry 1 is u[r + 1, c] - u[r, c].
    The difference is 0 on the last column and the last row.
    """
    values = _as_image(image, "image")
    step = np.zeros((2,) + values.shape)
    step[0, :, :-1] = values[:, 1:] - values[:, :-1]
    step[1, :-1, :] = values[1:, :] - values[:-1, :]
    return step


def gradient_adjoint(field):
    """
    Adjoint of `gradient`, which is minus the divergence. The input has shape (2, H, W).
    """
    values = np.asarray(field, dtype=float)
    if values.ndim != 3 or values.shape[0] != 2:
        raise ValueError("field must have shape (2, H, W)")
    result = np.zeros(values.shape[1:])
    result[:, 1:] += values[0, :, :-1]
    result[:, :-1] -= values[0, :, :-1]
    result[1:, :] += values[1, :-1, :]
    result[:-1, :] -= values[1, :-1, :]
    return result


def total_variation(image):
    """
    Isotropic total variation: the sum over pixels of |grad u|.
    """
    return float(np.sum(np.sqrt(np.sum(gradient(image) ** 2, axis=0))))


def denoise_image(image, lam, data="l1", n_iter=5000, theta=1.0, tol=RELATIVE_CHANGE_TOL):
    """
    Denoise a grayscale image with isotropic total variation.

    `data` is "l1" for the TV-L1 model or "l2" for the ROF model.
    `lam` multiplies the data term.
    """
    observed = _as_image(image, "image")
    lam = _as_nonnegative(lam, "lam")
    prox_data = _data_prox(data)
    return chambolle_pock(
        observed,
        K=gradient,
        K_adjoint=gradient_adjoint,
        prox_f_conj=_project_onto_unit_balls,
        prox_g=lambda point, tau: prox_data(point, observed, tau * lam),
        norm_K=OPERATOR_NORM_BOUND,
        n_iter=n_iter,
        theta=theta,
        tol=tol,
    )


def tvl1_energy(observed, restored, lam):
    """
    TV(u) + lam * ||u - f||_1 for u = `restored` and f = `observed`.
    """
    noisy = _as_image(observed, "observed")
    candidate = _as_image(restored, "restored")
    if noisy.shape != candidate.shape:
        raise ValueError("observed and restored must have the same shape")
    lam = _as_nonnegative(lam, "lam")
    return total_variation(candidate) + lam * float(np.sum(np.abs(candidate - noisy)))


#--------------------------------------------------------------------------------


def _project_onto_unit_balls(field, _sigma):
    # Prox of sigma * F*, where F* is the indicator of {|p(r, c)| <= 1 for every pixel}.
    # The projection divides each pixel's pair of values by max(1, |p|).
    norms = np.sqrt(np.sum(field**2, axis=0))
    return field / np.maximum(1.0, norms)


def _data_prox(data):
    if not isinstance(data, str) or data.strip().lower() not in ("l1", "l2"):
        raise ValueError("data must be 'l1' or 'l2'")
    return prox_l1_data if data.strip().lower() == "l1" else prox_quadratic_data


def _as_image(array, name):
    values = np.asarray(array, dtype=float)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError(f"{name} must have shape (H, W) with H >= 1 and W >= 1")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    return values


def _as_nonnegative(value, name):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number >= 0") from exc
    if not np.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be a finite number >= 0")
    return number
