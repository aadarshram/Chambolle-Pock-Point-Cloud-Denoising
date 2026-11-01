"""
The Chambolle-Pock primal-dual iteration, Algorithm 1 of Chambolle and Pock
(2011), for min_x F(Kx) + G(x).

Nothing here depends on the problem. The caller supplies the linear operator
K, its adjoint, the proximal maps of F* and G, and a bound on ||K||. Arrays
may have any shape. The dual variable takes the shape of K x.
"""

import numpy as np

# tau * sigma * ||K||^2 = STEP_FRACTION**2 < 1.
STEP_FRACTION = 0.99
RELATIVE_CHANGE_TOL = 1e-5

#--------------------------------------------------------------------------------


def chambolle_pock(
    x0,
    K,
    K_adjoint,
    prox_f_conj,
    prox_g,
    norm_K,
    n_iter=5000,
    theta=1.0,
    tol=RELATIVE_CHANGE_TOL,
):
    """
    Run Algorithm 1 from the primal start `x0` and a zero dual start.

    K(x) is the linear operator and K_adjoint(y) its adjoint. prox_f_conj(y, sigma)
    is the proximal map of sigma * F*, and prox_g(x, tau) that of tau * G.
    `norm_K` is an upper bound on ||K||. The steps are tau = sigma = STEP_FRACTION / norm_K.

    Each iteration is

        y  <- prox_f_conj(y + sigma * K(x_bar), sigma)
        x  <- prox_g(x - tau * K_adjoint(y), tau)
        x_bar <- x + theta * (x - x_previous)

    It stops when both x and y change by less than the relative `tol`,
    or after `n_iter` iterations. Returns x.
    """
    primal = np.array(x0, dtype=float)
    if not np.all(np.isfinite(primal)):
        raise ValueError("x0 must contain only finite values")
    _as_callable(K, "K")
    _as_callable(K_adjoint, "K_adjoint")
    _as_callable(prox_f_conj, "prox_f_conj")
    _as_callable(prox_g, "prox_g")
    norm_K = _as_positive(norm_K, "norm_K")
    n_iter = _as_positive_int(n_iter, "n_iter")
    theta = _as_theta(theta)
    tol = _as_positive(tol, "tol")
    tau = STEP_FRACTION / norm_K
    sigma = STEP_FRACTION / norm_K
    extrapolated = primal.copy()
    dual = np.zeros_like(K(primal))
    for _ in range(n_iter):
        previous_dual = dual
        dual = prox_f_conj(dual + sigma * K(extrapolated), sigma)
        previous = primal
        primal = prox_g(primal - tau * K_adjoint(dual), tau)
        extrapolated = primal + theta * (primal - previous)
        if _has_settled(primal, previous, tol) and _has_settled(dual, previous_dual, tol):
            break
    return primal


#--------------------------------------------------------------------------------


def _has_settled(current, previous, tol):
    # Relative change. A variable that stays at 0 has settled.
    return np.linalg.norm(current - previous) <= tol * np.linalg.norm(previous)


def _as_callable(value, name):
    if not callable(value):
        raise ValueError(f"{name} must be callable")


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


def _as_positive(value, name):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number > 0") from exc
    if not np.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a finite number > 0")
    return number


def _as_theta(value):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("theta must lie in [0, 1]") from exc
    if not np.isfinite(number) or number < 0.0 or number > 1.0:
        raise ValueError("theta must lie in [0, 1]")
    return number
