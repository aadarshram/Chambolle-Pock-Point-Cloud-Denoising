"""
Proximal maps of the two data terms used in the experiments.

Both act on every entry separately, so they do not depend on the shape of the
signal or on K. `step` is tau * lam.
"""

import numpy as np


def prox_quadratic_data(point, observed, step):
    """
    Prox of step * 0.5 * (u - f)^2 at `point`, the ROF data term: a weighted average.
    """
    return (point + step * observed) / (1.0 + step)


def prox_l1_data(point, observed, step):
    """
    Prox of step * |u - f| at `point`, the TV-L1 data term: soft-threshold the residual.
    """
    residual = point - observed
    return observed + np.sign(residual) * np.maximum(np.abs(residual) - step, 0.0)
