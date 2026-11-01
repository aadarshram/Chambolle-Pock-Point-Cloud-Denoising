"""Checks for the problem-independent Chambolle-Pock iteration (Algorithm 1 of the 2011 paper)."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from cp_solver import chambolle_pock
from cp_solver.prox import prox_l1_data, prox_quadratic_data

SRC = Path(__file__).resolve().parents[1] / "src"


def _soft_threshold_problem(b, mu):
    """
    min_x mu * ||x||_1 + 0.5 * ||x - b||^2 with K = identity.

    F(z) = mu * ||z||_1 has F* = indicator of {|y| <= mu}, so prox_{sigma F*} is a clip.
    G(x) = 0.5 * ||x - b||^2 has prox_{tau G}(x) = (x + tau * b) / (1 + tau).
    """
    return dict(
        x0=b.copy(),
        K=lambda x: x.copy(),
        K_adjoint=lambda y: y.copy(),
        prox_f_conj=lambda y, sigma: np.clip(y, -mu, mu),
        prox_g=lambda x, tau: (x + tau * b) / (1.0 + tau),
        norm_K=1.0,
    )


def test_identity_operator_recovers_the_soft_threshold():
    b = np.array([-3.0, -0.4, 0.0, 0.2, 1.0, 5.0])
    mu = 0.5
    restored = chambolle_pock(**_soft_threshold_problem(b, mu), n_iter=5000, tol=1e-12)
    expected = np.sign(b) * np.maximum(np.abs(b) - mu, 0.0)
    assert restored == pytest.approx(expected, abs=1e-6)


def test_arrays_of_any_shape_are_accepted_and_the_dual_takes_the_shape_of_K():
    rng = np.random.default_rng(0)
    b = rng.normal(size=(4, 5))
    mu = 0.3
    problem = _soft_threshold_problem(b, mu)
    # K stacks two copies of x. The dual then has shape (2, 4, 5), and the norm is sqrt(2).
    problem["K"] = lambda x: np.stack([x, x])
    problem["K_adjoint"] = lambda y: y[0] + y[1]
    problem["prox_f_conj"] = lambda y, sigma: np.clip(y, -mu, mu)
    problem["norm_K"] = np.sqrt(2.0)
    restored = chambolle_pock(**problem, n_iter=20000, tol=1e-13)
    expected = np.sign(b) * np.maximum(np.abs(b) - 2.0 * mu, 0.0)
    # F(Kx) = mu * (||x||_1 + ||x||_1), so the threshold doubles.
    assert restored.shape == b.shape
    assert restored == pytest.approx(expected, abs=1e-5)


def test_the_steps_satisfy_the_step_size_condition_and_reach_the_callbacks():
    seen = {"sigma": set(), "tau": set()}
    b = np.array([1.0, -2.0])
    problem = _soft_threshold_problem(b, 0.5)

    def prox_f_conj(y, sigma):
        seen["sigma"].add(sigma)
        return np.clip(y, -0.5, 0.5)

    def prox_g(x, tau):
        seen["tau"].add(tau)
        return (x + tau * b) / (1.0 + tau)

    problem.update(prox_f_conj=prox_f_conj, prox_g=prox_g, norm_K=2.0)
    chambolle_pock(**problem, n_iter=5)
    (sigma,), (tau,) = seen["sigma"], seen["tau"]
    assert tau * sigma * 2.0**2 < 1.0


def test_the_solver_imports_nothing_problem_specific():
    code = (
        "import sys; import cp_solver, cp_solver.prox; "
        "bad = [m for m in ('cp_pointcloud', 'cp_image', 'scipy') if m in sys.modules]; "
        "sys.exit(len(bad))"
    )
    env = dict(os.environ, PYTHONPATH=str(SRC))
    assert subprocess.run([sys.executable, "-c", code], env=env).returncode == 0


def test_quadratic_data_prox_is_the_weighted_average():
    observed = np.array([1.0, 3.0])
    point = np.array([0.0, 0.0])
    assert prox_quadratic_data(point, observed, 1.0) == pytest.approx([0.5, 1.5])


def test_l1_data_prox_soft_thresholds_toward_the_data():
    observed = np.ones(3)
    point = np.array([3.0, 1.2, -1.0])
    assert prox_l1_data(point, observed, 0.5) == pytest.approx([2.5, 1.0, -0.5])


@pytest.mark.parametrize(
    "change",
    [
        {"theta": 1.5},
        {"theta": -0.1},
        {"n_iter": 0},
        {"n_iter": 2.5},
        {"tol": 0.0},
        {"tol": float("nan")},
        {"norm_K": 0.0},
        {"norm_K": -1.0},
        {"norm_K": float("inf")},
        {"prox_g": 3},
    ],
)
def test_bad_input_raises(change):
    problem = _soft_threshold_problem(np.array([1.0, 2.0]), 0.5)
    problem.update(change)
    with pytest.raises(ValueError):
        chambolle_pock(**problem)
