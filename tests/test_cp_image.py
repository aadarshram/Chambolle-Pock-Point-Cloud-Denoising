"""Checks for TV-L1 and ROF denoising on a 2D image (Chambolle and Pock, 2011)."""

import numpy as np
import pytest

from cp_image import (
    OPERATOR_NORM_BOUND,
    denoise_image,
    gradient,
    gradient_adjoint,
)

def _spike(size=7):
    image = np.zeros((size, size))
    image[size // 2, size // 2] = 1.0
    return image


def _salt_and_pepper_square():
    rng = np.random.default_rng(0)
    clean = np.full((16, 16), 0.2)
    clean[4:12, 4:12] = 0.8
    picked = rng.random(clean.shape) < 0.2
    salt = rng.random(clean.shape) < 0.5
    noisy = clean.copy()
    noisy[picked] = salt[picked].astype(float)
    return clean, noisy


def test_gradient_is_the_forward_difference_with_neumann_boundary():
    image = np.random.default_rng(1).normal(size=(3, 4))
    step = gradient(image)
    assert step.shape == (2, 3, 4)
    assert step[0, :, :-1] == pytest.approx(np.diff(image, axis=1))
    assert step[1, :-1, :] == pytest.approx(np.diff(image, axis=0))
    assert np.all(step[0, :, -1] == 0.0)
    assert np.all(step[1, -1, :] == 0.0)


def test_gradient_adjoint_is_the_transpose():
    rng = np.random.default_rng(2)
    image = rng.normal(size=(4, 5))
    field = rng.normal(size=(2, 4, 5))
    assert np.sum(gradient(image) * field) == pytest.approx(np.sum(image * gradient_adjoint(field)))


def test_gradient_of_a_constant_image_is_zero_and_adjoint_has_zero_sum():
    assert np.all(gradient(np.full((3, 3), 0.7)) == 0.0)
    field = np.random.default_rng(4).normal(size=(2, 5, 6))
    assert gradient_adjoint(field).sum() == pytest.approx(0.0, abs=1e-12)


def test_norm_bound_is_the_paper_bound_and_covers_power_iteration():
    assert OPERATOR_NORM_BOUND == pytest.approx(np.sqrt(8.0))
    vector = np.random.default_rng(3).normal(size=(6, 5))
    vector /= np.linalg.norm(vector)
    for _ in range(300):
        vector = gradient_adjoint(gradient(vector))
        scale = np.linalg.norm(vector)
        vector /= scale
    assert OPERATOR_NORM_BOUND >= np.sqrt(scale) - 1e-6


@pytest.mark.parametrize("lam", [2.0, 3.0])
def test_spike_below_the_threshold_is_removed(lam):
    restored = denoise_image(_spike(), lam, n_iter=20000, tol=1e-10)
    assert restored == pytest.approx(np.zeros((7, 7)), abs=1e-4)


@pytest.mark.parametrize("lam", [4.0, 5.0])
def test_spike_above_the_threshold_is_kept(lam):
    restored = denoise_image(_spike(), lam, n_iter=20000, tol=1e-10)
    assert restored == pytest.approx(_spike(), abs=1e-4)


def test_salt_and_pepper_error_drops():
    clean, noisy = _salt_and_pepper_square()
    restored = denoise_image(noisy, lam=1.0, n_iter=5000, tol=1e-8)
    noisy_mse = np.mean((noisy - clean) ** 2)
    restored_mse = np.mean((restored - clean) ** 2)
    assert restored_mse < 0.1 * noisy_mse


def test_rof_preserves_the_mean_and_smooths():
    _, noisy = _salt_and_pepper_square()
    restored = denoise_image(noisy, 4.0, data="l2", n_iter=3000, tol=1e-10)
    assert restored.mean() == pytest.approx(noisy.mean(), abs=1e-9)
    assert np.abs(gradient(restored)).sum() < np.abs(gradient(noisy)).sum()


def test_l1_and_l2_give_different_restorations():
    _, noisy = _salt_and_pepper_square()
    l1 = denoise_image(noisy, 1.0, data="l1", n_iter=300, tol=1e-8)
    l2 = denoise_image(noisy, 1.0, data="l2", n_iter=300, tol=1e-8)
    assert not np.allclose(l1, l2)


def test_single_pixel_image_is_returned_unchanged():
    assert denoise_image(np.array([[0.3]]), lam=1.0) == pytest.approx(np.array([[0.3]]))


@pytest.mark.parametrize(
    "image, kwargs",
    [
        (np.zeros(4), {"lam": 1.0}),
        (np.zeros((2, 2, 1)), {"lam": 1.0}),
        (np.array([[0.0, np.nan]]), {"lam": 1.0}),
        (np.zeros((2, 2)), {"lam": -1.0}),
        (np.zeros((2, 2)), {"lam": 1.0, "data": "l3"}),
        (np.zeros((2, 2)), {"lam": 1.0, "data": 1}),
    ],
)
def test_bad_image_input_raises(image, kwargs):
    with pytest.raises(ValueError):
        denoise_image(image, **kwargs)
