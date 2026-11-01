"""Checks for TV-L1 denoising on the pixel grid (Chambolle and Pock, 2011)."""

import numpy as np
import pytest

from cp_pointcloud import chambolle_pock, denoise_image, grid_graph, tvl1_energy
from cp_pointcloud.denoise import _data_prox, apply_adjoint, apply_gradient, operator_norm_bound

# A single pixel of height h on a flat image costs (2 + sqrt(2)) * h in total
# variation and lam * h in the data term to remove, so TV-L1 removes it
# exactly when lam < 2 + sqrt(2).
SPIKE_THRESHOLD = 2.0 + np.sqrt(2.0)


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


def test_grid_has_one_edge_per_forward_difference():
    sources, targets, weights = grid_graph(3, 4)
    assert sources.shape[0] == 2 * 3 * 4 - 3 - 4
    assert np.all(weights == 1.0)
    edges = set(zip(sources.tolist(), targets.tolist()))
    assert (0, 1) in edges and (0, 4) in edges
    # No backward edges, and no edge wraps from the end of a row to the next.
    assert (1, 0) not in edges
    assert (3, 4) not in edges


def test_grid_gradient_is_the_forward_difference_with_neumann_boundary():
    rng = np.random.default_rng(1)
    image = rng.normal(size=(3, 4))
    sources, targets, weights = grid_graph(3, 4)
    gradient = apply_gradient(image.ravel(), sources, targets, weights)
    across = np.zeros((3, 4))
    across[:, :-1] = np.diff(image, axis=1)
    down = np.zeros((3, 4))
    down[:-1, :] = np.diff(image, axis=0)
    # Squared jumps out of a pixel sum to that pixel's |grad u|^2.
    squared = np.zeros(12)
    np.add.at(squared, sources, gradient**2)
    assert squared == pytest.approx((across**2 + down**2).ravel())


def test_grid_adjoint_is_the_transpose():
    rng = np.random.default_rng(2)
    image = rng.normal(size=(4, 5))
    sources, targets, weights = grid_graph(4, 5)
    dual = rng.normal(size=sources.shape[0])
    gradient = apply_gradient(image.ravel(), sources, targets, weights)
    adjoint = apply_adjoint(dual, sources, targets, weights, image.size)
    assert np.sum(gradient * dual) == pytest.approx(np.sum(image.ravel() * adjoint))


def test_grid_norm_bound_is_the_paper_bound_and_covers_power_iteration():
    rng = np.random.default_rng(3)
    sources, targets, weights = grid_graph(6, 5)
    bound = operator_norm_bound(sources, targets, weights, 30)
    assert bound == pytest.approx(np.sqrt(8.0))
    vector = rng.normal(size=30)
    vector /= np.linalg.norm(vector)
    for _ in range(200):
        vector = apply_adjoint(
            apply_gradient(vector, sources, targets, weights), sources, targets, weights, 30
        )
        scale = np.linalg.norm(vector)
        vector /= scale
    assert bound >= np.sqrt(scale) - 1e-6


def test_l1_prox_soft_thresholds_toward_the_data():
    observed = np.ones(3)
    point = np.array([3.0, 1.2, -1.0])
    restored = _data_prox(point, observed, 0.5, "l1")
    assert restored == pytest.approx(np.array([2.5, 1.0, -0.5]))


def test_energy_of_a_spike_matches_the_threshold():
    spike = _spike()
    assert tvl1_energy(spike, spike, lam=1.0) == pytest.approx(SPIKE_THRESHOLD)
    assert tvl1_energy(spike, np.zeros_like(spike), lam=2.0) == pytest.approx(2.0)


@pytest.mark.parametrize("lam", [2.0, 3.0])
def test_spike_below_the_threshold_is_removed(lam):
    restored = denoise_image(_spike(), lam, n_iter=20000, tol=1e-10)
    assert restored == pytest.approx(np.zeros((7, 7)), abs=1e-4)


@pytest.mark.parametrize("lam", [4.0, 5.0])
def test_spike_above_the_threshold_is_kept(lam):
    restored = denoise_image(_spike(), lam, n_iter=20000, tol=1e-10)
    assert restored == pytest.approx(_spike(), abs=1e-4)


def test_salt_and_pepper_error_and_energy_drop():
    clean, noisy = _salt_and_pepper_square()
    restored = denoise_image(noisy, lam=1.0, n_iter=5000, tol=1e-8)
    noisy_mse = np.mean((noisy - clean) ** 2)
    restored_mse = np.mean((restored - clean) ** 2)
    assert restored_mse < 0.1 * noisy_mse
    assert tvl1_energy(noisy, restored, 1.0) <= tvl1_energy(noisy, noisy, 1.0)


def test_l2_is_the_default_data_term():
    _, noisy = _salt_and_pepper_square()
    sources, targets, weights = grid_graph(*noisy.shape)
    kwargs = dict(n_iter=300, tol=1e-8)
    default = chambolle_pock(noisy.ravel(), sources, targets, weights, 4.0, **kwargs)
    quadratic = chambolle_pock(
        noisy.ravel(), sources, targets, weights, 4.0, data="l2", **kwargs
    )
    assert default == pytest.approx(quadratic, abs=0.0)
    image = denoise_image(noisy, 4.0, data="l2", **kwargs)
    assert image.ravel() == pytest.approx(default, abs=0.0)


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


@pytest.mark.parametrize("height, width", [(0, 3), (3, 0), (2.5, 3), (True, 3)])
def test_bad_grid_size_raises(height, width):
    with pytest.raises(ValueError):
        grid_graph(height, width)


def test_energy_shape_mismatch_raises():
    with pytest.raises(ValueError):
        tvl1_energy(np.zeros((2, 2)), np.zeros((2, 3)), lam=1.0)
