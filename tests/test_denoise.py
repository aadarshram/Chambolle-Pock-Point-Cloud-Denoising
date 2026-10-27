"""Checks for the 2015 graph ROF model and its Chambolle-Pock iteration."""

import numpy as np
import pytest

from cp_pointcloud import chambolle_pock, denoise_point_cloud, knn_graph
from cp_pointcloud.denoise import apply_adjoint, apply_gradient, operator_norm_bound, total_variation


def _path_edges(n):
    left = np.arange(n - 1)
    right = left + 1
    sources = np.concatenate([left, right])
    targets = np.concatenate([right, left])
    weights = np.ones(sources.shape[0])
    return sources, targets, weights


def _rof(signal, restored, sources, targets, weights, lam):
    fidelity = 0.5 * lam * np.sum((restored - signal) ** 2)
    variation = 0.0
    for channel in range(signal.shape[1]):
        variation += total_variation(restored[:, channel], sources, targets, weights)
    return fidelity + variation


def test_three_vertex_total_variation_matches_equation_7():
    signal = np.array([0.0, 1.0, 3.0])
    sources, targets, weights = _path_edges(3)
    variation = total_variation(signal, sources, targets, weights)
    assert variation == pytest.approx(1.0 + np.sqrt(5.0) + 2.0)


def test_adjoint_matches_equation_4_on_a_path():
    signal = np.array([0.0, 1.0, 3.0])
    # Edges in order (0, 1), (1, 0), (1, 2), (2, 1).
    sources = np.array([0, 1, 1, 2])
    targets = np.array([1, 0, 2, 1])
    weights = np.ones(4)
    dual = np.array([1.0, 0.5, -2.0, 4.0])
    gradient = apply_gradient(signal, sources, targets, weights)
    adjoint = apply_adjoint(dual, sources, targets, weights, len(signal))
    assert np.sum(gradient * dual) == pytest.approx(np.sum(signal * adjoint))
    assert adjoint == pytest.approx(np.array([-0.5, 6.5, -6.0]))


def test_gradient_is_adjoint_to_its_transpose():
    rng = np.random.default_rng(0)
    signal = rng.normal(size=(12, 3))
    sources = rng.integers(0, 12, size=17)
    targets = (sources + rng.integers(1, 12, size=17)) % 12
    weights = rng.uniform(0.2, 2.0, size=17)
    dual = rng.normal(size=(17, 3))
    gradient = apply_gradient(signal, sources, targets, weights)
    adjoint = apply_adjoint(dual, sources, targets, weights, signal.shape[0])
    assert np.sum(gradient * dual) == pytest.approx(np.sum(signal * adjoint), rel=1e-10, abs=1e-10)


def test_operator_norm_bound_covers_power_iteration():
    rng = np.random.default_rng(1)
    n = 15
    hop = rng.uniform(0.5, 1.5, size=n - 1)
    sources, targets, _ = _path_edges(n)
    weights = np.concatenate([hop, hop])
    bound = operator_norm_bound(sources, targets, weights, n)
    vector = rng.normal(size=n)
    vector /= np.linalg.norm(vector)
    for _ in range(40):
        vector = apply_adjoint(
            apply_gradient(vector, sources, targets, weights),
            sources,
            targets,
            weights,
            n,
        )
        scale = np.linalg.norm(vector)
        vector /= scale
    estimated = np.sqrt(scale)
    assert bound >= estimated - 1e-6


def test_two_point_jump_matches_stationarity():
    signal = np.array([[0.0], [1.0]])
    sources, targets, weights = _path_edges(2)
    restored = chambolle_pock(
        signal, sources, targets, weights, lam=8.0, n_iter=5000, tol=1e-10
    )
    assert restored == pytest.approx(np.array([[0.25], [0.75]]), abs=1e-5)
    assert restored.mean() == pytest.approx(signal.mean(), abs=1e-8)


def test_mean_is_preserved_and_step_mse_drops():
    rng = np.random.default_rng(2)
    clean = np.concatenate([np.zeros(20), np.ones(20)])[:, None]
    noisy = clean + rng.normal(scale=0.2, size=clean.shape)
    sources, targets, weights = _path_edges(len(clean))
    restored = chambolle_pock(
        noisy, sources, targets, weights, lam=8.0, n_iter=8000, tol=1e-8
    )
    assert restored.mean() == pytest.approx(noisy.mean(), abs=1e-8)
    noisy_mse = np.mean((noisy - clean) ** 2)
    restored_mse = np.mean((restored - clean) ** 2)
    assert restored_mse < noisy_mse
    assert _rof(noisy, restored, sources, targets, weights, 8.0) <= _rof(
        noisy, noisy, sources, targets, weights, 8.0
    )


def test_small_lambda_flattens_a_connected_signal():
    rng = np.random.default_rng(3)
    signal = rng.normal(size=(30, 1))
    sources, targets, weights = _path_edges(len(signal))
    restored = chambolle_pock(
        signal, sources, targets, weights, lam=1e-2, n_iter=20000, tol=1e-8
    )
    assert restored.mean() == pytest.approx(signal.mean(), abs=1e-6)
    assert restored.std() < 0.25 * signal.std()


def test_large_lambda_stays_near_the_input():
    rng = np.random.default_rng(4)
    signal = rng.normal(size=(30, 1))
    sources, targets, weights = _path_edges(len(signal))
    restored = chambolle_pock(
        signal, sources, targets, weights, lam=1e4, n_iter=2000, tol=1e-10
    )
    assert restored == pytest.approx(signal, abs=1e-3)
    assert restored.mean() == pytest.approx(signal.mean(), abs=1e-8)


def test_knn_graph_keeps_both_directions():
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [100.0, 0.0, 0.0], [101.0, 0.0, 0.0]]
    )
    sources, targets, weights = knn_graph(points, k=1)
    edges = {tuple(pair) for pair in np.stack([sources, targets], axis=1)}
    assert edges == {(0, 1), (1, 0), (2, 3), (3, 2)}
    assert np.all(weights == 1.0)
    assert np.all(sources != targets)


def _edge_weight(sources, targets, weights, source, target):
    mask = (sources == source) & (targets == target)
    assert int(mask.sum()) == 1
    return float(weights[mask][0])


def test_distance_weight_decays_with_length():
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    sources, targets, weights = knn_graph(points, k=2, weight="distance", bandwidth=1.0)
    short = _edge_weight(sources, targets, weights, 0, 1)
    middle = _edge_weight(sources, targets, weights, 1, 2)
    long = _edge_weight(sources, targets, weights, 0, 2)
    assert short == pytest.approx(np.exp(-1.0))
    assert middle == pytest.approx(np.exp(-16.0))
    assert long == pytest.approx(np.exp(-25.0))
    assert short > middle > long
    assert _edge_weight(sources, targets, weights, 1, 0) == pytest.approx(short)


def test_default_distance_bandwidth_is_the_median_length():
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    sources, targets, weights = knn_graph(points, k=2, weight="distance")
    lengths = np.linalg.norm(points[targets] - points[sources], axis=1)
    bandwidth = float(np.median(lengths))
    short = _edge_weight(sources, targets, weights, 0, 1)
    assert short == pytest.approx(np.exp(-(1.0**2) / bandwidth**2))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"weight": "patch"},
        {"weight": "distance", "bandwidth": 0.0},
        {"weight": "distance", "bandwidth": -1.0},
        {"weight": "equal", "bandwidth": 1.0},
    ],
)
def test_bad_weight_raises(kwargs):
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    with pytest.raises(ValueError):
        knn_graph(points, k=2, **kwargs)


def test_point_cloud_runs_each_coordinate_separately():
    rng = np.random.default_rng(5)
    points = rng.normal(size=(25, 3))
    restored = denoise_point_cloud(points, k=4, lam=2.0, n_iter=2000, tol=1e-8)
    sources, targets, weights = knn_graph(points, k=4)
    separate = [
        chambolle_pock(
            points[:, channel : channel + 1],
            sources,
            targets,
            weights,
            lam=2.0,
            n_iter=2000,
            tol=1e-8,
        )
        for channel in range(3)
    ]
    assert restored == pytest.approx(np.hstack(separate), abs=1e-8)
    before = _rof(points, points, sources, targets, weights, 2.0)
    after = _rof(points, restored, sources, targets, weights, 2.0)
    assert after <= before + 1e-8
    assert restored.mean(axis=0) == pytest.approx(points.mean(axis=0), abs=1e-8)


@pytest.mark.parametrize(
    "points, kwargs",
    [
        (np.zeros(3), {"k": 1, "lam": 1.0}),
        (np.zeros((4, 3)), {"k": 0, "lam": 1.0}),
        (np.zeros((4, 3)), {"k": 4, "lam": 1.0}),
        (np.zeros((4, 3)), {"k": 1, "lam": -0.1}),
        (np.array([[0.0, 0.0, np.nan], [1.0, 0.0, 0.0]]), {"k": 1, "lam": 1.0}),
        (np.zeros((4, 3)), {"k": 1, "lam": 1.0, "n_iter": 0}),
    ],
)
def test_bad_input_raises(points, kwargs):
    with pytest.raises(ValueError):
        denoise_point_cloud(points, **kwargs)
