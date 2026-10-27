"""Checks for textured-mesh sampling and color denoising."""

from pathlib import Path

import numpy as np
import pytest

from cp_pointcloud import chambolle_pock, knn_graph
from experiments.exp_color.color import add_color_noise, color_mse, denoise_color
from experiments.exp_color.mesh import sample_textured_mesh


def test_samples_lie_in_the_triangle_and_take_its_color():
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    uvs = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    vertex_ids = np.array([[0, 1, 2]])
    uv_ids = np.array([[0, 1, 2]])
    texture = np.zeros((4, 4, 3))
    texture[..., 0] = 1.0
    rng = np.random.default_rng(0)
    points, colors = sample_textured_mesh(
        vertices, uvs, vertex_ids, uv_ids, texture, n=80, rng=rng
    )
    assert points.shape == (80, 3)
    assert colors.shape == (80, 3)
    assert points[:, 2] == pytest.approx(np.zeros(80), abs=1e-8)
    assert np.all(points[:, :2] >= -1e-8)
    assert np.all(points[:, 0] + points[:, 1] <= 1.0 + 1e-8)
    assert colors[:, 0] == pytest.approx(np.ones(80))
    assert colors[:, 1] == pytest.approx(np.zeros(80))


def test_color_noise_stays_in_range_and_denoise_lowers_the_error():
    rng = np.random.default_rng(1)
    points = rng.random((60, 3))
    clean = np.full((60, 3), 0.5)
    noisy = add_color_noise(clean, noise_std=0.08, rng=rng)
    assert noisy.shape == clean.shape
    assert np.all(noisy >= 0.0) and np.all(noisy <= 1.0)
    assert color_mse(noisy, clean) > 0.0
    restored = denoise_color(points, noisy, k=6, lam=30.0, n_iter=80, tol=1e-8)
    assert restored.shape == noisy.shape
    assert color_mse(restored, clean) < color_mse(noisy, clean)
    assert restored.mean(axis=0) == pytest.approx(noisy.mean(axis=0), abs=1e-5)


def _cluster_gap(colors):
    return float(np.linalg.norm(colors[:20].mean(axis=0) - colors[20:].mean(axis=0)))


def test_distance_weight_keeps_separated_colors():
    rng = np.random.default_rng(2)
    left = rng.normal(scale=0.02, size=(20, 3))
    right = np.array([8.0, 0.0, 0.0]) + rng.normal(scale=0.02, size=(20, 3))
    points = np.vstack([left, right])
    clean = np.vstack([np.full((20, 3), 0.2), np.full((20, 3), 0.8)])
    noisy = add_color_noise(clean, noise_std=0.05, rng=rng)
    kwargs = dict(k=30, lam=2.0, n_iter=400, tol=1e-8)
    equal = denoise_color(points, noisy, weight="equal", **kwargs)
    distant = denoise_color(points, noisy, weight="distance", **kwargs)
    sources, targets, weights = knn_graph(points, k=30, weight="distance")
    wired = chambolle_pock(noisy, sources, targets, weights, lam=2.0, n_iter=400, tol=1e-8)
    assert distant == pytest.approx(wired, abs=1e-8)
    assert _cluster_gap(distant) > _cluster_gap(equal) + 0.2
    assert distant.mean(axis=0) == pytest.approx(noisy.mean(axis=0), abs=1e-5)


@pytest.mark.parametrize(
    "call",
    [
        lambda: sample_textured_mesh(
            np.zeros((3, 3)),
            np.zeros((3, 2)),
            np.array([[0, 1, 2]]),
            np.array([[0, 1, 2]]),
            np.zeros((2, 2, 3)),
            n=0,
            rng=np.random.default_rng(0),
        ),
        lambda: add_color_noise(np.zeros((4, 3)), noise_std=-0.1, rng=np.random.default_rng(0)),
        lambda: denoise_color(np.zeros((5, 3)), np.zeros((4, 3)), k=2, lam=1.0, n_iter=2),
        lambda: denoise_color(
            np.zeros((5, 3)), np.zeros((5, 3)), k=2, lam=1.0, n_iter=2, weight="patch"
        ),
        lambda: color_mse(np.zeros((4, 3)), np.zeros((4, 2))),
    ],
)
def test_bad_input_raises(call):
    with pytest.raises(ValueError):
        call()


def test_obj_reader_pairs_positions_with_texture_coordinates(tmp_path):
    from experiments.exp_color.mesh import read_obj_triangles

    path = Path(tmp_path) / "one.obj"
    path.write_text(
        "\n".join(
            [
                "v 0 0 0",
                "v 1 0 0",
                "v 0 1 0",
                "vt 0 0",
                "vt 1 0",
                "vt 0 1",
                "f 1/1 2/2 3/3",
            ]
        )
        + "\n"
    )
    vertices, uvs, vertex_ids, uv_ids = read_obj_triangles(path)
    assert vertices.shape == (3, 3)
    assert uvs.shape == (3, 2)
    assert vertex_ids.tolist() == [[0, 1, 2]]
    assert uv_ids.tolist() == [[0, 1, 2]]
