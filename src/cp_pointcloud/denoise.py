"""
Extension of the Chambolle-Pock method for 2D image denoising to color denoising in 3D point clouds.

We first define the point cloud as a symmetric k-nearest-neighbor graph. This establishes the ordering structure to construct the gradient operator.
K = grad is the weighted edge difference. sqrt(w) * (signal[target] - signal[source]) where the signal[] are the color values at the vertices.

The gradient norm first finds L2 norm over all edges. The isotropic total variation is then the sum over all vertices of the Euclidean norm of the gradients.

The weight for the edges can be set to be equal or a function of the distance between the vertices. The latter helps in smoothing the point cloud by giving more weight to closer neighbors but for smaller patches it doesn't make much difference.

Each color channel is denoised independently.

Note: Essentially, in the Chambolle-Pock method, only the operator K is changed. Remaining follows exactly as that of 2D image denoising.
"""

import numpy as np
from scipy.spatial import cKDTree # for k-nearest-neighbor graph

from cp_solver import RELATIVE_CHANGE_TOL, chambolle_pock
from cp_solver.prox import prox_l1_data, prox_quadratic_data

#--------------------------------------------------------------------------------

def denoise_point_cloud(
    points, k, lam, n_iter=5000, theta=1.0, tol=RELATIVE_CHANGE_TOL, weight="equal", bandwidth=None
):
    """
    Denoise each coordinate on the symmetric k-nearest-neighbor graph of `points`.

    Solves the denoising problem for each color channel independently. `lam` multiplies the data term.
    `weight` is "equal" or "distance", as in `knn_graph`.
    """
    cloud = _as_matrix(points, "points")

    sources, targets, weights = knn_graph(cloud, k, weight=weight, bandwidth=bandwidth)
    return denoise_graph_signal(
        cloud, sources, targets, weights, lam, n_iter=n_iter, theta=theta, tol=tol
    )

def knn_graph(points, k, weight="equal", bandwidth=None):
    """
    Directed edges of a symmetric k-nearest-neighbor graph.

    Both (i, j) and (j, i) are returned when either point is a neighbor of the
    other. There are no self-loops. `weight="equal"` sets every edge to 1.
    `weight="distance"` sets w_ij = exp(-||x_i - x_j||^2 / h^2). `h` is
    `bandwidth`, or the median positive edge length when `bandwidth` is omitted.
    """

    def edge_weights(sources, targets):
        if not isinstance(weight, str) or weight.strip().lower() not in ("equal", "distance"):
            raise ValueError("weight must be 'equal' or 'distance'") 
        kind = weight.strip().lower()
        if kind == "equal":
            if bandwidth is not None:
                raise ValueError("bandwidth is used only with weight='distance'")
            return np.ones(sources.shape[0])
        if isinstance(bandwidth, (bool, np.bool_)):
            raise ValueError("bandwidth must be a finite number > 0")
        lengths = np.linalg.norm(cloud[targets] - cloud[sources], axis=1)
        if bandwidth is None:
            positive = lengths[lengths > 0.0]
            if positive.size == 0:
                raise ValueError("bandwidth is undefined when every neighbor distance is 0")
            scale = float(np.median(positive))
        else:
            try:
                scale = float(bandwidth)
            except (TypeError, ValueError) as exc:
                raise ValueError("bandwidth must be a finite number > 0") from exc
            if not np.isfinite(scale) or scale <= 0.0:
                raise ValueError("bandwidth must be a finite number > 0")
        return np.exp(-(lengths**2) / (scale**2))

    cloud = _as_matrix(points, "points")
    k = _as_positive_int(k, "k")
    n = cloud.shape[0]
    if k >= n:
        raise ValueError("k must be smaller than the number of points")

    # Drop the point itself from the neighbor query.
    _distances, neighbors = cKDTree(cloud).query(cloud, k=k + 1)
    is_self = neighbors == np.arange(n)[:, None]
    if not np.all(is_self.sum(axis=1) == 1):
        raise ValueError("nearest-neighbor query did not return each point itself")
    neighbors = neighbors[~is_self].reshape(n, k)
    rows = np.repeat(np.arange(n), k)
    cols = neighbors.reshape(-1)
    # Add the reverse of every neighbor so the neighborhood is symmetric.
    sources = np.concatenate([rows, cols])
    targets = np.concatenate([cols, rows])
    order = np.lexsort((targets, sources))
    sources, targets = sources[order], targets[order]
    # Keep one copy of each directed edge.
    keep = np.ones(sources.shape[0], dtype=bool)
    keep[1:] = (sources[1:] != sources[:-1]) | (targets[1:] != targets[:-1])
    sources, targets = sources[keep], targets[keep]
    weights = edge_weights(sources, targets)
    return sources.astype(int, copy=False), targets.astype(int, copy=False), weights

#--------------------------------------------------------------------------------

def apply_gradient(signal, sources, targets, weights):
    """
    Weighted differences sqrt(w) * (signal[target] - signal[source]).
    """
    values = np.asarray(signal, dtype=float)
    sqrt_w = np.sqrt(weights)
    if values.ndim == 1:
        return sqrt_w * (values[targets] - values[sources])
    return sqrt_w[:, None] * (values[targets] - values[sources])

def apply_adjoint(dual, sources, targets, weights, n_vertices):
    """
    Adjoint of `apply_gradient`.
    Sketch:
    For graph, adjoint of the edge gradient is the edge sum of the dual. The dual variable is the projection onto [-1, 1].
    """
    values = np.asarray(dual, dtype=float)
    sqrt_w = np.sqrt(weights)
    if values.ndim == 1:
        adjoint = np.zeros(n_vertices)
        scaled = sqrt_w * values
    else:
        adjoint = np.zeros((n_vertices, values.shape[1]))
        scaled = sqrt_w[:, None] * values
    np.add.at(adjoint, targets, scaled)
    np.add.at(adjoint, sources, -scaled)
    return adjoint


def total_variation(signal, sources, targets, weights):
    """
    Isotropic vertex total variation.

    Channels are summed separately. The norm at each vertex stacks the
    weighted jumps along outgoing edges.
    """
    values = np.asarray(signal, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    total = 0.0
    sqrt_w = np.sqrt(weights)
    for channel in range(values.shape[1]):
        jump = sqrt_w * (values[targets, channel] - values[sources, channel])
        squared = np.zeros(values.shape[0])
        np.add.at(squared, sources, jump**2)
        total += float(np.sum(np.sqrt(squared)))
    return total

def operator_norm_bound(sources, targets, weights, n_vertices):
    """
    Sketch:
    Using the formula for the operator norm and the fact that this is a symmetric undirected graph, a sufficient uppr bound can be found.

    Upper bound 2 * sqrt(max outgoing weight) on ||K||.
    ||K||^2 <= 4 * max_i sum_{j ~ i} w_ij.
    """
    outgoing = np.zeros(n_vertices)
    np.add.at(outgoing, np.asarray(sources, dtype=int), weights)
    maximum = float(outgoing.max()) if n_vertices else 0.0
    if maximum == 0.0:
        return 0.0
    return float(np.sqrt(4.0 * maximum))

#--------------------------------------------------------------------------------

def denoise_graph_signal(
    signal,
    sources,
    targets,
    weights,
    lam,
    n_iter=5000,
    theta=1.0,
    tol=RELATIVE_CHANGE_TOL,
    data="l2",
):
    """
    Minimize the denoising problem on the given graph, one scalar channel at a time.

    `data` is "l2" for the quadratic data term, or "l1" for the TV-L1 data term.
    """
    observed, squeeze = _as_signal(signal)
    n_vertices = observed.shape[0]
    sources, targets, weights = _as_edges(sources, targets, weights, n_vertices)
    lam = _as_nonnegative(lam, "lam")
    prox_data = _data_prox(data)
    bound = operator_norm_bound(sources, targets, weights, n_vertices)
    restored = observed.copy()

    if bound > 0.0:
        for channel in range(observed.shape[1]):
            restored[:, channel] = chambolle_pock(
                observed[:, channel],
                K=lambda x: apply_gradient(x, sources, targets, weights),
                K_adjoint=lambda y: apply_adjoint(y, sources, targets, weights, n_vertices),
                prox_f_conj=lambda y, _sigma: y * _unit_ball_scale(y, sources, n_vertices),
                prox_g=lambda x, tau, f=observed[:, channel]: prox_data(x, f, tau * lam),
                norm_K=bound,
                n_iter=n_iter,
                theta=theta,
                tol=tol,
            )
    if squeeze:
        return restored[:, 0]
    return restored

def _unit_ball_scale(dual, sources, n_vertices):
    squared = np.zeros(n_vertices)
    np.add.at(squared, sources, dual**2)
    norms = np.sqrt(squared)
    scale = np.ones(n_vertices)
    outside = norms > 1.0
    scale[outside] = 1.0 / norms[outside]
    return scale[sources]

def _data_prox(data):
    if not isinstance(data, str) or data.strip().lower() not in ("l2", "l1"):
        raise ValueError("data must be 'l2' or 'l1'")
    return prox_quadratic_data if data.strip().lower() == "l2" else prox_l1_data

#--------------------------------------------------------------------------------

def _as_matrix(array, name):
    values = np.asarray(array, dtype=float)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError(f"{name} must have shape (N, C) with N >= 1 and C >= 1")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    return values

def _as_signal(array):
    values = np.asarray(array, dtype=float)
    squeeze = values.ndim == 1
    if squeeze:
        values = values[:, None]
    elif values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError("signal must have shape (N,) or (N, C)")
    if not np.all(np.isfinite(values)):
        raise ValueError("signal must contain only finite values")
    return values, squeeze

def _as_edges(sources, targets, weights, n_vertices):
    sources = np.asarray(sources)
    targets = np.asarray(targets)
    weights = np.asarray(weights, dtype=float)
    same_shape = sources.ndim == 1 and targets.shape == sources.shape and weights.shape == sources.shape
    if not same_shape:
        raise ValueError("sources, targets, and weights must be 1-D arrays of equal length")
    if not np.array_equal(sources, np.asarray(sources, dtype=int)) or not np.array_equal(
        targets, np.asarray(targets, dtype=int)
    ):
        raise ValueError("sources and targets must be integer indices")
    sources = np.asarray(sources, dtype=int)
    targets = np.asarray(targets, dtype=int)
    if sources.shape[0] == 0:
        return sources, targets, weights
    inside = sources.min() >= 0 and targets.min() >= 0
    inside = inside and sources.max() < n_vertices and targets.max() < n_vertices
    if not inside:
        raise ValueError("an edge index is outside the signal")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("weights must be finite and non-negative")
    return sources, targets, weights

def _as_positive_int(value, name):
    if isinstance(value, bool) or isinstance(value, np.bool_):
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
