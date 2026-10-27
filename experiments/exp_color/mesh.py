"""
Load a textured triangle mesh and sample colored points on its surface.
"""

from pathlib import Path

import numpy as np
from PIL import Image

#--------------------------------------------------------------------------------


def read_obj_triangles(path):
    """
    Read positions, texture coordinates, and triangles from a Wavefront OBJ file.

    Faces are fan-triangulated. A face with no texture coordinate raises ValueError.
    Indices in the returned arrays start at 0.
    """
    vertices = []
    uvs = []
    vertex_ids = []
    uv_ids = []
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        kind = parts[0]
        if kind == "v" and len(parts) >= 4:
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif kind == "vt" and len(parts) >= 3:
            uvs.append([float(parts[1]), float(parts[2])])
        elif kind == "f" and len(parts) >= 4:
            corners = [_face_corner(part, len(vertices), len(uvs)) for part in parts[1:]]
            for second in range(1, len(corners) - 1):
                triangle = (corners[0], corners[second], corners[second + 1])
                vertex_ids.append([item[0] for item in triangle])
                uv_ids.append([item[1] for item in triangle])
    if not vertex_ids:
        raise ValueError("the OBJ file has no textured faces")
    return (
        np.asarray(vertices, dtype=float),
        np.asarray(uvs, dtype=float),
        np.asarray(vertex_ids, dtype=int),
        np.asarray(uv_ids, dtype=int),
    )


def read_texture(path):
    """
    Read an RGB image as float colors in [0, 1].
    """
    image = np.asarray(Image.open(path).convert("RGB"), dtype=float)
    return image / 255.0


def sample_textured_mesh(vertices, uvs, vertex_ids, uv_ids, texture, n, rng):
    """
    Draw `n` points uniformly by area, and the texture color at each point.
    """
    count = _as_positive_int(n, "n")
    _as_generator(rng)
    verts = _as_vertices(vertices)
    texcoord = _as_uvs(uvs)
    triangles = _as_triangles(vertex_ids, "vertex_ids")
    uv_triangles = _as_triangles(uv_ids, "uv_ids")
    if uv_triangles.shape != triangles.shape:
        raise ValueError("vertex_ids and uv_ids must have the same shape")
    image = _as_texture(texture)
    corners = verts[triangles]
    edges = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    areas = 0.5 * np.linalg.norm(edges, axis=1)
    total = float(areas.sum())
    if total <= 0.0:
        raise ValueError("the mesh has no area")
    choice = rng.choice(triangles.shape[0], size=count, p=areas / total)
    bary = _barycentric(rng, count)
    points = np.sum(bary[:, :, None] * corners[choice], axis=1)
    uv = np.sum(bary[:, :, None] * texcoord[uv_triangles[choice]], axis=1)
    return points, _sample_texture(image, uv)


#--------------------------------------------------------------------------------


def _face_corner(part, n_vertices, n_uvs):
    fields = part.split("/")
    if len(fields) < 2 or fields[1] == "":
        raise ValueError("a face has no texture coordinate")
    return _obj_index(fields[0], n_vertices), _obj_index(fields[1], n_uvs)


def _obj_index(text, count):
    index = int(text)
    if index > 0:
        return index - 1
    if index < 0:
        return count + index
    raise ValueError("an OBJ index must be nonzero")


def _barycentric(rng, count):
    root = np.sqrt(rng.random(count))
    fraction = rng.random(count)
    return np.column_stack([1.0 - root, root * (1.0 - fraction), root * fraction])


def _sample_texture(image, uv):
    height, width = image.shape[:2]
    column = np.clip(np.rint(uv[:, 0] * (width - 1)), 0, width - 1).astype(int)
    row = np.clip(np.rint((1.0 - uv[:, 1]) * (height - 1)), 0, height - 1).astype(int)
    return image[row, column]


def _as_vertices(values):
    points = np.asarray(values, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 1:
        raise ValueError("vertices must have shape (V, 3)")
    return points


def _as_uvs(values):
    coords = np.asarray(values, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2 or coords.shape[0] < 1:
        raise ValueError("uvs must have shape (T, 2)")
    return coords


def _as_triangles(values, name):
    triangles = np.asarray(values, dtype=int)
    if triangles.ndim != 2 or triangles.shape[1] != 3 or triangles.shape[0] < 1:
        raise ValueError(f"{name} must have shape (F, 3)")
    return triangles


def _as_texture(values):
    image = np.asarray(values, dtype=float)
    if image.ndim != 3 or image.shape[2] != 3 or image.shape[0] < 1 or image.shape[1] < 1:
        raise ValueError("texture must have shape (H, W, 3)")
    return image


def _as_generator(rng):
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a numpy Generator")


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
