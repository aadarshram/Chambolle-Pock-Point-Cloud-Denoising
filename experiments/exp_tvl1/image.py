"""
Test images, salt-and-pepper noise, and the image error for exp_tvl1.
"""

import numpy as np
from PIL import Image

#--------------------------------------------------------------------------------


def synthetic_image(size):
    """
    Piecewise-constant size x size image in [0, 1].

    A square, a disk, and a thin bar on a flat background.
    """
    size = _as_positive_int(size, "size")
    row, col = np.mgrid[0:size, 0:size] / size
    image = np.full((size, size), 0.2)
    image[(row > 0.12) & (row < 0.52) & (col > 0.1) & (col < 0.5)] = 0.8
    image[(row - 0.6) ** 2 + (col - 0.68) ** 2 < 0.2**2] = 0.5
    image[(row > 0.84) & (row < 0.87) & (col > 0.1) & (col < 0.9)] = 0.65
    return image


def read_grayscale(path, max_side):
    """
    Read an image file as grayscale in [0, 1].

    The image is shrunk, keeping its aspect ratio, so its longer side is at
    most `max_side` pixels.
    """
    max_side = _as_positive_int(max_side, "max_side")
    with Image.open(path) as picture:
        gray = picture.convert("L")
        gray.thumbnail((max_side, max_side))
        return np.asarray(gray, dtype=float) / 255.0


def add_salt_and_pepper(image, fraction, rng):
    """
    Replace each pixel, with probability `fraction`, by 0 or 1 with equal chance.
    """
    clean = _as_image(image, "image")
    fraction = _as_fraction(fraction)
    _as_generator(rng)
    picked = rng.random(clean.shape) < fraction
    salt = rng.random(clean.shape) < 0.5
    noisy = clean.copy()
    noisy[picked] = salt[picked].astype(float)
    return noisy


def image_mse(restored, clean):
    """
    Mean square error between two images of the same shape.
    """
    left = _as_image(restored, "restored")
    right = _as_image(clean, "clean")
    if left.shape != right.shape:
        raise ValueError("restored and clean must have the same shape")
    return float(np.mean((left - right) ** 2))


#--------------------------------------------------------------------------------


def _as_image(array, name):
    values = np.asarray(array, dtype=float)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError(f"{name} must have shape (H, W) with H >= 1 and W >= 1")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    return values


def _as_fraction(value):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("fraction must lie in [0, 1]") from exc
    if not np.isfinite(number) or number < 0.0 or number > 1.0:
        raise ValueError("fraction must lie in [0, 1]")
    return number


def _as_generator(rng):
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a numpy Generator")


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
