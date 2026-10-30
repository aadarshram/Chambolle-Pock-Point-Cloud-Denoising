"""Checks for the salt-and-pepper experiment helpers."""

import numpy as np
import pytest
from PIL import Image

from experiments.exp_tvl1.image import add_salt_and_pepper, image_mse, read_grayscale, synthetic_image


def test_salt_and_pepper_hits_the_fraction_with_only_0_and_1():
    clean = np.full((100, 100), 0.4)
    noisy = add_salt_and_pepper(clean, fraction=0.25, rng=np.random.default_rng(0))
    hit = noisy != clean
    assert hit.mean() == pytest.approx(0.25, abs=0.02)
    assert set(np.unique(noisy[hit]).tolist()) == {0.0, 1.0}
    assert noisy[hit].mean() == pytest.approx(0.5, abs=0.05)
    assert np.all(clean == 0.4)


def test_salt_and_pepper_is_reproducible_from_the_seed():
    clean = synthetic_image(32)
    first = add_salt_and_pepper(clean, 0.3, np.random.default_rng(7))
    second = add_salt_and_pepper(clean, 0.3, np.random.default_rng(7))
    assert np.array_equal(first, second)


def test_zero_fraction_leaves_the_image_alone():
    clean = synthetic_image(16)
    noisy = add_salt_and_pepper(clean, 0.0, np.random.default_rng(0))
    assert np.array_equal(noisy, clean)


def test_synthetic_image_is_piecewise_constant_in_range():
    image = synthetic_image(64)
    assert image.shape == (64, 64)
    assert set(np.unique(image).tolist()) == {0.2, 0.5, 0.65, 0.8}


def test_image_mse_of_a_constant_offset():
    assert image_mse(np.full((3, 4), 0.5), np.zeros((3, 4))) == pytest.approx(0.25)


def test_read_grayscale_scales_to_unit_range_and_shrinks(tmp_path):
    path = tmp_path / "ramp.png"
    pixels = np.tile(np.linspace(0, 255, 40).astype(np.uint8), (20, 1))
    Image.fromarray(pixels).save(path)
    image = read_grayscale(path, max_side=10)
    assert image.shape == (5, 10)
    assert image.min() >= 0.0 and image.max() <= 1.0


@pytest.mark.parametrize(
    "call",
    [
        lambda: add_salt_and_pepper(np.zeros((4, 4)), fraction=1.5, rng=np.random.default_rng(0)),
        lambda: add_salt_and_pepper(np.zeros((4, 4)), fraction=0.1, rng=0),
        lambda: add_salt_and_pepper(np.zeros(4), fraction=0.1, rng=np.random.default_rng(0)),
        lambda: image_mse(np.zeros((2, 2)), np.zeros((2, 3))),
        lambda: synthetic_image(0),
    ],
)
def test_bad_input_raises(call):
    with pytest.raises(ValueError):
        call()
