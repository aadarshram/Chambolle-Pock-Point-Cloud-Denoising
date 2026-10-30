"""
Draw the clean, noisy, TV-L1, and ROF images side by side.
"""

import matplotlib.pyplot as plt

from experiments.exp_tvl1.image import image_mse


def make_comparison_figure(clean, noisy, panels):
    """
    Build a row of grayscale panels with the error against `clean` in each title.

    `panels` is a list of `(title, image)` shown after the clean and noisy images.
    """
    shown = [("clean", clean), ("noisy", noisy)] + list(panels)
    fig, axes = plt.subplots(1, len(shown), figsize=(3.2 * len(shown), 3.6))
    for axis, (title, image) in zip(axes, shown):
        axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
        axis.set_title(f"{title}\nmse = {image_mse(image, clean):.2e}")
        axis.set_axis_off()
    fig.tight_layout()
    return fig
