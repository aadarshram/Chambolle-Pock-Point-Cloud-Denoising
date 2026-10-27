"""
Draw the clean, noisy, and denoised colors on a fixed point cloud.
"""

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d  # noqa: F401  (registers the 3d projection)
import numpy as np

from experiments.exp_color.color import color_mse

# Specific view for Spot. Can be changed to a more general view.
# Spot stands with +y up and the snout toward +z. This eye sits in front of
# the face, off to one side, so the view is a diagonal of the head.
DIAGONAL_ELEVATION = 16
DIAGONAL_AZIMUTH = -160
DIAGONAL_VERTICAL = "y"

def make_comparison_figure(points, clean, noisy, denoised_by_lam):
    """
    Build a 3D comparison painted with the color at each point.

    `denoised_by_lam` is a list of `(lam, colors)` in the order large, moderate, small.
    Every panel uses the same axis box.
    """
    cloud = np.asarray(points, dtype=float)
    panels = [("clean", clean), ("noisy", noisy)]
    panels.extend([(f"λ = {lam:g}", colors) for lam, colors in denoised_by_lam])
    low = cloud.min(axis=0)
    high = cloud.max(axis=0)
    center = (low + high) / 2.0
    half = float(np.max(high - low) / 2.0 * 1.05)
    fig = plt.figure(figsize=(14, 8))
    grid = fig.add_gridspec(2, 3)
    slots = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
    for (row, col), (title, colors) in zip(slots, panels):
        axis = fig.add_subplot(grid[row, col], projection="3d")
        axis.scatter(
            cloud[:, 0],
            cloud[:, 1],
            cloud[:, 2],
            c=np.clip(np.asarray(colors, dtype=float), 0.0, 1.0),
            s=8,
            depthshade=False,
        )
        _style_axis(axis, title, center, half)
    notes = fig.add_subplot(grid[1, 2])
    notes.set_axis_off()
    notes.set_title("color, points fixed")
    note_lines = []
    for title, colors in panels:
        note_lines.append(f"{title}\n  color mse {color_mse(colors, clean):.3e}")
    notes.text(0.0, 1.0, "\n".join(note_lines), va="top", family="monospace", fontsize=9)
    fig.suptitle("exp_color: ROF denoising of the color on Spot")
    return fig


def draw_colored_cloud(axis, points, colors, title):
    """
    Draw one fixed cloud, painted with `colors`, on the Spot camera.
    """
    cloud = np.asarray(points, dtype=float)
    axis.scatter(
        cloud[:, 0],
        cloud[:, 1],
        cloud[:, 2],
        c=np.clip(np.asarray(colors, dtype=float), 0.0, 1.0),
        s=6,
        depthshade=False,
    )
    low = cloud.min(axis=0)
    high = cloud.max(axis=0)
    center = (low + high) / 2.0
    half = float(np.max(high - low) / 2.0 * 1.05)
    _style_axis(axis, title, center, half)
    axis.view_init(
        elev=DIAGONAL_ELEVATION,
        azim=DIAGONAL_AZIMUTH,
        vertical_axis=DIAGONAL_VERTICAL,
    )


def set_diagonal_view(fig):
    """
    Point every 3D panel at a diagonal view of the face, with the back upright.
    """
    for axis in fig.axes:
        if hasattr(axis, "view_init"):
            axis.view_init(
                elev=DIAGONAL_ELEVATION,
                azim=DIAGONAL_AZIMUTH,
                vertical_axis=DIAGONAL_VERTICAL,
            )


def _style_axis(axis, title, center, half):
    axis.set_title(title)
    axis.set_xlim(center[0] - half, center[0] + half)
    axis.set_ylim(center[1] - half, center[1] + half)
    axis.set_zlim(center[2] - half, center[2] + half)
    axis.set_box_aspect((1.0, 1.0, 1.0))
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_zlabel("z")
