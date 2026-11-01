#!/usr/bin/env python3
"""
Run exp_tvl1: remove salt-and-pepper noise from an image with TV-L1, and
compare it with ROF.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cp_pointcloud import denoise_image
from cp_pointcloud.denoise import RELATIVE_CHANGE_TOL
from experiments.exp_tvl1.figure import make_comparison_figure
from experiments.exp_tvl1.image import add_salt_and_pepper, image_mse, read_grayscale, synthetic_image

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "results" / "experiments" / "exp_tvl1" / "exp_tvl1_comparison.png"


def main():
    parser = argparse.ArgumentParser(
        description="Remove salt-and-pepper noise with TV-L1 and compare with ROF."
    )
    parser.add_argument(
        "--image", type=Path, default=None, help="Grayscale source image. Default: a synthetic image"
    )
    parser.add_argument("--size", type=int, default=128, help="Longer side of the image in pixels")
    parser.add_argument("--fraction", type=float, default=0.25, help="Fraction of pixels set to 0 or 1")
    parser.add_argument("--lam-l1", type=float, default=1.0, help="Data weight of TV-L1")
    parser.add_argument("--lam-l2", type=float, default=3.0, help="Data weight of ROF")
    parser.add_argument("--n-iter", type=int, default=5000, help="Maximum primal-dual iterations")
    parser.add_argument("--tol", type=float, default=RELATIVE_CHANGE_TOL)
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Path for the comparison PNG")
    parser.add_argument("--no-show", action="store_true", help="Save the figure without opening a window")
    args = parser.parse_args()
    if args.image is None:
        clean = synthetic_image(args.size)
    else:
        clean = read_grayscale(args.image, args.size)
    rng = np.random.default_rng(args.seed)
    noisy = add_salt_and_pepper(clean, args.fraction, rng)
    restored = {}
    for data, lam in (("l1", args.lam_l1), ("l2", args.lam_l2)):
        print(f"denoising data={data} lam={lam:g}", flush=True)
        restored[data] = denoise_image(noisy, lam, data=data, n_iter=args.n_iter, tol=args.tol)
    panels = [
        (f"TV-L1, λ = {args.lam_l1:g}", restored["l1"]),
        (f"ROF, λ = {args.lam_l2:g}", restored["l2"]),
    ]
    figure = make_comparison_figure(clean, noisy, panels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=150, bbox_inches="tight")
    metrics = _metrics_text(clean, noisy, restored, args)
    args.output.with_name("exp_tvl1_metrics.txt").write_text(metrics)
    print(metrics, end="", flush=True)
    print(f"saved {args.output}", flush=True)
    if not args.no_show:
        plt.show(block=True)


def _metrics_text(clean, noisy, restored, args):
    source = "synthetic" if args.image is None else args.image.name
    lines = [
        f"image={source} shape={clean.shape[0]}x{clean.shape[1]} "
        f"fraction={args.fraction} seed={args.seed}",
        f"noisy: mse={image_mse(noisy, clean):.6e}",
        f"tvl1 lam={args.lam_l1:g}: mse={image_mse(restored['l1'], clean):.6e}",
        f"rof lam={args.lam_l2:g}: mse={image_mse(restored['l2'], clean):.6e}",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
