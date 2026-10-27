#!/usr/bin/env python3
"""
Run exp_color and open an interactive view of Spot with denoised color.
"""

import argparse
from pathlib import Path
import zipfile

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

from cp_pointcloud.denoise import RELATIVE_CHANGE_TOL
from experiments.exp_color.color import add_color_noise, color_mse, denoise_color
from experiments.exp_color.figure import make_comparison_figure, set_diagonal_view
from experiments.exp_color.mesh import read_obj_triangles, read_texture, sample_textured_mesh

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "experiments" / "exp_color"
DEFAULT_OUTPUT = ROOT / "results" / "experiments" / "exp_color" / "exp_color_diagonal.png"


def main():
    parser = argparse.ArgumentParser(
        description="Denoise the color of a textured point cloud and show the clouds."
    )
    parser.add_argument("--n", type=int, default=8000, help="Number of points sampled on the mesh")
    parser.add_argument("--noise-std", type=float, default=0.35, help="Gaussian noise on each color channel")
    parser.add_argument("--k", type=int, default=8, help="Neighbors per point, from the positions")
    parser.add_argument(
        "--lam",
        type=float,
        nargs=3,
        default=(80.0, 12.0, 1.0),
        metavar=("LARGE", "MODERATE", "SMALL"),
        help="Data weights. Larger values stay closer to the noisy color.",
    )
    parser.add_argument("--n-iter", type=int, default=5000, help="Maximum primal-dual iterations")
    parser.add_argument("--tol", type=float, default=RELATIVE_CHANGE_TOL)
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Path for the diagonal-view PNG")
    parser.add_argument("--no-show", action="store_true", help="Save the figure without opening a window")
    args = parser.parse_args()
    obj_path, texture_path = _spot_paths()
    vertices, uvs, vertex_ids, uv_ids = read_obj_triangles(obj_path)
    texture = read_texture(texture_path)
    rng = np.random.default_rng(args.seed)
    points, clean = sample_textured_mesh(
        vertices, uvs, vertex_ids, uv_ids, texture, args.n, rng
    )
    noisy = add_color_noise(clean, args.noise_std, rng)
    denoised_by_lam = []
    for lam in args.lam:
        print(f"denoising lam={lam:g}", flush=True)
        restored = denoise_color(
            points, noisy, k=args.k, lam=lam, n_iter=args.n_iter, tol=args.tol
        )
        denoised_by_lam.append((lam, restored))
    figure = make_comparison_figure(points, clean, noisy, denoised_by_lam)
    set_diagonal_view(figure)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=150, bbox_inches="tight")
    metrics = _metrics_text(clean, noisy, denoised_by_lam, args)
    args.output.with_name("exp_color_metrics.txt").write_text(metrics)
    np.savez(
        args.output.with_name("exp_color_clouds.npz"),
        points=points,
        clean=clean,
        noisy=noisy,
        **{f"lam_{lam:g}": colors for lam, colors in denoised_by_lam},
    )
    print(metrics, end="", flush=True)
    print(f"saved {args.output}", flush=True)
    if not args.no_show:
        print(f"opening window backend={matplotlib.get_backend()}", flush=True)
        plt.show(block=True)
        print("show returned", flush=True)


def _spot_paths():
    obj_path = MODEL_DIR / "spot" / "spot_triangulated.obj"
    texture_path = MODEL_DIR / "spot" / "spot_texture.png"
    if obj_path.is_file() and texture_path.is_file():
        return obj_path, texture_path
    archive = MODEL_DIR / "spot.zip"
    if not archive.is_file():
        raise FileNotFoundError(
            "Spot is missing. Download https://www.cs.cmu.edu/~kmcrane/Projects/ModelRepository/spot.zip "
            f"into {archive}"
        )
    with zipfile.ZipFile(archive) as bundle:
        bundle.extract("spot/spot_triangulated.obj", MODEL_DIR)
        bundle.extract("spot/spot_texture.png", MODEL_DIR)
    return obj_path, texture_path


def _metrics_text(clean, noisy, denoised_by_lam, args):
    lines = [
        f"n={args.n} noise_std={args.noise_std} k={args.k} seed={args.seed}",
        _cloud_line("clean", clean, clean),
        _cloud_line("noisy", noisy, clean),
    ]
    for lam, colors in denoised_by_lam:
        lines.append(_cloud_line(f"lam={lam:g}", colors, clean))
    return "\n".join(lines) + "\n"


def _cloud_line(name, colors, clean):
    return f"{name}: color_mse={color_mse(colors, clean):.6e}"


if __name__ == "__main__":
    main()
