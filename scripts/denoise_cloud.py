#!/usr/bin/env python3
"""
Denoise a point cloud stored as a NumPy array of shape (N, C).
"""

import argparse

import numpy as np

from cp_pointcloud import denoise_point_cloud
from cp_pointcloud.denoise import RELATIVE_CHANGE_TOL

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Denoise a point cloud with the graph ROF model of "
            "Tenbrinck, Lozes, and Elmoataz (2015)."
        )
    )
    parser.add_argument("--input", required=True, help="Path to a .npy array of shape (N, C)")
    parser.add_argument("--output", required=True, help="Path for the denoised .npy array")
    parser.add_argument("--k", type=int, required=True, help="Number of neighbors per point")
    parser.add_argument(
        "--lam",
        type=float,
        required=True,
        help="Data-fidelity weight. Larger values stay closer to the input.",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=5000,
        help="Maximum number of primal-dual iterations",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=RELATIVE_CHANGE_TOL,
        help="Stop when the relative change in the signal falls below this value",
    )
    args = parser.parse_args()
    points = np.load(args.input)
    restored = denoise_point_cloud(
        points,
        k=args.k,
        lam=args.lam,
        n_iter=args.n_iter,
        tol=args.tol,
    )
    np.save(args.output, restored)


if __name__ == "__main__":
    main()
