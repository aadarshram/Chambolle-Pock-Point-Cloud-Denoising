# exp_sphere

Denoise the coordinates of a noisy sphere with the graph ROF model, and check whether the cloud becomes a sphere again.

The unknown is the point position. The symmetric 8-neighbor graph is built from the noisy points and then held fixed. Each coordinate is a separate scalar problem. \(\lambda\) multiplies the data term: a larger value stays closer to the noisy cloud.

## Setup

Sample 800 points uniformly on the sphere of radius 1 centered at the origin.

Add independent Gaussian noise of standard deviation 0.03 to each coordinate. That scale is below the typical gap between neighbors, so a point’s neighbors are still nearby points on the sphere.

Denoise that same noisy cloud three times.

- \(\lambda = 50\). The result stays close to the noisy cloud.
- \(\lambda = 5\). The starting value for a rounder sphere.
- \(\lambda = 0.2\). The cloud shrinks toward the origin.

The small-\(\lambda\) run is a check on the energy. Total variation of the coordinates is zero for a constant signal, and the iteration keeps the mean of each coordinate. A centered sphere therefore collapses toward the origin when \(\lambda\) is very small. A collapsed cloud means the solver is following the ROF problem.

## What to look at

The window has the clean sphere, the noisy cloud, and the three denoised clouds. The camera starts on a diagonal and can be rotated. Color is the absolute gap between a point’s distance to the origin and radius 1. Every panel uses the same color scale and the same axis box, so a smaller sphere stays visibly smaller.

Two numbers are printed for each cloud, and written beside the figure.

- Mean square error to the clean samples, using the original point index.
- The mean distance to the origin, and the standard deviation of those distances.

## What counts as the expected outcome

At \(\lambda = 5\), the error to the clean samples drops, the spread of the radii drops, and the mean radius stays near 1.

At \(\lambda = 50\), the error barely moves.

At \(\lambda = 0.2\), the mean radius falls.

A slightly smaller sphere with a tight radius spread is a successful smooth paid for by shrinkage. The error to the original samples can rise again when shrinkage exceeds the noise that was removed. The mean radius is there so that case is not read as a failed sphere.

## Run

```bash
PYTHONPATH=src python -m experiments.exp_sphere.run_exp_sphere
```

The diagonal view is saved to `results/experiments/exp_sphere/exp_sphere_diagonal.png`. The window stays open until it is closed. `--no-show` saves the figure without opening a window.

This folder is not a git repository, so the figure name has no commit hash.

## This run

Seed 0, the defaults above. The diagonal view is the saved figure.

| Cloud | MSE to clean samples | Mean radius | Radius spread |
| --- | ---: | ---: | ---: |
| clean | 0 | 1.000 | 0 |
| noisy | 8.85e-4 | 1.002 | 0.030 |
| λ = 50 | 1.21e-3 | 0.987 | 0.017 |
| λ = 5 | 1.03e-2 | 0.873 | 0.052 |
| λ = 0.2 | 3.32e-1 | 0.062 | 0 |

λ = 0.2 collapses to the origin, which is the ROF check. λ = 50 halves the radius spread and stays near radius 1, while the error to the clean samples rises a little because of the small inward shift. λ = 5 does not recover the ball: the mean radius falls to 0.87 and the radius spread grows. Shrinkage has already overtaken the noise removal at that value.
