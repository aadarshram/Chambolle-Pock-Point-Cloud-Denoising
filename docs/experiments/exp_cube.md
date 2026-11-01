# exp_cube

Denoise the coordinates of a noisy cube with the graph ROF model, and check whether the faces become flat again.

The unknown is the point position. The symmetric 8-neighbor graph is built from the noisy points and then held fixed. Each coordinate is a separate scalar problem. \(\lambda\) multiplies the data term: a larger value stays closer to the noisy cloud.

A cube is the shape this energy prefers. A face is constant in one coordinate, and an edge is a jump. Total variation is small on a flat face and it keeps a sharp jump. The sphere is curved in every direction, so the same energy starts to pull the radius in before the noise is gone.

## Setup

Sample 900 points uniformly on the surface of the cube of edge length 1 centered at the origin. Each of the six faces is chosen with equal probability.

Add independent Gaussian noise of standard deviation 0.05 to each coordinate. That is 5% of the edge length, and it is below the typical gap between neighbors, so a point’s neighbors are still nearby points on the same face.

Denoise that same noisy cloud three times.

- \(\lambda = 400\). The result stays close to the noisy cloud.
- \(\lambda = 80\). The value that flattens the faces.
- \(\lambda = 0.3\). The cloud shrinks toward the origin.

The small-\(\lambda\) run is a check on the energy. Total variation of the coordinates is zero for a constant signal, and the iteration keeps the mean of each coordinate. A centered cube therefore collapses toward the origin when \(\lambda\) is very small. A collapsed cloud means the solver is following the ROF problem.

## What to look at

The window has the clean cube, the noisy cloud, and the three denoised clouds. The camera starts on a diagonal and can be rotated. Color is the absolute offset from the clean face, measured on the coordinate that was exactly \(\pm 0.5\) before the noise. Every panel uses the same color scale and the same axis box, so a smaller cube stays visibly smaller.

Two numbers are printed for each cloud, and written beside the figure.

- Mean square error to the clean samples, using the original point index.
- The mean absolute face offset, its standard deviation, and the mean absolute value of the face coordinate. On the clean cube that position is 0.5.

## What counts as the expected outcome

At \(\lambda = 80\), the error to the clean samples drops, the face offset drops, and the face position stays near 0.5.

At \(\lambda = 400\), the face offset moves only a little and the face position stays at 0.5.

At \(\lambda = 0.3\), the face position falls to the origin.

A slightly smaller cube with flatter faces is a successful smooth paid for by shrinkage. The error to the original samples rises again when the faces move off the clean cube. The face position is there so that case is not read as a flatter cube.

## Run

```bash
PYTHONPATH=src python -m experiments.exp_cube.run_exp_cube
```

The diagonal view is saved to `results/experiments/exp_cube/exp_cube_diagonal.png`. The window stays open until it is closed. `--no-show` saves the figure without opening a window.

This folder is not a git repository, so the figure name has no commit hash.

## This run

Seed 0, the defaults above. The diagonal view is the saved figure.

| Cloud | MSE to clean samples | Mean face offset | Face position |
| --- | ---: | ---: | ---: |
| clean | 0 | 0 | 0.500 |
| noisy | 2.46e-3 | 0.041 | 0.501 |
| λ = 400 | 2.24e-3 | 0.035 | 0.499 |
| λ = 80 | 1.96e-3 | 0.021 | 0.491 |
| λ = 0.3 | 1.38e-1 | 0.500 | 0.007 |

λ = 0.3 collapses to the origin, which is the ROF check. λ = 80 halves the face offset and lowers the error to the clean samples, while the faces stay near 0.5. λ = 400 trims the offset only slightly and leaves the cube on the noisy scan.
