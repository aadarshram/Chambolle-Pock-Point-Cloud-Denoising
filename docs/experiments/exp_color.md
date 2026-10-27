# exp_color

Denoise the color on a point cloud with the graph ROF model.

## Setup

We consider the point cloud model of Spot, the public-domain cow released by Keenan Crane. The mesh and the texture come from `https://www.cs.cmu.edu/~kmcrane/Projects/ModelRepository/spot.zip`.

Sample 8000 points uniformly by area on the triangulated surface, and read each point's color from the texture.

Add independent Gaussian noise of standard deviation 0.35 to each color channel, then clip to \([0, 1]\).
Denoise that same noisy color three times.

## Usage

```bash
PYTHONPATH=src python -m experiments.exp_color.run_exp_color
```

The diagonal view is saved to `results/experiments/exp_color/exp_color_diagonal.png`. The window stays open until it is closed. `--no-show` saves the figure without opening a window.

## Visualization
The notebook `notebooks/chambolle_pock_color_denoising.ipynb` demonstrates the denoising process and the effects of the parameters on this experiment.

## Layout
The relevant files can be found as follows:
1. `src/cp_pointcloud/denoise.py`: The implementation of the Chambolle-Pock algorithm for point cloud denoising.
2. `src/experiments/exp_color/color.py`: The implementation of the experiment.
3. `src/experiments/exp_color/run_exp_color.py`: The entry point of the experiment.
4. `src/experiments/exp_color/spot.zip`: The dataset for the experiment.
5. `notebooks/chambolle_pock_color_denoising.ipynb`: The notebook demonstrating the denoising process and the effects of the parameters on this experiment.