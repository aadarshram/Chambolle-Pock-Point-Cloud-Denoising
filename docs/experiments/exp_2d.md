# exp_2d
Remove gaussian noise and salt-and-pepper noise from grayscale images using the ROF model and the TV-L1 model.

## Setup

We consider the louvre image from the paper (Figure 4(a)) and the astronaut image from scikit-image.
The images are stored in `models/experiments/exp_2d/louvre.png` and `models/experiments/exp_2d/astronaut.png` respectively.

Add gaussian noise with standard deviation 0.1 to the image.
Add salt-and-pepper noise with probability 0.25 to the image.

Denoise the noisy image using the ROF model and the TV-L1 model. ROF uses \(\lambda = 3\) and TV-L1 uses \(\lambda = 1\).

We compare the results using the mean square error.

## Usage

```bash
PYTHONPATH=src python -m experiments.exp_2d.run_exp_2d
```

The comparison is saved to `results/experiments/exp_2d/exp_2d_comparison.png` and the errors to `exp_2d_metrics.txt` next to it. `--no-show` saves the figure without opening a window.

To repeat Figure 4 of the paper on the Louvre image, with the paper's \(\lambda\) values:

```bash
PYTHONPATH=src python -m experiments.exp_2d.run_exp_2d \
    --image models/experiments/exp_2d/louvre.png --size 500 \
    --lam-l1 1.5 --lam-l2 8 \
    --output results/experiments/exp_2d/louvre/exp_2d_comparison.png --no-show
```

## Visualization
The notebook `notebooks/chambolle_pock_2D_denoising.ipynb` demonstrates the denoising process and the effects of the parameters on this experiment.

## Layout
The relevant files can be found as follows:
1. `src/cp_image/denoise.py`: The implementation of the ROF and TV-L1 models for image denoising.
2. `src/cp_solver/`: The implementation of the Chambolle-Pock algorithm for image denoising.
3. `experiments/exp_2d/image.py`: The implementation of the synthetic image and the error.
4. `experiments/exp_2d/figure.py`: The implementation of the comparison figure.
5. `experiments/exp_2d/run_exp_2d.py`: The entry point of the experiment.
6. `models/experiments/exp_2d/louvre.png`: The clean test image of Figure 4 of the paper.
7. `models/experiments/exp_2d/astronaut.png`: The astronaut test image from scikit-image.