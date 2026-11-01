# Chambolle-Pock for 3D Point Cloud Denoising

Extends the Chambolle-Pock method for 2D image denoising to 3D point clouds inspired from the work of Tenbrinck, Lozes, and Elmoataz (SSVM 2015). The implementation is demonstrated on the Spot dataset ([https://www.cs.cmu.edu/~kmcrane/Projects/ModelRepository/](https://www.cs.cmu.edu/~kmcrane/Projects/ModelRepository/)).

Example of point cloud denoising (gaussian noise) on Spot (Keenan Crane, CMU):
![Spot](assets/spot_color_denoising.png)

The same solver also runs the TV-L1 image denoising model of Chambolle and Pock on a pixel grid. See [docs/experiments/exp_tvl1.md](docs/experiments/exp_tvl1.md).

Example of reproduced 2D denoising result (salt-and-pepper noise) from Chambolle and Pock (2011) on the Louvre image (Figure 4(a)):
![Louvre](assets/2d_louvre_denoising.png)

## Layout

The relevant files can be found as follows:

1. `src/cp_image/denoise.py`: The implementation of the Chambolle-Pock algorithm for 2D image denoising, with a quadratic (`data="l2"`) or TV-L1 (`data="l1"`) data term.
2. `src/cp_pointcloud/denoise.py`: The implementation of the Chambolle-Pock algorithm for point cloud denoising, with a quadratic (`data="l2"`) or TV-L1 (`data="l1"`) data term.
3. `src/cp_solver/algorithm.py`: The implementation of the Chambolle-Pock algorithm.    
4. `src/cp_solver/prox.py`: The implementation of the proximal maps for the quadratic and TV-L1 data terms.
5. `experiments/exp_2d/denoise.py`: The implementation of the 2D denoising experiment.
6. `experiments/exp_2d/run_exp_2d.py`: The entry point of the 2D denoising experiment.
7. `models/experiments/exp_2d/louvre.png`: The Louvre image from Figure 4(a) of Chambolle and Pock (2011).
8. `models/experiments/exp_2d/astronaut.png`: The astronaut image from scikit-image.
9. `notebooks/chambolle_pock_2D_denoising.ipynb`: The notebook demonstrating the denoising process and the effects of the parameters on the 2D denoising experiment.

## Usage

```bash
git clone https://github.com/aadarsh-ram/chambolle-pock-point-cloud-denoising.git
cd chambolle-pock-point-cloud-denoising
pip install -r requirements.txt
pip install -e .
pytest
```

## References

Chambolle and Pock, [A first-order primal-dual algorithm for convex problems with applications to imaging](https://hal.science/hal-00490826), 2011. Algorithm 1, and the TV-L1 image denoising model.

Tenbrinck, Lozes, and Elmoataz, [Solving minimal surface problems on surfaces and point clouds](https://elmoatazbill.users.greyc.fr/point_cloud/2015_tenbrinck_ssvm.pdf), SSVM 2015. Equations (7), (14), and (20). We only consider the isotropic graph for denoising.