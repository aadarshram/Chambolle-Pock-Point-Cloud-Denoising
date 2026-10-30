# Chambolle-Pock for 3D Point Cloud Denoising

Extends the Chambolle-Pock method for 2D image denoising to 3D point clouds inspired from the work of Tenbrinck, Lozes, and Elmoataz (SSVM 2015). The implementation is demonstrated on the Spot dataset ([https://www.cs.cmu.edu/~kmcrane/Projects/ModelRepository/](https://www.cs.cmu.edu/~kmcrane/Projects/ModelRepository/)).

Example of point cloud denoising on Spot (Keenan Crane, CMU):
![Spot](assets/spot_color_denoising.png)

The same solver also runs the TV-L1 image denoising model of Chambolle and Pock on a pixel grid. See [docs/experiments/exp_tvl1.md](docs/experiments/exp_tvl1.md).

## Layout

The relevant files can be found as follows:

1. `src/cp_pointcloud/denoise.py`: The implementation of the Chambolle-Pock algorithm for point cloud denoising, with a quadratic (`data="l2"`) or TV-L1 (`data="l1"`) data term.
2. `src/cp_pointcloud/tvl1_denoise.py`: The pixel grid and TV-L1 denoising of grayscale images.
3. `experiments/exp_color/color.py`: The implementation of the color experiment.
4. `experiments/exp_color/run_exp_color.py`: The entry point of the color experiment.
5. `models/experiments/exp_color/spot.zip`: The dataset for the color experiment.
6. `experiments/exp_tvl1/run_exp_tvl1.py`: The entry point of the salt-and-pepper experiment.
7. `models/experiments/exp_tvl1/`: The test images for the salt-and-pepper experiment: the Louvre image from Figure 4 of Chambolle and Pock, and the public-domain astronaut image from scikit-image.
8. `notebooks/chambolle_pock_color_denoising.ipynb`: The notebook demonstrating the denoising process and the effects of the parameters on the color experiment.



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