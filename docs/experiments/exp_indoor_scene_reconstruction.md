# exp_indoor_scene_reconstruction

Denoise one posed depth frame from a synthetic indoor room, and score the points against the room's ground-truth surface.

The unknown is the point position. The symmetric 8-neighbor graph is built from the noisy points and then held fixed. Each coordinate is a separate scalar problem, solved by the same graph ROF model as `exp_sphere`. \(\lambda\) multiplies the data term: a larger value stays closer to the noisy cloud.

The sequence is not fused. This is one frame, not a reconstruction pipeline.

## The rooms

The geometry is the living room and the office from Handa, Whelan, McDonald, and Davison, "A benchmark for RGB-D visual odometry, 3D reconstruction and SLAM," ICRA 2014. Both rooms are synthetic. The surfaces and the camera poses are known exactly. Color and depth were rendered with POV-Ray.

Choi, Zhou, and Koltun, "Robust reconstruction of indoor scenes," CVPR 2015, built four handheld trajectories on those rooms, two per room. The paths average 36 m and see about 88% of the surface. Living room 1 has 2870 frames and office 1 has 2690 frames, each 640×480. They also published a dense ground-truth point cloud for each room.

Living room 1 and office 1 are the runs used here. Each one has a clean depth sequence, a noisy depth sequence, the ground-truth trajectory, and the room's point cloud.

Living room 2 and office 2 walk through the same two rooms on a second path. They have no separate surface, so they are not downloaded.

## How the depth noise was made

The noisy sequence is already in the release. Their `simdepth.py` is the record of how it was built. It is not copied or run.

A clean pixel is stored in millimeters. The value divided by 1000 is meters. The simulator then, in order:

- shifts the pixel by a fraction of a pixel
- takes the depth from the even pixel of a 2×2 block
- applies a low-frequency distortion measured on a real depth camera
- quantizes disparity and adds a small high-frequency term

The pixel shuffle means a noisy pixel is not the same ray as the clean pixel at that coordinate. Error is the distance to the nearest ground-truth point, not a difference of matched indices.

`interpolate.py` and the POV-Ray scene files make new trajectories. They are not used.

## Files on disk

The scene files live in `models/experiments/exp_indoor_scene_reconstruction/`. Eight items are downloaded from `http://redwood-data.org/indoor/data/`:

- `livingroom.ply.zip`, `livingroom1-depth-clean.zip`, `livingroom1-depth-simulated.zip`, `livingroom1-traj.txt`, `livingroom1-color.zip`
- `office.ply.zip`, `office1-depth-clean.zip`, `office1-depth-simulated.zip`, `office1-traj.txt`, `office1-color.zip`

Color, the ONI streams, the fragment reconstructions, the distortion table, and trajectories 2 are left on the server.

```text
models/experiments/exp_indoor_scene_reconstruction/
  living_room/ground_truth.ply
  living_room/living_room_1/depth_clean/
  living_room/living_room_1/depth_noisy/
  living_room/living_room_1/color/
  living_room/living_room_1/trajectory.txt
  office/ground_truth.ply
  office/office_1/depth_clean/
  office/office_1/depth_noisy/
  office/office_1/color/
  office/office_1/trajectory.txt
```

## From a depth frame to a cloud

Handa publishes \(f_x = 481.20\), \(f_y = -480.00\), \(c_x = 319.50\), \(c_y = 239.50\), and the negative focal length is required for their POV-Ray projection. On these depth PNGs the clean frame meets the ground-truth cloud with \(f_y = +480\): the median distance is about 2 cm, against about 9 cm with the negative sign. The run uses the positive sign. A trajectory record is three integers and a 4×4 camera-to-world matrix.

Non-positive depth is missing and is dropped. The remaining points are transformed by that frame's pose. A voxel grid then keeps about 4000 points, and never more than 4000. The graph uses \(k = 8\). The stop is the same as the sphere: relative change \(10^{-5}\), at most 5000 iterations.

The frame is chosen by a fixed rule, then kept. Every 50th noisy frame is a candidate. The first candidate whose valid depth is between 40% and 90%, and whose valid depths have standard deviation at least 0.25 m, is the frame. The spread cut drops a view that is only a flat wall. Living room 1 selects frame 0, and that index is the default.

Before any denoised cloud is kept, the posed noisy points are compared with the ground-truth cloud. A median distance of centimeters means the pose and the camera matrix agree with the room. A median of meters means the focal-length sign is wrong, and that run is discarded.

## Choosing \(\lambda\)

Let \(d\) be the median distance from the subsampled noisy points to the ground-truth cloud, and let \(B\) be the operator-norm bound used for the step size. The three data weights satisfy

\[
B / \lambda \in \{0.25d,\, d,\, 20d\}.
\]

Those are the large, moderate, and small settings: stay near the scan, match the measured noise, and the collapse check. They are computed from the noisy cloud, before the denoised clouds are inspected.

Coordinate total variation is zero only for a constant signal, and the iteration keeps the mean of each coordinate. A small \(\lambda\) therefore pulls the patch toward its centroid. On the sphere this was a pull toward the origin. Here the patch flattens and contracts. A smaller distance to the surface together with a much smaller patch is shrinkage, not a recovered room.

## What to look at

The default window shows the noisy frame and the three denoised clouds, in the room coordinates. The camera starts on a diagonal and can be rotated. Color is the distance to the nearest ground-truth point. Every panel uses the same color scale and the same axis box, so a contracted patch stays visibly smaller.

`--room-view` opens a separate window for the ground-truth room, the noisy frame, and each denoised cloud. The point-cloud file stores a color channel, but every point in it is white, so the color comes from the rendered frames instead. A frame keeps the color of the pixel it was sampled from. The room behind it is colored by projecting each displayed point into a nearby color frame. The floor is upright, and the camera looks from that frame's pose toward the scan. The median distance and the patch diagonal stay in the window title.

The room picture is the wrong scale for the noise. On the largest flat face the points sit a few millimetres off the surface, and the large data weight moves each point by less than a millimetre. Drawn in the color of the photograph, on a room several metres across, the noisy cloud and the denoised clouds are the same picture.

`--detail` draws only that face, in one window with a panel for the noisy points and each denoised cloud. Position along the wall is true scale. The offset from the plane fit to the noisy face is drawn 20 times larger, and the color is that same offset in millimetres. The black rectangle is the plane. Every panel uses that plane and the noisy face's box, so a point that leaves the wall, or a face that shrinks, stays visible. The panel title gives the median absolute offset and how far the points moved.

Two numbers are printed for each cloud.

- The median distance to the ground-truth points.
- The diagonal of the axis-aligned bounding box, which is the patch size.

## What counts as the expected outcome

At the large \(\lambda\), the median distance stays near the noisy distance and the patch diagonal barely moves.

At the moderate \(\lambda\), the median distance drops and the patch diagonal stays near the noisy diagonal.

At the small \(\lambda\), the patch diagonal falls. That is the ROF check.

A slightly smaller patch with a lower median distance is a successful smooth paid for by shrinkage. The median distance can rise again when shrinkage moves the points off the surface. The patch diagonal is there so that case is not read as a failed room.

## Run

```bash
PYTHONPATH=src python -m experiments.exp_indoor_scene_reconstruction.run_exp_indoor_scene_reconstruction
```

Office 1 is the same command with `--scene office_1`. `--no-show` saves the figure without opening a window. `--room-view` saves and opens the colored room windows instead of the diagonal comparison.

`--detail` saves and opens the magnified face instead of the diagonal comparison. It can be combined with `--room-view`.

The figures are saved to `results/experiments/exp_indoor_scene_reconstruction/`. A window stays open until it is closed.

This folder is not a git repository, so the figure name has no commit hash.

## This run

Living room 1, frame 0, 4000 points, \(k = 8\). The operator-norm bound is 8.49. The noisy median distance is \(d = 0.0206\) m, so the three data weights are 1651, 413, and 20.6.

The clean rendering of this same frame, posed with the same camera, has a median distance of 0.021 m. The stored ground-truth cloud is itself about 2 cm away from the clean depth, so the noisy median has little room to fall.

| Cloud | Median distance | Patch diagonal |
| --- | ---: | ---: |
| noisy | 0.0206 m | 3.543 m |
| λ = 1651 | 0.0205 m | 3.542 m |
| λ = 413 | 0.0205 m | 3.519 m |
| λ = 20.6 | 0.0176 m | 3.343 m |

The large and moderate weights stay on the scan. The patch diagonal barely moves, and the median stays at the 2 cm floor.

The small weight shortens the diagonal by 0.20 m. The patch contracts, and it does not collapse to its centroid: the nominal shift \(B/\lambda\) is 0.41 m on a patch 3.5 m across. The median falls slightly while that contraction happens.

The diagonal view is `results/experiments/exp_indoor_scene_reconstruction/living_room_1_diagonal.png`.

On the largest flat face, the noisy points sit a median of 4.5 mm off the fitted plane. At λ = 1651 each point moves by 0.4 mm and the offset stays 4.5 mm. At λ = 413 the points move by 3.9 mm and the median offset falls to 3.3 mm. At λ = 20.6 the points move by 21 mm and the median offset falls to 1.5 mm, while the face pulls away from the plane. That close-up is `results/experiments/exp_indoor_scene_reconstruction/living_room_1_detail.png`. The offsets in the figure are drawn 20 times larger than the room.
