# Test images for exp_tvl1

The run of exp_tvl1 adds the salt-and-pepper noise itself, so both images are clean.

## Louvre

The clean test image of Figure 4(a) in Chambolle and Pock, A first-order primal-dual algorithm for convex problems with applications to imaging, 2011.

https://hal.science/hal-00490826

`louvre.png` is the 500 x 375 grayscale image embedded in the paper. It was extracted from the PDF unchanged with `pdfimages`.

## Astronaut

Photograph of the astronaut Eileen Collins from the NASA Great Images database, as shipped by scikit-image (`skimage.data.astronaut`).

https://flic.kr/p/r9qvLn

No known copyright restrictions, released into the public domain. `astronaut.png` is the 512 x 512 color original. The experiment reads it as grayscale.
