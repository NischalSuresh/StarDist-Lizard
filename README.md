# StarDist-Lizard
- Segment histopathological images using modified StarDist (PyTorch)
- The authors' implementation in https://github.com/stardist/stardist for calculating star distances involves ray tracing along n_ray directions for each pixel(dense prediction task) to find the distance to the boundary
- In my implementation instead of doing ray tracing for each pixel I fix n points on the boundary of each nuclei and calculate the distances to these n points which can be easily parallelized
- Modify distances
