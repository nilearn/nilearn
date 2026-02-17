Each file named ball_cloud_n_samples.csv contains the 3D coordinates of n points
evenly spaced in the unit ball. They have been precomputed and stored to save
time when using the 'ball' sampling in nilearn.surface.vol_to_surf.
They can be re-created like this:

```python
import numpy as np
from nilearn.surface import surface

for n in [10, 20, 40, 80, 160]:
    ball_cloud = surface._uniform_ball_cloud(n_points=n)
    np.savetxt(f"./ball_cloud_{b}_samples.csv", ball_cloud)
```

test_load_uniform_ball_cloud in `nilearn/surface/tests/test_surface.py` compares
these loaded values and freshly computed ones.


These values were computed with version 0.2 of scikit-learn, so positions
computed with scikit-learn < 0.18 would be different (but just as good for our
purposes), because the k-means implementation changed in 0.18.
