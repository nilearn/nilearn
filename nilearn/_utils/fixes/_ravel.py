from functools import partial
import numpy as np

np_version = []
for x in np.__version__.split('.'):
    try:
        np_version.append(int(x))
    except ValueError:
        # x may be of the form dev-1ea1592
        np_version.append(x)
np_version = tuple(np_version)

# Newer NumPy has a ravel that needs less copying.
if np_version < (1, 7, 1):
    _ravel = np.ravel
else:
    _ravel = partial(np.ravel, order='K')
