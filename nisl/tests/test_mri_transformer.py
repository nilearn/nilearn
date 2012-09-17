from nose.tools import assert_raises
from nose.tools import assert_equal
from nose.tools import assert_true

import numpy as np
from numpy.testing import assert_array_equal

from ..mri_transformer import MRITransformer
from nibabel import Nifti1Image

###############################################################################
# Image Loading                                                               #
###############################################################################


# Creating arrays with several dimensions and depth
def generate_image(dimension, depth, affine=None):
    if (affine == None):
        affine = np.random.random(size=(4, 4))
    if depth == 0:
        shape = tuple(np.random.random_integers(10, size=dimension))
        return Nifti1Image(np.random.random(size=shape), affine)
    items = np.random.random_integers(10)
    array = []
    for i in range(items):
        array.append(generate_image(dimension, depth - 1, affine))
    return array


def generate_sessions(n_sessions, array):
    return np.random.random_integers(n_sessions,
            size=np.ravel(array).shape[0])

# No session array | 3D | Depth 0
# Expected: Error
t = MRITransformer()
img = generate_image(3, 0)
assert_raises(ValueError, t.load_imgs, img)

# No session array | 3D | Depth 1
# Expected: OK, no session
t = MRITransformer()
img = generate_image(3, 1)
t.load_imgs(img)
assert_equal(t.sessions_, None)

# No session array | 3D | Depth 2
# Expected: OK, generates session array
t = MRITransformer()
img = generate_image(3, 2)
assert_raises(ValueError, t.load_imgs, img)

# No session array | 4D | Depth 0
# Expected: OK, no session
t = MRITransformer()
img = generate_image(4, 0)
t.load_imgs(img)
assert_equal(t.sessions_, None)

# No session array | 4D | Depth 1
# Expected: OK, generates session array
t = MRITransformer()
img = generate_image(4, 1)
assert_raises(ValueError, t.load_imgs, img)

# No session array | 4D | Depth 2
# Expected: Error
t = MRITransformer()
img = generate_image(4, 2)
assert_raises(ValueError, t.load_imgs, img)

# Session array    | 3D | Depth 0
# Expected: Error
img = generate_image(3, 0)
sessions = generate_sessions(3, img)
t = MRITransformer(sessions=sessions)
assert_raises(ValueError, t.load_imgs, img)

# Session array    | 3D | Depth 1
# Expected: OK, session array as given
img = generate_image(3, 1)
sessions = generate_sessions(3, img)
t = MRITransformer(sessions=sessions)
t.load_imgs(img)
assert_array_equal(t.sessions_, sessions)

# Session array    | 3D | Depth 2
# Expected: Error
img = generate_image(3, 2)
sessions = generate_sessions(3, img)
t = MRITransformer(sessions=sessions)
assert_raises(ValueError, t.load_imgs, img)

# Session array    | 4D | Depth 0
# Expected: OK, session array as given
img = generate_image(4, 0)
sessions = generate_sessions(3, img)
t = MRITransformer(sessions=sessions)
t.load_imgs(img)
assert_array_equal(t.sessions_, sessions)

# Session array    | 4D | Depth 1
# Expected: Error
img = generate_image(4, 1)
sessions = generate_sessions(3, img)
t = MRITransformer(sessions=sessions)
assert_raises(ValueError, t.load_imgs, img)

# Session array    | 4D | Depth 2
# Expected: Error
img = generate_image(4, 2)
sessions = generate_sessions(3, img)
t = MRITransformer(sessions=sessions)
assert_raises(ValueError, t.load_imgs, img)
