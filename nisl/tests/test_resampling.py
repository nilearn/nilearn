"""
Test the resampling code.
"""

import nose
import numpy as np

from nibabel import Nifti1Image

from ..resampling import resample_img


###############################################################################
# Helper function
def rotation(theta, phi):
    """ Returns a rotation 3x3 matrix.
    """
    cos = np.cos
    sin = np.sin
    a1 = np.array([[cos(theta), -sin(theta), 0],
                  [sin(theta),  cos(theta), 0],
                  [0, 0, 1]])
    a2 = np.array([[1, 0, 0],
                  [0, cos(phi), -sin(phi)],
                  [0, sin(phi),  cos(phi)]])
    return np.dot(a1, a2)


###############################################################################
# Tests
def test_identity_resample():
    """ Test resampling of the VolumeImg with an identity affine.
    """
    shape = (3., 2., 5., 2.)
    data = np.random.randint(0, 10, shape)
    affine = np.eye(4)
    affine[:3, -1] = 0.5 * np.array(shape[:3])
    rot_img = resample_img(Nifti1Image(data, affine),
                           target_affine=affine, interpolation='nearest')
    np.testing.assert_almost_equal(data, rot_img.get_data())
    # Test with a 3x3 affine
    rot_img = resample_img(Nifti1Image(data, affine),
                           target_affine=affine[:3, :3],
                           interpolation='nearest')
    np.testing.assert_almost_equal(data, rot_img.get_data())


def test_downsample():
    """ Test resampling of the VolumeImg with a 1/2 down-sampling affine.
    """
    shape = (6., 3., 6, 2.)
    data = np.random.random(shape)
    affine = np.eye(4)
    rot_img = resample_img(Nifti1Image(data, affine),
                           2 * affine, interpolation='nearest')
    downsampled = data[::2, ::2, ::2, ...]
    x, y, z = downsampled.shape[:3]
    np.testing.assert_almost_equal(downsampled,
                                   rot_img.get_data()[:x, :y, :z, ...])


def test_resampling_with_affine():
    """ Test resampling with a given rotation part of the affine.
    """
    prng = np.random.RandomState(10)
    data = prng.randint(4, size=(1, 4, 4))
    for angle in (0, np.pi, np.pi / 2, np.pi / 4, np.pi / 3):
        rot = rotation(0, angle)
        rot_img = resample_img(Nifti1Image(data, np.eye(4)),
                               target_affine=rot,
                               interpolation='nearest')
        np.testing.assert_almost_equal(np.max(data),
                                       np.max(rot_img.get_data()))


def test_missing_parameter():
    """ Test Error when shape provided without affine.
    """
    shape = (3., 2., 5., 2.)
    target_shape = (5., 3., 2., 2.)
    affine = np.eye(4)
    data = np.random.randint(0, 10, shape)
    nose.tools.assert_raises(ValueError, resample_img,
                             Nifti1Image(data, affine),
                             target_shape=target_shape,
                             interpolation='nearest')
