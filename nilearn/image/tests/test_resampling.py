"""
Test the resampling code.
"""

from nose.tools import assert_equal, assert_raises, assert_false, \
    assert_almost_equal
from numpy.testing import assert_array_equal
import numpy as np

from nibabel import Nifti1Image

from ..resampling import resample_img, BoundingBoxError
from ..._utils import testing

###############################################################################
# Helper function
def rotation(theta, phi):
    """ Returns a rotation 3x3 matrix.
    """
    cos = np.cos
    sin = np.sin
    a1 = np.array([[cos(theta), -sin(theta), 0],
                   [sin(theta), cos(theta), 0],
                  [0, 0, 1]])
    a2 = np.array([[1, 0, 0],
                  [0, cos(phi), -sin(phi)],
                  [0, sin(phi), cos(phi)]])
    return np.dot(a1, a2)


###############################################################################
# Tests
def test_identity_resample():
    """ Test resampling with an identity affine.
    """
    shape = (3, 2, 5, 2)
    data = np.random.randint(0, 10, shape)
    affine = np.eye(4)
    affine[:3, -1] = 0.5 * np.array(shape[:3])
    rot_img = resample_img(Nifti1Image(data, affine),
                           target_affine=affine, interpolation='nearest')
    np.testing.assert_almost_equal(data, rot_img.get_data())
    # Smoke-test with a list affine
    rot_img = resample_img(Nifti1Image(data, affine),
                           target_affine=affine.tolist(),
                           interpolation='nearest')
    # Test with a 3x3 affine
    rot_img = resample_img(Nifti1Image(data, affine),
                           target_affine=affine[:3, :3],
                           interpolation='nearest')
    np.testing.assert_almost_equal(data, rot_img.get_data())


def test_downsample():
    """ Test resampling with a 1/2 down-sampling affine.
    """
    rand_gen = np.random.RandomState(0)
    shape = (6, 3, 6, 2)
    data = rand_gen.random_sample(shape)
    affine = np.eye(4)
    rot_img = resample_img(Nifti1Image(data, affine),
                           target_affine=2 * affine, interpolation='nearest')
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


def test_resampling_error_checks():
    shape = (3, 2, 5, 2)
    target_shape = (5, 3, 2)
    affine = np.eye(4)
    data = np.random.randint(0, 10, shape)
    img = Nifti1Image(data, affine)

    # Correct parameters: no exception
    resample_img(img, target_shape=target_shape, target_affine=affine)
    resample_img(img, target_affine=affine)

    with testing.write_tmp_imgs(img) as filename:
        resample_img(filename, target_shape=target_shape, target_affine=affine)

    # Missing parameter
    assert_raises(ValueError, resample_img, img, target_shape=target_shape)

    # Invalid shape
    assert_raises(ValueError, resample_img, img, target_shape=(2, 3),
                  target_affine=affine)

    # Invalid interpolation
    assert_raises(ValueError, resample_img, img, target_shape=target_shape,
                  target_affine=affine, interpolation="invalid")

    # Noop
    target_shape = shape[:3]

    img_r = resample_img(img, copy=False)
    assert_equal(img_r, img)

    img_r = resample_img(img, copy=True)
    assert_false(np.may_share_memory(img_r.get_data(), img.get_data()))

    np.testing.assert_almost_equal(img_r.get_data(), img.get_data())
    np.testing.assert_almost_equal(img_r.get_affine(), img.get_affine())

    img_r = resample_img(img, target_affine=affine, target_shape=target_shape,
                         copy=False)
    assert_equal(img_r, img)

    img_r = resample_img(img, target_affine=affine, target_shape=target_shape,
                         copy=True)
    assert_false(np.may_share_memory(img_r.get_data(), img.get_data()))
    np.testing.assert_almost_equal(img_r.get_data(), img.get_data())
    np.testing.assert_almost_equal(img_r.get_affine(), img.get_affine())


def test_4D_affine_bounding_box_error():

    small_data = np.ones([4, 4, 4])
    small_data_4D_affine = np.eye(4)
    small_data_4D_affine[:3, -1] = np.array([5, 4, 5])

    small_img = Nifti1Image(small_data,
                            small_data_4D_affine)

    bigger_data_4D_affine = np.eye(4)
    bigger_data = np.zeros([10, 10, 10])
    bigger_img = Nifti1Image(bigger_data,
                             bigger_data_4D_affine)

    # We would like to check whether all/most of the data
    # will be contained in the resampled image
    # The measure will be the l2 norm, since some resampling
    # schemes approximately conserve it

    def l2_norm(arr):
        return (arr ** 2).sum()

    # resample using 4D affine and specified target shape
    small_to_big_with_shape = resample_img(
        small_img,
        target_affine=bigger_img.get_affine(),
        target_shape=bigger_img.shape)
    # resample using 3D affine and no target shape
    small_to_big_without_shape_3D_affine = resample_img(
        small_img,
        target_affine=bigger_img.get_affine()[:3, :3])
    # resample using 4D affine and no target shape
    small_to_big_without_shape = resample_img(
        small_img,
        target_affine=bigger_img.get_affine())

    # The first 2 should pass
    assert_almost_equal(l2_norm(small_data),
                 l2_norm(small_to_big_with_shape.get_data()))
    assert_almost_equal(l2_norm(small_data),
                 l2_norm(small_to_big_without_shape_3D_affine.get_data()))

    # This now passes as well
    assert_almost_equal(l2_norm(small_data),
                 l2_norm(small_to_big_without_shape.get_data()))

    assert_array_equal(small_to_big_without_shape.shape,
                 small_data_4D_affine[:3, -1] + np.array(small_img.shape))


def test_raises_upon_3x3_affine_and_no_shape():
    img = Nifti1Image(np.zeros([8, 9, 10]),
                      affine=np.eye(4))
    # with assert_raises(ValueError("Given target shape without anchor "
    #                               "vector: Affine should be (4, 4) and "
    #                               "not (3, 3)")):
    with assert_raises(ValueError):
        new_img = resample_img(img, target_affine=np.eye(3) * 2,
                               target_shape=(10, 10, 10))


def test_raises_bbox_error_if_data_outside_box():

    # Make some cases which should raise exceptions

    # original image
    data = np.zeros([8, 9, 10])
    affine = np.eye(4)
    affine_offset = np.array([1, 1, 1])
    affine[:3, 3] = affine_offset

    img = Nifti1Image(data, affine)

    # some axis flipping affines
    axis_flips = np.array(map(np.diag,
                              [[-1, 1, 1, 1],
                               [1, -1, 1, 1],
                               [1, 1, -1, 1],
                               [-1, -1, 1, 1],
                               [-1, 1, -1, 1],
                               [1, -1, -1, 1]]))

    # some in plane 90 degree rotations base on these
    # (by permuting two lines)
    af = axis_flips
    rotations = np.array([af[0][[1, 0, 2, 3]],
                          af[0][[2, 1, 0, 3]],
                          af[1][[1, 0, 2, 3]],
                          af[1][[0, 2, 1, 3]],
                          af[2][[2, 1, 0, 3]],
                          af[2][[0, 2, 1, 3]]])

    new_affines = np.concatenate([axis_flips,
                                  rotations])
    new_offset = np.array([0., 0., 0.])
    new_affines[:, :3, 3] = new_offset[np.newaxis, :]

    for new_affine in new_affines:
        # with assert_raises(
        #     BoundingBoxError("The field of view given "
        #                      "by the target affine does "
        #                      "not contain any of the data")):
        with assert_raises(BoundingBoxError):
            new_img = resample_img(img, target_affine=new_affine)
