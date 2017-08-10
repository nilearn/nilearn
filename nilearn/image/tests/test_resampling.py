"""
Test the resampling code.
"""
import os
import copy
import math

from nose import SkipTest
from nose.tools import assert_equal, assert_raises, \
    assert_false, assert_true, assert_almost_equal, assert_not_equal
from numpy.testing import assert_array_equal, assert_array_almost_equal

import numpy as np

from nibabel import Nifti1Image

from nilearn.image.resampling import resample_img, resample_to_img, reorder_img
from nilearn.image.resampling import from_matrix_vector, coord_transform
from nilearn.image.resampling import BoundingBoxError
from nilearn._utils import testing, compat


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


def pad(array, *args):
    """Pad an ndarray with zeros of quantity specified
    in args as follows args = (x1minpad, x1maxpad, x2minpad,
    x2maxpad, x3minpad, ...)
    """

    if len(args) % 2 != 0:
        raise ValueError("Please specify as many max paddings as min"
                         " paddings. You have specified %d arguments" %
                         len(args))

    all_paddings = np.zeros([array.ndim, 2], dtype=np.int64)
    all_paddings[:len(args) // 2] = np.array(args).reshape(-1, 2)

    lower_paddings, upper_paddings = all_paddings.T
    new_shape = np.array(array.shape) + upper_paddings + lower_paddings

    padded = np.zeros(new_shape, dtype=array.dtype)
    source_slices = [slice(max(-lp, 0), min(s + up, s))
                     for lp, up, s in zip(lower_paddings,
                                          upper_paddings,
                                          array.shape)]
    target_slices = [slice(max(lp, 0), min(s - up, s))
                     for lp, up, s in zip(lower_paddings,
                                          upper_paddings,
                                          new_shape)]

    padded[target_slices] = array[source_slices].copy()
    return padded


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

    # Test with non native endian data

    # Test with big endian data ('>f8')
    for interpolation in ['nearest', 'continuous']:
        rot_img = resample_img(Nifti1Image(data.astype('>f8'), affine),
                               target_affine=affine.tolist(),
                               interpolation=interpolation)
        np.testing.assert_almost_equal(data, rot_img.get_data())

    # Test with little endian data ('<f8')
    for interpolation in ['nearest', 'continuous']:
        rot_img = resample_img(Nifti1Image(data.astype('<f8'), affine),
                               target_affine=affine.tolist(),
                               interpolation=interpolation)
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

    # Test with non native endian data

    # Test to check that if giving non native endian data as input should
    # work as normal and expected to return the same output as above tests.

    # Big endian data ('>f8')
    for copy in [True, False]:
        rot_img = resample_img(Nifti1Image(data.astype('>f8'), affine),
                               target_affine=2 * affine,
                               interpolation='nearest',
                               copy=copy)
        np.testing.assert_almost_equal(downsampled,
                                       rot_img.get_data()[:x, :y, :z, ...])

    # Little endian data
    for copy in [True, False]:
        rot_img = resample_img(Nifti1Image(data.astype('<f8'), affine),
                               target_affine=2 * affine,
                               interpolation='nearest',
                               copy=copy)
        np.testing.assert_almost_equal(downsampled,
                                       rot_img.get_data()[:x, :y, :z, ...])


def test_resampling_with_affine():
    """ Test resampling with a given rotation part of the affine.
    """
    prng = np.random.RandomState(10)

    data_3d = prng.randint(4, size=(1, 4, 4))
    data_4d = prng.randint(4, size=(1, 4, 4, 3))

    for data in [data_3d, data_4d]:
        for angle in (0, np.pi, np.pi / 2., np.pi / 4., np.pi / 3.):
            rot = rotation(0, angle)
            rot_img = resample_img(Nifti1Image(data, np.eye(4)),
                                   target_affine=rot,
                                   interpolation='nearest')
            assert_equal(np.max(data),
                         np.max(rot_img.get_data()))
            assert_equal(rot_img.get_data().dtype, data.dtype)

    # We take the same rotation logic as above and test with nonnative endian
    # data as input
    for data in [data_3d, data_4d]:
        img = Nifti1Image(data.astype('>f8'), np.eye(4))
        for angle in (0, np.pi, np.pi / 2., np.pi / 4., np.pi / 3.):
            rot = rotation(0, angle)
            rot_img = resample_img(img, target_affine=rot,
                                   interpolation='nearest')
            assert_equal(np.max(data),
                         np.max(rot_img.get_data()))


def test_resampling_continuous_with_affine():
    prng = np.random.RandomState(10)

    data_3d = prng.randint(1, 4, size=(1, 10, 10))
    data_4d = prng.randint(1, 4, size=(1, 10, 10, 3))

    for data in [data_3d, data_4d]:
        for angle in (0, np.pi / 2., np.pi, 3 * np.pi / 2.):
            rot = rotation(0, angle)

            img = Nifti1Image(data, np.eye(4))
            rot_img = resample_img(
                img,
                target_affine=rot,
                interpolation='continuous')
            rot_img_back = resample_img(
                rot_img,
                target_affine=np.eye(4),
                interpolation='continuous')

            center = slice(1, 9)
            # values on the edges are wrong for some reason
            mask = (0, center, center)
            np.testing.assert_allclose(
                img.get_data()[mask],
                rot_img_back.get_data()[mask])
            assert_equal(rot_img.get_data().dtype,
                         np.dtype(data.dtype.name.replace('int', 'float')))


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
    interpolation = 'an_invalid_interpolation'
    pattern = "interpolation must be either.+{0}".format(interpolation)
    testing.assert_raises_regex(ValueError, pattern,
                                resample_img, img, target_shape=target_shape,
                                target_affine=affine,
                                interpolation="an_invalid_interpolation")

    # Noop
    target_shape = shape[:3]

    img_r = resample_img(img, copy=False)
    assert_equal(img_r, img)

    img_r = resample_img(img, copy=True)
    assert_false(np.may_share_memory(img_r.get_data(), img.get_data()))

    np.testing.assert_almost_equal(img_r.get_data(), img.get_data())
    np.testing.assert_almost_equal(compat.get_affine(img_r), compat.get_affine(img))

    img_r = resample_img(img, target_affine=affine, target_shape=target_shape,
                         copy=False)
    assert_equal(img_r, img)

    img_r = resample_img(img, target_affine=affine, target_shape=target_shape,
                         copy=True)
    assert_false(np.may_share_memory(img_r.get_data(), img.get_data()))
    np.testing.assert_almost_equal(img_r.get_data(), img.get_data())
    np.testing.assert_almost_equal(compat.get_affine(img_r), compat.get_affine(img))


def test_4d_affine_bounding_box_error():

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
        target_affine=compat.get_affine(bigger_img),
        target_shape=bigger_img.shape)
    # resample using 3D affine and no target shape
    small_to_big_without_shape_3D_affine = resample_img(
        small_img,
        target_affine=compat.get_affine(bigger_img)[:3, :3])
    # resample using 4D affine and no target shape
    small_to_big_without_shape = resample_img(
        small_img,
        target_affine=compat.get_affine(bigger_img))

    # The first 2 should pass
    assert_almost_equal(l2_norm(small_data),
                 l2_norm(small_to_big_with_shape.get_data()))
    assert_almost_equal(l2_norm(small_data),
                 l2_norm(small_to_big_without_shape_3D_affine.get_data()))

    # After correcting decision tree for 4x4 affine given + no target shape
    # from "use initial shape" to "calculate minimal bounding box respecting
    # the affine anchor and the data"
    assert_almost_equal(l2_norm(small_data),
                 l2_norm(small_to_big_without_shape.get_data()))

    assert_array_equal(small_to_big_without_shape.shape,
                 small_data_4D_affine[:3, -1] + np.array(small_img.shape))


def test_raises_upon_3x3_affine_and_no_shape():
    img = Nifti1Image(np.zeros([8, 9, 10]),
                      affine=np.eye(4))
    exception = ValueError
    message = ("Given target shape without anchor "
               "vector: Affine shape should be \(4, 4\) and "
               "not \(3, 3\)")
    testing.assert_raises_regex(
        exception, message,
        resample_img, img, target_affine=np.eye(3) * 2,
        target_shape=(10, 10, 10))


def test_3x3_affine_bbox():
    # Test that the bounding-box is properly computed when
    # transforming with a negative affine component
    # This is specifically to test for a change in behavior between
    # scipy < 0.18 and scipy >= 0.18, which is an interaction between
    # offset and a diagonal affine
    image = np.ones((20, 30))
    source_affine = np.eye(4)
    # Give the affine an offset
    source_affine[:2, 3] = np.array([96, 64])

    # We need to turn this data into a nibabel image
    img = Nifti1Image(image[:, :, np.newaxis], affine=source_affine)

    target_affine_3x3 = np.eye(3) * 2
    # One negative axes
    target_affine_3x3[1] *= -1

    img_3d_affine = resample_img(img, target_affine=target_affine_3x3)

    # If the bounding box is computed wrong, the image will be only
    # zeros
    np.testing.assert_allclose(img_3d_affine.get_data().max(), image.max())


def test_raises_bbox_error_if_data_outside_box():
    # Make some cases which should raise exceptions

    # original image
    data = np.zeros([8, 9, 10])
    affine = np.eye(4)
    affine_offset = np.array([1, 1, 1])
    affine[:3, 3] = affine_offset

    img = Nifti1Image(data, affine)

    # some axis flipping affines
    axis_flips = np.array(list(map(np.diag,
                              [[-1, 1, 1, 1],
                               [1, -1, 1, 1],
                               [1, 1, -1, 1],
                               [-1, -1, 1, 1],
                               [-1, 1, -1, 1],
                               [1, -1, -1, 1]])))

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
        exception = BoundingBoxError
        message = ("The field of view given "
                   "by the target affine does "
                   "not contain any of the data")

        testing.assert_raises_regex(
            exception, message,
            resample_img, img, target_affine=new_affine)


def test_resampling_result_axis_permutation():
    # Transform real data using easily checkable transformations
    # For now: axis permutations
    # create a cuboid full of deterministic data, padded with one
    # voxel thickness of zeros
    core_shape = (3, 5, 4)
    core_data = np.arange(np.prod(core_shape)).reshape(core_shape)
    full_data_shape = np.array(core_shape) + 2
    full_data = np.zeros(full_data_shape)
    full_data[[slice(1, 1 + s) for s in core_shape]] = core_data

    source_img = Nifti1Image(full_data, np.eye(4))

    axis_permutations = [[0, 1, 2],
                         [1, 0, 2],
                         [2, 1, 0],
                         [0, 2, 1]]

    # check 3x3 transformation matrix
    for ap in axis_permutations:
        target_affine = np.eye(3)[ap]
        resampled_img = resample_img(source_img,
                                     target_affine=target_affine)

        resampled_data = resampled_img.get_data()
        what_resampled_data_should_be = full_data.transpose(ap)
        assert_array_almost_equal(resampled_data,
                                  what_resampled_data_should_be)

    # check 4x4 transformation matrix
    offset = np.array([-2, 1, -3])
    for ap in axis_permutations:
        target_affine = np.eye(4)
        target_affine[:3, :3] = np.eye(3)[ap]
        target_affine[:3, 3] = offset

        resampled_img = resample_img(source_img,
                                     target_affine=target_affine)
        resampled_data = resampled_img.get_data()
        offset_cropping = np.vstack([-offset[ap][np.newaxis, :],
                                     np.zeros([1, 3])]
                                    ).T.ravel().astype(int)
        what_resampled_data_should_be = pad(full_data.transpose(ap),
                                            *list(offset_cropping))

        assert_array_almost_equal(resampled_data,
                                  what_resampled_data_should_be)


def test_resampling_nan():
    # Test that when the data has NaNs they do not propagate to the
    # whole image

    for core_shape in [(3, 5, 4), (3, 5, 4, 2)]:
        # create deterministic data, padded with one
        # voxel thickness of zeros
        core_data = np.arange(np.prod(core_shape)
                              ).reshape(core_shape).astype(np.float)
        # Introduce a nan
        core_data[2, 2:4, 1] = np.nan
        full_data_shape = np.array(core_shape) + 2
        full_data = np.zeros(full_data_shape)
        full_data[[slice(1, 1 + s) for s in core_shape]] = core_data

        source_img = Nifti1Image(full_data, np.eye(4))

        # Transform real data using easily checkable transformations
        # For now: axis permutations
        axis_permutation = [0, 1, 2]

        # check 3x3 transformation matrix
        target_affine = np.eye(3)[axis_permutation]
        resampled_img = testing.assert_warns(
            RuntimeWarning, resample_img, source_img,
            target_affine=target_affine)

        resampled_data = resampled_img.get_data()
        if full_data.ndim == 4:
            axis_permutation.append(3)
        what_resampled_data_should_be = full_data.transpose(axis_permutation)
        non_nan = np.isfinite(what_resampled_data_should_be)

        # Check that the input data hasn't been modified:
        assert_false(np.all(non_nan))

        # Check that for finite value resampling works without problems
        assert_array_almost_equal(resampled_data[non_nan],
                                  what_resampled_data_should_be[non_nan])

        # Check that what was not finite is still not finite
        assert_false(np.any(np.isfinite(
                        resampled_data[np.logical_not(non_nan)])))

    # Test with an actual resampling, in the case of a bigish hole
    # This checks the extrapolation mechanism: if we don't do any
    # extrapolation before resampling, the hole creates big
    # artefacts
    data = 10 * np.ones((10, 10, 10))
    data[4:6, 4:6, 4:6] = np.nan
    source_img = Nifti1Image(data, 2 * np.eye(4))
    resampled_img = testing.assert_warns(
        RuntimeWarning, resample_img, source_img,
        target_affine=np.eye(4))

    resampled_data = resampled_img.get_data()
    np.testing.assert_allclose(10, resampled_data[np.isfinite(resampled_data)])


def test_resample_to_img():
    # Testing resample to img function
    rand_gen = np.random.RandomState(0)
    shape = (6, 3, 6, 3)
    data = rand_gen.random_sample(shape)

    source_affine = np.eye(4)
    source_img = Nifti1Image(data, source_affine)

    target_affine = 2 * source_affine
    target_img = Nifti1Image(data, target_affine)


    result_img = resample_to_img(source_img, target_img,
                                 interpolation='nearest')

    downsampled = data[::2, ::2, ::2, ...]
    x, y, z = downsampled.shape[:3]
    np.testing.assert_almost_equal(downsampled,
                                   result_img.get_data()[:x, :y, :z, ...])


def test_reorder_img():
    # We need to test on a square array, as rotation does not change
    # shape, whereas reordering does.
    shape = (5, 5, 5, 2, 2)
    rng = np.random.RandomState(42)
    data = rng.rand(*shape)
    affine = np.eye(4)
    affine[:3, -1] = 0.5 * np.array(shape[:3])
    ref_img = Nifti1Image(data, affine)
    # Test with purely positive matrices and compare to a rotation
    for theta, phi in np.random.randint(4, size=(5, 2)):
        rot = rotation(theta * np.pi / 2, phi * np.pi / 2)
        rot[np.abs(rot) < 0.001] = 0
        rot[rot > 0.9] = 1
        rot[rot < -0.9] = 1
        b = 0.5 * np.array(shape[:3])
        new_affine = from_matrix_vector(rot, b)
        rot_img = resample_img(ref_img, target_affine=new_affine)
        np.testing.assert_array_equal(compat.get_affine(rot_img), new_affine)
        np.testing.assert_array_equal(rot_img.get_data().shape, shape)
        reordered_img = reorder_img(rot_img)
        np.testing.assert_array_equal(compat.get_affine(reordered_img)[:3, :3],
                                      np.eye(3))
        np.testing.assert_almost_equal(reordered_img.get_data(),
                                       data)

    # Create a non-diagonal affine, and check that we raise a sensible
    # exception
    affine[1, 0] = 0.1
    ref_img = Nifti1Image(data, affine)
    testing.assert_raises_regex(ValueError, 'Cannot reorder the axes',
                                reorder_img, ref_img)

    # Test that no exception is raised when resample='continuous'
    reorder_img(ref_img, resample='continuous')

    # Test that resample args gets passed to resample_img
    interpolation = 'nearest'
    reordered_img = reorder_img(ref_img, resample=interpolation)
    resampled_img = resample_img(ref_img,
                                 target_affine=compat.get_affine(reordered_img),
                                 interpolation=interpolation)
    np.testing.assert_array_equal(reordered_img.get_data(),
                                  resampled_img.get_data())

    # Make sure invalid resample argument is included in the error message
    interpolation = 'an_invalid_interpolation'
    pattern = "interpolation must be either.+{0}".format(interpolation)
    testing.assert_raises_regex(ValueError, pattern,
                                reorder_img, ref_img,
                                resample=interpolation)

    # Test flipping an axis
    data = rng.rand(*shape)
    for i in (0, 1, 2):
        # Make a diagonal affine with a negative axis, and check that
        # can be reordered, also vary the shape
        shape = (i + 1, i + 2, 3 - i)
        affine = np.eye(4)
        affine[i, i] *= -1
        img = Nifti1Image(data, affine)
        orig_img = copy.copy(img)
        #x, y, z = img.get_world_coords()
        #sample = img.values_in_world(x, y, z)
        img2 = reorder_img(img)
        # Check that img has not been changed
        np.testing.assert_array_equal(compat.get_affine(img),
                                      compat.get_affine(orig_img))
        np.testing.assert_array_equal(img.get_data(),
                                      orig_img.get_data())
        # Test that the affine is indeed diagonal:
        np.testing.assert_array_equal(compat.get_affine(img2)[:3, :3],
                                      np.diag(np.diag(
                                              compat.get_affine(img2)[:3, :3])))
        assert_true(np.all(np.diag(compat.get_affine(img2)) >= 0))


def test_reorder_img_non_native_endianness():
    def _get_resampled_img(dtype):
        data = np.ones((10, 10, 10), dtype=dtype)
        data[3:7, 3:7, 3:7] = 2

        affine = np.eye(4)

        theta = math.pi / 6.
        c = math.cos(theta)
        s = math.sin(theta)

        affine = np.array([[1, 0, 0, 0],
                           [0, c, -s, 0],
                           [0, s, c, 0],
                           [0, 0, 0, 1]])

        img = Nifti1Image(data, affine)
        return resample_img(img, target_affine=np.eye(4))

    img_1 = _get_resampled_img('<f8')
    img_2 = _get_resampled_img('>f8')

    np.testing.assert_equal(img_1.get_data(), img_2.get_data())


def test_coord_transform_trivial():
    sform = np.eye(4)
    x = np.random.random((10,))
    y = np.random.random((10,))
    z = np.random.random((10,))

    x_, y_, z_ = coord_transform(x, y, z, sform)
    np.testing.assert_array_equal(x, x_)
    np.testing.assert_array_equal(y, y_)
    np.testing.assert_array_equal(z, z_)

    sform[:, -1] = 1
    x_, y_, z_ = coord_transform(x, y, z, sform)
    np.testing.assert_array_equal(x + 1, x_)
    np.testing.assert_array_equal(y + 1, y_)
    np.testing.assert_array_equal(z + 1, z_)

    # Test the output in case of one item array
    x, y, z = x[:1], y[:1], z[:1]
    x_, y_, z_ = coord_transform(x, y, z, sform)
    np.testing.assert_array_equal(x + 1, x_)
    np.testing.assert_array_equal(y + 1, y_)
    np.testing.assert_array_equal(z + 1, z_)

    # Test the output in case of simple items
    x, y, z = x[0], y[0], z[0]
    x_, y_, z_ = coord_transform(x, y, z, sform)
    np.testing.assert_array_equal(x + 1, x_)
    np.testing.assert_array_equal(y + 1, y_)
    np.testing.assert_array_equal(z + 1, z_)


def test_resample_img_segmentation_fault():
    if os.environ.get('APPVEYOR') == 'True':
        raise SkipTest('This test too slow (7-8 minutes) on AppVeyor')

    # see https://github.com/nilearn/nilearn/issues/346
    shape_in = (64, 64, 64)
    aff_in = np.diag([2., 2., 2., 1.])
    aff_out = np.diag([3., 3., 3., 1.])
    # fourth_dim = 1024 works fine but for 1025 creates a segmentation
    # fault with scipy < 0.14.1
    fourth_dim = 1025

    try:
        data = np.ones(shape_in + (fourth_dim, ), dtype=np.float64)
    except MemoryError:
        # This can happen on AppVeyor and for 32-bit Python on Windows
        raise SkipTest('Not enough RAM to run this test')

    img_in = Nifti1Image(data, aff_in)

    resample_img(img_in,
                 target_affine=aff_out,
                 interpolation='nearest')


def test_resampling_with_int_types_no_crash():
    affine = np.eye(4)
    data = np.zeros((2, 2, 2))

    for dtype in [np.int, np.int8, np.int16, np.int32, np.int64,
                  np.uint, np.uint8, np.uint16, np.uint32, np.uint64,
                  np.float32, np.float64, np.float, '>i8', '<i8']:
        img = Nifti1Image(data.astype(dtype), affine)
        resample_img(img, target_affine=2. * affine)
