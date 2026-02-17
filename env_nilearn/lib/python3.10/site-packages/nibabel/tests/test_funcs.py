# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Test for image funcs"""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from ..analyze import AnalyzeImage
from ..funcs import OrientationError, as_closest_canonical, concat_images
from ..loadsave import save
from ..nifti1 import Nifti1Image
from ..tmpdirs import InTemporaryDirectory

_counter = 0


def _as_fname(img):
    global _counter
    fname = f'img{_counter:3d}.nii'
    _counter = _counter + 1
    save(img, fname)
    return fname


def test_concat():
    # Smoke test: concat empty list.
    with pytest.raises(ValueError):
        concat_images([])

    # Build combinations of 3D, 4D w/size[3] == 1, and 4D w/size[3] == 3
    all_shapes_5D = ((1, 4, 5, 3, 3), (7, 3, 1, 4, 5), (0, 2, 1, 4, 5))

    affine = np.eye(4)
    for dim in range(2, 6):
        all_shapes_ND = tuple(shape[:dim] for shape in all_shapes_5D)
        all_shapes_N1D_unary = tuple(shape + (1,) for shape in all_shapes_ND)
        all_shapes = all_shapes_ND + all_shapes_N1D_unary

        # Loop over all possible combinations of images, in first and
        #   second position.
        for data0_shape in all_shapes:
            data0_numel = np.asarray(data0_shape).prod()
            data0 = np.arange(data0_numel, dtype='int32').reshape(data0_shape)
            img0_mem = Nifti1Image(data0, affine)

            for data1_shape in all_shapes:
                data1_numel = np.asarray(data1_shape).prod()
                data1 = np.arange(data1_numel, dtype='int32').reshape(data1_shape)
                img1_mem = Nifti1Image(data1, affine)
                img2_mem = Nifti1Image(data1, affine + 1)  # bad affine

                # Loop over every possible axis, including None (explicit and implied)
                for axis in list(range(-(dim - 2), (dim - 1))) + [None, '__default__']:
                    # Allow testing default vs. passing explicit param
                    if axis == '__default__':
                        np_concat_kwargs = dict(axis=-1)
                        concat_imgs_kwargs = dict()
                        axis = None  # Convert downstream
                    elif axis is None:
                        np_concat_kwargs = dict(axis=-1)
                        concat_imgs_kwargs = dict(axis=axis)
                    else:
                        np_concat_kwargs = dict(axis=axis)
                        concat_imgs_kwargs = dict(axis=axis)

                    # Create expected output
                    try:
                        # Error will be thrown if the np.concatenate fails.
                        #   However, when axis=None, the concatenate is possible
                        #   but our efficient logic (where all images are
                        #   3D and the same size) fails, so we also
                        #   have to expect errors for those.
                        if axis is None:  # 3D from here and below
                            all_data = np.concatenate(
                                [data0[..., np.newaxis], data1[..., np.newaxis]],
                                **np_concat_kwargs,
                            )
                        else:  # both 3D, appending on final axis
                            all_data = np.concatenate([data0, data1], **np_concat_kwargs)
                        expect_error = False
                    except ValueError:
                        # Shapes are not combinable
                        expect_error = True

                    # Check filenames and in-memory images work
                    with InTemporaryDirectory():
                        # Try mem-based, file-based, and mixed
                        imgs = [img0_mem, img1_mem, img2_mem]
                        img_files = [_as_fname(img) for img in imgs]
                        imgs_mixed = [imgs[0], img_files[1], imgs[2]]
                        for img0, img1, img2 in (imgs, img_files, imgs_mixed):
                            try:
                                all_imgs = concat_images([img0, img1], **concat_imgs_kwargs)
                            except ValueError as ve:
                                assert expect_error, str(ve)
                            else:
                                assert (
                                    not expect_error
                                ), 'Expected a concatenation error, but got none.'
                                assert_array_equal(all_imgs.get_fdata(), all_data)
                                assert_array_equal(all_imgs.affine, affine)

                            # check that not-matching affines raise error
                            with pytest.raises(ValueError):
                                concat_images([img0, img2], **concat_imgs_kwargs)

                            # except if check_affines is False
                            try:
                                all_imgs = concat_images([img0, img1], **concat_imgs_kwargs)
                            except ValueError as ve:
                                assert expect_error, str(ve)
                            else:
                                assert (
                                    not expect_error
                                ), 'Expected a concatenation error, but got none.'
                                assert_array_equal(all_imgs.get_fdata(), all_data)
                                assert_array_equal(all_imgs.affine, affine)


def test_closest_canonical():
    # Use 32-bit data so that the AnalyzeImage class doesn't complain
    arr = np.arange(24, dtype=np.int32).reshape((2, 3, 4, 1))

    # Test with an AnalyzeImage first
    img = AnalyzeImage(arr, np.eye(4))
    xyz_img = as_closest_canonical(img)
    assert img is xyz_img

    # And a case where the Analyze image has to be flipped
    img = AnalyzeImage(arr, np.diag([-1, 1, 1, 1]))
    xyz_img = as_closest_canonical(img)
    assert img is not xyz_img
    out_arr = xyz_img.get_fdata()
    assert_array_equal(out_arr, np.flipud(arr))

    # Now onto the NIFTI cases (where dim_info also has to be updated)

    # No funky stuff, returns same thing
    img = Nifti1Image(arr, np.eye(4))
    # set freq/phase/slice dim so that we can check that we
    # re-order them properly
    img.header.set_dim_info(0, 1, 2)
    xyz_img = as_closest_canonical(img)
    assert img is xyz_img

    # a axis flip
    img = Nifti1Image(arr, np.diag([-1, 1, 1, 1]))
    img.header.set_dim_info(0, 1, 2)
    xyz_img = as_closest_canonical(img)
    assert img is not xyz_img
    assert img.header.get_dim_info() == xyz_img.header.get_dim_info()
    out_arr = xyz_img.get_fdata()
    assert_array_equal(out_arr, np.flipud(arr))

    # no error for enforce_diag in this case
    xyz_img = as_closest_canonical(img, True)
    # but there is if the affine is not diagonal
    aff = np.eye(4)
    aff[0, 1] = 0.1
    # although it's more or less canonical already
    img = Nifti1Image(arr, aff)
    xyz_img = as_closest_canonical(img)
    assert img is xyz_img
    # it's still not diagnonal
    with pytest.raises(OrientationError):
        as_closest_canonical(img, True)

    # an axis swap
    aff = np.diag([1, 0, 0, 1])
    aff[1, 2] = 1
    aff[2, 1] = 1
    img = Nifti1Image(arr, aff)
    img.header.set_dim_info(0, 1, 2)

    xyz_img = as_closest_canonical(img)
    assert img is not xyz_img
    # Check both the original and new objects
    assert img.header.get_dim_info() == (0, 1, 2)
    assert xyz_img.header.get_dim_info() == (0, 2, 1)
    out_arr = xyz_img.get_fdata()
    assert_array_equal(out_arr, np.transpose(arr, (0, 2, 1, 3)))

    # same axis swap but with None dim info (except for slice dim)
    img.header.set_dim_info(None, None, 2)
    xyz_img = as_closest_canonical(img)
    assert xyz_img.header.get_dim_info() == (None, None, 1)
