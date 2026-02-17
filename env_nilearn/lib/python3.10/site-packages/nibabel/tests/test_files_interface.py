# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Testing filesets - a draft"""

from io import BytesIO

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from .. import MGHImage, Nifti1Image, Nifti1Pair, all_image_classes
from ..fileholders import FileHolderError
from ..spatialimages import SpatialImage


def test_files_spatialimages():
    # test files creation in image classes
    arr = np.zeros((2, 3, 4))
    aff = np.eye(4)
    klasses = [
        klass for klass in all_image_classes if klass.rw and issubclass(klass, SpatialImage)
    ]
    for klass in klasses:
        file_map = klass.make_file_map()
        for value in file_map.values():
            assert value.filename is None
            assert value.fileobj is None
            assert value.pos == 0
        # If we can't create new images in memory without loading, bail here
        if not klass.makeable:
            continue
        # MGHImage accepts only a few datatypes
        # so we force a type change to float32
        if klass == MGHImage:
            img = klass(arr.astype(np.float32), aff)
        else:
            img = klass(arr, aff)
        for value in img.file_map.values():
            assert value.filename is None
            assert value.fileobj is None
            assert value.pos == 0


def test_files_interface():
    # test high-level interface to files mapping
    arr = np.zeros((2, 3, 4))
    aff = np.eye(4)
    img = Nifti1Image(arr, aff)
    # single image
    img.set_filename('test')
    assert img.get_filename() == 'test.nii'
    assert img.file_map['image'].filename == 'test.nii'
    with pytest.raises(KeyError):
        img.file_map['header']
    # pair - note new class
    img = Nifti1Pair(arr, aff)
    img.set_filename('test')
    assert img.get_filename() == 'test.img'
    assert img.file_map['image'].filename == 'test.img'
    assert img.file_map['header'].filename == 'test.hdr'
    # fileobjs - single image
    img = Nifti1Image(arr, aff)
    img.file_map['image'].fileobj = BytesIO()
    img.to_file_map()  # saves to files
    img2 = Nifti1Image.from_file_map(img.file_map)
    # img still has correct data
    assert_array_equal(img2.get_fdata(), img.get_fdata())
    # fileobjs - pair
    img = Nifti1Pair(arr, aff)
    img.file_map['image'].fileobj = BytesIO()
    # no header yet
    with pytest.raises(FileHolderError):
        img.to_file_map()
    img.file_map['header'].fileobj = BytesIO()
    img.to_file_map()  # saves to files
    img2 = Nifti1Pair.from_file_map(img.file_map)
    # img still has correct data
    assert_array_equal(img2.get_fdata(), img.get_fdata())


def test_round_trip_spatialimages():
    # write an image to files
    data = np.arange(24, dtype='i4').reshape((2, 3, 4))
    aff = np.eye(4)
    klasses = [
        klass
        for klass in all_image_classes
        if klass.rw and klass.makeable and issubclass(klass, SpatialImage)
    ]
    for klass in klasses:
        file_map = klass.make_file_map()
        for key in file_map:
            file_map[key].fileobj = BytesIO()
        img = klass(data, aff)
        img.file_map = file_map
        img.to_file_map()
        # read it back again from the written files
        img2 = klass.from_file_map(file_map)
        assert_array_equal(img2.get_fdata(), data)
        # write, read it again
        img2.to_file_map()
        img3 = klass.from_file_map(file_map)
        assert_array_equal(img3.get_fdata(), data)
