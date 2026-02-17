import getpass
import hashlib
import os
import struct
import time
import unittest
from os.path import isdir
from os.path import join as pjoin
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from ...fileslice import strided_scalar
from ...testing import clear_and_catch_warnings
from ...tests.nibabel_data import get_nibabel_data, needs_nibabel_data
from ...tmpdirs import InTemporaryDirectory
from .. import (
    read_annot,
    read_geometry,
    read_label,
    read_morph_data,
    write_annot,
    write_geometry,
    write_morph_data,
)
from ..io import _pack_rgb

DATA_SDIR = 'fsaverage'

have_freesurfer = False
if 'SUBJECTS_DIR' in os.environ:
    # May have Freesurfer installed with data
    data_path = pjoin(os.environ['SUBJECTS_DIR'], DATA_SDIR)
    have_freesurfer = isdir(data_path)
else:
    # May have nibabel test data submodule checked out
    nib_data = get_nibabel_data()
    if nib_data != '':
        data_path = pjoin(nib_data, 'nitest-freesurfer', DATA_SDIR)
        have_freesurfer = isdir(data_path)

freesurfer_test = unittest.skipUnless(
    have_freesurfer, f'cannot find freesurfer {DATA_SDIR} directory'
)


@freesurfer_test
def test_geometry():
    """Test IO of .surf"""
    surf_path = pjoin(data_path, 'surf', 'lh.inflated')
    coords, faces = read_geometry(surf_path)
    assert 0 == faces.min()
    assert coords.shape[0] == faces.max() + 1

    surf_path = pjoin(data_path, 'surf', 'lh.sphere')
    coords, faces, volume_info, create_stamp = read_geometry(
        surf_path, read_metadata=True, read_stamp=True
    )

    assert 0 == faces.min()
    assert coords.shape[0] == faces.max() + 1
    assert 9 == len(volume_info)
    assert np.array_equal([2, 0, 20], volume_info['head'])
    assert create_stamp == 'created by greve on Thu Jun  8 19:17:51 2006'

    # Test equivalence of freesurfer- and nibabel-generated triangular files
    # with respect to read_geometry()
    with InTemporaryDirectory():
        surf_path = 'test'
        create_stamp = f'created by {getpass.getuser()} on {time.ctime()}'
        volume_info['cras'] = [1.0, 2.0, 3.0]
        write_geometry(surf_path, coords, faces, create_stamp, volume_info)

        coords2, faces2, volume_info2 = read_geometry(surf_path, read_metadata=True)

        for key in ('xras', 'yras', 'zras', 'cras'):
            assert_allclose(volume_info2[key], volume_info[key], rtol=1e-7, atol=1e-30)

        assert np.array_equal(volume_info2['cras'], volume_info['cras'])
        with open(surf_path, 'rb') as fobj:
            np.fromfile(fobj, '>u1', 3)
            read_create_stamp = fobj.readline().decode().rstrip('\n')

        # now write an incomplete file
        write_geometry(surf_path, coords, faces)
        with pytest.warns(UserWarning) as w:
            read_geometry(surf_path, read_metadata=True)
        assert any('volume information contained' in str(ww.message) for ww in w)
        assert any('extension code' in str(ww.message) for ww in w)

        volume_info['head'] = [1, 2]
        with pytest.warns(UserWarning, match='Unknown extension'):
            write_geometry(surf_path, coords, faces, create_stamp, volume_info)

        volume_info['a'] = 0
        with pytest.raises(ValueError):
            write_geometry(surf_path, coords, faces, create_stamp, volume_info)

    assert create_stamp == read_create_stamp

    assert np.array_equal(coords, coords2)
    assert np.array_equal(faces, faces2)

    # Validate byte ordering
    coords_swapped = coords.byteswap()
    coords_swapped = coords_swapped.view(coords_swapped.dtype.newbyteorder())
    faces_swapped = faces.byteswap()
    faces_swapped = faces_swapped.view(faces_swapped.dtype.newbyteorder())
    assert np.array_equal(coords_swapped, coords)
    assert np.array_equal(faces_swapped, faces)


@freesurfer_test
@needs_nibabel_data('nitest-freesurfer')
def test_quad_geometry():
    """Test IO of freesurfer quad files."""
    new_quad = pjoin(
        get_nibabel_data(), 'nitest-freesurfer', 'subjects', 'bert', 'surf', 'lh.inflated.nofix'
    )
    coords, faces = read_geometry(new_quad)
    assert 0 == faces.min()
    assert coords.shape[0] == (faces.max() + 1)
    with InTemporaryDirectory():
        new_path = 'test'
        write_geometry(new_path, coords, faces)
        coords2, faces2 = read_geometry(new_path)
        assert np.array_equal(coords, coords2)
        assert np.array_equal(faces, faces2)


@freesurfer_test
def test_morph_data():
    """Test IO of morphometry data file (eg. curvature)."""
    curv_path = pjoin(data_path, 'surf', 'lh.curv')
    curv = read_morph_data(curv_path)
    assert -1.0 < curv.min() < 0
    assert 0 < curv.max() < 1.0
    with InTemporaryDirectory():
        new_path = 'test'
        write_morph_data(new_path, curv)
        curv2 = read_morph_data(new_path)
        assert np.array_equal(curv2, curv)


def test_write_morph_data():
    """Test write_morph_data edge cases"""
    values = np.arange(20, dtype='>f4')
    okay_shapes = [(20,), (20, 1), (20, 1, 1), (1, 20)]
    bad_shapes = [(10, 2), (1, 1, 20, 1, 1)]
    big_num = np.iinfo('i4').max + 1
    with InTemporaryDirectory():
        for shape in okay_shapes:
            write_morph_data('test.curv', values.reshape(shape))
            # Check ordering is preserved, regardless of shape
            assert np.array_equal(read_morph_data('test.curv'), values)

        with pytest.raises(ValueError):
            write_morph_data('test.curv', np.zeros(shape), big_num)
        # Windows 32-bit overflows Python int
        if np.dtype(int) != np.dtype(np.int32):
            with pytest.raises(ValueError):
                write_morph_data('test.curv', strided_scalar((big_num,)))
        for shape in bad_shapes:
            with pytest.raises(ValueError):
                write_morph_data('test.curv', values.reshape(shape))


@freesurfer_test
def test_annot():
    """Test IO of .annot against freesurfer example data."""
    annots = ['aparc', 'aparc.a2005s']
    for a in annots:
        annot_path = pjoin(data_path, 'label', f'lh.{a}.annot')

        labels, ctab, names = read_annot(annot_path)
        assert labels.shape == (163842,)
        assert ctab.shape == (len(names), 5)

        labels_orig = None
        if a == 'aparc':
            labels_orig, _, _ = read_annot(annot_path, orig_ids=True)
            np.testing.assert_array_equal(labels == -1, labels_orig == 0)
            # Handle different version of fsaverage
            content_hash = hashlib.md5(Path(annot_path).read_bytes()).hexdigest()
            if content_hash == 'bf0b488994657435cdddac5f107d21e8':
                assert np.sum(labels_orig == 0) == 13887
            elif content_hash == 'd4f5b7cbc2ed363ac6fcf89e19353504':
                assert np.sum(labels_orig == 1639705) == 13327
            else:
                raise RuntimeError(
                    'Unknown freesurfer file. Please report '
                    'the problem to the maintainer of nibabel.'
                )

        # Test equivalence of freesurfer- and nibabel-generated annot files
        # with respect to read_annot()
        with InTemporaryDirectory():
            annot_path = 'test'
            write_annot(annot_path, labels, ctab, names)

            labels2, ctab2, names2 = read_annot(annot_path)
            if labels_orig is not None:
                labels_orig_2, _, _ = read_annot(annot_path, orig_ids=True)

        assert np.array_equal(labels, labels2)
        if labels_orig is not None:
            assert np.array_equal(labels_orig, labels_orig_2)
        assert np.array_equal(ctab, ctab2)
        assert names == names2


def test_read_write_annot():
    """Test generating .annot file and reading it back."""
    # This annot file will store a LUT for a mesh made of 10 vertices, with
    # 3 colours in the LUT.
    nvertices = 10
    nlabels = 3
    names = [f'label {l}' for l in range(1, nlabels + 1)]
    # randomly generate a label for each vertex, making sure
    # that at least one of each label value is present. Label
    # values are in the range (0, nlabels-1) - they are used
    # as indices into the lookup table (generated below).
    labels = list(range(nlabels)) + list(np.random.randint(0, nlabels, nvertices - nlabels))
    labels = np.array(labels, dtype=np.int32)
    np.random.shuffle(labels)
    # Generate some random colours for the LUT
    rgbal = np.zeros((nlabels, 5), dtype=np.int32)
    rgbal[:, :4] = np.random.randint(0, 255, (nlabels, 4))
    # But make sure we have at least one large alpha, to make sure that when
    # it is packed into a signed 32 bit int, it results in a negative value
    # for the annotation value.
    rgbal[0, 3] = 255
    # Generate the annotation values for each LUT entry
    rgbal[:, 4] = rgbal[:, 0] + rgbal[:, 1] * (2**8) + rgbal[:, 2] * (2**16)
    annot_path = 'c.annot'
    with InTemporaryDirectory():
        write_annot(annot_path, labels, rgbal, names, fill_ctab=False)
        labels2, rgbal2, names2 = read_annot(annot_path)
        names2 = [n.decode('ascii') for n in names2]
        assert np.all(np.isclose(rgbal2, rgbal))
        assert np.all(np.isclose(labels2, labels))
        assert names2 == names


def test_write_annot_fill_ctab():
    """Test the `fill_ctab` parameter to :func:`.write_annot`."""
    nvertices = 10
    nlabels = 3
    names = [f'label {l}' for l in range(1, nlabels + 1)]
    labels = list(range(nlabels)) + list(np.random.randint(0, nlabels, nvertices - nlabels))
    labels = np.array(labels, dtype=np.int32)
    np.random.shuffle(labels)
    rgba = np.array(np.random.randint(0, 255, (nlabels, 4)), dtype=np.int32)
    annot_path = 'c.annot'
    with InTemporaryDirectory():
        write_annot(annot_path, labels, rgba, names, fill_ctab=True)
        labels2, rgbal2, names2 = read_annot(annot_path)
        names2 = [n.decode('ascii') for n in names2]
        assert np.all(np.isclose(rgbal2[:, :4], rgba))
        assert np.all(np.isclose(labels2, labels))
        assert names2 == names
        # make sure a warning is emitted if fill_ctab is False, and the
        # annotation values are wrong. Use orig_ids=True so we get those bad
        # values back.
        badannot = (10 * np.arange(nlabels, dtype=np.int32)).reshape(-1, 1)
        rgbal = np.hstack((rgba, badannot))
        with pytest.warns(
            UserWarning, match=f'Annotation values in {annot_path} will be incorrect'
        ):
            write_annot(annot_path, labels, rgbal, names, fill_ctab=False)
        labels2, rgbal2, names2 = read_annot(annot_path, orig_ids=True)
        names2 = [n.decode('ascii') for n in names2]
        assert np.all(np.isclose(rgbal2[:, :4], rgba))
        assert np.all(np.isclose(labels2, badannot[labels].squeeze()))
        assert names2 == names
        # make sure a warning is *not* emitted if fill_ctab is False, but the
        # annotation values are correct.
        rgbal = np.hstack((rgba, np.zeros((nlabels, 1), dtype=np.int32)))
        rgbal[:, 4] = rgbal[:, 0] + rgbal[:, 1] * (2**8) + rgbal[:, 2] * (2**16)
        with clear_and_catch_warnings() as w:
            write_annot(annot_path, labels, rgbal, names, fill_ctab=False)
        assert all(
            f'Annotation values in {annot_path} will be incorrect' != str(ww.message) for ww in w
        )
        labels2, rgbal2, names2 = read_annot(annot_path)
        names2 = [n.decode('ascii') for n in names2]
        assert np.all(np.isclose(rgbal2[:, :4], rgba))
        assert np.all(np.isclose(labels2, labels))
        assert names2 == names


def test_read_annot_old_format():
    """Test reading an old-style .annot file."""

    def gen_old_annot_file(fpath, nverts, labels, rgba, names):
        dt = '>i'
        vdata = np.zeros((nverts, 2), dtype=dt)
        vdata[:, 0] = np.arange(nverts)
        vdata[:, [1]] = _pack_rgb(rgba[labels, :3])
        fbytes = b''
        # number of vertices
        fbytes += struct.pack(dt, nverts)
        # vertices + annotation values
        fbytes += vdata.astype(dt).tobytes()
        # is there a colour table?
        fbytes += struct.pack(dt, 1)
        # number of entries in colour table
        fbytes += struct.pack(dt, rgba.shape[0])
        # length of orig_tab string
        fbytes += struct.pack(dt, 5)
        fbytes += b'abcd\x00'
        for i in range(rgba.shape[0]):
            # length of entry name (+1 for terminating byte)
            fbytes += struct.pack(dt, len(names[i]) + 1)
            fbytes += names[i].encode('ascii') + b'\x00'
            fbytes += rgba[i, :].astype(dt).tobytes()
        with open(fpath, 'wb') as f:
            f.write(fbytes)

    with InTemporaryDirectory():
        nverts = 10
        nlabels = 3
        names = [f'Label {l}' for l in range(nlabels)]
        labels = np.concatenate(
            (np.arange(nlabels), np.random.randint(0, nlabels, nverts - nlabels))
        )
        np.random.shuffle(labels)
        rgba = np.random.randint(0, 255, (nlabels, 4))
        # write an old .annot file
        gen_old_annot_file('blah.annot', nverts, labels, rgba, names)
        # read it back
        rlabels, rrgba, rnames = read_annot('blah.annot')
        rnames = [n.decode('ascii') for n in rnames]
        assert np.all(np.isclose(labels, rlabels))
        assert np.all(np.isclose(rgba, rrgba[:, :4]))
        assert names == rnames


@freesurfer_test
def test_label():
    """Test IO of .label"""
    label_path = pjoin(data_path, 'label', 'lh.cortex.label')
    label = read_label(label_path)
    # XXX : test more
    assert label.min() >= 0
    assert label.max() <= 163841
    assert label.shape[0] <= 163842

    labels, scalars = read_label(label_path, True)
    assert np.all(labels == label)
    assert len(labels) == len(scalars)


def test_write_annot_maxstruct():
    """Test writing ANNOT files with repeated labels"""
    with InTemporaryDirectory():
        nlabels = 3
        names = [f'label {l}' for l in range(1, nlabels + 1)]
        # max label < n_labels
        labels = np.array([1, 1, 1], dtype=np.int32)
        rgba = np.array(np.random.randint(0, 255, (nlabels, 4)), dtype=np.int32)
        annot_path = 'c.annot'

        write_annot(annot_path, labels, rgba, names)
        # Validate the file can be read
        rt_labels, rt_ctab, rt_names = read_annot(annot_path)
        # Check round-trip
        assert np.array_equal(labels, rt_labels)
        assert np.array_equal(rgba, rt_ctab[:, :4])
        assert names == [n.decode('ascii') for n in rt_names]
