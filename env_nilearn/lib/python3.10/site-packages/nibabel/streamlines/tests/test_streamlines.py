import os
import unittest
import warnings
from io import BytesIO
from os.path import join as pjoin

import numpy as np
import pytest

import nibabel as nib
from nibabel.testing import clear_and_catch_warnings, data_path, error_warnings
from nibabel.tmpdirs import InTemporaryDirectory

from .. import FORMATS, trk
from ..tractogram import LazyTractogram, Tractogram
from ..tractogram_file import ExtensionWarning, TractogramFile
from .test_tractogram import assert_tractogram_equal

DATA = {}


def setup_module():
    global DATA
    DATA['empty_filenames'] = [pjoin(data_path, 'empty' + ext) for ext in FORMATS.keys()]
    DATA['simple_filenames'] = [pjoin(data_path, 'simple' + ext) for ext in FORMATS.keys()]
    DATA['complex_filenames'] = [
        pjoin(data_path, 'complex' + ext)
        for ext, cls in FORMATS.items()
        if (cls.SUPPORTS_DATA_PER_POINT or cls.SUPPORTS_DATA_PER_STREAMLINE)
    ]

    DATA['streamlines'] = [
        np.arange(1 * 3, dtype='f4').reshape((1, 3)),
        np.arange(2 * 3, dtype='f4').reshape((2, 3)),
        np.arange(5 * 3, dtype='f4').reshape((5, 3)),
    ]

    fa = [
        np.array([[0.2]], dtype='f4'),
        np.array([[0.3], [0.4]], dtype='f4'),
        np.array([[0.5], [0.6], [0.6], [0.7], [0.8]], dtype='f4'),
    ]

    colors = [
        np.array([(1, 0, 0)] * 1, dtype='f4'),
        np.array([(0, 1, 0)] * 2, dtype='f4'),
        np.array([(0, 0, 1)] * 5, dtype='f4'),
    ]

    mean_curvature = [
        np.array([1.11], dtype='f4'),
        np.array([2.11], dtype='f4'),
        np.array([3.11], dtype='f4'),
    ]

    mean_torsion = [
        np.array([1.22], dtype='f4'),
        np.array([2.22], dtype='f4'),
        np.array([3.22], dtype='f4'),
    ]

    mean_colors = [
        np.array([1, 0, 0], dtype='f4'),
        np.array([0, 1, 0], dtype='f4'),
        np.array([0, 0, 1], dtype='f4'),
    ]

    DATA['data_per_point'] = {'colors': colors, 'fa': fa}
    DATA['data_per_streamline'] = {
        'mean_curvature': mean_curvature,
        'mean_torsion': mean_torsion,
        'mean_colors': mean_colors,
    }

    DATA['empty_tractogram'] = Tractogram(affine_to_rasmm=np.eye(4))
    DATA['simple_tractogram'] = Tractogram(DATA['streamlines'], affine_to_rasmm=np.eye(4))
    DATA['complex_tractogram'] = Tractogram(
        DATA['streamlines'],
        DATA['data_per_streamline'],
        DATA['data_per_point'],
        affine_to_rasmm=np.eye(4),
    )


def test_is_supported_detect_format(tmp_path):
    # Test is_supported and detect_format functions
    # Empty file/string
    f = BytesIO()
    assert not nib.streamlines.is_supported(f)
    assert not nib.streamlines.is_supported('')
    assert nib.streamlines.detect_format(f) is None
    assert nib.streamlines.detect_format('') is None

    # Valid file without extension
    for tfile_cls in FORMATS.values():
        f = BytesIO()
        f.write(tfile_cls.MAGIC_NUMBER)
        f.seek(0, os.SEEK_SET)
        assert nib.streamlines.is_supported(f)
        assert nib.streamlines.detect_format(f) is tfile_cls

    # Wrong extension but right magic number
    for tfile_cls in FORMATS.values():
        fpath = tmp_path / 'test.txt'
        with open(fpath, 'w+b') as f:
            f.write(tfile_cls.MAGIC_NUMBER)
            f.seek(0, os.SEEK_SET)
            assert nib.streamlines.is_supported(f)
            assert nib.streamlines.detect_format(f) is tfile_cls

    # Good extension but wrong magic number
    for ext, tfile_cls in FORMATS.items():
        fpath = tmp_path / f'test{ext}'
        with open(fpath, 'w+b') as f:
            f.write(b'pass')
            f.seek(0, os.SEEK_SET)
            assert not nib.streamlines.is_supported(f)
            assert nib.streamlines.detect_format(f) is None

    # Wrong extension, string only
    f = 'my_tractogram.asd'
    assert not nib.streamlines.is_supported(f)
    assert nib.streamlines.detect_format(f) is None

    # Good extension, string only
    for ext, tfile_cls in FORMATS.items():
        f = 'my_tractogram' + ext
        assert nib.streamlines.is_supported(f)
        assert nib.streamlines.detect_format(f) == tfile_cls

    # Extension should not be case-sensitive.
    for ext, tfile_cls in FORMATS.items():
        f = 'my_tractogram' + ext.upper()
        assert nib.streamlines.detect_format(f) is tfile_cls


class TestLoadSave(unittest.TestCase):
    def test_load_empty_file(self):
        for lazy_load in [False, True]:
            for empty_filename in DATA['empty_filenames']:
                tfile = nib.streamlines.load(empty_filename, lazy_load=lazy_load)
                assert isinstance(tfile, TractogramFile)

                if lazy_load:
                    assert type(tfile.tractogram), Tractogram
                else:
                    assert type(tfile.tractogram), LazyTractogram

                with pytest.warns(Warning) if lazy_load else error_warnings():
                    assert_tractogram_equal(tfile.tractogram, DATA['empty_tractogram'])

    def test_load_simple_file(self):
        for lazy_load in [False, True]:
            for simple_filename in DATA['simple_filenames']:
                tfile = nib.streamlines.load(simple_filename, lazy_load=lazy_load)
                assert isinstance(tfile, TractogramFile)

                if lazy_load:
                    assert type(tfile.tractogram), Tractogram
                else:
                    assert type(tfile.tractogram), LazyTractogram

                with pytest.warns(Warning) if lazy_load else error_warnings():
                    assert_tractogram_equal(tfile.tractogram, DATA['simple_tractogram'])

    def test_load_complex_file(self):
        for lazy_load in [False, True]:
            for complex_filename in DATA['complex_filenames']:
                tfile = nib.streamlines.load(complex_filename, lazy_load=lazy_load)
                assert isinstance(tfile, TractogramFile)

                if lazy_load:
                    assert type(tfile.tractogram), Tractogram
                else:
                    assert type(tfile.tractogram), LazyTractogram

                tractogram = Tractogram(DATA['streamlines'], affine_to_rasmm=np.eye(4))

                if tfile.SUPPORTS_DATA_PER_POINT:
                    tractogram.data_per_point = DATA['data_per_point']

                if tfile.SUPPORTS_DATA_PER_STREAMLINE:
                    data = DATA['data_per_streamline']
                    tractogram.data_per_streamline = data

                with pytest.warns(Warning) if lazy_load else error_warnings():
                    assert_tractogram_equal(tfile.tractogram, tractogram)

    def test_save_tractogram_file(self):
        tractogram = Tractogram(DATA['streamlines'], affine_to_rasmm=np.eye(4))
        trk_file = trk.TrkFile(tractogram)

        # No need for keyword arguments.
        with pytest.raises(ValueError):
            nib.streamlines.save(trk_file, 'dummy.trk', header={})

        # Wrong extension.
        with pytest.warns(ExtensionWarning, match='extension'):
            trk_file = trk.TrkFile(tractogram)
            with pytest.raises(ValueError):
                nib.streamlines.save(trk_file, 'dummy.tck', header={})

        with InTemporaryDirectory():
            nib.streamlines.save(trk_file, 'dummy.trk')
            tfile = nib.streamlines.load('dummy.trk', lazy_load=False)
            assert_tractogram_equal(tfile.tractogram, tractogram)

    def test_save_empty_file(self):
        tractogram = Tractogram(affine_to_rasmm=np.eye(4))
        for ext in FORMATS:
            with InTemporaryDirectory():
                filename = 'streamlines' + ext
                nib.streamlines.save(tractogram, filename)
                tfile = nib.streamlines.load(filename, lazy_load=False)
                assert_tractogram_equal(tfile.tractogram, tractogram)

    def test_save_simple_file(self):
        tractogram = Tractogram(DATA['streamlines'], affine_to_rasmm=np.eye(4))
        for ext in FORMATS:
            with InTemporaryDirectory():
                filename = 'streamlines' + ext
                nib.streamlines.save(tractogram, filename)
                tfile = nib.streamlines.load(filename, lazy_load=False)
                assert_tractogram_equal(tfile.tractogram, tractogram)

    def test_save_complex_file(self):
        complex_tractogram = Tractogram(
            DATA['streamlines'],
            DATA['data_per_streamline'],
            DATA['data_per_point'],
            affine_to_rasmm=np.eye(4),
        )

        for ext, cls in FORMATS.items():
            with InTemporaryDirectory():
                filename = 'streamlines' + ext

                # If streamlines format does not support saving data
                # per point or data per streamline, warning messages
                # should be issued.
                nb_expected_warnings = (not cls.SUPPORTS_DATA_PER_POINT) + (
                    not cls.SUPPORTS_DATA_PER_STREAMLINE
                )

                with clear_and_catch_warnings() as w:
                    warnings.simplefilter('always')
                    nib.streamlines.save(complex_tractogram, filename)
                assert len(w) == nb_expected_warnings

                tractogram = Tractogram(DATA['streamlines'], affine_to_rasmm=np.eye(4))

                if cls.SUPPORTS_DATA_PER_POINT:
                    tractogram.data_per_point = DATA['data_per_point']

                if cls.SUPPORTS_DATA_PER_STREAMLINE:
                    data = DATA['data_per_streamline']
                    tractogram.data_per_streamline = data

                tfile = nib.streamlines.load(filename, lazy_load=False)
                assert_tractogram_equal(tfile.tractogram, tractogram)

    def test_save_sliced_tractogram(self):
        tractogram = Tractogram(DATA['streamlines'], affine_to_rasmm=np.eye(4))
        original_tractogram = tractogram.copy()
        for ext in FORMATS:
            with InTemporaryDirectory():
                filename = 'streamlines' + ext
                nib.streamlines.save(tractogram[::2], filename)
                tfile = nib.streamlines.load(filename, lazy_load=False)
                assert_tractogram_equal(tfile.tractogram, tractogram[::2])
                # Make sure original tractogram hasn't changed.
                assert_tractogram_equal(tractogram, original_tractogram)

    def test_load_unknown_format(self):
        with pytest.raises(ValueError):
            nib.streamlines.load('')

    def test_save_unknown_format(self):
        with pytest.raises(ValueError):
            nib.streamlines.save(Tractogram(), '')

    def test_save_from_generator(self):
        tractogram = Tractogram(DATA['streamlines'], affine_to_rasmm=np.eye(4))

        # Just to create a generator
        for ext in FORMATS:
            filtered = (s for s in tractogram.streamlines if True)
            lazy_tractogram = LazyTractogram(lambda: filtered, affine_to_rasmm=np.eye(4))

            with InTemporaryDirectory():
                filename = 'streamlines' + ext
                nib.streamlines.save(lazy_tractogram, filename)
                tfile = nib.streamlines.load(filename, lazy_load=False)
                assert_tractogram_equal(tfile.tractogram, tractogram)
