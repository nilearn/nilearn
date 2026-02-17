# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Test scripts

Test running scripts
"""

from io import StringIO
from os.path import join as pjoin

import numpy as np
import pytest

import nibabel as nib
from nibabel.cmdline.diff import (
    display_diff,
    get_data_diff,
    get_data_hash_diff,
    get_headers_diff,
    main,
)
from nibabel.cmdline.utils import (
    ap,
    safe_get,
    table2string,
)
from nibabel.testing import data_path


def test_table2string():
    assert table2string([['A', 'B', 'C', 'D'], ['E', 'F', 'G', 'H']]) == 'A B C D\nE F G H\n'
    assert (
        table2string(
            [
                ["Let's", 'Make', 'Tests', 'And'],
                ['Have', 'Lots', 'Of', 'Fun'],
                ['With', 'Python', 'Guys', '!'],
            ]
        )
        == "Let's  Make  Tests And\n Have  Lots    Of  Fun" + '\n With Python  Guys  !\n'
    )


def test_ap():
    assert ap([1, 2], '%2d') == ' 1,  2'
    assert ap([1, 2], '%3d') == '  1,   2'
    assert ap([1, 2], '%-2d') == '1 , 2 '
    assert ap([1, 2], '%d', '+') == '1+2'
    assert ap([1, 2, 3], '%d', '-') == '1-2-3'


def test_safe_get():
    class TestObject:
        def __init__(self, test=None):
            self.test = test

        def get_test(self):
            return self.test

    test = TestObject()
    test.test = 2

    assert safe_get(test, 'test') == 2
    assert safe_get(test, 'failtest') == '-'


def test_get_headers_diff():
    fnames = [pjoin(data_path, f) for f in ('standard.nii.gz', 'example4d.nii.gz')]
    actual_difference = get_headers_diff([nib.load(f).header for f in fnames])
    expected_difference = {
        'regular': [np.asarray(b''), np.asarray(b'r')],
        'dim_info': [np.asarray(0, 'uint8'), np.asarray(57, 'uint8')],
        'dim': [
            np.array([3, 4, 5, 7, 1, 1, 1, 1], 'int16'),
            np.array([4, 128, 96, 24, 2, 1, 1, 1], 'int16'),
        ],
        'datatype': [np.array(2, 'uint8'), np.array(4, 'uint8')],
        'bitpix': [np.array(8, 'uint8'), np.array(16, 'uint8')],
        'pixdim': [
            np.array([1.0, 1.0, 3.0, 2.0, 1.0, 1.0, 1.0, 1.0], 'float32'),
            np.array(
                [
                    -1.00000000e00,
                    2.00000000e00,
                    2.00000000e00,
                    2.19999909e00,
                    2.00000000e03,
                    1.00000000e00,
                    1.00000000e00,
                    1.00000000e00,
                ],
                'float32',
            ),
        ],
        'slice_end': [np.array(0, 'uint8'), np.array(23, 'uint8')],
        'xyzt_units': [np.array(0, 'uint8'), np.array(10, 'uint8')],
        'cal_max': [
            np.array(0.0, 'float32'),
            np.asarray(1162.0, 'float32'),
        ],
        'descrip': [
            np.array(b'', 'S80'),
            np.array(b'FSL3.3\x00 v2.25 NIfTI-1 Single file format', 'S80'),
        ],
        'qform_code': [np.array(0, 'int16'), np.array(1, 'int16')],
        'sform_code': [np.array(2, 'int16'), np.array(1, 'int16')],
        'quatern_b': [
            np.array(0.0, 'float32'),
            np.array(-1.9451068140294884e-26, 'float32'),
        ],
        'quatern_c': [
            np.array(0.0, 'float32'),
            np.array(-0.9967085123062134, 'float32'),
        ],
        'quatern_d': [
            np.array(0.0, 'float32'),
            np.array(-0.0810687392950058, 'float32'),
        ],
        'qoffset_x': [
            np.array(0.0, 'float32'),
            np.array(117.8551025390625, 'float32'),
        ],
        'qoffset_y': [
            np.array(0.0, 'float32'),
            np.array(-35.72294235229492, 'float32'),
        ],
        'qoffset_z': [
            np.array(0.0, 'float32'),
            np.array(-7.248798370361328, 'float32'),
        ],
        'srow_x': [
            np.array([1.0, 0.0, 0.0, 0.0], 'float32'),
            np.array([-2.00000000e00, 6.71471565e-19, 9.08102451e-18, 1.17855103e02], 'float32'),
        ],
        'srow_y': [
            np.array([0.0, 3.0, 0.0, 0.0], 'float32'),
            np.array([-6.71471565e-19, 1.97371149e00, -3.55528235e-01, -3.57229424e01], 'float32'),
        ],
        'srow_z': [
            np.array([0.0, 0.0, 2.0, 0.0], 'float32'),
            np.array([8.25548089e-18, 3.23207617e-01, 2.17108178e00, -7.24879837e00], 'float32'),
        ],
    }

    np.testing.assert_equal(actual_difference, expected_difference)


def test_display_diff():
    bogus_names = ['hellokitty.nii.gz', 'privettovarish.nii.gz']

    dict_values = {
        'datatype': [np.array(2, 'uint8'), np.array(4, 'uint8')],
        'bitpix': [np.array(8, 'uint8'), np.array(16, 'uint8')],
    }

    expected_output = """\
These files are different.
Field/File     \
1:hellokitty.nii.gz                                    \
2:privettovarish.nii.gz                                \n\
datatype       \
2                                                      \
4                                                      \n\
bitpix         \
8                                                      \
16                                                     \n"""

    assert display_diff(bogus_names, dict_values) == expected_output


def test_get_data_diff():
    #  testing for identical files specifically as md5 may vary by computer
    test_names = [pjoin(data_path, f) for f in ('standard.nii.gz', 'standard.nii.gz')]
    assert get_data_hash_diff(test_names) == []

    #  testing the maximum relative and absolute differences' different use cases
    test_array = np.arange(16).reshape(4, 4)
    test_array_2 = np.arange(1, 17).reshape(4, 4)
    test_array_3 = np.arange(2, 18).reshape(4, 4)
    test_array_4 = np.arange(100).reshape(10, 10)
    test_array_5 = np.arange(64).reshape(8, 8)

    # same shape, 2 files
    assert get_data_diff([test_array, test_array_2]) == {
        'DATA(diff 1:)': [None, {'abs': 1, 'rel': 2.0}]
    }

    # same shape, 3 files
    assert get_data_diff([test_array, test_array_2, test_array_3]) == {
        'DATA(diff 1:)': [
            None,
            {'abs': 1, 'rel': 2.0},
            {'abs': 2, 'rel': 2.0},
        ],
        'DATA(diff 2:)': [None, None, {'abs': 1, 'rel': 0.66666666666666663}],
    }

    # same shape, 2 files, modified maximum abs/rel
    assert get_data_diff([test_array, test_array_2], max_abs=2, max_rel=2) == {}

    # different shape, 2 files
    assert get_data_diff([test_array_2, test_array_4]) == {
        'DATA(diff 1:)': [None, {'CMP': 'incompat'}]
    }

    # different shape, 3 files
    assert get_data_diff([test_array_4, test_array_5, test_array_2]) == {
        'DATA(diff 1:)': [None, {'CMP': 'incompat'}, {'CMP': 'incompat'}],
        'DATA(diff 2:)': [None, None, {'CMP': 'incompat'}],
    }

    test_return = get_data_diff([test_array, test_array_2], dtype=np.float32)
    assert type(test_return['DATA(diff 1:)'][1]['abs']) is np.float32
    assert type(test_return['DATA(diff 1:)'][1]['rel']) is np.float32

    test_return_2 = get_data_diff([test_array, test_array_2, test_array_3])
    assert type(test_return_2['DATA(diff 1:)'][1]['abs']) is np.float64
    assert type(test_return_2['DATA(diff 1:)'][1]['rel']) is np.float64
    assert type(test_return_2['DATA(diff 2:)'][2]['abs']) is np.float64
    assert type(test_return_2['DATA(diff 2:)'][2]['rel']) is np.float64


def test_main():
    test_names = [pjoin(data_path, f) for f in ('standard.nii.gz', 'example4d.nii.gz')]
    expected_difference = {
        'regular': [np.asarray(b''), np.asarray(b'r')],
        'dim_info': [np.asarray(0, 'uint8'), np.asarray(57, 'uint8')],
        'dim': [
            np.array([3, 4, 5, 7, 1, 1, 1, 1], 'int16'),
            np.array([4, 128, 96, 24, 2, 1, 1, 1], 'int16'),
        ],
        'datatype': [np.array(2, 'uint8'), np.array(4, 'uint8')],
        'bitpix': [np.array(8, 'uint8'), np.array(16, 'uint8')],
        'pixdim': [
            np.array([1.0, 1.0, 3.0, 2.0, 1.0, 1.0, 1.0, 1.0], 'float32'),
            np.array(
                [
                    -1.00000000e00,
                    2.00000000e00,
                    2.00000000e00,
                    2.19999909e00,
                    2.00000000e03,
                    1.00000000e00,
                    1.00000000e00,
                    1.00000000e00,
                ],
                'float32',
            ),
        ],
        'slice_end': [np.array(0, 'uint8'), np.array(23, 'uint8')],
        'xyzt_units': [np.array(0, 'uint8'), np.array(10, 'uint8')],
        'cal_max': [
            np.array(0.0, 'float32'),
            np.asarray(1162.0, 'float32'),
        ],
        'descrip': [
            np.array(b'', 'S80'),
            np.array(b'FSL3.3\x00 v2.25 NIfTI-1 Single file format', 'S80'),
        ],
        'qform_code': [np.array(0, 'int16'), np.array(1, 'int16')],
        'sform_code': [np.array(2, 'int16'), np.array(1, 'int16')],
        'quatern_b': [
            np.array(0.0, 'float32'),
            np.array(-1.9451068140294884e-26, 'float32'),
        ],
        'quatern_c': [
            np.array(0.0, 'float32'),
            np.array(-0.9967085123062134, 'float32'),
        ],
        'quatern_d': [
            np.array(0.0, 'float32'),
            np.array(-0.0810687392950058, 'float32'),
        ],
        'qoffset_x': [
            np.array(0.0, 'float32'),
            np.array(117.8551025390625, 'float32'),
        ],
        'qoffset_y': [
            np.array(0.0, 'float32'),
            np.array(-35.72294235229492, 'float32'),
        ],
        'qoffset_z': [
            np.array(0.0, 'float32'),
            np.array(-7.248798370361328, 'float32'),
        ],
        'srow_x': [
            np.array([1.0, 0.0, 0.0, 0.0], 'float32'),
            np.array([-2.00000000e00, 6.71471565e-19, 9.08102451e-18, 1.17855103e02], 'float32'),
        ],
        'srow_y': [
            np.array([0.0, 3.0, 0.0, 0.0], 'float32'),
            np.array([-6.71471565e-19, 1.97371149e00, -3.55528235e-01, -3.57229424e01], 'float32'),
        ],
        'srow_z': [
            np.array([0.0, 0.0, 2.0, 0.0], 'float32'),
            np.array([8.25548089e-18, 3.23207617e-01, 2.17108178e00, -7.24879837e00], 'float32'),
        ],
        'DATA(md5)': ['0a2576dd6badbb25bfb3b12076df986b', 'b0abbc492b4fd533b2c80d82570062cf'],
    }

    with pytest.raises(SystemExit):
        np.testing.assert_equal(main(test_names, StringIO()), expected_difference)

    test_names_2 = [pjoin(data_path, f) for f in ('standard.nii.gz', 'standard.nii.gz')]

    with pytest.raises(SystemExit):
        assert main(test_names_2, StringIO()) == 'These files are identical.'
