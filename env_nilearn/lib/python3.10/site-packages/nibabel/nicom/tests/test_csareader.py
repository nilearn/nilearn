"""Testing Siemens CSA header reader"""

import gzip
from copy import deepcopy
from os.path import join as pjoin

import numpy as np
import pytest

from .. import csareader as csa
from .. import dwiparams as dwp
from . import dicom_test, pydicom
from .test_dicomwrappers import DATA, IO_DATA_PATH

CSA2_B0 = open(pjoin(IO_DATA_PATH, 'csa2_b0.bin'), 'rb').read()
CSA2_B1000 = open(pjoin(IO_DATA_PATH, 'csa2_b1000.bin'), 'rb').read()
CSA2_0len = gzip.open(pjoin(IO_DATA_PATH, 'csa2_zero_len.bin.gz'), 'rb').read()
CSA_STR_valid = open(pjoin(IO_DATA_PATH, 'csa_str_valid.bin'), 'rb').read()
CSA_STR_1001n_items = open(pjoin(IO_DATA_PATH, 'csa_str_1001n_items.bin'), 'rb').read()


@dicom_test
def test_csa_header_read():
    hdr = csa.get_csa_header(DATA, 'image')
    assert hdr['n_tags'] == 83
    assert csa.get_csa_header(DATA, 'series')['n_tags'] == 65
    with pytest.raises(ValueError):
        csa.get_csa_header(DATA, 'xxxx')
    assert csa.is_mosaic(hdr)
    # Get a shallow copy of the data, lacking the CSA marker
    # Need to do it this way because del appears broken in pydicom 0.9.7
    data2 = pydicom.dataset.Dataset()
    for element in DATA:
        if (element.tag.group, element.tag.elem) != (0x29, 0x10):
            data2.add(element)
    assert csa.get_csa_header(data2, 'image') is None
    # Add back the marker - CSA works again
    data2[(0x29, 0x10)] = DATA[(0x29, 0x10)]
    assert csa.is_mosaic(csa.get_csa_header(data2, 'image'))


def test_csas0():
    for csa_str in (CSA2_B0, CSA2_B1000):
        csa_info = csa.read(csa_str)
        assert csa_info['type'] == 2
        assert csa_info['n_tags'] == 83
        tags = csa_info['tags']
        assert len(tags) == 83
        n_o_m = tags['NumberOfImagesInMosaic']
        assert n_o_m['items'] == [48]
    csa_info = csa.read(CSA2_B1000)
    b_matrix = csa_info['tags']['B_matrix']
    assert len(b_matrix['items']) == 6
    b_value = csa_info['tags']['B_value']
    assert b_value['items'] == [1000]


def test_csa_len0():
    # We did get a failure for item with item_len of 0 - gh issue #92
    csa_info = csa.read(CSA2_0len)
    assert csa_info['type'] == 2
    assert csa_info['n_tags'] == 44
    tags = csa_info['tags']
    assert len(tags) == 44


def test_csa_nitem():
    # testing csa.read's ability to raise an error when n_items >= 200
    with pytest.raises(csa.CSAReadError):
        csa.read(CSA_STR_1001n_items)
    # OK when < 1000
    csa_info = csa.read(CSA_STR_valid)
    assert len(csa_info['tags']) == 1
    # OK after changing module global
    n_items_thresh = csa.MAX_CSA_ITEMS
    try:
        csa.MAX_CSA_ITEMS = 2000
        csa_info = csa.read(CSA_STR_1001n_items)
        assert len(csa_info['tags']) == 1
    finally:
        csa.MAX_CSA_ITEMS = n_items_thresh


def test_csa_params():
    for csa_str in (CSA2_B0, CSA2_B1000):
        csa_info = csa.read(csa_str)
        n_o_m = csa.get_n_mosaic(csa_info)
        assert n_o_m == 48
        snv = csa.get_slice_normal(csa_info)
        assert snv.shape == (3,)
        assert np.allclose(1, np.sqrt((snv * snv).sum()))
        amt = csa.get_acq_mat_txt(csa_info)
        assert amt == '128p*128'
    csa_info = csa.read(CSA2_B0)
    b_matrix = csa.get_b_matrix(csa_info)
    assert b_matrix is None
    b_value = csa.get_b_value(csa_info)
    assert b_value == 0
    g_vector = csa.get_g_vector(csa_info)
    assert g_vector is None
    csa_info = csa.read(CSA2_B1000)
    b_matrix = csa.get_b_matrix(csa_info)
    assert b_matrix.shape == (3, 3)
    # check (by absence of error) that the B matrix is positive
    # semi-definite.
    dwp.B2q(b_matrix)  # no error
    b_value = csa.get_b_value(csa_info)
    assert b_value == 1000
    g_vector = csa.get_g_vector(csa_info)
    assert g_vector.shape == (3,)
    assert np.allclose(1, np.sqrt((g_vector * g_vector).sum()))


def test_ice_dims():
    ex_dims0 = ['X', '1', '1', '1', '1', '1', '1', '48', '1', '1', '1', '1', '201']
    ex_dims1 = ['X', '1', '1', '1', '2', '1', '1', '48', '1', '1', '1', '1', '201']
    for csa_str, ex_dims in ((CSA2_B0, ex_dims0), (CSA2_B1000, ex_dims1)):
        csa_info = csa.read(csa_str)
        assert csa.get_ice_dims(csa_info) == ex_dims
    assert csa.get_ice_dims({}) is None


@dicom_test
def test_missing_csa_elem():
    # Test that we get None instead of raising an Exception when the file has
    # the PrivateCreator element for the CSA dict but not the element with the
    # actual CSA header (perhaps due to anonymization)
    dcm = deepcopy(DATA)
    csa_tag = pydicom.dataset.Tag(0x29, 0x1010)
    del dcm[csa_tag]
    hdr = csa.get_csa_header(dcm, 'image')
    assert hdr is None
