"""Tests for ``get_nibabel_data``"""

import os
from os.path import dirname, isdir, realpath
from os.path import join as pjoin

from . import nibabel_data as nibd

MY_DIR = dirname(__file__)


def setup_module():
    nibd.environ = {}


def teardown_module():
    nibd.environ = os.environ


def test_get_nibabel_data():
    # Test getting directory
    local_data = realpath(pjoin(MY_DIR, '..', '..', 'nibabel-data'))
    if isdir(local_data):
        assert nibd.get_nibabel_data() == local_data
    else:
        assert nibd.get_nibabel_data() == ''
    nibd.environ['NIBABEL_DATA_DIR'] = 'not_a_path'
    assert nibd.get_nibabel_data() == ''
    nibd.environ['NIBABEL_DATA_DIR'] = MY_DIR
    assert nibd.get_nibabel_data() == MY_DIR
