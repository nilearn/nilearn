"""
Test the datasets module
"""
# Author: Alexandre Abraham
# License: simplified BSD

import os
import shutil
from tempfile import mkdtemp, mktemp
import numpy as np

from nose import with_setup
from nose.tools import assert_true, assert_false, assert_equal

from .. import datasets
from .._utils.testing import mock_urllib2, mock_chunk_read_,\
    mock_uncompress_file, mock_get_dataset

currdir = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(currdir, 'data')
tmpdir = None


def setup_tmpdata():
    # create temporary dir
    global tmpdir
    tmpdir = mkdtemp()


def teardown_tmpdata():
    # remove temporary dir
    if tmpdir is not None:
        shutil.rmtree(tmpdir)


def test_md5_sum_file():
    # Create dummy temporary file
    f = mktemp()
    out = open(f, 'w')
    out.write('abcfeg')
    out.close()
    assert_equal(datasets._md5_sum_file(f), '18f32295c556b2a1a3a8e68fe1ad40f7')
    os.remove(f)


def test_read_md5_sum_file():
    # Create dummy temporary file
    f = mktemp()
    out = open(f, 'w')
    out.write('20861c8c3fe177da19a7e9539a5dbac  /tmp/test\n'
              '70886dcabe7bf5c5a1c24ca24e4cbd94  test/some_image.nii')
    out.close()
    h = datasets._read_md5_sum_file(f)
    assert_true('/tmp/test' in h)
    assert_false('/etc/test' in h)
    assert_equal(h['test/some_image.nii'], '70886dcabe7bf5c5a1c24ca24e4cbd94')
    assert_equal(h['/tmp/test'], '20861c8c3fe177da19a7e9539a5dbac')
    os.remove(f)


@with_setup(setup_tmpdata, teardown_tmpdata)
def test_fetch_haxby_simple():
    local_url = "file://" + os.path.join(datadir, "pymvpa-exampledata.tar.bz2")
    haxby = datasets.fetch_haxby_simple(data_dir=tmpdir, url=local_url)
    datasetdir = os.path.join(tmpdir, 'haxby2001_simple', 'pymvpa-exampledata')
    for key, file in [
            ('session_target', 'attributes.txt'),
            ('func', 'bold.nii.gz'),
            ('mask', 'mask.nii.gz'),
            ('conditions_target', 'attributes_literal.txt')]:
        assert_equal(haxby[key], os.path.join(datasetdir, file))
        assert_true(os.path.exists(os.path.join(datasetdir, file)))
