"""
Test the datasets module
"""
# Author: Alexandre Abraham
# License: simplified BSD

import os
import shutil
from tempfile import mkdtemp
import numpy as np

from nose import with_setup
from nose.tools import assert_true, assert_equal

from .. import datasets
from ..testing import mock_urllib2, mock_chunk_read_, mock_uncompress_file, \
    mock_get_dataset

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


def test_fetch_nyu_rest():
    # Mock urllib2 of the dataset fetcher
    # _urllib2_ref = datasets.mldata.urllib2
    mock = mock_urllib2()
    datasets.urllib2 = mock
    datasets._chunk_read_ = mock_chunk_read_
    datasets._uncompress_file = mock_uncompress_file
    datasets._get_dataset = mock_get_dataset

    # First session, all subjects
    setup_tmpdata()
    nyu = datasets.fetch_nyu_rest(data_dir=tmpdir)
    assert_equal(len(mock.urls), 2)
    assert_equal(len(nyu.func), 25)
    assert_equal(len(nyu.anat_anon), 25)
    assert_equal(len(nyu.anat_skull), 25)
    assert_true(np.all(np.asarray(nyu.session) == 1))
    teardown_tmpdata()

    # All sessions, 12 subjects
    setup_tmpdata()
    mock.reset()
    nyu = datasets.fetch_nyu_rest(data_dir=tmpdir, sessions=[1, 2, 3],
                                  n_subjects=12)
    assert_equal(len(mock.urls), 3)
    assert_equal(len(nyu.func), 36)
    assert_equal(len(nyu.anat_anon), 36)
    assert_equal(len(nyu.anat_skull), 36)
    s = np.asarray(nyu.session)
    assert_true(np.all(s[:12] == 1))
    assert_true(np.all(s[12:24] == 2))
    assert_true(np.all(s[24:] == 3))
    teardown_tmpdata()
    return
