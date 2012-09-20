"""
Test the datasets module
"""
# Author: Alexandre Abraham
# License: simplified BSD

import os
import shutil
from tempfile import mkdtemp

from nose.tools import assert_true

from .. import datasets

currdir = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(currdir, 'data')

def test_fetch_haxby():
    testdir = mkdtemp()
    local_url = "file://" + os.path.join(datadir, "pymvpa-exampledata.tar.bz2")
    haxby = datasets.fetch_haxby(data_dir=testdir, url=local_url)
    datasetdir = os.path.join(testdir, 'pymvpa-exampledata')
    for key, file in [
            ('session_target', 'attributes.txt'),
            ('func', 'bold.nii.gz'),
            ('mask', 'mask.nii.gz'),
            ('conditions_target', 'attributes_literal.txt')]:
        assert_true(haxby[key] == os.path.join(datasetdir, file))
        assert_true(os.exists(os.path.join(datasetdir, file)))
    shutil.rmtree(testdir)
