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
from nose.tools import assert_true, assert_false, assert_equal, assert_raises

from .. import datasets
from .._utils.testing import mock_urllib2, wrap_chunk_read_,\
    mock_fetch_files

currdir = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(currdir, 'data')
tmpdir = None
mock = None


def setup_tmpdata():
    # create temporary dir
    global tmpdir
    tmpdir = mkdtemp()


def setup_tmpdata_and_mock():
    setup_tmpdata()
    global mock
    mock = mock_urllib2()
    datasets.urllib2 = mock
    datasets._chunk_read_ = wrap_chunk_read_(datasets._chunk_read_)
    datasets._fetch_files = mock_fetch_files


def teardown_tmpdata():
    # remove temporary dir
    global tmpdir
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


def test_tree():
    # Create a dummy directory tree
    parent = mkdtemp()

    open(os.path.join(parent, 'file1'), 'w').close()
    open(os.path.join(parent, 'file2'), 'w').close()
    dir1 = os.path.join(parent, 'dir1')
    dir11 = os.path.join(dir1, 'dir11')
    dir12 = os.path.join(dir1, 'dir12')
    dir2 = os.path.join(parent, 'dir2')
    os.mkdir(dir1)
    os.mkdir(dir11)
    os.mkdir(dir12)
    os.mkdir(dir2)
    open(os.path.join(dir1, 'file11'), 'w').close()
    open(os.path.join(dir1, 'file12'), 'w').close()
    open(os.path.join(dir11, 'file111'), 'w').close()
    open(os.path.join(dir2, 'file21'), 'w').close()

    tree_ = datasets._tree(parent)

    # Check the tree
    #assert_equal(tree_[0]['dir1'][0]['dir11'][0], 'file111')
    #assert_equal(len(tree_[0]['dir1'][1]['dir12']), 0)
    #assert_equal(tree_[0]['dir1'][2], 'file11')
    #assert_equal(tree_[0]['dir1'][3], 'file12')
    #assert_equal(tree_[1]['dir2'][0], 'file21')
    #assert_equal(tree_[2], 'file1')
    #assert_equal(tree_[3], 'file2')
    assert_equal(tree_[0][1][0][1][0], os.path.join(dir11, 'file111'))
    assert_equal(len(tree_[0][1][1][1]), 0)
    assert_equal(tree_[0][1][2], os.path.join(dir1, 'file11'))
    assert_equal(tree_[0][1][3], os.path.join(dir1, 'file12'))
    assert_equal(tree_[1][1][0], os.path.join(dir2, 'file21'))
    assert_equal(tree_[2], os.path.join(parent, 'file1'))
    assert_equal(tree_[3], os.path.join(parent, 'file2'))

    # Clean
    shutil.rmtree(parent)


def test_movetree():
    # Create a dummy directory tree
    parent = mkdtemp()

    dir1 = os.path.join(parent, 'dir1')
    dir11 = os.path.join(dir1, 'dir11')
    dir12 = os.path.join(dir1, 'dir12')
    dir2 = os.path.join(parent, 'dir2')
    os.mkdir(dir1)
    os.mkdir(dir11)
    os.mkdir(dir12)
    os.mkdir(dir2)
    os.mkdir(os.path.join(dir2, 'dir12'))
    open(os.path.join(dir1, 'file11'), 'w').close()
    open(os.path.join(dir1, 'file12'), 'w').close()
    open(os.path.join(dir11, 'file111'), 'w').close()
    open(os.path.join(dir12, 'file121'), 'w').close()
    open(os.path.join(dir2, 'file21'), 'w').close()

    datasets.movetree(dir1, dir2)

    assert_false(os.path.exists(dir11))
    assert_false(os.path.exists(dir12))
    assert_false(os.path.exists(os.path.join(dir1, 'file11')))
    assert_false(os.path.exists(os.path.join(dir1, 'file12')))
    assert_false(os.path.exists(os.path.join(dir11, 'file111')))
    assert_false(os.path.exists(os.path.join(dir12, 'file121')))
    dir11 = os.path.join(dir2, 'dir11')
    dir12 = os.path.join(dir2, 'dir12')

    assert_true(os.path.exists(dir11))
    assert_true(os.path.exists(dir12))
    assert_true(os.path.exists(os.path.join(dir2, 'file11')))
    assert_true(os.path.exists(os.path.join(dir2, 'file12')))
    assert_true(os.path.exists(os.path.join(dir11, 'file111')))
    assert_true(os.path.exists(os.path.join(dir12, 'file121')))


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


@with_setup(setup_tmpdata, teardown_tmpdata)
def test_fail_fetch_haxby_simple():
    # Test a dataset fetching failure to validate sandboxing
    local_url = "file://" + os.path.join(datadir, "pymvpa-exampledata.tar.bz2")
    datasetdir = os.path.join(tmpdir, 'haxby2001_simple', 'pymvpa-exampledata')
    os.makedirs(datasetdir)
    # Create a dummy file. If sandboxing is successful, it won't be overwritten
    dummy = open(os.path.join(datasetdir, 'attributes.txt'), 'w')
    dummy.write('stuff')
    dummy.close()

    path = 'pymvpa-exampledata'

    opts = {'uncompress': True}
    files = [
            (os.path.join(path, 'attributes.txt'), local_url, opts),
            # The following file does not exists. It will cause an abortion of
            # the fetching procedure
            (os.path.join(path, 'bald.nii.gz'), local_url, opts)
    ]

    assert_raises(IOError, datasets._fetch_files, 'haxby2001_simple', files,
            data_dir=tmpdir)
    dummy = open(os.path.join(datasetdir, 'attributes.txt'), 'r')
    stuff = dummy.read(5)
    dummy.close()
    assert_equal(stuff, 'stuff')


# Smoke tests for the rest of the fetchers


@with_setup(setup_tmpdata_and_mock, teardown_tmpdata)
def test_fetch_craddock_2011_atlas():
    bunch = datasets.fetch_craddock_2011_atlas(data_dir=tmpdir)

    keys = ("scorr_mean", "tcorr_mean",
            "scorr_2level", "tcorr_2level",
            "random")
    filenames = [
            "scorr05_mean_all.nii.gz",
            "tcorr05_mean_all.nii.gz",
            "scorr05_2level_all.nii.gz",
            "tcorr05_2level_all.nii.gz",
            "random_all.nii.gz",
    ]
    assert_equal(len(mock.urls), 1)
    for key, fn in zip(keys, filenames):
        assert_equal(bunch[key], os.path.join(tmpdir, 'craddock_2011', fn))


@with_setup(setup_tmpdata_and_mock, teardown_tmpdata)
def test_fetch_smith_2009_atlas():
    bunch = datasets.fetch_smith_2009(data_dir=tmpdir)

    keys = ("rsn20", "rsn10", "rsn70",
            "bm20", "bm10", "bm70")
    filenames = [
            "rsn20.nii.gz",
            "PNAS_Smith09_rsn10.nii.gz",
            "rsn70.nii.gz",
            "bm20.nii.gz",
            "PNAS_Smith09_bm10.nii.gz",
            "bm70.nii.gz",
    ]

    assert_equal(len(mock.urls), 6)
    for key, fn in zip(keys, filenames):
        assert_equal(bunch[key], os.path.join(tmpdir, 'smith_2009', fn))


@with_setup(setup_tmpdata_and_mock, teardown_tmpdata)
def test_fetch_haxby():
    for i in range(1, 6):
        haxby = datasets.fetch_haxby(data_dir=tmpdir, n_subjects=i)
        assert_equal(len(mock.urls), 1 + (i == 1))  # subject_data + md5 file
        assert_equal(len(haxby.func), i)
        assert_equal(len(haxby.anat), i)
        assert_equal(len(haxby.session_target), i)
        assert_equal(len(haxby.mask_vt), i)
        assert_equal(len(haxby.mask_face), i)
        assert_equal(len(haxby.mask_house), i)
        assert_equal(len(haxby.mask_face_little), i)
        assert_equal(len(haxby.mask_house_little), i)
        mock.reset()


@with_setup(setup_tmpdata_and_mock, teardown_tmpdata)
def test_fetch_nyu_rest():
    # First session, all subjects
    nyu = datasets.fetch_nyu_rest(data_dir=tmpdir)
    assert_equal(len(mock.urls), 2)
    assert_equal(len(nyu.func), 25)
    assert_equal(len(nyu.anat_anon), 25)
    assert_equal(len(nyu.anat_skull), 25)
    assert_true(np.all(np.asarray(nyu.session) == 1))

    # All sessions, 12 subjects
    mock.reset()
    nyu = datasets.fetch_nyu_rest(data_dir=tmpdir, sessions=[1, 2, 3],
                                  n_subjects=12)
    # Session 1 has already been downloaded
    assert_equal(len(mock.urls), 2)
    assert_equal(len(nyu.func), 36)
    assert_equal(len(nyu.anat_anon), 36)
    assert_equal(len(nyu.anat_skull), 36)
    s = np.asarray(nyu.session)
    assert_true(np.all(s[:12] == 1))
    assert_true(np.all(s[12:24] == 2))
    assert_true(np.all(s[24:] == 3))


@with_setup(setup_tmpdata_and_mock, teardown_tmpdata)
def test_fetch_adhd():
    local_url = "file://" + datadir
    # Disabled: cannot be tested without actually fetching the phenotypic file
    adhd = datasets.fetch_adhd(data_dir=tmpdir, url=local_url, n_subjects=12)
    assert_equal(len(adhd.func), 12)
    assert_equal(len(adhd.confounds), 12)
    assert_equal(len(mock.urls), 2)


@with_setup(setup_tmpdata_and_mock, teardown_tmpdata)
def test_miyawaki2008():
    dataset = datasets.fetch_miyawaki2008(data_dir=tmpdir)
    assert_equal(len(dataset.func), 32)
    assert_equal(len(dataset.label), 32)
    assert_true(isinstance(dataset.mask, basestring))
    assert_equal(len(dataset.mask_roi), 38)
    assert_equal(len(mock.urls), 1)


@with_setup(setup_tmpdata_and_mock, teardown_tmpdata)
def test_fetch_msdl_atlas():
    dataset = datasets.fetch_msdl_atlas(data_dir=tmpdir)
    assert_true(isinstance(dataset.labels, basestring))
    assert_true(isinstance(dataset.maps, basestring))
    assert_equal(len(mock.urls), 1)


@with_setup(setup_tmpdata_and_mock, teardown_tmpdata)
def test_fetch_icbm152_2009():
    dataset = datasets.fetch_icbm152_2009(data_dir=tmpdir)
    assert_true(isinstance(dataset.csf, basestring))
    assert_true(isinstance(dataset.eye_mask, basestring))
    assert_true(isinstance(dataset.face_mask, basestring))
    assert_true(isinstance(dataset.gm, basestring))
    assert_true(isinstance(dataset.mask, basestring))
    assert_true(isinstance(dataset.pd, basestring))
    assert_true(isinstance(dataset.t1, basestring))
    assert_true(isinstance(dataset.t2, basestring))
    assert_true(isinstance(dataset.t2_relax, basestring))
    assert_true(isinstance(dataset.wm, basestring))
    assert_equal(len(mock.urls), 1)


@with_setup(setup_tmpdata_and_mock, teardown_tmpdata)
def test_fetch_yeo_2011_atlas():
    dataset = datasets.fetch_yeo_2011_atlas(data_dir=tmpdir)
    assert_true(isinstance(dataset.anat, basestring))
    assert_true(isinstance(dataset.colors_17, basestring))
    assert_true(isinstance(dataset.colors_7, basestring))
    assert_true(isinstance(dataset.liberal_17, basestring))
    assert_true(isinstance(dataset.liberal_7, basestring))
    assert_true(isinstance(dataset.tight_17, basestring))
    assert_true(isinstance(dataset.tight_7, basestring))
    assert_equal(len(mock.urls), 1)


@with_setup(setup_tmpdata_and_mock, teardown_tmpdata)
def test_fetch_localizer_contrasts():
    dataset = datasets.fetch_localizer_contrasts(["checkerboard"],
                                                 data_dir=tmpdir)
    assert_true(dataset.anats is None)
    assert_true(dataset.tmaps is None)
    assert_true(dataset.masks is None)
    assert_true(isinstance(dataset.cmaps[0], basestring))


@with_setup(setup_tmpdata_and_mock, teardown_tmpdata)
def test_fetch_localizer_calculation_task():
    dataset = datasets.fetch_localizer_calculation_task(data_dir=tmpdir)
    assert_true(dataset.anats is None)
    assert_true(dataset.tmaps is None)
    assert_true(dataset.masks is None)
    assert_true(isinstance(dataset.cmaps[0], basestring))
