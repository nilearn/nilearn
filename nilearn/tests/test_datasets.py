"""
Test the datasets module
"""
# Author: Alexandre Abraham
# License: simplified BSD

import contextlib
import os
import shutil
from tempfile import mkdtemp, mkstemp
import numpy as np
import zipfile
import tarfile
import gzip

from nose import with_setup
from nose.tools import assert_true, assert_false, assert_equal, assert_raises

from .. import datasets
from .._utils.testing import mock_urllib2, wrap_chunk_read_,\
    FetchFilesMock, assert_raises_regexp


currdir = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(currdir, 'data')
tmpdir = None
url_mock = None
file_mock = None


def setup_tmpdata():
    # create temporary dir
    global tmpdir
    tmpdir = mkdtemp()


def setup_mock():
    global url_mock
    url_mock = mock_urllib2()
    datasets.urllib2 = url_mock
    datasets._chunk_read_ = wrap_chunk_read_(datasets._chunk_read_)
    global file_mock
    file_mock = FetchFilesMock()
    datasets._fetch_files = file_mock


def teardown_tmpdata():
    # remove temporary dir
    global tmpdir
    if tmpdir is not None:
        shutil.rmtree(tmpdir)


def test_md5_sum_file():
    # Create dummy temporary file
    out, f = mkstemp()
    os.write(out, 'abcfeg')
    os.close(out)
    assert_equal(datasets._md5_sum_file(f), '18f32295c556b2a1a3a8e68fe1ad40f7')
    os.remove(f)

@with_setup(setup_tmpdata, teardown_tmpdata)
def test_get_dataset_dir():

    os.chdir(tmpdir)
    #testing folder creation under different environments, enforcing a custom
    #clean install
    os.environ.pop('NILEARN_DATA', None)
    os.environ.pop('NILEARN_SHARED_DATA', None)

    data_dir = datasets._get_dataset_dir('test')
    assert_equal(data_dir, os.path.abspath(os.path.join('nilearn_data', 'test')))
    assert os.path.exists(data_dir)
    shutil.rmtree(data_dir)

    os.environ['NILEARN_DATA'] = 'test_nilearn_data'
    data_dir = datasets._get_dataset_dir('test')
    assert_equal(data_dir, os.path.join('test_nilearn_data', 'test'))
    assert os.path.exists(data_dir)
    shutil.rmtree(data_dir)

    os.environ['NILEARN_SHARED_DATA'] = 'nilearn_shared_data'
    data_dir = datasets._get_dataset_dir('test')
    assert_equal(data_dir, os.path.join('nilearn_shared_data', 'test'))
    assert os.path.exists(data_dir)
    shutil.rmtree(data_dir)

    os.chdir(currdir)

    #Verify exception is raised on read-only directories
    no_write = mkdtemp()
    os.chmod(no_write, 0400)
    assert_raises_regexp(OSError, 'Permission denied',
                         datasets._get_dataset_dir, 'test', no_write)
    #Verify exception for not paths as files
    test_file = mktemp()
    out = open(test_file, 'w')
    out.write('abcfeg')
    out.close()
    assert_raises_regexp(OSError, 'Not a directory',
                         datasets._get_dataset_dir, 'test', test_file)

@with_setup(setup_tmpdata, teardown_tmpdata)
def test_get_dataset_dir():
    # testing folder creation under different environments, enforcing
    # a custom clean install
    os.environ.pop('NILEARN_DATA', None)
    os.environ.pop('NILEARN_SHARED_DATA', None)

    expected_base_dir = os.path.expanduser('~/nilearn_data')
    data_dir = datasets._get_dataset_dir('test', verbose=0)
    assert_equal(data_dir, os.path.join(expected_base_dir, 'test'))
    assert os.path.exists(data_dir)
    shutil.rmtree(data_dir)

    expected_base_dir = os.path.join(tmpdir, 'test_nilearn_data')
    os.environ['NILEARN_DATA'] = expected_base_dir
    data_dir = datasets._get_dataset_dir('test', verbose=0)
    assert_equal(data_dir, os.path.join(expected_base_dir, 'test'))
    assert os.path.exists(data_dir)
    shutil.rmtree(data_dir)

    expected_base_dir = os.path.join(tmpdir, 'nilearn_shared_data')
    os.environ['NILEARN_SHARED_DATA'] = expected_base_dir
    data_dir = datasets._get_dataset_dir('test', verbose=0)
    assert_equal(data_dir, os.path.join(expected_base_dir, 'test'))
    assert os.path.exists(data_dir)
    shutil.rmtree(data_dir)

    # Verify exception is raised on read-only directories
    no_write = os.path.join(tmpdir, 'no_write')
    os.makedirs(no_write)
    os.chmod(no_write, 0400)
    assert_raises_regexp(OSError, 'Permission denied',
                         datasets._get_dataset_dir, 'test', no_write,
                         verbose=0)

    # Verify exception for a path which exists and is a file
    test_file = os.path.join(tmpdir, 'some_file')
    with open(test_file, 'w') as out:
        out.write('abcfeg')
    assert_raises_regexp(OSError, 'Not a directory',
                         datasets._get_dataset_dir, 'test', test_file,
                         verbose=0)


def test_read_md5_sum_file():
    # Create dummy temporary file
    out, f = mkstemp()
    os.write(out, '20861c8c3fe177da19a7e9539a5dbac  /tmp/test\n'
              '70886dcabe7bf5c5a1c24ca24e4cbd94  test/some_image.nii')
    os.close(out)
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
    haxby = datasets.fetch_haxby_simple(data_dir=tmpdir, url=local_url,
                                        verbose=0)
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

    assert_raises(IOError, datasets._fetch_files,
            os.path.join(tmpdir, 'haxby2001_simple'), files,
            verbose=0)
    dummy = open(os.path.join(datasetdir, 'attributes.txt'), 'r')
    stuff = dummy.read(5)
    dummy.close()
    assert_equal(stuff, 'stuff')


# Smoke tests for the rest of the fetchers


@with_setup(setup_mock)
@with_setup(setup_tmpdata, teardown_tmpdata)
def test_fetch_craddock_2011_atlas():
    bunch = datasets.fetch_craddock_2011_atlas(data_dir=tmpdir, verbose=0)

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
    assert_equal(len(url_mock.urls), 1)
    for key, fn in zip(keys, filenames):
        assert_equal(bunch[key], os.path.join(tmpdir, 'craddock_2011', fn))


@with_setup(setup_mock)
@with_setup(setup_tmpdata, teardown_tmpdata)
def test_fetch_smith_2009_atlas():
    bunch = datasets.fetch_smith_2009(data_dir=tmpdir, verbose=0)

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

    assert_equal(len(url_mock.urls), 6)
    for key, fn in zip(keys, filenames):
        assert_equal(bunch[key], os.path.join(tmpdir, 'smith_2009', fn))


@with_setup(setup_mock)
@with_setup(setup_tmpdata, teardown_tmpdata)
def test_fetch_haxby():
    for i in range(1, 6):
        haxby = datasets.fetch_haxby(data_dir=tmpdir, n_subjects=i,
                                     verbose=0)
        assert_equal(len(url_mock.urls), 1 + (i == 1))  # subject_data + md5
        assert_equal(len(haxby.func), i)
        assert_equal(len(haxby.anat), i)
        assert_equal(len(haxby.session_target), i)
        assert_equal(len(haxby.mask_vt), i)
        assert_equal(len(haxby.mask_face), i)
        assert_equal(len(haxby.mask_house), i)
        assert_equal(len(haxby.mask_face_little), i)
        assert_equal(len(haxby.mask_house_little), i)
        url_mock.reset()


@with_setup(setup_mock)
@with_setup(setup_tmpdata, teardown_tmpdata)
def test_fetch_nyu_rest():
    # First session, all subjects
    nyu = datasets.fetch_nyu_rest(data_dir=tmpdir, verbose=0)
    assert_equal(len(url_mock.urls), 2)
    assert_equal(len(nyu.func), 25)
    assert_equal(len(nyu.anat_anon), 25)
    assert_equal(len(nyu.anat_skull), 25)
    assert_true(np.all(np.asarray(nyu.session) == 1))

    # All sessions, 12 subjects
    url_mock.reset()
    nyu = datasets.fetch_nyu_rest(data_dir=tmpdir, sessions=[1, 2, 3],
                                  n_subjects=12, verbose=0)
    # Session 1 has already been downloaded
    assert_equal(len(url_mock.urls), 2)
    assert_equal(len(nyu.func), 36)
    assert_equal(len(nyu.anat_anon), 36)
    assert_equal(len(nyu.anat_skull), 36)
    s = np.asarray(nyu.session)
    assert_true(np.all(s[:12] == 1))
    assert_true(np.all(s[12:24] == 2))
    assert_true(np.all(s[24:] == 3))


@with_setup(setup_mock)
@with_setup(setup_tmpdata, teardown_tmpdata)
def test_fetch_adhd():
    local_url = "file://" + datadir

    sub1 = ['3902469', '7774305', '3699991']
    sub2 = ['2014113', '4275075', '1019436', '3154996', '3884955', '0027034',
            '4134561', '0027018', '6115230', '0027037', '8409791', '0027011']
    sub3 = ['3007585', '8697774', '9750701', '0010064', '0021019', '0010042',
            '0010128', '2497695', '4164316', '1552181', '4046678', '0023012']
    sub4 = ['1679142', '1206380', '0023008', '4016887', '1418396', '2950754',
            '3994098', '3520880', '1517058', '9744150', '1562298', '3205761',
            '3624598']
    subs = np.asarray(sub1 + sub2 + sub3 + sub4)
    subs = subs.view(dtype=[('Subject', 'S7')])
    file_mock.add_csv('ADHD200_40subs_motion_parameters_and_phenotypics.csv',
            subs)

    adhd = datasets.fetch_adhd(data_dir=tmpdir, url=local_url,
                               n_subjects=12, verbose=0)
    assert_equal(len(adhd.func), 12)
    assert_equal(len(adhd.confounds), 12)
    assert_equal(len(url_mock.urls), 2)


@with_setup(setup_mock)
@with_setup(setup_tmpdata, teardown_tmpdata)
def test_miyawaki2008():
    dataset = datasets.fetch_miyawaki2008(data_dir=tmpdir, verbose=0)
    assert_equal(len(dataset.func), 32)
    assert_equal(len(dataset.label), 32)
    assert_true(isinstance(dataset.mask, basestring))
    assert_equal(len(dataset.mask_roi), 38)
    assert_equal(len(url_mock.urls), 1)


@with_setup(setup_mock)
@with_setup(setup_tmpdata, teardown_tmpdata)
def test_fetch_msdl_atlas():
    dataset = datasets.fetch_msdl_atlas(data_dir=tmpdir, verbose=0)
    assert_true(isinstance(dataset.labels, basestring))
    assert_true(isinstance(dataset.maps, basestring))
    assert_equal(len(url_mock.urls), 1)


@with_setup(setup_mock)
@with_setup(setup_tmpdata, teardown_tmpdata)
def test_fetch_icbm152_2009():
    dataset = datasets.fetch_icbm152_2009(data_dir=tmpdir, verbose=0)
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
    assert_equal(len(url_mock.urls), 1)


@with_setup(setup_mock)
@with_setup(setup_tmpdata, teardown_tmpdata)
def test_fetch_yeo_2011_atlas():
    dataset = datasets.fetch_yeo_2011_atlas(data_dir=tmpdir, verbose=0)
    assert_true(isinstance(dataset.anat, basestring))
    assert_true(isinstance(dataset.colors_17, basestring))
    assert_true(isinstance(dataset.colors_7, basestring))
    assert_true(isinstance(dataset.thick_17, basestring))
    assert_true(isinstance(dataset.thick_7, basestring))
    assert_true(isinstance(dataset.thin_17, basestring))
    assert_true(isinstance(dataset.thin_7, basestring))
    assert_equal(len(url_mock.urls), 1)


@with_setup(setup_mock)
@with_setup(setup_tmpdata, teardown_tmpdata)
def test_fetch_localizer_contrasts():
    local_url = "file://" + datadir
    ids = np.asarray(['S%2d' % i for i in range(94)])
    ids = ids.view(dtype=[('subject_id', 'S3')])
    file_mock.add_csv('cubicwebexport.csv', ids)
    file_mock.add_csv('cubicwebexport2.csv', ids)

    # Disabled: cannot be tested without actually fetching covariates CSV file
    # All subjects
    dataset = datasets.fetch_localizer_contrasts(["checkerboard"],
                                                 data_dir=tmpdir,
                                                 url=local_url,
                                                 verbose=0)
    assert_true(dataset.anats is None)
    assert_true(dataset.tmaps is None)
    assert_true(dataset.masks is None)
    assert_true(isinstance(dataset.ext_vars, np.recarray))
    assert_true(isinstance(dataset.cmaps[0], basestring))
    assert_equal(dataset.ext_vars.size, 94)
    assert_equal(len(dataset.cmaps), 94)

    # 20 subjects
    dataset = datasets.fetch_localizer_contrasts(["checkerboard"],
                                                 n_subjects=20,
                                                 data_dir=tmpdir,
                                                 url=local_url,
                                                 verbose=0)
    assert_true(dataset.anats is None)
    assert_true(dataset.tmaps is None)
    assert_true(dataset.masks is None)
    assert_true(isinstance(dataset.cmaps[0], basestring))
    assert_true(isinstance(dataset.ext_vars, np.recarray))
    assert_equal(len(dataset.cmaps), 20)
    assert_equal(dataset.ext_vars.size, 20)

    # Multiple contrasts
    dataset = datasets.fetch_localizer_contrasts(
        ["checkerboard", "horizontal checkerboard"],
        n_subjects=20, data_dir=tmpdir,
        verbose=0)
    assert_true(dataset.anats is None)
    assert_true(dataset.tmaps is None)
    assert_true(dataset.masks is None)
    assert_true(isinstance(dataset.ext_vars, np.recarray))
    assert_true(isinstance(dataset.cmaps[0], basestring))
    assert_equal(len(dataset.cmaps), 20 * 2)  # two contrasts are fetched
    assert_equal(dataset.ext_vars.size, 20)

    # get_anats=True
    dataset = datasets.fetch_localizer_contrasts(["checkerboard"],
                                                 data_dir=tmpdir,
                                                 url=local_url,
                                                 get_anats=True,
                                                 verbose=0)
    assert_true(dataset.masks is None)
    assert_true(dataset.tmaps is None)
    assert_true(isinstance(dataset.ext_vars, np.recarray))
    assert_true(isinstance(dataset.anats[0], basestring))
    assert_true(isinstance(dataset.cmaps[0], basestring))
    assert_equal(dataset.ext_vars.size, 94)
    assert_equal(len(dataset.anats), 94)
    assert_equal(len(dataset.cmaps), 94)

    # get_masks=True
    dataset = datasets.fetch_localizer_contrasts(["checkerboard"],
                                                 data_dir=tmpdir,
                                                 url=local_url,
                                                 get_masks=True,
                                                 verbose=0)
    assert_true(dataset.anats is None)
    assert_true(dataset.tmaps is None)
    assert_true(isinstance(dataset.ext_vars, np.recarray))
    assert_true(isinstance(dataset.cmaps[0], basestring))
    assert_true(isinstance(dataset.masks[0], basestring))
    assert_equal(dataset.ext_vars.size, 94)
    assert_equal(len(dataset.cmaps), 94)
    assert_equal(len(dataset.masks), 94)

    # get_tmaps=True
    dataset = datasets.fetch_localizer_contrasts(["checkerboard"],
                                                 data_dir=tmpdir,
                                                 url=local_url,
                                                 get_tmaps=True,
                                                 verbose=0)
    assert_true(dataset.anats is None)
    assert_true(dataset.masks is None)
    assert_true(isinstance(dataset.ext_vars, np.recarray))
    assert_true(isinstance(dataset.cmaps[0], basestring))
    assert_true(isinstance(dataset.tmaps[0], basestring))
    assert_equal(dataset.ext_vars.size, 94)
    assert_equal(len(dataset.cmaps), 94)
    assert_equal(len(dataset.tmaps), 94)

    # all get_*=True
    dataset = datasets.fetch_localizer_contrasts(["checkerboard"],
                                                 data_dir=tmpdir,
                                                 url=local_url,
                                                 get_anats=True,
                                                 get_masks=True,
                                                 get_tmaps=True,
                                                 verbose=0)

    assert_true(isinstance(dataset.ext_vars, np.recarray))
    assert_true(isinstance(dataset.anats[0], basestring))
    assert_true(isinstance(dataset.cmaps[0], basestring))
    assert_true(isinstance(dataset.masks[0], basestring))
    assert_true(isinstance(dataset.tmaps[0], basestring))
    assert_equal(dataset.ext_vars.size, 94)
    assert_equal(len(dataset.anats), 94)
    assert_equal(len(dataset.cmaps), 94)
    assert_equal(len(dataset.masks), 94)
    assert_equal(len(dataset.tmaps), 94)


@with_setup(setup_mock)
@with_setup(setup_tmpdata, teardown_tmpdata)
def test_fetch_localizer_calculation_task():
    local_url = "file://" + datadir
    ids = np.asarray(['S%2d' % i for i in range(94)])
    ids = ids.view(dtype=[('subject_id', 'S3')])
    file_mock.add_csv('cubicwebexport.csv', ids)
    file_mock.add_csv('cubicwebexport2.csv', ids)

    # Disabled: cannot be tested without actually fetching covariates CSV file
    # All subjects
    dataset = datasets.fetch_localizer_calculation_task(data_dir=tmpdir,
                                                        url=local_url,
                                                        verbose=0)
    assert_true(isinstance(dataset.ext_vars, np.recarray))
    assert_true(isinstance(dataset.cmaps[0], basestring))
    assert_equal(dataset.ext_vars.size, 94)
    assert_equal(len(dataset.cmaps), 94)

    # 20 subjects
    dataset = datasets.fetch_localizer_calculation_task(n_subjects=20,
                                                        data_dir=tmpdir,
                                                        url=local_url,
                                                        verbose=0)
    assert_true(isinstance(dataset.ext_vars, np.recarray))
    assert_true(isinstance(dataset.cmaps[0], basestring))
    assert_equal(dataset.ext_vars.size, 20)
    assert_equal(len(dataset.cmaps), 20)


@with_setup(setup_mock)
@with_setup(setup_tmpdata, teardown_tmpdata)
def test_fetch_oasis_vbm():
    local_url = "file://" + datadir
    ids = np.asarray(['OAS1_%4d' % i for i in range(457)])
    ids = ids.view(dtype=[('ID', 'S9')])
    file_mock.add_csv('oasis_cross-sectional.csv', ids)

    # Disabled: cannot be tested without actually fetching covariates CSV file
    dataset = datasets.fetch_oasis_vbm(data_dir=tmpdir, url=local_url,
                                       verbose=0)
    assert_equal(len(dataset.gray_matter_maps), 403)
    assert_equal(len(dataset.white_matter_maps), 403)
    assert_true(isinstance(dataset.gray_matter_maps[0], basestring))
    assert_true(isinstance(dataset.white_matter_maps[0], basestring))
    assert_true(isinstance(dataset.ext_vars, np.recarray))
    assert_true(isinstance(dataset.data_usage_agreement, basestring))
    assert_equal(len(url_mock.urls), 3)

    dataset = datasets.fetch_oasis_vbm(data_dir=tmpdir, url=local_url,
                                       dartel_version=False, verbose=0)
    assert_equal(len(dataset.gray_matter_maps), 415)
    assert_equal(len(dataset.white_matter_maps), 415)
    assert_true(isinstance(dataset.gray_matter_maps[0], basestring))
    assert_true(isinstance(dataset.white_matter_maps[0], basestring))
    assert_true(isinstance(dataset.ext_vars, np.recarray))
    assert_true(isinstance(dataset.data_usage_agreement, basestring))
    assert_equal(len(url_mock.urls), 4)


def test_load_mni152_template():
    # All subjects
    template_nii = datasets.load_mni152_template()
    assert_equal(template_nii.shape, (91, 109, 91))
    assert_equal(template_nii.get_header().get_zooms(), (2.0, 2.0, 2.0))


@with_setup(setup_mock)
@with_setup(setup_tmpdata, teardown_tmpdata)
def test_fetch_abide_pcp():
    local_url = "file://" + datadir
    ids = ['50%03d' % i for i in range(800)]
    filenames = ['no_filename'] * 800
    filenames[::2] = ['filename'] * 400
    pheno = np.asarray(zip(ids, filenames), dtype=[('subject_id', int),
                                                   ('FILE_ID', 'S11')])
    #pheno = pheno.T.view()
    file_mock.add_csv('Phenotypic_V1_0b_preprocessed1.csv', pheno)

    # All subjects
    dataset = datasets.fetch_abide_pcp(data_dir=tmpdir, url=local_url,
                                       quality_checked=False, verbose=0)
    assert_equal(len(dataset.func_preproc), 400)


def test_filter_columns():
    # Create fake recarray
    value1 = np.arange(500)
    strings = np.asarray(['a', 'b', 'c'])
    value2 = strings[value1 % 3]

    values = np.asarray(zip(value1, value2),
                        dtype=[('INT', int), ('STR', 'S1')])

    f = datasets._filter_columns(values, {'INT': (23, 46)})
    assert_equal(np.sum(f), 24)

    f = datasets._filter_columns(values, {'INT': [0, 9, (12, 24)]})
    assert_equal(np.sum(f), 15)

    value1 = value1 % 2
    values = np.asarray(zip(value1, value2),
                        dtype=[('INT', int), ('STR', 'S1')])

    # No filter
    f = datasets._filter_columns(values, [])
    assert_equal(np.sum(f), 500)

    f = datasets._filter_columns(values, {'STR': 'b'})
    assert_equal(np.sum(f), 167)

    f = datasets._filter_columns(values, {'INT': 1, 'STR': 'b'})
    assert_equal(np.sum(f), 84)

    f = datasets._filter_columns(values, {'INT': 1, 'STR': 'b'},
            combination='or')
    assert_equal(np.sum(f), 333)


def test_uncompress():
    # Create dummy file
    fd, temp = mkstemp()
    os.close(fd)
    # Create a zipfile
    dtemp = mkdtemp()
    ztemp = os.path.join(dtemp, 'test.zip')
    with contextlib.closing(zipfile.ZipFile(ztemp, 'w')) as testzip:
        testzip.write(temp)
    datasets._uncompress_file(ztemp, verbose=0)
    assert(os.path.exists(os.path.join(dtemp, temp)))
    shutil.rmtree(dtemp)

    dtemp = mkdtemp()
    ztemp = os.path.join(dtemp, 'test.tar')
    with contextlib.closing(tarfile.open(ztemp, 'w')) as tar:
        tar.add(temp)
    datasets._uncompress_file(ztemp, verbose=0)
    assert(os.path.exists(os.path.join(dtemp, temp)))
    shutil.rmtree(dtemp)

    dtemp = mkdtemp()
    ztemp = os.path.join(dtemp, 'test.gz')
    f = gzip.open(ztemp, 'wb')
    f.close()
    datasets._uncompress_file(ztemp, verbose=0)
    assert(os.path.exists(os.path.join(dtemp, temp)))
    shutil.rmtree(dtemp)

    os.remove(temp)
