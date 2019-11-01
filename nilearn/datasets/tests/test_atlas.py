"""
Test the datasets module
"""
# Author: Alexandre Abraham
# License: simplified BSD

import os
import shutil
import itertools

import numpy as np

import nibabel

from distutils.version import LooseVersion

from nose import with_setup, SkipTest
from nose.tools import (assert_true, assert_equal, assert_raises,
                        assert_not_equal)
from numpy.testing import assert_array_equal

from nilearn._utils.testing import assert_raises_regex
from . import test_utils as tst

from nilearn._utils.compat import _basestring, _urllib

from nilearn.datasets import utils, atlas
from nilearn.image import get_data


def setup_mock():
    return tst.setup_mock(utils, atlas)


def teardown_mock():
    return tst.teardown_mock(utils, atlas)


@with_setup(tst.setup_tmpdata, tst.teardown_tmpdata)
def test_get_dataset_dir():
    # testing folder creation under different environments, enforcing
    # a custom clean install
    os.environ.pop('NILEARN_DATA', None)
    os.environ.pop('NILEARN_SHARED_DATA', None)

    expected_base_dir = os.path.expanduser('~/nilearn_data')
    data_dir = utils._get_dataset_dir('test', verbose=0)
    assert_equal(data_dir, os.path.join(expected_base_dir, 'test'))
    assert os.path.exists(data_dir)
    shutil.rmtree(data_dir)

    expected_base_dir = os.path.join(tst.tmpdir, 'test_nilearn_data')
    os.environ['NILEARN_DATA'] = expected_base_dir
    data_dir = utils._get_dataset_dir('test', verbose=0)
    assert_equal(data_dir, os.path.join(expected_base_dir, 'test'))
    assert os.path.exists(data_dir)
    shutil.rmtree(data_dir)

    expected_base_dir = os.path.join(tst.tmpdir, 'nilearn_shared_data')
    os.environ['NILEARN_SHARED_DATA'] = expected_base_dir
    data_dir = utils._get_dataset_dir('test', verbose=0)
    assert_equal(data_dir, os.path.join(expected_base_dir, 'test'))
    assert os.path.exists(data_dir)
    shutil.rmtree(data_dir)

    expected_base_dir = os.path.join(tst.tmpdir, 'env_data')
    expected_dataset_dir = os.path.join(expected_base_dir, 'test')
    data_dir = utils._get_dataset_dir(
        'test', default_paths=[expected_dataset_dir], verbose=0)
    assert_equal(data_dir, os.path.join(expected_base_dir, 'test'))
    assert os.path.exists(data_dir)
    shutil.rmtree(data_dir)

    no_write = os.path.join(tst.tmpdir, 'no_write')
    os.makedirs(no_write)
    os.chmod(no_write, 0o400)

    expected_base_dir = os.path.join(tst.tmpdir, 'nilearn_shared_data')
    os.environ['NILEARN_SHARED_DATA'] = expected_base_dir
    data_dir = utils._get_dataset_dir('test',
                                      default_paths=[no_write],
                                      verbose=0)
    # Non writeable dir is returned because dataset may be in there.
    assert_equal(data_dir, no_write)
    assert os.path.exists(data_dir)
    # Set back write permissions in order to be able to remove the file
    os.chmod(no_write, 0o600)
    shutil.rmtree(data_dir)

    # Verify exception for a path which exists and is a file
    test_file = os.path.join(tst.tmpdir, 'some_file')
    with open(test_file, 'w') as out:
        out.write('abcfeg')
    assert_raises_regex(OSError,
                        'Nilearn tried to store the dataset '
                        'in the following directories, but',
                        utils._get_dataset_dir,
                        'test', test_file, verbose=0)


@with_setup(tst.setup_tmpdata, tst.teardown_tmpdata)
def test_downloader():

    # Sandboxing test
    # ===============

    # When nilearn downloads a file, everything is first downloaded in a
    # temporary directory (sandbox) and moved to the "real" data directory if
    # all files are present. In case of error, the sandbox is deleted.

    # To test this feature, we do as follow:
    # - create the data dir with a file that has a specific content
    # - try to download the dataset but make it fail on purpose (by requesting a
    #   file that is not in the archive)
    # - check that the previously created file is untouched :
    #   - if sandboxing is faulty, the file would be replaced by the file of the
    #     archive
    #   - if sandboxing works, the file must be untouched.

    local_url = "file:" + _urllib.request.pathname2url(
        os.path.join(tst.datadir, "craddock_2011_parcellations.tar.gz"))
    datasetdir = os.path.join(tst.tmpdir, 'craddock_2012')
    os.makedirs(datasetdir)

    # Create a dummy file. If sandboxing is successful, it won't be overwritten
    dummy = open(os.path.join(datasetdir, 'random_all.nii.gz'), 'w')
    dummy.write('stuff')
    dummy.close()

    opts = {'uncompress': True}
    files = [
        ('random_all.nii.gz', local_url, opts),
        # The following file does not exists. It will cause an abortion of
        # the fetching procedure
        ('bald.nii.gz', local_url, opts)
    ]

    assert_raises(IOError, utils._fetch_files,
                  os.path.join(tst.tmpdir, 'craddock_2012'), files,
                  verbose=0)
    dummy = open(os.path.join(datasetdir, 'random_all.nii.gz'), 'r')
    stuff = dummy.read(5)
    dummy.close()
    assert_equal(stuff, 'stuff')

    # Downloading test
    # ================

    # Now, we use the regular downloading feature. This will override the dummy
    # file created before.

    atlas.fetch_atlas_craddock_2012(data_dir=tst.tmpdir, url=local_url)
    dummy = open(os.path.join(datasetdir, 'random_all.nii.gz'), 'r')
    stuff = dummy.read()
    dummy.close()
    assert_equal(stuff, '')


@with_setup(tst.setup_tmpdata, tst.teardown_tmpdata)
def test_fail_fetch_atlas_harvard_oxford():
    # specify non-existing atlas item
    assert_raises_regex(ValueError, 'Invalid atlas name',
                        atlas.fetch_atlas_harvard_oxford,
                        'not_inside')

    # specify existing atlas item
    target_atlas = 'cort-maxprob-thr0-1mm'
    target_atlas_fname = 'HarvardOxford-' + target_atlas + '.nii.gz'

    ho_dir = os.path.join(tst.tmpdir, 'fsl', 'data', 'atlases')
    os.makedirs(ho_dir)
    nifti_dir = os.path.join(ho_dir, 'HarvardOxford')
    os.makedirs(nifti_dir)

    target_atlas_nii = os.path.join(nifti_dir, target_atlas_fname)

    # Create false atlas
    atlas_data = np.zeros((10, 10, 10), dtype=int)

    # Create an interhemispheric map
    atlas_data[:, :2, :] = 1

    # Create a left map
    atlas_data[:5, 3:5, :] = 2

    # Create a right map, with one voxel on the left side
    atlas_data[5:, 7:9, :] = 3
    atlas_data[4, 7, 0] = 3

    nibabel.Nifti1Image(atlas_data, np.eye(4) * 3).to_filename(
        target_atlas_nii)

    dummy = open(os.path.join(ho_dir, 'HarvardOxford-Cortical.xml'), 'w')
    dummy.write("<?xml version='1.0' encoding='us-ascii'?>\n"
                "<data>\n"
                '<label index="0" x="48" y="94" z="35">R1</label>\n'
                '<label index="1" x="25" y="70" z="32">R2</label>\n'
                '<label index="2" x="33" y="73" z="63">R3</label>\n'
                "</data>")
    dummy.close()

    # when symmetric_split=False (by default), then atlas fetcher should
    # have maps as string and n_labels=4 with background. Since, we relay on xml
    # file to retrieve labels.
    ho_wo_symm = atlas.fetch_atlas_harvard_oxford(target_atlas,
                                                  data_dir=tst.tmpdir)
    assert_true(isinstance(ho_wo_symm.maps, _basestring))
    assert_true(isinstance(ho_wo_symm.labels, list))
    assert_equal(ho_wo_symm.labels[0], "Background")
    assert_equal(ho_wo_symm.labels[1], "R1")
    assert_equal(ho_wo_symm.labels[2], "R2")
    assert_equal(ho_wo_symm.labels[3], "R3")

    # This section tests with lateralized version. In other words,
    # symmetric_split=True

    # Dummy xml file for lateralized control of cortical atlas images
    # shipped with FSL 5.0. Atlases are already lateralized in this version
    # for cortical type atlases denoted with maxprob but not full prob and but
    # not also with subcortical.

    # So, we test the fetcher with symmetric_split=True by creating a new
    # dummy local file and fetch them and test the output variables
    # accordingly.
    dummy2 = open(os.path.join(ho_dir, 'HarvardOxford-Cortical-Lateralized.xml'), 'w')
    dummy2.write("<?xml version='1.0' encoding='us-ascii'?>\n"
                 "<data>\n"
                 '<label index="0" x="63" y="86" z="49">Left R1</label>\n'
                 '<label index="1" x="21" y="86" z="33">Right R1</label>\n'
                 '<label index="2" x="64" y="69" z="32">Left R2</label>\n'
                 '<label index="3" x="26" y="70" z="32">Right R2</label>\n'
                 '<label index="4" x="47" y="75" z="66">Left R3</label>\n'
                 '<label index="5" x="43" y="80" z="61">Right R3</label>\n'
                 "</data>")
    dummy2.close()

    # Here, with symmetric_split=True, atlas maps are returned as nibabel Nifti
    # image but not string. Now, with symmetric split number of labels should be
    # more than without split and contain Left and Right tags in the labels.

    # Create dummy image files too with cortl specified for symmetric split.
    split_atlas_fname = 'HarvardOxford-' + 'cortl-maxprob-thr0-1mm' + '.nii.gz'
    nifti_target_split = os.path.join(nifti_dir, split_atlas_fname)
    nibabel.Nifti1Image(atlas_data, np.eye(4) * 3).to_filename(
        nifti_target_split)
    ho = atlas.fetch_atlas_harvard_oxford(target_atlas,
                                          data_dir=tst.tmpdir,
                                          symmetric_split=True)

    assert_true(isinstance(ho.maps, nibabel.Nifti1Image))
    assert_true(isinstance(ho.labels, list))
    assert_equal(len(ho.labels), 7)
    assert_equal(ho.labels[0], "Background")
    assert_equal(ho.labels[1], "Left R1")
    assert_equal(ho.labels[2], "Right R1")
    assert_equal(ho.labels[3], "Left R2")
    assert_equal(ho.labels[4], "Right R2")
    assert_equal(ho.labels[5], "Left R3")
    assert_equal(ho.labels[6], "Right R3")


@with_setup(setup_mock, teardown_mock)
@with_setup(tst.setup_tmpdata, tst.teardown_tmpdata)
def test_fetch_atlas_craddock_2012():
    bunch = atlas.fetch_atlas_craddock_2012(data_dir=tst.tmpdir,
                                            verbose=0)

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
    assert_equal(len(tst.mock_url_request.urls), 1)
    for key, fn in zip(keys, filenames):
        assert_equal(bunch[key], os.path.join(tst.tmpdir, 'craddock_2012', fn))
    assert_not_equal(bunch.description, '')


@with_setup(setup_mock, teardown_mock)
@with_setup(tst.setup_tmpdata, tst.teardown_tmpdata)
def test_fetch_atlas_smith_2009():
    bunch = atlas.fetch_atlas_smith_2009(data_dir=tst.tmpdir, verbose=0)

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

    assert_equal(len(tst.mock_url_request.urls), 6)
    for key, fn in zip(keys, filenames):
        assert_equal(bunch[key], os.path.join(tst.tmpdir, 'smith_2009', fn))
    assert_not_equal(bunch.description, '')


def test_fetch_coords_power_2011():
    bunch = atlas.fetch_coords_power_2011()
    assert_equal(len(bunch.rois), 264)
    assert_not_equal(bunch.description, '')


def test_fetch_coords_seitzman_2018():
    bunch = atlas.fetch_coords_seitzman_2018()
    assert_equal(len(bunch.rois), 300)
    assert_equal(len(bunch.radius), 300)
    assert_equal(len(bunch.networks), 300)
    assert_equal(len(bunch.regions), 300)
    assert_equal(len(np.unique(bunch.networks)), 14)
    assert_equal(len(np.unique(bunch.regions)), 8)
    np.testing.assert_array_equal(bunch.networks, np.sort(bunch.networks))
    assert_not_equal(bunch.description, '')

    assert bunch.regions[0] == "cortexL"

    bunch = atlas.fetch_coords_seitzman_2018(ordered_regions=False)
    assert_true(np.any(bunch.networks != np.sort(bunch.networks)))



@with_setup(setup_mock, teardown_mock)
@with_setup(tst.setup_tmpdata, tst.teardown_tmpdata)
def test_fetch_atlas_destrieux_2009():
    datadir = os.path.join(tst.tmpdir, 'destrieux_2009')
    os.mkdir(datadir)
    dummy = open(os.path.join(
        datadir, 'destrieux2009_rois_labels_lateralized.csv'), 'w')
    dummy.write("name,index")
    dummy.close()
    bunch = atlas.fetch_atlas_destrieux_2009(data_dir=tst.tmpdir,
                                             verbose=0)

    assert_equal(len(tst.mock_url_request.urls), 1)
    assert_equal(bunch['maps'], os.path.join(
        tst.tmpdir, 'destrieux_2009', 'destrieux2009_rois_lateralized.nii.gz'))

    dummy = open(os.path.join(
        datadir, 'destrieux2009_rois_labels.csv'), 'w')
    dummy.write("name,index")
    dummy.close()
    bunch = atlas.fetch_atlas_destrieux_2009(
        lateralized=False, data_dir=tst.tmpdir, verbose=0)

    assert_equal(len(tst.mock_url_request.urls), 1)
    assert_equal(bunch['maps'], os.path.join(
        tst.tmpdir, 'destrieux_2009', 'destrieux2009_rois.nii.gz'))


@with_setup(setup_mock, teardown_mock)
@with_setup(tst.setup_tmpdata, tst.teardown_tmpdata)
def test_fetch_atlas_msdl():
    datadir = os.path.join(tst.tmpdir, 'msdl_atlas')
    os.mkdir(datadir)
    os.mkdir(os.path.join(datadir, 'MSDL_rois'))
    data_dir = os.path.join(datadir, 'MSDL_rois', 'msdl_rois_labels.csv')
    csv = np.rec.array([(1.5, 1.5, 1.5, 'Aud', 'Aud'),
                        (1.2, 1.3, 1.4, 'DMN', 'DMN')],
                       dtype=[('x', '<f8'), ('y', '<f8'),
                              ('z', '<f8'), ('name', 'S12'),
                              ('net_name', 'S19')])
    with open(data_dir, 'wb') as csv_file:
        header = '{0}\n'.format(','.join(csv.dtype.names))
        csv_file.write(header.encode())
        np.savetxt(csv_file, csv, delimiter=',', fmt='%s')

    dataset = atlas.fetch_atlas_msdl(data_dir=tst.tmpdir, verbose=0)
    assert_true(isinstance(dataset.labels, list))
    assert_true(isinstance(dataset.region_coords, list))
    assert_true(isinstance(dataset.networks, list))
    assert_true(isinstance(dataset.maps, _basestring))
    assert_equal(len(tst.mock_url_request.urls), 1)
    assert_not_equal(dataset.description, '')


@with_setup(setup_mock, teardown_mock)
@with_setup(tst.setup_tmpdata, tst.teardown_tmpdata)
def test_fetch_atlas_yeo_2011():
    dataset = atlas.fetch_atlas_yeo_2011(data_dir=tst.tmpdir, verbose=0)
    assert_true(isinstance(dataset.anat, _basestring))
    assert_true(isinstance(dataset.colors_17, _basestring))
    assert_true(isinstance(dataset.colors_7, _basestring))
    assert_true(isinstance(dataset.thick_17, _basestring))
    assert_true(isinstance(dataset.thick_7, _basestring))
    assert_true(isinstance(dataset.thin_17, _basestring))
    assert_true(isinstance(dataset.thin_7, _basestring))
    assert_equal(len(tst.mock_url_request.urls), 1)
    assert_not_equal(dataset.description, '')


@with_setup(setup_mock, teardown_mock)
@with_setup(tst.setup_tmpdata, tst.teardown_tmpdata)
def test_fetch_atlas_aal():
    ho_dir = os.path.join(tst.tmpdir, 'aal_SPM12', 'aal', 'atlas')
    os.makedirs(ho_dir)
    with open(os.path.join(ho_dir, 'AAL.xml'), 'w') as xml_file:
        xml_file.write("<?xml version='1.0' encoding='us-ascii'?> "
                       "<metadata>"
                       "</metadata>")
    dataset = atlas.fetch_atlas_aal(data_dir=tst.tmpdir, verbose=0)
    assert_true(isinstance(dataset.maps, _basestring))
    assert_true(isinstance(dataset.labels, list))
    assert_true(isinstance(dataset.indices, list))
    assert_equal(len(tst.mock_url_request.urls), 1)

    assert_raises_regex(ValueError, 'The version of AAL requested "FLS33"',
                        atlas.fetch_atlas_aal, version="FLS33",
                        data_dir=tst.tmpdir, verbose=0)

    assert_not_equal(dataset.description, '')


@with_setup(setup_mock, teardown_mock)
@with_setup(tst.setup_tmpdata, tst.teardown_tmpdata)
def test_fetch_atlas_basc_multiscale_2015():
    # default version='sym'
    data_sym = atlas.fetch_atlas_basc_multiscale_2015(data_dir=tst.tmpdir,
                                                      verbose=0)
    # version='asym'
    data_asym = atlas.fetch_atlas_basc_multiscale_2015(version='asym',
                                                       verbose=0,
                                                       data_dir=tst.tmpdir)

    keys = ['scale007', 'scale012', 'scale020', 'scale036', 'scale064',
            'scale122', 'scale197', 'scale325', 'scale444']

    dataset_name = 'basc_multiscale_2015'
    name_sym = 'template_cambridge_basc_multiscale_nii_sym'
    basenames_sym = ['template_cambridge_basc_multiscale_sym_' +
                     key + '.nii.gz' for key in keys]
    for key, basename_sym in zip(keys, basenames_sym):
        assert_equal(data_sym[key], os.path.join(tst.tmpdir, dataset_name,
                                                 name_sym, basename_sym))

    name_asym = 'template_cambridge_basc_multiscale_nii_asym'
    basenames_asym = ['template_cambridge_basc_multiscale_asym_' +
                      key + '.nii.gz' for key in keys]
    for key, basename_asym in zip(keys, basenames_asym):
        assert_equal(data_asym[key], os.path.join(tst.tmpdir, dataset_name,
                                                  name_asym, basename_asym))

    assert_equal(len(data_sym), 10)
    assert_raises_regex(ValueError,
                        'The version of Brain parcellations requested "aym"',
                        atlas.fetch_atlas_basc_multiscale_2015, version="aym",
                        data_dir=tst.tmpdir, verbose=0)

    assert_equal(len(tst.mock_url_request.urls), 2)
    assert_not_equal(data_sym.description, '')
    assert_not_equal(data_asym.description, '')


def test_fetch_coords_dosenbach_2010():
    bunch = atlas.fetch_coords_dosenbach_2010()
    assert_equal(len(bunch.rois), 160)
    assert_equal(len(bunch.labels), 160)
    assert_equal(len(np.unique(bunch.networks)), 6)
    assert_not_equal(bunch.description, '')
    np.testing.assert_array_equal(bunch.networks, np.sort(bunch.networks))

    bunch = atlas.fetch_coords_dosenbach_2010(ordered_regions=False)
    assert_true(np.any(bunch.networks != np.sort(bunch.networks)))


@with_setup(setup_mock, teardown_mock)
@with_setup(tst.setup_tmpdata, tst.teardown_tmpdata)
def test_fetch_atlas_allen_2011():
    bunch = atlas.fetch_atlas_allen_2011(data_dir=tst.tmpdir, verbose=0)
    keys = ("maps",
            "rsn28",
            "comps")

    filenames = ["ALL_HC_unthresholded_tmaps.nii.gz",
                 "RSN_HC_unthresholded_tmaps.nii.gz",
                 "rest_hcp_agg__component_ica_.nii.gz"]

    assert_equal(len(tst.mock_url_request.urls), 1)
    for key, fn in zip(keys, filenames):
        assert_equal(bunch[key], os.path.join(tst.tmpdir, 'allen_rsn_2011',
                                              'allen_rsn_2011', fn))

    assert_not_equal(bunch.description, '')


@with_setup(setup_mock, teardown_mock)
@with_setup(tst.setup_tmpdata, tst.teardown_tmpdata)
def test_fetch_atlas_surf_destrieux(data_dir=tst.tmpdir, verbose=0):

    # Old nibabel versions does not support 'write_annot'
    if LooseVersion(nibabel.__version__) <= LooseVersion('1.2.0'):
        raise SkipTest

    data_dir = os.path.join(tst.tmpdir, 'destrieux_surface')
    os.mkdir(data_dir)
    # Create mock annots
    for hemi in ('left', 'right'):
        nibabel.freesurfer.write_annot(
                os.path.join(data_dir,
                             '%s.aparc.a2009s.annot' % hemi),
                np.arange(4), np.zeros((4, 5)), 5 * ['a'],
                )

    bunch = atlas.fetch_atlas_surf_destrieux(data_dir=tst.tmpdir, verbose=0)
    # Our mock annots have 4 labels
    assert_equal(len(bunch.labels), 4)
    assert_equal(bunch.map_left.shape, (4, ))
    assert_equal(bunch.map_right.shape, (4, ))
    assert_not_equal(bunch.description, '')


def _get_small_fake_talairach():
    labels = ['*', 'b', 'a']
    all_labels = itertools.product(*(labels,) * 5)
    labels_txt = '\n'.join(map('.'.join, all_labels))
    extensions = nibabel.nifti1.Nifti1Extensions([
        nibabel.nifti1.Nifti1Extension(
            'afni', labels_txt.encode('utf-8'))
    ])
    img = nibabel.Nifti1Image(
        np.arange(243).reshape((3, 9, 9)),
        np.eye(4), nibabel.Nifti1Header(extensions=extensions))
    return img, all_labels


def _mock_talairach_fetch_files(data_dir, *args, **kwargs):
    img, all_labels = _get_small_fake_talairach()
    file_name = os.path.join(data_dir, 'talairach.nii')
    img.to_filename(file_name)
    return [file_name]


@with_setup(setup_mock, teardown_mock)
@with_setup(tst.setup_tmpdata, tst.teardown_tmpdata)
def test_fetch_atlas_talairach(data_dir=tst.tmpdir):
    atlas._fetch_files = _mock_talairach_fetch_files
    level_values = np.ones((81, 3)) * [0, 1, 2]
    talairach = atlas.fetch_atlas_talairach('hemisphere', data_dir=tst.tmpdir)
    assert_array_equal(get_data(talairach.maps).ravel(),
                       level_values.T.ravel())
    assert_array_equal(talairach.labels, ['Background', 'b', 'a'])
    talairach = atlas.fetch_atlas_talairach('ba', data_dir=tst.tmpdir)
    assert_array_equal(get_data(talairach.maps).ravel(),
                       level_values.ravel())
    assert_raises(ValueError, atlas.fetch_atlas_talairach, 'bad_level')

@with_setup(tst.setup_tmpdata, tst.teardown_tmpdata)
def test_fetch_atlas_pauli_2017():
    data_dir = os.path.join(tst.tmpdir, 'pauli_2017')

    data = atlas.fetch_atlas_pauli_2017('labels', data_dir)
    assert_equal(len(data.labels), 16)

    values = get_data(nibabel.load(data.maps))
    assert_equal(len(np.unique(values)), 17)

    data = atlas.fetch_atlas_pauli_2017('prob', data_dir)
    assert_equal(nibabel.load(data.maps).shape[-1], 16)

@with_setup(tst.setup_tmpdata, tst.teardown_tmpdata)
def test_fetch_atlas_schaefer_2018():
    valid_n_rois = list(range(100, 1100, 100))
    valid_yeo_networks = [7, 17]
    valid_resolution_mm = [1, 2]

    assert_raises(ValueError, atlas.fetch_atlas_schaefer_2018, n_rois=44)
    assert_raises(ValueError, atlas.fetch_atlas_schaefer_2018, yeo_networks=10)
    assert_raises(ValueError, atlas.fetch_atlas_schaefer_2018, resolution_mm=3)

    for n_rois, yeo_networks, resolution_mm in \
            itertools.product(valid_n_rois, valid_yeo_networks,
                              valid_resolution_mm):
        data = atlas.fetch_atlas_schaefer_2018(n_rois=n_rois,
                                               yeo_networks=yeo_networks,
                                               resolution_mm=resolution_mm,
                                               data_dir=tst.tmpdir,
                                               verbose=0)
        assert_not_equal(data.description, '')
        assert_true(isinstance(data.maps, _basestring))
        assert_true(isinstance(data.labels, np.ndarray))
        assert_equal(len(data.labels), n_rois)
        assert_true(data.labels[0].astype(str).startswith("{}Networks".
                                              format(yeo_networks)))
        img = nibabel.load(data.maps)
        assert_equal(img.header.get_zooms()[0], resolution_mm)
        assert_true(np.array_equal(np.unique(img.dataobj),
                                   np.arange(n_rois+1)))
