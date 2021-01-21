"""
Test the datasets module
"""
# Author: Alexandre Abraham
# License: simplified BSD

import os
import shutil
from pathlib import Path

import nibabel
import numpy as np
import pandas as pd
import pytest

from nilearn.datasets import utils, struct
from nilearn.datasets._testing import list_to_archive, dict_to_archive



def test_get_dataset_dir(tmp_path):
    # testing folder creation under different environments, enforcing
    # a custom clean install
    os.environ.pop('NILEARN_DATA', None)
    os.environ.pop('NILEARN_SHARED_DATA', None)

    expected_base_dir = os.path.expanduser('~/nilearn_data')
    data_dir = utils._get_dataset_dir('test', verbose=0)
    assert data_dir == os.path.join(expected_base_dir, 'test')
    assert os.path.exists(data_dir)
    shutil.rmtree(data_dir)

    expected_base_dir = str(tmp_path / 'test_nilearn_data')
    os.environ['NILEARN_DATA'] = expected_base_dir
    data_dir = utils._get_dataset_dir('test', verbose=0)
    assert data_dir == os.path.join(expected_base_dir, 'test')
    assert os.path.exists(data_dir)
    shutil.rmtree(data_dir)

    expected_base_dir = str(tmp_path / 'nilearn_shared_data')
    os.environ['NILEARN_SHARED_DATA'] = expected_base_dir
    data_dir = utils._get_dataset_dir('test', verbose=0)
    assert data_dir == os.path.join(expected_base_dir, 'test')
    assert os.path.exists(data_dir)
    shutil.rmtree(data_dir)

    expected_base_dir = str(tmp_path / 'env_data')
    expected_dataset_dir = os.path.join(expected_base_dir, 'test')
    data_dir = utils._get_dataset_dir(
        'test', default_paths=[expected_dataset_dir], verbose=0)
    assert data_dir == os.path.join(expected_base_dir, 'test')
    assert os.path.exists(data_dir)
    shutil.rmtree(data_dir)

    no_write = str(tmp_path / 'no_write')
    os.makedirs(no_write)
    os.chmod(no_write, 0o400)

    expected_base_dir = str(tmp_path / 'nilearn_shared_data')
    os.environ['NILEARN_SHARED_DATA'] = expected_base_dir
    data_dir = utils._get_dataset_dir('test',
                                      default_paths=[no_write],
                                      verbose=0)
    # Non writeable dir is returned because dataset may be in there.
    assert data_dir == no_write
    assert os.path.exists(data_dir)
    os.chmod(no_write, 0o600)
    shutil.rmtree(data_dir)

    # Verify exception for a path which exists and is a file
    test_file = str(tmp_path / 'some_file')
    with open(test_file, 'w') as out:
        out.write('abcfeg')
    with pytest.raises(OSError,
                       match='Nilearn tried to store the dataset '
                             'in the following directories, but'):
        utils._get_dataset_dir('test', test_file, verbose=0)


def test_fetch_icbm152_2009(tmp_path, request_mocker):
    dataset = struct.fetch_icbm152_2009(data_dir=str(tmp_path), verbose=0)
    assert isinstance(dataset.csf, str)
    assert isinstance(dataset.eye_mask, str)
    assert isinstance(dataset.face_mask, str)
    assert isinstance(dataset.gm, str)
    assert isinstance(dataset.mask, str)
    assert isinstance(dataset.pd, str)
    assert isinstance(dataset.t1, str)
    assert isinstance(dataset.t2, str)
    assert isinstance(dataset.t2_relax, str)
    assert isinstance(dataset.wm, str)
    assert request_mocker.url_count == 1
    assert dataset.description != ''


def _make_oasis_data(dartel=True):
    n_subjects = 457
    prefix = "mwrc" if dartel else "mwc"
    ids = pd.DataFrame(
        {"ID": list(map("OAS1_{:04}".format, range(n_subjects)))}
    ).to_csv(index=False, sep="\t")
    data = {"oasis_cross-sectional.csv": ids, "data_usage_agreement.txt": ""}
    path_pattern = str(Path(
        "OAS1_{subj:04}_MR1",
        "{prefix}{kind}OAS1_{subj:04}_MR1_mpr_anon_fslswapdim_bet.nii.gz"))
    for i in range(457):
        for kind in [1, 2]:
            data[path_pattern.format(subj=i, kind=kind, prefix=prefix)] = ""
    return dict_to_archive(data)


def test_fetch_oasis_vbm(tmp_path, request_mocker):
    request_mocker.url_mapping["*archive_dartel.tgz*"] = _make_oasis_data()
    request_mocker.url_mapping["*archive.tgz*"] = _make_oasis_data(False)

    dataset = struct.fetch_oasis_vbm(
        data_dir=str(tmp_path), verbose=0)
    assert len(dataset.gray_matter_maps) == 403
    assert len(dataset.white_matter_maps) == 403
    assert isinstance(dataset.gray_matter_maps[0], str)
    assert isinstance(dataset.white_matter_maps[0], str)
    assert isinstance(dataset.ext_vars, np.recarray)
    assert isinstance(dataset.data_usage_agreement, str)
    assert request_mocker.url_count == 1

    dataset = struct.fetch_oasis_vbm(data_dir=str(tmp_path),
                                     dartel_version=False, verbose=0)
    assert len(dataset.gray_matter_maps) == 415
    assert len(dataset.white_matter_maps) == 415
    assert isinstance(dataset.gray_matter_maps[0], str)
    assert isinstance(dataset.white_matter_maps[0], str)
    assert isinstance(dataset.ext_vars, np.recarray)
    assert isinstance(dataset.data_usage_agreement, str)
    assert request_mocker.url_count == 2
    assert dataset.description != ''


def test_load_mni152_template():
    # All subjects
    template_nii = struct.load_mni152_template()
    assert template_nii.shape == (91, 109, 91)
    assert template_nii.header.get_zooms() == (2.0, 2.0, 2.0)


def test_load_mni152_brain_mask():
    brain_mask = struct.load_mni152_brain_mask()
    assert isinstance(brain_mask, nibabel.Nifti1Image)
    # standard MNI template shape
    assert brain_mask.shape == (91, 109, 91)


def test_fetch_icbm152_brain_gm_mask(tmp_path, request_mocker):
    dataset = struct.fetch_icbm152_2009(data_dir=str(tmp_path), verbose=0)
    struct.load_mni152_template().to_filename(dataset.gm)
    grey_matter_img = struct.fetch_icbm152_brain_gm_mask(
        data_dir=str(tmp_path), verbose=0)
    assert isinstance(grey_matter_img, nibabel.Nifti1Image)


@pytest.mark.parametrize("mesh", ["fsaverage5", "fsaverage"])
def test_fetch_surf_fsaverage(mesh, tmp_path, request_mocker):
    keys = {'pial_left', 'pial_right', 'infl_left', 'infl_right',
            'sulc_left', 'sulc_right', 'white_left', 'white_right'}
    request_mocker.url_mapping["*fsaverage.tar.gz"] = list_to_archive(
        [Path("fsaverage") / "{}.gii".format(k.replace("infl", "inflated")) for
         k in keys])
    dataset = struct.fetch_surf_fsaverage(mesh, data_dir=str(tmp_path))
    assert keys.issubset(set(dataset.keys()))
    assert dataset.description != ''


def test_fetch_surf_fsaverage5_sphere(tmp_path, request_mocker):
    request_mocker.url_mapping["*b79fy*"] = list_to_archive([
        "sphere_right.gii", "sphere_left.gii"
    ])
    for mesh in ['fsaverage5_sphere']:

        dataset = struct.fetch_surf_fsaverage(
            mesh, data_dir=str(tmp_path))

        keys = {'sphere_left', 'sphere_right'}

        assert keys.issubset(set(dataset.keys()))
        assert dataset.description != ''
