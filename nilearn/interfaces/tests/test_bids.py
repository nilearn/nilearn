"""Tests for the nilearn.interfaces.bids submodule."""
import os

import numpy as np
import pandas as pd
import pytest
from nibabel import load
from nibabel.tmpdirs import InTemporaryDirectory

from nilearn.maskers import NiftiMasker
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.second_level import SecondLevelModel
from nilearn._utils.data_gen import (
    create_fake_bids_dataset,
    generate_fake_fmri_data_and_design,
    write_fake_fmri_data_and_design,
)
from nilearn.interfaces.bids import (
    get_bids_files,
    parse_bids_filename,
    save_glm_to_bids,
)


def test_get_bids_files():
    with InTemporaryDirectory():
        bids_path = create_fake_bids_dataset(n_sub=10, n_ses=2,
                                             tasks=['localizer', 'main'],
                                             n_runs=[1, 3])
        # For each possible option of file selection we check that we
        # recover the appropriate amount of files, as included in the
        # fake bids dataset.

        # 250 files in total related to subject images. Top level files like
        # README not included
        selection = get_bids_files(bids_path)
        assert len(selection) == 250
        # 160 bold files expected. .nii and .json files
        selection = get_bids_files(bids_path, file_tag='bold')
        assert len(selection) == 160
        # Only 90 files are nii.gz. Bold and T1w files.
        selection = get_bids_files(bids_path, file_type='nii.gz')
        assert len(selection) == 90
        # Only 25 files correspond to subject 01
        selection = get_bids_files(bids_path, sub_label='01')
        assert len(selection) == 25
        # There are only 10 files in anat folders. One T1w per subject.
        selection = get_bids_files(bids_path, modality_folder='anat')
        assert len(selection) == 10
        # 20 files corresponding to run 1 of session 2 of main task.
        # 10 bold.nii.gz and 10 bold.json files. (10 subjects)
        filters = [('task', 'main'), ('run', '01'), ('ses', '02')]
        selection = get_bids_files(bids_path, file_tag='bold', filters=filters)
        assert len(selection) == 20
        # Get Top level folder files. Only 1 in this case, the README file.
        selection = get_bids_files(bids_path, sub_folder=False)
        assert len(selection) == 1
        # 80 counfonds (4 runs per ses & sub), testing `fmriprep` >= 20.2 path
        selection = get_bids_files(os.path.join(bids_path, 'derivatives'),
                                   file_tag='desc-confounds_timeseries')
        assert len(selection) == 80

    with InTemporaryDirectory():
        bids_path = create_fake_bids_dataset(n_sub=10, n_ses=2,
                                             tasks=['localizer', 'main'],
                                             n_runs=[1, 3],
                                             confounds_tag="desc-confounds_"
                                                           "regressors")
        # 80 counfonds (4 runs per ses & sub), testing `fmriprep` >= 20.2 path
        selection = get_bids_files(os.path.join(bids_path, 'derivatives'),
                                   file_tag='desc-confounds_regressors')
        assert len(selection) == 80


def test_parse_bids_filename():
    fields = ['sub', 'ses', 'task', 'lolo']
    labels = ['01', '01', 'langloc', 'lala']
    file_name = 'sub-01_ses-01_task-langloc_lolo-lala_bold.nii.gz'
    file_path = os.path.join('dataset', 'sub-01', 'ses-01', 'func', file_name)
    file_dict = parse_bids_filename(file_path)
    for fidx, field in enumerate(fields):
        assert file_dict[field] == labels[fidx]
    assert file_dict['file_type'] == 'nii.gz'
    assert file_dict['file_tag'] == 'bold'
    assert file_dict['file_path'] == file_path
    assert file_dict['file_basename'] == file_name
    assert file_dict['file_fields'] == fields


def test_save_glm_to_bids(tmp_path_factory):
    """Test that save_glm_to_bids saves the appropriate files.

    This test reuses code from
    nilearn.glm.tests.test_first_level.test_high_level_glm_one_session.
    """
    tmpdir = tmp_path_factory.mktemp('test_save_glm_results')

    EXPECTED_FILENAMES = [
        'dataset_description.json',
        'sub-01_ses-01_task-nback_contrast-effectsOfInterest_design.svg',
        (
            'sub-01_ses-01_task-nback_contrast-effectsOfInterest_'
            'stat-F_statmap.nii.gz'
        ),
        (
            'sub-01_ses-01_task-nback_contrast-effectsOfInterest_'
            'stat-effect_statmap.nii.gz'
        ),
        (
            'sub-01_ses-01_task-nback_contrast-effectsOfInterest_'
            'stat-p_statmap.nii.gz'
        ),
        (
            'sub-01_ses-01_task-nback_contrast-effectsOfInterest_'
            'stat-variance_statmap.nii.gz'
        ),
        (
            'sub-01_ses-01_task-nback_contrast-effectsOfInterest_'
            'stat-z_statmap.nii.gz'
        ),
        'sub-01_ses-01_task-nback_design.svg',
        'sub-01_ses-01_task-nback_design.tsv',
        'sub-01_ses-01_task-nback_stat-errorts_statmap.nii.gz',
        'sub-01_ses-01_task-nback_stat-rSquare_statmap.nii.gz',
        'sub-01_ses-01_task-nback_statmap.json',
    ]

    shapes, rk = [(7, 8, 9, 15)], 3
    mask, fmri_data, design_matrices = generate_fake_fmri_data_and_design(
        shapes,
        rk,
    )

    masker = NiftiMasker(mask)
    masker.fit()

    # Call with verbose (improve coverage)
    single_session_model = FirstLevelModel(
        mask_img=None,
        minimize_memory=False,
    ).fit(
        fmri_data[0],
        design_matrices=design_matrices[0],
    )

    contrasts = {
        'effects of interest': np.eye(rk),
    }
    contrast_types = {
        'effects of interest': 'F',
    }
    save_glm_to_bids(
        model=single_session_model,
        contrasts=contrasts,
        contrast_types=contrast_types,
        out_dir=tmpdir,
        prefix='sub-01_ses-01_task-nback'
    )

    for fname in EXPECTED_FILENAMES:
        full_filename = os.path.join(tmpdir, fname)
        assert os.path.isfile(full_filename)


def test_save_glm_to_bids_contrast_definitions(tmp_path_factory):
    """Test that save_glm_to_bids operates on different contrast definitions as
    expected.

    This test reuses code from
    nilearn.glm.tests.test_first_level.test_high_level_glm_one_session.
    """
    tmpdir = tmp_path_factory.mktemp(
        'test_save_glm_to_bids_contrast_definitions'
    )

    EXPECTED_FILENAMES = [
        'dataset_description.json',
        (
            'sub-01_ses-01_task-nback_contrast-aaaMinusBbb'
            '_stat-effect_statmap.nii.gz'
        ),
        'sub-01_ses-01_task-nback_contrast-aaaMinusBbb_stat-p_statmap.nii.gz',
        'sub-01_ses-01_task-nback_contrast-aaaMinusBbb_stat-t_statmap.nii.gz',
        (
            'sub-01_ses-01_task-nback_contrast-aaaMinusBbb'
            '_stat-variance_statmap.nii.gz'
        ),
        'sub-01_ses-01_task-nback_contrast-aaaMinusBbb_stat-z_statmap.nii.gz',
        'sub-01_ses-01_task-nback_run-1_contrast-aaaMinusBbb_design.svg',
        'sub-01_ses-01_task-nback_run-1_design.svg',
        'sub-01_ses-01_task-nback_run-1_design.tsv',
        'sub-01_ses-01_task-nback_run-2_contrast-aaaMinusBbb_design.svg',
        'sub-01_ses-01_task-nback_run-2_design.svg',
        'sub-01_ses-01_task-nback_run-2_design.tsv',
        'sub-01_ses-01_task-nback_stat-errorts_statmap.nii.gz',
        'sub-01_ses-01_task-nback_stat-rSquare_statmap.nii.gz',
        'sub-01_ses-01_task-nback_statmap.json',
    ]

    # Create two runs of data
    shapes, rk = [(7, 8, 9, 15), (7, 8, 9, 15)], 3
    mask, fmri_data, design_matrices = generate_fake_fmri_data_and_design(
        shapes,
        rk,
    )
    # Rename two conditions in design matrices
    mapper = {
        design_matrices[0].columns[0]: 'AAA',
        design_matrices[0].columns[1]: 'BBB',
    }
    design_matrices[0] = design_matrices[0].rename(columns=mapper)
    mapper = {
        design_matrices[1].columns[0]: 'AAA',
        design_matrices[1].columns[1]: 'BBB',
    }
    design_matrices[1] = design_matrices[1].rename(columns=mapper)

    masker = NiftiMasker(mask)
    masker.fit()

    # Call with verbose (improve coverage)
    single_session_model = FirstLevelModel(
        mask_img=None,
        minimize_memory=False,
    ).fit(
        fmri_data,
        design_matrices=design_matrices,
    )

    # Contrast names must be strings
    contrasts = {5: np.eye(rk)}
    contrast_types = {5: 'F'}

    with pytest.raises(ValueError):
        save_glm_to_bids(
            model=single_session_model,
            contrasts=contrasts,
            contrast_types=contrast_types,
            out_dir=tmpdir,
            prefix='sub-01_ses-01_task-nback'
        )

    # Contrast definitions must be strings, numpy arrays, or lists
    contrasts = {'effects of interest': 5}
    contrast_types = {'effects of interest': 'F'}

    with pytest.raises(ValueError):
        save_glm_to_bids(
            model=single_session_model,
            contrasts=contrasts,
            contrast_types=contrast_types,
            out_dir=tmpdir,
            prefix='sub-01_ses-01_task-nback'
        )

    # Test string-based contrasts and undefined contrast types
    contrasts = ['AAA - BBB']

    save_glm_to_bids(
        model=single_session_model,
        contrasts=contrasts,
        contrast_types=None,
        out_dir=tmpdir,
        prefix='sub-01_ses-01_task-nback'
    )

    for fname in EXPECTED_FILENAMES:
        full_filename = os.path.join(tmpdir, fname)
        assert os.path.isfile(full_filename)


def test_save_glm_to_bids_second_level():
    """Test save_glm_to_bids on a SecondLevelModel.

    This test reuses code from
    nilearn.glm.tests.test_second_level.test_high_level_glm_with_paths.
    """
    EXPECTED_FILENAMES = [
        'dataset_description.json',
        'task-nback_contrast-effectsOfInterest_design.svg',
        'task-nback_contrast-effectsOfInterest_stat-F_statmap.nii.gz',
        'task-nback_contrast-effectsOfInterest_stat-effect_statmap.nii.gz',
        'task-nback_contrast-effectsOfInterest_stat-p_statmap.nii.gz',
        'task-nback_contrast-effectsOfInterest_stat-variance_statmap.nii.gz',
        'task-nback_contrast-effectsOfInterest_stat-z_statmap.nii.gz',
        'task-nback_design.svg',
        'task-nback_design.tsv',
        'task-nback_stat-errorts_statmap.nii.gz',
        'task-nback_stat-rSquare_statmap.nii.gz',
        'task-nback_statmap.json',
    ]

    with InTemporaryDirectory():
        shapes = ((7, 8, 9, 1),)
        mask, FUNCFILE, _ = write_fake_fmri_data_and_design(shapes)
        FUNCFILE = FUNCFILE[0]
        func_img = load(FUNCFILE)

        # Ordinary Least Squares case
        model = SecondLevelModel(mask_img=mask, minimize_memory=False)

        # fit model
        Y = [func_img] * 4
        X = pd.DataFrame([[1]] * 4, columns=['intercept'])
        model = model.fit(Y, design_matrix=X)

        contrasts = {
            'effects of interest': np.eye(
                len(model.design_matrix_.columns)
            )[0],
        }
        contrast_types = {
            'effects of interest': 'F',
        }

        save_glm_to_bids(
            model=model,
            contrasts=contrasts,
            contrast_types=contrast_types,
            out_dir='.',
            prefix='task-nback'
        )

        for fname in EXPECTED_FILENAMES:
            assert os.path.isfile(fname)
