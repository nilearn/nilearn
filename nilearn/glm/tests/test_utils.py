#!/usr/bin/env python
import os

import numpy as np
import pandas as pd
import pytest
import scipy.linalg as spl

from nibabel.tmpdirs import InTemporaryDirectory
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from scipy.stats import norm

from nilearn._utils.data_gen import create_fake_bids_dataset, generate_fake_fmri
from nilearn._utils.glm import (_check_and_load_tables,
                                _check_list_length_match, _check_run_tables,
                                full_rank, get_bids_files,
                                get_design_from_fslmat, multiple_fast_inverse,
                                multiple_mahalanobis, parse_bids_filename,
                                positive_reciprocal, z_score)
from nilearn.input_data import NiftiMasker
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix
from nilearn.datasets import fetch_spm_auditory
from nilearn.image import mean_img


def test_full_rank():
    n, p = 10, 5
    X = np.random.randn(n, p)
    X_, _ = full_rank(X)
    assert_array_almost_equal(X, X_)
    X[:, -1] = X[:, :-1].sum(1)
    X_, cond = full_rank(X)
    assert cond > 1.e10
    assert_array_almost_equal(X, X_)


def test_z_score():
    p = np.random.rand(10)
    assert_array_almost_equal(norm.sf(z_score(p)), p)
    # check the numerical precision
    for p in [1.e-250, 1 - 1.e-16]:
        assert_array_almost_equal(z_score(p), norm.isf(p))
    assert_array_almost_equal(z_score(np.float32(1.e-100)), norm.isf(1.e-300))


def test_z_score_opposite_contrast():
    fmri, mask = generate_fake_fmri(shape=(50, 20, 50), length=96)

    nifti_masker = NiftiMasker(mask_img=mask)
    data = nifti_masker.fit_transform(fmri)

    frametimes = np.linspace(0, (96 - 1) * 2, 96)

    for i in [0, 20]:
        design_matrix = make_first_level_design_matrix(
            frametimes, hrf_model='spm',
            add_regs=np.array(data[:, i]).reshape(-1, 1))
        c1 = np.array([1] + [0] * (design_matrix.shape[1] - 1))
        c2 = np.array([0] + [1] + [0] * (design_matrix.shape[1] - 2))
        contrasts = {'seed1 - seed2': c1 - c2, 'seed2 - seed1': c2 - c1}
        fmri_glm = FirstLevelModel(t_r=2., 
                                   noise_model='ar1', 
                                   standardize=False, 
                                   hrf_model='spm', 
                                   drift_model='cosine')
        fmri_glm.fit(fmri, design_matrices=design_matrix)
        z_map_seed1_vs_seed2 = fmri_glm.compute_contrast(
            contrasts['seed1 - seed2'], output_type='z_score')
        z_map_seed2_vs_seed1 = fmri_glm.compute_contrast(
            contrasts['seed2 - seed1'], output_type='z_score')

        a = np.allclose([z_map_seed1_vs_seed2.get_fdata().min()],
                    [-z_map_seed2_vs_seed1.get_fdata().max()])
        print(z_map_seed1_vs_seed2.get_fdata().min())
        print(-z_map_seed2_vs_seed1.get_fdata().max())
        print(a)
        b = np.allclose([z_map_seed1_vs_seed2.get_fdata().max()],
                    [-z_map_seed2_vs_seed1.get_fdata().min()])
        print(z_map_seed1_vs_seed2.get_fdata().max())
        print(-z_map_seed2_vs_seed1.get_fdata().min())
        print(b)


def test_mahalanobis():
    n = 50
    x = np.random.rand(n) / n
    A = np.random.rand(n, n) / n
    A = np.dot(A.transpose(), A) + np.eye(n)
    mah = np.dot(x, np.dot(spl.inv(A), x))
    assert_almost_equal(mah, multiple_mahalanobis(x, A), decimal=1)


def test_mahalanobis2():
    n = 50
    x = np.random.randn(n, 3)
    Aa = np.zeros([n, n, 3])
    for i in range(3):
        A = np.random.randn(120, n)
        A = np.dot(A.T, A)
        Aa[:, :, i] = A
    i = np.random.randint(3)
    mah = np.dot(x[:, i], np.dot(spl.inv(Aa[:, :, i]), x[:, i]))
    f_mah = (multiple_mahalanobis(x, Aa))[i]
    assert np.allclose(mah, f_mah)


def test_mahalanobis_errors():
    effect = np.zeros((1, 2, 3))
    cov = np.zeros((3, 3, 3))
    with pytest.raises(ValueError):
        multiple_mahalanobis(effect, cov)

    cov = np.zeros((1, 2, 3))
    with pytest.raises(ValueError):
        multiple_mahalanobis(effect, cov)


def test_multiple_fast_inv():
    shape = (10, 20, 20)
    X = np.random.randn(shape[0], shape[1], shape[2])
    X_inv_ref = np.zeros(shape)
    for i in range(shape[0]):
        X[i] = np.dot(X[i], X[i].T)
        X_inv_ref[i] = spl.inv(X[i])
    X_inv = multiple_fast_inverse(X)
    assert_almost_equal(X_inv_ref, X_inv)


def test_multiple_fast_inverse_errors():
    shape = (2, 2, 2)
    X = np.zeros(shape)
    with pytest.raises(ValueError):
        multiple_fast_inverse(X)

    shape = (10, 20, 20)
    X = np.zeros(shape)
    with pytest.raises(ValueError):
        multiple_fast_inverse(X)


def test_pos_recipr():
    X = np.array([2, 1, -1, 0], dtype=np.int8)
    eX = np.array([0.5, 1, 0, 0])
    Y = positive_reciprocal(X)
    assert_array_almost_equal, Y, eX
    assert Y.dtype.type == np.float64
    X2 = X.reshape((2, 2))
    Y2 = positive_reciprocal(X2)
    assert_array_almost_equal, Y2, eX.reshape((2, 2))
    # check that lists have arrived
    XL = [0, 1, -1]
    assert_array_almost_equal, positive_reciprocal(XL), [0, 1, 0]
    # scalars
    assert positive_reciprocal(-1) == 0
    assert positive_reciprocal(0) == 0
    assert positive_reciprocal(2) == 0.5


def test_img_table_checks():
    # check matching lengths
    with pytest.raises(ValueError):
        _check_list_length_match([''] * 2, [''], "", "")
    # check tables type and that can be loaded
    with pytest.raises(ValueError):
        _check_and_load_tables(['.csv', '.csv'], "")
    with pytest.raises(TypeError):
        _check_and_load_tables([np.array([0]), pd.DataFrame()], "")
    with pytest.raises(ValueError):
        _check_and_load_tables(['.csv', pd.DataFrame()], "")
    # check high level wrapper keeps behavior
    with pytest.raises(ValueError):
        _check_run_tables([''] * 2, [''], "")
    with pytest.raises(ValueError):
        _check_run_tables([''] * 2, ['.csv', '.csv'], "")
    with pytest.raises(TypeError):
        _check_run_tables([''] * 2, [np.array([0]), pd.DataFrame()], "")
    with pytest.raises(ValueError):
        _check_run_tables([''] * 2, ['.csv', pd.DataFrame()], "")


def test_get_bids_files():
    with InTemporaryDirectory():
        bids_path = create_fake_bids_dataset(n_sub=10, n_ses=2,
                                             tasks=['localizer', 'main'],
                                             n_runs=[1, 3])
        # For each possible possible option of file selection we check
        # that we recover the appropriate amount of files, as included
        # in the fake bids dataset.

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


def test_get_design_from_fslmat(tmp_path):
    fsl_mat_path = os.path.join(str(tmp_path), 'fsl_mat.txt')
    matrix = np.ones((5, 5))
    with open(fsl_mat_path, 'w') as fsl_mat:
        fsl_mat.write('/Matrix\n')
        for row in matrix:
            for val in row:
                fsl_mat.write(str(val) + '\t')
            fsl_mat.write('\n')
    design_matrix = get_design_from_fslmat(fsl_mat_path)
    assert design_matrix.shape == matrix.shape
