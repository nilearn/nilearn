import itertools
import os
import shutil
import unittest.mock
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from nibabel import Nifti1Image, load
from numpy.testing import (
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
    assert_array_less,
)
from sklearn.cluster import KMeans

from nilearn._utils.data_gen import (
    add_metadata_to_bids_dataset,
    basic_paradigm,
    create_fake_bids_dataset,
    generate_fake_fmri_data_and_design,
    write_fake_fmri_data_and_design,
)
from nilearn.glm.contrasts import compute_fixed_effects
from nilearn.glm.first_level import (
    FirstLevelModel,
    first_level_from_bids,
    mean_scaling,
    run_glm,
)
from nilearn.glm.first_level.design_matrix import (
    check_design_matrix,
    make_first_level_design_matrix,
)
from nilearn.glm.first_level.first_level import (
    _check_and_load_tables,
    _check_list_length_match,
    _check_run_tables,
    _check_trial_type,
    _yule_walker,
)
from nilearn.glm.regression import ARModel, OLSModel
from nilearn.image import get_data
from nilearn.interfaces.bids import get_bids_files
from nilearn.maskers import NiftiMasker

BASEDIR = os.path.dirname(os.path.abspath(__file__))
FUNCFILE = os.path.join(BASEDIR, "functional.nii.gz")


def test_high_level_glm_one_run():
    shapes, rk = [(7, 8, 9, 15)], 3
    mask, fmri_data, design_matrices = generate_fake_fmri_data_and_design(
        shapes, rk
    )

    # Give an unfitted NiftiMasker as mask_img and check that we get an error
    masker = NiftiMasker(mask)
    with pytest.raises(
        ValueError, match="It seems that NiftiMasker has not been fitted."
    ):
        FirstLevelModel(mask_img=masker).fit(
            fmri_data[0], design_matrices=design_matrices[0]
        )

    # Give a fitted NiftiMasker with a None mask_img_ attribute
    # and check that the masker parameters are overridden by the
    # FirstLevelModel parameters
    masker.fit()
    masker.mask_img_ = None
    with pytest.warns(
        UserWarning, match="Parameter memory of the masker overridden"
    ):
        FirstLevelModel(mask_img=masker).fit(
            fmri_data[0], design_matrices=design_matrices[0]
        )

    # Give a fitted NiftiMasker
    masker = NiftiMasker(mask)
    masker.fit()
    single_run_model = FirstLevelModel(mask_img=masker).fit(
        fmri_data[0], design_matrices=design_matrices[0]
    )
    assert single_run_model.masker_ == masker

    # Call with verbose (improve coverage)
    single_run_model = FirstLevelModel(mask_img=None, verbose=1).fit(
        fmri_data[0], design_matrices=design_matrices[0]
    )

    single_run_model = FirstLevelModel(mask_img=None).fit(
        fmri_data[0], design_matrices=design_matrices[0]
    )
    assert isinstance(single_run_model.masker_.mask_img_, Nifti1Image)

    single_run_model = FirstLevelModel(mask_img=mask).fit(
        fmri_data[0], design_matrices=design_matrices[0]
    )
    z1 = single_run_model.compute_contrast(np.eye(rk)[:1])
    assert isinstance(z1, Nifti1Image)


def test_explicit_fixed_effects(tmp_path):
    """Test the fixed effects performed manually/explicitly."""
    shapes, rk = ((7, 8, 7, 15), (7, 8, 7, 16)), 3
    mask, fmri_data, design_matrices = write_fake_fmri_data_and_design(
        shapes, rk, file_path=tmp_path
    )
    contrast = np.eye(rk)[1]

    # run 1
    multi_run_model = FirstLevelModel(mask_img=mask).fit(
        fmri_data[0], design_matrices=design_matrices[:1]
    )
    dic1 = multi_run_model.compute_contrast(contrast, output_type="all")

    # run 2
    multi_run_model.fit(fmri_data[1], design_matrices=design_matrices[1:])
    dic2 = multi_run_model.compute_contrast(contrast, output_type="all")

    # fixed effects model
    multi_run_model.fit(fmri_data, design_matrices=design_matrices)
    fixed_fx_dic = multi_run_model.compute_contrast(
        contrast, output_type="all"
    )

    contrasts = [dic1["effect_size"], dic2["effect_size"]]
    variance = [dic1["effect_variance"], dic2["effect_variance"]]

    (
        fixed_fx_contrast,
        fixed_fx_variance,
        fixed_fx_stat,
    ) = compute_fixed_effects(contrasts, variance, mask)

    assert_almost_equal(
        get_data(fixed_fx_contrast), get_data(fixed_fx_dic["effect_size"])
    )
    assert_almost_equal(
        get_data(fixed_fx_variance), get_data(fixed_fx_dic["effect_variance"])
    )
    assert_almost_equal(
        get_data(fixed_fx_stat), get_data(fixed_fx_dic["stat"])
    )

    # ensure that using unbalanced effects size and variance images
    # raises an error
    with pytest.raises(
        ValueError,
        match=(
            "The number of contrast images .* differs "
            "from the number of variance images"
        ),
    ):
        compute_fixed_effects(contrasts * 2, variance, mask)

    # ensure that not providing the right number of dofs
    with pytest.raises(
        ValueError, match="degrees of freedom .* differs .* contrast images"
    ):
        compute_fixed_effects(contrasts, variance, mask, dofs=[100])


def test_explicit_fixed_effects_without_mask(tmp_path):
    """Test the fixed effects performed manually/explicitly with no mask."""
    shapes, rk = ((7, 8, 7, 15), (7, 8, 7, 16)), 3
    _, fmri_data, design_matrices = write_fake_fmri_data_and_design(
        shapes, rk, file_path=tmp_path
    )
    contrast = np.eye(rk)[1]

    # run 1
    multi_run_model = FirstLevelModel().fit(
        fmri_data[0], design_matrices=design_matrices[:1]
    )
    dic1 = multi_run_model.compute_contrast(contrast, output_type="all")

    # run 2
    multi_run_model.fit(fmri_data[1], design_matrices=design_matrices[1:])
    dic2 = multi_run_model.compute_contrast(contrast, output_type="all")

    # fixed effects model
    multi_run_model.fit(fmri_data, design_matrices=design_matrices)
    fixed_fx_dic = multi_run_model.compute_contrast(
        contrast, output_type="all"
    )

    contrasts = [dic1["effect_size"], dic2["effect_size"]]
    variance = [dic1["effect_variance"], dic2["effect_variance"]]

    # test without mask variable
    (
        fixed_fx_contrast,
        fixed_fx_variance,
        fixed_fx_stat,
        _,
    ) = compute_fixed_effects(contrasts, variance, return_z_score=True)
    assert_almost_equal(
        get_data(fixed_fx_contrast), get_data(fixed_fx_dic["effect_size"])
    )
    assert_almost_equal(
        get_data(fixed_fx_variance), get_data(fixed_fx_dic["effect_variance"])
    )
    assert_almost_equal(
        get_data(fixed_fx_stat), get_data(fixed_fx_dic["stat"])
    )


def test_high_level_glm_with_data(tmp_path):
    shapes, rk = ((7, 8, 7, 15), (7, 8, 7, 16)), 3
    _, fmri_data, design_matrices = write_fake_fmri_data_and_design(
        shapes, rk, file_path=tmp_path
    )

    multi_run_model = FirstLevelModel(mask_img=None).fit(
        fmri_data, design_matrices=design_matrices
    )
    n_voxels = get_data(multi_run_model.masker_.mask_img_).sum()
    z_image = multi_run_model.compute_contrast(np.eye(rk)[1])

    assert np.sum(get_data(z_image) != 0) == n_voxels
    assert get_data(z_image).std() < 3.0


def test_high_level_glm_with_data_with_mask(tmp_path):
    shapes, rk = ((7, 8, 7, 15), (7, 8, 7, 16)), 3
    mask, fmri_data, design_matrices = write_fake_fmri_data_and_design(
        shapes, rk, file_path=tmp_path
    )

    multi_run_model = FirstLevelModel(mask_img=mask).fit(
        fmri_data, design_matrices=design_matrices
    )

    z_image = multi_run_model.compute_contrast(
        np.eye(rk)[:2], output_type="z_score"
    )
    p_value = multi_run_model.compute_contrast(
        np.eye(rk)[:2], output_type="p_value"
    )
    stat_image = multi_run_model.compute_contrast(
        np.eye(rk)[:2], output_type="stat"
    )
    effect_image = multi_run_model.compute_contrast(
        np.eye(rk)[:2], output_type="effect_size"
    )
    variance_image = multi_run_model.compute_contrast(
        np.eye(rk)[:2], output_type="effect_variance"
    )

    assert_array_equal(get_data(z_image) == 0.0, get_data(load(mask)) == 0.0)
    assert (get_data(variance_image)[get_data(load(mask)) > 0] > 0.001).all()

    all_images = multi_run_model.compute_contrast(
        np.eye(rk)[:2], output_type="all"
    )

    assert_array_equal(get_data(all_images["z_score"]), get_data(z_image))
    assert_array_equal(get_data(all_images["p_value"]), get_data(p_value))
    assert_array_equal(get_data(all_images["stat"]), get_data(stat_image))
    assert_array_equal(
        get_data(all_images["effect_size"]), get_data(effect_image)
    )
    assert_array_equal(
        get_data(all_images["effect_variance"]), get_data(variance_image)
    )


def test_high_level_glm_with_paths(tmp_path):
    shapes, rk = ((7, 8, 7, 15), (7, 8, 7, 14)), 3
    mask_file, fmri_files, design_files = write_fake_fmri_data_and_design(
        shapes, rk, file_path=tmp_path
    )
    multi_run_model = FirstLevelModel(mask_img=None).fit(
        fmri_files, design_matrices=design_files
    )
    z_image = multi_run_model.compute_contrast(np.eye(rk)[1])

    assert_array_equal(z_image.affine, load(mask_file).affine)
    assert get_data(z_image).std() < 3.0


def test_high_level_glm_null_contrasts():
    # test that contrast computation is resilient to 0 values.
    shapes, rk = ((7, 8, 7, 15), (7, 8, 7, 19)), 3
    _, fmri_data, design_matrices = generate_fake_fmri_data_and_design(
        shapes, rk
    )

    multi_run_model = FirstLevelModel(mask_img=None).fit(
        fmri_data, design_matrices=design_matrices
    )
    single_run_model = FirstLevelModel(mask_img=None).fit(
        fmri_data[0], design_matrices=design_matrices[0]
    )
    z1 = multi_run_model.compute_contrast(
        [np.eye(rk)[:1], np.zeros((1, rk))], output_type="stat"
    )
    z2 = single_run_model.compute_contrast(np.eye(rk)[:1], output_type="stat")

    np.testing.assert_almost_equal(get_data(z1), get_data(z2))


def test_high_level_glm_different_design_matrices():
    # test that one can estimate a contrast when design matrices are different
    shapes, rk = ((7, 8, 7, 15), (7, 8, 7, 19)), 3
    mask, fmri_data, design_matrices = generate_fake_fmri_data_and_design(
        shapes, rk
    )

    # add a column to the second design matrix
    design_matrices[1]["new"] = np.ones((19, 1))

    # Fit a glm with two runs and design matrices
    multi_run_model = FirstLevelModel(mask_img=mask).fit(
        fmri_data, design_matrices=design_matrices
    )
    z_joint = multi_run_model.compute_contrast(
        [np.eye(rk)[:1], np.eye(rk + 1)[:1]], output_type="effect_size"
    )
    assert z_joint.shape == (7, 8, 7, 1)

    # compare the estimated effects to seprarately-fitted models
    model1 = FirstLevelModel(mask_img=mask).fit(
        fmri_data[0], design_matrices=design_matrices[0]
    )
    z1 = model1.compute_contrast(np.eye(rk)[:1], output_type="effect_size")
    model2 = FirstLevelModel(mask_img=mask).fit(
        fmri_data[1], design_matrices=design_matrices[1]
    )
    z2 = model2.compute_contrast(np.eye(rk + 1)[:1], output_type="effect_size")

    assert_almost_equal(get_data(z1) + get_data(z2), 2 * get_data(z_joint))


def test_high_level_glm_different_design_matrices_formulas():
    # test that one can estimate a contrast when design matrices are different
    shapes, rk = ((7, 8, 7, 15), (7, 8, 7, 19)), 3
    mask, fmri_data, design_matrices = generate_fake_fmri_data_and_design(
        shapes, rk
    )

    # make column names identical
    design_matrices[1].columns = design_matrices[0].columns
    # add a column to the second design matrix
    design_matrices[1]["new"] = np.ones((19, 1))

    # Fit a glm with two runs and design matrices
    multi_run_model = FirstLevelModel(mask_img=mask).fit(
        fmri_data, design_matrices=design_matrices
    )

    # Compute contrast with formulas
    cols_formula = tuple(design_matrices[0].columns[:2])
    formula = f"{cols_formula[0]}-{cols_formula[1]}"

    with pytest.warns(
        UserWarning, match="One contrast given, assuming it for all 2 runs"
    ):
        multi_run_model.compute_contrast(formula, output_type="effect_size")


def test_compute_contrast_num_contrasts():
    shapes, rk = ((7, 8, 7, 15), (7, 8, 7, 19), (7, 8, 7, 13)), 3
    mask, fmri_data, design_matrices = generate_fake_fmri_data_and_design(
        shapes, rk
    )

    # Fit a glm with 3 runs and design matrices
    multi_run_model = FirstLevelModel(mask_img=mask).fit(
        fmri_data, design_matrices=design_matrices
    )

    # raise when n_contrast != n_runs | 1
    with pytest.raises(
        ValueError, match="2 contrasts given, while there are 3 runs."
    ):
        multi_run_model.compute_contrast([np.eye(rk)[1]] * 2)

    multi_run_model.compute_contrast([np.eye(rk)[1]] * 3)

    with pytest.warns(
        UserWarning, match="One contrast given, assuming it for all 3 runs"
    ):
        multi_run_model.compute_contrast([np.eye(rk)[1]])


def test_run_glm_ols(rng):
    # Ordinary Least Squares case
    n, p, q = 33, 80, 10
    X, Y = rng.standard_normal(size=(p, q)), rng.standard_normal(size=(p, n))

    labels, results = run_glm(Y, X, "ols")

    assert_array_equal(labels, np.zeros(n))
    assert list(results.keys()) == [0.0]
    assert results[0.0].theta.shape == (q, n)
    assert_almost_equal(results[0.0].theta.mean(), 0, 1)
    assert_almost_equal(results[0.0].theta.var(), 1.0 / p, 1)
    assert isinstance(results[labels[0]].model, OLSModel)


def test_run_glm_ar1(rng):
    # ar(1) case
    n, p, q = 33, 80, 10
    X, Y = rng.standard_normal(size=(p, q)), rng.standard_normal(size=(p, n))

    labels, results = run_glm(Y, X, "ar1")

    assert len(labels) == n
    assert len(results.keys()) > 1
    tmp = sum(val.theta.shape[1] for val in results.values())
    assert tmp == n
    assert results[labels[0]].model.order == 1
    assert isinstance(results[labels[0]].model, ARModel)


def test_run_glm_ar3(rng):
    # ar(3) case
    n, p, q = 33, 80, 10
    X, Y = rng.standard_normal(size=(p, q)), rng.standard_normal(size=(p, n))

    labels_ar3, results_ar3 = run_glm(Y, X, "ar3", bins=10)

    assert len(labels_ar3) == n
    assert len(results_ar3.keys()) > 1
    tmp = sum(val.theta.shape[1] for val in results_ar3.values())
    assert tmp == n
    assert isinstance(results_ar3[labels_ar3[0]].model, ARModel)
    assert results_ar3[labels_ar3[0]].model.order == 3
    assert len(results_ar3[labels_ar3[0]].model.rho) == 3


def test_run_glm_errors(rng):
    """Check correct errors are thrown for nonsense noise model requests."""
    n, p, q = 33, 80, 10
    X, Y = rng.standard_normal(size=(p, q)), rng.standard_normal(size=(p, n))

    with pytest.raises(ValueError, match="AR order must be positive"):
        run_glm(Y, X, "ar0")
    match = (
        "AR order must be a positive integer specified as arN, "
        "where N is an integer."
    )
    with pytest.raises(ValueError, match=match):
        run_glm(Y, X, "arfoo")
    with pytest.raises(ValueError, match=match):
        run_glm(Y, X, "arr3")
    with pytest.raises(ValueError, match=match):
        run_glm(Y, X, "ar1.2")
    with pytest.raises(ValueError, match=match):
        run_glm(Y, X, "ar")
    with pytest.raises(ValueError, match="Acceptable noise models are "):
        run_glm(Y, X, "3ar")


@pytest.mark.parametrize(
    "ar_vals", [[-0.2], [-0.2, -0.5], [-0.2, -0.5, -0.7, -0.3]]
)
def test_glm_AR_estimates(rng, ar_vals):
    """Test that Yule-Walker AR fits are correct."""
    n, p, q = 1, 500, 2
    X_orig = rng.standard_normal((p, q))
    Y_orig = rng.standard_normal((p, n))

    ar_order = len(ar_vals)
    ar_arg = f"ar{ar_order}"

    X = X_orig.copy()
    Y = Y_orig.copy()

    for idx, lag in itertools.product(range(1, len(Y)), range(ar_order)):
        Y[idx] += ar_vals[lag] * Y[idx - 1 - lag]

    # Test using run_glm
    labels, results = run_glm(Y, X, ar_arg, bins=100)

    assert len(labels) == n

    for lab in results.keys():
        ar_estimate = lab.split("_")
        for lag in range(ar_order):
            assert_almost_equal(
                float(ar_estimate[lag]), ar_vals[lag], decimal=1
            )

    # Test using _yule_walker
    yw = _yule_walker(Y.T, ar_order)
    assert_almost_equal(yw[0], ar_vals, decimal=1)


def test_glm_AR_estimates_errors(rng):
    """Test Yule-Walker errors."""
    (n, p) = (1, 500)
    Y_orig = rng.standard_normal((p, n))

    with pytest.raises(TypeError, match="AR order must be an integer"):
        _yule_walker(Y_orig, 1.2)
    with pytest.raises(ValueError, match="AR order must be positive"):
        _yule_walker(Y_orig, 0)
    with pytest.raises(ValueError, match="AR order must be positive"):
        _yule_walker(Y_orig, -2)
    with pytest.raises(TypeError, match="at least 1 dim"):
        _yule_walker(np.array(0.0), 2)


@pytest.mark.parametrize("random_state", [3, np.random.RandomState(42)])
def test_glm_random_state(random_state):
    rng = np.random.RandomState(42)
    n, p, q = 33, 80, 10
    X, Y = rng.standard_normal(size=(p, q)), rng.standard_normal(size=(p, n))

    with unittest.mock.patch.object(
        KMeans,
        "__init__",
        autospec=True,
        side_effect=KMeans.__init__,
    ) as spy_kmeans:
        run_glm(Y, X, "ar3", random_state=random_state)
        spy_kmeans.assert_called_once_with(
            unittest.mock.ANY,
            n_clusters=unittest.mock.ANY,
            n_init=unittest.mock.ANY,
            random_state=random_state,
        )


def test_scaling(rng):
    """Test the scaling function."""
    shape = (400, 10)
    u = rng.standard_normal(size=shape)
    mean = 100 * rng.uniform(size=shape[1]) + 1
    Y = u + mean
    Y_, mean_ = mean_scaling(Y)
    assert_almost_equal(Y_.mean(0), 0, 5)
    assert_almost_equal(mean_, mean, 0)
    assert Y.std() > 1


def test_fmri_inputs(tmp_path):
    # Test processing of FMRI inputs
    shapes = ((7, 8, 9, 10),)
    mask, FUNCFILE, _ = write_fake_fmri_data_and_design(
        shapes, file_path=tmp_path
    )
    FUNCFILE = FUNCFILE[0]
    func_img = load(FUNCFILE)

    T = func_img.shape[-1]
    des = pd.DataFrame(np.ones((T, 1)), columns=[""])
    des_fname = str(tmp_path / "design.csv")
    des.to_csv(des_fname)

    conf = pd.DataFrame([0, 0])

    events = basic_paradigm()

    for fi in func_img, FUNCFILE:
        for d in des, des_fname:
            FirstLevelModel().fit(fi, design_matrices=d)

            FirstLevelModel(mask_img=None).fit([fi], design_matrices=d)

            FirstLevelModel(mask_img=mask).fit(fi, design_matrices=[d])

            FirstLevelModel(mask_img=mask).fit([fi], design_matrices=[d])

            # Provide t_r, confounds, and events but no design matrix
            FirstLevelModel(mask_img=mask, t_r=2.0).fit(
                fi,
                confounds=pd.DataFrame([0] * 10, columns=["conf"]),
                events=events,
            )

            # test with confounds as numpy array
            FirstLevelModel(mask_img=mask).fit(
                [fi], design_matrices=[d], confounds=conf.values
            )

            FirstLevelModel(mask_img=mask).fit(
                [fi, fi], design_matrices=[d, d]
            )
            FirstLevelModel(mask_img=None).fit(
                (fi, fi), design_matrices=(d, d)
            )


def test_fmri_inputs_errors(tmp_path):
    # Test processing of FMRI with wrong inputs
    shapes = ((7, 8, 9, 10),)
    mask, FUNCFILE, _ = write_fake_fmri_data_and_design(
        shapes, file_path=tmp_path
    )
    FUNCFILE = FUNCFILE[0]
    func_img = load(FUNCFILE)

    T = func_img.shape[-1]
    des = pd.DataFrame(np.ones((T, 1)), columns=[""])
    des_fname = str(tmp_path / "design.csv")
    des.to_csv(des_fname)

    conf = pd.DataFrame([0, 0])

    events = basic_paradigm()

    for fi in func_img, FUNCFILE:
        for d in des, des_fname:
            # test with confounds
            with pytest.warns(
                UserWarning, match="If design matrices are supplied"
            ):
                FirstLevelModel(mask_img=mask).fit(
                    [fi], design_matrices=[d], confounds=conf
                )

            # check that an error is raised if there is a
            # mismatch in the dimensions of the inputs
            with pytest.raises(
                ValueError, match="Rows in confounds does not match"
            ):
                FirstLevelModel(mask_img=mask, t_r=2.0).fit(
                    fi, confounds=conf, events=events
                )

            # test mismatch nupmber of image and events file
            match = r"len\(run_imgs\) .* does not match len\(events\) .*"
            with pytest.raises(ValueError, match=match):
                FirstLevelModel(mask_img=None, t_r=2.0).fit([fi, fi], d)
            with pytest.raises(ValueError, match=match):
                FirstLevelModel(mask_img=None, t_r=2.0).fit(fi, [d, d])

            # At least paradigms or design have to be given
            with pytest.raises(
                ValueError,
                match="events or design matrices must be provided",
            ):
                FirstLevelModel(mask_img=None).fit(fi)

            # If paradigms are given then both tr and slice time ref were
            # required
            match = (
                "t_r not given to FirstLevelModel object "
                "to compute design from events"
            )
            with pytest.raises(ValueError, match=match):
                FirstLevelModel(mask_img=None).fit(fi, d)
            with pytest.raises(ValueError, match=match):
                FirstLevelModel(mask_img=None, slice_time_ref=0.0).fit(fi, d)
            with pytest.raises(
                ValueError,
                match="The provided events data has no onset column.",
            ):
                FirstLevelModel(mask_img=None, t_r=1.0).fit(fi, d)

        # confounds rows do not match n_scans
        with pytest.raises(
            ValueError,
            match=(
                "Rows in confounds does not match "
                "n_scans in run_img at index 0."
            ),
        ):
            FirstLevelModel(mask_img=None, t_r=2.0).fit(fi, d, conf)


def test_first_level_design_creation(tmp_path):
    # Test processing of FMRI inputs
    shapes = ((7, 8, 9, 10),)
    mask, FUNCFILE, _ = write_fake_fmri_data_and_design(
        shapes, file_path=tmp_path
    )
    t_r = 10
    slice_time_ref = 0.0
    # basic test based on basic_paradigm and glover hrf
    model = FirstLevelModel(
        t_r=t_r,
        slice_time_ref=slice_time_ref,
        mask_img=mask,
        drift_model="polynomial",
        drift_order=3,
    )
    func_img = load(FUNCFILE[0])
    events = basic_paradigm()
    model = model.fit(func_img, events)

    frame1, X1, names1 = check_design_matrix(model.design_matrices_[0])

    # check design computation is identical
    n_scans = get_data(func_img).shape[3]
    start_time = slice_time_ref * t_r
    end_time = (n_scans - 1 + slice_time_ref) * t_r
    frame_times = np.linspace(start_time, end_time, n_scans)
    design = make_first_level_design_matrix(
        frame_times, events, drift_model="polynomial", drift_order=3
    )

    frame2, X2, names2 = check_design_matrix(design)

    assert_array_equal(frame1, frame2)
    assert_array_equal(X1, X2)
    assert_array_equal(names1, names2)


def test_first_level_glm_computation(tmp_path):
    shapes = ((7, 8, 9, 10),)
    mask, FUNCFILE, _ = write_fake_fmri_data_and_design(
        shapes, file_path=tmp_path
    )
    # basic test based on basic_paradigm and glover hrf
    model = FirstLevelModel(
        t_r=10,
        slice_time_ref=0.0,
        mask_img=mask,
        drift_model="polynomial",
        drift_order=3,
        minimize_memory=False,
    )
    func_img = load(FUNCFILE[0])
    events = basic_paradigm()
    model.fit(func_img, events)


def test_first_level_glm_computation_with_memory_caching(tmp_path):
    shapes = ((7, 8, 9, 10),)
    mask, FUNCFILE, _ = write_fake_fmri_data_and_design(
        shapes, file_path=tmp_path
    )
    # initialize FirstLevelModel with memory option enabled
    model = FirstLevelModel(
        t_r=10.0,
        slice_time_ref=0.0,
        mask_img=mask,
        drift_model="polynomial",
        drift_order=3,
        memory="nilearn_cache",
        memory_level=1,
        minimize_memory=False,
    )
    func_img = load(FUNCFILE[0])
    events = basic_paradigm()
    model.fit(func_img, events)


def test_first_level_from_bids_set_repetition_time_warnings(tmp_path):
    """Raise a warning when there is no bold.json file in the derivatives \
       and no TR value is passed as argument.

    create_fake_bids_dataset does not add JSON files in derivatives,
    so the TR value will be inferred from the raw.
    """
    bids_path = create_fake_bids_dataset(
        base_dir=tmp_path, n_sub=10, n_ses=1, tasks=["main"], n_runs=[1]
    )
    t_r = None
    warning_msg = "No bold.json .* BIDS"
    with pytest.warns(UserWarning, match=warning_msg):
        models, *_ = first_level_from_bids(
            dataset_path=str(tmp_path / bids_path),
            task_label="main",
            space_label="MNI",
            img_filters=[("desc", "preproc")],
            t_r=t_r,
            slice_time_ref=None,
            verbose=1,
        )

        # If no t_r is provided it is inferred from the raw dataset
        # create_fake_bids_dataset generates a dataset
        # with bold data with TR=1.5 secs
        expected_t_r = 1.5
        assert models[0].t_r == expected_t_r


@pytest.mark.parametrize(
    "t_r, error_type, error_msg",
    [
        ("not a number", TypeError, "must be a float"),
        (-1, ValueError, "positive"),
    ],
)
def test_first_level_from_bids_set_repetition_time_errors(
    tmp_path, t_r, error_type, error_msg
):
    """Throw errors for impossible values of TR."""
    bids_path = create_fake_bids_dataset(
        base_dir=tmp_path, n_sub=1, n_ses=1, tasks=["main"], n_runs=[1]
    )

    with pytest.raises(error_type, match=error_msg):
        first_level_from_bids(
            dataset_path=str(tmp_path / bids_path),
            task_label="main",
            space_label="MNI",
            img_filters=[("desc", "preproc")],
            slice_time_ref=None,
            t_r=t_r,
        )


def test_first_level_from_bids_set_slice_timing_ref_warnings(tmp_path):
    """Check that a warning is raised when slice_time_ref is not provided \
    and cannot be inferred from the dataset.

    In this case the model should be created with a slice_time_ref of 0.0.
    """
    bids_path = create_fake_bids_dataset(
        base_dir=tmp_path, n_sub=10, n_ses=1, tasks=["main"], n_runs=[1]
    )

    slice_time_ref = None
    warning_msg = "not provided and cannot be inferred"
    with pytest.warns(UserWarning, match=warning_msg):
        models, *_ = first_level_from_bids(
            dataset_path=str(tmp_path / bids_path),
            task_label="main",
            space_label="MNI",
            img_filters=[("desc", "preproc")],
            slice_time_ref=slice_time_ref,
        )

        expected_slice_time_ref = 0.0
        assert models[0].slice_time_ref == expected_slice_time_ref


@pytest.mark.parametrize(
    "slice_time_ref, error_type, error_msg",
    [
        ("not a number", TypeError, "must be a float"),
        (2, ValueError, "between 0 and 1"),
    ],
)
def test_first_level_from_bids_set_slice_timing_ref_errors(
    tmp_path, slice_time_ref, error_type, error_msg
):
    """Throw errors for impossible values of slice_time_ref."""
    bids_path = create_fake_bids_dataset(
        base_dir=tmp_path, n_sub=1, n_ses=1, tasks=["main"], n_runs=[1]
    )

    with pytest.raises(error_type, match=error_msg):
        first_level_from_bids(
            dataset_path=str(tmp_path / bids_path),
            task_label="main",
            space_label="MNI",
            img_filters=[("desc", "preproc")],
            slice_time_ref=slice_time_ref,
        )


def test_first_level_from_bids_get_metadata_from_derivatives(tmp_path):
    """No warning should be thrown given derivatives have metadata.

    The model created should use the values found in the derivatives.
    """
    bids_path = create_fake_bids_dataset(
        base_dir=tmp_path, n_sub=10, n_ses=1, tasks=["main"], n_runs=[1]
    )

    RepetitionTime = 6.0
    StartTime = 2.0
    add_metadata_to_bids_dataset(
        bids_path=tmp_path / bids_path,
        metadata={"RepetitionTime": RepetitionTime, "StartTime": StartTime},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        models, *_ = first_level_from_bids(
            dataset_path=str(tmp_path / bids_path),
            task_label="main",
            space_label="MNI",
            img_filters=[("desc", "preproc")],
            slice_time_ref=None,
        )
        assert models[0].t_r == RepetitionTime
        assert models[0].slice_time_ref == StartTime / RepetitionTime


def test_first_level_from_bids_get_RepetitionTime_from_derivatives(tmp_path):
    """Only RepetitionTime is provided in derivatives.

    Warning about missing StarTime time in derivatives.
    slice_time_ref cannot be inferred: defaults to 0.
    """
    bids_path = create_fake_bids_dataset(
        base_dir=tmp_path, n_sub=10, n_ses=1, tasks=["main"], n_runs=[1]
    )
    RepetitionTime = 6.0
    add_metadata_to_bids_dataset(
        bids_path=tmp_path / bids_path,
        metadata={"RepetitionTime": RepetitionTime},
    )

    with pytest.warns(UserWarning, match="StartTime' not found in file"):
        models, *_ = first_level_from_bids(
            dataset_path=str(tmp_path / bids_path),
            task_label="main",
            space_label="MNI",
            slice_time_ref=None,
            img_filters=[("desc", "preproc")],
        )
        assert models[0].t_r == 6.0
        assert models[0].slice_time_ref == 0.0


def test_first_level_from_bids_get_StartTime_from_derivatives(tmp_path):
    """Only StartTime is provided in derivatives.

    Warning about missing repetition time in derivatives,
    but RepetitionTime is still read from raw dataset.
    """
    bids_path = create_fake_bids_dataset(
        base_dir=tmp_path, n_sub=10, n_ses=1, tasks=["main"], n_runs=[1]
    )
    StartTime = 1.0
    add_metadata_to_bids_dataset(
        bids_path=tmp_path / bids_path, metadata={"StartTime": StartTime}
    )

    with pytest.warns(UserWarning, match="RepetitionTime' not found in file"):
        models, *_ = first_level_from_bids(
            dataset_path=str(tmp_path / bids_path),
            task_label="main",
            space_label="MNI",
            img_filters=[("desc", "preproc")],
            slice_time_ref=None,
        )

        # create_fake_bids_dataset generates a dataset
        # with bold data with TR=1.5 secs
        assert models[0].t_r == 1.5
        assert models[0].slice_time_ref == StartTime / 1.5


def test_first_level_contrast_computation(tmp_path):
    shapes = ((7, 8, 9, 10),)
    mask, FUNCFILE, _ = write_fake_fmri_data_and_design(
        shapes, file_path=tmp_path
    )

    # Ordinary Least Squares case
    model = FirstLevelModel(
        t_r=10.0,
        slice_time_ref=0.0,
        mask_img=mask,
        drift_model="polynomial",
        drift_order=3,
        minimize_memory=False,
    )
    c1, c2, cnull = np.eye(7)[0], np.eye(7)[1], np.zeros(7)

    # fit model
    # basic test based on basic_paradigm and glover hrf
    events = basic_paradigm()
    func_img = load(FUNCFILE[0])
    model = model.fit([func_img, func_img], [events, events])

    # smoke test for different contrasts in fixed effects
    model.compute_contrast([c1, c2])

    # smoke test for same contrast in fixed effects
    model.compute_contrast([c2, c2])

    # smoke test for contrast that will be repeated
    model.compute_contrast(c2)
    model.compute_contrast(c2, "F")
    model.compute_contrast(c2, "t", "z_score")
    model.compute_contrast(c2, "t", "stat")
    model.compute_contrast(c2, "t", "p_value")
    model.compute_contrast(c2, None, "effect_size")
    model.compute_contrast(c2, None, "effect_variance")

    # formula should work (passing variable name directly)
    model.compute_contrast("c0")
    model.compute_contrast("c1")
    model.compute_contrast("c2")

    # smoke test for one null contrast in group
    model.compute_contrast([c2, cnull])


def test_first_level_contrast_computation_errors(tmp_path):
    shapes = ((7, 8, 9, 10),)
    mask, FUNCFILE, _ = write_fake_fmri_data_and_design(
        shapes, file_path=tmp_path
    )

    # Ordinary Least Squares case
    model = FirstLevelModel(
        t_r=10.0,
        slice_time_ref=0.0,
        mask_img=mask,
        drift_model="polynomial",
        drift_order=3,
        minimize_memory=False,
    )
    c1, cnull = np.eye(7)[0], np.zeros(7)

    # asking for contrast before model fit gives error
    with pytest.raises(ValueError, match="The model has not been fit yet"):
        model.compute_contrast(c1)

    # fit model
    # basic test based on basic_paradigm and glover hrf
    events = basic_paradigm()
    func_img = load(FUNCFILE[0])
    model = model.fit([func_img, func_img], [events, events])

    # Check that an error is raised for invalid contrast_def
    with pytest.raises(
        ValueError, match="contrast_def must be an array or str or list"
    ):
        model.compute_contrast(37)

    # only passing null contrasts should give back a value error
    with pytest.raises(
        ValueError, match="All contrasts provided were null contrasts."
    ):
        model.compute_contrast(cnull)
    with pytest.raises(
        ValueError, match="All contrasts provided were null contrasts."
    ):
        model.compute_contrast([cnull, cnull])

    # passing wrong parameters
    match = ".* contrasts given, while there are .* runs."
    with pytest.raises(ValueError, match=match):
        model.compute_contrast([c1, c1, c1])
    with pytest.raises(ValueError, match=match):
        model.compute_contrast([])

    match = "output_type must be one of "
    with pytest.raises(ValueError, match=match):
        model.compute_contrast(c1, "", "")
    with pytest.raises(ValueError, match=match):
        model.compute_contrast(c1, "", [])

    with pytest.raises(
        ValueError,
        match="t contrasts cannot be empty",
    ):
        model.compute_contrast([c1, []])


def test_first_level_with_scaling(affine_eye):
    shapes, rk = [(3, 1, 1, 2)], 1
    fmri_data = [Nifti1Image(np.zeros((1, 1, 1, 2)) + 6, affine_eye)]
    design_matrices = [
        pd.DataFrame(
            np.ones((shapes[0][-1], rk)),
            columns=list("abcdefghijklmnopqrstuvwxyz")[:rk],
        )
    ]
    fmri_glm = FirstLevelModel(
        mask_img=False,
        noise_model="ols",
        signal_scaling=0,
        minimize_memory=True,
    )
    assert fmri_glm.signal_scaling == 0
    assert not fmri_glm.standardize
    with pytest.warns(
        DeprecationWarning, match="Deprecated. `scaling_axis` will be removed"
    ):
        assert fmri_glm.scaling_axis == 0
    glm_parameters = fmri_glm.get_params()
    test_glm = FirstLevelModel(**glm_parameters)
    fmri_glm = fmri_glm.fit(fmri_data, design_matrices=design_matrices)
    test_glm = test_glm.fit(fmri_data, design_matrices=design_matrices)
    assert glm_parameters["signal_scaling"] == 0


def test_first_level_with_no_signal_scaling(affine_eye):
    """Test to ensure that the FirstLevelModel works correctly \
       with a signal_scaling==False.

    In particular, that derived theta are correct for a
    constant design matrix with a single valued fmri image
    """
    # Check error with invalid signal_scaling values
    with pytest.raises(ValueError, match="signal_scaling must be"):
        FirstLevelModel(
            mask_img=False, noise_model="ols", signal_scaling="foo"
        )

    shapes, rk = [(3, 1, 1, 2)], 1
    design_matrices = [
        pd.DataFrame(
            np.ones((shapes[0][-1], rk)),
            columns=list("abcdefghijklmnopqrstuvwxyz")[:rk],
        )
    ]
    fmri_data = [Nifti1Image(np.zeros((1, 1, 1, 2)) + 6, affine_eye)]
    first_level = FirstLevelModel(
        mask_img=False, noise_model="ols", signal_scaling=False
    )

    first_level.fit(fmri_data, design_matrices=design_matrices)
    # trivial test of signal_scaling value
    assert first_level.signal_scaling is False
    # assert that our design matrix has one constant
    assert first_level.design_matrices_[0].equals(
        pd.DataFrame([1.0, 1.0], columns=["a"])
    )
    # assert that we only have one theta as there is only on voxel in our image
    assert first_level.results_[0][0].theta.shape == (1, 1)
    # assert that the theta is equal to the one voxel value
    assert_almost_equal(first_level.results_[0][0].theta[0, 0], 6.0, 2)


def test_first_level_residuals():
    shapes, rk = [(10, 10, 10, 100)], 3
    mask, fmri_data, design_matrices = generate_fake_fmri_data_and_design(
        shapes, rk
    )

    for i in range(len(design_matrices)):
        design_matrices[i][design_matrices[i].columns[0]] = 1

    model = FirstLevelModel(
        mask_img=mask, minimize_memory=False, noise_model="ols"
    )

    model.fit(fmri_data, design_matrices=design_matrices)

    residuals = model.residuals[0]
    mean_residuals = model.masker_.transform(residuals).mean(0)

    assert_array_almost_equal(mean_residuals, 0)


def test_first_level_residuals_errors():
    shapes, rk = [(10, 10, 10, 100)], 3
    mask, fmri_data, design_matrices = generate_fake_fmri_data_and_design(
        shapes, rk
    )

    for i in range(len(design_matrices)):
        design_matrices[i][design_matrices[i].columns[0]] = 1

    # Check that voxelwise model attributes cannot be
    # accessed if minimize_memory is set to True
    model = FirstLevelModel(
        mask_img=mask, minimize_memory=True, noise_model="ols"
    )
    model.fit(fmri_data, design_matrices=design_matrices)

    with pytest.raises(ValueError, match="To access voxelwise attributes"):
        model.residuals[0]

    model = FirstLevelModel(
        mask_img=mask, minimize_memory=False, noise_model="ols"
    )

    # Check that trying to access residuals without fitting
    # raises an error
    with pytest.raises(ValueError, match="The model has not been fit yet"):
        model.residuals[0]

    model.fit(fmri_data, design_matrices=design_matrices)

    # For coverage
    with pytest.raises(ValueError, match="attribute must be one of"):
        model._get_voxelwise_model_attribute("foo", True)


@pytest.mark.parametrize(
    "shapes",
    [
        [(10, 10, 10, 25)],
        [(10, 10, 10, 25), (10, 10, 10, 100)],
    ],
)
def test_get_voxelwise_attributes_should_return_as_many_as_design_matrices(
    shapes,
):
    mask, fmri_data, design_matrices = generate_fake_fmri_data_and_design(
        shapes
    )

    for i in range(len(design_matrices)):
        design_matrices[i][design_matrices[i].columns[0]] = 1

    model = FirstLevelModel(
        mask_img=mask, minimize_memory=False, noise_model="ols"
    )
    model.fit(fmri_data, design_matrices=design_matrices)

    # Check that length of outputs is the same as the number of design matrices
    assert len(model._get_voxelwise_model_attribute("residuals", True)) == len(
        shapes
    )


def test_first_level_predictions_r_square():
    shapes, rk = [(10, 10, 10, 25)], 3
    mask, fmri_data, design_matrices = generate_fake_fmri_data_and_design(
        shapes, rk
    )

    for i in range(len(design_matrices)):
        design_matrices[i][design_matrices[i].columns[0]] = 1

    model = FirstLevelModel(
        mask_img=mask,
        signal_scaling=False,
        minimize_memory=False,
        noise_model="ols",
    )
    model.fit(fmri_data, design_matrices=design_matrices)

    pred = model.predicted[0]
    data = fmri_data[0]
    r_square_3d = model.r_square[0]

    y_predicted = model.masker_.transform(pred)
    y_measured = model.masker_.transform(data)

    assert_almost_equal(np.mean(y_predicted - y_measured), 0)

    r_square_2d = model.masker_.transform(r_square_3d)
    assert_array_less(0.0, r_square_2d)


@pytest.mark.parametrize(
    "hrf_model",
    [
        "spm",
        "spm + derivative",
        "glover",
        lambda tr, ov: np.ones(int(tr * ov)),
    ],
)
@pytest.mark.parametrize("spaces", [False, True])
def test_first_level_hrf_model(hrf_model, spaces):
    """Ensure that FirstLevelModel runs without raising errors \
       for different values of hrf_model.

    In particular, one checks that it runs
    without raising errors when given a custom response function.
    When :meth:`~nilearn.glm.first_level.FirstLevelModel.compute_contrast`
    is used errors should be raised when event (ie condition) names are not
    valid identifiers.
    """
    shapes, rk = [(10, 10, 10, 25)], 3
    mask, fmri_data, _ = generate_fake_fmri_data_and_design(shapes, rk)

    events = basic_paradigm(condition_names_have_spaces=spaces)

    model = FirstLevelModel(t_r=2.0, mask_img=mask, hrf_model=hrf_model)

    model.fit(fmri_data, events)

    columns = model.design_matrices_[0].columns
    exp = f"{columns[0]}-{columns[1]}"
    try:
        model.compute_contrast(exp)
    except Exception:
        with pytest.raises(ValueError, match="invalid python identifiers"):
            model.compute_contrast(exp)


def test_glm_sample_mask():
    """Ensure the sample mask is performing correctly in GLM."""
    shapes, rk = [(10, 10, 10, 25)], 3
    mask, fmri_data, design_matrix = generate_fake_fmri_data_and_design(
        shapes, rk
    )
    model = FirstLevelModel(t_r=2.0, mask_img=mask, minimize_memory=False)
    sample_mask = np.arange(25)[3:]  # censor the first three volumes
    model.fit(
        fmri_data, design_matrices=design_matrix, sample_masks=sample_mask
    )

    assert model.design_matrices_[0].shape[0] == 22
    assert model.predicted[0].shape[-1] == 22


"""Test the first level model on BIDS datasets."""


def _inputs_for_new_bids_dataset():
    n_sub = 2
    n_ses = 2
    tasks = ["main"]
    n_runs = [2]
    return n_sub, n_ses, tasks, n_runs


@pytest.fixture(scope="session")
def bids_dataset(tmp_path_factory):
    """Create a fake BIDS dataset for testing purposes.

    Only use if the dataset does not need to me modified.
    """
    base_dir = tmp_path_factory.mktemp("bids")
    n_sub, n_ses, tasks, n_runs = _inputs_for_new_bids_dataset()
    return create_fake_bids_dataset(
        base_dir=base_dir, n_sub=n_sub, n_ses=n_ses, tasks=tasks, n_runs=n_runs
    )


def _new_bids_dataset(base_dir=Path()):
    """Create a new BIDS dataset for testing purposes.

    Use if the dataset needs to be modified after creation.
    """
    n_sub, n_ses, tasks, n_runs = _inputs_for_new_bids_dataset()
    return create_fake_bids_dataset(
        base_dir=base_dir, n_sub=n_sub, n_ses=n_ses, tasks=tasks, n_runs=n_runs
    )


@pytest.mark.parametrize("n_runs", ([1, 0], [1, 1], [1, 2]))
@pytest.mark.parametrize("n_ses", [0, 1, 2])
@pytest.mark.parametrize("task_index", [0, 1])
@pytest.mark.parametrize("space_label", ["MNI", "T1w"])
def test_first_level_from_bids(
    tmp_path, n_runs, n_ses, task_index, space_label
):
    """Test several BIDS structure."""
    n_sub = 2
    tasks = ["localizer", "main"]

    bids_path = create_fake_bids_dataset(
        base_dir=tmp_path, n_sub=n_sub, n_ses=n_ses, tasks=tasks, n_runs=n_runs
    )

    models, imgs, events, confounds = first_level_from_bids(
        dataset_path=bids_path,
        task_label=tasks[task_index],
        space_label=space_label,
        img_filters=[("desc", "preproc")],
        slice_time_ref=None,
    )

    _check_output_first_level_from_bids(n_sub, models, imgs, events, confounds)

    n_imgs_expected = n_ses * n_runs[task_index]

    # no run entity in filename or session level
    # when they take a value of 0 when generating a dataset
    no_run_entity = n_runs[task_index] <= 1
    no_session_level = n_ses <= 1

    if no_session_level:
        n_imgs_expected = 1 if no_run_entity else n_runs[task_index]
    elif no_run_entity:
        n_imgs_expected = n_ses

    assert len(imgs[0]) == n_imgs_expected


def test_first_level_from_bids_select_one_run_per_session(bids_dataset):
    n_sub, n_ses, *_ = _inputs_for_new_bids_dataset()

    models, imgs, events, confounds = first_level_from_bids(
        dataset_path=bids_dataset,
        task_label="main",
        space_label="MNI",
        img_filters=[("run", "01"), ("desc", "preproc")],
        slice_time_ref=None,
    )

    _check_output_first_level_from_bids(n_sub, models, imgs, events, confounds)

    n_imgs_expected = n_ses
    assert len(imgs[0]) == n_imgs_expected


def test_first_level_from_bids_select_all_runs_of_one_session(bids_dataset):
    n_sub, _, _, n_runs = _inputs_for_new_bids_dataset()

    models, imgs, events, confounds = first_level_from_bids(
        dataset_path=bids_dataset,
        task_label="main",
        space_label="MNI",
        img_filters=[("ses", "01"), ("desc", "preproc")],
        slice_time_ref=None,
    )

    _check_output_first_level_from_bids(n_sub, models, imgs, events, confounds)

    n_imgs_expected = n_runs[0]
    assert len(imgs[0]) == n_imgs_expected


@pytest.mark.parametrize("verbose", [0, 1])
def test_first_level_from_bids_smoke_test_for_verbose_argument(
    bids_dataset, verbose
):
    first_level_from_bids(
        dataset_path=bids_dataset,
        task_label="main",
        space_label="MNI",
        img_filters=[("desc", "preproc")],
        verbose=verbose,
        slice_time_ref=None,
    )


@pytest.mark.parametrize(
    "entity", ["acq", "ce", "dir", "rec", "echo", "res", "den"]
)
def test_first_level_from_bids_several_labels_per_entity(tmp_path, entity):
    """Correct files selected when an entity has several possible labels.

    Regression test for https://github.com/nilearn/nilearn/issues/3524
    """
    n_sub = 1
    n_ses = 1
    tasks = ["main"]
    n_runs = [1]

    bids_path = create_fake_bids_dataset(
        base_dir=tmp_path,
        n_sub=n_sub,
        n_ses=n_ses,
        tasks=tasks,
        n_runs=n_runs,
        entities={entity: ["A", "B"]},
    )

    models, imgs, events, confounds = first_level_from_bids(
        dataset_path=bids_path,
        task_label="main",
        space_label="MNI",
        img_filters=[("desc", "preproc"), (entity, "A")],
        slice_time_ref=None,
    )

    _check_output_first_level_from_bids(n_sub, models, imgs, events, confounds)
    n_imgs_expected = n_ses * n_runs[0]
    assert len(imgs[0]) == n_imgs_expected


def _check_output_first_level_from_bids(
    n_sub, models, imgs, events, confounds
):
    assert len(models) == n_sub
    assert all(isinstance(model, FirstLevelModel) for model in models)
    assert len(models) == len(imgs)
    for img_ in imgs:
        assert isinstance(img_, list)
        assert all(Path(x).exists() for x in img_)
    assert len(models) == len(events)
    for event_ in events:
        assert isinstance(event_, list)
        assert all(isinstance(x, pd.DataFrame) for x in event_)
    assert len(models) == len(confounds)
    for confound_ in confounds:
        assert isinstance(confound_, list)
        assert all(isinstance(x, pd.DataFrame) for x in confound_)


def test_first_level_from_bids_with_subject_labels(bids_dataset):
    """Test that the subject labels arguments works \
    with proper warning for missing subjects.

    Check that the incorrect label `foo` raises a warning,
    but that we still get a model for existing subject.
    """
    warning_message = "Subject label 'foo' is not present in*"
    with pytest.warns(UserWarning, match=warning_message):
        models, *_ = first_level_from_bids(
            dataset_path=bids_dataset,
            task_label="main",
            sub_labels=["foo", "01"],
            space_label="MNI",
            img_filters=[("desc", "preproc")],
            slice_time_ref=None,
        )

        assert models[0].subject_label == "01"


def test_first_level_from_bids_no_duplicate_sub_labels(bids_dataset):
    """Make sure that if a subject label is repeated, \
    only one model is created.

    See https://github.com/nilearn/nilearn/issues/3585
    """
    models, *_ = first_level_from_bids(
        dataset_path=bids_dataset,
        task_label="main",
        sub_labels=["01", "01"],
        space_label="MNI",
        img_filters=[("desc", "preproc")],
        slice_time_ref=None,
    )

    assert len(models) == 1


def test_first_level_from_bids_validation_input_dataset_path():
    with pytest.raises(TypeError, match="must be a string or pathlike"):
        first_level_from_bids(
            dataset_path=2,
            task_label="main",
            space_label="MNI",
            slice_time_ref=None,
        )
    with pytest.raises(ValueError, match="'dataset_path' does not exist"):
        first_level_from_bids(
            dataset_path="lolo",
            task_label="main",
            space_label="MNI",
            slice_time_ref=None,
        )
    with pytest.raises(TypeError, match="derivatives_.* must be a string"):
        first_level_from_bids(
            dataset_path=Path(),
            task_label="main",
            space_label="MNI",
            derivatives_folder=1,
            slice_time_ref=None,
        )


@pytest.mark.parametrize(
    "task_label, error_type",
    [(42, TypeError), ("$$$", ValueError)],
)
def test_first_level_from_bids_validation_task_label(
    bids_dataset, task_label, error_type
):
    with pytest.raises(error_type, match="All bids labels must be "):
        first_level_from_bids(
            dataset_path=bids_dataset, task_label=task_label, space_label="MNI"
        )


@pytest.mark.parametrize(
    "sub_labels, error_type, error_msg",
    [
        ("42", TypeError, "must be a list"),
        (["1", 1], TypeError, "must be string"),
        ([1], TypeError, "must be string"),
    ],
)
def test_first_level_from_bids_validation_sub_labels(
    bids_dataset, sub_labels, error_type, error_msg
):
    with pytest.raises(error_type, match=error_msg):
        first_level_from_bids(
            dataset_path=bids_dataset,
            task_label="main",
            sub_labels=sub_labels,
            slice_time_ref=None,
        )


@pytest.mark.parametrize(
    "space_label, error_type",
    [(42, TypeError), ("$$$", ValueError)],
)
def test_first_level_from_bids_validation_space_label(
    bids_dataset, space_label, error_type
):
    with pytest.raises(error_type, match="All bids labels must be "):
        first_level_from_bids(
            dataset_path=bids_dataset,
            task_label="main",
            space_label=space_label,
            slice_time_ref=None,
        )


@pytest.mark.parametrize(
    "img_filters, error_type,match",
    [
        ("foo", TypeError, "'img_filters' must be a list"),
        ([(1, 2)], TypeError, "Filters in img"),
        ([("desc", "*/-")], ValueError, "bids labels must be alphanumeric."),
        ([("foo", "bar")], ValueError, "is not a possible filter."),
    ],
)
def test_first_level_from_bids_validation_img_filter(
    bids_dataset, img_filters, error_type, match
):
    with pytest.raises(error_type, match=match):
        first_level_from_bids(
            dataset_path=bids_dataset,
            task_label="main",
            img_filters=img_filters,
            slice_time_ref=None,
        )


def test_first_level_from_bids_too_many_bold_files(bids_dataset):
    """Too many bold files if img_filters is underspecified, \
       should raise an error.

    Here there is a desc-preproc and desc-fmriprep image for the space-T1w.
    """
    with pytest.raises(ValueError, match="Too many images found"):
        first_level_from_bids(
            dataset_path=bids_dataset,
            task_label="main",
            space_label="T1w",
            slice_time_ref=None,
        )


def test_first_level_from_bids_with_missing_events(tmp_path_factory):
    """All events.tsv files are missing, should raise an error."""
    bids_dataset = _new_bids_dataset(tmp_path_factory.mktemp("no_events"))
    events_files = get_bids_files(main_path=bids_dataset, file_tag="events")
    for f in events_files:
        os.remove(f)

    with pytest.raises(ValueError, match="No events.tsv files found"):
        first_level_from_bids(
            dataset_path=bids_dataset,
            task_label="main",
            space_label="MNI",
            slice_time_ref=None,
        )


def test_first_level_from_bids_no_tr(tmp_path_factory):
    """Throw warning when t_r information cannot be inferred from the data \
    and t_r=None is passed."""
    bids_dataset = _new_bids_dataset(tmp_path_factory.mktemp("no_events"))
    json_files = get_bids_files(
        main_path=bids_dataset, file_tag="bold", file_type="json"
    )
    for f in json_files:
        os.remove(f)

    with pytest.warns(
        UserWarning, match="'t_r' not provided and cannot be inferred"
    ):
        first_level_from_bids(
            dataset_path=bids_dataset,
            task_label="main",
            space_label="MNI",
            slice_time_ref=None,
            t_r=None,
        )


def test_first_level_from_bids_no_bold_file(tmp_path_factory):
    bids_dataset = _new_bids_dataset(tmp_path_factory.mktemp("no_bold"))
    imgs = get_bids_files(
        main_path=bids_dataset / "derivatives",
        file_tag="bold",
        file_type="*gz",
    )
    for img_ in imgs:
        os.remove(img_)

    with pytest.raises(ValueError, match="No BOLD files found "):
        first_level_from_bids(
            dataset_path=bids_dataset,
            task_label="main",
            space_label="MNI",
            slice_time_ref=None,
        )


def test_first_level_from_bids_with_one_events_missing(tmp_path_factory):
    """Only one events.tsv file is missing, should raise an error."""
    bids_dataset = _new_bids_dataset(
        tmp_path_factory.mktemp("one_event_missing")
    )
    events_files = get_bids_files(main_path=bids_dataset, file_tag="events")
    os.remove(events_files[0])

    with pytest.raises(ValueError, match="Same number of event files "):
        first_level_from_bids(
            dataset_path=bids_dataset,
            task_label="main",
            space_label="MNI",
            slice_time_ref=None,
        )


def test_first_level_from_bids_one_confound_missing(tmp_path_factory):
    """There must be only one confound file per image or none.

    If only one is missing, it should raise an error.
    """
    bids_dataset = _new_bids_dataset(
        tmp_path_factory.mktemp("one_confound_missing")
    )
    confound_files = get_bids_files(
        main_path=bids_dataset / "derivatives",
        file_tag="desc-confounds_timeseries",
    )
    os.remove(confound_files[-1])

    with pytest.raises(ValueError, match="Same number of confound"):
        first_level_from_bids(
            dataset_path=bids_dataset,
            task_label="main",
            space_label="MNI",
            slice_time_ref=None,
        )


def test_first_level_from_bids_all_confounds_missing(tmp_path_factory):
    """If all confound files are missing, \
    confounds should be an array of None."""
    bids_dataset = _new_bids_dataset(tmp_path_factory.mktemp("no_confounds"))
    confound_files = get_bids_files(
        main_path=bids_dataset / "derivatives",
        file_tag="desc-confounds_timeseries",
    )
    for f in confound_files:
        os.remove(f)

    models, imgs, events, confounds = first_level_from_bids(
        dataset_path=bids_dataset,
        task_label="main",
        space_label="MNI",
        img_filters=[("desc", "preproc")],
        verbose=0,
        slice_time_ref=None,
    )

    assert len(models) == len(imgs)
    assert len(models) == len(events)
    assert len(models) == len(confounds)
    for condounds_ in confounds:
        assert condounds_ is None


def test_first_level_from_bids_no_derivatives(tmp_path):
    """Raise error if the derivative folder does not exist."""
    bids_path = create_fake_bids_dataset(
        base_dir=tmp_path,
        n_sub=1,
        n_ses=1,
        tasks=["main"],
        n_runs=[1],
        with_derivatives=False,
    )
    with pytest.raises(ValueError, match="derivatives folder not found"):
        first_level_from_bids(
            dataset_path=bids_path,
            task_label="main",
            space_label="MNI",
            slice_time_ref=None,
        )


def test_first_level_from_bids_no_session(tmp_path):
    """Check runs are not repeated when ses field is not used."""
    bids_path = create_fake_bids_dataset(
        base_dir=tmp_path, n_sub=3, n_ses=0, tasks=["main"], n_runs=[2]
    )
    # repeated run entity error
    # when run entity is in filenames and not ses
    # can arise when desc or space is present and not specified
    with pytest.raises(ValueError, match="Too many images found"):
        first_level_from_bids(
            dataset_path=bids_path,
            task_label="main",
            space_label="T1w",
            slice_time_ref=None,
        )


def test_first_level_from_bids_mismatch_run_index(tmp_path_factory):
    """Test error when run index is zero padded in raw but not in derivatives.

    Regression test for https://github.com/nilearn/nilearn/issues/3029

    """
    bids_dataset = _new_bids_dataset(tmp_path_factory.mktemp("renamed_runs"))
    files_to_rename = (bids_dataset / "derivatives").glob(
        "**/func/*_task-main_*desc-*"
    )
    for file_ in files_to_rename:
        new_file = file_.parent / file_.name.replace("run-0", "run-")
        file_.rename(new_file)

    with pytest.raises(ValueError, match=".*events.tsv files.*"):
        first_level_from_bids(
            dataset_path=bids_dataset,
            task_label="main",
            space_label="MNI",
            img_filters=[("desc", "preproc")],
            slice_time_ref=None,
        )


def test_first_level_from_bids_deprecated_slice_time_default(bids_dataset):
    with pytest.deprecated_call(match="slice_time_ref will default to None."):
        first_level_from_bids(
            dataset_path=bids_dataset,
            task_label="main",
            space_label="MNI",
            img_filters=[("desc", "preproc")],
            slice_time_ref=0,
        )


def test_slice_time_ref_warning_only_when_not_provided(bids_dataset):
    # catch all warnings
    with pytest.warns() as record:
        first_level_from_bids(
            dataset_path=bids_dataset,
            task_label="main",
            space_label="MNI",
            img_filters=[("desc", "preproc")],
            slice_time_ref=0.6,
            verbose=0,
        )

    # check that no warnings were raised
    for r in record:
        assert "'slice_time_ref' not provided" not in r.message.args[0]


def test_check_trial_type_warning(tmp_path):
    """Check that warning is thrown when an events file has no trial_type."""
    events = pd.DataFrame({"onset": [0, 1, 2], "duration": [1, 1, 1]})
    event_file = tmp_path / "events.tsv"
    events.to_csv(event_file, sep="\t", index=False)
    with pytest.warns(UserWarning, match="No column named 'trial_type' found"):
        _check_trial_type([event_file])


def test_missing_trial_type_column_warning(tmp_path_factory):
    """Check that warning is thrown when an events file has no trial_type.

    Ensure that the warning is thrown when running first_level_from_bids.
    """
    bids_dataset = _new_bids_dataset(
        tmp_path_factory.mktemp("one_event_missing")
    )
    events_files = get_bids_files(main_path=bids_dataset, file_tag="events")
    # remove trial type column from one events.tsv file
    events = pd.read_csv(events_files[0], sep="\t")
    events.drop(columns="trial_type", inplace=True)
    events.to_csv(events_files[0], sep="\t", index=False)

    with pytest.warns() as record:
        first_level_from_bids(
            dataset_path=bids_dataset,
            task_label="main",
            space_label="MNI",
            slice_time_ref=None,
        )
        assert any(
            "No column named 'trial_type' found" in r.message.args[0]
            for r in record
        )


def test_first_level_from_bids_load_confounds(tmp_path):
    """Test that only a subset of confounds can be loaded."""
    n_sub = 2

    bids_path = create_fake_bids_dataset(
        base_dir=tmp_path, n_sub=n_sub, n_ses=2, tasks=["main"], n_runs=[2]
    )

    _, _, _, confounds = first_level_from_bids(
        dataset_path=bids_path,
        task_label="main",
        space_label="MNI",
        img_filters=[("desc", "preproc")],
    )

    assert len(confounds[0][0].columns) == 189

    models, imgs, events, confounds = first_level_from_bids(
        dataset_path=bids_path,
        task_label="main",
        space_label="MNI",
        img_filters=[("desc", "preproc")],
        confounds_strategy=("motion", "wm_csf"),
        confounds_motion="full",
        confounds_wm_csf="basic",
    )

    _check_output_first_level_from_bids(n_sub, models, imgs, events, confounds)

    assert len(confounds[0][0].columns) == 26

    assert all(x in confounds[0][0].columns for x in ["csf", "white_matter"])
    for dir, motion, der, power in product(
        ["x", "y", "z"],
        ["rot", "trans"],
        ["", "_derivative1"],
        ["", "_power2"],
    ):
        assert f"{motion}_{dir}{der}{power}" in confounds[0][0].columns


def test_first_level_from_bids_load_confounds_warnings(tmp_path):
    """Throw warning when incompatible confound loading strategy are used."""
    n_sub = 2

    bids_path = create_fake_bids_dataset(
        base_dir=tmp_path, n_sub=n_sub, n_ses=2, tasks=["main"], n_runs=[2]
    )

    # high pass is loaded from the confounds: no warning
    first_level_from_bids(
        dataset_path=bids_path,
        task_label="main",
        space_label="MNI",
        img_filters=[("desc", "preproc")],
        drift_model=None,
        confounds_strategy=("high_pass",),
    )

    with pytest.warns(
        UserWarning, match=("duplicate .*the cosine one used in the model.")
    ):
        # cosine loaded from confounds may duplicate
        # the one created during model specification
        first_level_from_bids(
            dataset_path=bids_path,
            task_label="main",
            space_label="MNI",
            img_filters=[("desc", "preproc")],
            drift_model="cosine",
            confounds_strategy=("high_pass",),
        )

    with pytest.warns(
        UserWarning, match=("conflict .*the polynomial one used in the model.")
    ):
        # cosine loaded from confounds may conflict
        # the one created during model specification
        first_level_from_bids(
            dataset_path=bids_path,
            task_label="main",
            space_label="MNI",
            img_filters=[("desc", "preproc")],
            drift_model="polynomial",
            confounds_strategy=("high_pass",),
        )


def test_first_level_from_bids_no_subject(tmp_path):
    """Throw error when no subject found."""
    bids_path = create_fake_bids_dataset(
        base_dir=tmp_path, n_sub=1, n_ses=0, tasks=["main"], n_runs=[2]
    )
    shutil.rmtree(bids_path / "derivatives" / "sub-01")
    with pytest.raises(RuntimeError, match="No subject found in:"):
        first_level_from_bids(
            dataset_path=bids_path,
            task_label="main",
            space_label="MNI",
            slice_time_ref=None,
        )


def test_check_run_tables_errors():
    # check high level wrapper keeps behavior
    with pytest.raises(ValueError, match="len.* does not match len.*"):
        _check_run_tables([""] * 2, [""], "")
    with pytest.raises(ValueError, match="table path .* could not be loaded"):
        _check_run_tables([""] * 2, [".csv", ".csv"], "")
    with pytest.raises(
        TypeError, match="can only be a pandas DataFrames or a string"
    ):
        _check_run_tables([""] * 2, [[0], pd.DataFrame()], "")
    with pytest.raises(ValueError, match="table path .* could not be loaded"):
        _check_run_tables([""] * 2, [".csv", pd.DataFrame()], "")


def test_img_table_checks():
    # check matching lengths
    with pytest.raises(ValueError, match="len.* does not match len.*"):
        _check_list_length_match([""] * 2, [""], "", "")

    # check tables type and that can be loaded
    with pytest.raises(ValueError, match="table path .* could not be loaded"):
        _check_and_load_tables([".csv", ".csv"], "")
    with pytest.raises(
        TypeError, match="can only be a pandas DataFrames or a string"
    ):
        _check_and_load_tables([[], pd.DataFrame()], "")
    with pytest.raises(ValueError, match="table path .* could not be loaded"):
        _check_and_load_tables([".csv", pd.DataFrame()], "")
