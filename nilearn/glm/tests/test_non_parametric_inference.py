"""Test non_parametric_inference."""

import numpy as np
import pandas as pd
import pytest
from nibabel import Nifti1Image, load
from numpy.testing import assert_array_equal
from scipy import stats

from nilearn._utils.data_gen import (
    generate_fake_fmri_data_and_design,
    write_fake_fmri_data_and_design,
)
from nilearn.exceptions import NotImplementedWarning
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.second_level import non_parametric_inference
from nilearn.glm.tests.conftest import SHAPE, fake_fmri_data
from nilearn.image import concat_imgs, get_data, new_img_like, smooth_img
from nilearn.maskers import NiftiMasker, SurfaceMasker
from nilearn.reporting import get_clusters_table

N_PERM = 5


@pytest.mark.slow
def test_cluster_level_with_covariates(rng, n_subjects):
    """Test non-parametric inference with cluster-level inference in \
    the context of covariates.
    """
    mask, fmri_data = fake_fmri_data()

    unc_pval = 0.1

    # Set up one sample t-test design with two random covariates
    cov1 = rng.random(n_subjects)
    cov2 = rng.random(n_subjects)
    X = pd.DataFrame({"cov1": cov1, "cov2": cov2, "intercept": 1})

    # make sure there is variability in the images
    kernels = rng.uniform(low=0, high=5, size=n_subjects)
    Y = [smooth_img(fmri_data[0], kernel) for kernel in kernels]

    # Set up non-parametric test
    out = non_parametric_inference(
        Y,
        design_matrix=X,
        mask=mask,
        model_intercept=False,
        second_level_contrast="intercept",
        n_perm=int(1 / unc_pval),
        threshold=unc_pval,
    )

    # Calculate uncorrected cluster sizes
    df = len(Y) - X.shape[1]
    neg_log_pval = -np.log10(stats.t.sf(get_data(out["t"]), df=df))
    logp_unc = new_img_like(out["t"], neg_log_pval)
    logp_unc_cluster_sizes = list(
        get_clusters_table(logp_unc, -np.log10(unc_pval))["Cluster Size (mm3)"]
    )

    # Calculate corrected cluster sizes
    logp_max_cluster_sizes = list(
        get_clusters_table(out["logp_max_size"], unc_pval)[
            "Cluster Size (mm3)"
        ]
    )

    # Compare cluster sizes
    logp_unc_cluster_sizes.sort()
    logp_max_cluster_sizes.sort()
    assert logp_unc_cluster_sizes == logp_max_cluster_sizes


@pytest.mark.slow
def test_cluster_level_with_single_covariates(rng, n_subjects):
    """Test non-parametric inference with cluster-level inference in \
    the context of covariates.
    """
    mask, fmri_data = fake_fmri_data()

    unc_pval = 0.1

    # make sure there is variability in the images
    kernels = rng.uniform(low=0, high=5, size=n_subjects)
    Y = [smooth_img(fmri_data[0], kernel) for kernel in kernels]

    # Test single covariate
    X = pd.DataFrame({"intercept": [1] * len(Y)})
    non_parametric_inference(
        Y,
        design_matrix=X,
        mask=mask,
        model_intercept=False,
        second_level_contrast="intercept",
        n_perm=N_PERM,
        threshold=unc_pval,
    )


@pytest.mark.slow
def test_cluster_level(n_subjects):
    """Test non-parametric inference with cluster-level inference."""
    func_img, mask = fake_fmri_data()

    Y = [func_img] * n_subjects
    X = pd.DataFrame([[1]] * n_subjects, columns=["intercept"])

    out = non_parametric_inference(
        Y,
        design_matrix=X,
        model_intercept=False,
        mask=mask,
        n_perm=N_PERM,
        threshold=0.001,
    )
    assert isinstance(out, dict)
    assert "t" in out
    assert "logp_max_t" in out
    assert "size" in out
    assert "logp_max_size" in out
    assert "mass" in out
    assert "logp_max_mass" in out

    assert get_data(out["logp_max_t"]).shape == SHAPE[:3]


@pytest.mark.slow
def test_permutation_computation(n_subjects):
    """Check shape of computed output."""
    func_img, mask = fake_fmri_data()

    Y = [func_img] * n_subjects
    X = pd.DataFrame([[1]] * n_subjects, columns=["intercept"])

    neg_log_pvals_img = non_parametric_inference(
        Y, design_matrix=X, model_intercept=False, mask=mask, n_perm=N_PERM
    )

    assert get_data(neg_log_pvals_img).shape == SHAPE[:3]


@pytest.mark.slow
def test_tfce(n_subjects):
    """Test non-parametric inference with TFCE inference."""
    mask, fmri_data = fake_fmri_data()
    Y = [fmri_data] * n_subjects
    X = pd.DataFrame([[1]] * n_subjects, columns=["intercept"])

    out = non_parametric_inference(
        Y,
        design_matrix=X,
        model_intercept=False,
        mask=mask,
        n_perm=N_PERM,
        tfce=True,
    )
    assert isinstance(out, dict)
    assert "t" in out
    assert "tfce" in out
    assert "logp_max_t" in out
    assert "logp_max_tfce" in out

    assert get_data(out["tfce"]).shape == SHAPE[:3]
    assert get_data(out["logp_max_tfce"]).shape == SHAPE[:3]


@pytest.mark.slow
def test_with_flm_objects(shape_3d_default):
    """See https://github.com/nilearn/nilearn/issues/3579 ."""
    mask, fmri_data, design_matrices = generate_fake_fmri_data_and_design(
        shapes=[(*shape_3d_default, 15)]
    )

    masker = NiftiMasker(mask)
    masker.fit()
    single_run_model = FirstLevelModel(mask_img=masker).fit(
        fmri_data[0], design_matrices=design_matrices[0]
    )
    single_run_model.compute_contrast("x")

    second_level_input = [single_run_model, single_run_model]

    design_matrix = pd.DataFrame(
        [1] * len(second_level_input), columns=["intercept"]
    )

    non_parametric_inference(
        second_level_input=second_level_input,
        design_matrix=design_matrix,
        first_level_contrast="x",
        n_perm=N_PERM,
    )


@pytest.mark.slow
def test_inputs_errors(rng, confounds, shape_4d_default):
    """Check errors for several inputs."""
    # Test processing of FMRI inputs
    # prepare fake data
    _, fmri_data, design_matrices = generate_fake_fmri_data_and_design(
        [shape_4d_default], rk=1
    )

    # prepare correct input first level models
    flm = FirstLevelModel(subject_label="01").fit(
        fmri_data, design_matrices=design_matrices
    )

    # prepare correct input dataframe and lists
    p, q = 80, 10
    X = rng.standard_normal(size=(p, q))
    sdes = pd.DataFrame(X[:3, :3], columns=["intercept", "b", "c"])

    _, fmri_data = fake_fmri_data()
    niimgs = [fmri_data] * 3
    niimg_4d = concat_imgs(niimgs)

    # test missing second-level contrast
    match = "No second-level contrast is specified."
    # niimgs as input
    with pytest.raises(ValueError, match=match):
        non_parametric_inference(niimgs, None, sdes)
    with pytest.raises(ValueError, match=match):
        non_parametric_inference(niimgs, confounds, sdes)
    # 4d niimg as input
    with pytest.raises(ValueError, match=match):
        non_parametric_inference(niimg_4d, None, sdes)

    # test wrong input errors
    # test first level model
    with pytest.raises(TypeError, match="second_level_input must be"):
        non_parametric_inference(flm)

    # test list of less than two niimgs
    with pytest.raises(TypeError, match="at least two"):
        non_parametric_inference([fmri_data])

    # test niimgs requirements
    with pytest.raises(ValueError, match="require a design matrix"):
        non_parametric_inference(niimgs)
    with pytest.raises(TypeError):
        non_parametric_inference([*niimgs, []], confounds)

    # test other objects
    with pytest.raises(ValueError, match=r"File not found: .*"):
        non_parametric_inference("random string object")


@pytest.mark.slow
def test_with_paths(tmp_path, n_subjects):
    """Test using path as inputs."""
    mask_file, fmri_files, _ = write_fake_fmri_data_and_design(
        (SHAPE,), file_path=tmp_path
    )
    fmri_files = fmri_files[0]
    df_input = pd.DataFrame(
        {
            "subject_label": [f"sub-{i}" for i in range(n_subjects)],
            "effects_map_path": [fmri_files] * n_subjects,
            "map_name": [fmri_files] * n_subjects,
        }
    )
    func_img = load(fmri_files)
    Y = [func_img] * n_subjects
    X = pd.DataFrame([[1]] * n_subjects, columns=["intercept"])
    c1 = np.eye(len(X.columns))[0]
    neg_log_pvals_imgs = [
        non_parametric_inference(
            second_level_input,
            design_matrix=X,
            second_level_contrast=c1,
            first_level_contrast=fmri_files,
            mask=mask_file,
            n_perm=N_PERM,
            verbose=1,
        )
        for second_level_input in [Y, df_input]
    ]

    assert all(isinstance(img, Nifti1Image) for img in neg_log_pvals_imgs)
    for img in neg_log_pvals_imgs:
        assert_array_equal(img.affine, load(mask_file).affine)

    neg_log_pvals_list = [get_data(i) for i in neg_log_pvals_imgs]
    for neg_log_pvals in neg_log_pvals_list:
        assert np.all(neg_log_pvals <= -np.log10(1.0 / (N_PERM + 1)))
        assert np.all(neg_log_pvals >= 0)


def test_override_masker_param(n_subjects):
    """Check that parameter of the masker are over-ridden."""
    func_img, mask = fake_fmri_data()
    Y = [func_img] * n_subjects
    X = pd.DataFrame([[1]] * n_subjects, columns=["intercept"])
    c1 = np.eye(len(X.columns))[0]

    masker = NiftiMasker(mask, smoothing_fwhm=2.0)
    with pytest.warns(
        UserWarning,
        match="Parameter 'smoothing_fwhm' of the masker overridden",
    ):
        non_parametric_inference(
            Y,
            design_matrix=X,
            second_level_contrast=c1,
            smoothing_fwhm=3.0,
            mask=masker,
            n_perm=N_PERM,
        )


@pytest.mark.slow
@pytest.mark.parametrize("second_level_contrast", [None, "intercept", [1]])
def test_contrast_computation(second_level_contrast, n_subjects):
    """Compute contrast with 1 column in design matrix."""
    func_img, mask = fake_fmri_data()

    Y = [func_img] * n_subjects
    X = pd.DataFrame([[1]] * n_subjects, columns=["intercept"])

    non_parametric_inference(
        Y,
        design_matrix=X,
        model_intercept=False,
        mask=mask,
        n_perm=N_PERM,
        second_level_contrast=second_level_contrast,
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "second_level_contrast", [[1, 0], "r1", "r1-r2", [-1, 1]]
)
def test_contrast_formula(second_level_contrast, rng, n_subjects):
    """Compute contrast with more than 1 column in design matrix."""
    func_img, _ = fake_fmri_data()
    Y = [func_img] * n_subjects
    X = pd.DataFrame(rng.uniform(size=(n_subjects, 2)), columns=["r1", "r2"])

    non_parametric_inference(
        second_level_input=Y,
        design_matrix=X,
        second_level_contrast=second_level_contrast,
    )


def test_contrast_computation_errors(rng, n_subjects):
    """Check several errors of with contrast computation."""
    func_img, mask = fake_fmri_data()

    # asking for contrast before model fit gives error
    with pytest.raises(TypeError, match="second_level_input must be either"):
        non_parametric_inference(
            second_level_input=None,
            second_level_contrast="intercept",
            mask=mask,
        )

    # fit model
    Y = [func_img] * n_subjects
    X = pd.DataFrame([[1]] * n_subjects, columns=["intercept"])

    ncol = len(X.columns)
    _, cnull = np.eye(ncol)[0, :], np.zeros(ncol)

    # passing null contrast should give back a value error
    with pytest.raises(
        ValueError,
        match=("Second_level_contrast must be a valid"),
    ):
        non_parametric_inference(
            second_level_input=Y,
            design_matrix=X,
            second_level_contrast=cnull,
            mask=mask,
        )
    with pytest.raises(
        ValueError,
        match=("Second_level_contrast must be a valid"),
    ):
        non_parametric_inference(
            second_level_input=Y,
            design_matrix=X,
            second_level_contrast=[],
            mask=mask,
        )

    # check that passing no explicit contrast when the design
    # matrix has more than one columns raises an error
    X = pd.DataFrame(rng.uniform(size=(n_subjects, 2)), columns=["r1", "r2"])
    with pytest.raises(
        ValueError, match=r"No second-level contrast is specified."
    ):
        non_parametric_inference(
            second_level_input=Y,
            design_matrix=X,
            second_level_contrast=None,
        )


# -----------------------surface tests----------------------- #


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"tfce": True},  # to run cluster inference
        {"threshold": 0.001},  # to run cluster inference
    ],
)
@pytest.mark.parametrize("two_sided_test", [True, False])
def test_surface_images(surf_img_1d, two_sided_test, kwargs, n_subjects):
    """Smoke test non_parametric_inference on list of 1D surfaces."""
    second_level_input = [surf_img_1d for _ in range(n_subjects)]

    design_matrix = pd.DataFrame([1] * n_subjects, columns=["intercept"])

    non_parametric_inference(
        second_level_input=second_level_input,
        design_matrix=design_matrix,
        n_perm=N_PERM,
        two_sided_test=two_sided_test,
        **kwargs,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"tfce": True},  # to run cluster inference
        {"threshold": 0.001},  # to run cluster inference
    ],
)
def test_surface_images_2d(surf_img_2d, n_subjects, kwargs):
    """Smoke test non_parametric_inference on 2d surfaces."""
    second_level_input = surf_img_2d(n_subjects)

    design_matrix = pd.DataFrame([1] * n_subjects, columns=["intercept"])

    non_parametric_inference(
        second_level_input=second_level_input,
        design_matrix=design_matrix,
        n_perm=N_PERM,
        **kwargs,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"tfce": True},  # to run cluster inference
        {"threshold": 0.001},  # to run cluster inference
    ],
)
def test_surface_images_2d_mask(surf_img_2d, surf_mask_1d, n_subjects, kwargs):
    """Smoke test non_parametric_inference on 2d surfaces and a mask."""
    second_level_input = surf_img_2d(n_subjects)

    design_matrix = pd.DataFrame([1] * n_subjects, columns=["intercept"])

    masker = SurfaceMasker(surf_mask_1d)

    non_parametric_inference(
        second_level_input=second_level_input,
        design_matrix=design_matrix,
        n_perm=N_PERM,
        mask=masker,
        **kwargs,
    )


def test_surface_images_warnings(surf_img_1d, n_subjects):
    """Throw warnings for non implemented features for surface."""
    second_level_input = [surf_img_1d for _ in range(n_subjects)]

    design_matrix = pd.DataFrame([1] * n_subjects, columns=["intercept"])

    with pytest.warns(
        NotImplementedWarning,
        match="'smoothing_fwhm' is not yet supported for surface data.",
    ):
        non_parametric_inference(
            second_level_input=second_level_input,
            design_matrix=design_matrix,
            n_perm=N_PERM,
            smoothing_fwhm=6,
        )
