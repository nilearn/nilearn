"""Test the second level model."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from nibabel import Nifti1Image
from numpy.testing import (
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
)
from sklearn.utils.estimator_checks import parametrize_with_checks

from nilearn._utils import testing
from nilearn._utils.data_gen import (
    generate_fake_fmri_data_and_design,
    write_fake_bold_img,
    write_fake_fmri_data_and_design,
)
from nilearn._utils.estimator_checks import (
    check_estimator,
    nilearn_check_estimator,
    return_expected_failed_checks,
)
from nilearn._utils.tags import SKLEARN_LT_1_6
from nilearn.exceptions import NotImplementedWarning
from nilearn.glm.first_level import FirstLevelModel, run_glm
from nilearn.glm.second_level import SecondLevelModel
from nilearn.glm.second_level.second_level import (
    _check_confounds,
    _check_first_level_contrast,
    _check_input_as_first_level_model,
    _check_n_rows_desmat_vs_n_effect_maps,
    _check_second_level_input,
    _infer_effect_maps,
    _process_second_level_input_as_dataframe,
    _process_second_level_input_as_firstlevelmodels,
    _sort_input_dataframe,
)
from nilearn.glm.tests.conftest import SHAPE, _confounds, fake_fmri_data
from nilearn.image import concat_imgs, get_data
from nilearn.maskers import NiftiMasker
from nilearn.surface.utils import assert_surface_image_equal

ESTIMATORS_TO_CHECK = [SecondLevelModel()]

if SKLEARN_LT_1_6:

    @pytest.mark.parametrize(
        "estimator, check, name",
        check_estimator(estimators=ESTIMATORS_TO_CHECK),
    )
    def test_check_estimator_sklearn_valid(estimator, check, name):  # noqa: ARG001
        """Check compliance with sklearn estimators."""
        check(estimator)

    @pytest.mark.xfail(reason="invalid checks should fail")
    @pytest.mark.parametrize(
        "estimator, check, name",
        check_estimator(estimators=ESTIMATORS_TO_CHECK, valid=False),
    )
    def test_check_estimator_sklearn_invalid(estimator, check, name):  # noqa: ARG001
        """Check compliance with sklearn estimators."""
        check(estimator)

else:

    @parametrize_with_checks(
        estimators=ESTIMATORS_TO_CHECK,
        expected_failed_checks=return_expected_failed_checks,
    )
    def test_check_estimator_sklearn(estimator, check):
        """Check compliance with sklearn estimators."""
        check(estimator)


@pytest.mark.slow
@pytest.mark.parametrize(
    "estimator, check, name",
    nilearn_check_estimator(estimators=ESTIMATORS_TO_CHECK),
)
def test_check_estimator_nilearn(estimator, check, name):  # noqa: ARG001
    """Check compliance with nilearn estimators rules."""
    check(estimator)


@pytest.fixture
def input_df():
    """Input DataFrame for testing."""
    return pd.DataFrame(
        {
            "effects_map_path": ["foo.nii", "bar.nii", "baz.nii"],
            "subject_label": ["foo", "bar", "baz"],
        }
    )


def test_process_second_level_input_as_dataframe(input_df):
    """Unit tests for function _process_second_level_input_as_dataframe()."""
    sample_map, subjects_label = _process_second_level_input_as_dataframe(
        input_df
    )
    assert sample_map == "foo.nii"
    assert subjects_label == ["foo", "bar", "baz"]


def test_sort_input_dataframe(input_df):
    """Unit tests for function _sort_input_dataframe()."""
    output_df = _sort_input_dataframe(input_df)

    assert output_df["subject_label"].to_list() == [
        "bar",
        "baz",
        "foo",
    ]
    assert output_df["effects_map_path"].to_list() == [
        "bar.nii",
        "baz.nii",
        "foo.nii",
    ]


def test_second_level_input_as_3d_images(
    rng, affine_eye, tmp_path, shape_3d_default, n_subjects
):
    """Test second level model with a list 3D image filenames as input.

    Should act as a regression test for:
    https://github.com/nilearn/nilearn/issues/3636

    """
    images = []
    for _ in range(n_subjects):
        data = rng.random(shape_3d_default)
        images.append(Nifti1Image(data, affine_eye))

    filenames = testing.write_imgs_to_path(
        *images, file_path=tmp_path, create_files=True
    )
    second_level_input = filenames
    design_matrix = pd.DataFrame(
        [1] * len(second_level_input), columns=["intercept"]
    )

    second_level_model = SecondLevelModel(smoothing_fwhm=8.0)
    second_level_model = second_level_model.fit(
        second_level_input,
        design_matrix=design_matrix,
    )


@pytest.mark.slow
def test_process_second_level_input_as_firstlevelmodels(
    shape_4d_default, n_subjects
):
    """Unit tests for function \
       _process_second_level_input_as_firstlevelmodels().
    """
    mask, fmri_data, design_matrices = generate_fake_fmri_data_and_design(
        shapes=[shape_4d_default]
    )
    list_of_flm = [
        FirstLevelModel(mask_img=mask, subject_label=f"sub-{i}").fit(
            fmri_data[0], design_matrices=design_matrices[0]
        )
        for i in range(n_subjects)
    ]
    (
        sample_map,
        subjects_label,
    ) = _process_second_level_input_as_firstlevelmodels(list_of_flm)

    assert subjects_label == [f"sub-{i}" for i in range(n_subjects)]
    assert isinstance(sample_map, Nifti1Image)
    assert sample_map.shape == shape_4d_default[:3]


@pytest.mark.slow
def test_check_affine_first_level_models(
    affine_eye, shape_4d_default, n_subjects
):
    """Check all FirstLevelModel have the same affine."""
    mask, fmri_data, design_matrices = generate_fake_fmri_data_and_design(
        shapes=[shape_4d_default]
    )
    list_of_flm = [
        FirstLevelModel(mask_img=mask, subject_label=f"sub-{i}").fit(
            fmri_data[0], design_matrices=design_matrices[0]
        )
        for i in range(n_subjects)
    ]
    # should pass
    _check_input_as_first_level_model(
        second_level_input=list_of_flm, none_confounds=False
    )

    # add a model with a different affine
    # should raise an error
    mask, fmri_data, design_matrices = generate_fake_fmri_data_and_design(
        shapes=[shape_4d_default], affine=affine_eye * 2
    )
    list_of_flm.append(
        FirstLevelModel(mask_img=mask, subject_label="sub-4").fit(
            fmri_data[0], design_matrices=design_matrices[0]
        )
    )

    with pytest.raises(
        ValueError, match="All first level models must have the same affine"
    ):
        _check_input_as_first_level_model(
            second_level_input=list_of_flm, none_confounds=False
        )


@pytest.mark.slow
def test_check_shape_first_level_models(shape_4d_default, n_subjects):
    """Check all FirstLevelModel have the same shape."""
    mask, fmri_data, design_matrices = generate_fake_fmri_data_and_design(
        shapes=[shape_4d_default]
    )
    list_of_flm = [
        FirstLevelModel(mask_img=mask, subject_label=f"sub-{i}").fit(
            fmri_data[0], design_matrices=design_matrices[0]
        )
        for i in range(n_subjects)
    ]
    # should pass
    _check_input_as_first_level_model(
        second_level_input=list_of_flm, none_confounds=False
    )

    # add a model with a different shape
    # should raise an error
    mask, fmri_data, design_matrices = generate_fake_fmri_data_and_design(
        shapes=[(8, 9, 10, 15)]
    )
    list_of_flm.append(
        FirstLevelModel(mask_img=mask, subject_label="sub-4").fit(
            fmri_data[0], design_matrices=design_matrices[0]
        )
    )

    with pytest.raises(
        ValueError, match="All first level models must have the same shape"
    ):
        _check_input_as_first_level_model(
            second_level_input=list_of_flm, none_confounds=False
        )


def test_check_second_level_input(shape_4d_default):
    """Raise errors when wrong inputs are passed to SecondLevelModel."""
    with pytest.raises(TypeError, match="'second_level_input' must be"):
        _check_second_level_input(1, None)

    with pytest.raises(
        TypeError,
        match="A second level model requires a list with at "
        "least two first level models or niimgs",
    ):
        _check_second_level_input([FirstLevelModel()], pd.DataFrame())

    with pytest.raises(
        TypeError, match="Got object type <class 'int'> at idx 1"
    ):
        _check_second_level_input(["foo", 1], pd.DataFrame())

    mask, fmri_data, design_matrices = generate_fake_fmri_data_and_design(
        shapes=[shape_4d_default]
    )

    input_models = [
        FirstLevelModel(mask_img=mask).fit(
            fmri_data[0], design_matrices=design_matrices[0]
        )
    ]

    obj = lambda: None  # noqa: E731
    obj.results_ = "foo"
    obj.labels_ = "bar"

    with pytest.raises(
        TypeError, match="Got object type <class 'function'> at idx 1"
    ):
        _check_second_level_input([*input_models, obj], pd.DataFrame())


def test_check_second_level_input_list_wrong_type():
    """Raise errors when wrong inputs are passed to SecondLevelModel.

    Integration test: slightly higher level test than those for
    _check_second_level_input.
    """
    model = SecondLevelModel()
    second_level_input = [1, 2]
    with pytest.raises(TypeError, match="'second_level_input' must be"):
        model.fit(second_level_input)


def test_check_second_level_input_unfit_model():
    """Test _check_second_level_input with unfitted first level models."""
    with pytest.raises(
        ValueError, match="Model sub_1 at index 0 has not been fit yet"
    ):
        _check_second_level_input(
            [FirstLevelModel(subject_label=f"sub_{i}") for i in range(1, 3)],
            pd.DataFrame(),
        )


def test_check_second_level_input_dataframe():
    """Test _check_second_level_input with DataFrame."""
    with pytest.raises(
        ValueError,
        match="'second_level_input' DataFrame must have columns "
        "'subject_label', 'map_name' and 'effects_map_path'",
    ):
        _check_second_level_input(
            pd.DataFrame(columns=["foo", "bar"]), pd.DataFrame()
        )

    with pytest.raises(
        ValueError, match="'subject_label' column must contain only strings"
    ):
        _check_second_level_input(
            pd.DataFrame(
                {
                    "subject_label": [1, 2],
                    "map_name": ["a", "b"],
                    "effects_map_path": ["c", "d"],
                }
            ),
            pd.DataFrame(),
        )


def test_check_second_level_input_confounds(shape_4d_default):
    """Test _check_second_level_input with confounds."""
    mask, fmri_data, design_matrices = generate_fake_fmri_data_and_design(
        shapes=[shape_4d_default]
    )

    input_models = [
        FirstLevelModel(mask_img=mask).fit(
            fmri_data[0], design_matrices=design_matrices[0]
        )
    ]

    with pytest.raises(
        ValueError,
        match="In case confounds are provided, first level "
        "objects need to provide the attribute 'subject_label'",
    ):
        _check_second_level_input(
            input_models * 2, pd.DataFrame(), confounds=pd.DataFrame()
        )


def test_check_second_level_input_design_matrix(shape_4d_default):
    """Raise errors when no design matrix is passed to SecondLevelModel.

    When passing niimg like objects.
    """
    _, fmri_data, _ = generate_fake_fmri_data_and_design(
        shapes=[shape_4d_default]
    )

    _check_second_level_input(fmri_data[0], pd.DataFrame())

    with pytest.raises(
        ValueError,
        match="List of niimgs as second_level_input "
        "require a design matrix to be provided",
    ):
        _check_second_level_input(fmri_data * 2, None)
    with pytest.raises(
        ValueError,
        match="List of niimgs as second_level_input "
        "require a design matrix to be provided",
    ):
        _check_second_level_input(fmri_data[0], None)


def test_check_confounds():
    """Test _check_confounds."""
    _check_confounds(None)  # Should not do anything
    with pytest.raises(
        TypeError, match="confounds must be a pandas DataFrame"
    ):
        _check_confounds("foo")
    with pytest.raises(
        ValueError, match="confounds DataFrame must contain column"
    ):
        _check_confounds(pd.DataFrame())
    with pytest.raises(
        ValueError, match="confounds should contain at least 2 columns"
    ):
        _check_confounds(pd.DataFrame(columns=["subject_label"]))
    with pytest.raises(
        ValueError, match="subject_label column must contain only strings"
    ):
        _check_confounds(
            pd.DataFrame(
                {"subject_label": [None, None, None], "conf": [4, 5, 6]}
            )
        )


def test_check_first_level_contrast():
    """Test _check_first_level_contrast."""
    _check_first_level_contrast(["foo"], None)  # Should not do anything
    _check_first_level_contrast([FirstLevelModel()], "foo")
    with pytest.raises(ValueError, match="If second_level_input was a list"):
        _check_first_level_contrast([FirstLevelModel()], None)


def test_check_n_rows_desmat_vs_n_effect_maps():
    """Check match dimension design matrix and number of input image."""
    _check_n_rows_desmat_vs_n_effect_maps(
        [1, 2, 3], np.array([[1, 2], [3, 4], [5, 6]])
    )
    with pytest.raises(
        ValueError,
        match="design_matrix does not match the number of maps considered",
    ):
        _check_n_rows_desmat_vs_n_effect_maps(
            [1, 2], np.array([[1, 2], [3, 4], [5, 6]])
        )


@pytest.mark.slow
def test_infer_effect_maps(tmp_path, shape_4d_default):
    """Check that the right input is inferred.

    second_level_input could for example
    be a list of images
    or a dataframe 'mapping' a string to an image.
    """
    rk = 3
    shapes = [SHAPE, shape_4d_default]
    mask_file, fmri_files, design_files = write_fake_fmri_data_and_design(
        shapes, rk=rk, file_path=tmp_path
    )
    second_level_input = pd.DataFrame(
        {"map_name": ["a", "b"], "effects_map_path": [fmri_files[0], "bar"]}
    )

    assert _infer_effect_maps(second_level_input, "a") == [fmri_files[0]]
    assert _infer_effect_maps([fmri_files[0]], None) == [fmri_files[0]]

    contrast = np.eye(rk)[1]
    second_level_input = [FirstLevelModel(mask_img=mask_file)] * 2
    for i, model in enumerate(second_level_input):
        model.fit(fmri_files[i], design_matrices=design_files[i])

    assert len(_infer_effect_maps(second_level_input, contrast)) == 2


def test_infer_effect_maps_error(tmp_path, shape_3d_default):
    """Check error raised when inferring 'type' for the images.

    For example if the image mapped in a dataframe does not exist.
    """
    shapes = [(*shape_3d_default, 5), (*shape_3d_default, 6)]
    _, fmri_files, _ = write_fake_fmri_data_and_design(
        shapes, file_path=tmp_path
    )
    second_level_input = pd.DataFrame(
        {"map_name": ["a", "b"], "effects_map_path": [fmri_files[0], "bar"]}
    )
    with pytest.raises(ValueError, match="File not found: 'bar'"):
        _infer_effect_maps(second_level_input, "b")


def test_affine_output_mask(n_subjects):
    """Make sure output image matches that of mask."""
    func_img, mask = fake_fmri_data()

    model = SecondLevelModel(mask_img=mask)

    Y = [func_img] * n_subjects
    X = pd.DataFrame([[1]] * n_subjects, columns=["intercept"])
    model = model.fit(Y, design_matrix=X)

    c1 = np.eye(len(model.design_matrix_.columns))[0]
    z_image = model.compute_contrast(c1, output_type="z_score")

    assert_array_equal(z_image.affine, mask.affine)


def test_affine_shape_output_when_provided(affine_eye, n_subjects):
    """Check fov output corresponds to the one passed to model."""
    func_img, mask = fake_fmri_data()

    Y = [func_img] * n_subjects
    X = pd.DataFrame([[1]] * n_subjects, columns=["intercept"])

    target_shape = (10, 10, 10)
    target_affine = affine_eye
    target_affine[0, 3] = 1

    model = SecondLevelModel(
        mask_img=mask,
        target_shape=target_shape,
        target_affine=target_affine,
    )
    model = model.fit(Y, design_matrix=X)

    c1 = np.eye(len(model.design_matrix_.columns))[0]
    z_image = model.fit(Y, design_matrix=X).compute_contrast(c1)

    assert_array_equal(z_image.shape, target_shape)
    assert_array_equal(z_image.affine, target_affine)


def test_slm_4d_image(img_4d_mni):
    """Compute contrast with 4D images as input.

    See https://github.com/nilearn/nilearn/issues/3058
    """
    model = SecondLevelModel()
    Y = img_4d_mni
    X = pd.DataFrame([[1]] * img_4d_mni.shape[3], columns=["intercept"])
    model = model.fit(Y, design_matrix=X)
    c1 = np.eye(len(model.design_matrix_.columns))[0]
    model.compute_contrast(c1, output_type="z_score")


def test_warning_overriding_with_masker_parameter(n_subjects):
    """Test over-riding slm default with masker params."""
    func_img, mask = fake_fmri_data()

    # fit model
    Y = [func_img] * n_subjects
    X = pd.DataFrame([[1]] * n_subjects, columns=["intercept"])

    # Provide a masker as mask_img
    masker = NiftiMasker(mask).fit()
    with pytest.warns(
        UserWarning,
        match=(
            "Overriding provided-default estimator parameters "
            "with provided masker parameters"
        ),
    ):
        SecondLevelModel(mask_img=masker, verbose=1).fit(Y, design_matrix=X)


@pytest.mark.slow
@pytest.mark.parametrize("confounds", [None, _confounds()])
def test_fmri_inputs_flms(rng, confounds, shape_4d_default):
    """Test second level model with first level model as inputs."""
    # prepare fake data
    mask, fmri_data, design_matrices = generate_fake_fmri_data_and_design(
        [shape_4d_default], rk=1
    )

    # prepare correct input first level models
    flm = FirstLevelModel(subject_label="01").fit(
        fmri_data, design_matrices=design_matrices
    )

    # prepare correct input dataframe and lists
    p, q = 80, 10
    X = rng.standard_normal(size=(p, q))
    design_matrix = pd.DataFrame(X[:3, :3], columns=["intercept", "b", "c"])

    # smoke tests with correct input
    flms = [flm, flm, flm]

    # First level models as input
    SecondLevelModel(mask_img=mask).fit(flms)
    SecondLevelModel().fit(flms)

    # Note : the following one creates a singular design matrix
    SecondLevelModel().fit(flms, confounds)
    SecondLevelModel().fit(flms, confounds, design_matrix)


@pytest.mark.parametrize("confounds", [None, _confounds()])
def test_fmri_inputs_images(rng, confounds):
    """Test second level model with image as inputs."""
    # prepare correct input dataframe and lists
    p, q = 80, 10
    X = rng.standard_normal(size=(p, q))
    design_matrix = pd.DataFrame(X[:3, :3], columns=["intercept", "b", "c"])

    fmri_data, _ = fake_fmri_data()

    # niimgs as input
    niimgs = [fmri_data, fmri_data, fmri_data]
    SecondLevelModel().fit(niimgs, confounds, design_matrix)

    # 4d niimg as input
    niimg_4d = concat_imgs(niimgs)
    SecondLevelModel().fit(niimg_4d, confounds, design_matrix)


@pytest.mark.parametrize("confounds", [None, _confounds()])
def test_fmri_inputs_dataframes_as_input(tmp_path, rng, confounds):
    """Test second level model with dataframe as inputs."""
    # prepare fake data
    p, q = 80, 10
    X = rng.standard_normal(size=(p, q))

    # prepare correct input dataframe and lists
    _, fmri_files, _ = write_fake_fmri_data_and_design(
        (SHAPE,), file_path=tmp_path
    )
    fmri_files = fmri_files[0]

    design_matrix = pd.DataFrame(X[:3, :3], columns=["intercept", "b", "c"])

    # dataframes as input
    dfcols = ["subject_label", "map_name", "effects_map_path"]
    dfrows = [
        ["01", "a", fmri_files],
        ["02", "a", fmri_files],
        ["03", "a", fmri_files],
    ]
    niidf = pd.DataFrame(dfrows, columns=dfcols)

    SecondLevelModel().fit(niidf, confounds)
    SecondLevelModel().fit(niidf, confounds, design_matrix)


def test_fmri_pandas_series_as_input(tmp_path, rng):
    """Use pandas series of file paths as inputs."""
    # prepare correct input dataframe and lists
    p, q = 80, 10
    X = rng.standard_normal(size=(p, q))
    _, fmri_files, _ = write_fake_fmri_data_and_design(
        (SHAPE,), file_path=tmp_path
    )
    fmri_files = fmri_files[0]

    # dataframes as input
    design_matrix = pd.DataFrame(X[:3, :3], columns=["intercept", "b", "c"])
    niidf = pd.DataFrame({"filepaths": [fmri_files, fmri_files, fmri_files]})
    SecondLevelModel().fit(
        second_level_input=niidf["filepaths"],
        confounds=None,
        design_matrix=design_matrix,
    )


def test_fmri_inputs_pandas_errors():
    """Test wrong second level inputs."""
    # test wrong input for list and pandas requirements
    nii_img = ["01", "02", "03"]
    with pytest.raises(ValueError, match="File not found: "):
        SecondLevelModel().fit(nii_img)

    nii_series = pd.Series(nii_img)
    with pytest.raises(ValueError, match="File not found: "):
        SecondLevelModel().fit(nii_series)

    # test dataframe requirements
    dfcols = [
        "not_the_right_column_name",
    ]
    dfrows = [["01"], ["02"], ["03"]]
    niidf = pd.DataFrame(dfrows, columns=dfcols)
    with pytest.raises(
        ValueError,
        match=(
            r"'second_level_input' DataFrame must have "
            r"columns 'subject_label', 'map_name' and 'effects_map_path'."
        ),
    ):
        SecondLevelModel().fit(niidf)


def test_secondlevelmodel_fit_inputs_errors(confounds, shape_4d_default):
    """Raise the proper errors when invalid inputs are passed to fit."""
    # prepare fake data
    _, fmri_data, design_matrices = generate_fake_fmri_data_and_design(
        (shape_4d_default,)
    )

    # prepare correct input first level models
    flm = FirstLevelModel(subject_label="01").fit(
        fmri_data[0], design_matrices=design_matrices[0]
    )

    # test first level model requirements
    with pytest.raises(TypeError, match="'second_level_input' must be"):
        SecondLevelModel().fit(second_level_input=flm)
    with pytest.raises(TypeError, match="at least two"):
        SecondLevelModel().fit(second_level_input=[flm])

    # test first_level_conditions, confounds, and design
    flms = [flm, flm, flm]
    with pytest.raises(
        TypeError, match="confounds must be a pandas DataFrame"
    ):
        SecondLevelModel().fit(second_level_input=flms, confounds=["", []])
    with pytest.raises(
        TypeError, match="confounds must be a pandas DataFrame"
    ):
        SecondLevelModel().fit(second_level_input=flms, confounds=[])
    with pytest.raises(
        TypeError, match="confounds must be a pandas DataFrame"
    ):
        SecondLevelModel().fit(
            second_level_input=flms, confounds=confounds["conf1"]
        )


@pytest.mark.slow
@pytest.mark.parametrize(
    "filename, sep", [("design.csv", ","), ("design.tsv", "\t")]
)
def test_secondlevelmodel_design_matrix_path(
    img_3d_mni, tmp_path, filename, sep
):
    """Test design matrix as file fit input requirements."""
    second_level_input = [img_3d_mni, img_3d_mni]
    design_matrix = pd.DataFrame(
        np.ones((len(second_level_input), 1)), columns=["a"]
    )

    SecondLevelModel().fit(
        second_level_input=second_level_input, design_matrix=design_matrix
    )

    design_matrix_fname = tmp_path / filename
    design_matrix.to_csv(design_matrix_fname, sep=sep)

    SecondLevelModel().fit(
        second_level_input=second_level_input,
        design_matrix=design_matrix_fname,
    )
    SecondLevelModel().fit(
        second_level_input=second_level_input,
        design_matrix=str(design_matrix_fname),
    )


@pytest.mark.parametrize("design_matrix", ["foo", Path("foo")])
def test_secondlevelmodel_design_matrix_error_path(img_3d_mni, design_matrix):
    """Test error design matrix as file fit input requirements."""
    second_level_input = [img_3d_mni, img_3d_mni, img_3d_mni]
    with pytest.raises(
        ValueError, match=r"Tables to load can only be TSV or CSV."
    ):
        SecondLevelModel().fit(
            second_level_input=second_level_input, design_matrix=design_matrix
        )


@pytest.mark.parametrize("design_matrix", [1, ["foo"]])
def test_secondlevelmodel_design_matrix_error_type(img_3d_mni, design_matrix):
    """Test design matrix fit input requirements."""
    second_level_input = [img_3d_mni, img_3d_mni, img_3d_mni]

    with pytest.raises(TypeError, match="'design_matrix' must be "):
        SecondLevelModel().fit(
            second_level_input=second_level_input, design_matrix=design_matrix
        )


def test_fmri_img_inputs_errors(confounds):
    """Test niimgs fit input requirements."""
    fmri_data, _ = fake_fmri_data()

    niimgs = [fmri_data, fmri_data, fmri_data]
    with pytest.raises(ValueError, match="require a design matrix"):
        SecondLevelModel().fit(niimgs)

    with pytest.raises(
        TypeError,
        match=r"Elements of second_level_input must be of the same type.",
    ):
        SecondLevelModel().fit([*niimgs, []], confounds)


def test_second_level_glm_computation(n_subjects):
    """Compare output of compute_contrast and run_glm."""
    func_img, mask = fake_fmri_data()

    model = SecondLevelModel(mask_img=mask)
    Y = [func_img] * n_subjects
    X = pd.DataFrame([[1]] * n_subjects, columns=["intercept"])

    model = model.fit(Y, design_matrix=X)
    model.compute_contrast()
    labels1 = model.labels_
    results1 = model.results_

    labels2, results2 = run_glm(model.masker_.transform(Y), X.values, "ols")
    assert_almost_equal(labels1, labels2, decimal=1)

    assert len(results1) == len(results2)


@pytest.mark.parametrize("attribute", ["residuals", "predicted", "r_square"])
def test_second_level_voxelwise_attribute_errors(attribute, n_subjects):
    """Tests that an error is raised when trying to access \
       voxelwise attributes before fitting the model, \
       before computing a contrast.
    """
    fmri_data, mask = fake_fmri_data()

    model = SecondLevelModel(mask_img=mask, minimize_memory=False)

    Y = [fmri_data] * n_subjects
    X = pd.DataFrame([[1]] * n_subjects, columns=["intercept"])
    model.fit(Y, design_matrix=X)

    with pytest.raises(ValueError, match=r"The model has no results."):
        getattr(model, attribute)
    with pytest.raises(ValueError, match="attribute must be one of"):
        model._get_element_wise_model_attribute("foo", True)


@pytest.mark.slow
@pytest.mark.parametrize("attribute", ["residuals", "predicted", "r_square"])
def test_second_level_voxelwise_attribute_errors_minimize_memory(
    attribute, n_subjects
):
    """Tests that an error is raised when trying to access \
       voxelwise attributes before fitting the model, \
       when not setting ``minimize_memory`` to ``True``.
    """
    fmri_data, mask = fake_fmri_data()

    model = SecondLevelModel(mask_img=mask, minimize_memory=True)

    Y = [fmri_data] * n_subjects
    X = pd.DataFrame([[1]] * n_subjects, columns=["intercept"])
    model.fit(Y, design_matrix=X)

    model.compute_contrast()

    with pytest.raises(ValueError, match="To access voxelwise attributes"):
        getattr(model, attribute)


@pytest.mark.slow
@pytest.mark.parametrize("attribute", ["residuals", "predicted", "r_square"])
def test_second_level_voxelwise_attribute(attribute, n_subjects):
    """Smoke test for voxelwise attributes for SecondLevelModel."""
    fmri_data, mask = fake_fmri_data()
    model = SecondLevelModel(mask_img=mask, minimize_memory=False)
    Y = [fmri_data] * n_subjects
    X = pd.DataFrame([[1]] * n_subjects, columns=["intercept"])
    model.fit(Y, design_matrix=X)
    model.compute_contrast()

    getattr(model, attribute)


@pytest.mark.slow
def test_second_level_residuals(n_subjects):
    """Tests residuals computation for SecondLevelModel."""
    fmri_data, mask = fake_fmri_data()
    model = SecondLevelModel(mask_img=mask, minimize_memory=False)
    Y = [fmri_data] * n_subjects
    X = pd.DataFrame([[1]] * n_subjects, columns=["intercept"])
    model.fit(Y, design_matrix=X)
    model.compute_contrast()

    assert isinstance(model.residuals, Nifti1Image)
    assert model.residuals.shape == (*SHAPE[:3], n_subjects)
    mean_residuals = model.masker_.transform(model.residuals).mean(0)
    assert_array_almost_equal(mean_residuals, 0)


@pytest.mark.slow
def test_second_level_contrast_computation_smoke(fitted_slm):
    """Smoke test for different contrasts in fixed effects."""
    ncol = len(fitted_slm.design_matrix_.columns)
    c1 = np.eye(ncol)[0, :]
    fitted_slm.compute_contrast(second_level_contrast=c1)

    # formula should work (passing variable name directly)
    fitted_slm.compute_contrast("intercept")

    # or simply pass nothing
    fitted_slm.compute_contrast()


@pytest.mark.slow
@pytest.mark.parametrize(
    "output_type",
    [
        "z_score",
        "stat",
        "p_value",
        "effect_size",
        "effect_variance",
    ],
)
def test_second_level_contrast_computation_all(fitted_slm, output_type):
    """Test output_type='all', and verify images are equivalent."""
    ncol = len(fitted_slm.design_matrix_.columns)
    c1 = np.eye(ncol)[0, :]

    all_images = fitted_slm.compute_contrast(
        second_level_contrast=c1, output_type="all"
    )

    assert_array_equal(
        get_data(all_images[output_type]),
        get_data(
            fitted_slm.compute_contrast(
                second_level_contrast=c1, output_type=output_type
            )
        ),
    )


def test_second_level_contrast_computation_unfitted_errors():
    """Check error is raised when computing contrast on unfitted SLM.

    # TODO
    # do in estimator_checks
    # asking for contrast before model fit gives error
    """
    model = SecondLevelModel()

    with pytest.raises(ValueError, match="not fitted yet"):
        model.compute_contrast(second_level_contrast="intercept")


def test_second_level_contrast_computation_errors(fitted_slm):
    """Check several errors during contrast computation."""
    ncol = len(fitted_slm.design_matrix_.columns)
    c1 = np.eye(ncol)[0, :]
    cnull = np.zeros(ncol)

    # passing null contrast should give back a value error
    with pytest.raises(ValueError, match="Contrast is null"):
        fitted_slm.compute_contrast(cnull)

    # passing wrong parameters
    with pytest.raises(ValueError, match="'stat_type' must be one of"):
        fitted_slm.compute_contrast(
            second_level_contrast=c1, second_level_stat_type=""
        )
    with pytest.raises(ValueError, match="'stat_type' must be one of"):
        fitted_slm.compute_contrast(
            second_level_contrast=c1, second_level_stat_type=[]
        )
    with pytest.raises(ValueError, match="'output_type' must be one of "):
        fitted_slm.compute_contrast(second_level_contrast=c1, output_type="")


def test_second_level_contrast_computation_none_errors(rng, n_subjects):
    """Check that passing no explicit contrast when the design
    matrix has more than one columns raises an error.
    """
    func_img, mask = fake_fmri_data()

    model = SecondLevelModel(mask_img=mask)

    Y = [func_img] * n_subjects
    X = pd.DataFrame([[1]] * n_subjects, columns=["intercept"])
    model = model.fit(Y, design_matrix=X)

    X = pd.DataFrame(rng.uniform(size=(n_subjects, 2)), columns=["r1", "r2"])
    model = model.fit(Y, design_matrix=X)
    with pytest.raises(
        ValueError, match="No second-level contrast is specified"
    ):
        model.compute_contrast(None)


def test_second_level_t_contrast_length_errors(fitted_slm):
    """Check SLM error T-contrast computation shape."""
    with pytest.raises(
        ValueError,
        match=(r"t contrasts should be of length P=1, but it has length 2."),
    ):
        fitted_slm.compute_contrast(second_level_contrast=[1, 2])


@pytest.mark.slow
def test_second_level_f_contrast_length_errors(fitted_slm):
    """Check SLM error F-contrast computation shape."""
    with pytest.raises(
        ValueError,
        match=(r"F contrasts should have .* columns, but it has .*"),
    ):
        fitted_slm.compute_contrast(second_level_contrast=np.eye(2))


@pytest.mark.slow
def test_second_level_contrast_computation_with_memory_caching(n_subjects):
    """Smoke test for caching of SLM."""
    func_img, mask = fake_fmri_data()

    model = SecondLevelModel(mask_img=mask, memory="nilearn_cache")

    Y = [func_img] * n_subjects
    X = pd.DataFrame([[1]] * n_subjects, columns=["intercept"])
    model = model.fit(Y, design_matrix=X)

    ncol = len(model.design_matrix_.columns)
    c1 = np.eye(ncol)[0, :]

    # test memory caching for compute_contrast
    model.compute_contrast(c1, output_type="z_score")
    # or simply pass nothing
    model.compute_contrast()


def test_second_lvl_dataframe_computation(tmp_path, shape_3d_default):
    """Check that contrast can be computed when using dataframes as input.

    See bug https://github.com/nilearn/nilearn/issues/3871
    """
    file_path = write_fake_bold_img(
        file_path=tmp_path / "img.nii.gz", shape=shape_3d_default
    )

    dfcols = ["subject_label", "map_name", "effects_map_path"]
    dfrows = [
        ["01", "a", file_path],
        ["02", "a", file_path],
        ["03", "a", file_path],
    ]
    niidf = pd.DataFrame(dfrows, columns=dfcols)

    model = SecondLevelModel().fit(niidf)
    model.compute_contrast(first_level_contrast="a")


# -----------------------surface tests----------------------- #


def test_second_level_input_as_surface_image(surf_img_1d, n_subjects):
    """Test slm with a list surface images as input."""
    second_level_input = [surf_img_1d for _ in range(n_subjects)]

    design_matrix = pd.DataFrame(
        [1] * len(second_level_input), columns=["intercept"]
    )

    model = SecondLevelModel()
    model = model.fit(second_level_input, design_matrix=design_matrix)


def test_second_level_input_as_surface_image_3d(surf_img_2d, n_subjects):
    """Fit with surface image with all subjects as timepoints."""
    second_level_input = surf_img_2d(n_subjects)

    design_matrix = pd.DataFrame([1] * n_subjects, columns=["intercept"])

    model = SecondLevelModel()

    model.fit(second_level_input, design_matrix=design_matrix)


def test_second_level_input_error_surface_image_2d(surf_img_2d):
    """Err when passing a single 2D SurfaceImage with."""
    n_subjects = 1
    second_level_input = surf_img_2d(n_subjects)

    design_matrix = pd.DataFrame([1] * n_subjects, columns=["intercept"])

    model = SecondLevelModel()

    with pytest.raises(TypeError, match="must be a 3D SurfaceImage"):
        model.fit(second_level_input, design_matrix=design_matrix)


def test_second_level_input_as_surface_image_3d_same_as_list_2d(
    surf_img_1d, n_subjects
):
    """Fit all subjects as timepoints same as list of subject."""
    second_level_input = [surf_img_1d for _ in range(n_subjects)]

    design_matrix = pd.DataFrame([1] * n_subjects, columns=["intercept"])

    model = SecondLevelModel()
    model.fit(second_level_input, design_matrix=design_matrix)
    result_2d = model.compute_contrast()

    second_level_input_3d = concat_imgs(second_level_input)
    model.fit(second_level_input_3d, design_matrix=design_matrix)
    result_3d = model.compute_contrast()

    assert_surface_image_equal(result_2d, result_3d)


def test_second_level_input_as_surface_no_design_matrix(
    surf_img_1d, n_subjects
):
    """Raise error when design matrix is missing."""
    second_level_input = [surf_img_1d for _ in range(n_subjects)]

    model = SecondLevelModel()

    with pytest.raises(
        ValueError, match="require a design matrix to be provided"
    ):
        model.fit(second_level_input, design_matrix=None)


@pytest.mark.parametrize("surf_mask_dim", [1, 2])
def test_second_level_input_as_surface_image_with_mask(
    surf_img_1d, surf_mask_dim, surf_mask_1d, surf_mask_2d, n_subjects
):
    """Test slm with surface mask and a list surface images as input."""
    second_level_input = [surf_img_1d for _ in range(n_subjects)]

    design_matrix = pd.DataFrame(
        [1] * len(second_level_input), columns=["intercept"]
    )
    surf_mask = surf_mask_1d if surf_mask_dim == 1 else surf_mask_2d()

    model = SecondLevelModel(mask_img=surf_mask)
    model = model.fit(second_level_input, design_matrix=design_matrix)


def test_second_level_input_with_wrong_mask(
    surf_img_1d, surf_mask_1d, img_mask_mni, n_subjects
):
    """Test slm with mask of the wrong type."""
    second_level_input = [surf_img_1d for _ in range(n_subjects)]

    design_matrix = pd.DataFrame(
        [1] * len(second_level_input), columns=["intercept"]
    )

    # volume mask with surface data
    model = SecondLevelModel(mask_img=img_mask_mni)

    with pytest.raises(
        TypeError, match=r"Mask and input images must be of compatible types."
    ):
        model = model.fit(second_level_input, design_matrix=design_matrix)

    # surface mask with volume data
    func_img, _ = fake_fmri_data()
    second_level_input = [func_img] * 3
    model = SecondLevelModel(mask_img=surf_mask_1d)

    with pytest.raises(
        TypeError, match=r"Mask and input images must be of compatible types."
    ):
        model = model.fit(second_level_input, design_matrix=design_matrix)


def test_second_level_input_as_surface_image_warning_smoothing(
    surf_img_1d, n_subjects
):
    """Warn smoothing surface not implemented."""
    second_level_input = [surf_img_1d for _ in range(n_subjects)]

    design_matrix = pd.DataFrame(
        [1] * len(second_level_input), columns=["intercept"]
    )

    model = SecondLevelModel(smoothing_fwhm=8.0)
    with pytest.warns(NotImplementedWarning, match="not yet supported"):
        model = model.fit(second_level_input, design_matrix=design_matrix)


def test_second_level_input_as_flm_of_surface_image(
    surface_glm_data, n_subjects
):
    """Test fitting of list of first level model with surface data."""
    second_level_input = []
    for _ in range(n_subjects):
        img, des = surface_glm_data(5)
        model = FirstLevelModel()
        model.fit(img, design_matrices=des)
        second_level_input.append(model)

    design_matrix = pd.DataFrame(
        [1] * len(second_level_input), columns=["intercept"]
    )

    model = SecondLevelModel()
    model = model.fit(second_level_input, design_matrix=design_matrix)


def test_second_level_surface_image_contrast_computation(
    surf_img_1d, n_subjects
):
    """Check several types of contrast computation with surface SLM."""
    second_level_input = [surf_img_1d for _ in range(n_subjects)]

    design_matrix = pd.DataFrame(
        [1] * len(second_level_input), columns=["intercept"]
    )

    model = SecondLevelModel()

    model = model.fit(second_level_input, design_matrix=design_matrix)

    # simply pass nothing
    model.compute_contrast()

    # formula should work (passing variable name directly)
    model.compute_contrast("intercept")

    # smoke test for different contrasts in fixed effects
    ncol = len(model.design_matrix_.columns)
    c1, _ = np.eye(ncol)[0, :], np.zeros(ncol)
    model.compute_contrast(second_level_contrast=c1)

    # Test output_type='all', and verify images are equivalent
    all_images = model.compute_contrast(
        second_level_contrast=c1, output_type="all"
    )
    for key in [
        "z_score",
        "stat",
        "p_value",
        "effect_size",
        "effect_variance",
    ]:
        assert_surface_image_equal(
            all_images[key],
            model.compute_contrast(second_level_contrast=c1, output_type=key),
        )
