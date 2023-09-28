"""Test the second level model."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from nibabel import Nifti1Image, load
from numpy.testing import (
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
)
from scipy import stats

from nilearn._utils import testing
from nilearn._utils.data_gen import (
    generate_fake_fmri_data_and_design,
    write_fake_bold_img,
    write_fake_fmri_data_and_design,
)
from nilearn.glm.first_level import FirstLevelModel, run_glm
from nilearn.glm.second_level import SecondLevelModel, non_parametric_inference
from nilearn.glm.second_level.second_level import (
    _check_confounds,
    _check_design_matrix,
    _check_effect_maps,
    _check_first_level_contrast,
    _check_input_as_first_level_model,
    _check_output_type,
    _check_second_level_input,
    _get_contrast,
    _infer_effect_maps,
    _process_second_level_input_as_dataframe,
    _process_second_level_input_as_firstlevelmodels,
    _sort_input_dataframe,
)
from nilearn.image import concat_imgs, get_data, new_img_like, smooth_img
from nilearn.maskers import NiftiMasker

try:
    from nilearn.reporting import get_clusters_table
except ImportError:
    have_mpl = False
else:
    have_mpl = True

# This directory path
BASEDIR = os.path.dirname(os.path.abspath(__file__))
FUNCFILE = os.path.join(BASEDIR, "functional.nii.gz")

N_PERM = 10
SHAPE = (7, 8, 9, 1)


@pytest.fixture
def input_df():
    """Input DataFrame for testing."""
    return pd.DataFrame(
        {
            "effects_map_path": ["foo.nii", "bar.nii", "baz.nii"],
            "subject_label": ["foo", "bar", "baz"],
        }
    )


def fake_fmri_data(shape=SHAPE, file_path=None):
    if file_path is None:
        file_path = Path.cwd()
    shapes = (shape,)
    mask, FUNCFILE, _ = write_fake_fmri_data_and_design(
        shapes, file_path=file_path
    )
    FUNCFILE = FUNCFILE[0]
    func_img = load(FUNCFILE)
    return func_img, mask


def test_non_parametric_inference_with_flm_objects():
    """See https://github.com/nilearn/nilearn/issues/3579 ."""
    shapes, rk = [(7, 8, 9, 15)], 3
    mask, fmri_data, design_matrices = generate_fake_fmri_data_and_design(
        shapes, rk
    )

    masker = NiftiMasker(mask)
    masker.fit()
    single_session_model = FirstLevelModel(mask_img=masker).fit(
        fmri_data[0], design_matrices=design_matrices[0]
    )
    single_session_model.compute_contrast("x")

    second_level_input = [single_session_model, single_session_model]

    design_matrix = pd.DataFrame(
        [1] * len(second_level_input),
        columns=["intercept"],
    )

    non_parametric_inference(
        second_level_input=second_level_input,
        design_matrix=design_matrix,
        first_level_contrast="x",
        n_perm=N_PERM,
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

    assert output_df["subject_label"].values.tolist() == ["bar", "baz", "foo"]
    assert output_df["effects_map_path"].values.tolist() == [
        "bar.nii",
        "baz.nii",
        "foo.nii",
    ]


def test_second_level_input_as_3D_images():
    """Test second level model with a list 3D image filenames as input.

    Should act as a regression test for:
    https://github.com/nilearn/nilearn/issues/3636

    """
    shape = (7, 8, 9)
    images = []
    nb_subjects = 10
    affine = np.eye(4)
    for _ in range(nb_subjects):
        data = np.random.rand(*shape)
        images.append(Nifti1Image(data, affine))

    with testing.write_tmp_imgs(*images, create_files=True) as filenames:
        second_level_input = filenames
        design_matrix = pd.DataFrame(
            [1] * len(second_level_input),
            columns=["intercept"],
        )

        second_level_model = SecondLevelModel(smoothing_fwhm=8.0)
        second_level_model = second_level_model.fit(
            second_level_input,
            design_matrix=design_matrix,
        )


def test_process_second_level_input_as_firstlevelmodels():
    """Unit tests for function
    _process_second_level_input_as_firstlevelmodels().
    """
    shapes, rk = [(7, 8, 9, 15)], 3
    mask, fmri_data, design_matrices = generate_fake_fmri_data_and_design(
        shapes, rk
    )
    list_of_flm = [
        FirstLevelModel(mask_img=mask, subject_label=f"sub-{i}").fit(
            fmri_data[0], design_matrices=design_matrices[0]
        )
        for i in range(3)
    ]
    (
        sample_map,
        subjects_label,
    ) = _process_second_level_input_as_firstlevelmodels(list_of_flm)
    assert subjects_label == [f"sub-{i}" for i in range(3)]
    assert isinstance(sample_map, Nifti1Image)
    assert sample_map.shape == (7, 8, 9, 1)


def test_check_affine_first_level_models():
    shapes, rk = [(7, 8, 9, 15)], 3
    mask, fmri_data, design_matrices = generate_fake_fmri_data_and_design(
        shapes, rk
    )
    list_of_flm = [
        FirstLevelModel(mask_img=mask, subject_label=f"sub-{i}").fit(
            fmri_data[0], design_matrices=design_matrices[0]
        )
        for i in range(3)
    ]
    # should pass
    _check_input_as_first_level_model(
        second_level_input=list_of_flm, none_confounds=False
    )

    # add a model with a different affine
    # should raise an error
    mask, fmri_data, design_matrices = generate_fake_fmri_data_and_design(
        shapes, rk, affine=np.eye(4) * 2
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


def test_check_shape_first_level_models():
    shapes, rk = [(7, 8, 9, 15)], 3
    mask, fmri_data, design_matrices = generate_fake_fmri_data_and_design(
        shapes, rk
    )
    list_of_flm = [
        FirstLevelModel(mask_img=mask, subject_label=f"sub-{i}").fit(
            fmri_data[0], design_matrices=design_matrices[0]
        )
        for i in range(3)
    ]
    # should pass
    _check_input_as_first_level_model(
        second_level_input=list_of_flm, none_confounds=False
    )

    # add a model with a different shape
    # should raise an error
    shapes, rk = [(8, 9, 10, 15)], 3
    mask, fmri_data, design_matrices = generate_fake_fmri_data_and_design(
        shapes, rk
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


def test_check_second_level_input():
    with pytest.raises(
        TypeError,
        match="A second level model requires a list with at "
        "least two first level models or niimgs",
    ):
        _check_second_level_input([FirstLevelModel()], pd.DataFrame())
    with pytest.raises(
        ValueError, match="Model sub_1 at index 0 has not been fit yet"
    ):
        _check_second_level_input(
            [FirstLevelModel(subject_label=f"sub_{i}") for i in range(1, 3)],
            pd.DataFrame(),
        )

    shapes, rk = [(7, 8, 9, 15)], 3
    mask, fmri_data, design_matrices = generate_fake_fmri_data_and_design(
        shapes, rk
    )
    input_models = [
        FirstLevelModel(mask_img=mask).fit(
            fmri_data[0], design_matrices=design_matrices[0]
        )
    ]
    obj = lambda: None  # noqa : E731
    obj.results_ = "foo"
    obj.labels_ = "bar"

    with pytest.raises(
        TypeError, match="Got object type <class 'function'> at idx 1"
    ):
        _check_second_level_input(input_models + [obj], pd.DataFrame())
    with pytest.raises(
        ValueError,
        match="In case confounds are provided, first level "
        "objects need to provide the attribute 'subject_label'",
    ):
        _check_second_level_input(
            input_models * 2, pd.DataFrame(), confounds=pd.DataFrame()
        )
    with pytest.raises(
        ValueError,
        match="List of niimgs as second_level_input "
        "require a design matrix to be provided",
    ):
        _check_second_level_input(fmri_data * 2, None)
    _check_second_level_input(fmri_data[0], pd.DataFrame())

    with pytest.raises(
        TypeError, match="Got object type <class 'int'> at idx 1"
    ):
        _check_second_level_input(["foo", 1], pd.DataFrame())
    with pytest.raises(
        ValueError,
        match="second_level_input DataFrame must have columns "
        "subject_label, map_name and effects_map_path",
    ):
        _check_second_level_input(
            pd.DataFrame(columns=["foo", "bar"]), pd.DataFrame()
        )
    with pytest.raises(
        ValueError, match="subject_label column must contain only strings"
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
    with pytest.raises(
        ValueError,
        match="List of niimgs as second_level_input "
        "require a design matrix to be provided",
    ):
        _check_second_level_input(fmri_data[0], None)
    with pytest.raises(TypeError, match="second_level_input must be"):
        _check_second_level_input(1, None)
    with pytest.raises(TypeError, match="second_level_input must be"):
        _check_second_level_input(1, None)


def test_check_output_type():
    _check_output_type(int, [str, int, float])
    with pytest.raises(ValueError, match="output_type must be one of"):
        _check_output_type("foo", [str, int, float])


def test_check_design_matrix():
    _check_design_matrix(None)  # Should not do anything
    _check_design_matrix(pd.DataFrame())
    with pytest.raises(
        ValueError, match="design matrix must be a pandas DataFrame"
    ):
        _check_design_matrix("foo")


def test_check_confounds():
    _check_confounds(None)  # Should not do anything
    with pytest.raises(
        ValueError, match="confounds must be a pandas DataFrame"
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
    _check_first_level_contrast(["foo"], None)  # Should not do anything
    _check_first_level_contrast([FirstLevelModel()], "foo")
    with pytest.raises(ValueError, match="If second_level_input was a list"):
        _check_first_level_contrast([FirstLevelModel()], None)


def test_check_effect_maps():
    _check_effect_maps([1, 2, 3], np.array([[1, 2], [3, 4], [5, 6]]))
    with pytest.raises(
        ValueError,
        match="design_matrix does not match the number of maps considered",
    ):
        _check_effect_maps([1, 2], np.array([[1, 2], [3, 4], [5, 6]]))


def test_get_contrast():
    design_matrix = pd.DataFrame([1, 2, 3], columns=["conf"])
    assert _get_contrast("conf", design_matrix) == "conf"

    design_matrix = pd.DataFrame({"conf1": [1, 2, 3], "conf2": [4, 5, 6]})
    assert _get_contrast([0, 1], design_matrix) == "conf2"
    assert _get_contrast([1, 0], design_matrix) == "conf1"


def test_get_contrast_errors():
    design_matrix = pd.DataFrame([1, 2, 3], columns=["conf"])
    with pytest.raises(ValueError, match='"foo" is not a valid contrast name'):
        _get_contrast("foo", design_matrix)

    design_matrix = pd.DataFrame({"conf1": [1, 2, 3], "conf2": [4, 5, 6]})
    with pytest.raises(
        ValueError, match="No second-level contrast is specified."
    ):
        _get_contrast(None, design_matrix)
    with pytest.raises(
        ValueError,
        match="second_level_contrast must be a list of 0s and 1s",
    ):
        _get_contrast([0, 0], design_matrix)


def test_infer_effect_maps(tmp_path):
    shapes, rk = (SHAPE, (7, 8, 7, 16)), 3
    mask, fmri_data, design_matrices = write_fake_fmri_data_and_design(
        shapes, rk, file_path=tmp_path
    )
    second_level_input = pd.DataFrame(
        {"map_name": ["a", "b"], "effects_map_path": [fmri_data[0], "bar"]}
    )

    assert _infer_effect_maps(second_level_input, "a") == [fmri_data[0]]
    assert _infer_effect_maps([fmri_data[0]], None) == [fmri_data[0]]

    contrast = np.eye(rk)[1]
    second_level_input = [FirstLevelModel(mask_img=mask)] * 2
    for i, model in enumerate(second_level_input):
        model.fit(fmri_data[i], design_matrices=design_matrices[i])

    assert len(_infer_effect_maps(second_level_input, contrast)) == 2


def test_infer_effect_maps_error(tmp_path):
    shapes, rk = (SHAPE, (7, 8, 7, 16)), 3
    _, fmri_data, _ = write_fake_fmri_data_and_design(
        shapes, rk, file_path=tmp_path
    )
    second_level_input = pd.DataFrame(
        {"map_name": ["a", "b"], "effects_map_path": [fmri_data[0], "bar"]}
    )
    with pytest.raises(ValueError, match="File not found: 'bar'"):
        _infer_effect_maps(second_level_input, "b")


def test_high_level_glm_with_paths(tmp_path):
    func_img, mask = fake_fmri_data(file_path=tmp_path)

    model = SecondLevelModel(mask_img=mask)

    # fit model
    Y = [func_img] * 4
    X = pd.DataFrame([[1]] * 4, columns=["intercept"])
    model = model.fit(Y, design_matrix=X)
    c1 = np.eye(len(model.design_matrix_.columns))[0]
    z_image = model.compute_contrast(c1, output_type="z_score")

    assert isinstance(z_image, Nifti1Image)
    assert_array_equal(z_image.affine, load(mask).affine)

    # try with target_shape
    target_shape = (10, 10, 10)
    target_affine = np.eye(4)
    target_affine[0, 3] = 1
    model = SecondLevelModel(
        mask_img=mask,
        target_shape=target_shape,
        target_affine=target_affine,
    )
    z_image = model.fit(Y, design_matrix=X).compute_contrast(c1)

    assert_array_equal(z_image.shape, target_shape)
    assert_array_equal(z_image.affine, target_affine)


def test_high_level_glm_with_paths_errors(tmp_path):
    func_img, mask = fake_fmri_data(file_path=tmp_path)

    model = SecondLevelModel(mask_img=mask)

    # asking for contrast before model fit gives error
    with pytest.raises(ValueError):
        model.compute_contrast([])

    # fit model
    Y = [func_img] * 4
    X = pd.DataFrame([[1]] * 4, columns=["intercept"])

    # Provide a masker as mask_img
    masker = NiftiMasker(mask)
    with pytest.warns(
        UserWarning, match="Parameter memory of the masker overridden"
    ):
        SecondLevelModel(mask_img=masker, verbose=1).fit(Y, design_matrix=X)


def test_high_level_non_parametric_inference_with_paths(tmp_path):
    shapes = (SHAPE,)
    mask, FUNCFILE, _ = write_fake_fmri_data_and_design(
        shapes, file_path=tmp_path
    )
    FUNCFILE = FUNCFILE[0]
    df_input = pd.DataFrame(
        {
            "subject_label": [f"sub-{i}" for i in range(4)],
            "effects_map_path": [FUNCFILE] * 4,
            "map_name": [FUNCFILE] * 4,
        }
    )
    func_img = load(FUNCFILE)
    Y = [func_img] * 4
    X = pd.DataFrame([[1]] * 4, columns=["intercept"])
    c1 = np.eye(len(X.columns))[0]
    neg_log_pvals_imgs = [
        non_parametric_inference(
            second_level_input,
            design_matrix=X,
            second_level_contrast=c1,
            first_level_contrast=FUNCFILE,
            mask=mask,
            n_perm=N_PERM,
            verbose=1,
        )
        for second_level_input in [Y, df_input]
    ]

    assert all(isinstance(img, Nifti1Image) for img in neg_log_pvals_imgs)
    for img in neg_log_pvals_imgs:
        assert_array_equal(img.affine, load(mask).affine)

    neg_log_pvals_list = [get_data(i) for i in neg_log_pvals_imgs]
    for neg_log_pvals in neg_log_pvals_list:
        assert np.all(neg_log_pvals <= -np.log10(1.0 / (N_PERM + 1)))
        assert np.all(0 <= neg_log_pvals)


def test_high_level_non_parametric_inference_with_paths_warning(tmp_path):
    func_img, mask = fake_fmri_data(file_path=tmp_path)
    Y = [func_img] * 4
    X = pd.DataFrame([[1]] * 4, columns=["intercept"])
    c1 = np.eye(len(X.columns))[0]

    masker = NiftiMasker(mask, smoothing_fwhm=2.0)
    with pytest.warns(
        UserWarning,
        match="Parameter smoothing_fwhm of the masker overridden",
    ):
        non_parametric_inference(
            Y,
            design_matrix=X,
            second_level_contrast=c1,
            smoothing_fwhm=3.0,
            mask=masker,
            n_perm=N_PERM,
        )


def test_fmri_inputs(tmp_path):
    # Test processing of FMRI inputs
    # prepare fake data
    rng = np.random.RandomState(42)
    p, q = 80, 10
    X = rng.standard_normal(size=(p, q))
    shapes = ((7, 8, 9, 10),)
    mask, FUNCFILE, _ = write_fake_fmri_data_and_design(
        shapes, file_path=tmp_path
    )
    FUNCFILE = FUNCFILE[0]
    func_img = load(FUNCFILE)
    T = func_img.shape[-1]
    des = pd.DataFrame(np.ones((T, 1)), columns=["a"])
    des_fname = str(tmp_path / "design.csv")
    des.to_csv(des_fname)

    # prepare correct input first level models
    flm = FirstLevelModel(subject_label="01").fit(
        FUNCFILE, design_matrices=des
    )

    # prepare correct input dataframe and lists
    shapes = (SHAPE,)
    _, FUNCFILE, _ = write_fake_fmri_data_and_design(
        shapes, file_path=tmp_path
    )
    FUNCFILE = FUNCFILE[0]

    confounds = pd.DataFrame(
        [["01", 1], ["02", 2], ["03", 3]],
        columns=["subject_label", "conf1"],
    )
    sdes = pd.DataFrame(X[:3, :3], columns=["intercept", "b", "c"])

    # smoke tests with correct input

    flms = [flm, flm, flm]

    # First level models as input
    SecondLevelModel(mask_img=mask).fit(flms)
    SecondLevelModel().fit(flms)
    # Note : the following one creates a singular design matrix
    SecondLevelModel().fit(flms, confounds)
    SecondLevelModel().fit(flms, None, sdes)

    # dataframes as input
    dfcols = ["subject_label", "map_name", "effects_map_path"]
    dfrows = [
        ["01", "a", FUNCFILE],
        ["02", "a", FUNCFILE],
        ["03", "a", FUNCFILE],
    ]
    niidf = pd.DataFrame(dfrows, columns=dfcols)

    SecondLevelModel().fit(niidf)
    SecondLevelModel().fit(niidf, confounds)
    SecondLevelModel().fit(niidf, confounds, sdes)
    SecondLevelModel().fit(niidf, None, sdes)

    # niimgs as input
    niimgs = [FUNCFILE, FUNCFILE, FUNCFILE]
    SecondLevelModel().fit(niimgs, None, sdes)

    # 4d niimg as input
    niimg_4d = concat_imgs(niimgs)
    SecondLevelModel().fit(niimg_4d, None, sdes)


def test_fmri_inputs_errors(tmp_path):
    # Test processing of FMRI inputs
    # prepare fake data
    shapes = ((7, 8, 9, 10),)
    _, FUNCFILE, _ = write_fake_fmri_data_and_design(
        shapes, file_path=tmp_path
    )
    FUNCFILE = FUNCFILE[0]
    func_img = load(FUNCFILE)
    n_samples = func_img.shape[-1]
    des = pd.DataFrame(np.ones((n_samples, 1)), columns=["a"])
    des_fname = str(tmp_path / "design.csv")
    des.to_csv(des_fname)

    # prepare correct input first level models
    flm = FirstLevelModel(subject_label="01").fit(
        FUNCFILE, design_matrices=des
    )

    # prepare correct input
    shapes = (SHAPE,)
    _, FUNCFILE, _ = write_fake_fmri_data_and_design(
        shapes, file_path=tmp_path
    )
    FUNCFILE = FUNCFILE[0]

    # test wrong input errors

    # test first level model requirements
    with pytest.raises(TypeError, match="second_level_input must be"):
        SecondLevelModel().fit(flm)
    with pytest.raises(TypeError, match="at least two"):
        SecondLevelModel().fit([flm])

    # test dataframe requirements
    dfcols = ["subject_label", "map_name", "effects_map_path"]
    dfrows = [
        ["01", "a", FUNCFILE],
        ["02", "a", FUNCFILE],
        ["03", "a", FUNCFILE],
    ]
    niidf = pd.DataFrame(dfrows, columns=dfcols)
    with pytest.raises(TypeError, match="second_level_input must be"):
        SecondLevelModel().fit(niidf["subject_label"])

    confounds = pd.DataFrame(
        [["01", 1], ["02", 2], ["03", 3]],
        columns=["subject_label", "conf1"],
    )

    # test niimgs requirements
    niimgs = [FUNCFILE, FUNCFILE, FUNCFILE]
    with pytest.raises(ValueError, match="require a design matrix"):
        SecondLevelModel().fit(niimgs)
    with pytest.raises(TypeError):
        SecondLevelModel().fit(niimgs + [[]], confounds)

    # test first_level_conditions, confounds, and design
    flms = [flm, flm, flm]
    with pytest.raises(ValueError):
        SecondLevelModel().fit(flms, ["", []])
    with pytest.raises(ValueError):
        SecondLevelModel().fit(flms, [])
    with pytest.raises(ValueError):
        SecondLevelModel().fit(flms, confounds["conf1"])
    with pytest.raises(ValueError):
        SecondLevelModel().fit(flms, None, [])


def test_fmri_inputs_for_non_parametric_inference_errors(tmp_path):
    # Test processing of FMRI inputs

    # prepare fake data
    rng = np.random.RandomState(42)
    p, q = 80, 10
    X = rng.standard_normal(size=(p, q))
    shapes = ((7, 8, 9, 10),)
    _, func_file, _ = write_fake_fmri_data_and_design(
        shapes, file_path=tmp_path
    )

    func_file = func_file[0]

    func_img = load(func_file)
    T = func_img.shape[-1]
    des = pd.DataFrame(np.ones((T, 1)), columns=["a"])
    des_fname = str(tmp_path / "design.csv")
    des.to_csv(des_fname)

    # prepare correct input first level models
    flm = FirstLevelModel(subject_label="01").fit(
        func_file, design_matrices=des
    )
    # prepare correct input dataframe and lists
    shapes = (SHAPE,)
    _, func_file, _ = write_fake_fmri_data_and_design(
        shapes, file_path=tmp_path
    )
    func_file = func_file[0]

    niimgs = [func_file, func_file, func_file]
    niimg_4d = concat_imgs(niimgs)
    confounds = pd.DataFrame(
        [["01", 1], ["02", 2], ["03", 3]],
        columns=["subject_label", "conf1"],
    )
    sdes = pd.DataFrame(X[:3, :3], columns=["intercept", "b", "c"])

    # test missing second-level contrast

    # niimgs as input
    with pytest.raises(ValueError):
        non_parametric_inference(niimgs, None, sdes)
    with pytest.raises(ValueError):
        non_parametric_inference(niimgs, confounds, sdes)

    # 4d niimg as input
    with pytest.raises(ValueError):
        non_parametric_inference(niimg_4d, None, sdes)

    # test wrong input errors
    # test first level model
    with pytest.raises(TypeError, match="second_level_input must be"):
        non_parametric_inference(flm)

    # test list of less than two niimgs
    with pytest.raises(TypeError, match="at least two"):
        non_parametric_inference([func_file])

    # test niimgs requirements
    with pytest.raises(ValueError, match="require a design matrix"):
        non_parametric_inference(niimgs)
    with pytest.raises(TypeError):
        non_parametric_inference(niimgs + [[]], confounds)

    # test other objects
    with pytest.raises(ValueError):
        non_parametric_inference("random string object")


def test_second_level_glm_computation(tmp_path):
    func_img, mask = fake_fmri_data(file_path=tmp_path)

    model = SecondLevelModel(mask_img=mask)
    Y = [func_img] * 4
    X = pd.DataFrame([[1]] * 4, columns=["intercept"])

    model = model.fit(Y, design_matrix=X)
    model.compute_contrast()
    labels1 = model.labels_
    results1 = model.results_

    labels2, results2 = run_glm(model.masker_.transform(Y), X.values, "ols")
    assert_almost_equal(labels1, labels2, decimal=1)

    assert len(results1) == len(results2)


@pytest.mark.parametrize("attribute", ["residuals", "predicted", "r_square"])
def test_second_level_voxelwise_attribute_errors(attribute):
    """Tests that an error is raised when trying to access
    voxelwise attributes before fitting the model, before
    computing a contrast, and when not setting
    ``minimize_memory`` to ``True``.
    """
    shapes = (SHAPE,)
    mask, fmri_data, _ = generate_fake_fmri_data_and_design(shapes)
    model = SecondLevelModel(mask_img=mask, minimize_memory=False)

    with pytest.raises(ValueError, match="The model has no results."):
        getattr(model, attribute)

    Y = fmri_data * 4
    X = pd.DataFrame([[1]] * 4, columns=["intercept"])
    model.fit(Y, design_matrix=X)

    with pytest.raises(ValueError, match="The model has no results."):
        getattr(model, attribute)
    with pytest.raises(ValueError, match="attribute must be one of"):
        model._get_voxelwise_model_attribute("foo", True)

    model = SecondLevelModel(mask_img=mask, minimize_memory=True)
    model.fit(Y, design_matrix=X)
    model.compute_contrast()

    with pytest.raises(ValueError, match="To access voxelwise attributes"):
        getattr(model, attribute)


@pytest.mark.parametrize("attribute", ["residuals", "predicted", "r_square"])
def test_second_level_voxelwise_attribute(attribute):
    """Smoke test for voxelwise attributes for SecondLevelModel."""
    shapes = (SHAPE,)
    mask, fmri_data, _ = generate_fake_fmri_data_and_design(shapes)
    model = SecondLevelModel(mask_img=mask, minimize_memory=False)
    Y = fmri_data * 4
    X = pd.DataFrame([[1]] * 4, columns=["intercept"])
    model.fit(Y, design_matrix=X)
    model.compute_contrast()
    getattr(model, attribute)


def test_second_level_residuals():
    """Tests residuals computation for SecondLevelModel."""
    shapes = (SHAPE,)
    mask, fmri_data, _ = generate_fake_fmri_data_and_design(shapes)
    model = SecondLevelModel(mask_img=mask, minimize_memory=False)
    Y = fmri_data * 4
    X = pd.DataFrame([[1]] * 4, columns=["intercept"])
    model.fit(Y, design_matrix=X)
    model.compute_contrast()

    assert isinstance(model.residuals, Nifti1Image)
    assert model.residuals.shape == (7, 8, 9, 4)
    mean_residuals = model.masker_.transform(model.residuals).mean(0)
    assert_array_almost_equal(mean_residuals, 0)


def test_non_parametric_inference_permutation_computation(tmp_path):
    func_img, mask = fake_fmri_data(file_path=tmp_path)

    Y = [func_img] * 4
    X = pd.DataFrame([[1]] * 4, columns=["intercept"])

    neg_log_pvals_img = non_parametric_inference(
        Y, design_matrix=X, model_intercept=False, mask=mask, n_perm=N_PERM
    )

    assert get_data(neg_log_pvals_img).shape == SHAPE[:3]


def test_non_parametric_inference_tfce(tmp_path):
    """Test non-parametric inference with TFCE inference."""
    shapes = [SHAPE] * 4
    mask, FUNCFILES, _ = write_fake_fmri_data_and_design(
        shapes, file_path=tmp_path
    )
    X = pd.DataFrame([[1]] * 4, columns=["intercept"])

    out = non_parametric_inference(
        FUNCFILES,
        design_matrix=X,
        model_intercept=False,
        mask=mask,
        n_perm=N_PERM,
        tfce=True,
    )
    assert isinstance(out, dict)
    assert "t" in out.keys()
    assert "tfce" in out.keys()
    assert "logp_max_t" in out.keys()
    assert "logp_max_tfce" in out.keys()

    assert get_data(out["tfce"]).shape == shapes[0][:3]
    assert get_data(out["logp_max_tfce"]).shape == shapes[0][:3]


def test_non_parametric_inference_cluster_level(tmp_path):
    """Test non-parametric inference with cluster-level inference."""
    func_img, mask = fake_fmri_data(file_path=tmp_path)

    Y = [func_img] * 4
    X = pd.DataFrame([[1]] * 4, columns=["intercept"])

    out = non_parametric_inference(
        Y,
        design_matrix=X,
        model_intercept=False,
        mask=mask,
        n_perm=N_PERM,
        threshold=0.001,
    )
    assert isinstance(out, dict)
    assert "t" in out.keys()
    assert "logp_max_t" in out.keys()
    assert "size" in out.keys()
    assert "logp_max_size" in out.keys()
    assert "mass" in out.keys()
    assert "logp_max_mass" in out.keys()

    assert get_data(out["logp_max_t"]).shape == SHAPE[:3]


@pytest.mark.skipif(
    not have_mpl, reason="Matplotlib not installed; required for this test"
)
def test_non_parametric_inference_cluster_level_with_covariates(
    tmp_path,
    random_state=0,
):
    """Test non-parametric inference with cluster-level inference in \
    the context of covariates."""
    rng = np.random.RandomState(random_state)

    shapes = ((7, 8, 9, 1),)
    mask, FUNCFILE, _ = write_fake_fmri_data_and_design(
        shapes, file_path=tmp_path
    )
    func_img = load(FUNCFILE[0])

    unc_pval = 0.01
    n_subjects = 2

    # Set up one sample t-test design with two random covariates
    cov1 = rng.random(n_subjects)
    cov2 = rng.random(n_subjects)
    X = pd.DataFrame({"cov1": cov1, "cov2": cov2, "intercept": 1})

    # make sure there is variability in the images
    kernels = rng.uniform(low=0, high=5, size=n_subjects)
    Y = [smooth_img(func_img, kernel) for kernel in kernels]

    # Set up non-parametric test
    out = non_parametric_inference(
        Y,
        design_matrix=X,
        mask=mask,
        model_intercept=False,
        second_level_contrast="intercept",
        n_perm=1 / unc_pval,
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


def test_second_level_contrast_computation(tmp_path):
    func_img, mask = fake_fmri_data(file_path=tmp_path)

    model = SecondLevelModel(mask_img=mask)

    # fit model
    Y = [func_img] * 4
    X = pd.DataFrame([[1]] * 4, columns=["intercept"])
    model = model.fit(Y, design_matrix=X)

    ncol = len(model.design_matrix_.columns)
    c1, _ = np.eye(ncol)[0, :], np.zeros(ncol)

    # smoke test for different contrasts in fixed effects
    model.compute_contrast(second_level_contrast=c1)
    z_image = model.compute_contrast(
        second_level_contrast=c1, output_type="z_score"
    )
    stat_image = model.compute_contrast(
        second_level_contrast=c1, output_type="stat"
    )
    p_image = model.compute_contrast(
        second_level_contrast=c1, output_type="p_value"
    )
    effect_image = model.compute_contrast(
        second_level_contrast=c1, output_type="effect_size"
    )
    variance_image = model.compute_contrast(
        second_level_contrast=c1, output_type="effect_variance"
    )

    # Test output_type='all', and verify images are equivalent
    all_images = model.compute_contrast(
        second_level_contrast=c1, output_type="all"
    )
    assert_array_equal(get_data(all_images["z_score"]), get_data(z_image))
    assert_array_equal(get_data(all_images["stat"]), get_data(stat_image))
    assert_array_equal(get_data(all_images["p_value"]), get_data(p_image))
    assert_array_equal(
        get_data(all_images["effect_size"]), get_data(effect_image)
    )
    assert_array_equal(
        get_data(all_images["effect_variance"]), get_data(variance_image)
    )

    # formula should work (passing variable name directly)
    model.compute_contrast("intercept")
    # or simply pass nothing
    model.compute_contrast()

    # formula as contrasts
    rng = np.random.RandomState(42)
    X = pd.DataFrame(rng.uniform(size=(4, 2)), columns=["r1", "r2"])
    model = model.fit(Y, design_matrix=X)
    model.compute_contrast(second_level_contrast="r1 - r2")


def test_second_level_contrast_computation_errors(tmp_path):
    func_img, mask = fake_fmri_data(file_path=tmp_path)

    model = SecondLevelModel(mask_img=mask)

    # asking for contrast before model fit gives error
    with pytest.raises(ValueError, match="The model has not been fit yet"):
        model.compute_contrast(second_level_contrast="intercept")

    # fit model
    Y = [func_img] * 4
    X = pd.DataFrame([[1]] * 4, columns=["intercept"])
    model = model.fit(Y, design_matrix=X)
    ncol = len(model.design_matrix_.columns)
    c1, cnull = np.eye(ncol)[0, :], np.zeros(ncol)

    # formula should work (passing variable name directly)
    model.compute_contrast(second_level_contrast="intercept")
    # or simply pass nothing
    model.compute_contrast()

    # passing null contrast should give back a value error
    with pytest.raises(ValueError):
        model.compute_contrast(cnull)

    # passing wrong parameters
    with pytest.raises(
        ValueError,
        match=("t contrasts should be length P=1, but this is length 0"),
    ):
        model.compute_contrast(second_level_contrast=[])
    with pytest.raises(ValueError, match="Allowed types are .*'t', 'F'"):
        model.compute_contrast(
            second_level_contrast=c1, second_level_stat_type=""
        )
    with pytest.raises(ValueError, match="Allowed types are .*'t', 'F'"):
        model.compute_contrast(
            second_level_contrast=c1, second_level_stat_type=[]
        )
    with pytest.raises(ValueError, match="output_type must be one of "):
        model.compute_contrast(second_level_contrast=c1, output_type="")

    # check that passing no explicit contrast when the design
    # matrix has more than one columns raises an error
    rng = np.random.RandomState(42)
    X = pd.DataFrame(rng.uniform(size=(4, 2)), columns=["r1", "r2"])
    model = model.fit(Y, design_matrix=X)
    with pytest.raises(
        ValueError, match="No second-level contrast is specified"
    ):
        model.compute_contrast(None)
    with pytest.raises(
        ValueError,
        match=("t contrasts should be length P=2, but this is length 1"),
    ):
        model.compute_contrast([1])


def test_non_parametric_inference_contrast_computation(tmp_path):
    func_img, mask = fake_fmri_data(file_path=tmp_path)

    # fit model
    Y = [func_img] * 4
    X = pd.DataFrame([[1]] * 4, columns=["intercept"])
    # formula should work without second-level contrast
    non_parametric_inference(
        Y, design_matrix=X, model_intercept=False, mask=mask, n_perm=N_PERM
    )

    ncol = len(X.columns)
    c1, _ = np.eye(ncol)[0, :], np.zeros(ncol)
    # formula should work with second-level contrast
    non_parametric_inference(
        Y,
        design_matrix=X,
        model_intercept=False,
        second_level_contrast=c1,
        mask=mask,
        n_perm=N_PERM,
    )
    # formula should work passing variable name directly
    non_parametric_inference(
        Y,
        design_matrix=X,
        second_level_contrast="intercept",
        model_intercept=False,
        mask=mask,
        n_perm=N_PERM,
    )


@pytest.mark.parametrize("second_level_contrast", [[1, 0], "r1"])
def test_non_parametric_inference_contrast_formula(
    tmp_path, second_level_contrast
):
    func_img, _ = fake_fmri_data(file_path=tmp_path)
    Y = [func_img] * 4
    rng = np.random.RandomState(42)
    X = pd.DataFrame(rng.uniform(size=(4, 2)), columns=["r1", "r2"])

    non_parametric_inference(
        second_level_input=Y,
        design_matrix=X,
        second_level_contrast=second_level_contrast,
    )


def test_non_parametric_inference_contrast_computation_errors(tmp_path):
    func_img, mask = fake_fmri_data(file_path=tmp_path)

    # asking for contrast before model fit gives error
    with pytest.raises(TypeError, match="second_level_input must be either"):
        non_parametric_inference(
            second_level_input=None,
            second_level_contrast="intercept",
            mask=mask,
        )

    # fit model
    Y = [func_img] * 4
    X = pd.DataFrame([[1]] * 4, columns=["intercept"])

    ncol = len(X.columns)
    _, cnull = np.eye(ncol)[0, :], np.zeros(ncol)

    # passing null contrast should give back a value error
    with pytest.raises(
        ValueError,
        match=("second_level_contrast must be a list of 0s and 1s."),
    ):
        non_parametric_inference(
            second_level_input=Y,
            design_matrix=X,
            second_level_contrast=cnull,
            mask=mask,
        )
    with pytest.raises(
        ValueError,
        match=("second_level_contrast must be a list of 0s and 1s."),
    ):
        non_parametric_inference(
            second_level_input=Y,
            design_matrix=X,
            second_level_contrast=[],
            mask=mask,
        )

    # check that passing no explicit contrast when the design
    # matrix has more than one columns raises an error
    rng = np.random.RandomState(42)
    X = pd.DataFrame(rng.uniform(size=(4, 2)), columns=["r1", "r2"])
    with pytest.raises(
        ValueError, match="No second-level contrast is specified."
    ):
        non_parametric_inference(
            second_level_input=Y,
            design_matrix=X,
            second_level_contrast=None,
        )


def test_second_level_contrast_computation_with_memory_caching(tmp_path):
    func_img, mask = fake_fmri_data(file_path=tmp_path)

    model = SecondLevelModel(mask_img=mask, memory="nilearn_cache")

    # fit model
    Y = [func_img] * 4
    X = pd.DataFrame([[1]] * 4, columns=["intercept"])
    model = model.fit(Y, design_matrix=X)
    ncol = len(model.design_matrix_.columns)
    c1 = np.eye(ncol)[0, :]
    # test memory caching for compute_contrast
    model.compute_contrast(c1, output_type="z_score")
    # or simply pass nothing
    model.compute_contrast()


def test_second_lvl_dataframe_computation(tmp_path):
    """Check that contrast can be computed when using dataframes as input.

    See bug https://github.com/nilearn/nilearn/issues/3871
    """
    shape = (7, 8, 9, 1)
    file_path = write_fake_bold_img(
        file_path=tmp_path / "img.nii.gz", shape=shape
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
