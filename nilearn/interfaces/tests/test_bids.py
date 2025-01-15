"""Tests for the nilearn.interfaces.bids submodule."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nilearn._utils.data_gen import (
    add_metadata_to_bids_dataset,
    create_fake_bids_dataset,
    generate_fake_fmri_data_and_design,
)
from nilearn._utils.helpers import is_matplotlib_installed
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.second_level import SecondLevelModel
from nilearn.interfaces.bids import (
    get_bids_files,
    parse_bids_filename,
    save_glm_to_bids,
)
from nilearn.interfaces.bids.query import (
    _get_metadata_from_bids,
    infer_repetition_time_from_dataset,
    infer_slice_timing_start_time_from_dataset,
)
from nilearn.maskers import NiftiMasker


def test_get_metadata_from_bids(tmp_path):
    """Ensure that metadata is correctly extracted from BIDS JSON files.

    Throw a warning when the field is not found.
    Throw a warning when there is no JSON file.
    """
    json_file = tmp_path / "sub-01_task-main_bold.json"
    json_files = [json_file]

    with json_file.open("w") as f:
        json.dump({"RepetitionTime": 2.0}, f)
    value = _get_metadata_from_bids(
        field="RepetitionTime", json_files=json_files
    )
    assert value == 2.0

    with json_file.open("w") as f:
        json.dump({"foo": 2.0}, f)
    with pytest.warns(UserWarning, match="'RepetitionTime' not found"):
        value = _get_metadata_from_bids(
            field="RepetitionTime", json_files=json_files
        )

    json_files = []
    with pytest.warns(UserWarning, match="No .*json found in BIDS"):
        value = _get_metadata_from_bids(
            field="RepetitionTime", json_files=json_files
        )
        assert value is None


def test_infer_repetition_time_from_dataset(tmp_path):
    """Test inferring repetition time from the BIDS dataset.

    When using create_fake_bids_dataset the value is 1.5 secs by default
    in the raw dataset.
    When using add_metadata_to_bids_dataset the value is 2.0 secs.
    """
    bids_path = create_fake_bids_dataset(
        base_dir=tmp_path, n_sub=1, n_ses=1, tasks=["main"], n_runs=[1]
    )

    t_r = infer_repetition_time_from_dataset(
        bids_path=tmp_path / bids_path, filters=[("task", "main")]
    )

    expected_t_r = 1.5
    assert t_r == expected_t_r

    expected_t_r = 2.0
    add_metadata_to_bids_dataset(
        bids_path=tmp_path / bids_path,
        metadata={"RepetitionTime": expected_t_r},
    )

    t_r = infer_repetition_time_from_dataset(
        bids_path=tmp_path / bids_path / "derivatives",
        filters=[("task", "main"), ("run", "01")],
    )

    assert t_r == expected_t_r


def test_infer_slice_timing_start_time_from_dataset(tmp_path):
    """Test inferring slice timing start time from the BIDS dataset.

    create_fake_bids_dataset does not add slice timing information
    by default so the value returned will be None.

    If the metadata is added to the BIDS dataset,
    then this value should be returned.
    """
    bids_path = create_fake_bids_dataset(
        base_dir=tmp_path, n_sub=1, n_ses=1, tasks=["main"], n_runs=[1]
    )

    StartTime = infer_slice_timing_start_time_from_dataset(
        bids_path=tmp_path / bids_path / "derivatives",
        filters=[("task", "main")],
    )

    expected_StartTime = None
    assert StartTime is expected_StartTime

    expected_StartTime = 1.0
    add_metadata_to_bids_dataset(
        bids_path=tmp_path / bids_path,
        metadata={"StartTime": expected_StartTime},
    )

    StartTime = infer_slice_timing_start_time_from_dataset(
        bids_path=tmp_path / bids_path / "derivatives",
        filters=[("task", "main")],
    )

    assert StartTime == expected_StartTime


def _rm_all_json_files_from_bids_dataset(bids_path):
    """Remove all json and make sure that get_bids_files does not find any."""
    for x in bids_path.glob("**/*.json"):
        x.unlink()
    selection = get_bids_files(bids_path, file_type="json", sub_folder=True)
    assert selection == []
    selection = get_bids_files(bids_path, file_type="json", sub_folder=False)
    assert selection == []


def test_get_bids_files_inheritance_principle_root_folder(tmp_path):
    """Check if json files are found if in root folder of a dataset.

    see https://bids-specification.readthedocs.io/en/latest/common-principles.html#the-inheritance-principle
    """
    bids_path = create_fake_bids_dataset(
        base_dir=tmp_path, n_sub=1, n_ses=1, tasks=["main"], n_runs=[1]
    )

    _rm_all_json_files_from_bids_dataset(bids_path)

    # add json file to root of dataset
    json_file = "task-main_bold.json"
    json_file = add_metadata_to_bids_dataset(
        bids_path=bids_path,
        metadata={"RepetitionTime": 1.5},
        json_file=json_file,
    )
    assert json_file.exists()

    # make sure that get_bids_files finds the json file
    # but only when looking in root of dataset
    selection = get_bids_files(
        bids_path,
        file_tag="bold",
        file_type="json",
        filters=[("task", "main")],
        sub_folder=True,
    )
    assert selection == []
    selection = get_bids_files(
        bids_path,
        file_tag="bold",
        file_type="json",
        filters=[("task", "main")],
        sub_folder=False,
    )
    assert selection != []
    assert selection[0] == str(json_file)


@pytest.mark.xfail(
    reason=(
        "get_bids_files does not find json files"
        " that are directly in the subject folder of a dataset."
    ),
    strict=True,
)
@pytest.mark.parametrize(
    "json_file",
    [
        "sub-01/sub-01_task-main_bold.json",
        "sub-01/ses-01/sub-01_ses-01_task-main_bold.json",
    ],
)
def test_get_bids_files_inheritance_principle_sub_folder(tmp_path, json_file):
    """Check if json files are found if in subject or session folder.

    see https://bids-specification.readthedocs.io/en/latest/common-principles.html#the-inheritance-principle
    """
    bids_path = create_fake_bids_dataset(
        base_dir=tmp_path, n_sub=1, n_ses=1, tasks=["main"], n_runs=[1]
    )

    _rm_all_json_files_from_bids_dataset(bids_path)

    new_json_file = add_metadata_to_bids_dataset(
        bids_path=bids_path,
        metadata={"RepetitionTime": 1.5},
        json_file=json_file,
    )
    assert new_json_file.exists()

    # make sure that get_bids_files finds the json file
    # but only when NOT looking in root of dataset
    selection = get_bids_files(
        bids_path,
        file_tag="bold",
        file_type="json",
        filters=[("task", "main")],
        sub_folder=False,
    )
    assert selection == []
    selection = get_bids_files(
        bids_path,
        file_tag="bold",
        file_type="json",
        filters=[("task", "main")],
        sub_folder=True,
    )
    assert selection != []
    assert selection[0] == str(new_json_file)


def test_get_bids_files(tmp_path):
    """Check proper number of files is returned.

    For each possible option of file selection
    we check that we recover the appropriate amount of files,
    as included in the fake bids dataset.
    """
    n_sub = 2

    bids_path = create_fake_bids_dataset(
        base_dir=tmp_path,
        n_sub=n_sub,
        n_ses=2,
        tasks=["localizer", "main"],
        n_runs=[1, 2],
    )

    # files in total related to subject images.
    # Top level files like README not included
    selection = get_bids_files(bids_path)
    assert len(selection) == 19 * n_sub
    # bold files expected. .nii and .json files
    selection = get_bids_files(bids_path, file_tag="bold")
    assert len(selection) == 12 * n_sub
    # files are nii.gz. Bold and T1w files.
    selection = get_bids_files(bids_path, file_type="nii.gz")
    assert len(selection) == 7 * n_sub
    # files correspond to subject 01
    selection = get_bids_files(bids_path, sub_label="01")
    assert len(selection) == 19
    # There are only n_sub files in anat folders. One T1w per subject.
    selection = get_bids_files(bids_path, modality_folder="anat")
    assert len(selection) == n_sub
    # files corresponding to run 1 of session 2 of main task.
    # n_sub bold.nii.gz and n_sub bold.json files.
    filters = [("task", "main"), ("run", "01"), ("ses", "02")]
    selection = get_bids_files(bids_path, file_tag="bold", filters=filters)
    assert len(selection) == 2 * n_sub
    # Get Top level folder files. Only 1 in this case, the README file.
    selection = get_bids_files(bids_path, sub_folder=False)
    assert len(selection) == 1
    # counfonds (4 runs per ses & sub), testing `fmriprep` >= 20.2 path
    selection = get_bids_files(
        bids_path / "derivatives",
        file_tag="desc-confounds_timeseries",
    )
    assert len(selection) == 12 * n_sub

    bids_path = create_fake_bids_dataset(
        base_dir=tmp_path,
        n_sub=n_sub,
        n_ses=2,
        tasks=["localizer", "main"],
        n_runs=[1, 2],
        confounds_tag="desc-confounds_regressors",
    )
    # counfonds (4 runs per ses & sub), testing `fmriprep` >= 20.2 path
    selection = get_bids_files(
        bids_path / "derivatives",
        file_tag="desc-confounds_regressors",
    )
    assert len(selection) == 12 * n_sub


def test_get_bids_files_no_space_entity(tmp_path):
    """Pass empty string for a label ignores files containing that label.

    - remove space entity only from subject 01
    - check that only files from the appropriate subject are returned
      when passing ("space", "T1w") or ("space", "")
    """
    n_sub = 2

    bids_path = create_fake_bids_dataset(
        base_dir=tmp_path,
        n_sub=n_sub,
        n_ses=2,
        tasks=["main"],
        n_runs=[2],
    )

    for file in (bids_path / "derivatives" / "sub-01").glob(
        "**/*_space-*.nii.gz"
    ):
        stem = [
            entity
            for entity in file.stem.split("_")
            if not entity.startswith("space")
        ]
        file.replace(file.with_stem("_".join(stem)))

    selection = get_bids_files(
        bids_path / "derivatives",
        file_tag="bold",
        file_type="nii.gz",
        filters=[("space", "T1w")],
    )

    assert selection
    assert all("sub-01" not in file for file in selection)

    selection = get_bids_files(
        bids_path / "derivatives",
        file_tag="bold",
        file_type="nii.gz",
        filters=[("space", "")],
    )

    assert selection
    assert all("sub-02" not in file for file in selection)


def test_parse_bids_filename():
    fields = ["sub", "ses", "task", "lolo"]
    labels = ["01", "01", "langloc", "lala"]
    file_name = "sub-01_ses-01_task-langloc_lolo-lala_bold.nii.gz"
    file_path = Path("dataset", "sub-01", "ses-01", "func", file_name)
    file_dict = parse_bids_filename(file_path)
    for fidx, field in enumerate(fields):
        assert file_dict[field] == labels[fidx]
    assert file_dict["file_type"] == "nii.gz"
    assert file_dict["file_tag"] == "bold"
    assert file_dict["file_path"] == file_path
    assert file_dict["file_basename"] == file_name
    assert file_dict["file_fields"] == fields


@pytest.mark.skipif(
    not is_matplotlib_installed(),
    reason="This test requires matplotlib to be installed.",
)
@pytest.mark.parametrize(
    "prefix", ["sub-01_ses-01_task-nback", "sub-01_task-nback", "task-nback"]
)
def test_save_glm_to_bids(tmp_path_factory, prefix):
    """Test that save_glm_to_bids saves the appropriate files.

    This test reuses code from
    nilearn.glm.tests.test_first_level.test_high_level_glm_one_session.
    """
    tmpdir = tmp_path_factory.mktemp("test_save_glm_results")

    EXPECTED_FILENAMES = [
        "contrast-effectsOfInterest_design.svg",
        "contrast-effectsOfInterest_stat-F_statmap.nii.gz",
        "contrast-effectsOfInterest_stat-effect_statmap.nii.gz",
        "contrast-effectsOfInterest_stat-p_statmap.nii.gz",
        "contrast-effectsOfInterest_stat-variance_statmap.nii.gz",
        "contrast-effectsOfInterest_stat-z_statmap.nii.gz",
        "design.svg",
        "design.tsv",
        "design.json",
        "stat-errorts_statmap.nii.gz",
        "stat-rsquared_statmap.nii.gz",
        "statmap.json",
        "report.html",
    ]

    shapes, rk = [(7, 8, 9, 15)], 3
    _, fmri_data, design_matrices = generate_fake_fmri_data_and_design(
        shapes,
        rk,
    )

    single_run_model = FirstLevelModel(
        mask_img=None,
        minimize_memory=False,
    ).fit(
        fmri_data[0],
        design_matrices=design_matrices[0],
    )

    contrasts = {
        "effects of interest": np.eye(rk),
    }
    contrast_types = {
        "effects of interest": "F",
    }
    save_glm_to_bids(
        model=single_run_model,
        contrasts=contrasts,
        contrast_types=contrast_types,
        out_dir=tmpdir,
        prefix=prefix,
    )

    assert (tmpdir / "dataset_description.json").exists()

    sub_prefix = prefix.split("_")[0] if prefix.startswith("sub-") else ""

    for fname in EXPECTED_FILENAMES:
        assert (tmpdir / sub_prefix / f"{prefix}_{fname}").exists()


@pytest.mark.skipif(
    not is_matplotlib_installed(),
    reason="This test requires matplotlib to be installed.",
)
def test_save_glm_to_bids_serialize_affine(tmp_path):
    """Test that affines are turned into a serializable type.

    Regression test for https://github.com/nilearn/nilearn/issues/4324.
    """
    shapes, rk = [(7, 8, 9, 15)], 3
    mask, fmri_data, design_matrices = generate_fake_fmri_data_and_design(
        shapes,
        rk,
    )

    target_affine = mask.affine

    single_run_model = FirstLevelModel(
        target_affine=target_affine,
        minimize_memory=False,
    ).fit(
        fmri_data[0],
        design_matrices=design_matrices[0],
    )

    save_glm_to_bids(
        model=single_run_model,
        contrasts={
            "effects of interest": np.eye(rk),
        },
        contrast_types={
            "effects of interest": "F",
        },
        out_dir=tmp_path,
        prefix="sub-01_ses-01_task-nback",
    )


@pytest.fixture
def n_cols_design_matrix():
    return 3


@pytest.fixture
def two_runs_model(n_cols_design_matrix):
    # Create two runs of data
    shapes, rk = [(7, 8, 9, 15), (7, 8, 9, 15)], n_cols_design_matrix
    mask, fmri_data, design_matrices = generate_fake_fmri_data_and_design(
        shapes,
        rk,
    )
    # Rename two conditions in design matrices
    mapper = {
        design_matrices[0].columns[0]: "AAA",
        design_matrices[0].columns[1]: "BBB",
    }
    design_matrices[0] = design_matrices[0].rename(columns=mapper)
    mapper = {
        design_matrices[1].columns[0]: "AAA",
        design_matrices[1].columns[1]: "BBB",
    }
    design_matrices[1] = design_matrices[1].rename(columns=mapper)

    masker = NiftiMasker(mask)
    masker.fit()

    return FirstLevelModel(
        mask_img=None,
        minimize_memory=False,
    ).fit(
        fmri_data,
        design_matrices=design_matrices,
    )


@pytest.mark.skipif(
    not is_matplotlib_installed(),
    reason="This test requires matplotlib to be installed.",
)
def test_save_glm_to_bids_errors(
    tmp_path_factory, two_runs_model, n_cols_design_matrix
):
    """Test errors of save_glm_to_bids."""
    tmpdir = tmp_path_factory.mktemp("test_save_glm_to_bids_errors")

    # Contrast names must be strings
    contrasts = {5: np.eye(n_cols_design_matrix)}
    contrast_types = {5: "F"}

    with pytest.raises(ValueError):
        save_glm_to_bids(
            model=two_runs_model,
            contrasts=contrasts,
            contrast_types=contrast_types,
            out_dir=tmpdir,
            prefix="sub-01_ses-01_task-nback",
        )

    # Contrast definitions must be strings, numpy arrays, or lists
    contrasts = {"effects of interest": 5}
    contrast_types = {"effects of interest": "F"}

    with pytest.raises(ValueError):
        save_glm_to_bids(
            model=two_runs_model,
            contrasts=contrasts,
            contrast_types=contrast_types,
            out_dir=tmpdir,
            prefix="sub-01_ses-01_task-nback",
        )

    with pytest.raises(
        ValueError, match="Extra key-word arguments must be one of"
    ):
        save_glm_to_bids(
            model=two_runs_model,
            contrasts=["AAA - BBB"],
            out_dir=tmpdir,
            prefix="sub-01_ses-01_task-nback",
            foo="bar",
        )


@pytest.mark.skipif(
    not is_matplotlib_installed(),
    reason="This test requires matplotlib to be installed.",
)
@pytest.mark.parametrize(
    "prefix", ["sub-01_ses-01_task-nback", "sub-01_task-nback_", 1]
)
@pytest.mark.parametrize("contrasts", [["AAA - BBB"], "AAA - BBB"])
def test_save_glm_to_bids_contrast_definitions(
    tmp_path_factory, two_runs_model, contrasts, prefix
):
    """Test that save_glm_to_bids operates on different contrast definitions \
       as expected.

    - Test string-based contrasts and undefined contrast types

    This test reuses code from
    nilearn.glm.tests.test_first_level.test_high_level_glm_one_session.
    """
    tmpdir = tmp_path_factory.mktemp(
        "test_save_glm_to_bids_contrast_definitions"
    )

    EXPECTED_FILENAME_ENDINGS = [
        "contrast-aaaMinusBbb_stat-effect_statmap.nii.gz",
        "contrast-aaaMinusBbb_stat-p_statmap.nii.gz",
        "contrast-aaaMinusBbb_stat-t_statmap.nii.gz",
        "contrast-aaaMinusBbb_stat-variance_statmap.nii.gz",
        "contrast-aaaMinusBbb_stat-z_statmap.nii.gz",
        "run-1_contrast-aaaMinusBbb_design.svg",
        "run-1_design.svg",
        "run-1_design.tsv",
        "run-1_design.json",
        "run-1_stat-errorts_statmap.nii.gz",
        "run-1_stat-rsquared_statmap.nii.gz",
        "run-2_contrast-aaaMinusBbb_design.svg",
        "run-2_design.svg",
        "run-2_design.tsv",
        "run-2_design.json",
        "run-2_stat-errorts_statmap.nii.gz",
        "run-2_stat-rsquared_statmap.nii.gz",
        "statmap.json",
        "report.html",
    ]

    save_glm_to_bids(
        model=two_runs_model,
        contrasts=contrasts,
        contrast_types=None,
        out_dir=tmpdir,
        prefix=prefix,
    )

    assert (tmpdir / "dataset_description.json").exists()

    if not isinstance(prefix, str):
        prefix = ""

    if prefix and not prefix.endswith("_"):
        prefix = f"{prefix}_"

    sub_prefix = prefix.split("_")[0] if prefix.startswith("sub-") else ""

    for fname in EXPECTED_FILENAME_ENDINGS:
        assert (tmpdir / sub_prefix / f"{prefix}{fname}").exists()


@pytest.mark.skipif(
    not is_matplotlib_installed(),
    reason="This test requires matplotlib to be installed.",
)
@pytest.mark.parametrize("prefix", ["task-nback"])
def test_save_glm_to_bids_second_level(tmp_path_factory, prefix):
    """Test save_glm_to_bids on a SecondLevelModel.

    This test reuses code from
    nilearn.glm.tests.test_second_level.test_high_level_glm_with_paths.
    """
    tmpdir = tmp_path_factory.mktemp("test_save_glm_to_bids_second_level")

    EXPECTED_FILENAMES = [
        "contrast-effectsOfInterest_design.svg",
        "contrast-effectsOfInterest_stat-F_statmap.nii.gz",
        "contrast-effectsOfInterest_stat-effect_statmap.nii.gz",
        "contrast-effectsOfInterest_stat-p_statmap.nii.gz",
        "contrast-effectsOfInterest_stat-variance_statmap.nii.gz",
        "contrast-effectsOfInterest_stat-z_statmap.nii.gz",
        "design.svg",
        "design.tsv",
        "stat-errorts_statmap.nii.gz",
        "stat-rsquared_statmap.nii.gz",
        "statmap.json",
        "report.html",
    ]

    shapes = ((3, 3, 3, 1),)
    rk = 3
    mask, func_img, _ = generate_fake_fmri_data_and_design(
        shapes,
        rk,
    )
    func_img = func_img[0]

    # Ordinary Least Squares case
    model = SecondLevelModel(mask_img=mask, minimize_memory=False)

    # fit model
    Y = [func_img] * 2
    X = pd.DataFrame([[1]] * 2, columns=["intercept"])
    model = model.fit(Y, design_matrix=X)

    contrasts = {
        "effects of interest": np.eye(len(model.design_matrix_.columns))[0],
    }
    contrast_types = {"effects of interest": "F"}

    save_glm_to_bids(
        model=model,
        contrasts=contrasts,
        contrast_types=contrast_types,
        out_dir=tmpdir,
        prefix=prefix,
    )

    assert (tmpdir / "dataset_description.json").exists()

    for fname in EXPECTED_FILENAMES:
        assert (tmpdir / "group" / f"{prefix}_{fname}").exists()
