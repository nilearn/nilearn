"""Tests saving glm to bids."""

import json
import warnings

import numpy as np
import pandas as pd
import pytest

from nilearn._utils.data_gen import (
    create_fake_bids_dataset,
    generate_fake_fmri_data_and_design,
)
from nilearn._utils.helpers import is_matplotlib_installed
from nilearn.glm.first_level import FirstLevelModel, first_level_from_bids
from nilearn.glm.io import save_glm_to_bids
from nilearn.glm.second_level import SecondLevelModel
from nilearn.maskers import NiftiMasker

# generic parameters to reduce n warnings in tests
KWARGS = {"height_control": None, "threshold": 1, "cut_coords": [0.5, 1, 1.5]}


@pytest.mark.slow
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
        "contrast-effectsOfInterest_stat-F_statmap.nii.gz",
        "contrast-effectsOfInterest_stat-effect_statmap.nii.gz",
        "contrast-effectsOfInterest_stat-p_statmap.nii.gz",
        "contrast-effectsOfInterest_stat-variance_statmap.nii.gz",
        "contrast-effectsOfInterest_stat-z_statmap.nii.gz",
        "contrast-effectsOfInterest_clusters.tsv",
        "contrast-effectsOfInterest_clusters.json",
        "design.tsv",
        "design.json",
        "stat-errorts_statmap.nii.gz",
        "stat-rsquared_statmap.nii.gz",
        "statmap.json",
        "mask.nii.gz",
        "report.html",
    ]

    if is_matplotlib_installed():
        EXPECTED_FILENAMES.extend(
            [
                "design.png",
                "contrast-effectsOfInterest_design.png",
            ]
        )

    shapes, rk = [(7, 8, 9, 15)], 3
    _, fmri_data, design_matrices = generate_fake_fmri_data_and_design(
        shapes,
        rk,
    )

    single_run_model = FirstLevelModel(
        mask_img=None,
        minimize_memory=False,
    ).fit(fmri_data[0], design_matrices=design_matrices[0])

    contrasts = {"effects of interest": np.eye(rk)}
    contrast_types = {"effects of interest": "F"}
    with warnings.catch_warnings(record=True) as warning_list:
        save_glm_to_bids(
            model=single_run_model,
            contrasts=contrasts,
            contrast_types=contrast_types,
            out_dir=tmpdir,
            prefix=prefix,
            height_control=None,
        )

        # TODO (nilearn >= 0.15.0) remove
        n_future_warnings = len(
            [x for x in warning_list if issubclass(x.category, FutureWarning)]
        )
        assert n_future_warnings == 1

        n_no_contrasts_warnings = len(
            [
                x
                for x in warning_list
                if "No contrast passed during report generation." in str(x)
            ]
        )
        assert n_no_contrasts_warnings == 0

    assert (tmpdir / "dataset_description.json").exists()

    sub_prefix = prefix.split("_")[0] if prefix.startswith("sub-") else ""

    for fname in EXPECTED_FILENAMES:
        assert (tmpdir / sub_prefix / f"{prefix}_{fname}").exists()


@pytest.mark.slow
def test_save_glm_to_bids_reset_threshold_warning(tmp_path_factory):
    """Get single warning threshold reset to None."""
    tmpdir = tmp_path_factory.mktemp("test_save_glm_results")

    shapes, rk = [(7, 8, 9, 15)], 3
    _, fmri_data, design_matrices = generate_fake_fmri_data_and_design(
        shapes,
        rk,
    )

    single_run_model = FirstLevelModel(
        mask_img=None,
        minimize_memory=False,
    ).fit(fmri_data[0], design_matrices=design_matrices[0])

    contrasts = {"effects of interest": np.eye(rk)}
    contrast_types = {"effects of interest": "F"}
    with warnings.catch_warnings(record=True) as warning_list:
        save_glm_to_bids(
            model=single_run_model,
            contrasts=contrasts,
            contrast_types=contrast_types,
            out_dir=tmpdir,
            threshold=1.0,
        )

        reset_threshold_warnings = len(
            [
                x
                for x in warning_list
                if "'threshold' was set to 'None'" in str(x)
            ]
        )
        assert reset_threshold_warnings == 1


@pytest.mark.slow
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
        contrasts={"effects of interest": np.eye(rk)},
        contrast_types={"effects of interest": "F"},
        out_dir=tmp_path,
        prefix="sub-01_ses-01_task-nback",
        **KWARGS,
    )


@pytest.fixture
def n_cols_design_matrix():
    """Return expected number of column in design matrix."""
    return 3


@pytest.fixture
def two_runs_model(n_cols_design_matrix) -> FirstLevelModel:
    """Create two runs of data."""
    shapes, rk = [(7, 8, 9, 10), (7, 8, 9, 10)], n_cols_design_matrix
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

    return FirstLevelModel(mask_img=None, minimize_memory=False).fit(
        fmri_data, design_matrices=design_matrices
    )


def test_save_glm_to_bids_errors(
    tmp_path_factory, two_runs_model, n_cols_design_matrix
):
    """Test errors of save_glm_to_bids."""
    tmpdir = tmp_path_factory.mktemp("test_save_glm_to_bids_errors")

    # Contrast names must be strings
    contrasts = {5: np.eye(n_cols_design_matrix)}
    with pytest.raises(TypeError, match="contrast names must be strings"):
        save_glm_to_bids(
            model=two_runs_model,
            contrasts=contrasts,
            out_dir=tmpdir,
            prefix="sub-01",
        )

    # Contrast definitions must be strings, numpy arrays, or lists
    contrasts = {"effects of interest": 5}
    with pytest.raises(
        TypeError, match="contrast definitions must be strings or array_likes"
    ):
        save_glm_to_bids(
            model=two_runs_model,
            contrasts=contrasts,
            out_dir=tmpdir,
            prefix="sub-01",
        )

    with pytest.raises(ValueError, match="must be one of"):
        save_glm_to_bids(
            model=two_runs_model,
            contrasts=["AAA - BBB"],
            out_dir=tmpdir,
            prefix="sub-01",
            foo="bar",
        )


@pytest.mark.slow
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
        "contrast-aaaMinusBbb_clusters.tsv",
        "contrast-aaaMinusBbb_clusters.json",
        "run-1_design.tsv",
        "run-1_design.json",
        "run-1_stat-errorts_statmap.nii.gz",
        "run-1_stat-rsquared_statmap.nii.gz",
        "run-2_design.tsv",
        "run-2_design.json",
        "run-2_stat-errorts_statmap.nii.gz",
        "run-2_stat-rsquared_statmap.nii.gz",
        "statmap.json",
        "mask.nii.gz",
        "report.html",
    ]
    if is_matplotlib_installed():
        EXPECTED_FILENAME_ENDINGS.extend(
            [
                "run-1_contrast-aaaMinusBbb_design.png",
                "run-1_design.png",
                "run-2_contrast-aaaMinusBbb_design.png",
                "run-2_design.png",
            ]
        )

    save_glm_to_bids(
        model=two_runs_model,
        contrasts=contrasts,
        contrast_types=None,
        out_dir=tmpdir,
        prefix=prefix,
        **KWARGS,
    )

    assert (tmpdir / "dataset_description.json").exists()

    if not isinstance(prefix, str):
        prefix = ""

    if prefix and not prefix.endswith("_"):
        prefix = f"{prefix}_"

    sub_prefix = prefix.split("_")[0] if prefix.startswith("sub-") else ""

    for fname in EXPECTED_FILENAME_ENDINGS:
        assert (tmpdir / sub_prefix / f"{prefix}{fname}").exists()


@pytest.mark.slow
@pytest.mark.parametrize("prefix", ["task-nback"])
def test_save_glm_to_bids_second_level(tmp_path_factory, prefix):
    """Test save_glm_to_bids on a SecondLevelModel.

    This test reuses code from
    nilearn.glm.tests.test_second_level.test_high_level_glm_with_paths.
    """
    tmpdir = tmp_path_factory.mktemp("test_save_glm_to_bids_second_level")

    EXPECTED_FILENAMES = [
        "contrast-effectsOfInterest_stat-F_statmap.nii.gz",
        "contrast-effectsOfInterest_stat-effect_statmap.nii.gz",
        "contrast-effectsOfInterest_stat-p_statmap.nii.gz",
        "contrast-effectsOfInterest_stat-variance_statmap.nii.gz",
        "contrast-effectsOfInterest_stat-z_statmap.nii.gz",
        "contrast-effectsOfInterest_clusters.tsv",
        "contrast-effectsOfInterest_clusters.json",
        "design.tsv",
        "stat-errorts_statmap.nii.gz",
        "stat-rsquared_statmap.nii.gz",
        "statmap.json",
        "mask.nii.gz",
        "report.html",
    ]
    if is_matplotlib_installed():
        EXPECTED_FILENAMES.extend(
            [
                "design.png",
                "contrast-effectsOfInterest_design.png",
            ]
        )

    shapes = ((3, 3, 3, 1),)
    rk = 3
    mask, fmri_data, _ = generate_fake_fmri_data_and_design(
        shapes,
        rk,
    )
    fmri_data = fmri_data[0]

    # Ordinary Least Squares case
    model = SecondLevelModel(mask_img=mask, minimize_memory=False)

    # fit model
    Y = [fmri_data] * 2
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
        **KWARGS,
    )

    assert (tmpdir / "dataset_description.json").exists()

    for fname in EXPECTED_FILENAMES:
        assert (tmpdir / "group" / f"{prefix}_{fname}").exists()


@pytest.mark.slow
def test_save_glm_to_bids_glm_report_no_contrast(two_runs_model, tmp_path):
    """Run generate_report with no contrasts after save_glm_to_bids.

    generate_report tries to rely on some of the generated output,
    if no contrasts are requested to generate_report
    then it will rely on the content of
    model._reporting_data["filenames"]

    report generated by save_glm_to_bids should contain relative paths
    to the figures displayed as the report and its figures are meant
    to go together

    report generated after using save_glm_to_bids could be saved anywhere
    so evengthough we reuse pre-generated figures,
    we will rely on full path in this case
    """
    contrasts = {"BBB-AAA": "BBB-AAA"}
    contrast_types = {"BBB-AAA": "t"}
    model = save_glm_to_bids(
        model=two_runs_model,
        contrasts=contrasts,
        contrast_types=contrast_types,
        out_dir=tmp_path,
        **KWARGS,
    )

    assert model._reporting_data.get("filenames", None) is not None

    EXPECTED_FILENAMES = [
        "run-1_design.png",
        "run-1_corrdesign.png",
        "run-1_contrast-bbbMinusAaa_design.png",
    ]

    with (tmp_path / "report.html").open("r") as f:
        content = f.read()
        assert "BBB-AAA" in content
        for file in EXPECTED_FILENAMES:
            assert f'src="{file}"' in content

    report = model.generate_report(**KWARGS)

    report.save_as_html(tmp_path / "new_report.html")

    assert "BBB-AAA" in report.__str__()
    for file in EXPECTED_FILENAMES:
        assert f'src="{tmp_path / file}"' in report.__str__()
        assert f'src="{file}"' not in report.__str__()


@pytest.mark.slow
def test_save_glm_to_bids_glm_report_new_contrast(two_runs_model, tmp_path):
    """Run generate_report after save_glm_to_bids with different contrasts.

    generate_report tries to rely on some of the generated output,
    but if different contrasts are requested
    then it will have to do some extra contrast computation.
    """
    contrasts = {"BBB-AAA": "BBB-AAA"}
    contrast_types = {"BBB-AAA": "t"}
    model = save_glm_to_bids(
        model=two_runs_model,
        contrasts=contrasts,
        contrast_types=contrast_types,
        out_dir=tmp_path,
        **KWARGS,
    )

    EXPECTED_FILENAMES = [
        "run-1_design.png",
        "run-1_corrdesign.png",
        "run-1_contrast-bbbMinusAaa_design.png",
    ]

    # check content of a new report
    report = model.generate_report(contrasts=["AAA-BBB"], **KWARGS)

    assert "AAA-BBB" in report.__str__()
    assert "BBB-AAA" not in report.__str__()
    for file in EXPECTED_FILENAMES:
        assert file not in report.__str__()


@pytest.mark.slow
@pytest.mark.parametrize("kwargs", ([{}, {"height_control": None}]))
def test_save_glm_to_bids_infer_filenames(tmp_path, kwargs):
    """Check that output filenames can be inferred from BIDS input."""
    n_sub = 1

    bids_path = create_fake_bids_dataset(
        base_dir=tmp_path,
        n_sub=n_sub,
        n_ses=2,
        tasks=["main"],
        n_runs=[2],
        n_voxels=20,
    )

    models, imgs, events, _ = first_level_from_bids(
        dataset_path=bids_path,
        task_label="main",
        space_label="MNI",
        img_filters=[("desc", "preproc")],
        slice_time_ref=0.0,  # set to 0.0 to avoid warnings
    )

    model = models[0]
    run_imgs = imgs[0]
    events = events[0]

    model.minimize_memory = False
    model.fit(run_imgs=run_imgs, events=events)

    # 2 sessions with 2 runs each
    assert len(model._reporting_data["run_imgs"]) == 4

    model = save_glm_to_bids(
        model=model, out_dir=tmp_path / "output", contrasts=["c0"], **kwargs
    )

    EXPECTED_FILENAME_ENDINGS = [
        "sub-01_task-main_space-MNI_contrast-c0_stat-z_statmap.nii.gz",
        "sub-01_task-main_space-MNI_contrast-c0_clusters.tsv",
        "sub-01_task-main_space-MNI_contrast-c0_clusters.json",
        "sub-01_ses-01_task-main_run-01_space-MNI_stat-rsquared_statmap.nii.gz",
        "sub-01_ses-02_task-main_run-02_space-MNI_design.tsv",
        "sub-01_ses-01_task-main_run-02_space-MNI_design.json",
        # mask is common to all sessions and runs
        "sub-01_task-main_space-MNI_mask.nii.gz",
    ]
    if is_matplotlib_installed():
        EXPECTED_FILENAME_ENDINGS.extend(
            [
                "sub-01_ses-02_task-main_run-01_space-MNI_design.png",
                "sub-01_ses-02_task-main_run-01_space-MNI_corrdesign.png",
                "sub-01_ses-01_task-main_run-02_space-MNI_contrast-c0_design.png",
            ]
        )

    for fname in EXPECTED_FILENAME_ENDINGS:
        assert (tmp_path / "output" / "sub-01" / fname).exists()

    with (
        tmp_path
        / "output"
        / "sub-01"
        / "sub-01_task-main_space-MNI_contrast-c0_clusters.json"
    ).open("r") as f:
        metadata = json.load(f)

    expected_keys = [
        "Cluster size threshold (voxels)",
        "Minimum distance (mm)",
    ]

    if "height_control" not in kwargs:
        expected_keys.extend(
            [
                "Height control",
                "Threshold (computed)",
            ]
        )
    else:
        expected_keys.extend(
            [
                "Height control",
                "Threshold Z",
            ]
        )

    for key in expected_keys:
        assert key in metadata


@pytest.mark.slow
def test_save_glm_to_bids_surface_prefix_override(tmp_path):
    """Save surface GLM results to disk with prefix."""
    n_sub = 1

    bids_path = create_fake_bids_dataset(
        base_dir=tmp_path,
        n_sub=n_sub,
        n_ses=2,
        tasks=["main"],
        n_runs=[2],
        n_vertices=10242,
    )

    models, imgs, events, _ = first_level_from_bids(
        dataset_path=bids_path,
        task_label="main",
        space_label="fsaverage5",
        slice_time_ref=0.0,  # set to 0.0 to avoid warnings
    )

    model = models[0]
    run_imgs = imgs[0]
    events = events[0]

    model.minimize_memory = False
    model.fit(run_imgs=run_imgs, events=events)

    prefix = "sub-01"

    model = save_glm_to_bids(
        model=model,
        out_dir=tmp_path / "output",
        contrasts=["c0"],
        prefix=prefix,
        **KWARGS,
    )

    EXPECTED_FILENAME_ENDINGS = [
        "run-2_design.tsv",
        "run-2_design.json",
        "hemi-L_den-10242_mask.gii",
        "hemi-R_den-10242_mask.gii",
        "hemi-L_den-10242_contrast-c0_stat-z_statmap.gii",
        "hemi-R_den-10242_contrast-c0_stat-z_statmap.gii",
        "run-1_hemi-L_den-10242_stat-rsquared_statmap.gii",
        "run-1_hemi-R_den-10242_stat-rsquared_statmap.gii",
        "contrast-c0_clusters.tsv",
        "contrast-c0_clusters.json",
    ]
    if is_matplotlib_installed():
        EXPECTED_FILENAME_ENDINGS.extend(
            [
                "run-1_design.png",
                "run-1_corrdesign.png",
                "run-2_contrast-c0_design.png",
            ]
        )

    if prefix != "" and not prefix.endswith("_"):
        prefix += "_"

    sub_prefix = prefix.split("_")[0] if prefix.startswith("sub-") else ""

    for fname in EXPECTED_FILENAME_ENDINGS:
        assert (tmp_path / "output" / sub_prefix / f"{prefix}{fname}").exists()


@pytest.mark.slow
@pytest.mark.parametrize("prefix", ["", "sub-01", "foo_"])
def test_save_glm_to_bids_infer_filenames_override(tmp_path, prefix):
    """Check that output filenames is not inferred when prefix is passed."""
    n_sub = 1

    bids_path = create_fake_bids_dataset(
        base_dir=tmp_path,
        n_sub=n_sub,
        n_ses=1,
        tasks=["main"],
        n_runs=[1],
        n_voxels=20,
    )

    models, imgs, events, _ = first_level_from_bids(
        dataset_path=bids_path,
        task_label="main",
        space_label="MNI",
        img_filters=[("desc", "preproc")],
        slice_time_ref=0.0,  # set to 0.0 to avoid warnings
    )

    model = models[0]
    run_imgs = imgs[0]
    events = events[0]

    model.minimize_memory = False
    model.fit(run_imgs=run_imgs, events=events)

    model = save_glm_to_bids(
        model=model,
        out_dir=tmp_path / "output",
        contrasts=["c0"],
        prefix=prefix,
        **KWARGS,
    )

    EXPECTED_FILENAME_ENDINGS = [
        "mask.nii.gz",
        "contrast-c0_stat-z_statmap.nii.gz",
        "contrast-c0_clusters.tsv",
        "contrast-c0_clusters.json",
        "stat-rsquared_statmap.nii.gz",
        "design.tsv",
        "design.json",
    ]

    if prefix != "" and not prefix.endswith("_"):
        prefix += "_"

    sub_prefix = prefix.split("_")[0] if prefix.startswith("sub-") else ""

    for fname in EXPECTED_FILENAME_ENDINGS:
        assert (tmp_path / "output" / sub_prefix / f"{prefix}{fname}").exists()
