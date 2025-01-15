"""Tests for the data generation utilities."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from nibabel import load
from numpy.testing import assert_almost_equal
from pandas.api.types import is_numeric_dtype, is_object_dtype
from pandas.testing import assert_frame_equal

from nilearn._utils.data_gen import (
    add_metadata_to_bids_dataset,
    basic_paradigm,
    create_fake_bids_dataset,
    generate_fake_fmri,
    generate_fake_fmri_data_and_design,
    generate_group_sparse_gaussian_graphs,
    generate_labeled_regions,
    generate_maps,
    generate_mni_space_img,
    generate_random_img,
    generate_regions_ts,
    generate_timeseries,
    write_fake_bold_img,
    write_fake_fmri_data_and_design,
)
from nilearn.image import get_data


def test_add_metadata_to_bids_derivatives_default_path(tmp_path):
    """Check the filename created is the default value \
    of add_metadata_to_bids_dataset.
    """
    target_dir = tmp_path / "derivatives" / "sub-01" / "ses-01" / "func"
    target_dir.mkdir(parents=True)
    json_file = add_metadata_to_bids_dataset(
        bids_path=tmp_path, metadata={"foo": "bar"}
    )
    assert json_file.exists()
    assert (
        json_file.name
        == "sub-01_ses-01_task-main_run-01_space-MNI_desc-preproc_bold.json"
    )
    with json_file.open() as f:
        metadata = json.load(f)
        assert metadata == {"foo": "bar"}


def test_add_metadata_to_bids_derivatives_with_json_path(tmp_path):
    # bare bone smoke test
    target_dir = tmp_path / "derivatives" / "sub-02"
    target_dir.mkdir(parents=True)
    json_file = "derivatives/sub-02/sub-02_task-main_bold.json"
    json_file = add_metadata_to_bids_dataset(
        bids_path=tmp_path, metadata={"foo": "bar"}, json_file=json_file
    )
    assert json_file.exists()
    assert json_file.name == "sub-02_task-main_bold.json"
    with json_file.open() as f:
        metadata = json.load(f)
        assert metadata == {"foo": "bar"}


@pytest.mark.parametrize("have_spaces", [False, True])
def test_basic_paradigm(have_spaces):
    events = basic_paradigm(condition_names_have_spaces=have_spaces)

    assert events.columns.equals(pd.Index(["trial_type", "onset", "duration"]))
    assert is_object_dtype(events["trial_type"])
    assert is_numeric_dtype(events["onset"])
    assert is_numeric_dtype(events["duration"])
    assert events["trial_type"].str.contains(" ").any() == have_spaces


@pytest.mark.parametrize("shape", [(3, 4, 5), (2, 3, 5, 7)])
@pytest.mark.parametrize("affine", [None, np.diag([0.5, 0.3, 1, 1])])
def test_write_fake_bold_img(tmp_path, shape, affine, rng):
    img_file = write_fake_bold_img(
        file_path=tmp_path / "fake_bold.nii",
        shape=shape,
        affine=affine,
        random_state=rng,
    )
    img = load(img_file)

    assert img.get_fdata().shape == shape
    if affine is not None:
        assert_almost_equal(img.affine, affine)


def _bids_path_template(
    task,
    suffix,
    n_runs=None,
    space=None,
    desc=None,
    extra_entity=None,
):
    """Create a BIDS filepath from a template.

    File path is relative to the BIDS root folder.

    File path contains a session level folder.

    """
    task = f"task-{task}_*"
    run = "run-*_*" if n_runs is not None else "*"
    space = f"space-{space}_*" if space is not None else "*"
    desc = f"desc-{desc}_*" if desc is not None else "*"

    # only using with resolution and acquisition entities (for now)
    acq = "*"
    res = "*"
    if extra_entity is not None:
        if "acq" in extra_entity:
            acq = f"acq-{extra_entity['acq']}_*"
        elif "res" in extra_entity:
            res = f"res-{extra_entity['res']}_*"

    path = "sub-*/ses-*/func/sub-*_ses-*_*"
    path += f"{task}{acq}{run}{space}{res}{desc}{suffix}"
    # TODO use regex
    path = path.replace("***", "*")
    path = path.replace("**", "*")
    return path


@pytest.mark.parametrize("n_sub", [1, 2])
@pytest.mark.parametrize("n_ses", [1, 2])
@pytest.mark.parametrize(
    "tasks,n_runs",
    [(["main"], [1]), (["main"], [2]), (["main", "localizer"], [2, 1])],
)
def test_fake_bids_raw_with_session_and_runs(
    tmp_path, n_sub, n_ses, tasks, n_runs
):
    """Check number of each file 'type' created in raw."""
    bids_path = create_fake_bids_dataset(
        base_dir=tmp_path, n_sub=n_sub, n_ses=n_ses, tasks=tasks, n_runs=n_runs
    )

    # raw
    file_pattern = "sub-*/ses-*/anat/sub-*ses-*T1w.nii.gz"
    raw_anat_files = list(bids_path.glob(file_pattern))
    assert len(raw_anat_files) == n_sub

    for i, task in enumerate(tasks):
        for suffix in ["bold.nii.gz", "bold.json", "events.tsv"]:
            file_pattern = _bids_path_template(
                task=task, suffix=suffix, n_runs=n_runs[i]
            )
            files = list(bids_path.glob(file_pattern))
            assert len(files) == n_sub * n_ses * n_runs[i]

    all_files = list(bids_path.glob("sub-*/ses-*/*/*"))
    # per subject: 1 anat + (1 event + 1 json + 1 bold) per run per session
    n_raw_files_expected = n_sub * (1 + 3 * sum(n_runs) * n_ses)
    assert len(all_files) == n_raw_files_expected


def _check_n_files_derivatives_for_task(
    bids_path,
    n_sub,
    n_ses,
    task,
    n_run,
    extra_entity=None,
):
    """Check number of each file 'type' in derivatives for a given task."""
    for suffix in ["timeseries.tsv"]:
        # 1 confound per raw file
        # so we do not use the extra entity for derivatives entities like res
        if extra_entity is None or "res" in extra_entity:
            file_pattern = _bids_path_template(
                task=task,
                suffix=suffix,
                n_runs=n_run,
                extra_entity=None,
            )
        elif "acq" in extra_entity:
            file_pattern = _bids_path_template(
                task=task,
                suffix=suffix,
                n_runs=n_run,
                extra_entity=extra_entity,
            )

        files = list(bids_path.glob(f"derivatives/{file_pattern}"))
        assert len(files) == n_sub * n_ses * n_run

    for space in ["MNI", "T1w"]:
        file_pattern = _bids_path_template(
            task=task,
            suffix="bold.nii.gz",
            n_runs=n_run,
            space=space,
            desc="preproc",
            extra_entity=extra_entity,
        )
        files = list(bids_path.glob(f"derivatives/{file_pattern}"))
        assert len(files) == n_sub * n_ses * n_run

    # only T1w have desc-fmriprep_bold
    file_pattern = _bids_path_template(
        task=task,
        suffix="bold.nii.gz",
        n_runs=n_run,
        space="T1w",
        desc="fmriprep",
        extra_entity=extra_entity,
    )
    files = list(bids_path.glob(f"derivatives/{file_pattern}"))
    assert len(files) == n_sub * n_ses * n_run

    file_pattern = _bids_path_template(
        task=task,
        suffix="bold.nii.gz",
        n_runs=n_run,
        space="MNI",
        desc="fmriprep",
        extra_entity=extra_entity,
    )
    files = list(bids_path.glob(f"derivatives/{file_pattern}"))
    assert not files


@pytest.mark.parametrize("n_sub", [1, 2])
@pytest.mark.parametrize("n_ses", [1, 2])
@pytest.mark.parametrize(
    "tasks,n_runs",
    [(["main"], [1]), (["main"], [2]), (["main", "localizer"], [2, 1])],
)
def test_fake_bids_derivatives_with_session_and_runs(
    tmp_path, n_sub, n_ses, tasks, n_runs
):
    """Check number of each file 'type' created in derivatives."""
    bids_path = create_fake_bids_dataset(
        base_dir=tmp_path, n_sub=n_sub, n_ses=n_ses, tasks=tasks, n_runs=n_runs
    )

    # derivatives
    for task, n_run in zip(tasks, n_runs):
        _check_n_files_derivatives_for_task(
            bids_path=bids_path,
            n_sub=n_sub,
            n_ses=n_ses,
            task=task,
            n_run=n_run,
        )

    all_files = list(bids_path.glob("derivatives/sub-*/ses-*/*/*"))
    # per subject: (2 confound + 3 bold + 2 gifti) per run per session
    n_derivatives_files_expected = n_sub * (7 * sum(n_runs) * n_ses)
    assert len(all_files) == n_derivatives_files_expected


def test_bids_dataset_no_run_entity(tmp_path):
    """n_runs = 0 produces files without the run entity."""
    bids_path = create_fake_bids_dataset(
        base_dir=tmp_path,
        n_sub=1,
        n_ses=1,
        tasks=["main"],
        n_runs=[0],
        with_derivatives=True,
    )

    files = list(bids_path.glob("**/*run-*"))
    assert not files

    # nifti: 1 anat + 1 raw bold + 3 derivatives bold
    files = list(bids_path.glob("**/*.nii.gz"))
    assert len(files) == 5

    # events or json or confounds: 1
    for suffix in ["events.tsv", "timeseries.tsv", "bold.json"]:
        files = list(bids_path.glob(f"**/*{suffix}"))
        assert len(files) == 1


def test_bids_dataset_no_session(tmp_path):
    """n_ses = 0 prevent creation of a session folder."""
    bids_path = create_fake_bids_dataset(
        base_dir=tmp_path,
        n_sub=1,
        n_ses=0,
        tasks=["main"],
        n_runs=[1],
        with_derivatives=True,
    )

    files = list(bids_path.glob("**/*ses-*"))
    assert not files

    # nifti: 1 anat + 1 raw bold + 3 derivatives bold
    files = list(bids_path.glob("**/*.nii.gz"))
    assert len(files) == 5

    # events or json or confounds: 1
    for suffix in ["events.tsv", "timeseries.tsv", "bold.json"]:
        files = list(bids_path.glob(f"**/*{suffix}"))
        assert len(files) == 1


def test_create_fake_bids_dataset_no_derivatives(tmp_path):
    """Check no file is created in derivatives."""
    bids_path = create_fake_bids_dataset(
        base_dir=tmp_path,
        n_sub=1,
        n_ses=1,
        tasks=["main"],
        n_runs=[2],
        with_derivatives=False,
    )
    files = list(bids_path.glob("derivatives/**"))
    assert not files


@pytest.mark.parametrize(
    "confounds_tag,with_confounds", [(None, True), ("_timeseries", False)]
)
def test_create_fake_bids_dataset_no_confounds(
    tmp_path, confounds_tag, with_confounds
):
    """Check that files are created in the derivatives but no confounds."""
    bids_path = create_fake_bids_dataset(
        base_dir=tmp_path,
        n_sub=1,
        n_ses=1,
        tasks=["main"],
        n_runs=[2],
        with_confounds=with_confounds,
        confounds_tag=confounds_tag,
    )
    assert list(bids_path.glob("derivatives/*"))
    files = list(bids_path.glob("derivatives/*/*/func/*timeseries.tsv"))
    assert not files


def test_fake_bids_errors(tmp_path):
    with pytest.raises(ValueError, match="labels.*alphanumeric"):
        create_fake_bids_dataset(
            base_dir=tmp_path, n_sub=1, n_ses=1, tasks=["foo_bar"], n_runs=[1]
        )

    with pytest.raises(ValueError, match="labels.*alphanumeric"):
        create_fake_bids_dataset(
            base_dir=tmp_path,
            n_sub=1,
            n_ses=1,
            tasks=["main"],
            n_runs=[1],
            entities={"acq": "foo_bar"},
        )

    with pytest.raises(ValueError, match="number.*tasks.*runs.*same"):
        create_fake_bids_dataset(
            base_dir=tmp_path,
            n_sub=1,
            n_ses=1,
            tasks=["main"],
            n_runs=[1, 2],
        )


def test_fake_bids_extra_raw_entity(tmp_path):
    """Check files with extra entity are created appropriately."""
    n_sub = 2
    n_ses = 2
    tasks = ["main"]
    n_runs = [2]
    entities = {"acq": ["foo", "bar"]}
    bids_path = create_fake_bids_dataset(
        base_dir=tmp_path,
        n_sub=n_sub,
        n_ses=n_ses,
        tasks=tasks,
        n_runs=n_runs,
        entities=entities,
    )

    # raw
    for i, task in enumerate(tasks):
        for suffix in ["bold.nii.gz", "bold.json", "events.tsv"]:
            for label in entities["acq"]:
                file_pattern = _bids_path_template(
                    task=task,
                    suffix=suffix,
                    n_runs=n_runs[i],
                    extra_entity={"acq": label},
                )
                files = list(bids_path.glob(file_pattern))
                assert len(files) == n_sub * n_ses * n_runs[i]

    all_files = list(bids_path.glob("sub-*/ses-*/*/*"))
    # per subject:
    # 1 anat + (1 event + 1 json + 1 bold) per entity per run per session
    n_raw_files_expected = n_sub * (
        1 + 3 * sum(n_runs) * n_ses * len(entities["acq"])
    )
    assert len(all_files) == n_raw_files_expected

    # derivatives
    for label in entities["acq"]:
        for task, n_run in zip(tasks, n_runs):
            _check_n_files_derivatives_for_task(
                bids_path=bids_path,
                n_sub=n_sub,
                n_ses=n_ses,
                task=task,
                n_run=n_run,
                extra_entity={"acq": label},
            )

    all_files = list(bids_path.glob("derivatives/sub-*/ses-*/*/*"))
    # per subject: (2 confound + 3 bold + 2 gifti)
    #              per run per session per entity
    n_derivatives_files_expected = (
        n_sub * (7 * sum(n_runs) * n_ses) * len(entities["acq"])
    )
    assert len(all_files) == n_derivatives_files_expected


def test_fake_bids_extra_derivative_entity(tmp_path):
    """Check files with extra entity are created appropriately."""
    n_sub = 2
    n_ses = 2
    tasks = ["main"]
    n_runs = [2]
    entities = {"res": ["foo", "bar"]}
    bids_path = create_fake_bids_dataset(
        base_dir=tmp_path,
        n_sub=n_sub,
        n_ses=n_ses,
        tasks=tasks,
        n_runs=n_runs,
        entities=entities,
    )

    # raw
    all_files = list(bids_path.glob("sub-*/ses-*/*/*res*"))
    assert not all_files

    # derivatives
    for label in entities["res"]:
        for task, n_run in zip(tasks, n_runs):
            _check_n_files_derivatives_for_task(
                bids_path=bids_path,
                n_sub=n_sub,
                n_ses=n_ses,
                task=task,
                n_run=n_run,
                extra_entity={"res": label},
            )

    all_files = list(bids_path.glob("derivatives/sub-*/ses-*/*/*"))
    # per subject:
    # 1 confound per run per session
    # + (3 bold + 2 gifti) per run per session per entity
    n_derivatives_files_expected = n_sub * (
        2 * sum(n_runs) * n_ses
        + 5 * sum(n_runs) * n_ses * len(entities["res"])
    )
    assert len(all_files) == n_derivatives_files_expected


def test_fake_bids_extra_entity_not_bids_entity(tmp_path):
    """Check files with extra entity are created appropriately."""
    with pytest.raises(ValueError, match="Invalid entity"):
        create_fake_bids_dataset(
            base_dir=tmp_path,
            entities={"egg": ["spam"]},
        )


@pytest.mark.parametrize("window", ["boxcar", "hamming"])
def test_generate_regions_ts_no_overlap(window):
    n_voxels = 50
    n_regions = 10

    regions = generate_regions_ts(
        n_voxels, n_regions, overlap=0, window=window
    )

    assert regions.shape == (n_regions, n_voxels)
    # check no overlap
    np.testing.assert_array_less(
        (regions > 0).sum(axis=0) - 0.1, np.ones(regions.shape[1])
    )
    # check: a region everywhere
    np.testing.assert_array_less(
        np.zeros(regions.shape[1]), (regions > 0).sum(axis=0)
    )


@pytest.mark.parametrize("window", ["boxcar", "hamming"])
def test_generate_regions_ts_with_overlap(window):
    n_voxels = 50
    n_regions = 10

    regions = generate_regions_ts(
        n_voxels, n_regions, overlap=1, window=window
    )

    assert regions.shape == (n_regions, n_voxels)
    # check overlap
    assert np.any((regions > 0).sum(axis=-1) > 1.9)
    # check: a region everywhere
    np.testing.assert_array_less(
        np.zeros(regions.shape[1]), (regions > 0).sum(axis=0)
    )


def test_generate_labeled_regions():
    """Minimal testing of generate_labeled_regions."""
    shape = (3, 4, 5)
    n_regions = 10
    regions = generate_labeled_regions(shape, n_regions)
    assert regions.shape == shape
    assert len(np.unique(get_data(regions))) == n_regions + 1


def test_generate_maps():
    # Basic testing of generate_maps()
    shape = (10, 11, 12)
    n_regions = 9
    maps_img, _ = generate_maps(shape, n_regions, border=1)
    maps = get_data(maps_img)
    assert maps.shape == (*shape, n_regions)
    # no empty map
    assert np.all(abs(maps).sum(axis=0).sum(axis=0).sum(axis=0) > 0)
    # check border
    assert np.all(maps[0, ...] == 0)
    assert np.all(maps[:, 0, ...] == 0)
    assert np.all(maps[:, :, 0, :] == 0)


@pytest.mark.parametrize("shape", [(10, 11, 12), (6, 6, 7)])
@pytest.mark.parametrize("length", [16, 20])
@pytest.mark.parametrize("kind", ["noise", "step"])
@pytest.mark.parametrize(
    "n_block,block_size,block_type",
    [
        (None, None, None),
        (1, 1, "classification"),
        (4, 3, "classification"),
        (4, 4, "regression"),
    ],
)
def test_generate_fake_fmri(
    shape, length, kind, n_block, block_size, block_type, rng
):
    fake_fmri = generate_fake_fmri(
        shape=shape,
        length=length,
        kind=kind,
        n_blocks=n_block,
        block_size=block_size,
        block_type=block_type,
        random_state=rng,
    )

    assert fake_fmri[0].shape[:-1] == shape
    assert fake_fmri[0].shape[-1] == length
    if n_block is not None:
        assert fake_fmri[2].size == length


def test_generate_fake_fmri_error(rng):
    with pytest.raises(ValueError, match="10 is too small"):
        generate_fake_fmri(
            length=10,
            n_blocks=10,
            block_size=3,
            random_state=rng,
        )


@pytest.mark.parametrize(
    "shapes", [[(2, 3, 5, 7)], [(5, 5, 5, 3), (5, 5, 5, 5)]]
)
@pytest.mark.parametrize("rank", [1, 3, 5])
@pytest.mark.parametrize("affine", [None, np.diag([0.5, 0.3, 1, 1])])
def test_fake_fmri_data_and_design_generate(shapes, rank, affine):
    # test generate
    mask, fmri_data, design_matrices = generate_fake_fmri_data_and_design(
        shapes, rk=rank, affine=affine, random_state=42
    )

    for fmri, shape in zip(fmri_data, shapes):
        assert mask.shape == shape[:3]
        assert fmri.shape == shape
        if affine is not None:
            assert_almost_equal(fmri.affine, affine)

    for design, shape in zip(design_matrices, shapes):
        assert design.shape == (shape[3], rank)


@pytest.mark.parametrize(
    "shapes", [[(2, 3, 5, 7)], [(5, 5, 5, 3), (5, 5, 5, 5)]]
)
@pytest.mark.parametrize("rank", [1, 3, 5])
@pytest.mark.parametrize("affine", [None, np.diag([0.5, 0.3, 1, 1])])
def test_fake_fmri_data_and_design_write(tmp_path, shapes, rank, affine):
    mask, fmri_data, design_matrices = generate_fake_fmri_data_and_design(
        shapes, rk=rank, affine=affine, random_state=42
    )
    mask_file, fmri_files, design_files = write_fake_fmri_data_and_design(
        shapes, rk=rank, affine=affine, random_state=42, file_path=tmp_path
    )

    mask_img = load(mask_file)
    assert_almost_equal(mask_img.get_fdata(), mask.get_fdata())
    assert_almost_equal(mask_img.affine, mask.affine)

    for fmri_file, fmri in zip(fmri_files, fmri_data):
        fmri_img = load(fmri_file)
        assert_almost_equal(fmri_img.get_fdata(), fmri.get_fdata())
        assert_almost_equal(fmri_img.affine, fmri.affine)

    for design_file, design in zip(design_files, design_matrices):
        assert_frame_equal(
            pd.read_csv(design_file, sep="\t"), design, check_exact=False
        )


@pytest.mark.parametrize("shape", [(3, 4, 5), (2, 3, 5, 7)])
@pytest.mark.parametrize("affine", [None, np.diag([0.5, 0.3, 1, 1])])
def test_generate_random_img(shape, affine, rng):
    img, mask = generate_random_img(
        shape=shape, affine=affine, random_state=rng
    )

    assert img.shape == shape
    assert mask.shape == shape[:3]
    if affine is not None:
        assert_almost_equal(img.affine, affine)
        assert_almost_equal(mask.affine, affine)


@pytest.mark.parametrize("n_subjects", [5, 9])
@pytest.mark.parametrize("n_features", [30, 9])
@pytest.mark.parametrize("n_samples_range", [(30, 50), (9, 9)])
@pytest.mark.parametrize("density", [0.1, 1])
def test_generate_group_sparse_gaussian_graphs(
    n_subjects, n_features, n_samples_range, density, rng
):
    signals, precisions, topology = generate_group_sparse_gaussian_graphs(
        n_subjects=n_subjects,
        n_features=n_features,
        min_n_samples=n_samples_range[0],
        max_n_samples=n_samples_range[1],
        density=density,
        random_state=rng,
    )

    assert len(signals) == n_subjects
    assert len(precisions) == n_subjects

    signal_shapes = np.array([s.shape for s in signals])
    precision_shapes = np.array([p.shape for p in precisions])
    assert np.all(
        (signal_shapes[:, 0] >= n_samples_range[0])
        & (signal_shapes[:, 0] <= n_samples_range[1])
    )
    assert np.all(signal_shapes[:, 1] == n_features)
    assert np.all(precision_shapes == (n_features, n_features))
    assert topology.shape == (n_features, n_features)

    eigenvalues = np.array([np.linalg.eigvalsh(p) for p in precisions])
    assert np.all(eigenvalues >= 0)


@pytest.mark.parametrize("n_timepoints", [1, 9])
@pytest.mark.parametrize("n_features", [1, 9])
def test_generate_timeseries(n_timepoints, n_features, rng):
    timeseries = generate_timeseries(n_timepoints, n_features, rng)
    assert timeseries.shape == (n_timepoints, n_features)


@pytest.mark.parametrize("n_scans", [1, 5])
@pytest.mark.parametrize("res", [1, 30])
@pytest.mark.parametrize("mask_dilation", [1, 2])
def test_generate_mni_space_img(n_scans, res, mask_dilation, rng):
    inverse_img, mask_img = generate_mni_space_img(
        n_scans=n_scans, res=res, mask_dilation=mask_dilation, random_state=rng
    )

    def resample_dim(orig, res):
        return (orig - 2) // res + 2

    expected_shape = (
        resample_dim(197, res),
        resample_dim(233, res),
        resample_dim(189, res),
    )
    assert inverse_img.shape[:3] == expected_shape
    assert inverse_img.shape[3] == n_scans
    assert mask_img.shape == expected_shape
    assert_almost_equal(inverse_img.affine, mask_img.affine)
