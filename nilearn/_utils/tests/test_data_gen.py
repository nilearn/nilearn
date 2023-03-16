"""Tests for the data generation utilities."""

from pathlib import Path

import pytest
from nibabel.tmpdirs import InTemporaryDirectory
from nilearn._utils.data_gen import create_fake_bids_dataset


def _bids_path_template(
    task, suffix, n_runs=None, space=None, desc=None, extra_entity=None
):
    task = f"task-{task}_*"
    run = "run-*_*" if n_runs is not None else "*"
    space = f"space-{space}_*" if space is not None else "*"
    desc = f"desc-{desc}_*" if desc is not None else "*"

    # only testing with res and acq (for now)
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
def test_fake_bids_raw_with_session_and_runs(n_sub, n_ses, tasks, n_runs):
    """Check number of each file created in raw and derivatives."""
    with InTemporaryDirectory():
        bids_path = create_fake_bids_dataset(
            n_sub=n_sub, n_ses=n_ses, tasks=tasks, n_runs=n_runs
        )

        bids_path = Path(bids_path)

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


def _check_nb_files_derivatives_for_task(
    bids_path, n_sub, n_ses, task, n_run, extra_entity=None
):
    for suffix in ["timeseries.tsv"]:
        # 1 confound per raw file
        # so we do not use the extra entity for derivatives entities
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
    n_sub, n_ses, tasks, n_runs
):
    """Check number of each file created in derivatives."""
    with InTemporaryDirectory():
        bids_path = create_fake_bids_dataset(
            n_sub=n_sub, n_ses=n_ses, tasks=tasks, n_runs=n_runs
        )

        bids_path = Path(bids_path)

        # derivatives
        for task, n_run in zip(tasks, n_runs):
            _check_nb_files_derivatives_for_task(
                bids_path=bids_path,
                n_sub=n_sub,
                n_ses=n_ses,
                task=task,
                n_run=n_run,
            )

        all_files = list(bids_path.glob("derivatives/sub-*/ses-*/*/*"))
        # per subject: (1 confound + 3 bold) per run per session
        n_derivatives_files_expected = n_sub * (4 * sum(n_runs) * n_ses)
        assert len(all_files) == n_derivatives_files_expected


def test_bids_dataset_no_run_entity():
    """n_runs==0 produces files without the run entity."""
    with InTemporaryDirectory():
        bids_path = create_fake_bids_dataset(
            n_sub=1,
            n_ses=1,
            tasks=["main"],
            n_runs=[0],
            with_derivatives=True,
        )
        bids_path = Path(bids_path)

        files = list(bids_path.glob("**/*run-*"))
        assert not files

        # nifti: 1 anat + 1 raw bold + 3 derivatives bold
        files = list(bids_path.glob("**/*.nii.gz"))
        assert len(files) == 5

        # events or json or confounds: 1
        for suffix in ["events.tsv", "timeseries.tsv", "bold.json"]:
            files = list(bids_path.glob(f"**/*{suffix}"))
            assert len(files) == 1


@pytest.mark.parametrize("n_ses,no_session", [(1, True), (0, False)])
def test_bids_dataset_no_session(n_ses, no_session):
    """n_ses =0 and no_session prevent the creation of a session folder."""
    with InTemporaryDirectory():
        bids_path = create_fake_bids_dataset(
            n_sub=1,
            n_ses=n_ses,
            tasks=["main"],
            n_runs=[1],
            no_session=no_session,
            with_derivatives=True,
        )
        bids_path = Path(bids_path)

        files = list(bids_path.glob("**/*ses-*"))
        assert not files

        # nifti: 1 anat + 1 raw bold + 3 derivatives bold
        files = list(bids_path.glob("**/*.nii.gz"))
        assert len(files) == 5

        # events or json or confounds: 1
        for suffix in ["events.tsv", "timeseries.tsv", "bold.json"]:
            files = list(bids_path.glob(f"**/*{suffix}"))
            assert len(files) == 1


def test_create_fake_bids_dataset_no_derivatives():
    """Check no file is created in derivatives."""
    with InTemporaryDirectory():
        bids_path = create_fake_bids_dataset(
            n_sub=1,
            n_ses=1,
            tasks=["main"],
            n_runs=[2],
            with_derivatives=False,
        )
        bids_path = Path(bids_path)
        files = list(bids_path.glob("derivatives/**"))
        assert not files


@pytest.mark.parametrize(
    "confounds_tag,with_confounds", [(None, True), ("_timeseries", False)]
)
def test_create_fake_bids_dataset_no_confounds(confounds_tag, with_confounds):
    """Check that files are created in the derivatives but no confounds."""
    with InTemporaryDirectory():
        bids_path = create_fake_bids_dataset(
            n_sub=1,
            n_ses=1,
            tasks=["main"],
            n_runs=[2],
            with_confounds=with_confounds,
            confounds_tag=confounds_tag,
        )
        bids_path = Path(bids_path)
        assert list(bids_path.glob("derivatives/*"))
        files = list(bids_path.glob("derivatives/*/*/func/*timeseries.tsv"))
        assert not files


def test_fake_bids_errors():
    with InTemporaryDirectory():
        with pytest.raises(ValueError, match="labels.*alphanumeric"):
            create_fake_bids_dataset(
                n_sub=1, n_ses=1, tasks=["foo_bar"], n_runs=[1]
            )

        with pytest.raises(ValueError, match="labels.*alphanumeric"):
            create_fake_bids_dataset(
                n_sub=1,
                n_ses=1,
                tasks=["main"],
                n_runs=[1],
                entities={"acq": "foo_bar"},
            )

        with pytest.raises(ValueError, match="number.*tasks.*runs.*same"):
            create_fake_bids_dataset(
                n_sub=1,
                n_ses=1,
                tasks=["main"],
                n_runs=[1, 2],
            )


def test_fake_bids_extra_raw_entity():
    """Check files with extra entity are created appropriately."""
    with InTemporaryDirectory():
        n_sub = 2
        n_ses = 2
        tasks = ["main"]
        n_runs = [2]
        entities = {"acq": ["foo", "bar"]}
        bids_path = create_fake_bids_dataset(
            n_sub=n_sub,
            n_ses=n_ses,
            tasks=tasks,
            n_runs=n_runs,
            entities=entities,
        )

        bids_path = Path(bids_path)

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
                _check_nb_files_derivatives_for_task(
                    bids_path=bids_path,
                    n_sub=n_sub,
                    n_ses=n_ses,
                    task=task,
                    n_run=n_run,
                    extra_entity={"acq": label},
                )

        all_files = list(bids_path.glob("derivatives/sub-*/ses-*/*/*"))
        # per subject: (1 confound + 3 bold) per run per session per entity
        n_derivatives_files_expected = (
            n_sub * (4 * sum(n_runs) * n_ses) * len(entities["acq"])
        )
        assert len(all_files) == n_derivatives_files_expected


def test_fake_bids_extra_derivative_entity():
    """Check files with extra entity are created appropriately."""
    with InTemporaryDirectory():
        n_sub = 2
        n_ses = 2
        tasks = ["main"]
        n_runs = [2]
        entities = {"res": ["foo", "bar"]}
        bids_path = create_fake_bids_dataset(
            n_sub=n_sub,
            n_ses=n_ses,
            tasks=tasks,
            n_runs=n_runs,
            entities=entities,
        )

        bids_path = Path(bids_path)

        # raw
        all_files = list(bids_path.glob("sub-*/ses-*/*/*res*"))
        assert not all_files

        # derivatives
        for label in entities["res"]:
            for task, n_run in zip(tasks, n_runs):
                _check_nb_files_derivatives_for_task(
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
        # + (3 bold) per run per session per entity
        n_derivatives_files_expected = n_sub * (
            1 * sum(n_runs) * n_ses
            + 3 * sum(n_runs) * n_ses * len(entities["res"])
        )
        assert len(all_files) == n_derivatives_files_expected
