"""Test related to first level model."""

import shutil
import warnings
from itertools import product
from pathlib import Path

import pandas as pd
import pytest

from nilearn._utils.data_gen import (
    add_metadata_to_bids_dataset,
    create_fake_bids_dataset,
)
from nilearn.glm.first_level import (
    FirstLevelModel,
    first_level_from_bids,
)
from nilearn.interfaces.bids import get_bids_files
from nilearn.surface import SurfaceImage


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


def _new_bids_dataset(base_dir=None):
    """Create a new BIDS dataset for testing purposes.

    Use if the dataset needs to be modified after creation.
    """
    if base_dir is None:
        base_dir = Path()
    n_sub, n_ses, tasks, n_runs = _inputs_for_new_bids_dataset()
    return create_fake_bids_dataset(
        base_dir=base_dir, n_sub=n_sub, n_ses=n_ses, tasks=tasks, n_runs=n_runs
    )


def _check_output_first_level_from_bids(
    n_sub, models, imgs, events, confounds
):
    assert len(models) == n_sub
    assert all(isinstance(model, FirstLevelModel) for model in models)

    assert len(models) == len(imgs)
    for img_ in imgs:
        assert isinstance(img_, list)

        # We should only get lists of valid paths or lists of SurfaceImages
        if all(isinstance(x, str) for x in img_):
            assert all(Path(x).exists() for x in img_)
        else:
            assert all(isinstance(x, SurfaceImage) for x in img_)

    assert len(models) == len(events)
    for event_ in events:
        assert isinstance(event_, list)
        assert all(isinstance(x, pd.DataFrame) for x in event_)

    assert len(models) == len(confounds)
    for confound_ in confounds:
        assert isinstance(confound_, list)
        assert all(isinstance(x, pd.DataFrame) for x in confound_)


def test_set_repetition_time_warnings(tmp_path):
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
            slice_time_ref=0.0,  # set to 0.0 to avoid warnings
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
        ("not a number", TypeError, "must be of type"),
        (-1, ValueError, "positive"),
    ],
)
def test_set_repetition_time_errors(tmp_path, t_r, error_type, error_msg):
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


def test_set_slice_timing_ref_warnings(tmp_path):
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
        ("not a number", TypeError, "must be of type"),
        (2, ValueError, "between 0 and 1"),
    ],
)
def test_set_slice_timing_ref_errors(
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


@pytest.mark.single_process
def test_get_metadata_from_derivatives(tmp_path):
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


def test_get_repetition_time_from_derivatives(tmp_path):
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

    with pytest.warns(RuntimeWarning, match="StartTime' not found in file"):
        models, *_ = first_level_from_bids(
            dataset_path=str(tmp_path / bids_path),
            task_label="main",
            space_label="MNI",
            img_filters=[("desc", "preproc")],
            slice_time_ref=0.0,  # set to 0.0 to avoid warnings
        )
        assert models[0].t_r == 6.0
        assert models[0].slice_time_ref == 0.0


def test_get_start_time_from_derivatives(tmp_path):
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

    with pytest.warns(
        RuntimeWarning, match="RepetitionTime' not found in file"
    ):
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


@pytest.mark.parametrize("n_runs", ([1, 0], [1, 2]))
@pytest.mark.parametrize("n_ses", [0, 2])
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
        slice_time_ref=0.0,  # set to 0.0 to avoid warnings
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


def test_exclude_subject(tmp_path):
    """Test several BIDS structure."""
    n_sub = 2
    n_ses = 1
    n_runs = [1]
    task_label = "main"
    space_label = "MNI"

    bids_path = create_fake_bids_dataset(
        base_dir=tmp_path,
        n_sub=n_sub,
        n_ses=n_ses,
        tasks=[task_label],
        n_runs=n_runs,
    )

    models, _, _, _ = first_level_from_bids(
        dataset_path=bids_path,
        task_label=task_label,
        space_label=space_label,
        img_filters=[("desc", "preproc")],
        exclude_subjects=["01"],
        slice_time_ref=0.0,  # set to 0.0 to avoid warnings
    )

    assert len(models) == 1


@pytest.mark.parametrize("slice_time_ref", [None, 0.0, 0.5, 1.0])
def test_slice_time_ref(bids_dataset, slice_time_ref):
    """Test several valid values of slice_time_ref."""
    n_sub, *_ = _inputs_for_new_bids_dataset()
    models, imgs, events, confounds = first_level_from_bids(
        dataset_path=bids_dataset,
        task_label="main",
        space_label="MNI",
        img_filters=[("run", "01"), ("desc", "preproc")],
        slice_time_ref=slice_time_ref,
    )

    _check_output_first_level_from_bids(n_sub, models, imgs, events, confounds)


def test_space_none(tmp_path):
    """Test behavior when no specific space is required .

    Function should look for images with MNI152NLin2009cAsym.
    """
    n_sub = 1
    bids_path = create_fake_bids_dataset(
        base_dir=tmp_path, n_sub=n_sub, spaces=["MNI152NLin2009cAsym"]
    )
    models, imgs, events, confounds = first_level_from_bids(
        dataset_path=bids_path,
        task_label="main",
        space_label=None,
        img_filters=[("run", "01"), ("desc", "preproc")],
        slice_time_ref=0.0,  # set to 0.0 to avoid warnings
    )

    _check_output_first_level_from_bids(n_sub, models, imgs, events, confounds)


def test_select_one_run_per_session(bids_dataset):
    """Check that img_filters can select a single file per run per session."""
    n_sub, n_ses, *_ = _inputs_for_new_bids_dataset()

    models, imgs, events, confounds = first_level_from_bids(
        dataset_path=bids_dataset,
        task_label="main",
        space_label="MNI",
        img_filters=[("run", "01"), ("desc", "preproc")],
        slice_time_ref=0.0,  # set to 0.0 to avoid warnings
    )

    _check_output_first_level_from_bids(n_sub, models, imgs, events, confounds)

    n_imgs_expected = n_ses
    assert len(imgs[0]) == n_imgs_expected


def test_select_all_runs_of_one_session(bids_dataset):
    """Check that img_filters can select all runs in a session."""
    n_sub, _, _, n_runs = _inputs_for_new_bids_dataset()

    models, imgs, events, confounds = first_level_from_bids(
        dataset_path=bids_dataset,
        task_label="main",
        space_label="MNI",
        img_filters=[("ses", "01"), ("desc", "preproc")],
        slice_time_ref=0.0,  # set to 0.0 to avoid warnings
    )

    _check_output_first_level_from_bids(n_sub, models, imgs, events, confounds)

    n_imgs_expected = n_runs[0]
    assert len(imgs[0]) == n_imgs_expected


def test_smoke_test_for_verbose_argument(bids_dataset, capsys):
    """Test with verbose mode.

    verbose = 0 is the default, so should be covered by other tests.
    """
    first_level_from_bids(
        dataset_path=bids_dataset,
        task_label="main",
        space_label="MNI",
        img_filters=[("desc", "preproc")],
        verbose=1,
        slice_time_ref=0.0,  # set to 0.0 to avoid warnings
    )
    assert len(capsys.readouterr().out) > 0


@pytest.mark.parametrize(
    "entity", ["acq", "ce", "dir", "rec", "echo", "res", "den"]
)
def test_several_labels_per_entity(tmp_path, entity):
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
        slice_time_ref=0.0,  # set to 0.0 to avoid warnings
    )

    _check_output_first_level_from_bids(n_sub, models, imgs, events, confounds)
    n_imgs_expected = n_ses * n_runs[0]
    assert len(imgs[0]) == n_imgs_expected


def test_with_subject_labels(bids_dataset):
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
            slice_time_ref=0.0,  # set to 0.0 to avoid warnings
        )

        assert models[0].subject_label == "01"


def test_no_duplicate_sub_labels(bids_dataset):
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
        slice_time_ref=0.0,  # set to 0.0 to avoid warnings
    )

    assert len(models) == 1


def test_validation_input_dataset_path():
    """Raise error when dataset_path is invalid."""
    with pytest.raises(TypeError, match="must be of type"):
        first_level_from_bids(
            dataset_path=2,
            task_label="main",
            space_label="MNI",
            slice_time_ref=0.0,  # set to 0.0 to avoid warnings
        )
    with pytest.raises(ValueError, match="'dataset_path' does not exist"):
        first_level_from_bids(
            dataset_path="lolo",
            task_label="main",
            space_label="MNI",
            slice_time_ref=0.0,  # set to 0.0 to avoid warnings
        )
    with pytest.raises(TypeError, match=r"derivatives_.* must be of type"):
        first_level_from_bids(
            dataset_path=Path(),
            task_label="main",
            space_label="MNI",
            derivatives_folder=1,
            slice_time_ref=0.0,  # set to 0.0 to avoid warnings
        )


@pytest.mark.parametrize(
    "task_label, error_type",
    [(42, TypeError), ("$$$", ValueError)],
)
def test_validation_task_label(bids_dataset, task_label, error_type):
    """Raise error for invalid task_label."""
    with pytest.raises(error_type, match="All bids labels must be "):
        first_level_from_bids(
            dataset_path=bids_dataset, task_label=task_label, space_label="MNI"
        )


@pytest.mark.parametrize(
    "sub_labels, error_type, error_msg",
    [
        ("42", TypeError, "must be of type"),
        (["1", 1], TypeError, "must be string"),
        ([1], TypeError, "must be string"),
    ],
)
def test_validation_sub_labels(
    bids_dataset, sub_labels, error_type, error_msg
):
    """Raise error for invalid sub_labels."""
    with pytest.raises(error_type, match=error_msg):
        first_level_from_bids(
            dataset_path=bids_dataset,
            task_label="main",
            sub_labels=sub_labels,
            slice_time_ref=0.0,  # set to 0.0 to avoid warnings
        )


@pytest.mark.parametrize(
    "space_label, error_type",
    [(42, TypeError), ("$$$", ValueError)],
)
def test_validation_space_label(bids_dataset, space_label, error_type):
    """Raise error when space_label is invalid."""
    with pytest.raises(error_type, match="All bids labels must be "):
        first_level_from_bids(
            dataset_path=bids_dataset,
            task_label="main",
            space_label=space_label,
            slice_time_ref=0.0,  # set to 0.0 to avoid warnings
        )


@pytest.mark.parametrize(
    "img_filters, error_type,match",
    [
        ("foo", TypeError, "'img_filters' must be of type"),
        ([(1, 2)], TypeError, "Filters in img"),
        ([("desc", "*/-")], ValueError, "bids labels must be alphanumeric."),
        ([("foo", "bar")], ValueError, "must be one of"),
    ],
)
def test_validation_img_filter(bids_dataset, img_filters, error_type, match):
    """Raise error when img_filters is invalid."""
    with pytest.raises(error_type, match=match):
        first_level_from_bids(
            dataset_path=bids_dataset,
            task_label="main",
            img_filters=img_filters,
            slice_time_ref=0.0,  # set to 0.0 to avoid warnings
        )


def test_too_many_bold_files(bids_dataset):
    """Too many bold files if img_filters is underspecified, \
       should raise an error.

    Here there is a desc-preproc and desc-fmriprep image for the space-T1w.
    """
    with pytest.raises(ValueError, match="Too many images found"):
        first_level_from_bids(
            dataset_path=bids_dataset,
            task_label="main",
            space_label="T1w",
            slice_time_ref=0.0,  # set to 0.0 to avoid warnings
        )


@pytest.mark.thread_unsafe
def test_with_missing_events(tmp_path_factory):
    """All events.tsv files are missing, should raise an error."""
    bids_dataset = _new_bids_dataset(tmp_path_factory.mktemp("no_events"))
    events_files = get_bids_files(main_path=bids_dataset, file_tag="events")
    for f in events_files:
        Path(f).unlink()

    with pytest.raises(ValueError, match=r"No events.tsv files found"):
        first_level_from_bids(
            dataset_path=bids_dataset,
            task_label="main",
            space_label="MNI",
            slice_time_ref=0.0,  # set to 0.0 to avoid warnings
        )


def test_no_tr(tmp_path_factory):
    """Throw warning when t_r information cannot be inferred from the data \
    and t_r=None is passed.
    """
    bids_dataset = _new_bids_dataset(tmp_path_factory.mktemp("no_events"))
    json_files = get_bids_files(
        main_path=bids_dataset, file_tag="bold", file_type="json"
    )
    for f in json_files:
        Path(f).unlink()

    with pytest.warns(
        RuntimeWarning, match="'t_r' not provided and cannot be inferred"
    ):
        first_level_from_bids(
            dataset_path=bids_dataset,
            task_label="main",
            space_label="MNI",
            slice_time_ref=0.0,  # set to 0.0 to avoid warnings
            t_r=None,
        )


def test_no_bold_file(tmp_path_factory):
    """Raise error when no bold file in BIDS dataset."""
    bids_dataset = _new_bids_dataset(tmp_path_factory.mktemp("no_bold"))
    imgs = get_bids_files(
        main_path=bids_dataset / "derivatives",
        file_tag="bold",
        file_type="*gz",
    )
    for img_ in imgs:
        Path(img_).unlink()

    with pytest.raises(ValueError, match="No BOLD files found "):
        first_level_from_bids(
            dataset_path=bids_dataset,
            task_label="main",
            space_label="MNI",
            slice_time_ref=0.0,  # set to 0.0 to avoid warnings
        )


@pytest.mark.thread_unsafe
def test_with_one_events_missing(tmp_path_factory):
    """Only one events.tsv file is missing, should raise an error."""
    bids_dataset = _new_bids_dataset(
        tmp_path_factory.mktemp("one_event_missing")
    )
    events_files = get_bids_files(main_path=bids_dataset, file_tag="events")
    Path(events_files[0]).unlink()

    with pytest.raises(ValueError, match="Same number of event files "):
        first_level_from_bids(
            dataset_path=bids_dataset,
            task_label="main",
            space_label="MNI",
            slice_time_ref=0.0,  # set to 0.0 to avoid warnings
        )


@pytest.mark.thread_unsafe
def test_one_confound_missing(tmp_path_factory):
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
    Path(confound_files[-1]).unlink()

    with pytest.raises(ValueError, match="Same number of confound"):
        first_level_from_bids(
            dataset_path=bids_dataset,
            task_label="main",
            space_label="MNI",
            slice_time_ref=0.0,  # set to 0.0 to avoid warnings
        )


def test_all_confounds_missing(tmp_path_factory):
    """If all confound files are missing, \
    confounds should be an array of None.
    """
    bids_dataset = _new_bids_dataset(tmp_path_factory.mktemp("no_confounds"))
    confound_files = get_bids_files(
        main_path=bids_dataset / "derivatives",
        file_tag="desc-confounds_timeseries",
    )
    for f in confound_files:
        Path(f).unlink()

    models, imgs, events, confounds = first_level_from_bids(
        dataset_path=bids_dataset,
        task_label="main",
        space_label="MNI",
        img_filters=[("desc", "preproc")],
        slice_time_ref=0.0,  # set to 0.0 to avoid warnings
    )

    assert len(models) == len(imgs)
    assert len(models) == len(events)
    assert len(models) == len(confounds)
    for condounds_ in confounds:
        assert condounds_ is None


def test_no_derivatives(tmp_path):
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
            slice_time_ref=0.0,  # set to 0.0 to avoid warnings
        )


def test_no_session(tmp_path):
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
            slice_time_ref=0.0,  # set to 0.0 to avoid warnings
        )


@pytest.mark.thread_unsafe
def test_mismatch_run_index(tmp_path_factory):
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

    with pytest.raises(ValueError, match=r".*events.tsv files.*"):
        first_level_from_bids(
            dataset_path=bids_dataset,
            task_label="main",
            space_label="MNI",
            img_filters=[("desc", "preproc")],
            slice_time_ref=0.0,  # set to 0.0 to avoid warnings
        )


def test_slice_time_ref_warning_only_when_not_provided(bids_dataset):
    """Catch warning when slice_time_ref is not provided."""
    with pytest.warns() as record:
        first_level_from_bids(
            dataset_path=bids_dataset,
            task_label="main",
            space_label="MNI",
            img_filters=[("desc", "preproc")],
            slice_time_ref=0.6,
        )

    # check that no warnings were raised
    for r in record:
        assert "'slice_time_ref' not provided" not in r.message.args[0]


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
    events = events.drop(columns="trial_type")
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


def test_load_confounds(tmp_path):
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
        slice_time_ref=0.0,  # set to 0.0 to avoid warnings
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
        slice_time_ref=0.0,  # set to 0.0 to avoid warnings
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


def test_load_confounds_warnings(tmp_path):
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
        slice_time_ref=0.0,  # set to 0.0 to avoid warnings
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
            slice_time_ref=0.0,  # set to 0.0 to avoid warnings
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
            slice_time_ref=0.0,  # set to 0.0 to avoid warnings
        )


def test_no_subject(tmp_path):
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
            slice_time_ref=0.0,  # set to 0.0 to avoid warnings
        )


def test_unused_kwargs(tmp_path):
    """Check that unused kwargs are properly handled."""
    bids_path = create_fake_bids_dataset(
        base_dir=tmp_path, n_sub=1, n_ses=1, tasks=["main"], n_runs=[2]
    )
    with pytest.raises(RuntimeError, match="Unknown keyword arguments"):
        # wrong kwarg name `confound_strategy` (wrong)
        # instead of `confounds_strategy` (correct)
        first_level_from_bids(
            dataset_path=bids_path,
            task_label="main",
            space_label="MNI",
            slice_time_ref=0.0,  # set to 0.0 to avoid warnings
            confound_strategy="motion",
        )


def test_subject_order(tmp_path):
    """Make sure subjects are returned in order.

    See https://github.com/nilearn/nilearn/issues/4581
    """
    n_sub = 10
    bids_path = create_fake_bids_dataset(
        base_dir=tmp_path, n_sub=n_sub, n_ses=1, tasks=["main"], n_runs=[1]
    )

    models, *_ = first_level_from_bids(
        dataset_path=str(tmp_path / bids_path),
        task_label="main",
        space_label="MNI",
        img_filters=[("desc", "preproc")],
        slice_time_ref=0.0,  # set to 0.0 to avoid warnings
    )

    # Check if the subjects are returned in order
    expected_subjects = [f"{label:02}" for label in range(1, n_sub + 1)]
    returned_subjects = [model.subject_label for model in models]
    assert returned_subjects == expected_subjects


@pytest.mark.single_process
def test_subject_order_with_labels(tmp_path):
    """Make sure subjects are returned in order.

    See https://github.com/nilearn/nilearn/issues/4581
    """
    n_sub = 10
    bids_path = create_fake_bids_dataset(
        base_dir=tmp_path, n_sub=n_sub, n_ses=1, tasks=["main"], n_runs=[1]
    )

    models, *_ = first_level_from_bids(
        dataset_path=str(tmp_path / bids_path),
        sub_labels=["01", "10", "04", "05", "02", "03"],
        task_label="main",
        space_label="MNI",
        img_filters=[("desc", "preproc")],
        slice_time_ref=0.0,  # set to 0.0 to avoid warnings
    )

    # Check if the subjects are returned in order
    expected_subjects = ["01", "02", "03", "04", "05", "10"]
    returned_subjects = [model.subject_label for model in models]
    assert returned_subjects == expected_subjects


def test_surface(tmp_path):
    """Test finding and loading Surface data in BIDS dataset."""
    n_sub = 2
    tasks = ["main"]
    n_runs = [2]

    bids_path = create_fake_bids_dataset(
        base_dir=tmp_path,
        n_sub=n_sub,
        n_ses=0,
        tasks=tasks,
        n_runs=n_runs,
        n_vertices=10242,
    )

    models, imgs, events, confounds = first_level_from_bids(
        dataset_path=bids_path,
        task_label="main",
        space_label="fsaverage5",
    )

    _check_output_first_level_from_bids(n_sub, models, imgs, events, confounds)


def test_one_condition_missing(tmp_path):
    """One condition is missing in one events.tsv file.

    Should raise error when using formula for contrast
    when one or more conditions are missing.
    """
    n_runs = 1
    n_ses = 1
    bids_dataset = create_fake_bids_dataset(
        base_dir=tmp_path,
        n_sub=1,
        n_ses=n_ses,
        tasks=["main"],
        n_runs=[n_runs],
        n_voxels=10,
    )

    # remove rows with c0 and c2
    # from "trial_type" columns in events.tsv
    events_files = get_bids_files(main_path=bids_dataset, file_tag="events")
    events = pd.read_csv(events_files[0], sep="\t")
    events = events[events["trial_type"] != "c0"]
    events = events[events["trial_type"] != "c2"]
    events.to_csv(events_files[0], sep="\t", index=False)

    models, models_run_imgs, models_events, _ = first_level_from_bids(
        dataset_path=bids_dataset,
        task_label="main",
        space_label="MNI",
        slice_time_ref=0,
    )

    models[0].fit(models_run_imgs[0], models_events[0])

    with pytest.raises(ValueError, match="'c0' is not defined"):
        models[0].compute_contrast("c0-c1+c2")
