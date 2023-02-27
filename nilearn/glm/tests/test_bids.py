"""Test the first level model on BIDS datasets."""
import os

from pathlib import Path

import pytest

from nilearn._utils.data_gen import create_fake_bids_dataset
from nilearn.glm.first_level import first_level_from_bids
from nilearn.interfaces.bids import get_bids_files
from nibabel.tmpdirs import InTemporaryDirectory


def test_first_level_from_bids_bug_3029():
    """Test error when events.tsv is missing for a bold file."""
    with InTemporaryDirectory():
        bids_path = create_fake_bids_dataset(
            n_sub=2, n_ses=2, tasks=["main"], n_runs=[2]
        )
        files_to_rename = (
            Path(bids_path)
            .joinpath("derivatives")
            .glob("**/func/*_task-main_*desc-*")
        )
        for file in files_to_rename:
            new_file = file.parent / file.name.replace("run-0", "run-")
            file.rename(new_file)

        with pytest.raises(ValueError, match=".*events.tsv files.*"):
            first_level_from_bids(
                dataset_path=bids_path,
                task_label="main",
                space_label="MNI",
                img_filters=[("desc", "preproc")]
            )


def test_first_level_from_bids():
    with InTemporaryDirectory():
        n_sub = 2
        n_ses = 2
        n_runs = [2]

        bids_path = create_fake_bids_dataset(
            n_sub=2, n_ses=2, tasks=["main"], n_runs=[2]
        )
        # test output is as expected
        models, m_imgs, m_events, m_confounds = first_level_from_bids(
            dataset_path=bids_path,
            task_label="main",
            space_label="MNI",
            img_filters=[("desc", "preproc")],
        )
        assert len(models) == n_sub
        assert len(models) == len(m_imgs)
        assert len(models) == len(m_events)
        assert len(models) == len(m_confounds)
        assert len(m_imgs[0]) == n_ses * n_runs[0]

        # test verbose
        models, m_imgs, m_events, m_confounds = first_level_from_bids(
            dataset_path=bids_path,
            task_label="main",
            space_label="MNI",
            img_filters=[("desc", "preproc")],
            verbose=1,
        )


@pytest.mark.parametrize(
    "entity", ["acq", "ce", "dir", "rec", "echo", "res", "den"]
)
def test_first_level_from_bids_bug_3524(entity):
    """Test right files are selected when entities have several labels."""
    with InTemporaryDirectory():
        n_sub = 2
        n_ses = 2
        n_runs = [3]

        bids_path = create_fake_bids_dataset(
            n_sub=n_sub,
            n_ses=n_ses,
            tasks=["main"],
            n_runs=n_runs,
            entities=[entity, ["A", "B"]],
        )

        models, m_imgs, m_events, m_confounds = first_level_from_bids(
            dataset_path=bids_path,
            task_label="main",
            space_label="MNI",
            img_filters=[("desc", "preproc"), (entity, "A")],
        )
        assert len(models) == n_sub
        assert len(models) == len(m_imgs)
        assert len(models) == len(m_events)
        assert len(models) == len(m_confounds)
        assert len(m_imgs[0]) == n_ses * n_runs[0]


def test_first_level_from_bids_validation_input():
    with InTemporaryDirectory():
        bids_path = create_fake_bids_dataset(
            n_sub=3, n_ses=2, tasks=["localizer", "main"], n_runs=[1, 3]
        )
        # test arguments are provided correctly
        with pytest.raises(TypeError):
            first_level_from_bids(2, task_label="main", space_label="MNI")
        with pytest.raises(ValueError):
            first_level_from_bids("lolo", task_label="main", space_label="MNI")
        with pytest.raises(TypeError):
            first_level_from_bids(
                dataset_path=bids_path, task_label=2, space_label="MNI"
            )
        with pytest.raises(TypeError):
            first_level_from_bids(
                dataset_path=bids_path,
                task_label="main",
                space_label="MNI",
                model_init=[],
            )

        with pytest.raises(TypeError, match="space_label must be a string"):
            first_level_from_bids(
                dataset_path=bids_path, task_label="main", space_label=42
            )

        with pytest.raises(TypeError, match="img_filters must be a list"):
            first_level_from_bids(
                dataset_path=bids_path, task_label="main", img_filters="foo"
            )
        with pytest.raises(TypeError, match="filters in img"):
            first_level_from_bids(
                dataset_path=bids_path, task_label="main", img_filters=[(1, 2)]
            )
        with pytest.raises(
            ValueError, match="field foo is not a possible filter."
        ):
            first_level_from_bids(
                dataset_path=bids_path,
                task_label="main",
                img_filters=[("foo", "bar")],
            )


def test_first_level_from_bids_with_missing_files():
    with InTemporaryDirectory():
        bids_path = create_fake_bids_dataset(
            n_sub=2, n_ses=2, tasks=["localizer", "main"], n_runs=[1, 2]
        )
        # test repeated run tag error when run tag is in filenames
        # can arise when desc or space is present and not specified
        #
        # desc not specified
        with pytest.raises(ValueError):
            first_level_from_bids(
                dataset_path=bids_path, task_label="main", space_label="T1w"
            )
        # test more than one ses file error when run tag is not in filenames
        # can arise when desc or space is present and not specified
        #
        # desc not specified
        with pytest.raises(ValueError):
            first_level_from_bids(
                dataset_path=bids_path,
                task_label="localizer",
                space_label="T1w",
            )


def test_first_level_from_bids_with_one_events_missing():
    with InTemporaryDirectory():
        bids_path = create_fake_bids_dataset(
            n_sub=2, n_ses=2, tasks=["main"], n_runs=[2]
        )
        # test issues with event files
        events_files = get_bids_files(bids_path, file_tag="events")
        os.remove(events_files[0])
        # one file missing
        with pytest.raises(
            ValueError, match="Same number of event files "
        ):
            first_level_from_bids(
                dataset_path=bids_path, task_label="main", space_label="MNI"
            )


def test_first_level_from_bids_with_missing_events():
    with InTemporaryDirectory():
        bids_path = create_fake_bids_dataset(
            n_sub=2, n_ses=2, tasks=["main"], n_runs=[2]
        )
        # test issues with event files
        events_files = get_bids_files(bids_path, file_tag="events")
        for f in events_files[1:]:
            os.remove(f)
        # all files missing
        with pytest.raises(ValueError, match="Same number of event files "):
            first_level_from_bids(
                dataset_path=bids_path, task_label="main", space_label="MNI"
            )


def test_first_level_from_bids_one_confound_missing():
    """Test issues with confound files.

    There should be only one confound file per img.
    And one per image or None.
    """
    with InTemporaryDirectory():
        bids_path = create_fake_bids_dataset(
            n_sub=2, n_ses=2, tasks=["main"], n_runs=[2]
        )
        # case when one is missing
        confound_files = get_bids_files(
            os.path.join(bids_path, "derivatives"),
            file_tag="desc-confounds_timeseries",
        )
        os.remove(confound_files[-1])

        with pytest.raises(ValueError, match="Same number of confound"):
            first_level_from_bids(
                dataset_path=bids_path, task_label="main", space_label="MNI"
            )


def test_first_level_from_bids_no_derivatives():
    with InTemporaryDirectory():
        bids_path = create_fake_bids_dataset(
            n_sub=1,
            n_ses=1,
            tasks=["main"],
            n_runs=[1],
            with_derivatives=False,
        )
        with pytest.raises(ValueError, match="derivatives folder does not"):
            first_level_from_bids(
                dataset_path=bids_path, task_label="main", space_label="MNI"
            )


def test_first_level_from_bids_no_session():
    """Check runs are not repeated when ses field is not used."""
    with InTemporaryDirectory():
        bids_path = create_fake_bids_dataset(
            n_sub=3, n_ses=1, tasks=["main"], n_runs=[2], no_session=True
        )
        # test repeated run tag error when run tag is in filenames and not ses
        # can arise when desc or space is present and not specified
        with pytest.raises(ValueError):
            first_level_from_bids(
                dataset_path=bids_path, task_label="main", space_label="T1w"
            )
