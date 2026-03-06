import json
from pathlib import Path

import pytest

from nilearn._utils.data_gen import (
    add_metadata_to_bids_dataset,
    create_fake_bids_dataset,
)
from nilearn.interfaces.bids.query import (
    _get_metadata_from_bids,
    get_bids_files,
    infer_repetition_time_from_dataset,
    infer_slice_timing_start_time_from_dataset,
    parse_bids_filename,
)


def test_parse_bids_filename():
    """Check that a typical BIDS file is properly parsed."""
    fields = ["sub", "ses", "task", "lolo"]
    labels = ["01", "01", "langloc+foo", "lala"]
    file_name = "sub-01_ses-01_task-langloc+foo_lolo-lala_bold.nii.gz"

    file_path = Path("dataset", "sub-01", "ses-01", "func", file_name)

    file_dict = parse_bids_filename(file_path)
    assert file_dict["extension"] == "nii.gz"
    assert file_dict["suffix"] == "bold"
    assert file_dict["file_path"] == file_path
    assert file_dict["file_basename"] == file_name
    entities = {field: labels[fidx] for fidx, field in enumerate(fields)}
    assert file_dict["entities"] == entities


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
    with pytest.warns(RuntimeWarning, match="'RepetitionTime' not found"):
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
    """Check if json files are found in root folder of a dataset.

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


@pytest.mark.parametrize(
    "params, files_per_subject",
    [
        # files in total related to subject images.
        # Top level files like README not included
        ({}, 19),
        # bold files expected. .nii and .json files
        ({"file_tag": "bold"}, 12),
        # files are nii.gz. Bold and T1w files.
        ({"file_type": "nii.gz"}, 7),
        # There are only n_sub files in anat folders. One T1w per subject.
        ({"modality_folder": "anat"}, 1),
        # files corresponding to run 1 of session 2 of main task.
        # n_sub bold.nii.gz and n_sub bold.json files.
        (
            {
                "file_tag": "bold",
                "filters": [("task", "main"), ("run", "01"), ("ses", "02")],
            },
            2,
        ),
    ],
)
def test_get_bids_files(tmp_path, params, files_per_subject):
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

    selection = get_bids_files(bids_path, **params)

    assert len(selection) == files_per_subject * n_sub

    # files correspond to subject 01
    selection = get_bids_files(bids_path, sub_label="01")

    assert len(selection) == 19

    # Get Top level folder files. Only 1 in this case, the README file.
    selection = get_bids_files(bids_path, sub_folder=False)

    assert len(selection) == 1


def test_get_bids_files_fmriprep(tmp_path):
    """Check proper number of files is returned for fmriprep version."""
    n_sub = 2

    bids_path = create_fake_bids_dataset(
        base_dir=tmp_path,
        n_sub=n_sub,
        n_ses=2,
        tasks=["localizer", "main"],
        n_runs=[1, 2],
        confounds_tag="desc-confounds_timeseries",
    )

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

    # counfonds (4 runs per ses & sub), testing `fmriprep` < 20.2 path
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
