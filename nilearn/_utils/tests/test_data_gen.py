"""Test for data generation utilities."""

import json

from nilearn._utils.data_gen import add_metadata_to_bids_derivatives


def test_add_metadata_to_bids_derivatives(tmp_path):
    # bare bone smoke test
    target_dir = tmp_path / 'derivatives' / 'sub-01' / 'ses-01' / 'func'
    target_dir.mkdir(parents=True)
    json_file = add_metadata_to_bids_derivatives(bids_path=tmp_path,
                                                  metadata={"foo": "bar"})
    assert json_file.exists()
    assert json_file.name == 'sub-01_ses-01_task-main_run-01_bold.json'
    with open(json_file, 'r') as f:
        metadata = json.load(f)
        assert metadata == {"foo": "bar"}