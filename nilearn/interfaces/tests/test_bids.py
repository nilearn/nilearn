"""Tests for the nilearn.interfaces.bids submodule."""
import os

from nibabel.tmpdirs import InTemporaryDirectory

from nilearn._utils.data_gen import create_fake_bids_dataset
from nilearn.interfaces.bids import get_bids_files, parse_bids_filename


def test_get_bids_files():
    with InTemporaryDirectory():
        bids_path = create_fake_bids_dataset(n_sub=10, n_ses=2,
                                             tasks=['localizer', 'main'],
                                             n_runs=[1, 3])
        # For each possible option of file selection we check that we
        # recover the appropriate amount of files, as included in the
        # fake bids dataset.

        # 250 files in total related to subject images. Top level files like
        # README not included
        selection = get_bids_files(bids_path)
        assert len(selection) == 250
        # 160 bold files expected. .nii and .json files
        selection = get_bids_files(bids_path, file_tag='bold')
        assert len(selection) == 160
        # Only 90 files are nii.gz. Bold and T1w files.
        selection = get_bids_files(bids_path, file_type='nii.gz')
        assert len(selection) == 90
        # Only 25 files correspond to subject 01
        selection = get_bids_files(bids_path, sub_label='01')
        assert len(selection) == 25
        # There are only 10 files in anat folders. One T1w per subject.
        selection = get_bids_files(bids_path, modality_folder='anat')
        assert len(selection) == 10
        # 20 files corresponding to run 1 of session 2 of main task.
        # 10 bold.nii.gz and 10 bold.json files. (10 subjects)
        filters = [('task', 'main'), ('run', '01'), ('ses', '02')]
        selection = get_bids_files(bids_path, file_tag='bold', filters=filters)
        assert len(selection) == 20
        # Get Top level folder files. Only 1 in this case, the README file.
        selection = get_bids_files(bids_path, sub_folder=False)
        assert len(selection) == 1
        # 80 counfonds (4 runs per ses & sub), testing `fmriprep` >= 20.2 path
        selection = get_bids_files(os.path.join(bids_path, 'derivatives'),
                                   file_tag='desc-confounds_timeseries')
        assert len(selection) == 80

    with InTemporaryDirectory():
        bids_path = create_fake_bids_dataset(n_sub=10, n_ses=2,
                                             tasks=['localizer', 'main'],
                                             n_runs=[1, 3],
                                             confounds_tag="desc-confounds_"
                                                           "regressors")
        # 80 counfonds (4 runs per ses & sub), testing `fmriprep` >= 20.2 path
        selection = get_bids_files(os.path.join(bids_path, 'derivatives'),
                                   file_tag='desc-confounds_regressors')
        assert len(selection) == 80


def test_parse_bids_filename():
    fields = ['sub', 'ses', 'task', 'lolo']
    labels = ['01', '01', 'langloc', 'lala']
    file_name = 'sub-01_ses-01_task-langloc_lolo-lala_bold.nii.gz'
    file_path = os.path.join('dataset', 'sub-01', 'ses-01', 'func', file_name)
    file_dict = parse_bids_filename(file_path)
    for fidx, field in enumerate(fields):
        assert file_dict[field] == labels[fidx]
    assert file_dict['file_type'] == 'nii.gz'
    assert file_dict['file_tag'] == 'bold'
    assert file_dict['file_path'] == file_path
    assert file_dict['file_basename'] == file_name
    assert file_dict['file_fields'] == fields
