"""Test the first level model on BIDS datasets."""
import os
import shutil

from pathlib import Path

import pytest

from nilearn._utils.data_gen import (create_fake_bids_dataset)
from nilearn.glm.first_level import (first_level_from_bids)
from nilearn.interfaces.bids import get_bids_files
from nibabel.tmpdirs import InTemporaryDirectory

def test_first_level_from_bids_bug_3029():
    "Test error when events.tsv is missing for a bold file."
    with InTemporaryDirectory():

        bids_path = create_fake_bids_dataset(n_sub=2,
                                            n_ses=2,
                                            tasks=['main'],
                                            n_runs=[2])
        files_to_rename = Path(bids_path).joinpath('derivatives').glob(
            '**/func/*_task-main_*desc-*')
        for file in files_to_rename:
            new_file = file.parent / file.name.replace('run-0', 'run-')
            file.rename(new_file)

        with pytest.raises(ValueError, 
                           match=".*events.tsv files.*"):
            first_level_from_bids(
                bids_path,
                task_label='main',
                space_label='MNI',
                img_filters=[('desc', 'preproc')],
                verbose=1)

def test_first_level_from_bids():
    with InTemporaryDirectory():

        bids_path = create_fake_bids_dataset(n_sub=3,
                                             n_ses=2,
                                             tasks=['localizer', 'main'],
                                             n_runs=[1, 3])
        # test output is as expected
        models, m_imgs, m_events, m_confounds = first_level_from_bids(
            bids_path,
            task_label='main',
            space_label='MNI',
            img_filters=[('desc', 'preproc')])
        assert len(models) == len(m_imgs)
        assert len(models) == len(m_events)
        assert len(models) == len(m_confounds)


@pytest.mark.parametrize('entity', ['acq',
                                    'ce',
                                    'dir',
                                    'rec',
                                    'echo',
                                    'res',
                                    'den'])
def test_first_level_from_bids_bug_3524(entity):
    "Test right files are selected when entities have several labels."
    with InTemporaryDirectory():

        bids_path = create_fake_bids_dataset(n_sub=2,
                                             n_ses=2,
                                             tasks=['main'],
                                             n_runs=[3],
                                             entities=[entity, ['A', 'B']])

        models, m_imgs, m_events, m_confounds = first_level_from_bids(
            dataset_path=bids_path,
            task_label='main',
            space_label='MNI',
            img_filters=[('desc', 'preproc'), (entity, 'A')])
        assert len(models) == len(m_imgs)
        assert len(models) == len(m_events)
        assert len(models) == len(m_confounds)    


def test_first_level_from_bids_validation_input():
    with InTemporaryDirectory():
        bids_path = create_fake_bids_dataset(n_sub=3,
                                             n_ses=2,
                                             tasks=['localizer', 'main'],
                                             n_runs=[1, 3])
        # test arguments are provided correctly
        with pytest.raises(TypeError):
            first_level_from_bids(2,
                                  task_label='main',
                                  space_label='MNI')
        with pytest.raises(ValueError):
            first_level_from_bids('lolo',
                                  task_label='main',
                                  space_label='MNI')
        with pytest.raises(TypeError):
            first_level_from_bids(bids_path,
                                  task_label=2,
                                  space_label='MNI')
        with pytest.raises(TypeError):
            first_level_from_bids(bids_path,
                                  task_label='main',
                                  space_label='MNI',
                                  model_init=[])

        with pytest.raises(TypeError,
                           match="space_label must be a string"):
            first_level_from_bids(bids_path,
                                  task_label='main',
                                  space_label=42)

        with pytest.raises(TypeError,
                           match="img_filters must be a list"):
            first_level_from_bids(bids_path,
                                  task_label='main',
                                  img_filters="foo")
        with pytest.raises(TypeError,
                           match="filters in img"):
            first_level_from_bids(bids_path,
                                  task_label='main',
                                  img_filters=[(1, 2)])
        with pytest.raises(ValueError,
                           match="field foo is not a possible filter."):
            first_level_from_bids(bids_path,
                                  task_label='main',
                                  img_filters=[("foo", "bar")])


def test_first_level_from_bids_with_missing_files():
    with InTemporaryDirectory():
        bids_path = create_fake_bids_dataset(n_sub=3,
                                             n_ses=2,
                                             tasks=['localizer', 'main'],
                                             n_runs=[1, 3])
        # test repeated run tag error when run tag is in filenames
        # can arise when desc or space is present and not specified
        #
        # desc not specified
        with pytest.raises(ValueError):
            first_level_from_bids(bids_path,
                                  task_label='main',
                                  space_label='T1w')
        # test more than one ses file error when run tag is not in filenames
        # can arise when desc or space is present and not specified
        #
        # desc not specified
        with pytest.raises(ValueError):
            first_level_from_bids(bids_path,
                                  task_label='localizer',
                                  space_label='T1w')

        # test issues with confound files. There should be only one confound
        # file per img. An one per image or None. Case when one is missing
        confound_files = get_bids_files(os.path.join(bids_path, 'derivatives'),
                                        file_tag='desc-confounds_timeseries')
        os.remove(confound_files[-1])
        with pytest.raises(ValueError):
            first_level_from_bids(bids_path,
                                  task_label='main',
                                  space_label='MNI')
        # test issues with event files
        events_files = get_bids_files(bids_path, file_tag='events')
        os.remove(events_files[0])
        # one file missing
        with pytest.raises(ValueError):
            first_level_from_bids(bids_path,
                                  task_label='main',
                                  space_label='MNI')
        for f in events_files[1:]:
            os.remove(f)
        # all files missing
        with pytest.raises(ValueError):
            first_level_from_bids(bids_path,
                                  task_label='main',
                                  space_label='MNI')

        # In case different desc and spaces exist and are not selected we
        # fail and ask for more specific information
        shutil.rmtree(os.path.join(bids_path, 'derivatives'))
        # issue if no derivatives folder is present
        with pytest.raises(ValueError):
            first_level_from_bids(bids_path,
                                  task_label='main',
                                  space_label='MNI')


def test_first_level_from_bids_no_session():
    """Check runs are not repeated when ses field is not used."""
    with InTemporaryDirectory():
        bids_path = create_fake_bids_dataset(n_sub=3,
                                             n_ses=1,
                                             tasks=['localizer', 'main'],
                                             n_runs=[1, 3],
                                             no_session=True)
        # test repeated run tag error when run tag is in filenames and not ses
        # can arise when desc or space is present and not specified
        with pytest.raises(ValueError):
            first_level_from_bids(
                bids_path,
                task_label='main',
                space_label='T1w')
