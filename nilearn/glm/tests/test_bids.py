import pytest

from nilearn._utils.data_gen import (create_fake_bids_dataset)
from nilearn.glm.first_level import (first_level_from_bids)
from nibabel.tmpdirs import InTemporaryDirectory


@pytest.mark.parametrize('entity', ['acq',
                                    'ce',
                                    'dir',
                                    'rec',
                                    'echo',
                                    'res',
                                    'den'])
def test_first_level_from_bids_bug_3524(entity):
    with InTemporaryDirectory():

        bids_path = create_fake_bids_dataset(n_sub=10,
                                             n_ses=2,
                                             tasks=['localizer', 'main'],
                                             n_runs=[1, 3],
                                             entities=[entity, ['A', 'B']])

        first_level_from_bids(dataset_path=bids_path,
                              task_label='main',
                              space_label='MNI',
                              img_filters=[('desc', 'preproc'), (entity, 'A')])
