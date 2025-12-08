"""Fixturs common for all GLM tests."""

import pandas as pd
import pytest

from nilearn._utils.data_gen import generate_fake_fmri_data_and_design
from nilearn.conftest import _shape_3d_default
from nilearn.glm.second_level import SecondLevelModel

SHAPE = (*_shape_3d_default(), 1)


@pytest.fixture()
def n_subjects():
    """Return nb subjects.

    Kept low to minimize run time of group level GLM tests.
    """
    return 3


def _confounds() -> pd.DataFrame:
    return pd.DataFrame(
        [["01", 1], ["02", 2], ["03", 3]],
        columns=["subject_label", "conf1"],
    )


@pytest.fixture
def confounds() -> pd.DataFrame:
    """Return confounds for 3 subjects for group level model."""
    return _confounds()


def fake_fmri_data(shape=SHAPE):
    """Return a single 4D nifti image and its mask."""
    shapes = (shape,)
    mask, fmri_data, _ = generate_fake_fmri_data_and_design(shapes)
    return fmri_data[0], mask


@pytest.fixture
def fitted_slm(n_subjects) -> SecondLevelModel:
    """Return a simple fitted second level model.

    Useful when just testing contrast computation.
    """
    func_img, mask = fake_fmri_data()

    model = SecondLevelModel(mask_img=mask)

    Y = [func_img] * n_subjects
    X = pd.DataFrame([[1]] * n_subjects, columns=["intercept"])
    model = model.fit(Y, design_matrix=X)
    return model
