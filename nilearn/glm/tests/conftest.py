"""Common fixtures and function for GLM tests."""

import pandas as pd
import pytest

from nilearn._utils.data_gen import generate_fake_fmri_data_and_design
from nilearn.conftest import _shape_3d_default

SHAPE = (*_shape_3d_default(), 1)


@pytest.fixture()
def n_subjects() -> int:
    """Return number of subjects for group level GLM."""
    return 3


@pytest.fixture
def input_df() -> pd.DataFrame:
    """Input DataFrame for testing."""
    return pd.DataFrame(
        {
            "effects_map_path": ["foo.nii", "bar.nii", "baz.nii"],
            "subject_label": ["foo", "bar", "baz"],
        }
    )


def _confounds() -> pd.DataFrame:
    return pd.DataFrame(
        [["01", 1], ["02", 2], ["03", 3]],
        columns=["subject_label", "conf1"],
    )


@pytest.fixture
def confounds() -> pd.DataFrame:
    """Confound DataFrame for testing."""
    return _confounds()


def fake_fmri_data(shape=SHAPE):
    """Return fMRI data and design."""
    shapes = (shape,)
    mask, fmri_data, _ = generate_fake_fmri_data_and_design(shapes)
    return fmri_data[0], mask
