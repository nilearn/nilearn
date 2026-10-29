"""Common fixtures and function for GLM tests."""

import pandas as pd
import pytest

from nilearn._utils.data_gen import generate_fake_fmri_data_and_design
from nilearn.conftest import _shape_3d_default
from nilearn.image import new_img_like

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


def fake_subjects_fmri_data(n_subjects, shape=SHAPE):
    """Return a list of distinct fMRI images (one per subject) and a mask.

    Using the same image for all subjects gives zero variance
    across subjects and leads to degenerate statistics.
    """
    shapes = [shape] * n_subjects
    mask, fmri_data, _ = generate_fake_fmri_data_and_design(shapes)
    return fmri_data, mask


@pytest.fixture
def surf_imgs_with_noise(surf_img_1d, n_subjects, rng):
    """Return a list of surface images (one per subject) with added noise.

    Using the same image for all subjects gives zero variance
    across subjects and leads to degenerate statistics.
    """
    return [
        new_img_like(
            surf_img_1d,
            {
                hemi: data + rng.standard_normal(data.shape)
                for hemi, data in surf_img_1d.data.parts.items()
            },
        )
        for _ in range(n_subjects)
    ]
