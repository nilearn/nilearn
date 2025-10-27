import numpy as np
import pytest

from nilearn._utils.data_gen import generate_fake_fmri_data_and_design
from nilearn.glm.first_level import FirstLevelModel
from nilearn.interfaces.bids import save_glm_to_bids


@pytest.mark.slow
def test_deprecation_save_glm_to_bids(tmp_path):
    """Check deprecation about moved function."""
    shapes, rk = [(7, 8, 9, 15)], 3
    _, fmri_data, design_matrices = generate_fake_fmri_data_and_design(
        shapes,
        rk,
    )

    single_run_model = FirstLevelModel(
        mask_img=None,
        minimize_memory=False,
    ).fit(fmri_data[0], design_matrices=design_matrices[0])

    contrasts = {"effects of interest": np.eye(rk)}
    contrast_types = {"effects of interest": "F"}

    with pytest.warns(
        FutureWarning, match="Please import from 'nilearn.glm' instead"
    ):
        save_glm_to_bids(
            model=single_run_model,
            contrasts=contrasts,
            contrast_types=contrast_types,
            out_dir=tmp_path,
        )
