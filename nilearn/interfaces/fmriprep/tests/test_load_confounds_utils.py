import pytest
from nilearn.interfaces.fmriprep.load_confounds_utils import (
    _sanitize_confounds
)


@pytest.mark.parametrize(
    "inputs,flag",
    [
        (["image.nii.gz"], True),
        ("image.nii.gz", True),
        (["image1.nii.gz", "image2.nii.gz"], False),
        (["image_L.func.gii", "image_R.func.gii"], True),
        ([["image_L.func.gii", "image_R.func.gii"]], True),
        ([["image1_L.func.gii", "image1_R.func.gii"],
          ["image2_L.func.gii", "image2_R.func.gii"]], False),
    ],
)
def test_sanitize_confounds(inputs, flag):
    """Should correctly catch inputs that are a single image."""
    _, singleflag = _sanitize_confounds(inputs)
    assert singleflag is flag
