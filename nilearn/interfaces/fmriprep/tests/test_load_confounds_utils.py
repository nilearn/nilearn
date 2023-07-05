import pytest

from nilearn.interfaces.fmriprep.load_confounds_utils import (
    _get_file_name,
    _sanitize_confounds,
)
from nilearn.interfaces.fmriprep.tests.utils import create_tmp_filepath


@pytest.mark.parametrize(
    "inputs,flag",
    [
        (["image.nii.gz"], True),
        ("image.nii.gz", True),
        (["image1.nii.gz", "image2.nii.gz"], False),
        (["image_L.func.gii", "image_R.func.gii"], True),
        ([["image_L.func.gii", "image_R.func.gii"]], True),
        (
            [
                ["image1_L.func.gii", "image1_R.func.gii"],
                ["image2_L.func.gii", "image2_R.func.gii"],
            ],
            False,
        ),
    ],
)
def test_sanitize_confounds(inputs, flag):
    """Should correctly catch inputs that are a single image."""
    _, singleflag = _sanitize_confounds(inputs)
    assert singleflag is flag


@pytest.mark.parametrize("flag", [True, False])
@pytest.mark.parametrize(
    "image_type", ["regular", "native", "res", "cifti", "den", "part", "gifti"]
)
def test_get_file_name(tmp_path, flag, image_type):
    """Test _get_file_name."""
    if flag:
        suffix = "_desc-confounds_regressors"
    else:
        suffix = "_desc-confounds_timeseries"

    if image_type == "part":
        kwargs = {"suffix": "sub-test01_task-test_part-mag_run-01"}
    else:
        kwargs = {}

    img, _ = create_tmp_filepath(
        tmp_path,
        image_type=image_type,
        old_derivative_suffix=flag,
        **kwargs,
    )

    conf = _get_file_name(img)
    assert suffix in conf
