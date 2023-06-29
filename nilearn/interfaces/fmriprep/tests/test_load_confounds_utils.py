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
        ([["image1_L.func.gii", "image1_R.func.gii"],
          ["image2_L.func.gii", "image2_R.func.gii"]], False),
    ],
)
def test_sanitize_confounds(inputs, flag):
    """Should correctly catch inputs that are a single image."""
    _, singleflag = _sanitize_confounds(inputs)
    assert singleflag is flag


@pytest.mark.parametrize(
    "flag,suffix,image_type",
    [
        (True, "_desc-confounds_regressors", "regular"),
        (False, "_desc-confounds_timeseries", "regular"),
        (True, "_desc-confounds_regressors", "native"),
        (False, "_desc-confounds_timeseries", "native"),
        (True, "_desc-confounds_regressors", "res"),
        (False, "_desc-confounds_timeseries", "res"),
        (True, "_desc-confounds_regressors", "cifti"),
        (False, "_desc-confounds_timeseries", "cifti"),
        (True, "_desc-confounds_regressors", "den"),
        (False, "_desc-confounds_timeseries", "den"),
        (True, "_desc-confounds_regressors", "gifti"),
        (False, "_desc-confounds_timeseries", "gifti"),
    ],
)
def test_get_file_name(tmp_path, flag, suffix, image_type):
    img, _ = create_tmp_filepath(
        tmp_path,
        image_type=image_type,
        old_derivative_suffix=flag,
    )
    conf = _get_file_name(img)
    assert suffix in conf
