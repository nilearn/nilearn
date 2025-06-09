import pandas as pd
import pytest

from nilearn.interfaces.fmriprep.load_confounds_utils import (
    _get_file_name,
    load_confounds_file_as_dataframe,
    sanitize_confounds,
)
from nilearn.interfaces.fmriprep.tests._testing import create_tmp_filepath


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
    _, singleflag = sanitize_confounds(inputs)
    assert singleflag is flag


@pytest.mark.parametrize(
    "flag,keyword",
    [
        ("1.2.x", "_desc-confounds_regressors"),
        ("1.4.x", "_desc-confounds_timeseries"),
        ("21.x.x", "21xx"),
    ],
)
@pytest.mark.parametrize(
    "image_type", ["regular", "native", "res", "cifti", "den", "part", "gifti"]
)
def test_get_file_name(tmp_path, flag, keyword, image_type):
    """Test _get_file_name."""
    if image_type == "part":
        kwargs = {
            "bids_fields": {
                "entities": {
                    "sub": flag.replace(".", ""),
                    "task": "test",
                    "part": "mag",
                    "run": "01",
                }
            }
        }
    else:
        kwargs = {}

    img, _ = create_tmp_filepath(
        tmp_path,
        image_type=image_type,
        fmriprep_version=flag,
        **kwargs,
    )

    conf = _get_file_name(img)
    assert keyword in conf


def test_get_file_name_raises_on_invalid_tedana_confounds(tmp_path):
    """
    Test that _get_file_name raises
    when TEDANA confound file count is not 2.
    """
    func_file = tmp_path / "sub-01_task-rest_desc-optcom_bold.nii.gz"
    func_file.write_text("dummy")

    # TEDANA expects exactly these two suffixes, so we can test with one
    for suffix in ["_mixing.tsv"]:
        (tmp_path / f"sub-01_task-rest_desc-ICA{suffix}").write_text("dummy")

    with pytest.raises(ValueError, match="expected 2 for TEDANA"):
        _get_file_name(str(func_file), flag_tedana=True)


def test_load_confounds_file_as_dataframe_tedana_invalid_columns(tmp_path):
    """
    Test that loading TEDANA confounds raises ValueError
    when the required columns are not present in the confound file.
    """
    # create a fake confound file with invalid TEDANA headers
    fake_confounds = pd.DataFrame(
        {
            "junk_col1": [0.1, 0.2],
            "junk_col2": [1, 0],
        }
    )
    conf_file = tmp_path / "fake_tedana.tsv"
    fake_confounds.to_csv(conf_file, sep="\t", index=False)

    # this should raise a ValueError because required TEDANA
    # headers are missing
    with pytest.raises(
        ValueError,
        match="The confound file does not contain the expected columns",
    ):
        load_confounds_file_as_dataframe(str(conf_file), flag_tedana=True)
