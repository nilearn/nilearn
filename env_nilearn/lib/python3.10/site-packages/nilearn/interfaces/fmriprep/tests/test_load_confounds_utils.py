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
    "flag,keyword,tedana_keyword",
    [
        ("1.2.x", "_desc-confounds_regressors", "_desc-ICA_mixing"),
        ("1.4.x", "_desc-confounds_timeseries", "_desc-tedana_metrics"),
        ("21.x.x", "21xx", "missing"),
        ("21.x.x", "21xx", "invalid_columns"),
    ],
)
@pytest.mark.parametrize(
    "image_type",
    ["regular", "native", "res", "cifti", "den", "part", "gifti", "tedana"],
)
def test_get_file_name(tmp_path, flag, keyword, tedana_keyword, image_type):
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
    # If image_type is "tedana", we expect the keyword to be different
    flag_tedana = False
    if image_type == "tedana":
        flag_tedana = True
        keyword = tedana_keyword

    # if flag_tedana and tedana_keyword is missing we test raise error
    if flag_tedana and tedana_keyword == "missing":
        (
            tmp_path
            / f"sub-{flag.replace('.', '')}_task-test_desc-ICA_mixing.tsv"
        ).unlink(missing_ok=True)
        with pytest.raises(ValueError, match="expected 2 for TEDANA"):
            _get_file_name(img, flag_tedana=flag_tedana)
        return

    # if flag_tedana and tedana_keyword is invalid_columns we test
    # raise error
    if flag_tedana and tedana_keyword == "invalid_columns":
        # create a fake confound file with invalid TEDANA headers
        fake_confounds = pd.DataFrame(
            {
                "junk_col1": [0.1, 0.2],
                "junk_col2": [1, 0],
            }
        )
        fake_confounds2 = pd.DataFrame(
            {
                "junk_col1": [0.1, 0.2],
                "junk_col2": [1, 0],
            }
        )
        conf_file = (
            tmp_path / f"sub-{flag.replace('.', '')}"
            "_task-test_desc-ICA_mixing.tsv"
        )
        conf_file2 = (
            tmp_path / f"sub-{flag.replace('.', '')}"
            "_task-test_desc-tedana_metrics.tsv"
        )
        fake_confounds.to_csv(
            conf_file,
            sep="\t",
            index=False,
        )
        fake_confounds2.to_csv(
            conf_file2,
            sep="\t",
            index=False,
        )
        print(conf_file)
        print(conf_file2)
        with pytest.raises(
            ValueError,
            match="The confound file does not contain the expected columns",
        ):
            load_confounds_file_as_dataframe(
                [str(conf_file), str(conf_file2)], flag_tedana=True
            )
        return

    conf = _get_file_name(img, flag_tedana=flag_tedana)

    if isinstance(conf, list):
        assert any(keyword in item for item in conf)
    else:
        assert keyword in conf
