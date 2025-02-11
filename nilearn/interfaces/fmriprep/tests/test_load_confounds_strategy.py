import re

import pandas as pd
import pytest

from nilearn.interfaces.fmriprep import load_confounds, load_confounds_strategy
from nilearn.interfaces.fmriprep.load_confounds_strategy import (
    preset_strategies,
)
from nilearn.interfaces.fmriprep.tests._testing import create_tmp_filepath


@pytest.mark.parametrize("fmriprep_version", ["1.4.x", "21.x.x"])
@pytest.mark.parametrize(
    "denoise_strategy,image_type",
    [
        ("simple", "regular"),
        ("scrubbing", "regular"),
        ("compcor", "regular"),
        ("ica_aroma", "ica_aroma"),
    ],
)
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_load_confounds_strategy(
    tmp_path, denoise_strategy, image_type, fmriprep_version
):
    """Smoke test with no extra inputs."""
    file_nii, _ = create_tmp_filepath(
        tmp_path,
        image_type=image_type,
        copy_confounds=True,
        copy_json=True,
        fmriprep_version=fmriprep_version,
    )
    confounds, _ = load_confounds_strategy(
        file_nii, denoise_strategy=denoise_strategy
    )
    assert isinstance(confounds, pd.DataFrame)


@pytest.mark.parametrize("fmriprep_version", ["1.4.x", "21.x.x"])
@pytest.mark.parametrize(
    "denoise_strategy,image_type",
    [
        ("simple", "regular"),
        ("scrubbing", "regular"),
        ("compcor", "regular"),
        ("ica_aroma", "ica_aroma"),
    ],
)
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_strategies(tmp_path, denoise_strategy, image_type, fmriprep_version):
    """Check defaults setting of each preset strategy."""
    file_nii, _ = create_tmp_filepath(
        tmp_path,
        image_type=image_type,
        copy_confounds=True,
        copy_json=True,
        fmriprep_version=fmriprep_version,
    )
    confounds, _ = load_confounds_strategy(
        file_nii, denoise_strategy=denoise_strategy
    )
    # Check that all fixed name model categories have been successfully loaded
    list_check = _get_headers(denoise_strategy)
    for col in confounds.columns:
        # Check that all possible names exists
        checker = [
            re.match(keyword, col) is not None for keyword in list_check
        ]
        assert sum(checker) == 1


patterns = {
    "high_pass": ["cosine+"],
    "motion": ["trans_[xyz]$", "rot_[xyz]$"],
    "wm_csf": ["csf$", "white_matter$"],
    "global": ["global_signal$"],
    "global_signal": ["global_signal$"],
    "compcor": ["[tawc]_comp_cor_+"],
}


def _get_headers(denoise_strategy):
    """Retrieve the relevant header pattern for each strategy to check."""
    default_params = preset_strategies[denoise_strategy].copy()

    list_check = []
    noise_components = default_params.pop("strategy")
    for strategy in noise_components:
        if strategy in patterns:
            list_check += patterns[strategy]

    for key, value in default_params.items():
        if key in ["motion", "wm_csf", "global_signal"] and value != "basic":
            for item in patterns[key]:
                list_check.append(item.replace("$", "_+"))
    return list_check


@pytest.mark.parametrize(
    "fmriprep_version, volumes_left, len_sample_mask",
    [("1.4.x", 22, 29), ("21.x.x", 0, 5)],
)
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_strategy_scrubbing(
    tmp_path, fmriprep_version, volumes_left, len_sample_mask
):
    """Check user specified input for scrubbing strategy."""
    file_nii, _ = create_tmp_filepath(
        tmp_path,
        image_type="regular",
        copy_confounds=True,
        copy_json=True,
        fmriprep_version=fmriprep_version,
    )
    confounds, sample_mask = load_confounds_strategy(
        file_nii, denoise_strategy="scrubbing", fd_threshold=0.15
    )
    # For "1.4.x"
    # out of 30 vols, should have 6 motion outliers from scrubbing,
    # and 2 vol removed by scrubbing strategy "full"
    assert len(sample_mask) == volumes_left

    # shape of confound regressors untouched
    assert confounds.shape[0] == 30

    # also load confounds with very liberal scrubbing thresholds
    # this should not produce an error
    confounds, sample_mask = load_confounds_strategy(
        file_nii,
        denoise_strategy="scrubbing",
        fd_threshold=1,
        std_dvars_threshold=5,
    )
    # only non-steady volumes removed
    assert len(sample_mask) == len_sample_mask

    # maker sure global signal works
    confounds, sample_mask = load_confounds_strategy(
        file_nii, denoise_strategy="scrubbing", global_signal="full"
    )
    for check in [
        "global_signal",
        "global_signal_derivative1",
        "global_signal_power2",
        "global_signal_derivative1_power2",
    ]:
        assert check in confounds.columns


@pytest.mark.parametrize("fmriprep_version", ["1.4.x", "21.x.x"])
def test_strategy_compcor(tmp_path, fmriprep_version):
    """Check user specified input for compcor strategy."""
    file_nii, _ = create_tmp_filepath(
        tmp_path,
        image_type="regular",
        copy_confounds=True,
        copy_json=True,
        fmriprep_version=fmriprep_version,
    )
    confounds, _ = load_confounds_strategy(
        file_nii, denoise_strategy="compcor"
    )
    compcor_col_str_anat = "".join(confounds.columns)
    assert "t_comp_cor_" not in compcor_col_str_anat
    # this one comes from the white matter mask
    expected = (
        "a_comp_cor_57" if fmriprep_version == "1.4.x" else "w_comp_cor_00"
    )
    assert expected not in compcor_col_str_anat
    assert "global_signal" not in compcor_col_str_anat

    # test the global signal option
    confounds, _ = load_confounds_strategy(
        file_nii, denoise_strategy="compcor", global_signal="basic"
    )
    compcor_col_str_anat = "".join(confounds.columns)
    assert "global_signal" in compcor_col_str_anat


@pytest.mark.parametrize("fmriprep_version", ["1.4.x", "21.x.x"])
def test_irrelevant_input(tmp_path, fmriprep_version):
    """Check invalid input raising correct warning or error message."""
    file_nii, _ = create_tmp_filepath(
        tmp_path,
        image_type="regular",
        copy_confounds=True,
        copy_json=True,
        fmriprep_version=fmriprep_version,
    )
    warning_message = (
        r"parameters accepted: \['motion', 'wm_csf', "
        "'global_signal', 'demean']"
    )
    with pytest.warns(UserWarning, match=warning_message):
        load_confounds_strategy(
            file_nii, denoise_strategy="simple", ica_aroma="full"
        )
    # invalid strategy
    with pytest.raises(KeyError, match="blah"):
        load_confounds_strategy(file_nii, denoise_strategy="blah")


def test_empty_strategy(tmp_path):
    """Ensure to return None for confounds and raise a warning
    when strategy is empty.
    """
    file_nii, _ = create_tmp_filepath(
        tmp_path,
        image_type="regular",
        copy_confounds=True,
        copy_json=True,
    )

    warning_message = "strategy is empty, confounds will return None."
    with pytest.warns(UserWarning, match=warning_message):
        confounds, sample_mask = load_confounds(file_nii, strategy=[])

    assert confounds is None
