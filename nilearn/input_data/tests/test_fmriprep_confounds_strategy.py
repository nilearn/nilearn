import re
import pytest
import pandas as pd
from ..fmriprep_confounds_strategy import fmriprep_confounds_strategy
from .utils import create_tmp_filepath


@pytest.mark.parametrize("denoise_strategy,image_type",
                         [("simple", "regular"),
                          ("scrubbing", "regular"),
                          ("compcor", "regular"),
                          ("ica_aroma", "icaaroma")])
def test_fmriprep_confounds_strategy(tmp_path, denoise_strategy, image_type):
    """Smoke test with no extra inputs."""
    file_nii, _ = create_tmp_filepath(tmp_path, image_type=image_type,
                                      copy_confounds=True, copy_json=True)
    confounds, _ = fmriprep_confounds_strategy(
        file_nii, denoise_strategy=denoise_strategy)
    assert isinstance(confounds, pd.DataFrame)


def test_strategy_simple(tmp_path):
    """Check user specified input for simple strategy."""
    file_nii, _ = create_tmp_filepath(tmp_path, image_type="regular",
                                      copy_confounds=True, copy_json=True)
    confounds, _ = fmriprep_confounds_strategy(
        file_nii, denoise_strategy="simple",
        motion="full", global_signal="basic")

    # Check that all fixed name model categories have been successfully loaded
    list_check = [
        "trans_[xyz]$",
        "rot_[xyz]$",
        "trans_[xyz]_+",
        "rot_[xyz]_+",
        "csf$",
        "white_matter$",
        "csf_+",
        "white_matter_+",
        "global_signal$",
        "cosine+"
    ]
    for col in confounds.columns:
        # Check that all possible names exists
        checker = [re.match(keyword, col) is not None
                   for keyword in list_check]
        assert sum(checker) == 1


def test_strategy_scrubbing(tmp_path):
    """Check user specified input for scrubbing strategy."""
    file_nii, _ = create_tmp_filepath(tmp_path, image_type="regular",
                                      copy_confounds=True, copy_json=True)
    confounds, sample_mask = fmriprep_confounds_strategy(
        file_nii, denoise_strategy="scrubbing", fd_thresh=0.15)
    # Check that all fixed name model categories have been successfully loaded
    list_check = [
        "trans_[xyz]$",
        "rot_[xyz]$",
        "trans_[xyz]_+",
        "rot_[xyz]_+",
        "csf$",
        "white_matter$",
        "csf_+",
        "white_matter_+",
        "global_signal$",
        "cosine+"
    ]
    for col in confounds.columns:
        # Check that all possible names exists
        checker = [re.match(keyword, col) is not None
                   for keyword in list_check]
        assert sum(checker) == 1

    # out of 30 vols, should have 6 motion outliers from scrubbing,
    # and 2 vol removed by srubbing strategy "full"
    assert len(sample_mask) == 22
    # shape of confound regressors untouched
    assert confounds.shape[0] == 30
    # also load confounds with very liberal scrubbing thresholds
    # this should not produce an error
    confounds, sample_mask = fmriprep_confounds_strategy(
        file_nii, denoise_strategy="scrubbing",
        fd_thresh=1, std_dvars_thresh=5)
    assert len(sample_mask) == 29  # only non-steady volumes removed

    # maker sure global signal works
    confounds, sample_mask = fmriprep_confounds_strategy(
        file_nii, denoise_strategy="scrubbing", global_signal="full")
    for check in ["global_signal",
                  "global_signal_derivative1",
                  "global_signal_power2",
                  "global_signal_derivative1_power2",
                  ]:
        assert check in confounds.columns


def test_strategy_compcor(tmp_path):
    """Check user specified input for compcor strategy."""
    file_nii, _ = create_tmp_filepath(tmp_path, image_type="regular",
                                      copy_confounds=True, copy_json=True)
    confounds, _ = fmriprep_confounds_strategy(
        file_nii, denoise_strategy="compcor")
    list_check = [
        "trans_[xyz]$",
        "rot_[xyz]$",
        "trans_[xyz]_+",
        "rot_[xyz]_+",
        "cosine+",
        "a_comp_cor_+",
    ]
    for col in confounds.columns:
        # Check that all possible names exists
        checker = [re.match(keyword, col) is not None
                   for keyword in list_check]
        assert sum(checker) == 1
    compcor_col_str_anat = "".join(confounds.columns)
    assert "t_comp_cor_" not in compcor_col_str_anat
    assert (
        "a_comp_cor_57" not in compcor_col_str_anat
    )  # this one comes from the white matter mask


def test_strategy_ica_aroma(tmp_path):
    """Check user specified input for ica_aroma strategy."""
    file_nii, _ = create_tmp_filepath(tmp_path, image_type="icaaroma",
                                      copy_confounds=True, copy_json=True)
    confounds, _ = fmriprep_confounds_strategy(
        file_nii, denoise_strategy="ica_aroma")

    # Check that all fixed name model categories have been successfully loaded
    list_check = [
        "csf$",
        "white_matter$",
        "csf_+",
        "white_matter_+",
        "cosine+"
    ]
    for col in confounds.columns:
        # Check that all possible names exists
        checker = [re.match(keyword, col) is not None
                   for keyword in list_check]
        assert sum(checker) == 1


def test_irrelevant_input(tmp_path):
    """Check invalid input raising correct warning or error message."""
    file_nii, _ = create_tmp_filepath(tmp_path, image_type="regular",
                                      copy_confounds=True, copy_json=True)
    warning_message = (r"parameters accepted: \['motion', 'wm_csf', "
                       "'global_signal', 'demean']")
    with pytest.warns(UserWarning, match=warning_message):
        fmriprep_confounds_strategy(
            file_nii, denoise_strategy="simple", ica_aroma="full")

    # invalid strategy
    with pytest.raises(KeyError, match="blah"):
        fmriprep_confounds_strategy(
            file_nii, denoise_strategy="blah")
