"""Utility functions for testing load_confounds"""
import os
import json
import pandas as pd
from nilearn.interfaces.fmriprep import load_confounds_utils


img_file_patterns = {
    "ica_aroma":
        "_space-MNI152NLin2009cAsym_desc-smoothAROMAnonaggr_bold.nii.gz",
    "regular":
        "_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
    "cifti":
        "_space-fsLR_den-91k_bold.dtseries.nii",
    "gifti": (
        "_space-fsaverage5_hemi-L_bold.func.gii",
        "_space-fsaverage5_hemi-R_bold.func.gii",
    ),
}


def get_testdata_path(non_steady_state=True, fmriprep_version="1.4.x"):
    """Get file path for the confound regressors."""
    derivative = "regressors" if fmriprep_version != "21.x.x" else "timeseries"
    path_data = os.path.join(os.path.dirname(
        load_confounds_utils.__file__), "data")
    suffix = "test-v21" if fmriprep_version == "21.x.x" else "test"
    if non_steady_state:
        return [
            os.path.join(path_data, filename)
            for filename in [
                f"{suffix}_desc-confounds_{derivative}.tsv",
                f"{suffix}_desc-confounds_{derivative}.json",
            ]
        ]
    else:
        return [
            os.path.join(path_data, filename)
            for filename in [
                f"no_nonsteady_desc-confounds_{derivative}.tsv",
                f"test_desc-confounds_{derivative}.json",
            ]
        ]


def create_tmp_filepath(
    base_path,
    image_type="regular",
    suffix="test",
    copy_confounds=False,
    copy_json=False,
    fmriprep_version="1.4.x"
):
    """Create test files in temporary directory."""
    derivative = "regressors" if fmriprep_version == "1.2.x" else "timeseries"
    suffix = "test-v21" if fmriprep_version == "21.x.x" else suffix

    # confound files
    confounds_root = f"_desc-confounds_{derivative}.tsv"
    tmp_conf = base_path / (suffix + confounds_root)

    if copy_confounds:
        conf, meta = get_legal_confound(fmriprep_version=fmriprep_version)
        conf.to_csv(tmp_conf, sep="\t", index=False)
    else:
        tmp_conf.touch()

    if copy_json:
        meta_root = f"_desc-confounds_{derivative}.json"
        tmp_meta = base_path / (suffix + meta_root)
        conf, meta = get_legal_confound(fmriprep_version=fmriprep_version)
        with open(tmp_meta, "w") as file:
            json.dump(meta, file, indent=2)

    # image data
    # convert path object to string as nibabel do strings
    img_root = img_file_patterns[image_type]
    if type(img_root) is str:
        tmp_img = suffix + img_root
        tmp_img = base_path / tmp_img
        tmp_img.touch()
        tmp_img = str(tmp_img)
    else:
        tmp_img = []
        for root in img_root:
            tmp_gii = suffix + root
            tmp_gii = base_path / tmp_gii
            tmp_gii.touch()
            tmp_img.append(str(tmp_gii))
    return tmp_img, tmp_conf


def get_legal_confound(non_steady_state=True, fmriprep_version="1.4.x"):
    """Load the valid confound files for manipulation."""
    conf, meta = get_testdata_path(non_steady_state=non_steady_state,
                                   fmriprep_version=fmriprep_version)
    conf = pd.read_csv(conf, delimiter="\t", encoding="utf-8")
    with open(meta, "r") as file:
        meta = json.load(file)
    return conf, meta
