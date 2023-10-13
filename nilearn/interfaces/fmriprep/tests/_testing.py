"""Utility functions for testing load_confounds"""
import json
import os

import pandas as pd

from nilearn._utils.data_gen import _create_bids_filename
from nilearn.interfaces.bids._utils import _bids_entities
from nilearn.interfaces.fmriprep import load_confounds_utils

img_file_patterns = {
    "ica_aroma":
        {"entities": {"space": "MNI152NLin2009cAsym",
                      "desc": "smoothAROMAnonaggr"}},
    "regular":
        {"entities": {"space": "MNI152NLin2009cAsym",
                      "desc": "preproc"}},
    "res":
        {"entities": {"space": "MNI152NLin2009cAsym",
                      "res": "2",
                      "desc": "preproc"}},
    "native":
        {"entities": {"desc": "preproc"}},
    "cifti":
        {"entities": {"desc": "preproc",
                      "space": "fsLR",
                      "den": "91k"},
         "extension": "dtseries.nii"},
    "den":
        {"entities": {"space": "fsLR",
                      "den": "32k",
                      "desc": "preproc"}},
    "part":
        {"entities": {"part": "mag",
                      "space": "MNI152NLin2009cAsym",
                      "desc": "preproc",
                      }},
    "gifti": (
        {"entities": {"hemi": "L",
                      "space": "fsaverage5"},
         "extension": "func.gii"},
        {"entities": {"hemi": "R",
                      "space": "fsaverage5"},
         "extension": "func.gii"})
}


def get_testdata_path(non_steady_state=True):
    """Get file path for the confound regressors."""
    derivative = "regressors"
    path_data = os.path.join(os.path.dirname(
        load_confounds_utils.__file__), "data")
    if non_steady_state:
        return [
            os.path.join(path_data, filename)
            for filename in [
                f"test_desc-confounds_{derivative}.tsv",
                f"test_desc-confounds_{derivative}.json",
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
    bids_fields=None,
    copy_confounds=False,
    copy_json=False,
    old_derivative_suffix=False
):
    entities_to_include = [
        *_bids_entities()["raw"],
        *_bids_entities()["derivatives"]
    ]
    if bids_fields is None:
        bids_fields = {"entities": {"sub": "test01",
                                    "task": "test"}}

    """Create test files in temporary directory."""
    derivative = "regressors" if old_derivative_suffix else "timeseries"

    # confound files
    bids_fields["entities"]["desc"] = "confounds"
    bids_fields["suffix"] = derivative
    bids_fields["extension"] = "tsv"
    confounds_filename = _create_bids_filename(
        fields=bids_fields,
        entities_to_include=entities_to_include
    )
    tmp_conf = base_path / confounds_filename

    if copy_confounds:
        conf, meta = get_legal_confound()
        conf.to_csv(tmp_conf, sep="\t", index=False)
    else:
        tmp_conf.touch()

    if copy_json:
        bids_fields["extension"] = "json"
        confounds_sidecar = _create_bids_filename(
            fields=bids_fields,
            entities_to_include=entities_to_include
        )
        tmp_meta = base_path / confounds_sidecar
        conf, meta = get_legal_confound()
        with open(tmp_meta, "w") as file:
            json.dump(meta, file, indent=2)

    # image data
    # convert path object to string as nibabel do strings
    img_file_patterns_type = img_file_patterns[image_type]
    if type(img_file_patterns_type) is dict:
        bids_fields = update_bids_fields(bids_fields, img_file_patterns_type)
        tmp_img = _create_bids_filename(
            fields=bids_fields,
            entities_to_include=entities_to_include
        )
        tmp_img = base_path / tmp_img
        tmp_img.touch()
        tmp_img = str(tmp_img)
    else:
        tmp_img = []
        for root in img_file_patterns_type:
            bids_fields = update_bids_fields(bids_fields, root)
            tmp_gii = _create_bids_filename(
                fields=bids_fields,
                entities_to_include=entities_to_include
            )
            tmp_gii = base_path / tmp_gii
            tmp_gii.touch()
            tmp_img.append(str(tmp_gii))
    return tmp_img, tmp_conf


def get_legal_confound(non_steady_state=True):
    """Load the valid confound files for manipulation."""
    conf, meta = get_testdata_path(non_steady_state=non_steady_state)
    conf = pd.read_csv(conf, delimiter="\t", encoding="utf-8")
    with open(meta) as file:
        meta = json.load(file)
    return conf, meta


def update_bids_fields(bids_fields, img_file_patterns_type):
    """Update the bids_fields dictionary with the img_file_patterns_type."""
    if "extension" not in img_file_patterns_type:
        bids_fields["extension"] = "nii.gz"
    if "suffix" not in img_file_patterns_type:
        bids_fields["suffix"] = "bold"
    for key in img_file_patterns_type:
        if key == "entities":
            for entity in img_file_patterns_type["entities"]:
                bids_fields["entities"][entity] = \
                    img_file_patterns_type["entities"][entity]
        else:
            bids_fields[key] = img_file_patterns_type[key]
    return bids_fields
