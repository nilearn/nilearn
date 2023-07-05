import pytest
import pandas as pd
from nilearn.interfaces.fmriprep import load_confounds_components as components
from nilearn.interfaces.fmriprep.load_confounds_compcor import _find_compcor
from nilearn.interfaces.fmriprep.load_confounds_utils import (
    _load_confounds_json,
)
from nilearn.interfaces.fmriprep import load_confounds
from nilearn.interfaces.fmriprep.tests.utils import (
    create_tmp_filepath
)


@pytest.mark.parametrize("component",
                         ["motion", "wm_csf",
                         "global_signal", "compcor", "ica_aroma"]
                         )
def test_missing_keywords(tmp_path, component):
    img, bad_conf = create_tmp_filepath(
        tmp_path, copy_confounds=True, copy_json=True
    )

    legal_confounds = pd.read_csv(bad_conf, delimiter="\t", encoding="utf-8")
    if component in ["motion", "wm_csf", "global_signal"]:
        remove_columns = getattr(components, f"_load_{component}")(legal_confounds, "full")
    elif component == "compcor":
        meta_json = bad_conf.parent / bad_conf.name.replace("tsv", "json")
        meta_json = _load_confounds_json(
            meta_json, flag_acompcor=True
        )
        remove_columns = _find_compcor(meta_json, compcor="anat_combined", n_compcor=6)
    elif component == "ica_aroma":
        legal_confounds = pd.read_csv(bad_conf, delimiter="\t", encoding="utf-8")
        remove_columns = getattr(components, f"_load_{component}")(legal_confounds, "basic")
    else:
        remove_columns = getattr(components, f"_load_{component}")(legal_confounds)

    confounds_raw = legal_confounds.drop(
        columns=remove_columns
    )
    confounds_raw.to_csv(bad_conf, sep="\t", index=False)

    # unit test
    with pytest.raises(TypeError) as exc_info:
        getattr(components, f"_load_{component}")(
            confounds_raw
        )
    assert component in exc_info.value.args[0]

    # higher level wrapper should behave the same
    if component != "ica_aroma":
        with pytest.raises(ValueError) as exc_info:
            load_confounds(img, strategy=[component])
        assert component in exc_info.value.args[0]
    else:
        with pytest.raises(ValueError) as exc_info:
            load_confounds(img, strategy=[component], ica_aroma="basic")
        assert component in exc_info.value.args[0]
