import pandas as pd
import pytest

from nilearn.interfaces.fmriprep import load_confounds
from nilearn.interfaces.fmriprep.load_confounds import _load_noise_component
from nilearn.interfaces.fmriprep.load_confounds_utils import (
    load_confounds_json,
)
from nilearn.interfaces.fmriprep.tests._testing import create_tmp_filepath


@pytest.fixture
def expected_parameters(strategy_keywords):
    expectation = {
        "compcor": {"compcor": "anat_combined", "n_compcor": 6},
        "ica_aroma": {"ica_aroma": "basic"},
    }
    if strategy_keywords in expectation:
        return expectation[strategy_keywords]
    elif strategy_keywords != "high_pass":
        return {strategy_keywords: "full"}
    else:
        return False


@pytest.mark.parametrize("fmriprep_version", ["1.4.x", "21.x.x"])
@pytest.mark.parametrize(
    "strategy_keywords",
    ["motion", "wm_csf", "global_signal", "compcor", "ica_aroma", "high_pass"],
)
def test_missing_keywords(
    tmp_path, strategy_keywords, expected_parameters, fmriprep_version
):
    """Check the strategy keywords are raising errors correctly in low \
       and high level functions with the exception of `high_pass`.
    """
    img, bad_conf = create_tmp_filepath(
        tmp_path,
        copy_confounds=True,
        copy_json=True,
        fmriprep_version=fmriprep_version,
    )
    legal_confounds = pd.read_csv(bad_conf, delimiter="\t", encoding="utf-8")
    meta_json = bad_conf.parent / bad_conf.name.replace("tsv", "json")
    meta_json = load_confounds_json(meta_json, flag_acompcor=True)

    # remove confound strategy keywords in the test data
    if expected_parameters:
        remove_columns, missing = _load_noise_component(
            legal_confounds,
            component=strategy_keywords,
            missing={"confounds": [], "keywords": []},
            meta_json=meta_json,
            **expected_parameters,
        )
    else:
        remove_columns, missing = _load_noise_component(
            legal_confounds,
            component=strategy_keywords,
            missing={"confounds": [], "keywords": []},
        )
    confounds_raw = legal_confounds.drop(columns=remove_columns)
    confounds_raw.to_csv(bad_conf, sep="\t", index=False)  # save in tmp_path

    # unit test of module `load_confounds_components`
    # and higher level wrapper `load_confounds`
    # The two should behave the same
    missing = {"confounds": [], "keywords": []}
    if strategy_keywords == "high_pass":
        # when no cosine regressors are present in the confound file,
        # it will not raise error
        confounds_unit_test, missing = _load_noise_component(
            confounds_raw, component=strategy_keywords, missing=missing
        )
        assert confounds_unit_test.empty is True
        assert len(missing["keywords"]) == 0

        confounds_lc, _ = load_confounds(img, strategy=[strategy_keywords])
        assert confounds_lc.empty is True

    else:
        confounds_unit_test, missing = _load_noise_component(
            confounds_raw,
            component=strategy_keywords,
            missing=missing,
            meta_json=meta_json,
            **expected_parameters,
        )
        assert confounds_unit_test.empty is True
        assert missing["keywords"] == [strategy_keywords]

        with pytest.raises(ValueError) as exc_info:
            load_confounds(
                img, strategy=[strategy_keywords], **expected_parameters
            )
        assert strategy_keywords in exc_info.value.args[0]
