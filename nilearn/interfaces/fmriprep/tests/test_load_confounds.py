import re

import numpy as np
import pandas as pd
import pytest
from nibabel import Nifti1Image
from scipy.stats import pearsonr
from sklearn.preprocessing import scale

from nilearn._utils.data_gen import create_fake_bids_dataset
from nilearn._utils.fmriprep_confounds import to_camel_case
from nilearn.conftest import _rng
from nilearn.interfaces.bids import get_bids_files
from nilearn.interfaces.fmriprep import load_confounds
from nilearn.interfaces.fmriprep.load_confounds import (
    _check_strategy,
    _load_single_confounds_file,
)
from nilearn.interfaces.fmriprep.tests._testing import (
    create_tmp_filepath,
    get_legal_confound,
)
from nilearn.maskers import NiftiMasker
from nilearn.tests.test_signal import generate_trends


def _simu_img(tmp_path, trend, demean):
    """Simulate an nifti image based on confound file \
    with some parts confounds and some parts noise.
    """
    file_nii, _ = create_tmp_filepath(tmp_path, copy_confounds=True)
    # set the size of the image matrix
    nx = 5
    ny = 5
    # the actual number of slices will actually be double of that
    # as we will stack slices with confounds on top of slices with noise
    nz = 2
    # Load a simple 6 parameters motion models as confounds
    # demean set to False just for simulating signal based on the original
    # state
    confounds, _ = load_confounds(
        file_nii, strategy=("motion",), motion="basic", demean=False
    )

    X = _handle_non_steady(confounds)
    X = X.to_numpy()
    # the number of time points is based on the example confound file
    nt = X.shape[0]
    # initialize an empty 4D volume
    vol = np.zeros([nx, ny, 2 * nz, nt])
    vol_conf = np.zeros([nx, ny, 2 * nz])
    vol_rand = np.zeros([nx, ny, 2 * nz])

    # create random noise and a random mixture of confounds standardized
    # to zero mean and unit variance
    rng = _rng()
    beta = rng.random((nx * ny * nz, X.shape[1]))
    tseries_rand = scale(rng.random((nx * ny * nz, nt)), axis=1)
    # create the confound mixture
    tseries_conf = scale(np.matmul(beta, X.transpose()), axis=1)

    # fill the first half of the 4D data with the random mixture
    vol[:, :, 0:nz, :] = tseries_conf.reshape(nx, ny, nz, nt)
    vol_conf[:, :, 0:nz] = 1

    # create random noise in the second half of the 4D data
    vol[:, :, range(nz, 2 * nz), :] = tseries_rand.reshape(nx, ny, nz, nt)
    vol_rand[:, :, range(nz, 2 * nz)] = 1

    # Shift the mean to non-zero
    vol = vol + 10

    # add a linear trend to the data
    if trend:
        signal_trend = generate_trends(n_features=nx * ny * 2 * nz, length=nt)
        vol += signal_trend.reshape(nx, ny, 2 * nz, nt)

    # create an nifti image with the data, and corresponding mask
    img = Nifti1Image(vol, np.eye(4))
    mask_conf = Nifti1Image(vol_conf, np.eye(4))
    mask_rand = Nifti1Image(vol_rand, np.eye(4))

    # generate the associated confounds for testing
    test_confounds, _ = load_confounds(
        file_nii, strategy=("motion",), motion="basic", demean=demean
    )
    # match how we extend the length to increase the degree of freedom
    test_confounds = _handle_non_steady(test_confounds)
    sample_mask = np.arange(test_confounds.shape[0])[1:]
    return img, mask_conf, mask_rand, test_confounds, sample_mask


def _handle_non_steady(confounds):
    """Simulate non steady state correctly while increase the length.

    - The first row is non-steady state,
      replace it with the input from the second row.

    - Repeat X in length (axis = 0) 10 times to increase
      the degree of freedom for numerical stability.

    - Put non-steady state volume back at the first sample.
    """
    X = confounds.to_numpy()
    non_steady = X[0, :]
    tmp = np.vstack((X[1, :], X[1:, :]))
    tmp = np.tile(tmp, (10, 1))
    return pd.DataFrame(
        np.vstack((non_steady, tmp[1:, :])), columns=confounds.columns
    )


def _regression(confounds, tmp_path):
    """Perform simple regression with NiftiMasker."""
    # Simulate data
    img, mask_conf, _, _, _ = _simu_img(tmp_path, trend=False, demean=False)
    confounds = _handle_non_steady(confounds)
    # Do the regression
    masker = NiftiMasker(mask_img=mask_conf, standardize=True)
    tseries_clean = masker.fit_transform(
        img, confounds=confounds, sample_mask=None
    )
    assert tseries_clean.shape[0] == confounds.shape[0]


@pytest.mark.parametrize("fmriprep_version", ["1.4.x", "21.x.x"])
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize(
    "test_strategy,param",
    [
        (("motion",), {}),
        (("high_pass",), {}),
        (("wm_csf",), {"wm_csf": "full"}),
        (("global_signal",), {"global_signal": "full"}),
        (("high_pass", "compcor"), {}),
        (("high_pass", "compcor"), {"compcor": "anat_separated"}),
        (("high_pass", "compcor"), {"compcor": "temporal"}),
        (("ica_aroma",), {"ica_aroma": "basic"}),
    ],
)
def test_nilearn_regress(tmp_path, test_strategy, param, fmriprep_version):
    """Try regressing out all motion types without sample mask."""
    img_nii, _ = create_tmp_filepath(
        tmp_path,
        copy_confounds=True,
        copy_json=True,
        fmriprep_version=fmriprep_version,
    )
    if fmriprep_version == "21.x.x" and test_strategy == ("ica_aroma",):
        return
    confounds, _ = load_confounds(img_nii, strategy=test_strategy, **param)
    _regression(confounds, tmp_path)


def _tseries_std(
    img,
    mask_img,
    confounds,
    sample_mask,
    standardize_signal=False,
    standardize_confounds=True,
    detrend=False,
):
    """Get the std of time series in a mask."""
    masker = NiftiMasker(
        mask_img=mask_img,
        standardize=standardize_signal,
        standardize_confounds=standardize_confounds,
        detrend=detrend,
    )
    tseries = masker.fit_transform(
        img, confounds=confounds, sample_mask=sample_mask
    )
    return tseries.std(axis=0)


def _denoise(
    img,
    mask_img,
    confounds,
    sample_mask,
    standardize_signal=False,
    standardize_confounds=True,
    detrend=False,
):
    """Extract time series with and without confounds."""
    masker = NiftiMasker(
        mask_img=mask_img,
        standardize=standardize_signal,
        standardize_confounds=standardize_confounds,
        detrend=detrend,
    )
    tseries_raw = masker.fit_transform(img, sample_mask=sample_mask)
    tseries_clean = masker.fit_transform(
        img, confounds=confounds, sample_mask=sample_mask
    )
    return tseries_raw, tseries_clean


def _corr_tseries(tseries1, tseries2):
    """Compute the correlation between two sets of time series."""
    corr = np.zeros(tseries1.shape[1])
    for ind in range(tseries1.shape[1]):
        corr[ind], _ = pearsonr(tseries1[:, ind], tseries2[:, ind])
    return corr


@pytest.mark.filterwarnings("ignore")
def test_nilearn_standardize_false(tmp_path):
    """Test removing confounds with no standardization."""
    # niftimasker default:
    # standardize=False, standardize_confounds=True, detrend=False

    # Simulate data; set demean to False as standardize_confounds=True
    (img, mask_conf, mask_rand, confounds, sample_mask) = _simu_img(
        tmp_path, trend=False, demean=False
    )

    # Check that most variance is removed
    # in voxels composed of pure confounds
    tseries_std = _tseries_std(
        img,
        mask_conf,
        confounds,
        sample_mask,
        standardize_signal=False,
        standardize_confounds=True,
        detrend=False,
    )
    assert np.mean(tseries_std < 0.0001)

    # Check that most variance is preserved
    # in voxels composed of random noise
    tseries_std = _tseries_std(
        img,
        mask_rand,
        confounds,
        sample_mask,
        standardize_signal=False,
        standardize_confounds=True,
        detrend=False,
    )
    assert np.mean(tseries_std > 0.9)


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("standardize_signal", ["zscore", "psc"])
@pytest.mark.parametrize(
    "standardize_confounds,detrend",
    [(True, False), (False, True), (True, True)],
)
def test_nilearn_standardize(
    tmp_path, standardize_signal, standardize_confounds, detrend
):
    """Test confounds removal with logical parameters for processing signal."""
    # demean is set to False to let signal.clean handle everything
    (img, mask_conf, mask_rand, confounds, mask) = _simu_img(
        tmp_path, trend=True, demean=False
    )
    # We now load the time series with vs without confounds
    # in voxels composed of pure confounds
    # the correlation before and after denoising should be very low
    # as most of the variance is removed by denoising
    tseries_raw, tseries_clean = _denoise(
        img,
        mask_conf,
        confounds,
        mask,
        standardize_signal=standardize_signal,
        standardize_confounds=standardize_confounds,
        detrend=detrend,
    )
    corr = _corr_tseries(tseries_raw, tseries_clean)
    assert np.absolute(np.mean(corr)) < 0.2

    # We now load the time series with zscore standardization
    # with vs without confounds in voxels where the signal is uncorrelated
    # with confounds. The correlation before and after denoising should be very
    # high as very little of the variance is removed by denoising
    tseries_raw, tseries_clean = _denoise(
        img,
        mask_rand,
        confounds,
        mask,
        standardize_signal=standardize_signal,
        standardize_confounds=standardize_confounds,
        detrend=detrend,
    )
    corr = _corr_tseries(tseries_raw, tseries_clean)
    assert corr.mean() > 0.8


@pytest.mark.parametrize("fmriprep_version", ["1.4.x", "21.x.x"])
def test_confounds2df(tmp_path, fmriprep_version):
    """Check auto-detect of confonds from an fMRI nii image."""
    img_nii, _ = create_tmp_filepath(
        tmp_path, copy_confounds=True, fmriprep_version=fmriprep_version
    )
    confounds, _ = load_confounds(img_nii)
    assert "trans_x" in confounds.columns


@pytest.mark.parametrize("fmriprep_version", ["1.4.x", "21.x.x"])
def test_load_single_confounds_file(tmp_path, fmriprep_version):
    """Check that the load_confounds function returns the same confounds \
    as _load_single_confounds_file.
    """
    nii_file, confounds_file = create_tmp_filepath(
        tmp_path, copy_confounds=True, fmriprep_version=fmriprep_version
    )

    # get defaults from load_confounds
    import inspect

    _defaults = {
        key: value.default
        for key, value in inspect.signature(load_confounds).parameters.items()
    }
    _defaults.pop("img_files")
    _default_strategy = _defaults.pop("strategy")

    _, confounds = _load_single_confounds_file(
        str(confounds_file), strategy=_default_strategy, **_defaults
    )
    confounds_nii, _ = load_confounds(
        nii_file, strategy=_default_strategy, **_defaults
    )
    pd.testing.assert_frame_equal(confounds, confounds_nii)


@pytest.mark.parametrize(
    "strategy,message",
    [
        (
            ["string"],
            "not a supported type of confounds.",
        ),
        ("error", "tuple or list of strings"),
        ((0,), "not a supported type of confounds."),
        (("compcor",), "high_pass"),
    ],
)
def test_check_strategy(strategy, message):
    """Check that flawed strategy options \
    generate meaningful error messages.
    """
    with pytest.raises(ValueError) as exc_info:
        _check_strategy(strategy=strategy)
    assert message in exc_info.value.args[0]


SUFFIXES = np.array(["", "_derivative1", "_power2", "_derivative1_power2"])


@pytest.fixture
def expected_suffixes(motion):
    expectation = {
        "basic": slice(1),
        "derivatives": slice(2),
        "power2": np.array([True, False, True, False]),
        "full": slice(4),
    }
    return SUFFIXES[expectation[motion]]


@pytest.mark.parametrize("fmriprep_version", ["1.4.x", "21.x.x"])
@pytest.mark.parametrize("motion", ["basic", "derivatives", "power2", "full"])
@pytest.mark.parametrize(
    "param", ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
)
def test_motion(tmp_path, motion, param, expected_suffixes, fmriprep_version):
    img_nii, _ = create_tmp_filepath(
        tmp_path, copy_confounds=True, fmriprep_version=fmriprep_version
    )
    conf, _ = load_confounds(img_nii, strategy=("motion",), motion=motion)
    for suff in SUFFIXES:
        if suff in expected_suffixes:
            assert f"{param}{suff}" in conf.columns
        else:
            assert f"{param}{suff}" not in conf.columns


@pytest.mark.parametrize(
    "compcor, n_compcor, test_keyword, test_n, fmriprep_version",
    [
        ("anat_combined", 2, "a_comp_cor_", 2, "1.4.x"),
        ("anat_separated", 2, "a_comp_cor_", 4, "1.4.x"),
        ("anat_combined", "all", "a_comp_cor_", 57, "1.4.x"),
        ("temporal", "all", "t_comp_cor_", 6, "1.4.x"),
        ("anat_combined", 2, "a_comp_cor_", 2, "21.x.x"),
        ("anat_separated", "all", "w_comp_cor_", 4, "21.x.x"),
        ("temporal_anat_separated", "all", "c_comp_cor_", 3, "21.x.x"),
        ("temporal", "all", "t_comp_cor_", 3, "21.x.x"),
    ],
)
def test_n_compcor(
    tmp_path, compcor, n_compcor, test_keyword, test_n, fmriprep_version
):
    img_nii, _ = create_tmp_filepath(
        tmp_path,
        copy_confounds=True,
        copy_json=True,
        fmriprep_version=fmriprep_version,
    )
    conf, _ = load_confounds(
        img_nii,
        strategy=(
            "high_pass",
            "compcor",
        ),
        compcor=compcor,
        n_compcor=n_compcor,
    )
    assert sum(True for col in conf.columns if test_keyword in col) == test_n


missing_params = ["trans_y", "trans_x_derivative1", "rot_z_power2"]
missing_keywords = ["cosine", "global_signal"]


def _remove_confounds(conf_file):
    legal_confounds = pd.read_csv(conf_file, delimiter="\t", encoding="utf-8")
    remove_columns = []
    for missing_kw in missing_keywords:
        remove_columns += [
            col_name
            for col_name in legal_confounds.columns
            if missing_kw in col_name
        ]

    aroma = [
        col_name for col_name in legal_confounds.columns if "aroma" in col_name
    ]
    missing_confounds = legal_confounds.drop(
        columns=missing_params + remove_columns + aroma
    )
    missing_confounds.to_csv(conf_file, sep="\t", index=False)


@pytest.mark.parametrize("fmriprep_version", ["1.4.x", "21.x.x"])
def test_not_found_exception(tmp_path, fmriprep_version):
    """Check various file or parameter missing scenario."""
    # Create invalid confound file in temporary dir
    img_missing_confounds, bad_conf = create_tmp_filepath(
        tmp_path,
        copy_confounds=True,
        copy_json=False,
        fmriprep_version=fmriprep_version,
    )

    _remove_confounds(bad_conf)

    with pytest.raises(ValueError) as exc_info:
        load_confounds(
            img_missing_confounds,
            strategy=(
                "high_pass",
                "motion",
                "global_signal",
            ),
            global_signal="full",
            motion="full",
        )
    assert f"{missing_params}" in exc_info.value.args[0]

    # missing cosine if it's not present in the file it's fine
    assert f"{missing_keywords[-1:]}" in exc_info.value.args[0]

    # loading anat compcor should also raise an error, because the json file is
    # missing for that example dataset
    with pytest.raises(ValueError):
        load_confounds(
            img_missing_confounds,
            strategy=("high_pass", "compcor"),
            compcor="anat_combined",
        )

    # catch invalid compcor option
    with pytest.raises(KeyError):
        load_confounds(
            img_missing_confounds,
            strategy=("high_pass", "compcor"),
            compcor="blah",
        )


@pytest.mark.parametrize("fmriprep_version", ["1.4.x", "21.x.x"])
def test_not_found_exception_ica_aroma(tmp_path, fmriprep_version):
    """Check various file or parameter for ICA-AROMA strategy."""
    # Create invalid confound file in temporary dir
    img_missing_confounds, bad_conf = create_tmp_filepath(
        tmp_path,
        copy_confounds=True,
        copy_json=False,
        fmriprep_version=fmriprep_version,
    )

    _remove_confounds(bad_conf)

    # Aggressive ICA-AROMA strategy requires
    # default nifti and noise ICs in confound file
    # correct nifti but missing noise regressor
    with pytest.raises(ValueError) as exc_info:
        load_confounds(
            img_missing_confounds, strategy=("ica_aroma",), ica_aroma="basic"
        )
    assert "ica_aroma" in exc_info.value.args[0]

    # Default nifti
    aroma_nii, _ = create_tmp_filepath(
        tmp_path,
        image_type="ica_aroma",
        bids_fields={"entities": {"sub": "icaAroma"}},
        fmriprep_version=fmriprep_version,
    )
    with pytest.raises(ValueError) as exc_info:
        load_confounds(aroma_nii, strategy=("ica_aroma",), ica_aroma="basic")
    assert "Invalid file type" in exc_info.value.args[0]

    # non aggressive ICA-AROMA strategy requires
    # desc-smoothAROMAnonaggr nifti file
    with pytest.raises(ValueError) as exc_info:
        load_confounds(
            img_missing_confounds, strategy=("ica_aroma",), ica_aroma="full"
        )
    assert "desc-smoothAROMAnonaggr_bold" in exc_info.value.args[0]

    # no confound files along the image file
    (tmp_path / bad_conf).unlink()
    with pytest.raises(ValueError) as exc_info:
        load_confounds(img_missing_confounds)
    assert "Could not find associated confound file." in exc_info.value.args[0]


@pytest.mark.parametrize("fmriprep_version", ["1.4.x", "21.x.x"])
def test_non_steady_state(tmp_path, fmriprep_version):
    """Warn when 'non_steady_state' is in strategy."""
    # supplying 'non_steady_state' in strategy is not necessary
    # check warning is correctly raised
    img, _ = create_tmp_filepath(
        tmp_path, copy_confounds=True, fmriprep_version=fmriprep_version
    )
    warning_message = r"Non-steady state"
    with pytest.warns(UserWarning, match=warning_message):
        load_confounds(img, strategy=("non_steady_state", "motion"))


def test_load_non_nifti(tmp_path):
    """Test non-nifti and invalid file type as input."""
    # tsv file - unsupported input
    _, tsv = create_tmp_filepath(tmp_path, copy_confounds=True, copy_json=True)

    with pytest.raises(ValueError):
        load_confounds(str(tsv))

    # cifti file should be supported
    cifti, _ = create_tmp_filepath(
        tmp_path, image_type="cifti", copy_confounds=True, copy_json=True
    )
    conf, _ = load_confounds(cifti)
    assert conf.size != 0

    # gifti support
    gifti, _ = create_tmp_filepath(
        tmp_path, image_type="gifti", copy_confounds=True, copy_json=True
    )
    conf, _ = load_confounds(gifti)
    assert conf.size != 0


def test_invalid_filetype(tmp_path, rng):
    """Invalid file types/associated files for load method."""
    bad_nii, bad_conf = create_tmp_filepath(
        tmp_path, copy_confounds=True, fmriprep_version="1.4.x"
    )
    _, _ = load_confounds(bad_nii)

    # more than one legal filename for confounds
    add_conf = "sub-14x_task-test_desc-confounds_regressors.tsv"
    legal_confounds, _ = get_legal_confound()
    legal_confounds.to_csv(tmp_path / add_conf, sep="\t", index=False)
    with pytest.raises(ValueError) as info:
        load_confounds(bad_nii)
    assert "more than one" in str(info.value)
    (tmp_path / add_conf).unlink()  # Remove for the rest of the tests to run

    # invalid fmriprep version: confound file with no header (<1.0)
    fake_confounds = rng.random((30, 20))
    np.savetxt(bad_conf, fake_confounds, delimiter="\t")
    with pytest.raises(ValueError) as error_log:
        load_confounds(bad_nii)
    assert "The confound file contains no header." in str(error_log.value)

    # invalid fmriprep version: old camel case header (<1.2)
    legal_confounds, _ = get_legal_confound()
    camel_confounds = legal_confounds.copy()
    camel_confounds.columns = [
        to_camel_case(col_name) for col_name in legal_confounds.columns
    ]
    camel_confounds.to_csv(bad_conf, sep="\t", index=False)
    with pytest.raises(ValueError) as error_log:
        load_confounds(bad_nii)
    assert "contains header in camel case." in str(error_log.value)

    # create a empty nifti file with no associated confound file
    # We only need the path to check this
    no_conf = "no_confound_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    no_confound = tmp_path / no_conf
    no_confound.touch()
    with pytest.raises(ValueError):
        load_confounds(bad_nii)


@pytest.mark.parametrize("fmriprep_version", ["1.4.x"])
def test_ica_aroma(tmp_path, fmriprep_version):
    """Test ICA AROMA related file input."""
    aroma_nii, _ = create_tmp_filepath(
        tmp_path,
        image_type="ica_aroma",
        copy_confounds=True,
        fmriprep_version=fmriprep_version,
    )
    regular_nii, _ = create_tmp_filepath(
        tmp_path,
        image_type="regular",
        copy_confounds=True,
        fmriprep_version=fmriprep_version,
    )
    # Aggressive strategy
    conf, _ = load_confounds(
        regular_nii, strategy=("ica_aroma",), ica_aroma="basic"
    )
    for col_name in conf.columns:
        # only aroma and non-steady state columns will be present
        assert re.match("(?:aroma_motion_+|non_steady_state+)", col_name)

    # Non-aggressive strategy
    conf, _ = load_confounds(
        aroma_nii, strategy=("ica_aroma",), ica_aroma="full"
    )
    assert conf.size == 0

    # invalid combination of strategy and option
    with pytest.raises(ValueError) as exc_info:
        conf, _ = load_confounds(
            regular_nii, strategy=("ica_aroma",), ica_aroma="invalid"
        )
    assert "Current input: invalid" in exc_info.value.args[0]


@pytest.mark.parametrize(
    "fmriprep_version, scrubbed_time_points, non_steady_outliers",
    [("1.4.x", 8, 1), ("21.x.x", 30, 3)],
)
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_sample_mask(
    tmp_path, fmriprep_version, scrubbed_time_points, non_steady_outliers
):
    """Test load method and sample mask."""
    regular_nii, regular_conf = create_tmp_filepath(
        tmp_path,
        image_type="regular",
        copy_confounds=True,
        fmriprep_version=fmriprep_version,
    )

    reg, mask = load_confounds(
        regular_nii, strategy=("motion", "scrub"), scrub=5, fd_threshold=0.15
    )
    # the "1.4.x" test data has 6 time points marked as motion outliers,
    # and one nonsteady state (overlap with the first motion outlier)
    # 2 time points removed due to the "full" scrubbing strategy
    # (remove segment shorter than 5 volumes)
    assert reg.shape[0] - len(mask) == scrubbed_time_points

    # nilearn requires unmasked confound regressors
    assert reg.shape[0] == 30

    # non steady state will always be removed
    reg, mask = load_confounds(regular_nii, strategy=("motion",))
    assert reg.shape[0] - len(mask) == non_steady_outliers

    # When no non-steady state volumes are present
    conf_data, _ = get_legal_confound(non_steady_state=False)
    conf_data.to_csv(regular_conf, sep="\t", index=False)  # save to tmp
    reg, mask = load_confounds(regular_nii, strategy=("motion",))
    assert mask is None

    # When no volumes needs removing (very liberal motion threshould)
    reg, mask = load_confounds(
        regular_nii, strategy=("motion", "scrub"), scrub=0, fd_threshold=4
    )
    assert mask is None


@pytest.mark.parametrize(
    "image_type",
    [
        "regular",
        "native",
        "ica_aroma",
        "gifti",
        "cifti",
        "res",
        "den",
        "part",
    ],
)
def test_inputs(tmp_path, image_type):
    """Test multiple images as input."""
    # generate files
    files = []
    for i in range(2):  # gifti edge case
        nii, _ = create_tmp_filepath(
            tmp_path,
            bids_fields={
                "entities": {
                    "sub": f"test{i + 1}",
                    "ses": "test",
                    "task": "testimg",
                    "run": "01",
                }
            },
            image_type=image_type,
            copy_confounds=True,
            copy_json=True,
        )
        files.append(nii)

    if image_type == "ica_aroma":
        conf, _ = load_confounds(files, strategy=("ica_aroma",))
    else:
        conf, _ = load_confounds(files)
    assert len(conf) == 2


def test_load_confounds_for_gifti(tmp_path):
    """Ensure that confounds are found for gifti files.

    Regression test for
    https://github.com/nilearn/nilearn/issues/3817
    Wrong order of space and hemi entity in filename pattern
    lead to confounds not being found.
    """
    bids_path = create_fake_bids_dataset(base_dir=tmp_path, n_sub=1, n_ses=1)
    selection = get_bids_files(
        bids_path / "derivatives",
        sub_label="01",
        file_tag="bold",
        file_type="func.gii",
        filters=[
            ("ses", "01"),
            ("task", "main"),
            ("run", "01"),
            ("hemi", "L"),
        ],
        sub_folder=True,
    )
    assert len(selection) == 1
    load_confounds(
        selection[0],
        strategy=["motion", "wm_csf"],
        motion="basic",
        demean=False,
    )
