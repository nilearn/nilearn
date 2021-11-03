import sys
import re
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.preprocessing import scale
import pytest
from nibabel import Nifti1Image
from nilearn.input_data import NiftiMasker
from nilearn.input_data import fmriprep_confounds
from ..fmriprep_confounds import _check_strategy

from nilearn._utils.fmriprep_confounds import _to_camel_case

from .utils import create_tmp_filepath, get_leagal_confound


def _simu_img(tmp_path, demean):
    """Simulate an nifti image based on confound file with some parts confounds
    and some parts noise."""
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
    confounds, _ = fmriprep_confounds(
        file_nii, strategy=("motion", ), motion="basic", demean=False
    )

    X = _handle_non_steady(confounds)
    X = X.values
    # the number of time points is based on the example confound file
    nt = X.shape[0]
    # initialize an empty 4D volume
    vol = np.zeros([nx, ny, 2 * nz, nt])
    vol_conf = np.zeros([nx, ny, 2 * nz])
    vol_rand = np.zeros([nx, ny, 2 * nz])

    # create random noise and a random mixture of confounds standardized
    # to zero mean and unit variance
    if sys.version_info < (3, 7):  # fall back to random state for 3.6
        np.random.RandomState(42)
        beta = np.random.rand(nx * ny * nz, X.shape[1])
        tseries_rand = scale(np.random.rand(nx * ny * nz, nt), axis=1)
    else:
        randome_state = np.random.default_rng(0)
        beta = randome_state.random((nx * ny * nz, X.shape[1]))
        tseries_rand = scale(randome_state.random((nx * ny * nz, nt)), axis=1)
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

    # create an nifti image with the data, and corresponding mask
    img = Nifti1Image(vol, np.eye(4))
    mask_conf = Nifti1Image(vol_conf, np.eye(4))
    mask_rand = Nifti1Image(vol_rand, np.eye(4))

    # generate the associated confounds for testing
    test_confounds, _ = fmriprep_confounds(
        file_nii, strategy=("motion",), motion="basic", demean=demean)
    # match how we extend the length to increase the degree of freedom
    test_confounds = _handle_non_steady(test_confounds)
    sample_mask = np.arange(test_confounds.shape[0])[1:]
    return img, mask_conf, mask_rand, test_confounds, sample_mask


def _handle_non_steady(confounds):
    """Simulate non steady state correctly while increase the length."""
    X = confounds.values
    # the first row is non-steady state, replace it with the input from the
    # second row
    non_steady = X[0, :]
    X[0, :] = X[1, :]
    # repeat X in length (axis = 0) 10 times to increase
    # the degree of freedom for numerical stability
    X = np.tile(X, (10, 1))
    # put non-steady state volume back at the first sample
    X[0, :] = non_steady
    X = pd.DataFrame(X, columns=confounds.columns)
    return X


def _regression(confounds, tmp_path):
    """Simple regression with NiftiMasker."""
    # Simulate data
    img, mask_conf, _, _, _ = _simu_img(tmp_path, demean=False)
    confounds = _handle_non_steady(confounds)
    # Do the regression
    masker = NiftiMasker(mask_img=mask_conf, standardize=True)
    tseries_clean = masker.fit_transform(
        img, confounds=confounds, sample_mask=None
    )
    assert tseries_clean.shape[0] == confounds.shape[0]


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize(
    "test_strategy,param",
    [
        (("motion", ), {}),
        (("high_pass", ), {}),
        (("wm_csf", ), {"wm_csf": "full"}),
        (("global_signal", ), {"global_signal": "full"}),
        (("high_pass", "compcor", ), {}),
        (("high_pass", "compcor", ), {"compcor": "anat_separated"}),
        (("high_pass", "compcor", ), {"compcor": "temporal"}),
        (("ica_aroma", ), {"ica_aroma": "basic"}),
    ],
)
def test_nilearn_regress(tmp_path, test_strategy, param):
    """Try regressing out all motion types without sample mask."""
    img_nii, _ = create_tmp_filepath(
        tmp_path, copy_confounds=True, copy_json=True
    )
    confounds, _ = fmriprep_confounds(img_nii, strategy=test_strategy, **param)
    _regression(confounds, tmp_path)


def _tseries_std(img, mask_img, confounds, sample_mask,
                 standardize_signal=False, standardize_confounds=True,
                 detrend=False):
    """Get the std of time series in a mask."""
    masker = NiftiMasker(
        mask_img=mask_img,
        standardize=standardize_signal,
        standardize_confounds=standardize_confounds,
        detrend=detrend
    )
    tseries = masker.fit_transform(img,
                                   confounds=confounds,
                                   sample_mask=sample_mask)
    return tseries.std(axis=0)


def _denoise(img, mask_img, confounds, sample_mask,
             standardize_signal=False, standardize_confounds=True,
             detrend=False):
    """Extract time series with and without confounds."""
    masker = NiftiMasker(mask_img=mask_img,
                         standardize=standardize_signal,
                         standardize_confounds=standardize_confounds,
                         detrend=detrend)
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
    (img, mask_conf, mask_rand,
     confounds, sample_mask) = _simu_img(tmp_path, demean=False)

    # Check that most variance is removed
    # in voxels composed of pure confounds
    tseries_std = _tseries_std(img, mask_conf, confounds, sample_mask,
                               standardize_signal=False,
                               standardize_confounds=True,
                               detrend=False)
    assert np.mean(tseries_std < 0.0001)

    # Check that most variance is preserved
    # in voxels composed of random noise
    tseries_std = _tseries_std(img, mask_rand, confounds, sample_mask,
                               standardize_signal=False,
                               standardize_confounds=True,
                               detrend=False)
    assert np.mean(tseries_std > 0.9)


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("standardize_signal", ["zscore", "psc"])
@pytest.mark.parametrize("standardize_confounds,detrend", [(True, False),
                                                           (False, True),
                                                           (True, True)])
def test_nilearn_standardize(tmp_path, standardize_signal,
                             standardize_confounds, detrend):
    """Test confounds removal with logical parameters for processing signal."""
    # demean is set to False to let signal.clean handle everything
    (img, mask_conf, mask_rand, confounds, mask) = _simu_img(tmp_path,
                                                             demean=False)
    # We now load the time series with vs without confounds
    # in voxels composed of pure confounds
    # the correlation before and after denoising should be very low
    # as most of the variance is removed by denoising
    tseries_raw, tseries_clean = _denoise(
        img, mask_conf, confounds, mask,
        standardize_signal=standardize_signal,
        standardize_confounds=standardize_confounds,
        detrend=detrend)
    corr = _corr_tseries(tseries_raw, tseries_clean)
    assert np.absolute(np.mean(corr)) < 0.2

    # We now load the time series with zscore standardization
    # with vs without confounds in voxels where the signal is uncorrelated
    # with confounds. The correlation before and after denoising should be very
    # high as very little of the variance is removed by denoising
    tseries_raw, tseries_clean = _denoise(
        img, mask_rand, confounds, mask,
        standardize_signal=standardize_signal,
        standardize_confounds=standardize_confounds,
        detrend=detrend)
    corr = _corr_tseries(tseries_raw, tseries_clean)
    assert corr.mean() > 0.8


def test_confounds2df(tmp_path):
    """Check auto-detect of confonds from an fMRI nii image."""
    img_nii, _ = create_tmp_filepath(tmp_path, copy_confounds=True)
    confounds, _ = fmriprep_confounds(img_nii)
    assert "trans_x" in confounds.columns


@pytest.mark.parametrize("strategy,message",
                         [(["string", ], "not a supported type of confounds."),
                          ("error", "tuple or list of strings"),
                          ((0, ), "not a supported type of confounds."),
                          (("compcor", ), "high_pass")])
def test_check_strategy(strategy, message):
    """Check that flawed strategy options generate meaningful error
    messages."""
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


@pytest.mark.parametrize("motion", ["basic", "derivatives", "power2", "full"])
@pytest.mark.parametrize(
    "param", ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
)
def test_motion(tmp_path, motion, param, expected_suffixes):
    img_nii, _ = create_tmp_filepath(tmp_path, copy_confounds=True)
    conf, _ = fmriprep_confounds(
        img_nii, strategy=("motion", ), motion=motion
    )
    for suff in SUFFIXES:
        if suff in expected_suffixes:
            assert f"{param}{suff}" in conf.columns
        else:
            assert f"{param}{suff}" not in conf.columns


@pytest.mark.parametrize("compcor,n_compcor,test_keyword,test_n",
                         [("anat_combined", 2, "a_comp_cor_", 2),
                          ("anat_combined", "all", "a_comp_cor_", 57),
                          ("temporal", "all", "t_comp_cor_", 6)])
def test_n_compcor(tmp_path, compcor, n_compcor, test_keyword, test_n):
    img_nii, _ = create_tmp_filepath(
        tmp_path, copy_confounds=True, copy_json=True
    )
    conf, _ = fmriprep_confounds(
        img_nii, strategy=("high_pass", "compcor", ), compcor=compcor,
        n_compcor=n_compcor
    )
    assert sum(True for col in conf.columns if test_keyword in col) == test_n


def test_not_found_exception(tmp_path):
    """Check various file or parameter missing scenario."""
    # Create invalid confound file in temporary dir
    img_missing_confounds, bad_conf = create_tmp_filepath(
        tmp_path, copy_confounds=True, copy_json=False
    )
    missing_params = ["trans_y", "trans_x_derivative1", "rot_z_power2"]
    missing_keywords = ["cosine"]

    leagal_confounds = pd.read_csv(bad_conf, delimiter="\t", encoding="utf-8")
    cosine = [
        col_name
        for col_name in leagal_confounds.columns
        if "cosine" in col_name
    ]
    aroma = [
        col_name
        for col_name in leagal_confounds.columns
        if "aroma" in col_name
    ]
    missing_confounds = leagal_confounds.drop(
        columns=missing_params + cosine + aroma
    )
    missing_confounds.to_csv(bad_conf, sep="\t", index=False)

    with pytest.raises(ValueError) as exc_info:
        fmriprep_confounds(
            img_missing_confounds,
            strategy=("high_pass", "motion", "global_signal", ),
            global_signal="full",
            motion="full",
        )
    assert f"{missing_params}" in exc_info.value.args[0]
    assert f"{missing_keywords}" in exc_info.value.args[0]

    # loading anat compcor should also raise an error, because the json file is
    # missing for that example dataset
    with pytest.raises(ValueError):
        fmriprep_confounds(
            img_missing_confounds,
            strategy=("high_pass", "compcor"),
            compcor="anat_combined",
        )

    # catch invalid compcor option
    with pytest.raises(KeyError):
        fmriprep_confounds(
            img_missing_confounds, strategy=("high_pass", "compcor"),
            compcor="blah"
        )

    # Aggressive ICA-AROMA strategy requires
    # default nifti and noise ICs in confound file
    # correct nifti but missing noise regressor
    with pytest.raises(ValueError) as exc_info:
        fmriprep_confounds(
            img_missing_confounds, strategy=("ica_aroma", ), ica_aroma="basic"
        )
    assert "aroma" in exc_info.value.args[0]

    # Aggressive ICA-AROMA strategy requires
    # default nifti
    aroma_nii, _ = create_tmp_filepath(
        tmp_path, image_type="ica_aroma", suffix="aroma"
    )
    with pytest.raises(ValueError) as exc_info:
        fmriprep_confounds(
            aroma_nii, strategy=("ica_aroma", ), ica_aroma="basic"
        )
    assert "Invalid file type" in exc_info.value.args[0]

    # non aggressive ICA-AROMA strategy requires
    # desc-smoothAROMAnonaggr nifti file
    with pytest.raises(ValueError) as exc_info:
        fmriprep_confounds(
            img_missing_confounds, strategy=("ica_aroma", ), ica_aroma="full"
        )
    assert "desc-smoothAROMAnonaggr_bold" in exc_info.value.args[0]

    # no confound files along the image file
    (tmp_path / bad_conf).unlink()
    with pytest.raises(ValueError) as exc_info:
        fmriprep_confounds(img_missing_confounds)
    assert "Could not find associated confound file." in exc_info.value.args[0]


def test_non_steady_state(tmp_path):
    """Warn when 'non_steady_state' is in strategy."""
    # supplying 'non_steady_state' in strategy is not necessary
    # check warning is correcly raised
    img, conf = create_tmp_filepath(
        tmp_path, copy_confounds=True
    )
    warning_message = (r"Non-steady state")
    with pytest.warns(UserWarning, match=warning_message):
        fmriprep_confounds(img, strategy=('non_steady_state', 'motion'))


def test_load_non_nifti(tmp_path):
    """Test non-nifti and invalid file type as input."""
    # tsv file - unsupported input
    _, tsv = create_tmp_filepath(tmp_path, copy_confounds=True, copy_json=True)

    with pytest.raises(ValueError):
        fmriprep_confounds(str(tsv))

    # cifti file should be supported
    cifti, _ = create_tmp_filepath(
        tmp_path, image_type="cifti", copy_confounds=True, copy_json=True
    )
    conf, _ = fmriprep_confounds(cifti)
    assert conf.size != 0

    # gifti support
    gifti, _ = create_tmp_filepath(
        tmp_path, image_type="gifti", copy_confounds=True, copy_json=True
    )
    conf, _ = fmriprep_confounds(gifti)
    assert conf.size != 0


def test_invalid_filetype(tmp_path):
    """Invalid file types/associated files for load method."""
    bad_nii, bad_conf = create_tmp_filepath(tmp_path, copy_confounds=True)
    conf, _ = fmriprep_confounds(bad_nii)

    # more than one legal filename for confounds
    add_conf = "test_desc-confounds_timeseries.tsv"
    leagal_confounds, _ = get_leagal_confound()
    leagal_confounds.to_csv(tmp_path / add_conf, sep="\t", index=False)
    with pytest.raises(ValueError) as info:
        fmriprep_confounds(bad_nii)
    assert "more than one" in str(info.value)
    (tmp_path / add_conf).unlink()  # Remove for the rest of the tests to run

    # invalid fmriprep version: confound file with no header (<1.0)
    fake_confounds = np.random.rand(30, 20)
    np.savetxt(bad_conf, fake_confounds, delimiter="\t")
    with pytest.raises(ValueError) as error_log:
        fmriprep_confounds(bad_nii)
    assert "The confound file contains no header." in str(error_log.value)

    # invalid fmriprep version: old camel case header (<1.2)
    leagal_confounds, _ = get_leagal_confound()
    camel_confounds = leagal_confounds.copy()
    camel_confounds.columns = [
        _to_camel_case(col_name) for col_name in leagal_confounds.columns
    ]
    camel_confounds.to_csv(bad_conf, sep="\t", index=False)
    with pytest.raises(ValueError) as error_log:
        fmriprep_confounds(bad_nii)
    assert "contains header in camel case." in str(error_log.value)

    # create a empty nifti file with no associated confound file
    # We only need the path to check this
    no_conf = "no_confound_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    no_confound = tmp_path / no_conf
    no_confound.touch()
    with pytest.raises(ValueError):
        fmriprep_confounds(bad_nii)


def test_ica_aroma(tmp_path):
    """Test ICA AROMA related file input."""
    aroma_nii, _ = create_tmp_filepath(
        tmp_path, image_type="ica_aroma", copy_confounds=True
    )
    regular_nii, _ = create_tmp_filepath(
        tmp_path, image_type="regular", copy_confounds=True
    )
    # Aggressive strategy
    conf, _ = fmriprep_confounds(
        regular_nii, strategy=("ica_aroma", ), ica_aroma="basic"
    )
    for col_name in conf.columns:
        # only aroma and non-steady state columns will be present
        assert re.match("(?:aroma_motion_+|non_steady_state+)", col_name)

    # Non-aggressive strategy
    conf, _ = fmriprep_confounds(
        aroma_nii, strategy=("ica_aroma", ), ica_aroma="full"
    )
    assert conf.size == 0

    # invalid combination of strategy and option
    with pytest.raises(ValueError) as exc_info:
        conf, _ = fmriprep_confounds(
            regular_nii, strategy=("ica_aroma", ), ica_aroma="invalid"
        )
    assert "Current input: invalid" in exc_info.value.args[0]


def test_sample_mask(tmp_path):
    """Test load method and sample mask."""
    regular_nii, regular_conf = create_tmp_filepath(
        tmp_path, image_type="regular", copy_confounds=True
    )

    reg, mask = fmriprep_confounds(
        regular_nii, strategy=("motion", "scrub"), scrub=5, fd_thresh=0.15
    )
    # the current test data has 6 time points marked as motion outliers,
    # and one nonsteady state (overlap with the first motion outlier)
    # 2 time points removed due to the "full" srubbing strategy (remove segment
    # shorter than 5 volumes)
    assert reg.shape[0] - len(mask) == 8
    # nilearn requires unmasked confound regressors
    assert reg.shape[0] == 30

    # non steady state will always be removed
    reg, mask = fmriprep_confounds(regular_nii, strategy=("motion", ))
    assert reg.shape[0] - len(mask) == 1

    # When no non-steady state volumes are present
    conf_data, _ = get_leagal_confound(non_steady_state=False)
    conf_data.to_csv(regular_conf, sep="\t", index=False)  # save to tmp
    reg, mask = fmriprep_confounds(regular_nii, strategy=("motion", ))
    assert mask is None

    # When no volumes needs removing (very liberal motion threshould)
    reg, mask = fmriprep_confounds(
        regular_nii, strategy=("motion", "scrub"), scrub=0, fd_thresh=4
    )
    assert mask is None


@pytest.mark.parametrize(
    "image_type", ["regular", "ica_aroma", "gifti", "cifti"]
)
def test_inputs(tmp_path, image_type):
    """Test multiple images as input."""
    # generate files
    files = []
    for i in range(2):  # gifti edge case
        nii, _ = create_tmp_filepath(
            tmp_path,
            suffix=f"img{i+1}",
            image_type=image_type,
            copy_confounds=True,
            copy_json=True,
        )
        files.append(nii)

    if image_type == "ica_aroma":
        conf, _ = fmriprep_confounds(files, strategy=("ica_aroma", ))
    else:
        conf, _ = fmriprep_confounds(files)
    assert len(conf) == 2
