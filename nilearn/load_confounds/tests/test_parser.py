import os
import re
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.preprocessing import scale
import pytest
from nibabel import Nifti1Image
from nilearn.input_data import NiftiMasker
from nilearn.load_confounds import parser as lc
from nilearn._utils.load_confounds import to_camel_case

from .utils import create_tmp_filepath, get_leagal_confound

path_data = os.path.join(os.path.dirname(lc.__file__), "data")
file_confounds = os.path.join(
    path_data, "test_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
)
file_no_none_steady = os.path.join(
    path_data,
    "no_nonsteady_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
)


def _simu_img(tmp_path, demean=True):
    """Simulate an nifti image based on confound file with some parts confounds
    and some parts noise."""
    file_nii, _ = create_tmp_filepath(tmp_path, copy_testdata=True)
    # set the size of the image matrix
    nx = 5
    ny = 5
    # the actual number of slices will actually be double of that
    # as we will stack slices with confounds on top of slices with noise
    nz = 2
    # Load a simple 6 parameters motion models as confounds
    confounds, _ = lc.Confounds(
        strategy=["motion"], motion="basic", demean=demean
    ).load(file_nii)
    X = confounds.values
    # the first row is non-steady state, replace it with the imput from the
    # second row
    non_steady = X[0, :]
    X[0, :] = X[1, :]
    # repeat X in length (axis = 0) three times to increase
    # the degree of freedom
    X = np.tile(X, (3, 1))
    # put non-steady state volume back at the first sample
    X[0, :] = non_steady
    # the number of time points is based on the example confound file
    nt = X.shape[0]
    # initialize an empty 4D volume
    vol = np.zeros([nx, ny, 2 * nz, nt])
    vol_conf = np.zeros([nx, ny, 2 * nz])
    vol_rand = np.zeros([nx, ny, 2 * nz])

    # create a random mixture of confounds
    # standardized to zero mean and unit variance
    beta = np.random.rand(nx * ny * nz, X.shape[1])
    tseries_conf = scale(np.matmul(beta, X.transpose()), axis=1)
    # fill the first half of the 4D data with the mixture
    vol[:, :, 0:nz, :] = tseries_conf.reshape(nx, ny, nz, nt)
    vol_conf[:, :, 0:nz] = 1

    # create random noise in the second half of the 4D data
    tseries_rand = scale(np.random.randn(nx * ny * nz, nt), axis=1)
    vol[:, :, range(nz, 2 * nz), :] = tseries_rand.reshape(nx, ny, nz, nt)
    vol_rand[:, :, range(nz, 2 * nz)] = 1

    # Shift the mean to non-zero
    vol = vol + 100

    # create an nifti image with the data, and corresponding mask
    img = Nifti1Image(vol, np.eye(4))
    mask_conf = Nifti1Image(vol_conf, np.eye(4))
    mask_rand = Nifti1Image(vol_rand, np.eye(4))

    return img, mask_conf, mask_rand, X


def _tseries_std(img, mask_img, confounds, sample_mask, standardize):
    """Get the std of time series in a mask."""
    masker = NiftiMasker(
        mask_img=mask_img, standardize=standardize, sample_mask=sample_mask
    )
    tseries = masker.fit_transform(img, confounds=confounds)
    return tseries.std(axis=0)


def _denoise(img, mask_img, confounds, sample_mask, standardize):
    """Extract time series with and without confounds."""
    masker = NiftiMasker(
        mask_img=mask_img, standardize=standardize, sample_mask=sample_mask
    )
    tseries_raw = masker.fit_transform(img)
    tseries_clean = masker.fit_transform(img, confounds=confounds)
    return tseries_raw, tseries_clean


def _corr_tseries(tseries1, tseries2):
    """Compute the correlation between two sets of time series."""
    corr = np.zeros(tseries1.shape[1])
    for ind in range(tseries1.shape[1]):
        corr[ind], _ = pearsonr(tseries1[:, ind], tseries2[:, ind])
    return corr


def _regression(confounds, sample_mask, tmp_path):
    """Simple regression with nilearn."""
    # Simulate data
    img, mask_conf, _, _ = _simu_img(tmp_path, demean=True)
    confounds = np.tile(confounds, (3, 1))  # matching L29 (_simu_img)

    # Do the regression
    masker = NiftiMasker(
        mask_img=mask_conf, standardize=True, sample_mask=sample_mask
    )
    tseries_clean = masker.fit_transform(img, confounds)
    assert tseries_clean.shape[0] == confounds.shape[0]


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize('strategy,param',
                         [("motion", {}), ("high_pass", {}),
                          ("wm_csf", {"wm_csf": "full"}),
                          ("global", {"global_signal": "full"}),
                          ("compcor", {}),
                          ("compcor", {"compcor": "anat_separated"}),
                          ("compcor", {"compcor": "temporal"}),
                          ("ica_aroma", {"ica_aroma": "basic"})])
def test_nilearn_regress(tmp_path, strategy, param):
    """Try regressing out all motion types in nilearn."""
    confounds, _ = lc.Confounds(strategy=[strategy], **param).load(
        file_confounds
    )
    _regression(confounds, None, tmp_path)


@pytest.mark.filterwarnings("ignore")
def test_nilearn_standardize_false(tmp_path):
    """Test removing confounds in nilearn with no standardization."""
    # Simulate data
    img, mask_conf, mask_rand, X = _simu_img(tmp_path, demean=True)

    # Check that most variance is removed
    # in voxels composed of pure confounds
    tseries_std = _tseries_std(img, mask_conf, X, None, False)
    assert np.mean(tseries_std < 0.0001)

    # Check that most variance is preserved
    # in voxels composed of random noise
    tseries_std = _tseries_std(img, mask_rand, X, None, False)
    assert np.mean(tseries_std > 0.9)


@pytest.mark.filterwarnings("ignore")
def test_nilearn_standardize_zscore(tmp_path):
    """Test removing confounds in nilearn with zscore standardization."""
    # Simulate data

    img, mask_conf, mask_rand, X = _simu_img(tmp_path, demean=True)

    # We now load the time series with vs without confounds
    # in voxels composed of pure confounds
    # the correlation before and after denoising should be very low
    # as most of the variance is removed by denoising
    tseries_raw, tseries_clean = _denoise(img, mask_conf, X, None, "zscore")
    corr = _corr_tseries(tseries_raw, tseries_clean)
    assert corr.mean() < 0.2

    # We now load the time series with zscore standardization
    # with vs without confounds in voxels where the signal is uncorrelated
    # with confounds. The correlation before and after denoising should be very
    # high as very little of the variance is removed by denoising
    tseries_raw, tseries_clean = _denoise(img, mask_rand, X, None, "zscore")
    corr = _corr_tseries(tseries_raw, tseries_clean)
    assert corr.mean() > 0.8


def test_nilearn_standardize_psc(tmp_path):
    """Test removing confounds in nilearn with psc standardization."""
    # Similar test to test_nilearn_standardize_zscore, but with psc
    # Simulate data

    img, mask_conf, mask_rand, X = _simu_img(tmp_path, demean=False)

    # Areas with confound
    tseries_raw, tseries_clean = _denoise(img, mask_conf, X, None, "psc")
    corr = _corr_tseries(tseries_raw, tseries_clean)
    assert corr.mean() < 0.2

    # Areas with random noise
    tseries_raw, tseries_clean = _denoise(img, mask_rand, X, None, "psc")
    corr = _corr_tseries(tseries_raw, tseries_clean)
    assert corr.mean() > 0.8


def test_confounds2df(tmp_path):
    """Check auto-detect of confonds from an fMRI nii image."""
    conf = lc.Confounds()
    img_nii, _ = create_tmp_filepath(tmp_path, copy_testdata=True)
    conf.load(img_nii)
    assert "trans_x" in conf.confounds_.columns


@pytest.mark.parametrize('strategy',
                         ["string", "error", [0], "motion"])
def test_sanitize_strategy(strategy):
    """Check that flawed strategy options generate meaningful error
    messages."""
    if strategy == "motion":
        conf = lc.Confounds(strategy=[strategy])
        assert "non_steady_state" in conf.strategy
    else:
        with pytest.raises(ValueError):
            lc.Confounds(strategy=[strategy])


SUFFIXES = np.array(["", "_derivative1", "_power2", "_derivative1_power2"])


@pytest.fixture
def expected_suffixes(motion):
    expectation = {"basic": slice(1),
                   "derivatives": slice(2),
                   "power2": np.array([True, False, True, False]),
                   "full": slice(4)}
    return SUFFIXES[expectation[motion]]


@pytest.mark.parametrize('motion',
                         ["basic", "derivatives", "power2", "full"])
@pytest.mark.parametrize('param',
                         ["trans_x", "trans_y", "trans_z",
                          "rot_x", "rot_y", "rot_z"])
def test_motion(tmp_path, motion, param, expected_suffixes):
    img_nii, _ = create_tmp_filepath(tmp_path, copy_testdata=True)
    conf = lc.Confounds(strategy=["motion"], motion=motion)
    conf.load(img_nii)
    for suff in SUFFIXES:
        if suff in expected_suffixes:
            assert f"{param}{suff}" in conf.confounds_.columns
        else:
            assert f"{param}{suff}" not in conf.confounds_.columns


def test_n_compcor(tmp_path):
    img_nii, _ = create_tmp_filepath(tmp_path, copy_testdata=True,
                                       copy_json=True)
    conf = lc.Confounds(strategy=["compcor"], compcor="anat_combined",
                        n_compcor=2)
    conf.load(img_nii)
    assert "a_comp_cor_00" in conf.confounds_.columns
    assert "a_comp_cor_01" in conf.confounds_.columns
    assert "a_comp_cor_02" not in conf.confounds_.columns


@pytest.fixture
def expected_components(n_comp):
    if n_comp == 2:
        return range(1, 3)
    if n_comp == .95:
        return range(1, 12)
    return [1]


@pytest.mark.parametrize('n_comp', [.2, 2, .95, 50])
def test_n_motion(tmp_path, n_comp, expected_components):
    img_nii, _ = create_tmp_filepath(tmp_path, copy_testdata=True,
                                       copy_json=True)
    conf = lc.Confounds(strategy=["motion"], motion="full",
                        n_motion_components=n_comp)
    if n_comp == 50:
        with pytest.raises(ValueError):
            conf.load(img_nii)
    else:
        conf.load(img_nii)
        for i in range(1, 12):
            if i in expected_components:
                assert f"motion_pca_{i}" in conf.confounds_.columns
            else:
                assert f"motion_pca_{i}" not in conf.confounds_.columns


def test_not_found_exception(tmp_path):
    missing_params = ["trans_y", "trans_x_derivative1", "rot_z_power2"]
    missing_keywords = ["cosine"]

    # Create invalid file in temporary dir
    img_missing_confounds, bad_conf = create_tmp_filepath(tmp_path,
                                                            copy_testdata=True,
                                                            copy_json=False)
    leagal_confounds = pd.read_csv(bad_conf, delimiter="\t", encoding="utf-8")
    cosine = [col_name
              for col_name in leagal_confounds.columns
              if "cosine" in col_name]
    aroma = [col_name
             for col_name in leagal_confounds.columns
             if "aroma" in col_name]
    missing_confounds = leagal_confounds.drop(
        columns=missing_params + cosine + aroma)
    missing_confounds.to_csv(bad_conf, sep="\t", index=False)
    conf = lc.Confounds(
        strategy=["high_pass", "motion", "global"],
        global_signal="full",
        motion="full",
    )
    with pytest.raises(ValueError) as exc_info:
        conf.load(img_missing_confounds)
    assert f"{missing_params}" in exc_info.value.args[0]
    assert f"{missing_keywords}" in exc_info.value.args[0]

    # loading anat compcor should also raise an error, because the json file is
    # missing for that example dataset
    with pytest.raises(ValueError):
        conf = lc.Confounds(strategy=["compcor"], compcor="anat_combined")
        conf.load(img_missing_confounds)

    # catch invalid compcor option
    with pytest.raises(KeyError):
        conf = lc.Confounds(strategy=["compcor"], compcor="blah")
        conf.load(file_confounds)

    # Aggressive ICA-AROMA strategy requires
    # default nifti and noise ICs in confound file
    # correct nifti but missing noise regressor
    with pytest.raises(ValueError) as exc_info:
        conf = lc.Confounds(strategy=["ica_aroma"], ica_aroma="basic")
        conf.load(img_missing_confounds)
    assert "aroma" in exc_info.value.args[0]

    # Aggressive ICA-AROMA strategy requires
    # default nifti
    aroma_nii, _ = create_tmp_filepath(tmp_path, image_type="icaaroma",
                                         suffix="aroma")
    with pytest.raises(ValueError) as exc_info:
        conf.load(aroma_nii)
    assert "Invalid file type" in exc_info.value.args[0]

    # non aggressive ICA-AROMA strategy requires
    # desc-smoothAROMAnonaggr nifti file
    with pytest.raises(ValueError) as exc_info:
        conf = lc.Confounds(strategy=["ica_aroma"], ica_aroma="full")
        conf.load(img_missing_confounds)
    assert "desc-smoothAROMAnonaggr_bold" in exc_info.value.args[0]


def test_load_non_nifti(tmp_path):
    """Test non-nifti and invalid file type as input."""
    conf = lc.Confounds()

    # tsv file - unsupported input
    _, tsv = create_tmp_filepath(tmp_path, copy_testdata=True,
                                   copy_json=True)

    with pytest.raises(ValueError):
        conf.load(str(tsv))

    # cifti file should be supported
    cifti, _ = create_tmp_filepath(tmp_path, image_type="cifti",
                                     copy_testdata=True, copy_json=True)
    conf.load(cifti)
    assert conf.confounds_.size != 0

    # gifti support
    gifti, _ = create_tmp_filepath(tmp_path, image_type="gifti",
                                     copy_testdata=True, copy_json=True)
    conf.load(gifti)
    assert conf.confounds_.size != 0


def test_invalid_filetype(tmp_path):
    """Invalid file types/associated files for load method."""

    # invalid fmriprep version: confound file with no header (<1.0)
    conf = lc.Confounds()
    bad_nii, bad_conf = create_tmp_filepath(tmp_path)
    fake_confounds = np.random.rand(30, 20)
    np.savetxt(bad_conf, fake_confounds, delimiter="\t")
    with pytest.raises(ValueError) as error_log:
        conf.load(str(bad_nii))
    assert "The confound file contains no header." in str(error_log.value)

    # invalid fmriprep version: old camel case header (<1.2)
    leagal_confounds, _ = get_leagal_confound()
    camel_confounds = leagal_confounds.copy()
    camel_confounds.columns = [to_camel_case(col_name)
                               for col_name in leagal_confounds.columns]
    camel_confounds.to_csv(bad_conf, sep="\t", index=False)
    with pytest.raises(ValueError) as error_log:
        conf.load(str(bad_nii))
    assert "contains header in camel case." in str(error_log.value)

    # create a empty nifti file with no associated confound file
    # We only need the path to check this
    no_conf = "no_confound_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    no_confound = tmp_path / no_conf
    no_confound.touch()
    with pytest.raises(ValueError):
        conf.load(str(no_confound))


def test_ica_aroma(tmp_path):
    """Test ICA AROMA related file input."""
    aroma_nii, _ = create_tmp_filepath(tmp_path, image_type="icaaroma",
                                         copy_testdata=True)
    regular_nii, _ = create_tmp_filepath(tmp_path, image_type="regular",
                                           copy_testdata=True)
    # Agressive strategy
    conf = lc.Confounds(strategy=["ica_aroma"], ica_aroma="basic")
    conf.load(regular_nii)
    for col_name in conf.confounds_.columns:
        # only aroma and non-steady state columns will be present
        assert re.match("(?:aroma_motion_+|non_steady_state+)", col_name)

    # Non-agressive strategy
    conf = lc.Confounds(strategy=["ica_aroma"], ica_aroma="full")
    conf.load(aroma_nii)
    assert conf.confounds_.size == 0

    # invalid combination of strategy and option
    with pytest.raises(ValueError) as exc_info:
        conf = lc.Confounds(strategy=["ica_aroma"], ica_aroma=None)
        conf.load(regular_nii)
    assert "ICA-AROMA strategy" in exc_info.value.args[0]


def test_sample_mask(tmp_path):
    """Test load method and sample mask."""
    regular_nii, regular_conf = create_tmp_filepath(
        tmp_path, image_type="regular", copy_testdata=True)
    # create a version with srub_mask not applied;
    # This is not recommanded
    conf = lc.Confounds(
        strategy=["motion", "scrub"], scrub="full", fd_thresh=0.15
    )
    reg, mask = conf.load(regular_nii)

    # the current test data has 6 time points marked as motion outliers,
    # and one nonsteady state (overlap with the first motion outlier)
    # 2 time points removed due to the "full" srubbing strategy
    assert reg.shape[0] - len(mask) == 8
    # nilearn requires unmasked confound regressors
    assert reg.shape[0] == 30

    # non steady state will always be removed
    conf = lc.Confounds(strategy=["motion"])
    reg, mask = conf.load(regular_nii)
    assert reg.shape[0] - len(mask) == 1

    # When no non-steady state volumes are present
    conf_data, _ = get_leagal_confound(non_steady_state=False)
    conf_data.to_csv(regular_conf, sep="\t", index=False)
    conf = lc.Confounds(strategy=["motion"])
    reg, mask = conf.load(file_no_none_steady)
    assert mask is None
