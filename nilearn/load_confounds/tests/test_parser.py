import os
import re
import numpy as np
from scipy.stats import pearsonr
from sklearn.preprocessing import scale
import pytest
from nibabel import Nifti1Image
from nilearn.input_data import NiftiMasker
from nilearn.load_confounds import parser as lc


path_data = os.path.join(os.path.dirname(lc.__file__), "data")
file_confounds = os.path.join(
    path_data, "test_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
)
file_no_none_steady = os.path.join(
    path_data, "nonss_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
)


def _simu_img(demean=True):
    """Simulate an nifti image based on confound file with some parts confounds
    and some parts noise."""
    # set the size of the image matrix
    nx = 5
    ny = 5
    # the actual number of slices will actually be double of that
    # as we will stack slices with confounds on top of slices with noise
    nz = 2
    # Load a simple 6 parameters motion models as confounds
    confounds, _ = lc.Confounds(
        strategy=["motion"], motion="basic", demean=demean
    ).load(file_confounds)
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


def _regression(confounds, sample_mask):
    """Simple regression with nilearn."""
    # Simulate data
    img, mask_conf, _, _ = _simu_img(demean=True)
    confounds = np.tile(confounds, (3, 1))  # matching L29 (_simu_img)

    # Do the regression
    masker = NiftiMasker(
        mask_img=mask_conf, standardize=True, sample_mask=sample_mask
    )
    tseries_clean = masker.fit_transform(img, confounds)
    assert tseries_clean.shape[0] == confounds.shape[0]


@pytest.mark.filterwarnings("ignore")
def test_nilearn_regress():
    """Try regressing out all motion types in nilearn."""
    # Regress full motion
    confounds, _ = lc.Confounds(strategy=["motion"], motion="full").load(
        file_confounds
    )
    sample_mask = None  # not testing sample mask here
    _regression(confounds, sample_mask)

    # Regress high_pass
    confounds, _ = lc.Confounds(strategy=["high_pass"]).load(file_confounds)
    _regression(confounds, sample_mask)

    # Regress wm_csf
    confounds, _ = lc.Confounds(strategy=["wm_csf"], wm_csf="full").load(
        file_confounds
    )
    _regression(confounds, sample_mask)
    # Regress global
    confounds, _ = lc.Confounds(
        strategy=["global"], global_signal="full"
    ).load(file_confounds)
    _regression(confounds, sample_mask)

    # Regress AnatCompCor
    confounds, _ = lc.Confounds(strategy=["compcor"], compcor="anat").load(
        file_confounds
    )
    _regression(confounds, sample_mask)

    # Regress TempCompCor
    confounds, _ = lc.Confounds(strategy=["compcor"], compcor="temporal").load(
        file_confounds
    )
    _regression(confounds, sample_mask)

    # Regress ICA-AROMA
    confounds, _ = lc.Confounds(
        strategy=["ica_aroma"], ica_aroma="basic"
    ).load(file_confounds)
    _regression(confounds, sample_mask)


@pytest.mark.filterwarnings("ignore")
def test_nilearn_standardize_false():
    """Test removing confounds in nilearn with no standardization."""
    # Simulate data
    img, mask_conf, mask_rand, X = _simu_img(demean=True)

    # Check that most variance is removed
    # in voxels composed of pure confounds
    tseries_std = _tseries_std(img, mask_conf, X, None, False)
    assert np.mean(tseries_std < 0.0001)

    # Check that most variance is preserved
    # in voxels composed of random noise
    tseries_std = _tseries_std(img, mask_rand, X, None, False)
    assert np.mean(tseries_std > 0.9)


@pytest.mark.filterwarnings("ignore")
def test_nilearn_standardize_zscore():
    """Test removing confounds in nilearn with zscore standardization."""
    # Simulate data

    img, mask_conf, mask_rand, X = _simu_img(demean=True)

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


def test_nilearn_standardize_psc():
    """Test removing confounds in nilearn with psc standardization."""
    # Similar test to test_nilearn_standardize_zscore, but with psc
    # Simulate data

    img, mask_conf, mask_rand, X = _simu_img(demean=False)

    # Areas with confound
    tseries_raw, tseries_clean = _denoise(img, mask_conf, X, None, "psc")
    corr = _corr_tseries(tseries_raw, tseries_clean)
    assert corr.mean() < 0.2

    # Areas with random noise
    tseries_raw, tseries_clean = _denoise(img, mask_rand, X, None, "psc")
    corr = _corr_tseries(tseries_raw, tseries_clean)
    assert corr.mean() > 0.8


def test_confounds2df():
    """Check auto-detect of confonds from an fMRI nii image."""
    conf = lc.Confounds()
    file_confounds_nii = os.path.join(
        path_data, "test_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    )
    conf.load(file_confounds_nii)
    assert "trans_x" in conf.confounds_.columns


def test_sanitize_strategy():
    """Check that flawed strategy options generate meaningful error
    messages."""
    with pytest.raises(ValueError):
        lc.Confounds(strategy="string")

    with pytest.raises(ValueError):
        lc.Confounds(strategy=["error"])

    with pytest.raises(ValueError):
        lc.Confounds(strategy=[0])

    conf = lc.Confounds(strategy=["motion"])
    assert "non_steady_state" in conf.strategy


def test_motion():

    conf_basic = lc.Confounds(strategy=["motion"], motion="basic")
    conf_basic.load(file_confounds)
    conf_derivatives = lc.Confounds(strategy=["motion"], motion="derivatives")
    conf_derivatives.load(file_confounds)
    conf_power2 = lc.Confounds(strategy=["motion"], motion="power2")
    conf_power2.load(file_confounds)
    conf_full = lc.Confounds(strategy=["motion"], motion="full")
    conf_full.load(file_confounds)

    params = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
    for param in params:
        # Basic 6 params motion model
        assert f"{param}" in conf_basic.confounds_.columns
        assert f"{param}_derivative1" not in conf_basic.confounds_.columns
        assert f"{param}_power2" not in conf_basic.confounds_.columns
        assert (
            f"{param}_derivative1_power2" not in conf_basic.confounds_.columns
        )

        # Use a 6 params + derivatives motion model
        assert f"{param}" in conf_derivatives.confounds_.columns
        assert f"{param}_derivative1" in conf_derivatives.confounds_.columns
        assert f"{param}_power2" not in conf_derivatives.confounds_.columns
        assert (
            f"{param}_derivative1_power2"
            not in conf_derivatives.confounds_.columns
        )

        # Use a 6 params + power2 motion model
        assert f"{param}" in conf_power2.confounds_.columns
        assert f"{param}_derivative1" not in conf_power2.confounds_.columns
        assert f"{param}_power2" in conf_power2.confounds_.columns
        assert (
            f"{param}_derivative1_power2" not in conf_power2.confounds_.columns
        )

        # Use a 6 params + derivatives + power2 + power2d derivatives motion
        # model
        assert f"{param}" in conf_full.confounds_.columns
        assert f"{param}_derivative1" in conf_full.confounds_.columns
        assert f"{param}_power2" in conf_full.confounds_.columns
        assert f"{param}_derivative1_power2" in conf_full.confounds_.columns


def test_n_compcor():

    conf = lc.Confounds(strategy=["compcor"], compcor="anat", n_compcor=2)
    conf.load(file_confounds)
    assert "a_comp_cor_00" in conf.confounds_.columns
    assert "a_comp_cor_01" in conf.confounds_.columns
    assert "a_comp_cor_02" not in conf.confounds_.columns


def test_n_motion():

    conf = lc.Confounds(strategy=["motion"], motion="full",
                        n_motion_components=0.2)
    conf.load(file_confounds)
    assert "motion_pca_1" in conf.confounds_.columns
    assert "motion_pca_2" not in conf.confounds_.columns

    conf = lc.Confounds(strategy=["motion"], motion="full",
                        n_motion_components=2)
    conf.load(file_confounds)
    assert "motion_pca_1" in conf.confounds_.columns
    assert "motion_pca_3" not in conf.confounds_.columns

    conf = lc.Confounds(strategy=["motion"], motion="full",
                        n_motion_components=0.95)
    conf.load(file_confounds)
    assert "motion_pca_6" in conf.confounds_.columns

    with pytest.raises(ValueError):
        conf = lc.Confounds(strategy=["motion"], motion="full",
                            n_motion_components=50)
        conf.load(file_confounds)


def test_not_found_exception():

    conf = lc.Confounds(
        strategy=["high_pass", "motion", "global"],
        global_signal="full",
        motion="full",
    )

    missing_params = ["trans_y", "trans_x_derivative1", "rot_z_power2"]
    missing_keywords = ["cosine"]

    file_missing_confounds = os.path.join(
        path_data, "missing_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    )

    with pytest.raises(ValueError) as exc_info:
        conf.load(file_missing_confounds)
    assert f"{missing_params}" in exc_info.value.args[0]
    assert f"{missing_keywords}" in exc_info.value.args[0]

    # loading anat compcor should also raise an error, because the json file is
    # missing for that example dataset
    with pytest.raises(ValueError):
        conf = lc.Confounds(strategy=["compcor"], compcor="anat")
        conf.load(file_missing_confounds)

    # catch invalid compcor option
    with pytest.raises(KeyError):
        conf = lc.Confounds(strategy=["compcor"], compcor="blah")
        conf.load(file_confounds)

    # catch invalid compcor option
    with pytest.raises(ValueError):
        conf = lc.Confounds(
            strategy=["compcor"], compcor="full", acompcor_combined=None
        )
        conf.load(file_confounds)

    # Aggressive ICA-AROMA strategy requires
    # default nifti and noise ICs in confound file
    # correct nifti but missing noise regressor
    with pytest.raises(ValueError) as exc_info:
        conf = lc.Confounds(strategy=["ica_aroma"], ica_aroma="basic")
        conf.load(file_missing_confounds)
    assert "aroma" in exc_info.value.args[0]

    # Aggressive ICA-AROMA strategy requires
    # default nifti
    aroma_nii = os.path.join(
        path_data,
        "test_space-MNI152NLin2009cAsym_desc-smoothAROMAnonaggr_bold.nii.gz",
    )
    with pytest.raises(ValueError) as exc_info:
        conf.load(aroma_nii)
    assert "Invalid file type" in exc_info.value.args[0]

    # non aggressive ICA-AROMA strategy requires
    # desc-smoothAROMAnonaggr nifti file
    with pytest.raises(ValueError) as exc_info:
        conf = lc.Confounds(strategy=["ica_aroma"], ica_aroma="full")
        conf.load(file_missing_confounds)
    assert "desc-smoothAROMAnonaggr_bold" in exc_info.value.args[0]


def test_load_non_nifti():
    """Test non-nifti and invalid file type as input."""
    conf = lc.Confounds()

    # tsv file - unsupported input
    tsv = os.path.join(path_data, "test_desc-confounds_regressors.tsv")
    with pytest.raises(ValueError):
        conf.load(tsv)

    # cifti file should be supported
    cifti = os.path.join(
        path_data, "test_space-fsLR_den-91k_bold.dtseries.nii"
    )
    conf.load(cifti)
    assert conf.confounds_.size != 0

    # gifti support
    gifti = [
        os.path.join(
            path_data, f"test_space-fsaverage5_hemi-{hemi}_bold.func.gii"
        )
        for hemi in ["L", "R"]
    ]
    conf.load(gifti)
    assert conf.confounds_.size != 0


def test_invalid_filetype():
    """Invalid file types/associated files for load method."""
    # invalid fmriprep version: contain confound files before and after v20.2.0
    conf = lc.Confounds()

    invalid_ver = os.path.join(
        path_data, "invalid_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    )
    with pytest.raises(ValueError):
        conf.load(invalid_ver)

    # nifti with no associated confound file
    no_confound = os.path.join(
        path_data,
        "noconfound_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
    )
    with pytest.raises(ValueError):
        conf.load(no_confound)


def test_ica_aroma():
    """Test ICA AROMA related file input."""
    aroma_nii = os.path.join(
        path_data,
        "test_space-MNI152NLin2009cAsym_desc-smoothAROMAnonaggr_bold.nii.gz",
    )
    # Agressive strategy
    conf = lc.Confounds(strategy=["ica_aroma"], ica_aroma="basic")
    conf.load(file_confounds)
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
        conf.load(file_confounds)
    assert "ICA-AROMA strategy" in exc_info.value.args[0]


def test_sample_mask():
    """Test load method and sample mask."""
    # create a version with srub_mask not applied;
    # This is not recommanded
    conf = lc.Confounds(
        strategy=["motion", "scrub"], scrub="full", fd_thresh=0.15
    )
    reg, mask = conf.load(file_confounds)

    # the current test data has 6 time points marked as motion outliers,
    # and one nonsteady state (overlap with the first motion outlier)
    # 2 time points removed due to the "full" srubbing strategy
    assert reg.shape[0] - len(mask) == 8
    # nilearn requires unmasked confound regressors
    assert reg.shape[0] == 30

    # non steady state will always be removed
    conf = lc.Confounds(strategy=["motion"])
    reg, mask = conf.load(file_confounds)
    assert reg.shape[0] - len(mask) == 1

    # When no non-steady state volumes are present
    conf = lc.Confounds(strategy=["motion"])
    reg, mask = conf.load(file_no_none_steady)
    assert mask is None
