"""Test the nifti_region module.

Functions in this file only test features added by the NiftiLabelsMasker class,
not the underlying functions (clean(), img_to_signals_labels(), etc.). See
test_masking.py and test_signal.py for details.
"""

from copy import deepcopy
from typing import Union

import numpy as np
import pandas as pd
import pytest
from nibabel import Nifti1Image
from numpy.testing import assert_almost_equal, assert_array_equal
from sklearn.utils.estimator_checks import parametrize_with_checks

from nilearn._utils.data_gen import (
    generate_labeled_regions,
    generate_random_img,
)
from nilearn._utils.estimator_checks import (
    check_estimator,
    nilearn_check_estimator,
    return_expected_failed_checks,
)
from nilearn._utils.tags import SKLEARN_LT_1_6
from nilearn.conftest import _img_labels
from nilearn.image import get_data
from nilearn.maskers import NiftiLabelsMasker, NiftiMasker

ESTIMATORS_TO_CHECK = [NiftiLabelsMasker()]

if SKLEARN_LT_1_6:

    @pytest.mark.parametrize(
        "estimator, check, name",
        check_estimator(estimators=ESTIMATORS_TO_CHECK),
    )
    def test_check_estimator_sklearn_valid(estimator, check, name):  # noqa: ARG001
        """Check compliance with sklearn estimators."""
        check(estimator)

    @pytest.mark.xfail(reason="invalid checks should fail")
    @pytest.mark.parametrize(
        "estimator, check, name",
        check_estimator(estimators=ESTIMATORS_TO_CHECK, valid=False),
    )
    def test_check_estimator_sklearn_invalid(estimator, check, name):  # noqa: ARG001
        """Check compliance with sklearn estimators."""
        check(estimator)

else:

    @parametrize_with_checks(
        estimators=ESTIMATORS_TO_CHECK,
        expected_failed_checks=return_expected_failed_checks,
    )
    def test_check_estimator_sklearn(estimator, check):
        """Check compliance with sklearn estimators."""
        check(estimator)


@pytest.mark.timeout(0)
@pytest.mark.parametrize(
    "estimator, check, name",
    nilearn_check_estimator(
        estimators=[NiftiLabelsMasker(labels_img=_img_labels())]
    ),
)
def test_check_estimator_nilearn(estimator, check, name):  # noqa: ARG001
    """Check compliance with sklearn estimators."""
    check(estimator)


def test_nifti_labels_masker(
    affine_eye, shape_3d_default, n_regions, length, img_labels
):
    """Check working of shape/affine checks."""
    shape1 = (*shape_3d_default, length)

    fmri_img, mask11_img = generate_random_img(
        shape1,
        affine=affine_eye,
    )

    # No exception raised here
    masker = NiftiLabelsMasker(img_labels, resampling_target=None)
    signals = masker.fit_transform(fmri_img)

    assert signals.shape == (length, n_regions)

    # No exception should be raised either
    masker = NiftiLabelsMasker(img_labels, resampling_target=None)

    masker.fit()

    # Check attributes defined at fit
    assert masker.n_elements_ == n_regions

    # now with mask_img
    masker = NiftiLabelsMasker(
        img_labels, mask_img=mask11_img, resampling_target=None
    )
    signals = masker.fit_transform(fmri_img)

    assert signals.shape == (length, n_regions)


def test_nifti_labels_masker_errors(
    affine_eye, shape_3d_default, n_regions, length
):
    """Check working of shape/affine checks."""
    masker = NiftiLabelsMasker()
    with pytest.raises(TypeError, match="input should be a NiftiLike object"):
        masker.fit()

    shape1 = (*shape_3d_default, length)

    shape2 = (12, 10, 14, length)
    affine2 = np.diag((1, 2, 3, 1))

    fmri12_img, mask12_img = generate_random_img(shape1, affine=affine2)
    fmri21_img, mask21_img = generate_random_img(shape2, affine=affine_eye)

    labels11_img = generate_labeled_regions(
        shape1[:3],
        affine=affine_eye,
        n_regions=n_regions,
    )

    # check exception when transform() called without prior fit()
    masker11 = NiftiLabelsMasker(labels11_img, resampling_target=None)

    # Test all kinds of mismatch between shapes and between affines
    masker11.fit()
    with pytest.raises(
        ValueError, match="Images have different affine matrices."
    ):
        masker11.transform(fmri12_img)
    with pytest.raises(ValueError, match="Images have incompatible shapes."):
        masker11.transform(fmri21_img)

    masker11 = NiftiLabelsMasker(
        labels11_img, mask_img=mask12_img, resampling_target=None
    )
    with pytest.raises(
        ValueError, match="Following field of view errors were detected"
    ):
        masker11.fit()

    masker11 = NiftiLabelsMasker(
        labels11_img, mask_img=mask21_img, resampling_target=None
    )
    with pytest.raises(
        ValueError, match="Following field of view errors were detected"
    ):
        masker11.fit()


def test_nifti_labels_masker_with_nans_and_infs(
    affine_eye, shape_3d_default, n_regions, length, img_labels
):
    """Deal with NaNs and infs in label image.

    The masker should replace those NaNs and infs with zeros,
    while raising a warning.
    """
    fmri_img, mask_img = generate_random_img(
        (*shape_3d_default, length),
        affine=affine_eye,
    )

    # Introduce nans with data type float
    # See issue: https://github.com/nilearn/nilearn/issues/2580
    data = get_data(img_labels).astype(np.float32)
    data[:, :, 7] = np.nan
    data[:, :, 4] = np.inf
    img_labels = Nifti1Image(data, affine_eye)

    masker = NiftiLabelsMasker(img_labels, mask_img=mask_img)

    with pytest.warns(UserWarning, match="Non-finite values detected."):
        sig = masker.fit_transform(fmri_img)

    assert sig.shape == (length, n_regions)
    assert np.all(np.isfinite(sig))


def test_nifti_labels_masker_with_nans_and_infs_in_data(
    affine_eye, shape_3d_default, n_regions, length, img_labels
):
    """Apply a NiftiLabelsMasker to 4D data containing NaNs and infs.

    The masker should replace those NaNs and infs with zeros,
    while raising a warning.
    """
    fmri_img, mask_img = generate_random_img(
        (*shape_3d_default, length),
        affine=affine_eye,
    )

    # Introduce nans with data type float
    # See issues:
    # - https://github.com/nilearn/nilearn/issues/2580 (why floats)
    # - https://github.com/nilearn/nilearn/issues/2711 (why test)
    fmri_data = get_data(fmri_img).astype(np.float32)
    fmri_data[:, :, 7, :] = np.nan
    fmri_data[:, :, 4, 0] = np.inf
    fmri_img = Nifti1Image(fmri_data, affine_eye)

    masker = NiftiLabelsMasker(img_labels, mask_img=mask_img)

    with pytest.warns(UserWarning, match="Non-finite values detected."):
        sig = masker.fit_transform(fmri_img)

    assert sig.shape == (length, n_regions)
    assert np.all(np.isfinite(sig))


@pytest.mark.parametrize(
    "strategy, function",
    [
        ("mean", np.mean),
        ("median", np.median),
        ("sum", np.sum),
        ("minimum", np.min),
        ("maximum", np.max),
        ("standard_deviation", np.std),
        ("variance", np.var),
    ],
)
def test_nifti_labels_masker_reduction_strategies(
    affine_eye, strategy, function
):
    """Tests NiftiLabelsMasker strategies.

    1. whether the usage of different reduction strategies work.
    2. whether unrecognized strategies raise a ValueError
    3. whether the default option is backwards compatible (calls "mean")
    """
    test_values = [-2.0, -1.0, 0.0, 1.0, 2]

    img_data = np.array([[test_values, test_values]])

    labels_data = np.array([[[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]]], dtype=np.int8)

    img = Nifti1Image(img_data, affine_eye)
    labels = Nifti1Image(labels_data, affine_eye)

    # What NiftiLabelsMasker should return for each reduction strategy?
    expected_result = function(test_values)

    masker = NiftiLabelsMasker(labels, strategy=strategy)
    # Here passing [img] within a list because it's a 3D object.
    result = masker.fit_transform([img]).squeeze()

    assert result == expected_result

    default_masker = NiftiLabelsMasker(labels)

    assert default_masker.strategy == "mean"


def test_nifti_labels_masker_reduction_strategies_error(affine_eye):
    """Tests NiftiLabelsMasker invalid strategy."""
    labels_data = np.array([[[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]]], dtype=np.int8)

    labels = Nifti1Image(labels_data, affine_eye)

    with pytest.raises(ValueError, match="Invalid strategy 'TESTRAISE'"):
        masker = NiftiLabelsMasker(labels, strategy="TESTRAISE")
        masker.fit()


def test_nifti_labels_masker_resampling_errors(img_labels):
    """Test errors of resampling in NiftiLabelsMasker."""
    with pytest.raises(
        ValueError,
        match="invalid value for 'resampling_target' parameter: mask",
    ):
        masker = NiftiLabelsMasker(img_labels, resampling_target="mask")
        masker.fit()

    with pytest.raises(
        ValueError,
        match="invalid value for 'resampling_target' parameter: invalid",
    ):
        masker = NiftiLabelsMasker(
            img_labels,
            resampling_target="invalid",
        )
        masker.fit()


def test_nifti_labels_masker_resampling_to_data(affine_eye, n_regions, length):
    """Test resampling to data in NiftiLabelsMasker."""
    # mask
    shape2 = (8, 9, 10, length)
    # maps
    shape3 = (16, 18, 20)

    _, mask_img = generate_random_img(shape2, affine=affine_eye)

    labels_img = generate_labeled_regions(shape3, n_regions, affine=affine_eye)

    # Test with data and atlas of different shape:
    # the atlas should be resampled to the data
    shape22 = (5, 5, 6, length)
    affine2 = 2 * affine_eye
    affine2[-1, -1] = 1

    fmri_img, _ = generate_random_img(shape22, affine=affine2)

    masker = NiftiLabelsMasker(
        labels_img, mask_img=mask_img, resampling_target="data"
    )
    masker.fit_transform(fmri_img)

    assert_array_equal(masker.labels_img_.affine, affine2)


@pytest.mark.parametrize("resampling_target", ["data", "labels"])
def test_nifti_labels_masker_resampling(
    affine_eye,
    shape_3d_default,
    resampling_target,
    length,
    img_labels,
):
    """Test to return resampled labels having number of labels \
       equal with transformed shape of 2nd dimension.

    See https://github.com/nilearn/nilearn/issues/1673
    """
    shape = (*shape_3d_default, length)
    affine = 2 * affine_eye

    fmri_img, _ = generate_random_img(shape, affine=affine)

    masker = NiftiLabelsMasker(
        labels_img=img_labels, resampling_target=resampling_target
    )
    if resampling_target == "data":
        with pytest.warns(
            UserWarning,
            match="After resampling the label image "
            "to the data image, the following "
            "labels were removed",
        ):
            signals = masker.fit_transform(fmri_img)
    else:
        signals = masker.fit_transform(fmri_img)

    resampled_labels_img = masker.labels_img_
    n_resampled_labels = len(np.unique(get_data(resampled_labels_img)))

    assert n_resampled_labels - 1 == signals.shape[1]

    # inverse transform
    compressed_img = masker.inverse_transform(signals)

    # Test that compressing the image a second time should yield an image
    # with the same data as compressed_img.
    signals2 = masker.fit_transform(fmri_img)
    # inverse transform again
    compressed_img2 = masker.inverse_transform(signals2)

    assert_array_equal(get_data(compressed_img), get_data(compressed_img2))


def test_nifti_labels_masker_resampling_to_labels(
    affine_eye, shape_3d_default, n_regions, length
):
    """Test resampling to labels in NiftiLabelsMasker."""
    # fmri
    shape1 = (*shape_3d_default, length)
    # mask
    shape2 = (16, 17, 18, length)
    # labels
    shape3 = (13, 14, 15)

    # With data of the same affine
    fmri_img, _ = generate_random_img(
        shape1,
        affine=affine_eye,
    )
    _, mask_img = generate_random_img(shape2, affine=affine_eye)

    labels_img = generate_labeled_regions(shape3, n_regions, affine=affine_eye)

    masker = NiftiLabelsMasker(
        labels_img, mask_img=mask_img, resampling_target="labels"
    )

    signals = masker.fit_transform(fmri_img)

    assert_almost_equal(masker.labels_img_.affine, labels_img.affine)
    assert masker.labels_img_.shape == labels_img.shape
    assert_almost_equal(masker.mask_img_.affine, masker.labels_img_.affine)
    assert masker.mask_img_.shape == masker.labels_img_.shape[:3]

    assert signals.shape == (length, n_regions)

    fmri11_img_r = masker.inverse_transform(signals)

    assert_almost_equal(fmri11_img_r.affine, masker.labels_img_.affine)
    assert fmri11_img_r.shape == (masker.labels_img_.shape[:3] + (length,))


def test_nifti_labels_masker_resampling_to_clipped_labels(
    affine_eye, shape_3d_default, n_regions, length
):
    """Test with clipped labels.

    Mask does not contain all labels.

    Shapes do matter in that case,
    because there is some resampling taking place.
    """
    # fmri
    shape1 = (*shape_3d_default, length)
    # mask
    shape2 = (8, 9, 10, length)
    # maps
    shape3 = (16, 18, 20)

    fmri11_img, _ = generate_random_img(shape1, affine=affine_eye)
    _, mask22_img = generate_random_img(shape2, affine=affine_eye)

    labels33_img = generate_labeled_regions(
        shape3, n_regions, affine=affine_eye
    )

    masker = NiftiLabelsMasker(
        labels33_img, mask_img=mask22_img, resampling_target="labels"
    )

    signals = masker.fit_transform(fmri11_img)

    assert_almost_equal(masker.labels_img_.affine, labels33_img.affine)

    assert masker.labels_img_.shape == labels33_img.shape
    assert_almost_equal(masker.mask_img_.affine, masker.labels_img_.affine)
    assert masker.mask_img_.shape == masker.labels_img_.shape[:3]

    uniq_labels = np.unique(get_data(masker.labels_img_))
    assert uniq_labels[0] == 0
    assert len(uniq_labels) - 1 == n_regions

    assert signals.shape == (length, n_regions)
    # Some regions have been clipped. Resulting signal must be zero
    assert (signals.var(axis=0) == 0).sum() < n_regions

    fmri11_img_r = masker.inverse_transform(signals)

    assert_almost_equal(fmri11_img_r.affine, masker.labels_img_.affine)
    assert fmri11_img_r.shape == (masker.labels_img_.shape[:3] + (length,))


def test_nifti_labels_masker_resampling_to_none(
    affine_eye, length, shape_3d_default, img_labels
):
    """Test resampling to None in NiftiLabelsMasker.

    All inputs must have same affine to avoid errors.
    """
    fmri_img, mask_img = generate_random_img(
        shape=(*shape_3d_default, length),
        affine=affine_eye,
    )

    masker = NiftiLabelsMasker(
        img_labels, mask_img=mask_img, resampling_target=None
    )
    masker.fit_transform(fmri_img)

    fmri_img, _ = generate_random_img(
        (*shape_3d_default, length),
        affine=affine_eye * 2,
    )
    masker = NiftiLabelsMasker(
        img_labels, mask_img=mask_img, resampling_target=None
    )
    with pytest.raises(
        ValueError, match="Following field of view errors were detected"
    ):
        masker.fit_transform(fmri_img)


def test_standardization(rng, affine_eye, shape_3d_default, img_labels):
    """Check output properly standardized with 'standardize' parameter."""
    n_samples = 400

    signals = rng.standard_normal(size=(np.prod(shape_3d_default), n_samples))
    means = (
        rng.standard_normal(size=(np.prod(shape_3d_default), 1)) * 50 + 1000
    )
    signals += means
    img = Nifti1Image(
        signals.reshape((*shape_3d_default, n_samples)), affine_eye
    )

    # Unstandarized
    masker = NiftiLabelsMasker(img_labels, standardize=False)
    unstandarized_label_signals = masker.fit_transform(img)

    # z-score
    masker = NiftiLabelsMasker(img_labels, standardize="zscore_sample")
    trans_signals = masker.fit_transform(img)

    assert_almost_equal(trans_signals.mean(0), 0)
    assert_almost_equal(trans_signals.std(0), 1, decimal=3)

    # psc
    masker = NiftiLabelsMasker(img_labels, standardize="psc")
    trans_signals = masker.fit_transform(img)

    assert_almost_equal(trans_signals.mean(0), 0)
    assert_almost_equal(
        trans_signals,
        (
            unstandarized_label_signals
            / unstandarized_label_signals.mean(0)
            * 100
            - 100
        ),
    )


def test_nifti_labels_masker_with_mask(
    shape_3d_default, affine_eye, length, img_labels
):
    """Test NiftiLabelsMasker with a separate mask_img parameter."""
    shape = (*shape_3d_default, length)
    fmri_img, mask_img = generate_random_img(shape, affine=affine_eye)

    masker = NiftiLabelsMasker(
        img_labels, resampling_target=None, mask_img=mask_img
    )
    signals = masker.fit_transform(fmri_img)

    bg_masker = NiftiMasker(mask_img)
    tmp = bg_masker.fit_transform(img_labels)
    masked_labels = bg_masker.inverse_transform(tmp)

    masked_masker = NiftiLabelsMasker(
        masked_labels, resampling_target=None, mask_img=mask_img
    )
    masked_signals = masked_masker.fit_transform(fmri_img)

    assert np.allclose(signals, masked_signals)

    #  masker.region_atlas_ should be the same as the masked_labels
    # masked_labels is a 3D image with shape (10,10,10)
    masked_labels_data = get_data(masked_labels)[:, :, :]
    assert np.allclose(get_data(masker.region_atlas_), masked_labels_data)


def generate_labels(n_regions: int, background: str = ""):
    """Create list of strings to use as labels."""
    labels = []
    if background:
        labels.append(background)
    labels.extend([f"region_{i + 1!s}" for i in range(n_regions)])
    return labels


def generate_expected_lut(region_names: list[str]):
    """Generate a look up table based on a list of regions names."""
    if "background" in region_names:
        idx = region_names.index("background")
        region_names[idx] = "Background"
    return pd.DataFrame(
        {"name": region_names, "index": list(range(len(region_names)))}
    )


def check_region_names_after_fit(
    masker: NiftiLabelsMasker,
    signals: np.ndarray,
    region_names: list[str],
    background: Union[str, None],
    resampling: bool = False,
):
    """Perform several checks on the expected attributes of the masker.

    Parameters
    ----------
        masker: NiftiLabelsMasker

        signals: np.ndarray
            output of fit_transfrom from the masker

        region_names: list[str]
            list of regions names expected after fit

        background: str | None
            if not None and present in region_names
            it will be removed from region_names
            to before checking the fitted content of the masker

        resampling: bool
            if some resampling was done
            some checks are skipped as some regions may have been dropped

    - region_names_ does not include background
      should have same length as signals
    - region_ids_ does include background
    - region_names_ should be the same as the region names
      passed to the masker minus that for "background"
    """
    n_regions = signals.shape[0] if signals.ndim == 1 else signals.shape[1]

    assert len(masker.region_names_) == n_regions
    assert len(list(masker.region_ids_.items())) == n_regions + 1

    # resampling may drop some labels so we do not check the region names
    # in this case
    if not resampling:
        region_names_after_fit = [
            masker.region_names_[i] for i in masker.region_names_
        ]
        region_names_after_fit.sort()
        region_names.sort()
        if background:
            region_names = deepcopy(region_names)
            region_names.pop(region_names.index(background))
        assert region_names_after_fit == region_names


def check_lut(masker: NiftiLabelsMasker, expected_lut: pd.DataFrame):
    """Check content of the look up table."""
    if isinstance(masker.lut, pd.DataFrame):
        assert list(masker.lut.columns) == list(masker.lut_.columns)
    assert masker.background_label in masker.lut_["index"].to_list()
    assert "Background" in masker.lut_["name"].to_list()
    pd.testing.assert_series_equal(
        masker.lut_["index"], expected_lut["index"], check_dtype=False
    )
    pd.testing.assert_series_equal(
        masker.lut_["name"], expected_lut["name"], check_dtype=False
    )


def check_region_names_ids_match_after_fit(
    masker: NiftiLabelsMasker, region_names: list[str], region_ids, background
):
    """Check the region names and ids correspondence.

    Check that the same region names and ids correspond to each other
    after fit by comparing with before fit.
    """
    # region_ids includes background, so we make
    # sure that the region_names also include it
    region_names = deepcopy(region_names)
    if not background:
        region_names.insert(0, "background")
    # if they don't have the same length, we can't compare them
    if len(region_names) == len(region_ids):
        region_id_names = {
            region_id: region_names[i]
            for i, region_id in enumerate(region_ids)
        }
        for key, region_name in masker.region_names_.items():
            assert region_id_names[masker.region_ids_[key]] == region_name


def test_regions_id_names_no_labels_no_lut(affine_eye, shape_3d_default):
    """Check match between id and names.

    When no label, no lut: names are inferred from id

    Regression test for https://github.com/nilearn/nilearn/issues/5542
    and https://github.com/nilearn/nilearn/issues/5544
    """
    atlas = np.zeros((8, 8, 8))
    atlas[4, 4, 3:5] = 1
    atlas[4, 4, 5:7] = 2
    atlas = Nifti1Image(atlas, affine_eye)

    masker = NiftiLabelsMasker(atlas)

    fmri_img, _ = generate_random_img(shape_3d_default, affine=affine_eye)
    signals = masker.fit_transform(fmri_img)

    expected_region_ids_ = {"background": 0.0, 0: 1.0, 1: 2.0}
    assert masker.region_ids_ == expected_region_ids_

    assert masker.region_names_ == {0: "1.0", 1: "2.0"}

    # Background is not returned by 'region_names_'
    # but has been added internally
    region_names = ["Background", "1.0", "2.0"]
    region_ids = [0.0, 1.0, 2.0]
    check_region_names_ids_match_after_fit(
        masker, region_names, region_ids, "Background"
    )
    check_region_names_after_fit(masker, signals, region_names, "Background")

    expected_lut = pd.DataFrame(
        columns=["index", "name"],
        data=[[0.0, "Background"], [1.0, "1.0"], [2.0, "2.0"]],
    )
    check_lut(masker, expected_lut)


@pytest.mark.parametrize("Background", [False, True])
def test_regions_id_names_with_labels(
    affine_eye, Background, shape_3d_default
):
    """Check match between id and names.

    Regression test for https://github.com/nilearn/nilearn/issues/5542
    and https://github.com/nilearn/nilearn/issues/5544
    """
    atlas = np.zeros((8, 8, 8))
    atlas[4, 4, 3:5] = 1
    atlas[4, 4, 5:7] = 2
    atlas = Nifti1Image(atlas, affine_eye)

    labels = ["Background", "A", "B"] if Background else ["A", "B"]
    masker = NiftiLabelsMasker(atlas, labels=labels)

    fmri_img, _ = generate_random_img(shape_3d_default, affine=affine_eye)
    signals = masker.fit_transform(fmri_img)

    expected_region_ids_ = {"background": 0.0, 0: 1.0, 1: 2.0}
    assert masker.region_ids_ == expected_region_ids_
    assert masker.region_names_ == {0: "A", 1: "B"}

    # Background is not returned by 'region_names_'
    # but has been added internally even if it was not passed
    region_names = ["Background", "A", "B"]
    region_ids = [0.0, 1.0, 2.0]
    check_region_names_ids_match_after_fit(
        masker, region_names, region_ids, "Background"
    )
    check_region_names_after_fit(masker, signals, region_names, "Background")

    expected_lut = pd.DataFrame(
        columns=["index", "name"],
        data=[[0.0, "Background"], [1.0, "A"], [2.0, "B"]],
    )
    check_lut(masker, expected_lut)


def test_regions_id_names_with_too_few_labels(affine_eye):
    """Check match between id and names when too few labels are passed."""
    atlas = np.zeros((8, 8, 8))
    atlas[4, 4, 3:5] = 1
    atlas[4, 4, 5:7] = 6
    atlas[4, 3, 5:7] = 10
    atlas = Nifti1Image(atlas, affine_eye)

    with pytest.warns(UserWarning, match="Too many indices for the names."):
        # label for 3rd region was not passed so we should get a warning
        masker = NiftiLabelsMasker(
            atlas, labels=["Background", "A", "B"]
        ).fit()

    expected_region_ids_ = {"background": 0.0, 0: 1.0, 1: 6.0, 2: 10.0}
    assert masker.region_ids_ == expected_region_ids_

    assert masker.region_names_ == {0: "A", 1: "B", 2: "unknown"}

    expected_lut = pd.DataFrame(
        columns=["index", "name"],
        data=[[0.0, "Background"], [1.0, "A"], [6.0, "B"], [10.0, "unknown"]],
    )
    check_lut(masker, expected_lut)


def test_regions_id_names_lut(affine_eye, shape_3d_default):
    """Check match between id and names.

    Regression test for https://github.com/nilearn/nilearn/issues/5542
    and https://github.com/nilearn/nilearn/issues/5544
    """
    atlas = np.zeros((8, 8, 8))
    atlas[4, 4, 3:5] = 1
    atlas[4, 4, 5:7] = 2
    atlas = Nifti1Image(atlas, affine_eye)

    # note that regions do not have to be in the right order
    lut = pd.DataFrame(
        columns=["index", "name"],
        data=[[2.0, "B"], [1.0, "A"]],
    )

    masker = NiftiLabelsMasker(atlas, lut=lut)

    fmri_img, _ = generate_random_img(shape_3d_default, affine=affine_eye)
    signals = masker.fit_transform(fmri_img)

    expected_region_ids_ = {"background": 0.0, 0: 1.0, 1: 2.0}
    assert masker.region_ids_ == expected_region_ids_
    assert masker.region_names_ == {0: "A", 1: "B"}

    # Background is not returned by 'region_names_'
    # but has been added internally even if it was not passed
    region_names = ["Background", "A", "B"]
    region_ids = [0.0, 1.0, 2.0]
    check_region_names_ids_match_after_fit(
        masker, region_names, region_ids, "Background"
    )
    check_region_names_after_fit(masker, signals, region_names, "Background")

    # fitted lut now includes background
    expected_lut = pd.DataFrame(
        columns=["index", "name"],
        data=[
            [0.0, "Background"],
            [1.0, "A"],
            [2.0, "B"],
        ],
    )
    check_lut(masker, expected_lut)


def test_regions_id_names_lut_too_few(affine_eye, shape_3d_default):
    """Check passing LUT with too few entries."""
    atlas = np.zeros((8, 8, 8))
    atlas[4, 4, 3:5] = 1
    atlas[4, 4, 5:7] = 6
    atlas[4, 3, 5:7] = 10
    atlas = Nifti1Image(atlas, affine_eye)

    n_expected_regions = 3

    # Too few entries
    # we are skipping region with index 6
    lut = pd.DataFrame(
        columns=["index", "name"],
        data=[[1.0, "A"], [10.0, "B"]],
    )

    masker = NiftiLabelsMasker(atlas, lut=lut).fit()

    expected_region_ids_ = {"background": 0.0, 0: 1.0, 1: 6.0, 2: 10.0}
    assert masker.region_ids_ == expected_region_ids_

    assert masker.region_names_ == {0: "A", 1: "unknown", 2: "B"}

    expected_lut = pd.DataFrame(
        columns=["index", "name"],
        data=[[0.0, "Background"], [1.0, "A"], [6.0, "unknown"], [10.0, "B"]],
    )
    check_lut(masker, expected_lut)

    fmri_img, _ = generate_random_img(shape_3d_default, affine=affine_eye)
    signals = masker.fit_transform(fmri_img)

    assert signals.shape[0] == masker.n_elements_ == n_expected_regions

    assert len(masker.region_names_) == n_expected_regions

    # Background is included in list of labels
    n_expected_labels = n_expected_regions + 1
    assert len(masker.labels_) == n_expected_labels
    assert len(masker.region_ids_) == n_expected_labels


def test_regions_id_names_lut_too_many_entries(affine_eye):
    """Check passing LUT with too many entries.

    The extra regions won't appear in region_ids_ or region_names_
    or in the fitted lut_
    (same behavior as when passing too many labels).
    """
    atlas = np.zeros((8, 8, 8))
    atlas[4, 4, 3:5] = 1
    atlas[4, 4, 5:7] = 6
    atlas[4, 3, 5:7] = 10
    atlas = Nifti1Image(atlas, affine_eye)

    lut = pd.DataFrame(
        columns=["index", "name"],
        data=[[1.0, "A"], [6.0, "C"], [10.0, "B"], [2.0, "missing region"]],
    )

    masker = NiftiLabelsMasker(atlas, lut=lut).fit()

    expected_region_ids_ = {"background": 0.0, 0: 1.0, 1: 6.0, 2: 10.0}
    assert masker.region_ids_ == expected_region_ids_

    assert masker.region_names_ == {0: "A", 1: "C", 2: "B"}

    # fitted lut keeps track of background
    # but the missing regions was dropped
    expected_lut = pd.DataFrame(
        columns=["index", "name"],
        data=[[0.0, "Background"], [1.0, "A"], [6.0, "C"], [10.0, "B"]],
    )
    check_lut(masker, expected_lut)


@pytest.mark.parametrize(
    "background",
    [None, "background", "Background"],
)
def test_warning_n_labels_not_equal_n_regions(
    shape_3d_default, affine_eye, background, n_regions
):
    """Check that n_labels provided match n_regions in image."""
    labels_img = generate_labeled_regions(
        shape_3d_default[:3],
        affine=affine_eye,
        n_regions=n_regions,
    )
    region_names = generate_labels(n_regions + 2, background=background)
    with pytest.warns(
        UserWarning,
        match="Too many names for the indices. Dropping excess names values.",
    ):
        masker = NiftiLabelsMasker(
            labels_img,
            labels=region_names,
        )
        masker.fit()


def test_check_labels_errors(shape_3d_default, affine_eye):
    """Check that invalid types for labels are caught at fit time."""
    labels_img = generate_labeled_regions(
        shape_3d_default, affine=affine_eye, n_regions=2
    )

    with pytest.raises(TypeError, match="'labels' must be a list."):
        NiftiLabelsMasker(labels_img, labels={"foo", "bar", "baz"}).fit()

    with pytest.raises(
        TypeError, match="All elements of 'labels' must be a string"
    ):
        masker = NiftiLabelsMasker(labels_img, labels=[1, 2, 3])
        masker.fit()


@pytest.mark.parametrize(
    "background",
    [
        None,
        "background",
        "Background",
    ],  # In case the list of labels includes one for background
)
@pytest.mark.parametrize(
    "dtype",
    ["int32", "float32"],  # In case regions are labeled with floats
)
@pytest.mark.parametrize(
    "affine_data",
    [
        None,  # no resampling
        np.diag(
            (4, 4, 4, 4)  # with resampling
        ),  # region_names_ matches signals after resampling drops labels
    ],
)
def test_region_names(
    shape_3d_default, affine_eye, background, affine_data, dtype, n_regions
):
    """Test region_names_ attribute in NiftiLabelsMasker."""
    resampling = True
    if affine_data is None:
        resampling = False
        affine_data = affine_eye
    fmri_img, _ = generate_random_img(shape_3d_default, affine=affine_data)
    labels_img = generate_labeled_regions(
        shape_3d_default,
        affine=affine_eye,
        n_regions=n_regions,
        dtype=dtype,
    )

    masker = NiftiLabelsMasker(
        labels_img,
        labels=generate_labels(n_regions, background=background),
        resampling_target="data",
    )

    signals = masker.fit_transform(fmri_img)

    tmp = generate_labels(n_regions, background=background)
    if background is None:
        expected_lut = generate_expected_lut(["Background", *tmp])
    else:
        expected_lut = generate_expected_lut(tmp)
    check_lut(masker, expected_lut)

    region_names = generate_labels(n_regions, background=background)
    check_region_names_after_fit(
        masker,
        signals,
        region_names,
        background,
        resampling,
    )


@pytest.mark.parametrize(
    "background",
    [None, "background", "Background"],
)
@pytest.mark.parametrize(
    "affine_data",
    [
        None,  # no resampling
        np.diag(
            (4, 4, 4, 4)  # with resampling
        ),  # region_names_ matches signals after resampling drops labels
    ],
)
@pytest.mark.parametrize(
    "masking",
    [
        False,  # no masking
        True,  # with masking
    ],
)
@pytest.mark.parametrize(
    "keep_masked_labels",
    [False, True],
)
def test_region_names_ids_match_after_fit(
    shape_3d_default,
    affine_eye,
    background,
    affine_data,
    n_regions,
    masking,
    keep_masked_labels,
    img_labels,
):
    """Test that the same region names and ids correspond after fit."""
    if affine_data is None:
        # no resampling
        affine_data = affine_eye
    fmri_img, _ = generate_random_img(shape_3d_default, affine=affine_data)

    region_names = generate_labels(n_regions, background=background)
    region_ids = list(np.unique(get_data(img_labels)))

    if masking:
        # create a mask_img with 3 regions
        labels_data = get_data(img_labels)
        mask_data = (
            (labels_data == 1) + (labels_data == 2) + (labels_data == 5)
        )
        mask_img = Nifti1Image(mask_data.astype(np.int8), img_labels.affine)
    else:
        mask_img = None

    masker = NiftiLabelsMasker(
        img_labels,
        labels=region_names,
        resampling_target="data",
        mask_img=mask_img,
        keep_masked_labels=keep_masked_labels,
    )

    masker.fit_transform(fmri_img)

    tmp = generate_labels(n_regions, background=background)
    if background is None:
        expected_lut = generate_expected_lut(["Background", *tmp])
    else:
        expected_lut = generate_expected_lut(tmp)
    check_lut(masker, expected_lut)

    check_region_names_ids_match_after_fit(
        masker, region_names, region_ids, background
    )


@pytest.mark.parametrize("background", [None, "background", "Background"])
def test_region_names_with_non_sequential_labels(
    shape_3d_default, affine_eye, background
):
    """Test for atlases with region id that are not consecutive.

    See the AAL atlas for an example of this.
    """
    labels = [2001, 2002, 2101, 2102, 9170]
    fmri_img, _ = generate_random_img(shape_3d_default, affine=affine_eye)
    labels_img = generate_labeled_regions(
        shape_3d_default[:3],
        affine=affine_eye,
        n_regions=len(labels),
        labels=[0, *labels],
    )

    masker = NiftiLabelsMasker(
        labels_img,
        labels=generate_labels(len(labels), background=background),
        resampling_target=None,
    )

    signals = masker.fit_transform(fmri_img)

    expected_lut = pd.DataFrame(
        {
            "index": [0, *labels],
            "name": ["Background"]
            + [f"region_{i}" for i in range(1, len(labels) + 1)],
        }
    )
    check_lut(masker, expected_lut)

    region_names = generate_labels(len(labels), background=background)

    check_region_names_after_fit(masker, signals, region_names, background)


@pytest.mark.parametrize("background", [None, "background", "Background"])
def test_more_labels_than_actual_region_in_atlas(
    shape_3d_default, affine_eye, background, n_regions, img_labels
):
    """Test region_names_ property in NiftiLabelsMasker.

    See fetch_atlas_destrieux_2009 for example.
    Some labels have no associated voxels.
    """
    n_regions_in_labels = n_regions + 5

    region_names = generate_labels(n_regions_in_labels, background=background)

    masker = NiftiLabelsMasker(
        img_labels,
        labels=region_names,
        resampling_target="data",
    )

    fmri_img, _ = generate_random_img(shape_3d_default, affine=affine_eye)
    with pytest.warns(
        UserWarning,
        match="Too many names for the indices. Dropping excess names values.",
    ):
        masker.fit_transform(fmri_img)

    # lut should have one extra row for background
    assert len(masker.lut_) == n_regions + 1


@pytest.mark.parametrize("background", [None, "Background"])
def test_pass_lut(
    shape_3d_default, affine_eye, n_regions, img_labels, tmp_path, background
):
    """Smoke test to pass LUT directly or as file."""
    region_names = generate_labels(n_regions, background=background)
    if background:
        lut = pd.DataFrame(
            {"name": region_names, "index": list(range(n_regions + 1))}
        )
    else:
        lut = pd.DataFrame(
            {
                "name": ["Background", *region_names],
                "index": list(range(n_regions + 1)),
            }
        )

    fmri_img, _ = generate_random_img(shape_3d_default, affine=affine_eye)

    masker = NiftiLabelsMasker(
        img_labels,
        lut=lut,
    )

    masker.fit_transform(fmri_img)

    assert masker.lut_["index"].to_list() == lut["index"].to_list()
    assert masker.lut_["name"].to_list() == lut["name"].to_list()

    lut_file = tmp_path / "lut.csv"
    lut.to_csv(lut_file, index=False)
    masker = NiftiLabelsMasker(img_labels, lut=lut_file)

    masker.fit_transform(fmri_img)


def test_pass_lut_error(n_regions, img_labels):
    """Cannot pass both LUT and labels."""
    region_names = generate_labels(n_regions, background=None)
    lut = pd.DataFrame(
        {
            "name": ["Background", *region_names],
            "index": list(range(n_regions + 1)),
        }
    )

    with pytest.raises(
        ValueError, match="Pass either labels or a lookup table"
    ):
        NiftiLabelsMasker(img_labels, lut=lut, labels=region_names).fit()


def test_no_background(n_regions, img_labels, shape_3d_default, affine_eye):
    """Test label image with no background."""
    region_names = generate_labels(n_regions + 1, background=None)
    lut = pd.DataFrame(
        {
            "name": [*region_names],
            "index": list(range(n_regions + 1)),
        }
    )

    fmri_img, _ = generate_random_img(shape_3d_default, affine=affine_eye)

    masker = NiftiLabelsMasker(img_labels, lut=lut, background_label=999)

    masker.fit()

    assert "Background" not in masker.lut_["name"].to_list()

    signal = masker.fit_transform(fmri_img)

    assert "Background" not in masker.lut_["name"].to_list()
    assert "Background" not in masker._lut_["name"].to_list()

    assert 999 not in masker.lut_["index"].to_list()
    assert 999 not in masker._lut_["index"].to_list()

    n_expected_regions = n_regions + 1
    assert masker.n_elements_ == n_expected_regions
    assert signal.shape[0] == n_expected_regions
    assert len(masker.labels_) == n_expected_regions
    assert len(masker.region_ids_) == n_expected_regions
    assert len(masker.region_names_) == n_expected_regions
