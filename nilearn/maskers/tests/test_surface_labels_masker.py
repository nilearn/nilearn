import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from sklearn.utils.estimator_checks import parametrize_with_checks

from nilearn._utils.estimator_checks import (
    check_estimator,
    nilearn_check_estimator,
    return_expected_failed_checks,
)
from nilearn._utils.tags import SKLEARN_LT_1_6
from nilearn.conftest import _make_mesh
from nilearn.maskers import SurfaceLabelsMasker
from nilearn.surface import SurfaceImage


def _sklearn_surf_label_img():
    """Create a sample surface label image using the sample mesh,
    just to use for scikit-learn checks.
    """
    labels = {
        "left": np.asarray([1, 1, 2, 2]),
        "right": np.asarray([1, 1, 2, 2, 2]),
    }
    return SurfaceImage(_make_mesh(), labels)


ESTIMATORS_TO_CHECK = [SurfaceLabelsMasker(_sklearn_surf_label_img())]

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


@pytest.mark.parametrize(
    "estimator, check, name",
    nilearn_check_estimator(estimators=ESTIMATORS_TO_CHECK),
)
def test_check_estimator_nilearn(estimator, check, name):  # noqa: ARG001
    """Check compliance with sklearn estimators."""
    check(estimator)


def test_surface_label_masker_fit(surf_label_img):
    """Test fit and check estimated attributes."""
    masker = SurfaceLabelsMasker(labels_img=surf_label_img)
    masker = masker.fit()

    assert masker.n_elements_ == 1
    assert masker.labels_ == [0, 1]
    assert masker._reporting_data is not None
    assert masker.lut_["name"].to_list() == ["Background", "1"]
    assert masker.region_names_ == {0: "1"}
    assert masker.region_ids_ == {"background": 0, 0: 1}


def test_surface_label_masker_fit_transform(surf_label_img, surf_img_1d):
    """Test transform does not return any value for the background.

    A single scalar should be return as the mask only has 1 region.
    """
    masker = SurfaceLabelsMasker(labels_img=surf_label_img)
    signal = masker.fit_transform(surf_img_1d)

    assert masker.n_elements_ == 1
    assert signal.size == masker.n_elements_


@pytest.mark.parametrize("labels", [["Background", "bar"], ["bar"]])
def test_surface_label_masker_fit_with_labels(surf_label_img, labels):
    """Check passing labels is reflected in attributes.

    Should behave the same even if "Background" is not in the list of labels
    """
    masker = SurfaceLabelsMasker(labels_img=surf_label_img, labels=labels)

    masker = masker.fit()

    assert masker.n_elements_ == 1
    assert masker.labels_ == [0, 1]
    assert masker.lut_["name"].to_list() == ["Background", "bar"]
    assert masker.region_names_ == {0: "bar"}
    assert masker.region_ids_ == {"background": 0, 0: 1}


@pytest.mark.parametrize("labels", [["bar"], ["bar", "Background"]])
def test_surface_label_masker_fit_background_label(surf_label_img, labels):
    """Check that labels and background label can be matched properly.

    Here we say that the background label is 1 and not 0.

    Should behave the same even if "Background" is not in the list of labels
    """
    masker = SurfaceLabelsMasker(
        labels_img=surf_label_img, labels=labels, background_label=1
    )

    masker = masker.fit()

    assert masker.n_elements_ == 1
    assert masker.labels_ == [0, 1]
    assert masker.lut_["name"].to_list() == ["bar", "Background"]
    assert masker.region_names_ == {0: "bar"}
    assert masker.region_ids_ == {"background": 1, 0: 0}


def test_surface_label_masker_fit_too_many_labels(surf_label_img):
    """Check passing labels is reflected in attributes.

    Check warning are thrown when too many are passed.
    """
    masker = SurfaceLabelsMasker(
        labels_img=surf_label_img, labels=["Background", "bar", "foo"]
    )

    with pytest.warns(UserWarning, match="Dropping excess names values."):
        masker = masker.fit()

    assert masker.n_elements_ == 1
    assert masker.labels_ == [0, 1]
    assert masker.lut_["name"].to_list() == ["Background", "bar"]
    assert masker.region_names_ == {0: "bar"}
    assert masker.region_ids_ == {"background": 0, 0: 1}


def test_surface_label_masker_fit_too_few_labels(surf_label_img):
    """Check passing labels is reflected in attributes.

    Check warning are thrown when too few are passed.
    """
    masker = SurfaceLabelsMasker(
        labels_img=surf_label_img, labels=["Background"]
    )

    with pytest.warns(UserWarning, match="Padding 'names' with 'unknown'"):
        masker = masker.fit()

    assert masker.n_elements_ == 1
    assert masker.labels_ == [0, 1]
    assert masker.lut_["name"].to_list() == ["Background", "unknown"]
    assert masker.region_names_ == {0: "unknown"}
    assert masker.region_ids_ == {"background": 0, 0: 1}


def test_surface_label_masker_fit_with_lut(surf_label_img, tmp_path):
    """Check passing lut is reflected in attributes.

    Check that lut can be read from:
    - a tsv file (str or path)
    - a csv file (doc strings only mention TSV but testing for robustness)
    - a dataframe
    """
    lut_df = pd.DataFrame({"index": [0, 1], "name": ["Background", "bar"]})

    lut_tsv = tmp_path / "lut.tsv"
    lut_df.to_csv(lut_tsv, sep="\t", index=False)

    lut_csv = tmp_path / "lut.csv"
    lut_df.to_csv(lut_csv, sep="\t", index=False)

    for lut in [lut_tsv, lut_csv, lut_df, str(lut_tsv)]:
        masker = SurfaceLabelsMasker(labels_img=surf_label_img, lut=lut).fit()

        if isinstance(lut, pd.DataFrame):
            assert list(masker.lut.columns) == list(masker.lut_.columns)
        assert masker.n_elements_ == 1
        assert masker.labels_ == [0, 1]
        assert masker.lut_["name"].to_list() == ["Background", "bar"]


@pytest.mark.parametrize(
    "background_label, n_expected_regions", [(0, 3), (1, 2)]
)
@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {
            "lut": pd.DataFrame(
                {
                    "index": [0, 1, 2, 3],
                    "name": ["Background", "Foo", "Bar", "Baz"],
                }
            )
        },
        {
            "lut": pd.DataFrame(
                {
                    "index": [0, 1, 2, 3],
                    "name": ["Unknown", "Foo", "Bar", "Baz"],
                }
            )
        },
        {"labels": ["background", "Foo", "Bar", "Baz"]},
    ],
)
def test_label_image_no_background_missing_regions(
    surf_mesh, surf_img_2d, background_label, n_expected_regions, kwargs
):
    """Test label image with no background.

    Compare behavior when background is present in label image
    (background_label=1) or not (background_label=0).

    Regression test for https://github.com/nilearn/nilearn/issues/5596
    """
    data = {
        "left": np.asarray([3, 3, 1, 1]),
        "right": np.asarray([1, 1, 3, 2, 3]),
    }
    label_img = SurfaceImage(surf_mesh, data)

    labels_masker = SurfaceLabelsMasker(
        labels_img=label_img, background_label=background_label, **kwargs
    ).fit()

    if "lut" in kwargs:
        # when reset_index() does not use drop=True,
        # extra columns were added
        assert list(kwargs["lut"].columns) == list(labels_masker.lut_.columns)

    masked_data = labels_masker.transform(surf_img_2d(2))

    assert masked_data.shape[1] == n_expected_regions
    assert len(labels_masker.region_names_) == n_expected_regions

    if background_label == 1:
        assert "Background" in labels_masker.lut_["name"].to_list()
        assert len(labels_masker.labels_) == n_expected_regions + 1
        assert len(labels_masker.region_ids_) == n_expected_regions + 1
    else:
        assert "Background" not in labels_masker.lut_["name"].to_list()
        assert len(labels_masker.labels_) == n_expected_regions
        assert len(labels_masker.region_ids_) == n_expected_regions


def test_surface_label_masker_error_names_and_lut(surf_label_img):
    """Cannot pass both look up table AND names."""
    lut = pd.DataFrame({"index": [0, 1], "name": ["background", "bar"]})
    masker = SurfaceLabelsMasker(
        labels_img=surf_label_img, labels=["background", "bar"], lut=lut
    )
    with pytest.raises(
        ValueError,
        match="Pass either labels or a lookup table .* but not both.",
    ):
        masker.fit()


def test_surface_label_masker_fit_no_report(surf_label_img):
    """Check no report data is stored."""
    masker = SurfaceLabelsMasker(labels_img=surf_label_img, reports=False)
    masker = masker.fit()
    assert masker._reporting_data is None


@pytest.mark.parametrize(
    "strategy",
    (
        "variance",
        "minimum",
        "mean",
        "standard_deviation",
        "sum",
        "median",
        "maximum",
    ),
)
def test_surface_label_masker_transform(surf_label_img, surf_img_1d, strategy):
    """Test transform extract signals.

    Also a smoke test for different strategies.
    """
    masker = SurfaceLabelsMasker(labels_img=surf_label_img, strategy=strategy)
    masker = masker.fit()

    signal = masker.transform(surf_img_1d)

    assert isinstance(signal, np.ndarray)
    assert signal.size == 1


def test_surface_label_masker_transform_with_mask(surf_mesh, surf_img_2d):
    """Test transform extract signals with a mask and check warning."""
    # create a labels image
    labels_data = {
        "left": np.asarray([1, 1, 1, 2]),
        "right": np.asarray([3, 3, 2, 2, 2]),
    }
    surf_label_img = SurfaceImage(surf_mesh, labels_data)

    # create a mask image
    # we are keeping labels 1 and 2 out of 3
    # so we should only get signals for labels 1 and 2
    # plus masker should throw a warning that label 3 is being removed due to
    # mask
    mask_data = {
        "left": np.asarray([1, 1, 1, 1]),
        "right": np.asarray([0, 0, 1, 1, 1]),
    }
    surf_mask = SurfaceImage(surf_mesh, mask_data)
    masker = SurfaceLabelsMasker(labels_img=surf_label_img, mask_img=surf_mask)

    with pytest.warns(
        UserWarning,
        match="the following labels were removed",
    ):
        masker = masker.fit()

    n_timepoints = 5
    signal = masker.transform(surf_img_2d(n_timepoints))

    assert isinstance(signal, np.ndarray)
    expected_n_regions = 2
    assert masker.n_elements_ == expected_n_regions
    assert signal.shape == (n_timepoints, masker.n_elements_)

    # note that even if the label image contained no vertex with
    # background level values, some vertices were masked
    # and given background value
    # so background should appear here.
    assert masker.labels_ == [0, 1, 2]
    assert masker.lut_["name"].to_list() == ["Background", "1", "2"]
    assert masker.region_names_ == {0: "1", 1: "2"}
    assert masker.region_ids_ == {"background": 0, 0: 1, 1: 2}


@pytest.fixture
def polydata_labels():
    """Return polydata with 4 regions."""
    return {
        "left": np.asarray([2, 0, 10, 1]),
        "right": np.asarray([10, 1, 20, 20, 0]),
    }


@pytest.fixture
def expected_mean_value():
    """Return expected values for some specific labels."""
    return {
        "1": 5,
        "2": 6,
        "10": 50,
        "20": 60,
    }


@pytest.fixture
def data_left_1d_with_expected_mean(rng, expected_mean_value):
    """Generate left data with given expected value for one sample."""
    return np.asarray(
        [
            expected_mean_value["2"],
            rng.random(),
            expected_mean_value["10"],
            expected_mean_value["1"],
        ]
    )


@pytest.fixture
def data_right_1d_with_expected_mean(rng, expected_mean_value):
    """Generate right data with given expected value for one sample."""
    return np.asarray(
        [
            expected_mean_value["10"],
            expected_mean_value["1"],
            expected_mean_value["20"],
            expected_mean_value["20"],
            rng.random(),
        ]
    )


@pytest.fixture
def expected_signal(expected_mean_value):
    """Return signal extract from data with expected mean."""
    return np.asarray(
        [
            expected_mean_value["1"],
            expected_mean_value["2"],
            expected_mean_value["10"],
            expected_mean_value["20"],
        ]
    )


@pytest.fixture
def inverse_data_left_1d_with_expected_mean(expected_mean_value):
    """Return inversed left data with given expected value for one sample."""
    return np.asarray(
        [
            expected_mean_value["2"],
            0.0,
            expected_mean_value["10"],
            expected_mean_value["1"],
        ]
    )


@pytest.fixture
def inverse_data_right_1d_with_expected_mean(expected_mean_value):
    """Return inversed right data with given expected value for one sample."""
    return np.asarray(
        [
            expected_mean_value["10"],
            expected_mean_value["1"],
            expected_mean_value["20"],
            expected_mean_value["20"],
            0.0,
        ]
    )


def test_surface_label_masker_check_output_1d(
    surf_mesh,
    polydata_labels,
    expected_signal,
    data_left_1d_with_expected_mean,
    data_right_1d_with_expected_mean,
    inverse_data_left_1d_with_expected_mean,
    inverse_data_right_1d_with_expected_mean,
):
    """Check actual content of the transform and inverse_transform.

    - Use a label mask with more than one label.
    - Use data with known content and expected mean.
      and background label data has random value.
    - Check that output data is properly averaged,
      even when labels are spread across hemispheres.
    """
    surf_label_img = SurfaceImage(surf_mesh, polydata_labels)
    masker = SurfaceLabelsMasker(labels_img=surf_label_img)
    masker = masker.fit()

    data = {
        "left": data_left_1d_with_expected_mean,
        "right": data_right_1d_with_expected_mean,
    }
    surf_img_1d = SurfaceImage(surf_mesh, data)
    signal = masker.transform(surf_img_1d)

    assert_array_equal(signal, np.asarray(expected_signal))

    assert masker.labels_ == [0, 1, 2, 10, 20]
    assert masker.lut_["name"].to_list() == [
        "Background",
        "1",
        "2",
        "10",
        "20",
    ]
    assert masker.region_names_ == {0: "1", 1: "2", 2: "10", 3: "20"}
    assert masker.region_ids_ == {"background": 0, 0: 1, 1: 2, 2: 10, 3: 20}

    # also check the output of inverse_transform
    img = masker.inverse_transform(signal)
    assert img.shape[0] == surf_img_1d.shape[0]
    # expected inverse data is the same as the input data
    # but with the random value replaced by zeros
    expected_inverse_data = {
        "left": np.asarray(inverse_data_left_1d_with_expected_mean).T,
        "right": np.asarray(inverse_data_right_1d_with_expected_mean).T,
    }

    assert_array_equal(img.data.parts["left"], expected_inverse_data["left"])
    assert_array_equal(img.data.parts["right"], expected_inverse_data["right"])


def test_surface_label_masker_lut_unsorted(
    surf_mesh,
    polydata_labels,
    expected_signal,
    data_left_1d_with_expected_mean,
    data_right_1d_with_expected_mean,
):
    """Test lut with wrong order of regions.

    LUT, region_ids, region_names should be properly sorted after fit.
    Result of region_ids, region_names
    should still match content of extracted signals.
    """
    surf_label_img = SurfaceImage(surf_mesh, polydata_labels)

    lut = pd.DataFrame(
        columns=["index", "name"],
        data=[[1.0, "one"], [20.0, "twenty"], [10.0, "ten"], [2.0, "two"]],
    )

    masker = SurfaceLabelsMasker(labels_img=surf_label_img, lut=lut)
    masker = masker.fit()

    assert list(masker.lut.columns) == list(masker.lut_.columns)
    assert masker.labels_ == [0.0, 1.0, 2.0, 10.0, 20.0]
    assert masker.lut_["name"].to_list() == [
        "Background",
        "one",
        "two",
        "ten",
        "twenty",
    ]
    assert masker.region_names_ == {0: "one", 1: "two", 2: "ten", 3: "twenty"}
    assert masker.region_ids_ == {
        "background": 0,
        0: 1.0,
        1: 2.0,
        2: 10.0,
        3: 20.0,
    }

    data = {
        "left": data_left_1d_with_expected_mean,
        "right": data_right_1d_with_expected_mean,
    }
    surf_img_1d = SurfaceImage(surf_mesh, data)
    signal = masker.transform(surf_img_1d)

    assert isinstance(signal, np.ndarray)
    assert_array_equal(signal, np.asarray(expected_signal))


def test_surface_label_masker_check_output_2d(
    surf_mesh,
    polydata_labels,
    expected_mean_value,
    expected_signal,
    data_left_1d_with_expected_mean,
    data_right_1d_with_expected_mean,
):
    """Check actual content of the transform and inverse_transform when
    we have multiple timepoints.

    - Use a label mask with more than one label.
    - Use data with known content and expected mean.
      and background label data has random value.
    - Check that output data is properly averaged,
      even when labels are spread across hemispheres.
    """
    surf_label_img = SurfaceImage(surf_mesh, polydata_labels)
    masker = SurfaceLabelsMasker(labels_img=surf_label_img)
    masker = masker.fit()

    # Now with 2 'time points'
    data = {
        "left": np.asarray(
            [
                data_left_1d_with_expected_mean - 1,
                data_left_1d_with_expected_mean + 1,
            ]
        ).T,
        "right": np.asarray(
            [
                data_right_1d_with_expected_mean - 1,
                data_right_1d_with_expected_mean + 1,
            ]
        ).T,
    }

    surf_img_2d = SurfaceImage(surf_mesh, data)
    signal = masker.transform(surf_img_2d)

    assert signal.shape == (surf_img_2d.shape[1], masker.n_elements_)

    expected_signal = np.asarray([expected_signal - 1, expected_signal + 1])
    assert_array_equal(signal, expected_signal)

    assert masker.labels_ == [0, 1, 2, 10, 20]
    assert masker.lut_["name"].to_list() == [
        "Background",
        "1",
        "2",
        "10",
        "20",
    ]
    assert masker.region_names_ == {0: "1", 1: "2", 2: "10", 3: "20"}
    assert masker.region_ids_ == {"background": 0, 0: 1, 1: 2, 2: 10, 3: 20}

    # also check the output of inverse_transform
    img = masker.inverse_transform(signal)

    assert img.shape[0] == surf_img_2d.shape[0]
    # expected inverse data is the same as the input data
    # but with the random values replaced by zeros
    expected_inverse_data = {
        "left": np.asarray(
            [
                [
                    expected_mean_value["2"] - 1,
                    0.0,
                    expected_mean_value["10"] - 1,
                    expected_mean_value["1"] - 1,
                ],
                [
                    expected_mean_value["2"] + 1,
                    0.0,
                    expected_mean_value["10"] + 1,
                    expected_mean_value["1"] + 1,
                ],
            ]
        ).T,
        "right": np.asarray(
            [
                [
                    expected_mean_value["10"] - 1,
                    expected_mean_value["1"] - 1,
                    expected_mean_value["20"] - 1,
                    expected_mean_value["20"] - 1,
                    0.0,
                ],
                [
                    expected_mean_value["10"] + 1,
                    expected_mean_value["1"] + 1,
                    expected_mean_value["20"] + 1,
                    expected_mean_value["20"] + 1,
                    0.0,
                ],
            ]
        ).T,
    }
    assert_array_equal(img.data.parts["left"], expected_inverse_data["left"])
    assert_array_equal(img.data.parts["right"], expected_inverse_data["right"])


def test_surface_label_masker_inverse_transform_with_mask(
    surf_mesh, surf_img_2d
):
    """Test inverse_transform with mask: inverted image's shape, warning if
    mask removes labels and data corresponding to removed labels is zeros.
    """
    # create a labels image
    labels_data = {
        "left": np.asarray([1, 1, 1, 2]),
        "right": np.asarray([3, 3, 2, 2, 2]),
    }
    surf_label_img = SurfaceImage(surf_mesh, labels_data)

    # create a mask image
    # we are keeping labels 1 and 3 out of 3
    # so we should only get signals for labels 1 and 3
    # plus masker should throw a warning that label 2 is being removed due to
    # mask
    mask_data = {
        "left": np.asarray([1, 1, 1, 0]),
        "right": np.asarray([1, 1, 0, 0, 0]),
    }
    surf_mask = SurfaceImage(surf_mesh, mask_data)
    masker = SurfaceLabelsMasker(labels_img=surf_label_img, mask_img=surf_mask)

    with pytest.warns(
        UserWarning,
        match="the following labels were removed",
    ):
        masker = masker.fit()

    n_timepoints = 5
    signal = masker.transform(surf_img_2d(n_timepoints))

    img_inverted = masker.inverse_transform(signal)

    assert img_inverted.shape == surf_img_2d(n_timepoints).shape
    # the data for label 2 should be zeros
    assert np.all(img_inverted.data.parts["left"][-1, :] == 0)
    assert np.all(img_inverted.data.parts["right"][2:, :] == 0)


def test_surface_label_masker_labels_img_none():
    """Test that an error is raised when labels_img is None."""
    with pytest.raises(
        ValueError,
        match="provide a labels_img to the masker",
    ):
        SurfaceLabelsMasker(labels_img=None).fit()


def test_error_wrong_strategy(surf_label_img):
    """Throw error for unsupported strategies."""
    masker = SurfaceLabelsMasker(labels_img=surf_label_img, strategy="foo")
    with pytest.raises(ValueError, match="Invalid strategy 'foo'."):
        masker.fit()
