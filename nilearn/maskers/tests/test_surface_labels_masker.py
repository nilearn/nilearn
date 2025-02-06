import warnings
from os.path import join
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from nilearn._utils.class_inspect import check_estimator
from nilearn._utils.helpers import is_matplotlib_installed
from nilearn.conftest import _make_mesh
from nilearn.maskers import SurfaceLabelsMasker
from nilearn.maskers.tests.conftest import check_valid_for_all_maskers
from nilearn.surface import SurfaceImage


def _sklearn_surf_label_img():
    """Create a sample surface label image using the sample mesh, just to use
    for scikit-learn checks.
    """
    labels = {
        "left": np.asarray([1, 2, 3, 5]),
        "right": np.asarray([4, 5, 6, 7, 9]),
    }
    return SurfaceImage(_make_mesh(), labels)


extra_valid_checks = [
    *check_valid_for_all_maskers(),
    "check_no_attributes_set_in_init",
    "check_do_not_raise_errors_in_init_or_set_params",
    "check_dont_overwrite_parameters",
    "check_estimators_fit_returns_self",
    "check_estimators_overwrite_params",
    "check_positive_only_tag_during_fit",
    "check_readonly_memmap_input",
]


@pytest.mark.parametrize(
    "estimator, check, name",
    check_estimator(
        estimator=[SurfaceLabelsMasker(_sklearn_surf_label_img())],
        extra_valid_checks=extra_valid_checks,
    ),
)
def test_check_estimator(estimator, check, name):  # noqa: ARG001
    """Check compliance with sklearn estimators."""
    check(estimator)


@pytest.mark.xfail(reason="invalid checks should fail")
@pytest.mark.parametrize(
    "estimator, check, name",
    check_estimator(
        estimator=[SurfaceLabelsMasker(_sklearn_surf_label_img())],
        valid=False,
        extra_valid_checks=extra_valid_checks,
    ),
)
def test_check_estimator_invalid(estimator, check, name):  # noqa: ARG001
    """Check compliance with sklearn estimators."""
    check(estimator)


def test_surface_label_masker_fit(surf_label_img):
    """Test fit and check estimated attributes.

    0 value in data is considered as background
    and should not be listed in the labels.
    """
    masker = SurfaceLabelsMasker(labels_img=surf_label_img)
    masker = masker.fit()

    assert masker.n_elements_ == 1
    assert masker._labels_ == [1]
    assert masker.label_names_ == ["0", "1"]
    assert masker._reporting_data is not None
    assert masker.lut_["name"].to_list() == ["0", "1"]
    assert masker.lut_["index"].to_list() == [0, 1]


def test_surface_label_masker_fit_with_names(surf_label_img):
    """Check passing labels is reflected in attributes."""
    masker = SurfaceLabelsMasker(
        labels_img=surf_label_img, labels=["background", "bar", "foo"]
    )

    with pytest.warns(UserWarning, match="Dropping excess names values."):
        masker = masker.fit()

    assert masker.n_elements_ == 1
    assert masker._labels_ == [1]
    assert masker.label_names_ == ["background", "bar"]
    assert masker.lut_["name"].to_list() == ["background", "bar"]
    assert masker.lut_["index"].to_list() == [0, 1]

    masker = SurfaceLabelsMasker(
        labels_img=surf_label_img, labels=["background"]
    )

    with pytest.warns(UserWarning, match="Padding 'names' with 'unknown'"):
        masker = masker.fit()

    assert masker.n_elements_ == 1
    assert masker._labels_ == [1]
    assert masker.label_names_ == ["background", "unknown"]
    assert masker.lut_["name"].to_list() == ["background", "unknown"]
    assert masker.lut_["index"].to_list() == [0, 1]


def test_surface_label_masker_fit_with_lut(surf_label_img, tmp_path):
    """Check passing lut is reflected in attributes.

    Check that lut can be read from:
    - a tsv file (str or path)
    - a csv file (doc strings only mention TSV but testing for robustness)
    - a dataframe
    """
    lut_df = pd.DataFrame({"index": [0, 1], "name": ["background", "bar"]})

    lut_tsv = tmp_path / "lut.tsv"
    lut_df.to_csv(lut_tsv, sep="\t", index=False)

    lut_csv = tmp_path / "lut.csv"
    lut_df.to_csv(lut_csv, sep="\t", index=False)

    for lut in [lut_tsv, lut_csv, lut_df, str(lut_tsv)]:
        masker = SurfaceLabelsMasker(labels_img=surf_label_img, lut=lut).fit()

        assert masker.n_elements_ == 1
        assert masker._labels_ == [1]
        assert masker.label_names_ == ["background", "bar"]


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
def test_surface_label_masker_transform(
    surf_label_img, surf_img_1d, surf_img_2d, strategy
):
    """Test transform extract signals.

    Also a smoke test for different strategies.
    """
    masker = SurfaceLabelsMasker(labels_img=surf_label_img, strategy=strategy)
    masker = masker.fit()

    # only one 'timepoint'
    signal = masker.transform(surf_img_1d)

    assert isinstance(signal, np.ndarray)
    n_labels = len(masker._labels_)
    assert signal.shape == (1, n_labels)

    # 5 'timepoint'
    n_timepoints = 5
    signal = masker.transform(surf_img_2d(n_timepoints))

    assert isinstance(signal, np.ndarray)
    assert signal.shape == (n_timepoints, n_labels)


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
    masker = masker.fit()
    n_timepoints = 5
    with pytest.warns(
        UserWarning,
        match="the following labels were removed",
    ):
        signal = masker.transform(surf_img_2d(n_timepoints))
    assert isinstance(signal, np.ndarray)
    assert signal.shape == (n_timepoints, 2)


def test_surface_label_masker_check_output_1d(surf_mesh, rng):
    """Check actual content of the transform and inverse_transform.

    - Use a label mask with more than one label.
    - Use data with known content and expected mean.
      and background label data has random value.
    - Check that output data is properly averaged,
      even when labels are spread across hemispheres.
    """
    labels = {
        "left": np.asarray([2, 0, 10, 1]),
        "right": np.asarray([10, 1, 20, 20, 0]),
    }
    surf_label_img = SurfaceImage(surf_mesh, labels)
    masker = SurfaceLabelsMasker(labels_img=surf_label_img)
    masker = masker.fit()

    expected_mean_value = {
        "1": 5,
        "2": 6,
        "10": 50,
        "20": 60,
    }

    data = {
        "left": np.asarray(
            [
                expected_mean_value["2"],
                rng.random(),
                expected_mean_value["10"],
                expected_mean_value["1"],
            ]
        ),
        "right": np.asarray(
            [
                expected_mean_value["10"],
                expected_mean_value["1"],
                expected_mean_value["20"],
                expected_mean_value["20"],
                rng.random(),
            ]
        ),
    }
    surf_img_1d = SurfaceImage(surf_mesh, data)
    signal = masker.transform(surf_img_1d)

    assert signal.shape == (1, masker.n_elements_)

    expected_signal = np.asarray(
        [
            [
                expected_mean_value["1"],
                expected_mean_value["2"],
                expected_mean_value["10"],
                expected_mean_value["20"],
            ]
        ]
    )

    assert_array_equal(signal, expected_signal)

    # also check the output of inverse_transform
    img = masker.inverse_transform(signal)
    assert img.shape[0] == surf_img_1d.shape[0]
    # expected inverse data is the same as the input data
    # but with the random value replaced by zeros
    expected_inverse_data = {
        "left": np.asarray(
            [
                [
                    expected_mean_value["2"],
                    0.0,
                    expected_mean_value["10"],
                    expected_mean_value["1"],
                ]
            ]
        ).T,
        "right": np.asarray(
            [
                [
                    expected_mean_value["10"],
                    expected_mean_value["1"],
                    expected_mean_value["20"],
                    expected_mean_value["20"],
                    0.0,
                ]
            ]
        ).T,
    }

    assert_array_equal(img.data.parts["left"], expected_inverse_data["left"])
    assert_array_equal(img.data.parts["right"], expected_inverse_data["right"])


def test_surface_label_masker_check_output_2d(surf_mesh, rng):
    """Check actual content of the transform and inverse_transform when
    we have multiple timepoints.

    - Use a label mask with more than one label.
    - Use data with known content and expected mean.
      and background label data has random value.
    - Check that output data is properly averaged,
      even when labels are spread across hemispheres.
    """
    labels = {
        "left": np.asarray([2, 0, 10, 1]),
        "right": np.asarray([10, 1, 20, 20, 0]),
    }
    surf_label_img = SurfaceImage(surf_mesh, labels)
    masker = SurfaceLabelsMasker(labels_img=surf_label_img)
    masker = masker.fit()

    expected_mean_value = {
        "1": 5,
        "2": 6,
        "10": 50,
        "20": 60,
    }

    # Now with 2 'time points'
    data = {
        "left": np.asarray(
            [
                [
                    expected_mean_value["2"] - 1,
                    rng.random(),
                    expected_mean_value["10"] - 1,
                    expected_mean_value["1"] - 1,
                ],
                [
                    expected_mean_value["2"] + 1,
                    rng.random(),
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
                    rng.random(),
                ],
                [
                    expected_mean_value["10"] + 1,
                    expected_mean_value["1"] + 1,
                    expected_mean_value["20"] + 1,
                    expected_mean_value["20"] + 1,
                    rng.random(),
                ],
            ]
        ).T,
    }

    surf_img_2d = SurfaceImage(surf_mesh, data)
    signal = masker.transform(surf_img_2d)

    assert signal.shape == (surf_img_2d.shape[1], masker.n_elements_)

    expected_signal = np.asarray(
        [
            [
                expected_mean_value["1"] - 1,
                expected_mean_value["2"] - 1,
                expected_mean_value["10"] - 1,
                expected_mean_value["20"] - 1,
            ],
            [
                expected_mean_value["1"] + 1,
                expected_mean_value["2"] + 1,
                expected_mean_value["10"] + 1,
                expected_mean_value["20"] + 1,
            ],
        ]
    )
    assert_array_equal(signal, expected_signal)

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


def test_surface_label_masker_fit_transform(surf_label_img, surf_img_1d):
    """Smoke test for fit_transform."""
    masker = SurfaceLabelsMasker(labels_img=surf_label_img)
    signal = masker.fit_transform(surf_img_1d)
    assert signal.shape == (1, masker.n_elements_)


def test_surface_label_masker_inverse_transform(surf_label_img, surf_img_1d):
    """Test transform extract signals."""
    masker = SurfaceLabelsMasker(labels_img=surf_label_img)
    masker = masker.fit()
    signal = masker.transform(surf_img_1d)
    img = masker.inverse_transform(signal)
    assert img.shape == (surf_img_1d.shape[0], 1)


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
    masker = masker.fit()
    n_timepoints = 5
    with pytest.warns(
        UserWarning,
        match="the following labels were removed",
    ):
        signal = masker.transform(surf_img_2d(n_timepoints))
        img_inverted = masker.inverse_transform(signal)
        assert img_inverted.shape == surf_img_2d(n_timepoints).shape
        # the data for label 2 should be zeros
        assert np.all(img_inverted.data.parts["left"][-1, :] == 0)
        assert np.all(img_inverted.data.parts["right"][2:, :] == 0)


def test_surface_label_masker_transform_list_surf_images(
    surf_label_img, surf_img_1d, surf_img_2d
):
    """Test transform on list of surface images."""
    masker = SurfaceLabelsMasker(surf_label_img).fit()
    signals = masker.transform([surf_img_1d, surf_img_1d, surf_img_1d])
    assert signals.shape == (3, masker.n_elements_)
    signals = masker.transform([surf_img_2d(5), surf_img_2d(4)])
    assert signals.shape == (9, masker.n_elements_)


def test_surface_label_masker_inverse_transform_list_surf_images(
    surf_label_img, surf_img_2d
):
    """Test inverse_transform on list of surface images."""
    masker = SurfaceLabelsMasker(surf_label_img).fit()
    signals = masker.transform([surf_img_2d(3), surf_img_2d(4)])
    img = masker.inverse_transform(signals)
    assert img.shape == (surf_label_img.mesh.n_vertices, 7)


@pytest.mark.parametrize("confounds", [None, np.ones((20, 3)), "str", "Path"])
def test_surface_labels_masker_confounds_to_fit_transform(
    surf_label_img, surf_img_2d, confounds
):
    """Test fit_transform with confounds."""
    masker = SurfaceLabelsMasker(surf_label_img)
    if isinstance(confounds, str) and confounds == "Path":
        nilearn_dir = Path(__file__).parent.parent.parent
        confounds = nilearn_dir / "tests" / "data" / "spm_confounds.txt"
    elif isinstance(confounds, str) and confounds == "str":
        # we need confound to be a string so using os.path.join
        confounds = join(  # noqa: PTH118
            Path(__file__).parent.parent.parent,
            "tests",
            "data",
            "spm_confounds.txt",
        )
    signals = masker.fit_transform(surf_img_2d(20), confounds=confounds)
    assert signals.shape == (20, masker.n_elements_)


def test_surface_labels_masker_sample_mask_to_fit_transform(
    surf_label_img, surf_img_2d
):
    """Test transform with sample_mask."""
    masker = SurfaceLabelsMasker(surf_label_img)
    masker = masker.fit()
    signals = masker.transform(
        surf_img_2d(5),
        sample_mask=np.asarray([True, False, True, False, True]),
    )
    # we remove two samples via sample_mask so we should have 3 samples
    assert signals.shape == (3, masker.n_elements_)


def test_surface_label_masker_labels_img_none():
    """Test that an error is raised when labels_img is None."""
    with pytest.raises(
        ValueError,
        match="provide a labels_img to the masker",
    ):
        SurfaceLabelsMasker(labels_img=None).fit()


@pytest.mark.skipif(
    is_matplotlib_installed(),
    reason="Test requires matplotlib not to be installed.",
)
def test_masker_reporting_mpl_warning(surf_label_img):
    """Raise warning after exception if matplotlib is not installed."""
    with warnings.catch_warnings(record=True) as warning_list:
        SurfaceLabelsMasker(surf_label_img).fit().generate_report()

    assert len(warning_list) == 1
    assert issubclass(warning_list[0].category, ImportWarning)


def test_error_wrong_strategy(surf_label_img):
    """Throw error for unsupported strategies."""
    masker = SurfaceLabelsMasker(labels_img=surf_label_img, strategy="foo")
    with pytest.raises(ValueError, match="Invalid strategy 'foo'."):
        masker.fit()
