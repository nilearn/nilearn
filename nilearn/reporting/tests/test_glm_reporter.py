import warnings

import numpy as np
import pandas as pd
import pytest

from nilearn._utils.data_gen import (
    basic_paradigm,
    generate_fake_fmri_data_and_design,
    write_fake_bold_img,
)
from nilearn.conftest import _img_mask_mni, _make_surface_mask
from nilearn.datasets import load_fsaverage
from nilearn.glm.first_level import (
    FirstLevelModel,
)
from nilearn.glm.second_level import SecondLevelModel
from nilearn.glm.thresholding import DEFAULT_Z_THRESHOLD
from nilearn.maskers import NiftiMasker
from nilearn.reporting import HTMLReport
from nilearn.surface import SurfaceImage


@pytest.fixture
def rk():
    """Return rank for design martrix."""
    return 3


@pytest.fixture
def contrasts(rk):
    """Return a contrast vector."""
    c = np.zeros((1, rk))
    c[0][0] = 1
    return c


@pytest.fixture()
def flm(rk):
    """Generate first level model."""
    shapes = ((7, 7, 7, 5),)
    mask, fmri_data, design_matrices = generate_fake_fmri_data_and_design(
        shapes, rk=rk
    )
    # generate_fake_fmri_data_and_design
    return FirstLevelModel().fit(fmri_data, design_matrices=design_matrices)


@pytest.fixture()
def slm():
    """Generate a fitted second level model."""
    shapes = ((7, 7, 7, 1),)
    _, fmri_data, _ = generate_fake_fmri_data_and_design(shapes)
    model = SecondLevelModel()
    Y = [fmri_data[0]] * 2
    X = pd.DataFrame([[1]] * 2, columns=["intercept"])
    return model.fit(Y, design_matrix=X)


def test_flm_report_no_activation_found(flm, contrasts):
    """Check presence message of no activation found.

    We use random data, so we should not get activations.
    """
    report = flm.generate_report(contrasts=contrasts)
    assert "No suprathreshold cluster" in report.__str__()


@pytest.mark.parametrize("model", [FirstLevelModel, SecondLevelModel])
@pytest.mark.parametrize("bg_img", [_img_mask_mni(), _make_surface_mask()])
def test_empty_surface_reports(tmp_path, model, bg_img):
    """Test that empty reports on unfitted model can be generated."""
    report = model(smoothing_fwhm=None).generate_report(bg_img=bg_img)

    assert isinstance(report, HTMLReport)

    report.save_as_html(tmp_path / "tmp.html")
    assert (tmp_path / "tmp.html").exists()


def test_flm_reporting_no_contrasts(flm):
    """Test for model report can be generated with no contrasts."""
    report = flm.generate_report(
        plot_type="glass",
        contrasts=None,
        min_distance=15,
        alpha=0.01,
    )
    assert "No statistical map was provided." in report.__str__()


def test_mask_coverage_in_report(flm):
    """Check that how much image is included in mask is in the report."""
    report = flm.generate_report()
    assert "The mask includes" in report.__str__()


@pytest.mark.timeout(0)
@pytest.mark.parametrize("height_control", ["fdr", "bonferroni", None])
def test_flm_reporting_height_control(flm, height_control, contrasts):
    """Test for first level model reporting.

    Also checks that passing threshold different from the default
    will throw a warning when height_control is not None.
    """
    with warnings.catch_warnings(record=True) as warnings_list:
        report_flm = flm.generate_report(
            contrasts=contrasts,
            plot_type="glass",
            height_control=height_control,
            min_distance=15,
            alpha=0.01,
            threshold=2,
        )
    if height_control is not None:
        assert any("will not be used with" in str(x) for x in warnings_list)
    # catches & raises UnicodeEncodeError in HTMLDocument.get_iframe()
    # in case certain unicode characters are mishandled,
    # like the greek alpha symbol.
    report_flm.get_iframe()

    # glover is the default hrf so it should appear in report
    assert "glover" in report_flm.__str__()

    # cosine is the default drift model so it should appear in report
    assert "cosine" in report_flm.__str__()


@pytest.mark.timeout(0)
@pytest.mark.parametrize("height_control", ["fpr", "fdr", "bonferroni", None])
def test_slm_reporting_method(slm, height_control):
    """Test for the second level reporting."""
    c1 = np.eye(len(slm.design_matrix_.columns))[0]
    report_slm = slm.generate_report(
        c1, height_control=height_control, alpha=0.01
    )
    # catches & raises UnicodeEncodeError in HTMLDocument.get_iframe()
    report_slm.get_iframe()


@pytest.mark.timeout(0)
def test_slm_with_flm_as_inputs(flm, contrasts):
    """Test second level reporting when inputs are first level models."""
    model = SecondLevelModel()

    Y = [flm] * 3
    X = pd.DataFrame([[1]] * 3, columns=["intercept"])
    first_level_contrast = contrasts

    model.fit(Y, design_matrix=X)

    c1 = np.eye(len(model.design_matrix_.columns))[0]

    model.generate_report(c1, first_level_contrast=first_level_contrast)


def test_slm_with_dataframes_as_input(tmp_path, shape_3d_default):
    """Test second level reporting when input is a dataframe."""
    file_path = write_fake_bold_img(
        file_path=tmp_path / "img.nii.gz", shape=shape_3d_default
    )

    dfcols = ["subject_label", "map_name", "effects_map_path"]
    dfrows = [
        ["01", "a", file_path],
        ["02", "a", file_path],
        ["03", "a", file_path],
    ]
    niidf = pd.DataFrame(dfrows, columns=dfcols)

    model = SecondLevelModel().fit(niidf)

    c1 = np.eye(len(model.design_matrix_.columns))[0]

    model.generate_report(c1, first_level_contrast="a")


@pytest.mark.timeout(0)
@pytest.mark.parametrize("plot_type", ["slice", "glass"])
def test_report_plot_type(flm, plot_type, contrasts):
    """Smoke test for valid plot type."""
    flm.generate_report(
        contrasts=contrasts,
        plot_type=plot_type,
    )


@pytest.mark.parametrize("plot_type", ["slice", "glass"])
@pytest.mark.parametrize("cut_coords", [None, (5, 4, 3)])
def test_report_cut_coords(flm, plot_type, cut_coords, contrasts):
    """Smoke test for valid cut_coords."""
    flm.generate_report(
        contrasts=contrasts,
        cut_coords=cut_coords,
        display_mode="z",
        plot_type=plot_type,
    )


def test_report_invalid_plot_type(matplotlib_pyplot, flm, contrasts):  # noqa: ARG001
    """Check errors when wrong plot type is requested."""
    with pytest.raises(KeyError, match="junk"):
        flm.generate_report(
            contrasts=contrasts,
            plot_type="junk",
        )

    expected_error = (
        "Invalid plot type provided. "
        "Acceptable options are 'slice' or 'glass'."
    )

    with pytest.raises(ValueError, match=expected_error):
        flm.generate_report(
            contrasts=contrasts,
            display_mode="glass",
            plot_type="junk",
        )


def test_masking_first_level_model(contrasts):
    """Check that using NiftiMasker when instantiating FirstLevelModel \
       doesn't raise Error when calling generate_report().
    """
    shapes, rk = ((7, 7, 7, 5),), 3
    mask, fmri_data, design_matrices = generate_fake_fmri_data_and_design(
        shapes,
        rk,
    )
    masker = NiftiMasker(mask_img=mask)
    masker.fit(fmri_data)
    flm = FirstLevelModel(mask_img=masker).fit(
        fmri_data, design_matrices=design_matrices
    )

    report_flm = flm.generate_report(
        contrasts=contrasts,
        plot_type="glass",
        height_control=None,
        min_distance=15,
        alpha=0.01,
    )

    report_flm.get_iframe()


def test_fir_delays_in_params(contrasts):
    """Check that fir_delays is in the report when hrf_model is fir.

    Also check that it's not in the report when using the default 'glover'.
    """
    shapes, rk = ((7, 7, 7, 5),), 3
    _, fmri_data, design_matrices = generate_fake_fmri_data_and_design(
        shapes, rk
    )
    model = FirstLevelModel(hrf_model="fir", fir_delays=[1, 2, 3])
    model.fit(fmri_data, design_matrices=design_matrices)

    report = model.generate_report(contrasts=contrasts)

    assert "fir_delays" in report.__str__()


@pytest.mark.timeout(0)
def test_drift_order_in_params(contrasts):
    """Check that drift_order is in the report when parameter is drift_model is
    polynomial.
    """
    shapes, rk = ((7, 7, 7, 5),), 3
    _, fmri_data, design_matrices = generate_fake_fmri_data_and_design(
        shapes, rk
    )
    model = FirstLevelModel(drift_model="polynomial", drift_order=3)
    model.fit(fmri_data, design_matrices=design_matrices)

    report = model.generate_report(contrasts=contrasts)

    assert "drift_order" in report.__str__()


def test_flm_generate_report_surface_data(rng):
    """Generate report from flm fitted surface.

    Need a larger image to avoid issues with colormap.
    """
    t_r = 2.0
    events = basic_paradigm()
    n_scans = 10

    mesh = load_fsaverage(mesh="fsaverage5")["pial"]
    data = {}
    for key, val in mesh.parts.items():
        data_shape = (val.n_vertices, n_scans)
        data_part = rng.normal(size=data_shape)
        data[key] = data_part
    fmri_data = SurfaceImage(mesh, data)

    # using smoothing_fwhm for coverage
    model = FirstLevelModel(t_r=t_r, smoothing_fwhm=None)

    model.fit(fmri_data, events=events)

    report = model.generate_report(
        "c0", height_control=None, threshold=DEFAULT_Z_THRESHOLD
    )

    assert isinstance(report, HTMLReport)

    assert "Results table not available for surface data." in report.__str__()


def test_flm_generate_report_surface_data_error(
    surf_mask_1d, surf_img_2d, img_3d_mni
):
    """Generate report from flm fitted surface."""
    model = FirstLevelModel(
        mask_img=surf_mask_1d, t_r=2.0, smoothing_fwhm=None
    )
    events = basic_paradigm()
    model.fit(surf_img_2d(9), events=events)

    with pytest.raises(
        TypeError, match="'bg_img' must a SurfaceImage instance"
    ):
        model.generate_report(
            "c0",
            bg_img=img_3d_mni,
            height_control=None,
            threshold=DEFAULT_Z_THRESHOLD,
        )


@pytest.mark.timeout(0)
def test_carousel_two_runs(
    matplotlib_pyplot,  # noqa: ARG001
    flm,
    slm,
    contrasts,
):
    """Check that a carousel is present when there is more than 1 run."""
    # Second level have a single "run" and do not need a carousel
    report_slm = slm.generate_report()

    assert 'id="carousel-navbar"' not in report_slm.__str__()

    # first level model with one run : no run carousel
    report_one_run = flm.generate_report(contrasts=contrasts)

    assert 'id="carousel-navbar"' not in report_one_run.__str__()

    # first level model with 2 runs : run carousel
    rk = 6
    shapes = ((7, 7, 7, 5), (7, 7, 7, 10))
    _, fmri_data, design_matrices = generate_fake_fmri_data_and_design(
        shapes, rk=rk
    )

    contrasts = np.zeros((1, rk))
    contrasts[0][1] = 1

    flm_two_runs = FirstLevelModel().fit(
        fmri_data, design_matrices=design_matrices
    )

    report = flm_two_runs.generate_report(contrasts=contrasts)

    assert 'id="carousel-navbar"' in report.__str__()


@pytest.mark.parametrize("threshold", [3.09, 2.9, DEFAULT_Z_THRESHOLD])
@pytest.mark.parametrize("height_control", [None, "bonferroni", "fdr", "fpr"])
def test_report_threshold_deprecation_warning(
    flm, contrasts, threshold, height_control
):
    """Check a single warning thrown when threshold==old threshold.

    # TODO (nilearn >= 0.15)
    # remove
    """
    with warnings.catch_warnings(record=True) as warning_list:
        flm.generate_report(
            contrasts=contrasts,
            threshold=threshold,
            height_control=height_control,
        )

    n_warnings = len(
        [x for x in warning_list if issubclass(x.category, FutureWarning)]
    )
    if height_control is None and threshold == 3.09:
        assert n_warnings == 1
    else:
        assert n_warnings == 0
