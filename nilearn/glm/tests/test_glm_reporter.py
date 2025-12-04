import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nilearn._utils.data_gen import (
    basic_paradigm,
    generate_fake_fmri_data_and_design,
    write_fake_bold_img,
)
from nilearn._utils.helpers import is_matplotlib_installed
from nilearn.conftest import _img_mask_mni, _make_surface_mask
from nilearn.datasets import load_fsaverage
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.second_level import SecondLevelModel
from nilearn.maskers import NiftiMasker
from nilearn.reporting import HTMLReport, make_glm_report
from nilearn.reporting.html_report import MISSING_ENGINE_MSG
from nilearn.surface import SurfaceImage


def check_glm_report(
    model: FirstLevelModel | SecondLevelModel,
    view=False,
    pth: Path | None = None,
    extend_includes: list[str] | None = None,
    extend_excludes: list[str] | None = None,
    **kwargs,
) -> HTMLReport:
    """Generate a GLM report and run generic checks on it.

    Parameters
    ----------
    model : FirstLevelModel or SecondLevelModel
        Model that generated the report

    view: bool, default=False
        if True the report is open in browser
        only used for debugging locally

    pth: Path or None, default=None
        Where to save the report

    extend_includes : Iterable[str] | None, default=None
        The function will check
        for the presence in the report
        of each string in this iterable.

    extend_includes : Iterable[str] | None, default=None
        The function will check
        for the absence in the report
        of each string in this iterable.
    """
    report = model.generate_report(**kwargs)

    assert isinstance(report, HTMLReport)

    # catches & raises UnicodeEncodeError in HTMLDocument.get_iframe()
    # in case certain unicode characters are mishandled,
    # like the greek alpha symbol.
    report.get_iframe()

    # only for debugging
    if view:
        report.open_in_browser()

    if pth:
        # save to disk
        # useful for visual inspection
        # for manual checks or in case of test failure
        report.save_as_html(pth / "tmp.html")
        assert (pth / "tmp.html").exists()

    includes = []
    excludes = []

    # check the navbar and its css is there
    includes.append('<nav class="navbar pure-g fw-bold" id="menu"')
    includes.append("Adapted from Pure CSS navbar")  # css

    if is_matplotlib_installed():
        excludes.extend(
            [MISSING_ENGINE_MSG, 'grey">No plotting engine found</p>']
        )
    else:
        includes.extend(
            [MISSING_ENGINE_MSG, 'grey">No plotting engine found</p>']
        )

    # 'Contrasts' and 'Statistical maps' should appear
    # as section and in navbar
    # if report was generated with contrasts.
    contrast_present_checks = [
        '<a id="navbar-contrasts-link',
        '<a href="#statistical-maps',
    ]
    # There should be a warning
    # that no contrast was passed
    # if that's the case
    contrasts_missing_checks = [
        "No contrast passed during report generation.",
    ]

    has_contrasts = "contrasts" in kwargs and kwargs["contrasts"] is not None

    if has_contrasts:
        includes.extend(contrast_present_checks)
        excludes.extend(contrasts_missing_checks)
    else:
        excludes.extend(contrast_present_checks)

    if not model.__sklearn_is_fitted__():
        includes.extend(
            [
                "This estimator has not been fit yet.",
                "No statistical map was provided.",
            ]
        )
        if is_matplotlib_installed():
            includes.extend(
                [
                    "No mask was provided.",
                ]
            )

        # no design matrix in navbar if model not fitted
        excludes.append('<a id="navbar-matrix-link')

    else:
        includes.extend(
            [
                "The mask includes",  # check that mask coverage is there
            ]
        )
        if is_matplotlib_installed():
            includes.extend(
                [
                    'id="design-matrix-',
                    'id="mask-',
                    'id="statistical-maps-',
                ]
            )

        if not has_contrasts:
            # the no contrast warning only appears for fitted models
            includes.extend(contrasts_missing_checks)

        if (
            isinstance(model, SecondLevelModel)
            or len(model.design_matrices_) < 2
        ):
            # SecondLevelModel have a single "run" and do not need a carousel
            # FirstLevelModel with a single neither
            excludes.append('id="carousel-navbar"')
        else:
            includes.append('id="carousel-navbar"')

    if extend_includes is not None:
        includes.extend(extend_includes)
    for check in set(includes):
        assert check in str(report)

    if extend_excludes is not None:
        excludes.extend(extend_excludes)
    for check in set(excludes):
        assert check not in str(report)

    return report


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
    _, fmri_data, design_matrices = generate_fake_fmri_data_and_design(
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


@pytest.mark.slow
def test_flm_report_no_activation_found(flm, contrasts, tmp_path):
    """Check presence message of no activation found.

    We use random data, so we should not get activations.
    """
    check_glm_report(
        model=flm,
        pth=tmp_path,
        extend_includes=["No suprathreshold cluster"],
        contrasts=contrasts,
    )


@pytest.mark.parametrize("model", [FirstLevelModel, SecondLevelModel])
@pytest.mark.parametrize("bg_img", [_img_mask_mni(), _make_surface_mask()])
def test_empty_reports(tmp_path, model, bg_img):
    """Test that empty reports on unfitted model can be generated.

    Both for volume and surface data.
    """
    check_glm_report(
        model=model(smoothing_fwhm=None),
        pth=tmp_path,
        bg_img=bg_img,
    )


def test_flm_reporting_no_contrasts(flm, tmp_path):
    """Test for model report can be generated with no contrasts."""
    check_glm_report(
        model=flm,
        pth=tmp_path,
        plot_type="glass",
        contrasts=None,
        min_distance=15,
        alpha=0.01,
    )


@pytest.mark.slow
@pytest.mark.parametrize("height_control", ["fdr", "bonferroni", None])
def test_flm_reporting_height_control(
    flm, height_control, contrasts, tmp_path
):
    """Test for first level model reporting.

    Also checks that passing threshold different from the default
    will throw a warning when height_control is not None.
    """
    with warnings.catch_warnings(record=True) as warnings_list:
        check_glm_report(
            model=flm,
            pth=tmp_path,
            # glover / cosine are the default
            # hrf / drift model so they should appear in report
            extend_includes=["glover", "cosine"],
            contrasts=contrasts,
            plot_type="glass",
            height_control=height_control,
            min_distance=15,
            alpha=0.01,
            threshold=2,
        )
    if height_control is not None:
        assert any("will not be used with" in str(x) for x in warnings_list)


@pytest.mark.slow
@pytest.mark.parametrize("height_control", ["fpr", "fdr", "bonferroni", None])
def test_slm_reporting_method(slm, height_control):
    """Test for the second level reporting."""
    c1 = np.eye(len(slm.design_matrix_.columns))[0]

    check_glm_report(
        slm, contrasts=c1, height_control=height_control, alpha=0.01
    )


@pytest.mark.slow
def test_slm_with_flm_as_inputs(flm, contrasts):
    """Test second level reporting when inputs are first level models."""
    model = SecondLevelModel()

    Y = [flm] * 3
    X = pd.DataFrame([[1]] * 3, columns=["intercept"])
    first_level_contrast = contrasts

    model.fit(Y, design_matrix=X)

    c1 = np.eye(len(model.design_matrix_.columns))[0]

    check_glm_report(
        model,
        contrasts=c1,
        first_level_contrast=first_level_contrast,
        # the following are to avoid warnings
        threshold=1e-8,
        height_control=None,
    )


@pytest.mark.slow
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

    check_glm_report(
        model,
        contrasts=c1,
        first_level_contrast="a",
        # the following are to avoid warnings
        threshold=1e-8,
        height_control=None,
    )


@pytest.mark.slow
@pytest.mark.parametrize("plot_type", ["slice", "glass"])
def test_report_plot_type(flm, plot_type, contrasts):
    """Smoke test for valid plot type."""
    check_glm_report(
        flm,
        contrasts=contrasts,
        plot_type=plot_type,
        # the following are to avoid warnings
        threshold=1e-8,
        height_control=None,
    )


@pytest.mark.slow
@pytest.mark.parametrize("plot_type", ["slice", "glass"])
@pytest.mark.parametrize("cut_coords", [None, (5, 4, 3)])
def test_report_cut_coords(flm, plot_type, cut_coords, contrasts):
    """Smoke test for valid cut_coords."""
    check_glm_report(
        flm,
        contrasts=contrasts,
        cut_coords=cut_coords,
        display_mode="z",
        plot_type=plot_type,
        # the following are to avoid warnings
        threshold=1e-8,
        height_control=None,
    )


@pytest.mark.slow
def test_report_invalid_plot_type(flm, contrasts):
    """Check errors when wrong plot type is requested."""
    with pytest.raises(KeyError, match="junk"):
        flm.generate_report(
            contrasts=contrasts,
            plot_type="junk",
            # the following are to avoid warnings
            threshold=1e-8,
            height_control=None,
        )
    if is_matplotlib_installed():
        with pytest.raises(ValueError, match="'plot_type' must be one of"):
            flm.generate_report(
                contrasts=contrasts,
                display_mode="glass",
                plot_type="junk",
                # the following are to avoid warnings
                threshold=1e-8,
                height_control=None,
            )


@pytest.mark.slow
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

    check_glm_report(
        flm,
        contrasts=contrasts,
        plot_type="glass",
        min_distance=15,
        alpha=0.01,
        # the following are to avoid warnings
        threshold=1e-8,
        height_control=None,
    )


@pytest.mark.slow
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

    # FIXME:
    # matrices were passed at fit time
    # so fir_delays should not appear in report
    # as we do not know which HRF was used to build the matrix
    check_glm_report(
        model,
        contrasts=contrasts,
        extend_includes=["fir_delays"],
        # the following are to avoid warnings
        threshold=1e-8,
        height_control=None,
    )


@pytest.mark.slow
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

    check_glm_report(
        model,
        contrasts=contrasts,
        extend_includes=["drift_order"],
        # the following are to avoid warnings
        threshold=1e-8,
        height_control=None,
    )


@pytest.mark.slow
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

    check_glm_report(
        model,
        contrasts="c0",
        # the following are to avoid warnings
        threshold=1e-8,
        height_control=None,
    )


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
            # the following are to avoid warnings
            threshold=1e-8,
            height_control=None,
        )


@pytest.mark.slow
def test_carousel_several_runs(
    matplotlib_pyplot,  # noqa: ARG001
    contrasts,
):
    """Check that a carousel is present when there is more than 1 run."""
    # first level model with 3 runs : run carousel
    # TODO
    # change name of design matrix columns and use contrast expression
    # to have different plots for each run
    rk = 6
    shapes = ((7, 7, 7, 5), (7, 7, 7, 10), (7, 7, 7, 15))
    _, fmri_data, design_matrices = generate_fake_fmri_data_and_design(
        shapes, rk=rk
    )

    contrasts = np.zeros((1, rk))
    contrasts[0][1] = 1

    flm_two_runs = FirstLevelModel().fit(
        fmri_data, design_matrices=design_matrices
    )

    report = check_glm_report(
        flm_two_runs,
        contrasts=contrasts,
        # the following are to avoid warnings
        threshold=1e-8,
        height_control=None,
    )

    # 3 runs should be in the carousel
    assert str(report).count('id="carousel-obj-') == len(shapes)


def test_report_make_glm_deprecation_warning(flm, contrasts):
    """Test deprecation warning for nilearn.reporting.make_glm_report.

    # TODO (nilearn >= 0.15)
    # remove
    """
    with pytest.warns(DeprecationWarning):
        make_glm_report(
            flm,
            contrasts=contrasts,
        )
