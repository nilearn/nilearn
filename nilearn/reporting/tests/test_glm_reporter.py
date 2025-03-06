import numpy as np
import pandas as pd
import pytest

from nilearn._utils.data_gen import (
    basic_paradigm,
    generate_fake_fmri_data_and_design,
)
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.first_level.design_matrix import (
    make_first_level_design_matrix,
)
from nilearn.glm.second_level import SecondLevelModel
from nilearn.maskers import NiftiMasker
from nilearn.reporting import HTMLReport, make_glm_report
from nilearn.reporting import glm_reporter as glmr
from nilearn.reporting.glm_reporter import (
    _make_surface_glm_report,
)


@pytest.fixture()
def flm():
    """Generate first level model."""
    shapes, rk = ((7, 7, 7, 5),), 3
    mask, fmri_data, design_matrices = generate_fake_fmri_data_and_design(
        shapes, rk
    )
    # generate_fake_fmri_data_and_design
    return FirstLevelModel(mask_img=mask).fit(
        fmri_data, design_matrices=design_matrices
    )


@pytest.mark.parametrize("height_control", ["fpr", "fdr", "bonferroni", None])
def test_flm_reporting(flm, height_control):
    """Smoke test for first level model reporting."""
    contrast = np.eye(3)[1]
    report_flm = glmr.make_glm_report(
        flm,
        contrast,
        plot_type="glass",
        height_control=height_control,
        min_distance=15,
        alpha=0.01,
        threshold=2,
    )
    # catches & raises UnicodeEncodeError in HTMLDocument.get_iframe()
    # in case certain unicode characters are mishandled,
    # like the greek alpha symbol.
    report_flm.get_iframe()


def test_flm_reporting_method(flm):
    """Smoke test for the first level generate method."""
    contrast = np.eye(3)[1]
    flm.generate_report(
        contrast,
        plot_type="glass",
        min_distance=15,
        alpha=0.01,
        threshold=2,
    )


@pytest.fixture()
def slm():
    """Generate a fitted second level model."""
    shapes = ((7, 7, 7, 1),)
    _, fmri_data, _ = generate_fake_fmri_data_and_design(shapes)
    model = SecondLevelModel()
    Y = [fmri_data[0]] * 2
    X = pd.DataFrame([[1]] * 2, columns=["intercept"])
    return model.fit(Y, design_matrix=X)


@pytest.mark.parametrize("height_control", ["fpr", "fdr", "bonferroni", None])
def test_slm_reporting_method(slm, height_control):
    """Smoke test for the second level reporting."""
    c1 = np.eye(len(slm.design_matrix_.columns))[0]
    report_slm = glmr.make_glm_report(
        slm, c1, height_control=height_control, threshold=2, alpha=0.01
    )
    # catches & raises UnicodeEncodeError in HTMLDocument.get_iframe()
    report_slm.get_iframe()


def test_slm_reporting(slm):
    """Smoke test for the second level model generate method."""
    c1 = np.eye(len(slm.design_matrix_.columns))[0]
    slm.generate_report(c1, threshold=2, alpha=0.01)


@pytest.mark.parametrize("cut_coords", [None, (5, 4, 3)])
def test_stat_map_to_svg_slice_z(img_3d_mni, cut_coords):
    table_details = pd.DataFrame.from_dict({"junk": 0}, orient="index")
    glmr._stat_map_to_svg(
        stat_img=img_3d_mni,
        bg_img=None,
        cut_coords=cut_coords,
        display_mode="ortho",
        plot_type="slice",
        table_details=table_details,
        threshold=2.76,
    )


@pytest.mark.parametrize("cut_coords", [None, (5, 4, 3)])
def test_stat_map_to_svg_glass_z(img_3d_mni, cut_coords):
    table_details = pd.DataFrame.from_dict({"junk": 0}, orient="index")
    glmr._stat_map_to_svg(
        stat_img=img_3d_mni,
        bg_img=None,
        cut_coords=cut_coords,
        display_mode="z",
        plot_type="glass",
        table_details=table_details,
        threshold=2.76,
    )


@pytest.mark.parametrize("cut_coords", [None, (5, 4, 3)])
def test_stat_map_to_svg_invalid_plot_type(img_3d_mni, cut_coords):
    expected_error = (
        "Invalid plot type provided. "
        "Acceptable options are 'slice' or 'glass'."
    )
    with pytest.raises(ValueError, match=expected_error):
        glmr._stat_map_to_svg(
            stat_img=img_3d_mni,
            bg_img=None,
            cut_coords=cut_coords,
            display_mode="z",
            plot_type="junk",
            table_details={"junk": 0},
            threshold=2.76,
        )


def _make_dummy_contrasts_dmtx():
    frame_times = np.linspace(0, 127 * 1.0, 128)
    dmtx = make_first_level_design_matrix(
        frame_times,
        drift_model="polynomial",
        drift_order=3,
    )
    contrast = {"test": np.ones(4)}
    return contrast, dmtx


def test_masking_first_level_model():
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
    contrast = np.eye(3)[1]

    report_flm = flm.generate_report(
        contrast,
        plot_type="glass",
        height_control=None,
        min_distance=15,
        alpha=0.01,
        threshold=2,
    )

    report_flm.get_iframe()


@pytest.mark.parametrize("hrf_model", ["glover", "fir"])
def test_fir_delays_in_params(hrf_model):
    """Check that fir_delays is in the report when hrf_model is fir.

    Also check that it's not in the report when using the default 'glover'.
    """
    shapes, rk = ((7, 7, 7, 5),), 3
    _, fmri_data, design_matrices = generate_fake_fmri_data_and_design(
        shapes, rk
    )
    model = FirstLevelModel(hrf_model=hrf_model, fir_delays=[1, 2, 3])
    model.fit(fmri_data, design_matrices=design_matrices)

    contrast = np.eye(3)[1]
    report = model.generate_report(contrast)

    if hrf_model == "fir":
        assert "fir_delays" in report.__str__()
    else:
        assert "fir_delays" not in report.__str__()


@pytest.mark.parametrize("drift_model", ["cosine", "polynomial"])
def test_drift_order_in_params(drift_model):
    """Check that drift_order is in the report when parameter is drift_model is
    polynomial.

    Also check that it's not in the report when using the default 'cosine'.
    """
    shapes, rk = ((7, 7, 7, 5),), 3
    _, fmri_data, design_matrices = generate_fake_fmri_data_and_design(
        shapes, rk
    )
    model = FirstLevelModel(drift_model=drift_model, drift_order=3)
    model.fit(fmri_data, design_matrices=design_matrices)

    contrast = np.eye(3)[1]
    report = model.generate_report(contrast)

    if drift_model == "polynomial":
        assert "drift_order" in report.__str__()
    else:
        assert "drift_order" not in report.__str__()


# -----------------------surface tests--------------------------------------- #


def test_flm_generate_report_error_with_surface_data(
    surf_mask_1d, surf_img_2d
):
    """Generate report from flm fitted surface."""
    model = FirstLevelModel(mask_img=surf_mask_1d, t_r=2.0)
    events = basic_paradigm()
    model.fit(surf_img_2d(9), events=events)
    report = model.generate_report("c0")

    assert isinstance(report, HTMLReport)

    report = make_glm_report(model, "c0")

    assert isinstance(report, HTMLReport)


@pytest.mark.parametrize("model", [FirstLevelModel, SecondLevelModel])
def test_empty_surface_reports(tmp_path, model, surf_img_1d):
    """Test that empty surface reports on unfitted model can be generated."""
    report = _make_surface_glm_report(model(), bg_img=surf_img_1d)

    assert isinstance(report, HTMLReport)

    report.save_as_html(tmp_path / "tmp.html")
    assert (tmp_path / "tmp.html").exists()


def test_empty_surface_reports_errors():
    """Test errors surface reports."""
    with pytest.raises(TypeError, match="must a SurfaceImage instance"):
        _make_surface_glm_report(
            FirstLevelModel(), bg_img="not a surface image"
        )
