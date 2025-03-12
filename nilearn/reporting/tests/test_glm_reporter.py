import numpy as np
import pandas as pd
import pytest

from nilearn._utils.data_gen import (
    basic_paradigm,
    generate_fake_fmri_data_and_design,
)
from nilearn.conftest import _img_mask_mni, _make_surface_mask
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.second_level import SecondLevelModel
from nilearn.maskers import NiftiMasker
from nilearn.reporting import HTMLReport


@pytest.fixture
def rk():
    return 3


@pytest.fixture
def contrasts(rk):
    c = np.zeros((1, rk))
    c[0] = 1
    return c


@pytest.fixture()
def flm(rk):
    """Generate first level model."""
    shapes = ((7, 7, 7, 5),)
    mask, fmri_data, design_matrices = generate_fake_fmri_data_and_design(
        shapes, rk=rk
    )
    # generate_fake_fmri_data_and_design
    return FirstLevelModel(mask_img=mask).fit(
        fmri_data, design_matrices=design_matrices
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


@pytest.mark.parametrize("model", [FirstLevelModel, SecondLevelModel])
@pytest.mark.parametrize("bg_img", [_img_mask_mni(), _make_surface_mask()])
def test_empty_surface_reports(tmp_path, model, bg_img):
    """Test that empty reports on unfitted model can be generated."""
    report = model().generate_report(bg_img=bg_img)

    assert isinstance(report, HTMLReport)

    report.save_as_html(tmp_path / "tmp.html")
    assert (tmp_path / "tmp.html").exists()


@pytest.mark.parametrize("height_control", ["fdr", "bonferroni", None])
def test_flm_reporting_height_control(flm, height_control, contrasts):
    """Test for first level model reporting."""
    report_flm = flm.generate_report(
        contrasts=contrasts,
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

    # glover is the default hrf so it should appear in report
    # but not details related to fir hrf
    assert "glover" in report_flm.__str__()
    assert "fir_delays" not in report_flm.__str__()

    # cosine is the default drift model so it should appear in report
    # but not details related to polynomial drift model
    assert "cosine" in report_flm.__str__()
    assert "drift_order" not in report_flm.__str__()


@pytest.mark.parametrize("height_control", ["fpr", "fdr", "bonferroni", None])
def test_slm_reporting_method(slm, height_control):
    """Test for the second level reporting."""
    c1 = np.eye(len(slm.design_matrix_.columns))[0]
    report_slm = slm.generate_report(
        c1, height_control=height_control, threshold=2, alpha=0.01
    )
    # catches & raises UnicodeEncodeError in HTMLDocument.get_iframe()
    report_slm.get_iframe()


@pytest.mark.parametrize("plot_type", ["slice", "glass"])
def test_report_plot_type(flm, plot_type, contrasts):
    """Smoke test for valid plot type."""
    flm.generate_report(
        contrasts=contrasts,
        plot_type=plot_type,
        threshold=2.76,
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
        threshold=2.76,
    )


def test_report_invalid_plot_type(flm, contrasts):
    with pytest.raises(KeyError, match="junk"):
        flm.generate_report(
            contrasts=contrasts,
            plot_type="junk",
            threshold=2.76,
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
            threshold=2.76,
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
        threshold=2,
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

    report = model.generate_report(contrasts=contrasts, threshold=0.1)

    assert "fir_delays" in report.__str__()
    assert "glover" not in report.__str__()


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
    assert "cosine" not in report.__str__()


def test_flm_generate_report_error_with_surface_data(
    surf_mask_1d, surf_img_2d
):
    """Generate report from flm fitted surface."""
    model = FirstLevelModel(mask_img=surf_mask_1d, t_r=2.0)
    events = basic_paradigm()
    model.fit(surf_img_2d(9), events=events)

    report = model.generate_report("c0")

    assert isinstance(report, HTMLReport)
