"""Functionality to create reports for GLM on surface data."""

import datetime
import warnings
from collections import OrderedDict
from decimal import Decimal
from string import Template

import numpy as np
from matplotlib import pyplot as plt

from nilearn import plotting
from nilearn._version import __version__
from nilearn.externals import tempita
from nilearn.plotting.matrix_plotting import (
    plot_contrast_matrix,
    plot_design_matrix,
)
from nilearn.reporting.html_report import (
    HTMLReport,
    _render_parameters_partial,
    _render_warnings_partial,
)
from nilearn.reporting.utils import (
    CSS_PATH,
    HTML_TEMPLATE_PATH,
    TEMPLATE_ROOT_PATH,
    coerce_to_dict,
    figure_to_png_base64,
)
from nilearn.surface import SurfaceImage

with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    from nilearn import glm


def _make_surface_glm_report(
    model,
    contrasts=None,
    title=None,
    threshold=3.09,
    alpha=0.001,
    cluster_threshold=0,
    height_control="fpr",
    bg_img=None,
):
    if bg_img == "MNI152TEMPLATE":
        bg_img = None

    title = f"<br>{title}" if title else ""

    selected_attributes = [
        "subject_label",
        "drift_model",
        "hrf_model",
        "standardize",
        "noise_model",
        "t_r",
        "target_shape",
        "signal_scaling",
        "scaling_axis",
        "smoothing_fwhm",
        "target_affine",
        "slice_time_ref",
    ]
    attribute_units = {
        "t_r": "seconds",
        "high_pass": "Hertz",
    }

    if hasattr(model, "hrf_model") and model.hrf_model == "fir":
        selected_attributes.append("fir_delays")

    if hasattr(model, "drift_model"):
        if model.drift_model == "cosine":
            selected_attributes.append("high_pass")
        elif model.drift_model == "polynomial":
            selected_attributes.append("drift_order")

    if bg_img:
        assert isinstance(bg_img, SurfaceImage)

    selected_attributes.sort()
    parameters = {
        attr_name: getattr(model, attr_name)
        for attr_name in selected_attributes
        if hasattr(model, attr_name)
    }
    for attribute_name_, attribute_unit_ in attribute_units.items():
        if attribute_name_ in parameters:
            parameters[f"{attribute_name_} ({attribute_unit_})"] = parameters[
                attribute_name_
            ]
            parameters.pop(attribute_name_)

    cluster_table_details = OrderedDict()
    threshold = np.around(threshold, 3)
    if height_control:
        cluster_table_details.update({"Height control": height_control})
        if alpha < 0.001:
            alpha = f"{Decimal(alpha):.2E}"
        cluster_table_details.update({"\u03b1": alpha})
        cluster_table_details.update({"Threshold (computed)": threshold})
    else:
        cluster_table_details.update({"Height control": "None"})
        cluster_table_details.update({"Threshold Z": threshold})
    cluster_table_details.update(
        {"Cluster size threshold (vertices)": cluster_threshold}
    )

    masker = getattr(model, "masker_", None)

    mask_plot = None
    if masker and masker.mask_img_:
        fig = masker._create_figure_for_report()
        mask_plot = figure_to_png_base64(fig)

    docstring = model.__doc__
    snippet = docstring.partition("Parameters\n    ----------\n")[0]

    try:
        design_matrices = model.design_matrices_
    except AttributeError:
        design_matrices = []
    design_matrices_dict = _return_design_matrices_dict(design_matrices)

    statistical_maps = None

    contrasts = coerce_to_dict(contrasts)
    contrasts_dict = _return_contrasts_dict(design_matrices, contrasts)

    if contrasts_dict is not None:
        statistical_maps = {}
        statistical_maps = {
            contrast_name: model.compute_contrast(
                contrast_val, output_type="z_score"
            )
            for contrast_name, contrast_val in contrasts.items()
        }

        surf_mesh = None
        if bg_img:
            surf_mesh = bg_img.mesh

        for contrast_name, contrast_val in contrasts.items():
            contrast_map = model.compute_contrast(
                contrast_val, output_type="z_score"
            )
            fig = plotting.plot_surf_stat_map(
                stat_map=contrast_map,
                hemi="left",
                colorbar=True,
                cmap="seismic",
                threshold=threshold,
                bg_map=bg_img,
                surf_mesh=surf_mesh,
            )
            statistical_maps[contrast_name] = {
                "stat_map_img": figure_to_png_base64(fig),
                "contrast_img": contrasts_dict[contrast_name],
            }

    warning_messages = []
    if model.labels_ is None or model.results_ is None:
        warning_messages.append("The model has not been fit yet.")

    body_template_path = HTML_TEMPLATE_PATH / "glm_report.html"
    tpl = tempita.HTMLTemplate.from_filename(
        str(body_template_path),
        encoding="utf-8",
    )

    css_file_path = CSS_PATH / "masker_report.css"
    with css_file_path.open(encoding="utf-8") as css_file:
        css = css_file.read()

    body = tpl.substitute(
        css=css,
        title=f"Statistical Report - {_return_model_type(model)}{title}",
        docstring=snippet,
        warning_messages=_render_warnings_partial(warning_messages),
        design_matrices_dict=design_matrices_dict,
        parameters=_render_parameters_partial(parameters),
        contrasts_dict=contrasts_dict,
        statistical_maps=statistical_maps,
        cluster_table_details=cluster_table_details,
        mask_plot=mask_plot,
        cluster_table=None,
        date=datetime.datetime.now().replace(microsecond=0).isoformat(),
    )

    # revert HTML safe substitutions in CSS sections
    body = body.replace(".pure-g &gt; div", ".pure-g > div")

    head_template_path = (
        TEMPLATE_ROOT_PATH / "html" / "report_head_template.html"
    )
    with head_template_path.open() as head_file:
        head_tpl = Template(head_file.read())

    head_css_file_path = CSS_PATH / "head.css"
    with head_css_file_path.open(encoding="utf-8") as head_css_file:
        head_css = head_css_file.read()

    report = HTMLReport(
        body=body,
        head_tpl=head_tpl,
        head_values={
            "head_css": head_css,
            "version": __version__,
            "page_title": (
                "Statistical Report - " f"{_return_model_type(model)}{title}"
            ),
        },
    )
    report.height = 800
    report.width = 1000
    return report


def _return_design_matrices_dict(design_matrices):
    # avoid circular import
    from nilearn.reporting.glm_reporter import (
        _plot_to_svg,
        _resize_plot_inches,
    )

    if not design_matrices:
        return None

    design_matrices_dict = {}
    for dmtx_count, design_matrix in enumerate(design_matrices, start=1):
        dmtx_plot = plot_design_matrix(design_matrix)
        dmtx_title = f"Run {dmtx_count}"
        if len(design_matrices) > 1:
            plt.title(dmtx_title, y=1.025, x=-0.1)
        dmtx_plot = _resize_plot_inches(dmtx_plot, height_change=0.3)
        url_design_matrix_svg = _plot_to_svg(dmtx_plot)
        # prevents sphinx-gallery & jupyter from scraping & inserting plots
        plt.close()

        design_matrices_dict[dmtx_title] = url_design_matrix_svg
    return design_matrices_dict


def _return_contrasts_dict(design_matrices, contrasts):
    # avoid circular import
    from nilearn.reporting.glm_reporter import _plot_to_svg

    if not design_matrices or not contrasts:
        return None

    contrasts_dict = {}
    for design_matrix in design_matrices:
        for contrast_name, contrast_data in contrasts.items():
            contrast_plot = plot_contrast_matrix(
                contrast_data, design_matrix, colorbar=True
            )
            contrast_plot.set_xlabel(contrast_name)
            contrast_plot.figure.set_figheight(2)
            contrast_plot.figure.set_tight_layout(True)
            url_contrast_plot_svg = _plot_to_svg(contrast_plot)
            # prevents sphinx-gallery & jupyter
            # from scraping & inserting plots
            plt.close()
            contrasts_dict[contrast_name] = url_contrast_plot_svg

    return contrasts_dict


def _return_model_type(model):
    if isinstance(model, glm.first_level.FirstLevelModel):
        return "First Level Model"
    elif isinstance(model, glm.second_level.SecondLevelModel):
        return "Second Level Model"
