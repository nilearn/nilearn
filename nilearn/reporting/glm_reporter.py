"""Functionality to create an HTML report using a fitted GLM & contrasts."""
# TODO (nilearn >= 0.15.0) remove this module

import warnings

from nilearn._utils.html_document import HEIGHT_DEFAULT, WIDTH_DEFAULT
from nilearn._utils.logger import find_stack_level
from nilearn.reporting.html_report import HTMLReport


def make_glm_report(
    model,
    contrasts=None,
    first_level_contrast=None,
    title=None,
    bg_img="MNI152TEMPLATE",
    threshold=3.09,
    alpha=0.001,
    cluster_threshold=0,
    height_control="fpr",
    two_sided=False,
    min_distance=8.0,
    plot_type="slice",
    cut_coords=None,
    display_mode=None,
    report_dims=(WIDTH_DEFAULT, HEIGHT_DEFAULT),
) -> HTMLReport:
    """Return HTMLReport object \
    for a report which shows all important aspects of a fitted GLM.

    From release 0.15.0, `make_glm_report` will be deprecated in favor of glm
    model's `generate_report` method.

    Returns
    -------
    report_text : HTMLReport Object
        Contains the HTML code for the :term:`GLM` Report.

    """
    make_glm_report_deprecation = (
        "From release 0.15.0, make_glm_report will be deprecated. "
        "Use generate_report method of the GLM model instead."
    )
    warnings.warn(
        category=FutureWarning,
        message=make_glm_report_deprecation,
        stacklevel=find_stack_level(),
    )
    if model.__str__() == "Second Level Model":
        return model.generate_report(
            contrasts=contrasts,
            first_level_contrast=first_level_contrast,
            title=title,
            bg_img=bg_img,
            threshold=threshold,
            alpha=alpha,
            cluster_threshold=cluster_threshold,
            height_control=height_control,
            two_sided=two_sided,
            min_distance=min_distance,
            plot_type=plot_type,
            cut_coords=cut_coords,
            display_mode=display_mode,
            report_dims=report_dims,
        )
    else:
        return model.generate_report(
            contrasts=contrasts,
            title=title,
            bg_img=bg_img,
            threshold=threshold,
            alpha=alpha,
            cluster_threshold=cluster_threshold,
            height_control=height_control,
            two_sided=two_sided,
            min_distance=min_distance,
            plot_type=plot_type,
            cut_coords=cut_coords,
            display_mode=display_mode,
            report_dims=report_dims,
        )
