import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from nilearn.externals import tempita
from nilearn.plotting import (
    plot_contrast_matrix,
    plot_design_matrix,
    plot_design_matrix_correlation,
)
from nilearn.reporting.utils import (
    figure_to_png_base64,
)


def generate_design_matrices_figures(
    design_matrices, design_matrices_dict=None, output=None
):
    """Generate plot for design matrices and their correlation matrices.

    After generating the figure it can either :

    - convert it to bytes for insertion into HTML report
    - save it to disk if the appropriate "output" was passed

    design_matrices_dict is a dict-like (tempita.bunc)
    that contains the figure (as bytes or relative path).
    A tempita bunch is used to facilitate injecting its content
    into HTML templates.
    If a design_matrices_dict is passed its content will be updated.

    Returns
    -------
    design_matrices_dict : tempita.bunch

        design_matrices_dict[i_run].design_matrix
        design_matrices_dict[i_run].correlation_matrix

    """
    if design_matrices_dict is None:
        design_matrices_dict = tempita.bunch()

    if design_matrices is None:
        return design_matrices_dict

    for i_run, design_matrix in enumerate(design_matrices):
        dmtx_plot = plot_design_matrix(design_matrix)
        dmtx_plot = resize_plot_inches(dmtx_plot, height_change=0.3)
        dmtx_fig = None
        if output:
            # the try is mostly here in case badly formed dict
            try:
                dmtx_fig = output["design_matrices_dict"][i_run][
                    "design_matrix_png"
                ]
                dmtx_plot.figure.savefig(output["dir"] / dmtx_fig)
            except Exception:  # pragma: no cover
                dmtx_fig = None
        if dmtx_fig is None:
            dmtx_fig = figure_to_png_base64(dmtx_plot)
        # prevents sphinx-gallery & jupyter
        # from scraping & inserting plots
        plt.close("all")

        dmtx_cor_fig = None
        # in case of second level model with a single regressor
        # (for example one-sample t-test)
        # no point in plotting the correlation
        if (
            isinstance(design_matrix, np.ndarray)
            and design_matrix.shape[1] > 1
        ) or (
            isinstance(design_matrix, pd.DataFrame)
            and len(design_matrix.columns) > 1
        ):
            dmtx_cor_plot = plot_design_matrix_correlation(
                design_matrix, tri="diag"
            )
            dmtx_cor_plot = resize_plot_inches(
                dmtx_cor_plot, height_change=0.3
            )
            if output:
                try:
                    dmtx_cor_fig = output["design_matrices_dict"][i_run][
                        "correlation_matrix_png"
                    ]
                    dmtx_cor_plot.figure.savefig(output["dir"] / dmtx_cor_fig)
                except KeyError:  # pragma: no cover
                    dmtx_cor_fig = None
            if dmtx_cor_fig is None:
                dmtx_cor_fig = figure_to_png_base64(dmtx_cor_plot)
            # prevents sphinx-gallery & jupyter
            # from scraping & inserting plots
            plt.close("all")

        if i_run not in design_matrices_dict:
            design_matrices_dict[i_run] = tempita.bunch(
                design_matrix=None, correlation_matrix=None
            )

        design_matrices_dict[i_run]["design_matrix_png"] = dmtx_fig
        design_matrices_dict[i_run]["correlation_matrix_png"] = dmtx_cor_fig

    return design_matrices_dict


def generate_contrast_matrices_figures(
    design_matrices, contrasts=None, contrasts_dict=None, output=None
):
    """Generate plot for contrasts matrices.

    After generating the figure it can either :

    - convert it to bytes for insertion into HTML report
    - save it to disk if the appropriate "output" was passed

    contrasts_dict is a dict-like (tempita.bunc)
    that contains the figure (as bytes or relative path).
    A tempita bunch is used to facilitate injecting its content
    into HTML templates.
    If a contrasts_dict is passed its content will be updated.

    Returns
    -------
    contrasts_dict : tempita.bunch

        contrasts_dict[contrast_name]


    """
    if contrasts_dict is None:
        contrasts_dict = tempita.bunch()

    if design_matrices is None or not contrasts:
        return contrasts_dict

    for i_run, design_matrix in enumerate(design_matrices):
        tmp = {}
        for contrast_name, contrast_data in contrasts.items():
            contrast_plot = plot_contrast_matrix(
                contrast_data, design_matrix, colorbar=True
            )

            contrast_plot.set_xlabel(contrast_name)

            contrast_plot.figure.set_figheight(2)

            contrast_fig = None
            if output:
                try:
                    contrast_fig = output["contrasts_dict"][i_run][
                        contrast_name
                    ]
                    contrast_plot.figure.savefig(output["dir"] / contrast_fig)
                except KeyError:  # pragma: no cover
                    contrast_fig = None
            if contrast_fig is None:
                with warnings.catch_warnings():
                    # ignore some warnings that we cannot avoid
                    warnings.filterwarnings(
                        "ignore", message=".*constrained_layout not applied.*"
                    )
                    contrast_fig = figure_to_png_base64(contrast_plot)
            # prevents sphinx-gallery & jupyter
            # from scraping & inserting plots
            plt.close("all")

            tmp[contrast_name] = contrast_fig

        contrasts_dict[i_run] = tempita.bunch(**tmp)

    return contrasts_dict


def resize_plot_inches(plot, width_change=0, height_change=0):
    """Accept a matplotlib figure or axes object and resize it (in inches).

    Returns the original object.

    Parameters
    ----------
    plot : matplotlib.Figure() or matplotlib.Axes()
        The matplotlib Figure/Axes object to be resized.

    width_change : float, default=0
        The amount of change to be added on to original width.
        Use negative values for reducing figure dimensions.

    height_change : float, default=0
        The amount of change to be added on to original height.
        Use negative values for reducing figure dimensions.

    Returns
    -------
    plot : matplotlib.Figure() or matplotlib.Axes()
        The matplotlib Figure/Axes object after being resized.

    """
    if not isinstance(plot, (plt.Figure)):
        orig_size = plot.figure.get_size_inches()
    else:
        orig_size = plot.get_size_inches()

    new_size = (
        orig_size[0] + width_change,
        orig_size[1] + height_change,
    )

    if not isinstance(plot, (plt.Figure)):
        plot.figure.set_size_inches(new_size, forward=True)
    else:
        plot.set_size_inches(new_size, forward=True)

    return plot
