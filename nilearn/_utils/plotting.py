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
    design_matrices, out_dir=None, output=None
):
    if design_matrices is None:
        return None

    design_matrices_dict = tempita.bunch()

    for i_run, design_matrix in enumerate(design_matrices, start=1):
        dmtx_plot = plot_design_matrix(design_matrix)
        dmtx_plot = resize_plot_inches(dmtx_plot, height_change=0.3)

        if out_dir and output:
            dmtx_plot.figure.savefig(
                out_dir
                / output["design_matrices_dict"][i_run]["design_matrix"]
            )

        else:
            dmtx_png = figure_to_png_base64(dmtx_plot)
            # prevents sphinx-gallery & jupyter
            # from scraping & inserting plots
            plt.close("all")

        # in case of second level model with a single regressor
        # (for example one-sample t-test)
        # no point in plotting the correlation
        dmtx_cor_png = None
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
            if out_dir and output:
                dmtx_cor_plot.figure.savefig(
                    out_dir
                    / output["design_matrices_dict"][i_run][
                        "correlation_matrix"
                    ]
                )

            else:
                dmtx_cor_png = figure_to_png_base64(dmtx_cor_plot)
                # prevents sphinx-gallery & jupyter
                # from scraping & inserting plots
                plt.close("all")

        design_matrices_dict[i_run] = tempita.bunch(
            design_matrix=dmtx_png, correlation_matrix=dmtx_cor_png
        )

    return design_matrices_dict


def generate_constrat_matrices_figures(
    design_matrices, contrasts, out_dir=None, output=None
):
    if design_matrices is None or not contrasts:
        return None

    contrasts_dict = {}
    for i_run, design_matrix in enumerate(design_matrices):
        for contrast_name, contrast_data in contrasts.items():
            contrast_plot = plot_contrast_matrix(
                contrast_data, design_matrix, colorbar=True
            )

            contrast_plot.set_xlabel(contrast_name)

            contrast_plot.figure.set_figheight(2)

            if out_dir and output:
                contrast_plot.figure.savefig(
                    out_dir / output["contrasts_dict"][i_run][contrast_name]
                )
            else:
                url_contrast_plot_png = figure_to_png_base64(contrast_plot)
                # prevents sphinx-gallery & jupyter
                # from scraping & inserting plots
                plt.close("all")
                contrasts_dict[contrast_name] = url_contrast_plot_png

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
