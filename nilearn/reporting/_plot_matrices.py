"""
This module implements plotting functions useful to report analysis results.

Author: Martin Perez-Guevara, Elvis Dohmatob, 2017
"""

import numpy as np
import matplotlib.pyplot as plt

from nilearn.glm.first_level import check_design_matrix
from nilearn.glm.contrasts import expression_to_contrast_vector


def plot_design_matrix(design_matrix, rescale=True, ax=None, output_file=None):
    """Plot a design matrix provided as a DataFrame

    Parameters
    ----------
    design matrix : pandas DataFrame,
        Describes a design matrix.

    rescale : bool, optional
        Rescale columns magnitude for visualization or not.

    ax : axis handle, optional
        Handle to axis onto which we will draw design matrix.

    output_file: string or None, optional,
        The name of an image file to export the plot to. Valid extensions
        are .png, .pdf, .svg. If output_file is not None, the plot
        is saved to a file, and the display is closed.

    Returns
    -------
    ax: axis handle
        The axis used for plotting.
    """
    # We import _set_mpl_backend because just the fact that we are
    # importing it sets the backend

    # normalize the values per column for better visualization
    _, X, names = check_design_matrix(design_matrix)
    if rescale:
        X = X / np.maximum(1.e-12, np.sqrt(
            np.sum(X ** 2, 0)))  # pylint: disable=no-member
    if ax is None:
        max_len = np.max([len(str(name)) for name in names])
        fig_height = 1 + .1 * X.shape[0] + .04 * max_len
        if fig_height < 3:
            fig_height = 3
        elif fig_height > 10:
            fig_height = 10
        plt.figure(figsize=(1 + .23 * len(names), fig_height))
        ax = plt.subplot(1, 1, 1)

    ax.imshow(X, interpolation='nearest', aspect='auto')
    ax.set_label('conditions')
    ax.set_ylabel('scan number')

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=60, ha='left')
    # Set ticks above, to have a display more similar to the display of a
    # corresponding dataframe
    ax.xaxis.tick_top()

    plt.tight_layout()
    if output_file is not None:
        plt.savefig(output_file)
        plt.close()
        ax = None
    return ax


def plot_contrast_matrix(contrast_def, design_matrix, colorbar=False, ax=None,
                         output_file=None):
    """Creates plot for contrast definition.

    Parameters
    ----------
    contrast_def : str or array of shape (n_col) or list of (string or
                   array of shape (n_col))

        where ``n_col`` is the number of columns of the design matrix, (one
        array per run). If only one array is provided when there are several
        runs, it will be assumed that the same contrast is desired for all
        runs. The string can be a formula compatible with
        `pandas.DataFrame.eval`. Basically one can use the name of the
        conditions as they appear in the design matrix of the fitted model
        combined with operators +- and combined with numbers with operators
        +-`*`/.

    design_matrix: pandas DataFrame

    colorbar: Boolean, optional (default False)
        Include a colorbar in the contrast matrix plot.

    ax: matplotlib Axes object, optional (default None)
        Directory where plotted figures will be stored.

    output_file: string or None, optional,
        The name of an image file to export the plot to. Valid extensions
        are .png, .pdf, .svg. If output_file is not None, the plot
        is saved to a file, and the display is closed.


    Returns
    -------
    Plot Axes object

    """

    design_column_names = design_matrix.columns.tolist()
    if isinstance(contrast_def, str):
        contrast_def = expression_to_contrast_vector(
            contrast_def, design_column_names)
    maxval = np.max(np.abs(contrast_def))
    con_matrix = np.asmatrix(contrast_def)
    max_len = np.max([len(str(name)) for name in design_column_names])
    
    if ax is None:
        plt.figure(figsize=(.4 * len(design_column_names),
                            1 + .5 * con_matrix.shape[0] + .04 * max_len))
        ax = plt.gca()

    mat = ax.matshow(con_matrix, aspect='equal',
                     cmap='gray', vmin=-maxval, vmax=maxval)

    ax.set_label('conditions')
    ax.set_ylabel('')
    ax.set_yticks(())

    ax.xaxis.set(ticks=np.arange(len(design_column_names)))
    ax.set_xticklabels(design_column_names, rotation=50, ha='left')
    
    if colorbar:
        plt.colorbar(mat, fraction=0.025, pad=0.04)

    plt.tight_layout()
    plt.subplots_adjust(top=np.min([.3 + .05 * con_matrix.shape[0], .55]))

    if output_file is not None:
        plt.savefig(output_file)
        plt.close()
        ax = None

    return ax
