"""
This module implements plotting functions useful to report analysis results.

Author: Martin Perez-Guevara, Elvis Dohmatob, 2017
"""

import numpy as np
import matplotlib.pyplot as plt

from nilearn.stats.first_level_model import check_design_matrix
from nilearn.stats.contrasts import expression_to_contrast_vector


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

    output_file : string or None, optional,
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
        plt.figure()
        ax = plt.subplot(1, 1, 1)

    ax.imshow(X, interpolation='nearest', aspect='auto')
    ax.set_label('conditions')
    ax.set_ylabel('scan number')

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=60, ha='right')

    plt.tight_layout()
    if output_file is not None:
        plt.savefig(output_file)
        plt.close()
        ax = None
    return ax


def plot_event(model_event, cmap=None, output_file=None, **fig_kwargs):
    """Creates plot for contrast definition.

    Parameters
    ----------
    model_event : pandas DataFrame or list of pandas DataFrame
        the `pandas.DataFrame` must have three columns
        ``event_type`` with event name, ``onset`` and ``duration``.
        The `pandas.DataFrame` can also be obtained from 
        :func:`~nilearn.stats.first_level_model.first_level_models_from_bids`.

    cmap : str or matplotlib.cmap
        the colormap used to label different events

    output_file : string or None, optional,
        The name of an image file to export the plot to. Valid extensions
        are .png, .pdf, .svg. If output_file is not None, the plot
        is saved to a file, and the display is closed.

    **fig_kwargs : extra keyword arguments, optional
        Extra arguments passed to matplotlib.pyplot.subplots

    Returns
    -------
    Plot Figure object

    """
    import matplotlib.patches as mpatches
    import pandas as pd

    if isinstance(model_event, pd.DataFrame):
        model_event = [model_event]
   
    n_runs = len(model_event)
    figure, ax = plt.subplots(1, 1, **fig_kwargs)

    # input validation
    if cmap is None:
        cmap = plt.cm.tab20
    elif isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    else:
        cmap = cmap

    event_labels = pd.concat(e['trial_type'] for e in model_event)
    event_labels = np.unique(event_labels)

    cmap_dictionary = {label:idx for idx, label in enumerate(event_labels)}

    if len(event_labels) > cmap.N:
        plt.close()
        raise ValueError("The number of event types is greater than "+ \
            " colors in colormap (%d > %d). Use a different colormap." \
            % (len(event_labels), cmap.N))

    for idx_run, event_df in enumerate(model_event):
        
        for _, event in event_df.iterrows():
            event_onset = event['onset']
            event_end = event['onset'] + event['duration']
            color = cmap.colors[cmap_dictionary[event['trial_type']]]
         
            ax.axvspan(event_onset, 
                       event_end, 
                       ymin=(idx_run + .25) / n_runs, 
                       ymax=(idx_run + .75) / n_runs, 
                       facecolor=color)

    handles = []
    for label, idx in cmap_dictionary.items():
        patch = mpatches.Patch(color=cmap.colors[idx], label=label)
        handles.append(patch)

    _ = ax.legend(handles=handles)

    ax.set_ylabel("Runs")
    ax.set_xlabel("Time (sec.)")
    ax.set_ylim(0, n_runs)
    ax.set_yticks(np.arange(n_runs) + .5)
    ax.set_yticklabels(np.arange(n_runs) + 1)
    
    plt.tight_layout()
    if output_file is not None:
        plt.savefig(output_file)
        plt.close()
        ax = None

    return figure


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

    design_matrix : pandas DataFrame

    colorbar : Boolean, optional (default False)
        Include a colorbar in the contrast matrix plot.

    ax : matplotlib Axes object, optional (default None)
        Directory where plotted figures will be stored.

    output_file : string or None, optional,
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

    if ax is None:
        plt.figure(figsize=(8, 4))
        ax = plt.gca()

    mat = ax.matshow(con_matrix, aspect='equal',
                     extent=[0, con_matrix.shape[1], 0, con_matrix.shape[0]],
                     cmap='gray', vmin=-maxval, vmax=maxval)

    ax.set_label('conditions')
    ax.set_ylabel('')
    ax.set_yticklabels(['' for x in ax.get_yticklabels()])

    # Shift ticks to be at 0.5, 1.5, etc
    ax.xaxis.set(ticks=np.arange(len(design_column_names)))
    ax.set_xticklabels(design_column_names, rotation=60, ha='left')

    if colorbar:
        plt.colorbar(mat, fraction=0.025, pad=0.04)

    plt.tight_layout()
    if output_file is not None:
        plt.savefig(output_file)
        plt.close()
        ax = None

    return ax
