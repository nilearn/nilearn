"""
Miscellaneous matrix plotting utilities.
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.tight_layout import get_renderer
from matplotlib.colorbar import make_axes
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    from nilearn.glm.contrasts import expression_to_contrast_vector
    from nilearn.glm.first_level import check_design_matrix


def fit_axes(ax):
    """ Redimension the given axes to have labels fitting.
    """
    fig = ax.get_figure()
    renderer = get_renderer(fig)
    ylabel_width = ax.yaxis.get_tightbbox(renderer).transformed(
        ax.figure.transFigure.inverted()).width
    if ax.get_position().xmin < 1.1 * ylabel_width:
        # we need to move it over
        new_position = ax.get_position()
        new_position.x0 = 1.1 * ylabel_width  # pad a little
        ax.set_position(new_position)

    xlabel_height = ax.xaxis.get_tightbbox(renderer).transformed(
        ax.figure.transFigure.inverted()).height
    if ax.get_position().ymin < 1.1 * xlabel_height:
        # we need to move it over
        new_position = ax.get_position()
        new_position.y0 = 1.1 * xlabel_height  # pad a little
        ax.set_position(new_position)


def plot_matrix(mat, title=None, labels=None, figure=None, axes=None,
                colorbar=True, cmap=plt.cm.RdBu_r, tri='full',
                auto_fit=True, grid=False, reorder=False, **kwargs):
    """Plot the given matrix.

    Parameters
    ----------
    mat : 2-D numpy array
        Matrix to be plotted.

    title : string or None, optional
        A text to add in the upper left corner.

    labels : list, ndarray of strings, empty list, False, or None, optional
        The label of each row and column. Needs to be the same
        length as rows/columns of mat. If False, None, or an
        empty list, no labels are plotted.

    figure : figure instance, figsize tuple, or None, optional
        Sets the figure used. This argument can be either an existing
        figure, or a pair (width, height) that gives the size of a
        newly-created figure.
        Specifying both axes and figure is not allowed.

    axes : None or Axes, optional
        Axes instance to be plotted on. Creates a new one if None.
        Specifying both axes and figure is not allowed.

    colorbar : boolean, optional
        If True, an integrated colorbar is added. Default=True.

    cmap : matplotlib colormap, optional
        The colormap for the matrix. Default=plt.cm.RdBu_r.

    tri : {'full', 'lower', 'diag'}, optional
        Which triangular part of the matrix to plot:
        'lower' is the lower part, 'diag' is the lower including
        diagonal, and 'full' is the full matrix.
        Default='full'.

    auto_fit : boolean, optional
        If auto_fit is True, the axes are dimensioned to give room
        for the labels. This assumes that the labels are resting
        against the bottom and left edges of the figure.
        Default=True.

    grid : color or False, optional
        If not False, a grid is plotted to separate rows and columns
        using the given color. Default=False.

    reorder : boolean or {'single', 'complete', 'average'}, optional
        If not False, reorders the matrix into blocks of clusters.
        Accepted linkage options for the clustering are 'single',
        'complete', and 'average'. True defaults to average linkage.
        Default=False.

        .. note::
            This option is only available with SciPy >= 1.0.0.

        .. versionadded:: 0.4.1

    kwargs : extra keyword arguments, optional
        Extra keyword arguments are sent to pylab.imshow.

    Returns
    -------
    display : instance of matplotlib
        Axes image.

    """
    # we need a list so an empty one will be cast to False
    if isinstance(labels, np.ndarray):
        labels = labels.tolist()
    if labels and len(labels) != mat.shape[0]:
        raise ValueError("Length of labels unequal to length of matrix.")

    if reorder:
        if not labels:
            raise ValueError("Labels are needed to show the reordering.")
        try:
            from scipy.cluster.hierarchy import (linkage, optimal_leaf_ordering,
                                                 leaves_list)
        except ImportError:
            raise ImportError("A scipy version of at least 1.0 is needed "
                              "for ordering the matrix with "
                              "optimal_leaf_ordering.")
        valid_reorder_args = [True, 'single', 'complete', 'average']
        if reorder not in valid_reorder_args:
            raise ValueError("Parameter reorder needs to be "
                             "one of {}.".format(valid_reorder_args))
        if reorder is True:
            reorder = 'average'
        linkage_matrix = linkage(mat, method=reorder)
        ordered_linkage = optimal_leaf_ordering(linkage_matrix, mat)
        index = leaves_list(ordered_linkage)
        # make sure labels is an ndarray and copy it
        labels = np.array(labels).copy()
        mat = mat.copy()
        # and reorder labels and matrix
        labels = labels[index].tolist()
        mat = mat[index, :][:, index]

    if tri == 'lower':
        mask = np.tri(mat.shape[0], k=-1, dtype=bool) ^ True
        mat = np.ma.masked_array(mat, mask)
    elif tri == 'diag':
        mask = np.tri(mat.shape[0], dtype=bool) ^ True
        mat = np.ma.masked_array(mat, mask)
    if axes is not None and figure is not None:
        raise ValueError("Parameters figure and axes cannot be specified "
            "together. You gave 'figure=%s, axes=%s'"
            % (figure, axes))
    if figure is not None:
        if isinstance(figure, plt.Figure):
            fig = figure
        else:
            fig = plt.figure(figsize=figure)
        axes = plt.gca()
        own_fig = True
    else:
        if axes is None:
            fig, axes = plt.subplots(1, 1, figsize=(7, 5))
            own_fig = True
        else:
            fig = axes.figure
            own_fig = False
    display = axes.imshow(mat, aspect='equal', interpolation='nearest',
                        cmap=cmap, **kwargs)
    axes.set_autoscale_on(False)
    ymin, ymax = axes.get_ylim()
    if not labels:
        axes.xaxis.set_major_formatter(plt.NullFormatter())
        axes.yaxis.set_major_formatter(plt.NullFormatter())
    else:
        axes.set_xticks(np.arange(len(labels)))
        axes.set_xticklabels(labels, size='x-small')
        for label in axes.get_xticklabels():
            label.set_ha('right')
            label.set_rotation(50)
        axes.set_yticks(np.arange(len(labels)))
        axes.set_yticklabels(labels, size='x-small')
        for label in axes.get_yticklabels():
            label.set_ha('right')
            label.set_va('top')
            label.set_rotation(10)

    if grid is not False:
        size = len(mat)
        # Different grids for different layouts
        if tri == 'lower':
            for i in range(size):
                # Correct for weird mis-sizing
                i = 1.001 * i
                axes.plot([i + 0.5, i + 0.5], [size - 0.5, i + 0.5],
                          color='grey')
                axes.plot([i + 0.5, -0.5], [i + 0.5, i + 0.5],
                          color='grey')
        elif tri == 'diag':
            for i in range(size):
                # Correct for weird mis-sizing
                i = 1.001 * i
                axes.plot([i + 0.5, i + 0.5], [size - 0.5, i - 0.5],
                          color='grey')
                axes.plot([i + 0.5, -0.5], [i - 0.5, i - 0.5], color='grey')
        else:
            for i in range(size):
                # Correct for weird mis-sizing
                i = 1.001 * i
                axes.plot([i + 0.5, i + 0.5], [size - 0.5, -0.5], color='grey')
                axes.plot([size - 0.5, -0.5], [i + 0.5, i + 0.5], color='grey')

    axes.set_ylim(ymin, ymax)

    if auto_fit:
        if labels:
            fit_axes(axes)
        elif own_fig:
            plt.tight_layout(pad=.1,
                             rect=((0, 0, .95, 1) if colorbar
                                   else (0, 0, 1, 1)))

    if colorbar:
        cax, kw = make_axes(axes, location='right', fraction=0.05, shrink=0.8,
                            pad=.0)
        fig.colorbar(mappable=display, cax=cax)
        # make some room
        fig.subplots_adjust(right=0.8)
        # change current axis back to matrix
        plt.sca(axes)

    if title is not None:
        # Adjust the size
        text_len = np.max([len(t) for t in title.split('\n')])
        size = axes.bbox.size[0] / text_len
        axes.text(0.95, 0.95, title,
                  horizontalalignment='right',
                  verticalalignment='top',
                  transform=axes.transAxes,
                  size=size)

    return display


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
        Design matrix to use.

    colorbar : Boolean, optional
        Include a colorbar in the contrast matrix plot. Default=False.

    ax : matplotlib Axes object, optional
        Axis on which to plot the figure. If None, a new figure will be created.

    output_file : string or None, optional
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


def plot_design_matrix(design_matrix, rescale=True, ax=None, output_file=None):
    """Plot a design matrix provided as a DataFrame

    Parameters
    ----------
    design matrix : pandas DataFrame,
        Describes a design matrix.

    rescale : bool, optional
        Rescale columns magnitude for visualization or not.
        Default=True.

    ax : axis handle, optional
        Handle to axis onto which we will draw design matrix.

    output_file : string or None, optional
        The name of an image file to export the plot to. Valid extensions
        are .png, .pdf, .svg. If output_file is not None, the plot
        is saved to a file, and the display is closed.

    Returns
    -------
    ax: axis handle
        The axis used for plotting.

    """
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


def plot_event(model_event, cmap=None, output_file=None, **fig_kwargs):
    """Creates plot for event visualization.

    Parameters
    ----------
    model_event : pandas DataFrame or list of pandas DataFrame
        The `pandas.DataFrame` must have three columns
        ``event_type`` with event name, ``onset`` and ``duration``.
        The `pandas.DataFrame` can also be obtained from
        :func:`nilearn.glm.first_level.first_level_from_bids`.

    cmap : str or matplotlib.cmap, optional
        The colormap used to label different events.

    output_file : string or None, optional
        The name of an image file to export the plot to. Valid extensions
        are .png, .pdf, .svg. If output_file is not None, the plot
        is saved to a file, and the display is closed.

    **fig_kwargs : extra keyword arguments, optional
        Extra arguments passed to matplotlib.pyplot.subplots.

    Returns
    -------
    Plot Figure object

    """
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

    event_labels = pd.concat(event['trial_type'] for event in model_event)
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

    _ = ax.legend(handles=handles, ncol=4)

    ax.set_xlabel("Time (sec.)")
    ax.set_ylabel("Runs")
    ax.set_ylim(0, n_runs)
    ax.set_yticks(np.arange(n_runs) + .5)
    ax.set_yticklabels(np.arange(n_runs) + 1)

    plt.tight_layout()
    if output_file is not None:
        plt.savefig(output_file)
        plt.close()
        figure = None

    return figure
