"""Miscellaneous matrix plotting utilities."""

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable

from nilearn import DEFAULT_DIVERGING_CMAP
from nilearn._utils.docs import fill_doc
from nilearn._utils.glm import check_and_load_tables
from nilearn._utils.helpers import constrained_layout_kwargs, rename_parameters
from nilearn.glm.first_level import check_design_matrix
from nilearn.glm.first_level.experimental_paradigm import check_events
from nilearn.plotting._utils import save_figure_if_needed
from nilearn.plotting.matrix._utils import (
    mask_matrix,
    pad_contrast_matrix,
    reorder_matrix,
    sanitize_labels,
    sanitize_reorder,
    sanitize_tri,
)


def _configure_axis(
    axes, labels, label_size, x_label_rotation, y_label_rotation
):
    """Help for plot_matrix."""
    if not labels:
        axes.xaxis.set_major_formatter(plt.NullFormatter())
        axes.yaxis.set_major_formatter(plt.NullFormatter())
    else:
        axes.set_xticks(np.arange(len(labels)))
        axes.set_xticklabels(labels, size=label_size)
        for label in axes.get_xticklabels():
            label.set_ha("right")
            label.set_rotation(x_label_rotation)
        axes.set_yticks(np.arange(len(labels)))
        axes.set_yticklabels(labels, size=label_size)
        for label in axes.get_yticklabels():
            label.set_ha("right")
            label.set_va("top")
            label.set_rotation(y_label_rotation)


def _configure_grid(axes, tri, size):
    """Help for plot_matrix."""
    # Different grids for different layouts
    if tri == "lower":
        for i in range(size):
            # Correct for weird mis-sizing
            i = 1.001 * i
            axes.plot([i + 0.5, i + 0.5], [size - 0.5, i + 0.5], color="gray")
            axes.plot([i + 0.5, -0.5], [i + 0.5, i + 0.5], color="gray")
    elif tri == "diag":
        for i in range(size):
            # Correct for weird mis-sizing
            i = 1.001 * i
            axes.plot([i + 0.5, i + 0.5], [size - 0.5, i - 0.5], color="gray")
            axes.plot([i + 0.5, -0.5], [i - 0.5, i - 0.5], color="gray")
    else:
        for i in range(size):
            # Correct for weird mis-sizing
            i = 1.001 * i
            axes.plot([i + 0.5, i + 0.5], [size - 0.5, -0.5], color="gray")
            axes.plot([size - 0.5, -0.5], [i + 0.5, i + 0.5], color="gray")


def _fit_axes(axes):
    """Help for plot_matrix.

    This function redimensions the given axes to have
    labels fitting.
    """
    fig = axes.get_figure()
    renderer = fig.canvas.get_renderer()
    ylabel_width = (
        axes.yaxis.get_tightbbox(renderer)
        .transformed(axes.figure.transFigure.inverted())
        .width
    )
    if axes.get_position().xmin < 1.1 * ylabel_width:
        # we need to move it over
        new_position = axes.get_position()
        new_position.x0 = 1.1 * ylabel_width  # pad a little
        axes.set_position(new_position)

    xlabel_height = (
        axes.xaxis.get_tightbbox(renderer)
        .transformed(axes.figure.transFigure.inverted())
        .height
    )
    if axes.get_position().ymin < 1.1 * xlabel_height:
        # we need to move it over
        new_position = axes.get_position()
        new_position.y0 = 1.1 * xlabel_height  # pad a little
        axes.set_position(new_position)


def _sanitize_figure_and_axes(figure, axes):
    """Help for plot_matrix."""
    if axes is not None and figure is not None:
        raise ValueError(
            "Parameters figure and axes cannot be specified together. "
            f"You gave 'figure={figure}, axes={axes}'."
        )
    if figure is not None:
        if isinstance(figure, plt.Figure):
            fig = figure
            if hasattr(fig, "set_layout_engine"):  # can be removed w/mpl 3.5
                fig.set_layout_engine("constrained")
        else:
            fig = plt.figure(figsize=figure, **constrained_layout_kwargs())
        axes = plt.gca()
        own_fig = True
    elif axes is None:
        fig, axes = plt.subplots(
            1,
            1,
            figsize=(7, 5),
            **constrained_layout_kwargs(),
        )
        own_fig = True
    else:
        fig = axes.figure
        own_fig = False
    return fig, axes, own_fig


def _sanitize_inputs_plot_matrix(
    mat_shape, tri, labels, reorder, figure, axes
):
    """Help for plot_matrix.

    This function makes sure the inputs to plot_matrix are valid.
    """
    sanitize_tri(tri)
    labels = sanitize_labels(mat_shape, labels)
    reorder = sanitize_reorder(reorder)
    fig, axes, own_fig = _sanitize_figure_and_axes(figure, axes)
    return labels, reorder, fig, axes, own_fig


@fill_doc
def plot_matrix(
    mat,
    title=None,
    labels=None,
    figure=None,
    axes=None,
    colorbar=True,
    cmap=DEFAULT_DIVERGING_CMAP,
    tri="full",
    auto_fit=True,
    grid=False,
    reorder=False,
    **kwargs,
):
    """Plot the given matrix.

    Parameters
    ----------
    mat : 2-D :class:`numpy.ndarray`
        Matrix to be plotted.
    %(title)s
    labels : :obj:`list`, or :class:`numpy.ndarray` of :obj:`str`,\
    or False, or None, default=None
        The label of each row and column. Needs to be the same
        length as rows/columns of mat. If False, None, or an
        empty list, no labels are plotted.

    figure : :class:`matplotlib.figure.Figure`, figsize :obj:`tuple`,\
    or None, default=None
        Sets the figure used. This argument can be either an existing
        figure, or a pair (width, height) that gives the size of a
        newly-created figure.

        .. note::

            Specifying both axes and figure is not allowed.

    axes : None or :class:`matplotlib.axes.Axes`, default=None
        Axes instance to be plotted on. Creates a new one if None.

        .. note::

            Specifying both axes and figure is not allowed.

    %(colorbar)s
        Default=True.

    %(cmap)s
        default="RdBu_r"

    tri : {'full', 'lower', 'diag'}, default='full'
        Which triangular part of the matrix to plot:

            - 'lower': Plot the lower part
            - 'diag': Plot the lower part with the diagonal
            - 'full': Plot the full matrix


    auto_fit : :obj:`bool`, default=True
        If auto_fit is True, the axes are dimensioned to give room
        for the labels. This assumes that the labels are resting
        against the bottom and left edges of the figure.

    grid : color or False, default=False
        If not False, a grid is plotted to separate rows and columns
        using the given color.

    reorder : :obj:`bool` or {'single', 'complete', 'average'}, default=False
        If not False, reorders the matrix into blocks of clusters.
        Accepted linkage options for the clustering are 'single',
        'complete', and 'average'. True defaults to average linkage.

        .. note::
            This option is only available with SciPy >= 1.0.0.

        .. versionadded:: 0.4.1

    kwargs : extra keyword arguments, optional
        Extra keyword arguments are sent to pylab.imshow.

    Returns
    -------
    display : :class:`matplotlib.axes.Axes`
        Axes image.

    """
    labels, reorder, fig, axes, _ = _sanitize_inputs_plot_matrix(
        mat.shape, tri, labels, reorder, figure, axes
    )
    if reorder:
        mat, labels = reorder_matrix(mat, labels, reorder)
    if tri != "full":
        mat = mask_matrix(mat, tri)
    display = axes.imshow(
        mat, aspect="equal", interpolation="nearest", cmap=cmap, **kwargs
    )
    axes.set_autoscale_on(False)
    ymin, ymax = axes.get_ylim()
    _configure_axis(
        axes,
        labels,
        label_size="x-small",
        x_label_rotation=50,
        y_label_rotation=10,
    )
    if grid is not False:
        _configure_grid(axes, tri, len(mat))
    axes.set_ylim(ymin, ymax)
    if auto_fit and labels:
        _fit_axes(axes)
    if colorbar:
        divider = make_axes_locatable(axes)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        fig.colorbar(display, cax=cax)

    if title is not None:
        axes.set_title(title, size=16)

    return display


# TODO (nilearn >= 0.13.0)
@fill_doc
@rename_parameters({"ax": "axes"}, end_version="0.13.0")
def plot_contrast_matrix(
    contrast_def, design_matrix, colorbar=True, axes=None, output_file=None
):
    """Create plot for :term:`contrast` definition.

    Parameters
    ----------
    contrast_def : :obj:`str` or :class:`numpy.ndarray` of shape[1] <= n_col \
        where ``n_col`` is the number of columns of the design matrix.
        The string can be a formula compatible
        with :meth:`pandas.DataFrame.eval`.
        Basically one can use the name of the conditions
        as they appear in the design matrix of the fitted model
        combined with operators +-
        and combined with numbers with operators +-`*`/.

    design_matrix : :class:`pandas.DataFrame`
        Design matrix to use.

    %(colorbar)s
        Default=True.

    axes : :class:`matplotlib.axes.Axes` or None, default=None
        Axis on which to plot the figure.
        If None, a new figure will be created.

    %(output_file)s

    Returns
    -------
    axes : :class:`matplotlib.axes.Axes`
        Figure object.

    """
    contrast_def = pad_contrast_matrix(contrast_def, design_matrix)
    con_matrix = np.array(contrast_def, ndmin=2)

    design_column_names = design_matrix.columns.tolist()
    max_len = np.max([len(str(name)) for name in design_column_names])

    n_columns_design_matrix = len(design_column_names)
    if axes is None:
        _, axes = plt.subplots(
            figsize=(
                0.4 * n_columns_design_matrix,
                1 + 0.5 * con_matrix.shape[0] + 0.04 * max_len,
            ),
            **constrained_layout_kwargs(),
        )

    maxval = np.max(np.abs(contrast_def))
    mat = axes.matshow(
        con_matrix, aspect="equal", cmap="gray", vmin=-maxval, vmax=maxval
    )

    axes.set_label("conditions")
    axes.set_ylabel("")
    axes.set_yticks(())

    axes.xaxis.set(ticks=np.arange(n_columns_design_matrix))
    axes.set_xticklabels(design_column_names, rotation=50, ha="left")

    if colorbar:
        fig = axes.figure
        fig.colorbar(mat, fraction=0.025, pad=0.04)

    return save_figure_if_needed(axes, output_file)


# TODO (nilearn >= 0.13.0)
@fill_doc
@rename_parameters({"ax": "axes"}, end_version="0.13.0")
def plot_design_matrix(
    design_matrix,
    rescale=True,
    axes=None,
    output_file=None,
):
    """Plot a design matrix.

    Parameters
    ----------
    design matrix : :class:`pandas.DataFrame` or \
                    :obj:`str` or :obj:`pathlib.Path` to a TSV event file
        Describes a design matrix.

    rescale : :obj:`bool`, default=True
        Rescale columns magnitude for visualization or not.

    axes : :class:`matplotlib.axes.Axes` or None, default=None
        Handle to axes onto which we will draw the design matrix.

    %(output_file)s

    Returns
    -------
    axes : :class:`matplotlib.axes.Axes`
        The axes used for plotting.

    """
    design_matrix = check_and_load_tables(design_matrix, "design_matrix")[0]

    _, X, names = check_design_matrix(design_matrix)
    # normalize the values per column for better visualization
    if rescale:
        X = X / np.maximum(1.0e-12, np.sqrt(np.sum(X**2, 0)))
    if axes is None:
        max_len = np.max([len(str(name)) for name in names])
        fig_height = 1 + 0.1 * X.shape[0] + 0.04 * max_len
        if fig_height < 3:
            fig_height = 3
        elif fig_height > 10:
            fig_height = 10
        _, axes = plt.subplots(
            figsize=(1 + 0.23 * len(names), fig_height),
            **constrained_layout_kwargs(),
        )

    axes.imshow(X, interpolation="nearest", aspect="auto")
    axes.set_label("conditions")
    axes.set_ylabel("scan number")

    axes.set_xticks(range(len(names)))
    axes.set_xticklabels(names, rotation=60, ha="left")
    # Set ticks above, to have a display more similar to the display of a
    # corresponding dataframe
    axes.xaxis.tick_top()

    return save_figure_if_needed(axes, output_file)


@fill_doc
def plot_event(model_event, cmap=None, output_file=None, **fig_kwargs):
    """Create plot for event visualization.

    .. warning::

        Events with a duration of 0 seconds will be plotted
        by a 'delta function'.

    Parameters
    ----------
    model_event : :class:`pandas.DataFrame`, \
                  :obj:`str` or :obj:`pathlib.Path` to a TSV event file, \
                  or a :obj:`list` or  :obj:`tuple` \
                  of :class:`pandas.DataFrame`, \
                  :obj:`str` or :obj:`pathlib.Path` to a TSV event file.
        The :class:`pandas.DataFrame` must have three columns:
        ``trial_type`` with event name, ``onset`` and ``duration``.
        See :func:`~nilearn.glm.first_level.make_first_level_design_matrix`
        for details on the required content of events dataframes.

        .. note::

            The :class:`pandas.DataFrame` can also be obtained
            from :func:`nilearn.glm.first_level.first_level_from_bids`.

    %(cmap)s

    %(output_file)s

    **fig_kwargs : extra keyword arguments, optional
        Extra arguments passed to :func:`matplotlib.pyplot.subplots`.

    Returns
    -------
    figure : :class:`matplotlib.figure.Figure`
        Plot Figure object.

    """
    model_event = check_and_load_tables(model_event, "model_event")

    for i, event in enumerate(model_event):
        event_copy = check_events(event)
        model_event[i] = event_copy

    n_runs = len(model_event)
    if "layout" not in fig_kwargs and "constrained_layout" not in fig_kwargs:
        fig_kwargs.update(**constrained_layout_kwargs())
    figure, axes = plt.subplots(1, 1, **fig_kwargs)

    # input validation
    if cmap is None:
        cmap = "tab20"
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    event_labels = pd.concat(event["trial_type"] for event in model_event)
    event_labels = np.unique(event_labels)

    cmap_dictionary = {label: idx for idx, label in enumerate(event_labels)}

    if len(event_labels) > cmap.N:
        plt.close(fig=figure)
        raise ValueError(
            "The number of event types is greater than "
            f"colors in colormap ({len(event_labels)} > {cmap.N}). "
            "Use a different colormap."
        )

    height = 0.5
    x_lim = []
    for idx_run, event_df in enumerate(model_event):
        for _, event in event_df.iterrows():
            modulation = 1.0
            if "modulation" in event:
                modulation = event["modulation"]

            ymin = (idx_run + 0.25) / n_runs
            ymax = (idx_run + 0.25 + height * modulation) / n_runs

            event_onset = event["onset"]
            event_end = event["onset"] + event["duration"]

            x_lim.append(event_end)

            color = cmap.colors[cmap_dictionary[event["trial_type"]]]

            if event["duration"] != 0:
                axes.axvspan(
                    xmin=event_onset,
                    xmax=event_end,
                    ymin=ymin,
                    ymax=ymax,
                    facecolor=color,
                )

            # events will 0 duration are plotted as lines
            else:
                axes.axvline(
                    event_onset,
                    ymin=ymin,
                    ymax=ymax,
                    color=color,
                )

    handles = []
    for label, idx in cmap_dictionary.items():
        patch = mpatches.Patch(color=cmap.colors[idx], label=label)
        handles.append(patch)

    _ = axes.legend(handles=handles, ncol=4)

    axes.set_xlabel("Time (sec.)")
    axes.set_ylabel("Runs")
    axes.set_ylim(0, n_runs)
    axes.set_xlim(-1, max(x_lim) + 1)
    axes.set_yticks(np.arange(n_runs) + 0.5)
    axes.set_yticklabels(np.arange(n_runs) + 1)

    return save_figure_if_needed(figure, output_file)


@fill_doc
def plot_design_matrix_correlation(
    design_matrix,
    tri="full",
    cmap=DEFAULT_DIVERGING_CMAP,
    colorbar=True,
    output_file=None,
    **kwargs,
):
    """Compute and plot the correlation between regressor of a design matrix.

    The drift and constant regressors are omitted from the plot.

    .. versionadded:: 0.11.0

    Parameters
    ----------
    design_matrix : :obj:`pandas.DataFrame`, :obj:`pandas.DataFrame` \
                    :obj:`pathlib.Path`
        Design matrix whose correlation matrix you want to plot.

    tri : {"full", "diag"}, default="full"
        Which triangular part of the matrix to plot:

        - ``"diag"``: Plot the lower part with the diagonal
        - ``"full"``: Plot the full matrix

    %(cmap)s
        default="RdBu_r"

        This must be a diverging colormap as the correlation matrix
        will be centered on 0.
        The allowed colormaps are:

        - ``"bwr"``
        - ``"RdBu_r"``
        - ``"seismic_r"``

    %(output_file)s

    kwargs : extra keyword arguments, optional
        Extra keyword arguments are sent to
        :func:`nilearn.plotting.plot_matrix`

    Returns
    -------
    display : :class:`matplotlib.axes.Axes`
        Axes image.
    """
    design_matrix = check_and_load_tables(design_matrix, "design_matrix")[0]

    check_design_matrix(design_matrix)

    ALLOWED_CMAP = ["RdBu_r", "bwr", "seismic_r"]
    cmap_name = cmap if isinstance(cmap, str) else cmap.name
    if cmap_name not in ALLOWED_CMAP:
        raise ValueError(f"cmap must be one of {ALLOWED_CMAP}")

    columns_to_drop = ["intercept", "constant"]
    columns_to_drop.extend(
        col for col in design_matrix.columns if col.startswith("drift_")
    )
    design_matrix = design_matrix.drop(
        columns=columns_to_drop, errors="ignore"
    )

    if len(design_matrix.columns) == 0:
        raise ValueError(
            "Nothing left to plot after "
            "removing drift and constant regressors."
        )

    sanitize_tri(tri, allowed_values=("full", "diag"))

    mat = design_matrix.corr()

    mat = mat.to_numpy()
    vmax = max(mat.min(), mat.max(), key=abs)
    if len(mat) > 1:
        # find the second-largest value in each row
        # to omit values on the diagonal that will always be == 1
        second_largest = np.partition(mat, -2, axis=1)[:, -2]
        vmax = max(abs(mat.min().min()), max(second_largest))

    col_labels = design_matrix.columns
    display = plot_matrix(
        mat,
        tri=tri,
        cmap=cmap,
        vmax=vmax,
        vmin=vmax * -1,
        labels=col_labels.to_list(),
        colorbar=colorbar,
        **kwargs,
    )

    return save_figure_if_needed(display, output_file)
