import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tight_layout import get_renderer
from matplotlib.colorbar import make_axes


def fit_axes(ax):
    """ Redimension the given axes to have labels fitting.
    """
    fig = ax.get_figure()
    renderer = get_renderer(fig)
    ylabel_width = ax.yaxis.get_tightbbox(renderer).inverse_transformed(
        ax.figure.transFigure).width
    if ax.get_position().xmin < 1.1 * ylabel_width:
        # we need to move it over
        new_position = ax.get_position()
        new_position.x0 = 1.1 * ylabel_width  # pad a little
        ax.set_position(new_position)

    xlabel_height = ax.xaxis.get_tightbbox(renderer).inverse_transformed(
        ax.figure.transFigure).height
    if ax.get_position().ymin < 1.1 * xlabel_height:
        # we need to move it over
        new_position = ax.get_position()
        new_position.y0 = 1.1 * xlabel_height  # pad a little
        ax.set_position(new_position)


def plot_matrix(mat, title=None, labels=None, figure=None, axes=None,
                colorbar=True, cmap=plt.cm.RdBu_r, tri='full',
                auto_fit=True, grid=False, reorder=False, **kwargs):
    """ Plot the given matrix.

        Parameters
        ----------
        mat : 2-D numpy array
            Matrix to be plotted.
        title : string or None, optional
            A text to add in the upper left corner.
        labels : list of strings, optional
            The label of each row and column
        figure : figure instance, figsize tuple, or None
            Sets the figure used. This argument can be either an existing
            figure, or a pair (width, height) that gives the size of a
            newly-created figure.
            Specifying both axes and figure is not allowed.
        axes : None or Axes, optional
            Axes instance to be plotted on. Creates a new one if None.
            Specifying both axes and figure is not allowed.
        colorbar : boolean, optional
            If True, an integrated colorbar is added.
        cmap : matplotlib colormap, optional
            The colormap for the matrix. Default is RdBu_r.
        tri : {'lower', 'diag', 'full'}, optional
            Which triangular part of the matrix to plot:
            'lower' is the lower part, 'diag' is the lower including
            diagonal, and 'full' is the full matrix.
        auto_fit : boolean, optional
            If auto_fit is True, the axes are dimensioned to give room
            for the labels. This assumes that the labels are resting
            against the bottom and left edges of the figure.
        grid : color or False, optional
            If not False, a grid is plotted to separate rows and columns
            using the given color.
        reorder : boolean or {'single', 'complete', 'average'}, optional
            If not False, reorders the matrix into blocks of clusters.
            Accepted linkage options for the clustering are 'single',
            'complete', and 'average'. True defaults to average linkage.

            .. note::
                This option is only available with SciPy >= 1.0.0.

            .. versionadded:: 0.4.1

        kwargs : extra keyword arguments
            Extra keyword arguments are sent to pylab.imshow

        Returns
        -------
        display : instance of matplotlib
            Axes image.
    """
    if reorder:
        if labels is None or labels is False:
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
        labels = labels[index]
        mat = mat[index, :][:, index]

    if tri == 'lower':
        mask = np.tri(mat.shape[0], k=-1, dtype=np.bool) ^ True
        mat = np.ma.masked_array(mat, mask)
    elif tri == 'diag':
        mask = np.tri(mat.shape[0], dtype=np.bool) ^ True
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
    if labels is False:
        axes.xaxis.set_major_formatter(plt.NullFormatter())
        axes.yaxis.set_major_formatter(plt.NullFormatter())
    elif labels is not None:
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
        if labels is not None and labels is not False:
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
