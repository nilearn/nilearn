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


def plot_matrix(mat, ax=None, tri='full', title=None, labels=None,
                auto_fit=True, grid=False, colorbar=True, cmap=plt.cm.RdBu_r,
                **kwargs):
    """ Plot the given matrix.

        Parameters
        ----------
        mat : 2-D numpy array
            Matrix to be plotted.
        ax  : None or Axes, optional
            Axes instance to be plotted on. Creates a new one if None.
        tri : {'lower', 'diag', 'full'}, optional
            Which triangular part of the matrix to plot:
            'lower' is the lower part, 'diag' is the lower including
            diagonal, and 'full' is the full matrix.
        title : string or None, optional
            A text to add in the upper left corner.
        labels : list of strings, optional
            The label of each row and column
        auto_fit : boolean, optional
            If auto_fit is True, the axes are dimensioned to give room
            for the labels. This assumes that the labels are resting
            against the bottom and left edges of the figure.
        grid : color or False, optional
            If not False, a grid is plotted to separate rows and columns
            using the given color.
        colorbar : boolean, optional
            If True, an integrated colorbar is added.
        cmap : matplotlib colormap, optional
            The colormap for the matrix. Default is RdBu_r.
        kwargs : extra keyword arguments
            Extra keyword arguments are sent to pylab.imshow

        Returns Matplotlib AxesImage instance
    """
    if tri == 'lower':
        mask = np.tri(mat.shape[0], k=-1, dtype=np.bool) ^ True
        mat = np.ma.masked_array(mat, mask)
    elif tri == 'diag':
        mask = np.tri(mat.shape[0], dtype=np.bool) ^ True
        mat = np.ma.masked_array(mat, mask)
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = ax.figure
    display = ax.imshow(mat, aspect='equal', interpolation='nearest',
                        cmap=cmap, **kwargs)
    ax.set_autoscale_on(False)
    ymin, ymax = ax.get_ylim()
    if labels is False:
        ax.xaxis.set_major_formatter(plt.NullFormatter())
        ax.yaxis.set_major_formatter(plt.NullFormatter())
    elif labels is not None:
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, size='x-small')
        for label in ax.get_xticklabels():
            label.set_ha('right')
            label.set_rotation(50)
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels, size='x-small')
        for label in ax.get_yticklabels():
            label.set_ha('right')
            label.set_rotation(10)

    if title is not None:
        ax.text(0.9 - .15 * colorbar, 0.9 + .05 * colorbar, title,
                horizontalalignment='right',
                verticalalignment='top',
                transform=ax.transAxes)

    if grid is not False:
        size = len(mat)
        # Different grids for different layouts
        if tri == 'lower':
            for i in range(size):
                # Correct for weird mis-sizing
                i = 1.001 * i
                ax.plot([i + 0.5, i + 0.5], [size - 0.5, i + 0.5],
                        color='grey')
                ax.plot([i + 0.5, -0.5], [i + 0.5, i + 0.5],
                        color='grey')
        elif tri == 'diag':
            for i in range(size):
                # Correct for weird mis-sizing
                i = 1.001 * i
                ax.plot([i + 0.5, i + 0.5], [size - 0.5, i - 0.5],
                        color='grey')
                ax.plot([i + 0.5, -0.5], [i - 0.5, i - 0.5], color='grey')
        else:
            for i in range(size):
                # Correct for weird mis-sizing
                i = 1.001 * i
                ax.plot([i + 0.5, i + 0.5], [size - 0.5, -0.5], color='grey')
                ax.plot([size - 0.5, -0.5], [i + 0.5, i + 0.5], color='grey')

    ax.set_ylim(ymin, ymax)

    if auto_fit and labels is not None and labels is not False:
            fit_axes(ax)

    if colorbar:
        cax, kw = make_axes(ax, location='right', fraction=0.05, shrink=0.8)
        fig.colorbar(mappable=display, cax=cax)
        # make some room
        fig.subplots_adjust(right=0.8)
        # change current axis back to matrix
        plt.sca(ax)

    return display
