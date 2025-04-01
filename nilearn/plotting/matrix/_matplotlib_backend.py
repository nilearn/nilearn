import matplotlib.pyplot as plt
import numpy as np

from nilearn._utils import constrained_layout_kwargs


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
