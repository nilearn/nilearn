"""Testing utilities."""

import matplotlib.colorbar as cbar
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def assert_colorbar_present(fig):
    """Check if a colorbar exists in the figure."""
    if (
        isinstance(fig, Figure)
        and not any(isinstance(ax, cbar.Colorbar) for ax in fig.get_children())
    ) or (isinstance(fig, Axes) and not isinstance(fig, cbar.Colorbar)):
        raise RuntimeError("Figures should have a colorbar by default.")
