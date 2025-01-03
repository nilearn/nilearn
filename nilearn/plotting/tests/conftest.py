"""Test utilities for plotting."""

import pytest


@pytest.fixture(scope="function")
def pyplot():
    """Set up and teardown fixture for matplotlib.

    This fixture checks if we can import matplotlib. If not, the tests will be
    skipped. Otherwise, we close the figures before and after running the
    functions.

    Returns
    -------
    pyplot : module
        The ``matplotlib.pyplot`` module.
    """
    pyplot = pytest.importorskip("matplotlib.pyplot")
    pyplot.close("all")
    yield pyplot
    pyplot.close("all")


@pytest.fixture(scope="function")
def plotly():
    """Check if we can import plotly.

    If not, the tests will be skipped.

    Returns
    -------
    plotly : module
        The ``plotly`` module.
    """
    plotly = pytest.importorskip("plotly")
    yield plotly
