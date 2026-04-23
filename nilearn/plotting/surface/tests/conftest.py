"""Test fixtures used by nilearn.plotting.surface tests."""

import numpy as np
import pytest

from nilearn._utils.helpers import is_matplotlib_installed, is_plotly_installed
from nilearn.surface import SurfaceImage


def pytest_generate_tests(metafunc):
    """Check installed packages and possible engines defined by
    ``@pytest.mark.engines`` for the test, and set engines to be used depending
    on availability of the engine.

    For example if the test should only be run for "niivue" and "plotly" (if
    installed) engines, put the below line on top of test function:

    @pytest.mark.engines(["plotly", "niivue"])

    If plotly is installed, the tests will be run with plotly; test for plotly
    will be skipped otherwise.

    https://docs.pytest.org/en/stable/example/parametrize.html#deferring-the-setup-of-parametrized-resources
    """
    if "engine" in metafunc.fixturenames:
        marker = metafunc.definition.get_closest_marker("engines")
        if marker:
            engines = marker.args[0]
        else:
            engines = ["matplotlib", "plotly"]

        installed_engines = []
        if is_matplotlib_installed() and "matplotlib" in engines:
            installed_engines.append("matplotlib")
        if is_plotly_installed() and "plotly" in engines:
            installed_engines.append("plotly")
        if "niivue" in engines:
            installed_engines.append("niivue")

        metafunc.parametrize("engine", installed_engines, indirect=True)


@pytest.fixture
def engine(request):
    """Return each of the engines detected by pytest_generate_tests."""
    return request.param


@pytest.fixture
def plt(request, engine):
    """Return the fixture for setup and teardown of test depending on the
    engine.
    """
    if engine == "matplotlib":
        return request.getfixturevalue("matplotlib_pyplot")
    elif engine == "plotly":
        return request.getfixturevalue("plotly")


@pytest.fixture
def bg_map(rng, in_memory_mesh) -> np.ndarray:
    """Return a background map with positive value."""
    return np.abs(rng.standard_normal(size=in_memory_mesh.n_vertices))


@pytest.fixture
def surface_image_roi(surf_mask_1d) -> SurfaceImage:
    """SurfaceImage for plotting."""
    return surf_mask_1d


@pytest.fixture
def parcellation(in_memory_mesh) -> np.ndarray:
    """Return array with parcellation with 2 regions."""
    parcellation = np.zeros((in_memory_mesh.n_vertices,))
    parcellation[in_memory_mesh.faces[3]] = 1
    parcellation[in_memory_mesh.faces[5]] = 2
    return parcellation


@pytest.fixture
def surface_image_parcellation(rng, in_memory_mesh) -> SurfaceImage:
    """Return surface image with random parcellation."""
    data = rng.integers(100, size=(in_memory_mesh.n_vertices, 1)).astype(float)
    parcellation = SurfaceImage(
        mesh={"left": in_memory_mesh, "right": in_memory_mesh},
        data={"left": data, "right": data},
    )
    return parcellation
