"""Test fixtures used by nilearn.plotting.surface tests."""

import pytest

from nilearn._utils.helpers import is_matplotlib_installed, is_plotly_installed


def pytest_generate_tests(metafunc):
    """Check installed packages and set engines to be used for the tests.

    https://docs.pytest.org/en/stable/example/parametrize.html#deferring-the-setup-of-parametrized-resources
    """
    if "engine" in metafunc.fixturenames:
        installed_engines = []
        if is_matplotlib_installed():
            installed_engines.append("matplotlib")
        if is_plotly_installed():
            installed_engines.append("plotly")
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
