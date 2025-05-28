"""Test nilearn.plotting.surface._backend functions."""

# ruff: noqa: ARG001

import pytest

from nilearn.plotting.surface._backend import (
    _check_hemisphere_is_valid,
    _check_view_is_valid,
)


@pytest.fixture
def backend(engine):
    if engine == "matplotlib":
        from nilearn.plotting.surface._matplotlib_backend import (
            MatplotlibSurfaceBackend,
        )

        return MatplotlibSurfaceBackend()
    elif engine == "plotly":
        from nilearn.plotting.surface._plotly_backend import (
            PlotlySurfaceBackend,
        )

        return PlotlySurfaceBackend()


@pytest.mark.parametrize(
    "view,is_valid",
    [
        ("lateral", True),
        ("medial", True),
        ("latreal", False),
        ((100, 100), True),
        ([100.0, 100.0], True),
        ((100, 100, 1), False),
        (("lateral", "medial"), False),
        ([100, "bar"], False),
    ],
)
def test_check_view_is_valid(view, is_valid):
    assert _check_view_is_valid(view) is is_valid


@pytest.mark.parametrize(
    "hemi,is_valid",
    [
        ("left", True),
        ("right", True),
        ("both", True),
        ("lft", False),
    ],
)
def test_check_hemisphere_is_valid(hemi, is_valid):
    assert _check_hemisphere_is_valid(hemi) is is_valid


def test_plot_surf(plt, engine, backend, tmp_path, in_memory_mesh, bg_map):
    """Test
    nilearn.plotting.surface._backend.MatplotlibBackend.plot_surf and
    nilearn.plotting.surface._backend.PlotlyBackend.plot_surf functions
    with available engine backends.
    """
    # to avoid extra warnings
    alpha = None
    cbar_vmin = None
    cbar_vmax = None
    if engine == "matplotlib":
        alpha = 0.5
        cbar_vmin = 0
        cbar_vmax = 150

    # Plot mesh only
    backend.plot_surf(in_memory_mesh)

    # Plot mesh with background
    backend.plot_surf(in_memory_mesh, bg_map=bg_map)
    backend.plot_surf(in_memory_mesh, bg_map=bg_map, darkness=0.5)
    backend.plot_surf(
        in_memory_mesh,
        bg_map=bg_map,
        alpha=alpha,
        output_file=tmp_path / "tmp.png",
    )

    # Plot with colorbar
    backend.plot_surf(in_memory_mesh, bg_map=bg_map, colorbar=True)
    backend.plot_surf(
        in_memory_mesh,
        bg_map=bg_map,
        colorbar=True,
        cbar_vmin=cbar_vmin,
        cbar_vmax=cbar_vmax,
        cbar_tick_format="%i",
    )


def test_plot_surf_stat_map(plt, engine, backend, in_memory_mesh, bg_map):
    """Smoke test when stat_map is specified to
    nilearn.plotting.surface._backend.MatplotlibBackend.plot_surf_stat_map
    and
    nilearn.plotting.surface._backend.PlotlyBackend.plot_surf_stat_map
    functions together with mesh.
    """
    alpha = 1 if engine == "matplotlib" else None

    backend.plot_surf_stat_map(in_memory_mesh, stat_map=bg_map)
    backend.plot_surf_stat_map(in_memory_mesh, stat_map=bg_map, alpha=alpha)
